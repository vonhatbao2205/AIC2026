import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

/** What reached PostgREST, in order. */
const updates: Record<string, unknown>[] = [];
const inserts: Record<string, unknown>[] = [];
const deletes: { filters: Record<string, unknown>; ids?: string[] }[] = [];
/** Pages the fake `select` hands back, keyed by the requested offset. */
let selectPages: Record<string, unknown>[][] = [[]];
const selectRanges: [number, number][] = [];

function thenable<T>(value: T) {
  return { then: (resolve: (v: T) => unknown) => Promise.resolve(value).then(resolve) };
}

/** Chainable stand-in for the PostgREST builder, faithful enough that a missing
 *  link (`.range` after `.order`) fails the test instead of passing silently. */
function makeBuilder() {
  const filters: Record<string, unknown> = {};
  const builder: Record<string, unknown> = {};
  Object.assign(builder, {
    select: () => builder,
    eq: (column: string, value: unknown) => {
      filters[column] = value;
      return builder;
    },
    in: (_column: string, values: string[]) => {
      filters.in = values;
      return builder;
    },
    order: () => builder,
    range: (from: number, to: number) => {
      selectRanges.push([from, to]);
      const page = selectPages[Math.floor(from / 1000)] ?? [];
      return Promise.resolve({ data: page, error: null });
    },
    insert: (record: Record<string, unknown> | Record<string, unknown>[]) => {
      for (const item of Array.isArray(record) ? record : [record]) inserts.push(item);
      return Promise.resolve({ error: null });
    },
    update: (payload: Record<string, unknown>) => {
      updates.push(payload);
      const tail: Record<string, unknown> = {
        eq: () => tail,
        in: () => tail,
        select: () =>
          Promise.resolve({ data: [{ ...serverRow, ...payload, revision: 2 }], error: null }),
      };
      return tail;
    },
    delete: () => {
      const record: { filters: Record<string, unknown>; ids?: string[] } = { filters };
      deletes.push(record);
      const tail: Record<string, unknown> = {
        eq: (column: string, value: unknown) => {
          filters[column] = value;
          return tail;
        },
        in: (_column: string, values: string[]) => {
          record.ids = values;
          return tail;
        },
        ...thenable({ error: null }),
      };
      return tail;
    },
  });
  return builder;
}

function makeClient() {
  return {
    from: vi.fn(() => makeBuilder()),
    channel: vi.fn(() => ({ on: vi.fn().mockReturnThis(), subscribe: vi.fn().mockReturnThis() })),
    removeChannel: vi.fn(),
  };
}

let serverRow: Record<string, unknown> = {};
const client = makeClient();

vi.mock("../lib/supabase", () => ({
  SUBMISSIONS_TABLE: "submissions",
  SUBMISSION_ROOM: "test-room",
  isSupabaseConfigured: () => true,
  getSupabase: () => client,
  getDisplayName: () => "Tester",
  setDisplayName: () => {},
  // useDisplayName subscribes to name changes; the name never changes here, so
  // an unsubscribe that does nothing is the whole contract.
  subscribeDisplayName: () => () => {},
}));

const { useSharedSubmission } = await import("../hooks/useSharedSubmission");

const baseRow = {
  id: "11111111-1111-4111-8111-111111111111",
  questionId: "query-p1-1-kis",
  videoId: "L27_V013",
  frames: [18350],
  answer: "",
  keyframeIds: ["L27/L27_V013/042"] as (string | null)[],
  ptsTimes: [734.0] as (number | null)[],
};

beforeEach(() => {
  updates.length = 0;
  inserts.length = 0;
  deletes.length = 0;
  selectRanges.length = 0;
  selectPages = [[]];
  serverRow = { id: baseRow.id, room: "test-room", question_id: baseRow.questionId };
  localStorage.clear();
});

describe("shared submission writes", () => {
  it("sends the EDITED row, not the one that was there before", async () => {
    // Regression: the update op used to look the row up from a ref at flush
    // time. `flush()` runs in the same tick as the optimistic setRows, so the
    // ref still held the pre-edit row: the old frame was sent, the server
    // answered with a higher revision, and the merge rolled the operator's
    // edit back. Re-picking a frame appeared to do nothing.
    const { result } = renderHook(() => useSharedSubmission());
    await act(async () => result.current.addRow({ ...baseRow }));
    await waitFor(() => expect(inserts).toHaveLength(1));

    await act(async () => result.current.updateRow(baseRow.id, { frames: [18367] }));

    await waitFor(() => expect(updates).toHaveLength(1));
    expect(updates[0].frames).toEqual([18367]);
    expect(result.current.rows[0].frames).toEqual([18367]);
  });

  it("collapses a burst of edits to one row instead of one request per keystroke", async () => {
    const { result } = renderHook(() => useSharedSubmission());
    await act(async () => result.current.addRow({ ...baseRow }));
    await waitFor(() => expect(inserts).toHaveLength(1));

    await act(async () => {
      result.current.updateRow(baseRow.id, { videoId: "L" });
      result.current.updateRow(baseRow.id, { videoId: "L2" });
      result.current.updateRow(baseRow.id, { videoId: "L27" });
    });

    await waitFor(() => expect(updates.length).toBeGreaterThan(0));
    // Whatever the queue did, the last value must win and must have been sent.
    expect(updates[updates.length - 1].video_id).toBe("L27");
    expect(updates.length).toBeLessThan(3);
    expect(result.current.rows[0].videoId).toBe("L27");
  });
});

describe("clearing a question", () => {
  it("sends one delete for the whole question, not one per row", async () => {
    // Regression: this used to enqueue a delete per row. A hundred answers meant
    // a hundred sequential round trips while the table already looked cleared —
    // closing the tab in that window left the rest on the server.
    const { result } = renderHook(() => useSharedSubmission("session-1"));
    await act(async () => {
      for (let i = 0; i < 12; i += 1) {
        result.current.addRow({ ...baseRow, id: `row-${i}`, questionId: "query-p1-16-trake" });
      }
      result.current.addRow({ ...baseRow, id: "keep", questionId: "query-p1-1-kis" });
    });
    await waitFor(() => expect(inserts.length).toBe(13));

    await act(async () => result.current.clearQuestion("query-p1-16-trake"));

    await waitFor(() => expect(deletes.length).toBeGreaterThan(0));
    expect(deletes).toHaveLength(1);
    expect(deletes[0].filters).toMatchObject({
      question_id: "query-p1-16-trake",
      session_id: "session-1",
    });
    // …and it is scoped, so a teammate's answers to other questions survive.
    expect(result.current.rows.map((row) => row.id)).toEqual(["keep"]);
  });

  it("deletes by question so a row this client never saw goes too", async () => {
    const { result } = renderHook(() => useSharedSubmission(null));
    await act(async () => result.current.clearQuestion("query-p1-2-kis"));
    await waitFor(() => expect(deletes).toHaveLength(1));
    // No session yet: fall back to the room rather than deleting across rooms.
    expect(deletes[0].filters).toMatchObject({ question_id: "query-p1-2-kis", room: "test-room" });
    expect(deletes[0].ids).toBeUndefined();
  });
});

describe("loading every answer", () => {
  it("keeps paging past the PostgREST row cap", async () => {
    // PostgREST stops at the project's "Max rows" (1000 by default) and only
    // says so in a header. 24 questions x 100 answers is 2400, so an unpaged
    // read would silently drop the tail — and the export would drop it too.
    const record = (i: number) => ({
      id: `id-${i}`, room: "test-room", session_id: null, question_id: "q",
      retrieval_database: "btc", video_id: "L01_V001", frames: [i],
      keyframe_ids: [null], pts_times: [null], answer: "", submitted_by: "x",
      source: "submit", revision: 1, created_at: `2026-01-01T00:00:${String(i % 60).padStart(2, "0")}Z`,
    });
    selectPages = [
      Array.from({ length: 1000 }, (_, i) => record(i)),
      Array.from({ length: 1000 }, (_, i) => record(1000 + i)),
      Array.from({ length: 400 }, (_, i) => record(2000 + i)),
    ];

    const { result } = renderHook(() => useSharedSubmission(null));

    await waitFor(() => expect(result.current.rows).toHaveLength(2400));
    expect(selectRanges).toEqual([[0, 999], [1000, 1999], [2000, 2999]]);
  });

  it("stops after one page when the table is small", async () => {
    selectPages = [[]];
    renderHook(() => useSharedSubmission(null));
    await waitFor(() => expect(selectRanges.length).toBeGreaterThan(0));
    expect(selectRanges).toEqual([[0, 999]]);
  });
});

describe("switching to a different question pack", () => {
  const cached = (over: Partial<Record<string, unknown>> = {}) => ({
    id: "cached-1", questionId: "query-p1-1-kis", videoId: "L01_V001", frames: [10],
    answer: "", syncState: "pending", sessionId: "old-session", ...over,
  });

  it("does not carry an unsynced answer from the old pack into the new one", async () => {
    // Re-importing a pack reuses the same question ids, so a stray local row
    // would be attributed to a question it was never an answer to.
    localStorage.setItem("aic26_submission_rows", JSON.stringify([cached()]));
    const { result } = renderHook(() => useSharedSubmission("new-session"));
    await waitFor(() => expect(selectRanges.length).toBeGreaterThan(0));
    await waitFor(() => expect(result.current.rows).toHaveLength(0));
  });

  it("keeps an unsynced answer that belongs to the session being loaded", async () => {
    localStorage.setItem(
      "aic26_submission_rows",
      JSON.stringify([cached({ sessionId: "new-session" })]),
    );
    const { result } = renderHook(() => useSharedSubmission("new-session"));
    await waitFor(() => expect(selectRanges.length).toBeGreaterThan(0));
    expect(result.current.rows.map((row) => row.id)).toEqual(["cached-1"]);
  });

  it("keeps a row written before any pack was published", async () => {
    localStorage.setItem(
      "aic26_submission_rows",
      JSON.stringify([cached({ sessionId: null })]),
    );
    const { result } = renderHook(() => useSharedSubmission("new-session"));
    await waitFor(() => expect(selectRanges.length).toBeGreaterThan(0));
    expect(result.current.rows.map((row) => row.id)).toEqual(["cached-1"]);
  });
});
