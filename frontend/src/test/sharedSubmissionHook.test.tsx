import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

/** Captured `update(...)` payloads, in the order they reached PostgREST. */
const updates: Record<string, unknown>[] = [];
const inserts: Record<string, unknown>[] = [];

/** Minimal stand-in for the PostgREST query builder the hook chains onto. */
function makeClient() {
  const result = { data: [] as unknown[], error: null };
  const builder: Record<string, unknown> = {};
  const chain = () => builder;
  Object.assign(builder, {
    select: vi.fn(() => builder),
    eq: vi.fn(() => builder),
    order: vi.fn(() => Promise.resolve(result)),
    insert: vi.fn((record: Record<string, unknown>) => {
      inserts.push(record);
      return Promise.resolve({ error: null });
    }),
    update: vi.fn((payload: Record<string, unknown>) => {
      updates.push(payload);
      // `.select()` at the end of an update resolves with the stored row.
      return {
        eq: chain,
        select: () => Promise.resolve({ data: [{ ...serverRow, ...payload, revision: 2 }], error: null }),
      };
    }),
    delete: vi.fn(() => ({ eq: () => Promise.resolve({ error: null }) })),
    then: undefined,
  });
  // The update chain needs `.eq().eq().select()`, so give eq its own shape.
  builder.update = vi.fn((payload: Record<string, unknown>) => {
    updates.push(payload);
    const tail = {
      eq: () => tail,
      select: () => Promise.resolve({ data: [{ ...serverRow, ...payload, revision: 2 }], error: null }),
    };
    return tail;
  });
  return {
    from: vi.fn(() => builder),
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
