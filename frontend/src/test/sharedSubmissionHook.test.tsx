import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

/** What reached PostgREST, in order. */
const updates: Record<string, unknown>[] = [];
/** Calls to `reorder_submissions`, and whether the project has that function. */
const rpcCalls: { name: string; args: Record<string, unknown> }[] = [];
let rpcMissing = false;
const inserts: Record<string, unknown>[] = [];
const deletes: { filters: Record<string, unknown>; ids?: string[] }[] = [];
/** Pages the fake `select` hands back, keyed by the requested offset. */
let selectPages: Record<string, unknown>[][] = [[]];
const selectRanges: [number, number][] = [];
/** Rows returned by the id lookup used to rebuild a lost Sticky outbox. */
let lookupRows: Record<string, unknown>[] = [];

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
      return Object.assign(builder, thenable({
        data: lookupRows.filter((row) => values.includes(String(row.id))),
        error: null,
      }));
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
      // Echo one row per targeted id, the way PostgREST does. Replying with a
      // single fixed row let a bulk write appear to land on rows it never
      // named, which hid what the merge would really do.
      let targets: string[] | null = null;
      const tail: Record<string, unknown> = {
        eq: (column: string, value: unknown) => {
          if (column === "id") targets = [String(value)];
          return tail;
        },
        in: (_column: string, values: string[]) => {
          targets = values;
          return tail;
        },
        select: () =>
          Promise.resolve({
            data: (targets ?? [serverRow.id as string]).map((id) => ({
              ...serverRow,
              id,
              ...payload,
              revision: 2,
            })),
            error: null,
          }),
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
    rpc: vi.fn((name: string, args: Record<string, unknown>) => {
      rpcCalls.push({ name, args });
      if (rpcMissing) {
        // What PostgREST answers when migration 004 has not been run.
        return Promise.resolve({
          data: null,
          error: { code: "PGRST202", message: "Could not find the function public.reorder_submissions" },
        });
      }
      const ids = (args.p_ids ?? []) as string[];
      const stamps = (args.p_created_at ?? []) as string[];
      return Promise.resolve({
        data: ids.map((id, index) => ({
          ...serverRow,
          id,
          created_at: stamps[index],
          revision: 2,
        })),
        error: null,
      });
    }),
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
const { planReorder } = await import("../lib/submissionOrder");

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
  rpcCalls.length = 0;
  rpcMissing = false;
  selectRanges.length = 0;
  selectPages = [[]];
  lookupRows = [];
  serverRow = { id: baseRow.id, room: "test-room", question_id: baseRow.questionId };
  localStorage.clear();
});

describe("shared submission writes", () => {
  it("recreates a Sticky insert lost with the in-memory outbox", async () => {
    const { result } = renderHook(() => useSharedSubmission("session-1"));
    await act(async () => result.current.ensureRows([{ ...baseRow }]));

    await waitFor(() => expect(inserts).toHaveLength(1));
    await waitFor(() => expect(result.current.rows[0].syncState).toBe("synced"));
    expect(inserts[0].id).toBe(baseRow.id);
  });

  it("confirms an existing Sticky row without inserting it twice", async () => {
    lookupRows = [{
      id: baseRow.id,
      room: "test-room",
      session_id: "session-1",
      question_id: baseRow.questionId,
      retrieval_database: "btc",
      video_id: baseRow.videoId,
      frames: baseRow.frames,
      keyframe_ids: baseRow.keyframeIds,
      pts_times: baseRow.ptsTimes,
      answer: "",
      submitted_by: "Tester",
      source: "submit",
      revision: 1,
      created_at: "2026-09-04T10:00:00.000Z",
    }];
    const { result } = renderHook(() => useSharedSubmission("session-1"));
    await act(async () => result.current.ensureRows([{ ...baseRow }]));

    await waitFor(() => expect(result.current.rows[0].syncState).toBe("synced"));
    expect(inserts).toHaveLength(0);
  });

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

describe("reordering answers", () => {
  /** Three answers to one question, a millisecond apart, already on the server. */
  function ranked() {
    return ["a", "b", "c"].map((suffix, index) => ({
      ...baseRow,
      id: `row-${suffix}`,
      frames: [index],
      createdAt: new Date(Date.parse("2026-09-04T10:00:00.000Z") + index).toISOString(),
    }));
  }

  it("writes the whole permutation as ONE statement", async () => {
    // Rank is the score. Sending it as N sequential PATCHes would leave the
    // list visibly half-reordered on four other screens, and a tab closed
    // halfway would leave an order nobody chose.
    const { result } = renderHook(() => useSharedSubmission());
    const list = ranked();
    await act(async () => result.current.addRows(list));
    await waitFor(() => expect(inserts).toHaveLength(3));

    await act(async () =>
      result.current.reorderRows(planReorder(result.current.rows, 2, 0), "hạng 3 → 1"),
    );

    expect(result.current.rows.map((row) => row.id)).toEqual(["row-c", "row-a", "row-b"]);
    await waitFor(() => expect(rpcCalls).toHaveLength(1));
    expect(rpcCalls[0].name).toBe("reorder_submissions");
    expect((rpcCalls[0].args.p_ids as string[]).sort()).toEqual(["row-a", "row-b", "row-c"]);
    // Ordering only: no frames, no answer text, so it cannot clobber a
    // teammate's edit to a row it moves past.
    expect(updates).toHaveLength(0);
  });

  it("falls back to one write per row when migration 004 is missing", async () => {
    // The outbox is a strict FIFO that stops on the first failure, so retrying
    // a call that can never succeed would wedge every later answer behind it.
    rpcMissing = true;
    const { result } = renderHook(() => useSharedSubmission());
    await act(async () => result.current.addRows(ranked()));
    await waitFor(() => expect(inserts).toHaveLength(3));

    await act(async () => result.current.reorderRows(planReorder(result.current.rows, 2, 0)));

    await waitFor(() => expect(updates).toHaveLength(3));
    // Still only the ordering column.
    expect(Object.keys(updates[0])).toEqual(["created_at"]);
    expect(result.current.rows.map((row) => row.id)).toEqual(["row-c", "row-a", "row-b"]);
    expect(result.current.error).toBeNull();
    expect(result.current.pending).toBe(0);
  });

  it("refreshes the revision of a content edit queued behind the move", async () => {
    // The reorder bumps `revision` on every row it touches. A queued update
    // still stating the old number would match zero rows and be reported as a
    // conflict the operator never caused.
    const { result } = renderHook(() => useSharedSubmission());
    await act(async () => result.current.addRows(ranked()));
    await waitFor(() => expect(inserts).toHaveLength(3));

    await act(async () => {
      result.current.reorderRows(planReorder(result.current.rows, 2, 0));
      result.current.updateRow("row-c", { frames: [4242] });
    });

    await waitFor(() => expect(updates).toHaveLength(1));
    expect(updates[0].frames).toEqual([4242]);
    expect(result.current.conflicts).toEqual([]);
    expect(result.current.rows.find((row) => row.id === "row-c")?.frames).toEqual([4242]);
  });

  it("drops a row a teammate deleted mid-drag instead of refusing the move", async () => {
    const { result } = renderHook(() => useSharedSubmission());
    await act(async () => result.current.addRows(ranked()));
    await waitFor(() => expect(inserts).toHaveLength(3));
    const plan = planReorder(result.current.rows, 2, 0);
    await act(async () => result.current.deleteRow("row-b"));

    await act(async () => result.current.reorderRows(plan));

    expect(result.current.rows.map((row) => row.id)).toEqual(["row-c", "row-a"]);
    await waitFor(() => expect(rpcCalls).toHaveLength(1));
    expect(rpcCalls[0].args.p_ids).not.toContain("row-b");
  });

  it("ignores an empty plan rather than writing nothing to the server", async () => {
    const { result } = renderHook(() => useSharedSubmission());
    await act(async () => result.current.addRows(ranked()));
    await waitFor(() => expect(inserts).toHaveLength(3));

    await act(async () => result.current.reorderRows(planReorder(result.current.rows, 1, 1)));

    expect(rpcCalls).toHaveLength(0);
    // …and no undo step either: Ctrl+Z must not be spent on a drag that
    // dropped the row back where it started.
    expect(result.current.undoLabel).toBe("add 3 rows");
  });

  it("undoes a move by writing the ranking back, not by patching rows", async () => {
    const { result } = renderHook(() => useSharedSubmission());
    await act(async () => result.current.addRows(ranked()));
    await waitFor(() => expect(inserts).toHaveLength(3));
    await act(async () =>
      result.current.reorderRows(planReorder(result.current.rows, 2, 0), "hạng 3 → 1"),
    );
    await waitFor(() => expect(rpcCalls).toHaveLength(1));

    let label: string | null = null;
    await act(async () => {
      label = result.current.undo();
    });

    expect(label).toBe("hạng 3 → 1");
    expect(result.current.rows.map((row) => row.id)).toEqual(["row-a", "row-b", "row-c"]);
    // A second ordering write, not a row update: `update` never sends
    // `created_at`, so undoing through it would come back on the next reload.
    await waitFor(() => expect(rpcCalls).toHaveLength(2));
    expect(updates).toHaveLength(0);
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

describe("undo", () => {
  const second = { ...baseRow, id: "22222222-2222-4222-8222-222222222222", frames: [900] };

  it("puts a deleted row back with its id and its rank", async () => {
    // `created_at` IS the position in the exported CSV, and rank is score. A
    // restore that stamped a fresh timestamp would quietly move a hand-picked
    // answer from rank 1 to the end of the list.
    const { result } = renderHook(() => useSharedSubmission());
    const row = { ...baseRow, createdAt: "2026-08-21T10:00:00.000Z" };
    await act(async () => result.current.addRow(row));
    await waitFor(() => expect(inserts).toHaveLength(1));

    await act(async () => result.current.deleteRow(row.id));
    expect(result.current.rows).toHaveLength(0);

    await act(async () => {
      result.current.undo();
    });
    expect(result.current.rows).toHaveLength(1);
    expect(result.current.rows[0].id).toBe(row.id);
    expect(result.current.rows[0].createdAt).toBe("2026-08-21T10:00:00.000Z");
    await waitFor(() => expect(inserts).toHaveLength(2));
    expect(inserts[1].id).toBe(row.id);
    expect(inserts[1].created_at).toBe("2026-08-21T10:00:00.000Z");
  });

  it("undoes a regeneration as ONE step, not as a delete and an insert", async () => {
    // Regenerating is deleteRows + addRows. Undoing half of it would leave the
    // question holding both the old rows and the new ones.
    const { result } = renderHook(() => useSharedSubmission());
    await act(async () => result.current.addRows([{ ...baseRow }, { ...second }]));
    await waitFor(() => expect(inserts).toHaveLength(2));
    const fresh = { ...baseRow, id: "33333333-3333-4333-8333-333333333333", frames: [77] };

    await act(async () => {
      result.current.transaction("regenerate", () => {
        result.current.deleteRows([baseRow.id, second.id]);
        result.current.addRows([fresh]);
      });
    });
    expect(result.current.rows.map((row) => row.id)).toEqual([fresh.id]);

    let label: string | null = null;
    await act(async () => {
      label = result.current.undo();
    });
    expect(label).toBe("regenerate");
    expect(result.current.rows.map((row) => row.id).sort()).toEqual(
      [baseRow.id, second.id].sort(),
    );
  });

  it("gives every row back its OWN previous answer", async () => {
    // A bulk fill overwrites rows that did not all hold the same text. Undo has
    // to restore each one, and still in one write per distinct value rather
    // than one per row.
    const { result } = renderHook(() => useSharedSubmission());
    await act(async () =>
      result.current.addRows([
        { ...baseRow, answer: "37.05" },
        { ...second, answer: "" },
      ]),
    );
    await waitFor(() => expect(inserts).toHaveLength(2));

    await act(async () => result.current.setAnswerForQuestion(baseRow.questionId, "Tà Pứa"));
    expect(result.current.rows.map((row) => row.answer)).toEqual(["Tà Pứa", "Tà Pứa"]);

    updates.length = 0;
    await act(async () => {
      result.current.undo();
    });
    const byId = new Map(result.current.rows.map((row) => [row.id, row.answer]));
    expect(byId.get(baseRow.id)).toBe("37.05");
    expect(byId.get(second.id)).toBe("");
    await waitFor(() => expect(updates.length).toBeGreaterThan(0));
    // Two distinct previous values -> two writes, not one per row.
    expect(updates).toHaveLength(2);
  });

  it("treats a burst of keystrokes in one cell as a single undo", async () => {
    const { result } = renderHook(() => useSharedSubmission());
    await act(async () => result.current.addRow({ ...baseRow, videoId: "L27_V013" }));
    await waitFor(() => expect(inserts).toHaveLength(1));

    await act(async () => {
      result.current.updateRow(baseRow.id, { videoId: "L" });
      result.current.updateRow(baseRow.id, { videoId: "L3" });
      result.current.updateRow(baseRow.id, { videoId: "L30" });
    });
    expect(result.current.rows[0].videoId).toBe("L30");

    await act(async () => {
      result.current.undo();
    });
    expect(result.current.rows[0].videoId).toBe("L27_V013");
  });

  it("forgets the steps of a question that was cleared", async () => {
    // "clear all" deletes BY QUESTION on the server, so this client cannot know
    // every row that went. A later Ctrl+Z must not resurrect the subset it did
    // know about, into a question the operator just emptied.
    const { result } = renderHook(() => useSharedSubmission("session-1"));
    await act(async () => result.current.addRows([{ ...baseRow }, { ...second }]));
    await waitFor(() => expect(inserts).toHaveLength(2));
    await act(async () => result.current.deleteRow(second.id));
    expect(result.current.undoLabel).not.toBeNull();

    await act(async () => result.current.clearQuestion(baseRow.questionId));
    expect(result.current.undoLabel).toBeNull();

    let label: string | null = "unset";
    await act(async () => {
      label = result.current.undo();
    });
    expect(label).toBeNull();
    expect(result.current.rows).toHaveLength(0);
  });

  it("removes rows that the undone action created", async () => {
    const { result } = renderHook(() => useSharedSubmission());
    await act(async () => result.current.addRows([{ ...baseRow }, { ...second }]));
    await waitFor(() => expect(inserts).toHaveLength(2));

    await act(async () => {
      result.current.undo();
    });
    expect(result.current.rows).toHaveLength(0);
    expect(result.current.undoLabel).toBeNull();
  });
});
