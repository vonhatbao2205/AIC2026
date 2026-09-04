import { describe, expect, it } from "vitest";
import { moveIndex, orderSlots, planReorder, planRestore } from "../lib/submissionOrder";
import { sortRows } from "../lib/sharedSubmission";
import type { SubmissionRow } from "../lib/submission";

/** A question's answers, in rank order, one millisecond apart. */
function rows(count: number, base = Date.parse("2026-09-04T10:00:00.000Z")): SubmissionRow[] {
  return Array.from({ length: count }, (_, index) => ({
    id: `row-${index + 1}`,
    questionId: "query-p1-3-kis",
    videoId: "L27_V013",
    frames: [1000 + index],
    answer: "",
    createdAt: new Date(base + index).toISOString(),
  }));
}

/** Apply a plan the way the hook does, then re-sort the way the panel reads. */
function applied(list: SubmissionRow[], plan: { id: string; createdAt: string }[]): string[] {
  const wanted = new Map(plan.map((change) => [change.id, change.createdAt]));
  return sortRows(
    list.map((row) => (wanted.has(row.id) ? { ...row, createdAt: wanted.get(row.id) } : row)),
  ).map((row) => row.id);
}

describe("planReorder", () => {
  it("puts rank 7 at rank 1 and shifts the rest down", () => {
    const list = rows(10);
    const plan = planReorder(list, 6, 0);
    expect(applied(list, plan)).toEqual([
      "row-7",
      "row-1",
      "row-2",
      "row-3",
      "row-4",
      "row-5",
      "row-6",
      "row-8",
      "row-9",
      "row-10",
    ]);
  });

  it("rewrites only the rows the move passes, not the whole question", () => {
    // The write is what costs during a contest: a hundred-row question must not
    // send a hundred timestamps because one answer moved three places.
    const plan = planReorder(rows(100), 6, 0);
    expect(plan).toHaveLength(7);
    expect(plan.map((change) => change.id).sort()).toEqual(
      ["row-1", "row-2", "row-3", "row-4", "row-5", "row-6", "row-7"].sort(),
    );
  });

  it("moves downwards too", () => {
    const list = rows(5);
    expect(applied(list, planReorder(list, 0, 3))).toEqual([
      "row-2",
      "row-3",
      "row-4",
      "row-1",
      "row-5",
    ]);
  });

  it("keeps the answers inside their own slice of the shared timeline", () => {
    // Stamping the moved rows "now" would jump this question's answers past
    // every other question's rows in the table. The instants are permuted, not
    // replaced, so the block occupies exactly the same span it did before.
    const list = rows(10);
    const before = list.map((row) => Date.parse(row.createdAt as string)).sort((a, b) => a - b);
    const plan = planReorder(list, 6, 0);
    const after = applied(list, plan);
    expect(after).toHaveLength(10);
    const stamps = new Map(plan.map((change) => [change.id, Date.parse(change.createdAt)]));
    const all = list
      .map((row) => stamps.get(row.id) ?? Date.parse(row.createdAt as string))
      .sort((a, b) => a - b);
    expect(all).toEqual(before);
  });

  it("is a no-op when nothing moves", () => {
    expect(planReorder(rows(5), 2, 2)).toEqual([]);
    expect(planReorder(rows(1), 0, 0)).toEqual([]);
    expect(planReorder([], 0, 0)).toEqual([]);
  });

  it("refuses an out-of-range source and clamps the destination", () => {
    expect(planReorder(rows(5), 9, 0)).toEqual([]);
    expect(planReorder(rows(5), -1, 0)).toEqual([]);
    const list = rows(5);
    expect(applied(list, planReorder(list, 0, 99))).toEqual([
      "row-2",
      "row-3",
      "row-4",
      "row-5",
      "row-1",
    ]);
  });

  it("breaks a tie instead of leaving the uuid tiebreak to decide rank 1", () => {
    // Two answers written in the same millisecond — a bulk insert used to do
    // exactly this. Permuting ambiguous stamps would leave the order to
    // `sortRows`'s id fallback, so the tie is normalised away.
    const stamp = "2026-09-04T10:00:00.000Z";
    const list: SubmissionRow[] = ["b", "a", "c"].map((suffix) => ({
      ...rows(1)[0],
      id: `row-${suffix}`,
      createdAt: stamp,
    }));
    const plan = planReorder(list, 2, 0);
    expect(applied(list, plan)).toEqual(["row-c", "row-b", "row-a"]);
    const stamps = plan.map((change) => change.createdAt);
    expect(new Set(stamps).size).toBe(stamps.length);
  });

  it("gives an unstamped row a place instead of dropping it to rank 1", () => {
    // Rows cached before `createdAt` existed parse as NaN, which sorts as 0 —
    // ahead of everything. A reorder has to leave every row with a real instant.
    const list = rows(3);
    delete list[1].createdAt;
    const plan = planReorder(list, 2, 0);
    expect(applied(list, plan)).toEqual(["row-3", "row-1", "row-2"]);
    for (const row of list) {
      const change = plan.find((item) => item.id === row.id);
      expect(Number.isFinite(Date.parse(change?.createdAt ?? row.createdAt ?? ""))).toBe(true);
    }
  });

  it("ignores the difference between the client's and PostgREST's spelling", () => {
    // `…T10:00:00.123Z` and `…T10:00:00.123456+00:00` are the same instant. A
    // plain string compare would report every row as moved on every drag.
    const list = rows(3).map((row, index) => ({
      ...row,
      createdAt: `2026-09-04T10:00:00.00${index}000+00:00`,
    }));
    expect(planReorder(list, 1, 1)).toEqual([]);
    expect(planReorder(list, 2, 1)).toHaveLength(2);
  });
});

describe("planRestore", () => {
  it("undoes a move exactly", () => {
    const list = rows(10);
    const forward = planReorder(list, 6, 0);
    const back = planRestore(list, forward);
    const moved = list.map((row) => {
      const change = forward.find((item) => item.id === row.id);
      return change ? { ...row, createdAt: change.createdAt } : row;
    });
    expect(applied(moved, back)).toEqual(list.map((row) => row.id));
  });

  it("writes back only what the move touched", () => {
    const list = rows(100);
    expect(planRestore(list, planReorder(list, 6, 0))).toHaveLength(7);
  });

  it("restores an order that was ambiguous to begin with", () => {
    // Nothing to hand back for a row that shared a millisecond, so the restore
    // is written from the normalised slots rather than from the raw stamps.
    const stamp = "2026-09-04T10:00:00.000Z";
    const list: SubmissionRow[] = ["x", "y", "z"].map((suffix) => ({
      ...rows(1)[0],
      id: `row-${suffix}`,
      createdAt: stamp,
    }));
    const forward = planReorder(list, 2, 0);
    const back = planRestore(list, forward);
    const moved = list.map((row) => {
      const change = forward.find((item) => item.id === row.id);
      return change ? { ...row, createdAt: change.createdAt } : row;
    });
    expect(applied(moved, back)).toEqual(["row-x", "row-y", "row-z"]);
  });
});

describe("orderSlots", () => {
  it("is strictly increasing whatever it is handed", () => {
    const list = rows(4);
    list[1].createdAt = list[0].createdAt;
    delete list[2].createdAt;
    const slots = orderSlots(list);
    for (let i = 1; i < slots.length; i += 1) expect(slots[i]).toBeGreaterThan(slots[i - 1]);
  });
});

describe("moveIndex", () => {
  it("lands the element on the destination index", () => {
    expect(moveIndex([1, 2, 3, 4], 3, 0)).toEqual([4, 1, 2, 3]);
    expect(moveIndex([1, 2, 3, 4], 0, 2)).toEqual([2, 3, 1, 4]);
  });
});
