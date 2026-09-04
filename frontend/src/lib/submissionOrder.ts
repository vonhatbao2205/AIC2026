/** Reordering the answers of one question.
 *
 *  Rank is not a column here: `created_at` IS the order key (see `sortRows` and
 *  `nextCreatedAt` in `sharedSubmission.ts`), and the order key is what the
 *  preliminary round scores — `Final = (R@1 + R@5 + R@20 + R@50 + R@100) / 5`,
 *  a max over the first k answers. Moving a row from rank 7 to rank 1 is
 *  therefore a real edit to the shared table, not a view preference, and it has
 *  to be written for the whole team.
 *
 *  Kept free of React and of Supabase so the arithmetic that decides which
 *  answer scores the query can be tested on its own.
 */
import type { SubmissionRow } from "./submission";

/** Drag payload for one answer row. The question id travels with it because a
 *  drop into another question's table is not a reorder — it would silently
 *  re-attribute an answer — so the target has to be able to refuse it. */
export const SUBMISSION_ROW_MIME = "text/x-submission-row";

export interface SubmissionRowDrag {
  questionId: string;
  rowId: string;
}

/** One row's new place in the order, as the instant that puts it there. */
export interface OrderChange {
  id: string;
  createdAt: string;
}

export function setRowDrag(event: React.DragEvent, payload: SubmissionRowDrag): void {
  event.dataTransfer.setData(SUBMISSION_ROW_MIME, JSON.stringify(payload));
  event.dataTransfer.effectAllowed = "move";
}

/** True while an answer row is being dragged over a target.
 *
 *  `dragover` may only see the TYPES — `getData` returns "" there in every
 *  browser — so a row that wants to light up has to ask this rather than parse
 *  the payload. */
export function hasRowDrag(event: React.DragEvent): boolean {
  return Array.from(event.dataTransfer.types).includes(SUBMISSION_ROW_MIME);
}

export function readRowDrag(event: React.DragEvent): SubmissionRowDrag | null {
  const raw = event.dataTransfer.getData(SUBMISSION_ROW_MIME);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as SubmissionRowDrag;
    if (!parsed?.questionId || !parsed?.rowId) return null;
    return { questionId: parsed.questionId, rowId: parsed.rowId };
  } catch {
    return null;
  }
}

/** The row's `created_at` as an instant, or NaN when it has none. */
function createdMs(row: SubmissionRow): number {
  const parsed = Date.parse(row.createdAt ?? "");
  return Number.isNaN(parsed) ? Number.NaN : parsed;
}

/** Same instant, one spelling. The two sides format it differently — the client
 *  writes `…T19:04:00.123Z`, PostgREST answers `…T19:04:00.123456+00:00` — so a
 *  plain string compare would report a change on every single row. */
function normalisedIso(row: SubmissionRow): string | null {
  const stamp = createdMs(row);
  return Number.isFinite(stamp) ? new Date(stamp).toISOString() : null;
}

/** The instants this question's rows occupy, forced to strictly increase.
 *
 *  Two rows can share a millisecond, and one written before `createdAt` existed
 *  carries none at all. In both cases the uuid tiebreak in `sortRows` — not the
 *  operator — decides which one is rank 1, and a permutation of an ambiguous
 *  list is still ambiguous. So the slots are normalised first: each is at least
 *  one millisecond after the one before it.
 *
 *  Only the question's OWN timestamps are reused. Reordering therefore keeps
 *  these answers in the same region of the shared table's timeline instead of
 *  stamping them "now" and jumping them past every other question's rows.
 *
 *  `rows` must already be in display order (which is what `sortRows` produces).
 */
export function orderSlots(rows: readonly SubmissionRow[]): number[] {
  const stamps = rows.map(createdMs);
  const firstFinite = stamps.find((value) => Number.isFinite(value));
  // Nothing was ever stamped: any monotone base is as good as another.
  let previous = (firstFinite ?? Date.now()) - 1;
  return stamps.map((stamp) => {
    previous = Math.max(Number.isFinite(stamp) ? stamp : previous + 1, previous + 1);
    return previous;
  });
}

/** Array move: the element at `from` ends up at index `to`. */
export function moveIndex<T>(items: readonly T[], from: number, to: number): T[] {
  const next = [...items];
  const [moved] = next.splice(from, 1);
  next.splice(to, 0, moved);
  return next;
}

/** What to write so `rows[from]` becomes rank `to + 1`.
 *
 *  Returns only the rows whose instant actually changes — a drag from rank 7 to
 *  rank 1 rewrites seven rows, not the whole hundred. An out-of-range source, or
 *  a move that changes nothing, returns an empty plan, and the callers treat
 *  that as "no write".
 */
export function planReorder(
  rows: readonly SubmissionRow[],
  from: number,
  to: number,
): OrderChange[] {
  if (from < 0 || from >= rows.length || rows.length < 2) return [];
  const target = Math.min(Math.max(to, 0), rows.length - 1);
  const slots = orderSlots(rows);
  const moved = moveIndex(rows, from, target);
  const changes: OrderChange[] = [];
  moved.forEach((row, index) => {
    const createdAt = new Date(slots[index]).toISOString();
    // A row whose stamp was tied or missing is rewritten even when it did not
    // move: leaving it ambiguous would let the tiebreak undo the drag.
    if (createdAt !== normalisedIso(row)) changes.push({ id: row.id, createdAt });
  });
  return changes;
}

/** The plan that puts the question back exactly as it is now.
 *
 *  This is what undo writes. It cannot be read off `row.createdAt` alone: a row
 *  that had no stamp, or shared one with its neighbour, has no instant to
 *  restore — the position it is at right now is defined by the normalised slots,
 *  so those are what gets written back.
 *
 *  `after` is the plan being applied, so rows the move leaves alone are left out
 *  of the undo write too.
 */
export function planRestore(
  rows: readonly SubmissionRow[],
  after: readonly OrderChange[],
): OrderChange[] {
  const applied = new Map(after.map((change) => [change.id, change.createdAt]));
  const slots = orderSlots(rows);
  const changes: OrderChange[] = [];
  rows.forEach((row, index) => {
    const before = new Date(slots[index]).toISOString();
    const next = applied.get(row.id) ?? normalisedIso(row);
    if (before !== next) changes.push({ id: row.id, createdAt: before });
  });
  return changes;
}
