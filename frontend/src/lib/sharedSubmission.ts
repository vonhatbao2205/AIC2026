/** Mapping between the Submission table's rows and the shared Postgres record.
 *
 * Kept free of React and of the Supabase client so the shape of what goes on the
 * wire can be tested directly — this is the layer where a mistake would corrupt
 * five people's answers at once.
 */
import type { RetrievalDatabase } from "../api/types";
import type { SubmissionRow } from "./submission";

export interface SubmissionRecord {
  id: string;
  room: string;
  session_id: string | null;
  question_id: string;
  retrieval_database: string;
  video_id: string;
  frames: number[];
  keyframe_ids: (string | null)[];
  pts_times: (number | null)[];
  answer: string;
  submitted_by: string;
  source: string;
  revision: number;
  created_at?: string;
  updated_at?: string;
}

/** Postgres function that rewrites a question's ranking in one statement.
 *  Defined by `supabase/migrations/004_reorder_submissions.sql`; optional, see
 *  `isMissingFunctionError`. */
export const REORDER_FUNCTION = "reorder_submissions";

export function newRowId(): string {
  const cryptoObj = globalThis.crypto as Crypto | undefined;
  if (cryptoObj && typeof cryptoObj.randomUUID === "function") return cryptoObj.randomUUID();
  // jsdom and older browsers: a v4-shaped id is enough, the column is just a uuid.
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

/** Pad a parallel array to the frame count so index i always means frame i. */
function align<T>(values: readonly T[] | undefined, length: number): (T | null)[] {
  const out: (T | null)[] = [];
  for (let i = 0; i < length; i += 1) out.push(values?.[i] ?? null);
  return out;
}

/** Milliseconds of the last stamp handed out, so the next one is never equal. */
let lastStampMs = 0;

/** A creation timestamp that is strictly later than the previous one.
 *
 *  `created_at` IS the ordering key of the answer list, and the answer list is
 *  scored by position — so two rows sharing a timestamp is not a cosmetic tie,
 *  it is a coin flip over which one gets rank 1. That is what used to happen to
 *  every generated batch: a hundred rows took one `Date.now()` on the client and
 *  one transaction `now()` on the server (a multi-row INSERT is a single
 *  statement), so the whole block fell through to the random-uuid tiebreak and
 *  the generator's ranking was replaced by a shuffle. Measured on a real
 *  submission, the strongest video sat at mean rank 50 of 100 — exactly chance.
 *
 *  Stepping 1 ms per row puts a batch of a hundred at most 100 ms ahead of the
 *  clock, which nothing here reads as a duration. */
export function nextCreatedAt(): string {
  const now = Date.now();
  lastStampMs = now > lastStampMs ? now : lastStampMs + 1;
  return new Date(lastStampMs).toISOString();
}

export function rowToRecord(
  row: SubmissionRow,
  room: string,
  user: string,
  sessionId: string | null = null,
): SubmissionRecord {
  const frames = row.frames.map((frame) => Math.round(frame)).filter(Number.isFinite);
  return {
    id: row.id,
    room,
    session_id: row.sessionId ?? sessionId,
    question_id: row.questionId,
    retrieval_database: row.retrievalDatabase ?? "btc",
    video_id: row.videoId,
    frames,
    keyframe_ids: align(row.keyframeIds, frames.length),
    pts_times: align(row.ptsTimes, frames.length),
    answer: row.answer,
    submitted_by: row.submittedBy || user || "unknown",
    source: row.source ?? "submit",
    revision: row.revision ?? 1,
    // Sent explicitly: the column defaults to `now()`, which is the TRANSACTION
    // time and therefore identical for every row of a bulk insert. Letting the
    // default win is what erased the order the rows were generated in.
    ...(row.createdAt ? { created_at: row.createdAt } : {}),
  };
}

export function recordToRow(record: SubmissionRecord): SubmissionRow {
  const frames = (record.frames ?? []).map((frame) => Number(frame));
  return {
    id: record.id,
    sessionId: record.session_id ?? null,
    questionId: record.question_id,
    videoId: record.video_id ?? "",
    frames,
    answer: record.answer ?? "",
    // Must round-trip exactly: a `generated` row read back as `submit` would
    // look hand-picked, and the generator would then refuse to ever replace it.
    source:
      record.source === "manual" || record.source === "generated"
        ? record.source
        : "submit",
    keyframeIds: align(record.keyframe_ids, frames.length),
    ptsTimes: align(record.pts_times, frames.length).map((value) =>
      value == null ? null : Number(value),
    ),
    retrievalDatabase: (record.retrieval_database === "infoshotpp"
      ? "infoshotpp"
      : "btc") as RetrievalDatabase,
    submittedBy: record.submitted_by ?? "unknown",
    revision: record.revision ?? 1,
    // Normalised, because the two sides format the same instant differently:
    // the client writes `…T19:04:00.123Z`, PostgREST returns
    // `…T19:04:00.123456+00:00`, and when the microseconds are zero it drops
    // the fraction entirely (`…T19:04:00+00:00`). Kept as text, that last form
    // sorts AFTER `…T19:04:00.500Z` — half a second later than it really is.
    createdAt: record.created_at ? new Date(record.created_at).toISOString() : undefined,
    updatedAt: record.updated_at,
    syncState: "synced",
  };
}

/** Merge a remote row into the local list, newest revision wins.
 *
 * A realtime event can arrive before or after the reply to our own write, and
 * a slow reply must never roll a row back, so the higher revision is kept. */
export function mergeRow(rows: SubmissionRow[], incoming: SubmissionRow): SubmissionRow[] {
  const index = rows.findIndex((row) => row.id === incoming.id);
  if (index < 0) return [...rows, incoming];
  const current = rows[index];
  if ((current.revision ?? 0) > (incoming.revision ?? 0)) return rows;
  const next = [...rows];
  next[index] = incoming;
  return next;
}

/** Order rows the way the panel and the CSV read them: oldest answer first.
 *
 * Deliberately keyed on creation, not update: ordering by `updated_at` would
 * make editing a frame move that answer to the bottom of its own CSV, silently
 * re-ranking the team's guesses. */
function createdMs(row: SubmissionRow): number {
  const parsed = Date.parse(row.createdAt ?? "");
  return Number.isNaN(parsed) ? 0 : parsed;
}

export function sortRows(rows: SubmissionRow[]): SubmissionRow[] {
  // By instant rather than by text. `recordToRow` normalises what comes back
  // from the server, but a row that never round-tripped is still in the client's
  // own format, and mixed formats do not compare as text: `…:00+00:00` sorts
  // after `…:00.500Z` even though it is half a second earlier. The uuid tiebreak
  // is a last resort for rows genuinely sharing a millisecond — `nextCreatedAt`
  // makes sure the ones we write never do.
  return [...rows].sort((a, b) => createdMs(a) - createdMs(b) || a.id.localeCompare(b.id));
}

/** Turn whatever Supabase rejected with into something an operator can act on.
 *
 *  PostgREST reports failures as plain `{ message, details, hint, code }`
 *  objects, not `Error` instances, so the usual
 *  `err instanceof Error ? err.message : String(err)` renders the useless
 *  "[object Object]". */
export function describeSupabaseError(error: unknown): string {
  if (!error) return "Unknown error";
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  const detail = error as { message?: string; details?: string; hint?: string; code?: string };
  const text = [detail.message, detail.details, detail.hint].filter(Boolean).join(" · ");
  if (!text) {
    try {
      return JSON.stringify(error);
    } catch {
      return String(error);
    }
  }
  return detail.code ? `${text} (${detail.code})` : text;
}

/** The one failure every team hits first: the migration has not been run. */
export function isMissingTableError(error: unknown): boolean {
  const detail = error as { code?: string; message?: string } | null;
  if (!detail) return false;
  if (detail.code === "PGRST205" || detail.code === "42P01") return true;
  return /schema cache|does not exist|could not find the table/i.test(detail.message ?? "");
}

export const MISSING_TABLE_HINT =
  "The `public.submissions` table is missing. Open Supabase → SQL Editor and run " +
  "supabase/migrations/001_shared_submission.sql, then reload the page.";

/** The project has not run migration 004, so `reorder_submissions` is missing.
 *
 *  Not an error the operator should ever see: the outbox is a strict FIFO that
 *  stops on the first failure, so retrying a call that can never succeed would
 *  wedge every later write behind it. The caller falls back to one PATCH per
 *  moved row instead. */
export function isMissingFunctionError(error: unknown): boolean {
  const detail = error as { code?: string; message?: string } | null;
  if (!detail) return false;
  // PGRST202: PostgREST could not find the function in its schema cache.
  // 42883: Postgres itself has no such function.
  if (detail.code === "PGRST202" || detail.code === "42883") return true;
  return /could not find the function|function .+ does not exist/i.test(detail.message ?? "");
}
