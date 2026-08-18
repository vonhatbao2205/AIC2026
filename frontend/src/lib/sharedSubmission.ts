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
    source: record.source === "manual" ? "manual" : "submit",
    keyframeIds: align(record.keyframe_ids, frames.length),
    ptsTimes: align(record.pts_times, frames.length).map((value) =>
      value == null ? null : Number(value),
    ),
    retrievalDatabase: (record.retrieval_database === "infoshotpp"
      ? "infoshotpp"
      : "btc") as RetrievalDatabase,
    submittedBy: record.submitted_by ?? "unknown",
    revision: record.revision ?? 1,
    createdAt: record.created_at,
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
export function sortRows(rows: SubmissionRow[]): SubmissionRow[] {
  return [...rows].sort(
    (a, b) => (a.createdAt ?? "").localeCompare(b.createdAt ?? "") || a.id.localeCompare(b.id),
  );
}

/** Turn whatever Supabase rejected with into something an operator can act on.
 *
 *  PostgREST reports failures as plain `{ message, details, hint, code }`
 *  objects, not `Error` instances, so the usual
 *  `err instanceof Error ? err.message : String(err)` renders the useless
 *  "[object Object]". */
export function describeSupabaseError(error: unknown): string {
  if (!error) return "Lỗi không xác định";
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
  "Chưa có bảng `public.submissions`. Mở Supabase → SQL Editor và chạy " +
  "supabase/migrations/001_shared_submission.sql, rồi tải lại trang.";
