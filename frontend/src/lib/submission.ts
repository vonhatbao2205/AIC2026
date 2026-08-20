/** The submission pack: one CSV per question, zipped under a `submission/` folder.
 *
 * Row shapes (from the organiser's format spec):
 *   kis    video_id,frame_idx
 *   qa     video_id,frame_idx,answer
 *   trake  video_id,frame_1,frame_2,…,frame_N
 *
 * No header row, UTF-8, comma delimiter, CRLF line endings.
 */
import type { RetrievalDatabase } from "../api/types";
import type { ImportedQuestion, QuestionKind } from "./questions";
import { buildZip, buildZipBytes } from "./zip";

export const MAX_ROWS_PER_QUESTION = 100;
export const MAX_ANSWER_LENGTH = 100;
export const SUBMISSION_DIR = "submission";

export interface SubmissionRow {
  id: string;
  questionId: string;
  videoId: string;
  /** One frame for kis/qa; one per event for trake. */
  frames: number[];
  answer: string;
  /** Where the row came from. `generated` is the one the answer generator owns:
   *  regenerating replaces those and leaves everything a person picked alone. */
  source?: "submit" | "manual" | "generated";

  // ---- shared-submission metadata (see supabase/migrations/001_*.sql) ----
  // `frames` stays the only value the CSV is built from. The two arrays below
  // are positionally parallel to it and exist so the panel can preview a row
  // without syncing any media; either entry may be null, because a frame taken
  // from a paused video need not be an extracted keyframe at all.
  keyframeIds?: (string | null)[];
  ptsTimes?: (number | null)[];
  /** Which profile's media the preview must resolve against. */
  retrievalDatabase?: RetrievalDatabase;
  /** Display name of whoever submitted it; local-only when working offline. */
  submittedBy?: string;
  /** Session (question pack) the answer belongs to, so a practice run and the
   *  real round can never be exported together. */
  sessionId?: string | null;
  /** Server revision this client last read, for optimistic concurrency. */
  revision?: number;
  createdAt?: string;
  updatedAt?: string;
  /** Local view of whether the shared store has this row yet. */
  syncState?: "synced" | "pending" | "local";
}

/** Video + frames, ignoring any answer text.
 *
 *  Two rows sharing this predict the same instant. In Q&A they can still be two
 *  different guesses, but a GENERATED row carries no answer yet, so one landing
 *  on an instant a person already picked is pure waste — it becomes an exact
 *  duplicate the moment the question's answer is filled in. */
export function frameKey(row: Pick<SubmissionRow, "videoId" | "frames">): string {
  return `${row.videoId.trim()}|${row.frames.join(",")}`;
}

/** Identity used for the duplicate warning — the exact tuple the CSV will hold. */
export function rowKey(row: Pick<SubmissionRow, "videoId" | "frames" | "answer">, kind: QuestionKind): string {
  // In Q&A the same frame with a different answer is a different guess, so the
  // text is part of the identity; elsewhere it is not part of the row at all.
  return kind === "qa"
    ? `${frameKey(row)}|${row.answer.trim().replace(/\s+/g, " ").toLowerCase()}`
    : frameKey(row);
}

export function findDuplicate(
  rows: SubmissionRow[],
  candidate: Pick<SubmissionRow, "questionId" | "videoId" | "frames" | "answer">,
  kind: QuestionKind,
  ignoreRowId?: string,
): SubmissionRow | null {
  const key = rowKey(candidate, kind);
  return (
    rows.find(
      (row) =>
        row.questionId === candidate.questionId &&
        row.id !== ignoreRowId &&
        rowKey(row, kind) === key,
    ) ?? null
  );
}

/** Quote only when the spec requires it: comma, quote, CR/LF or edge whitespace. */
export function csvField(value: string): string {
  if (!/[",\r\n]/.test(value) && value === value.trim()) return value;
  return `"${value.replace(/"/g, '""')}"`;
}

export function rowToCsvLine(row: SubmissionRow, kind: QuestionKind): string {
  const cells = [row.videoId.trim(), ...row.frames.map((frame) => String(frame))];
  if (kind === "qa") cells.push(row.answer);
  return cells.map(csvField).join(",");
}

export function questionCsv(rows: SubmissionRow[], kind: QuestionKind): string {
  return rows.map((row) => rowToCsvLine(row, kind)).join("\r\n");
}

/** `error` blocks the export; `warning` is shown but does not.
 *
 *  The split follows the organiser's rules, not taste: a malformed row is
 *  rejected by their parser, and a rejected submission still burns one of the
 *  three attempts allowed per pack. Duplicate predictions are nowhere forbidden,
 *  so they stay advisory. */
export type ProblemSeverity = "error" | "warning";

export interface RowProblem {
  rowId: string;
  message: string;
  severity: ProblemSeverity;
}

export interface SubmissionProblem extends RowProblem {
  questionId: string;
}

export interface SubmissionAudit {
  errors: SubmissionProblem[];
  warnings: SubmissionProblem[];
  /** Questions with no prediction at all. Exported as an empty CSV, not blocked. */
  unanswered: string[];
}

export class SubmissionFormatError extends Error {
  readonly problems: SubmissionProblem[];

  constructor(problems: SubmissionProblem[]) {
    super(
      `Không export: ${problems.length} lỗi định dạng.\n` +
        problems.map((problem) => `• ${problem.questionId} ${problem.message}`).join("\n"),
    );
    this.name = "SubmissionFormatError";
    this.problems = problems;
  }
}

/** True when a Q&A question is missing its answer text everywhere.
 *
 *  This is a property of the QUESTION, not of any one row: the answer generator
 *  fills a hundred frames and the text is typed once for all of them, so the
 *  usual state is a hundred rows with the same blank. Reported per row it is a
 *  hundred identical errors that bury every other problem in the panel — and it
 *  is one action to fix, not a hundred. */
export function qaAnswerMissingEverywhere(
  rows: SubmissionRow[],
  question: ImportedQuestion,
): boolean {
  return (
    question.kind === "qa" &&
    rows.length > 1 &&
    rows.every((row) => !row.answer.trim())
  );
}

/** Everything that would make the organiser's parser reject or misread a row. */
export function validateRows(rows: SubmissionRow[], question: ImportedQuestion): RowProblem[] {
  const problems: RowProblem[] = [];
  const seen = new Map<string, string>();
  const fail = (rowId: string, message: string) =>
    problems.push({ rowId, message, severity: "error" });
  // Still an error — the organisers require an answer — just stated once.
  const collapsedAnswer = qaAnswerMissingEverywhere(rows, question);
  if (collapsedAnswer) {
    fail(rows[0].id, `chưa có answer — cả ${rows.length} dòng sẽ bị chặn`);
  }
  rows.forEach((row, index) => {
    const where = `dòng ${index + 1}`;
    if (!row.videoId.trim()) {
      fail(row.id, `${where}: thiếu tên video`);
    } else if (/\.mp4$/i.test(row.videoId.trim())) {
      fail(row.id, `${where}: tên video không được có đuôi .mp4`);
    }
    if (!row.frames.length || row.frames.some((frame) => !Number.isInteger(frame) || frame < 0)) {
      fail(row.id, `${where}: frame phải là số nguyên ≥ 0`);
    }
    if (question.kind === "trake" && question.eventCount && row.frames.length !== question.eventCount) {
      fail(row.id, `${where}: cần đúng ${question.eventCount} frame (đang có ${row.frames.length})`);
    }
    if (question.kind === "trake" && row.frames.some((frame, i) => i > 0 && frame <= row.frames[i - 1])) {
      fail(row.id, `${where}: frame phải tăng dần theo thời gian`);
    }
    if (question.kind === "qa") {
      // Itemised only once SOME rows have an answer: then which ones are blank
      // is real per-row information rather than the same sentence repeated.
      if (!row.answer.trim()) {
        if (!collapsedAnswer) fail(row.id, `${where}: thiếu answer`);
      } else if (row.answer.length > MAX_ANSWER_LENGTH) {
        fail(row.id, `${where}: answer dài ${row.answer.length} > ${MAX_ANSWER_LENGTH} ký tự`);
      }
    }
    const key = rowKey(row, question.kind);
    const previous = seen.get(key);
    // Predicting the same frame twice wastes a line but is not a format error.
    if (previous) {
      problems.push({ rowId: row.id, message: `${where}: trùng với dòng ${previous}`, severity: "warning" });
    } else {
      seen.set(key, String(index + 1));
    }
  });
  if (rows.length > MAX_ROWS_PER_QUESTION) {
    fail(rows[MAX_ROWS_PER_QUESTION]?.id ?? "", `quá ${MAX_ROWS_PER_QUESTION} dòng (đang có ${rows.length})`);
  }
  return problems;
}

export function rowsForQuestion(rows: SubmissionRow[], questionId: string): SubmissionRow[] {
  return rows.filter((row) => row.questionId === questionId);
}

/** Whole-pack check, run before anything is written. */
export function auditSubmission(
  questions: ImportedQuestion[],
  rows: SubmissionRow[],
): SubmissionAudit {
  const errors: SubmissionProblem[] = [];
  const warnings: SubmissionProblem[] = [];
  const unanswered: string[] = [];
  for (const question of questions) {
    const questionRows = rowsForQuestion(rows, question.id);
    if (questionRows.length === 0) unanswered.push(question.id);
    for (const problem of validateRows(questionRows, question)) {
      (problem.severity === "error" ? errors : warnings).push({ ...problem, questionId: question.id });
    }
  }
  return { errors, warnings, unanswered };
}

/** Build the zip, or refuse.
 *
 *  The guard lives here rather than only in the UI so no call site can produce a
 *  file the organiser would reject — a rejected upload still counts as one of
 *  the three attempts.
 *
 *  Every question in the pack gets a CSV, including ones with no prediction:
 *  the spec asks for one file per query, so an unanswered question is an empty
 *  file rather than a missing one. */
export function buildSubmissionFiles(
  questions: ImportedQuestion[],
  rows: SubmissionRow[],
): { name: string; text: string }[] {
  const audit = auditSubmission(questions, rows);
  if (audit.errors.length) throw new SubmissionFormatError(audit.errors);
  return questions.map((question) => ({
    // The spec is explicit that the CSVs must sit inside a `submission/`
    // folder rather than at the root of the archive.
    name: `${SUBMISSION_DIR}/${question.id}.csv`,
    text: questionCsv(rowsForQuestion(rows, question.id), question.kind),
  }));
}

/** Raw bytes, so the archive can be inspected without a Blob reader. */
export function buildSubmissionZipBytes(
  questions: ImportedQuestion[],
  rows: SubmissionRow[],
): Uint8Array {
  return buildZipBytes(buildSubmissionFiles(questions, rows));
}

export function buildSubmissionZip(questions: ImportedQuestion[], rows: SubmissionRow[]): Blob {
  return buildZip(buildSubmissionFiles(questions, rows));
}
