/** The submission pack: one CSV per question, zipped under a `submission/` folder.
 *
 * Row shapes (from the organiser's format spec):
 *   kis    video_id,frame_idx
 *   qa     video_id,frame_idx,answer
 *   trake  video_id,frame_1,frame_2,…,frame_N
 *
 * No header row, UTF-8, comma delimiter, CRLF line endings.
 */
import type { ImportedQuestion, QuestionKind } from "./questions";
import { buildZip } from "./zip";

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
  /** Filled when the row came from a submit action rather than a manual edit. */
  source?: "submit" | "manual";
}

/** Identity used for the duplicate warning — the exact tuple the CSV will hold. */
export function rowKey(row: Pick<SubmissionRow, "videoId" | "frames" | "answer">, kind: QuestionKind): string {
  const base = `${row.videoId.trim()}|${row.frames.join(",")}`;
  // In Q&A the same frame with a different answer is a different guess, so the
  // text is part of the identity; elsewhere it is not part of the row at all.
  return kind === "qa" ? `${base}|${row.answer.trim().replace(/\s+/g, " ").toLowerCase()}` : base;
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

export interface RowProblem {
  rowId: string;
  message: string;
}

/** Everything that would make the organiser's parser reject or misread a row. */
export function validateRows(rows: SubmissionRow[], question: ImportedQuestion): RowProblem[] {
  const problems: RowProblem[] = [];
  const seen = new Map<string, string>();
  rows.forEach((row, index) => {
    const where = `dòng ${index + 1}`;
    if (!row.videoId.trim()) {
      problems.push({ rowId: row.id, message: `${where}: thiếu tên video` });
    } else if (/\.mp4$/i.test(row.videoId.trim())) {
      problems.push({ rowId: row.id, message: `${where}: tên video không được có đuôi .mp4` });
    }
    if (!row.frames.length || row.frames.some((frame) => !Number.isInteger(frame) || frame < 0)) {
      problems.push({ rowId: row.id, message: `${where}: frame phải là số nguyên ≥ 0` });
    }
    if (question.kind === "trake" && question.eventCount && row.frames.length !== question.eventCount) {
      problems.push({
        rowId: row.id,
        message: `${where}: cần đúng ${question.eventCount} frame (đang có ${row.frames.length})`,
      });
    }
    if (question.kind === "trake" && row.frames.some((frame, i) => i > 0 && frame <= row.frames[i - 1])) {
      problems.push({ rowId: row.id, message: `${where}: frame phải tăng dần theo thời gian` });
    }
    if (question.kind === "qa") {
      if (!row.answer.trim()) problems.push({ rowId: row.id, message: `${where}: thiếu answer` });
      else if (row.answer.length > MAX_ANSWER_LENGTH) {
        problems.push({ rowId: row.id, message: `${where}: answer dài ${row.answer.length} > ${MAX_ANSWER_LENGTH} ký tự` });
      }
    }
    const key = rowKey(row, question.kind);
    const previous = seen.get(key);
    if (previous) problems.push({ rowId: row.id, message: `${where}: trùng với dòng ${previous}` });
    else seen.set(key, String(index + 1));
  });
  if (rows.length > MAX_ROWS_PER_QUESTION) {
    problems.push({ rowId: rows[MAX_ROWS_PER_QUESTION]?.id ?? "", message: `Quá ${MAX_ROWS_PER_QUESTION} dòng` });
  }
  return problems;
}

/** Build the zip. Only questions that actually have rows get a CSV. */
export function buildSubmissionZip(questions: ImportedQuestion[], rows: SubmissionRow[]): Blob {
  const files = questions
    .map((question) => ({
      question,
      rows: rows.filter((row) => row.questionId === question.id).slice(0, MAX_ROWS_PER_QUESTION),
    }))
    .filter((entry) => entry.rows.length > 0)
    .map((entry) => ({
      // The spec is explicit that the CSVs must sit inside a `submission/`
      // folder rather than at the root of the archive.
      name: `${SUBMISSION_DIR}/${entry.question.id}.csv`,
      text: questionCsv(entry.rows, entry.question.kind),
    }));
  return buildZip(files);
}
