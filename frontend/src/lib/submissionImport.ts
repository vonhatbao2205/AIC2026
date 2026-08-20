/** Read a `submission.zip` back into the table it was exported from.
 *
 *  The export is one CSV per question under `submission/`, no header row, and
 *  the row shape depends on the question kind:
 *
 *      kis    video_id,frame_idx
 *      qa     video_id,frame_idx,answer
 *      trake  video_id,frame_1,…,frame_N
 *
 *  Which means a row cannot be parsed without knowing the kind: `L01_V028,1200,5`
 *  is a two-event TRAKE row or a Q&A row answered "5", and nothing in the line
 *  itself says which. The kind comes from the imported question pack, falling
 *  back to the `-kis|-qa|-trake` suffix the organiser puts in every file name.
 */
import { kindForQueryType, type ImportedQuestion, type QuestionKind } from "./questions";
import { MAX_ROWS_PER_QUESTION, type SubmissionRow } from "./submission";
import type { ZipTextEntry } from "./zip";
import { ZipError } from "./zip";

/** A CSV the archive holds, already attributed to a question. */
export interface ImportedAnswerFile {
  questionId: string;
  kind: QuestionKind;
  /** False when the question is not in the pack currently loaded. */
  inPack: boolean;
  rows: Omit<SubmissionRow, "id">[];
  /** Lines that could not be read, with the reason. Never silently dropped. */
  problems: string[];
}

export interface ImportedSubmission {
  files: ImportedAnswerFile[];
  totalRows: number;
  /** Questions in the archive that the current pack does not contain: they can
   *  be imported but will not be exported, so the operator has to see them. */
  orphanIds: string[];
  problems: string[];
}

const CSV_NAME_RE = /^(?:.*\/)?([^/]+)\.csv$/i;
const KIND_SUFFIX_RE = /-(kis|qa|trake|vkis)$/i;

/** One CSV line into cells, honouring the quoting `csvField` writes.
 *
 *  A Q&A answer may legitimately contain a comma ("Năm, sáu"), which is exactly
 *  why the writer quotes; splitting on "," would tear such a row in half. */
export function parseCsvLine(line: string): string[] {
  const cells: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (quoted) {
      if (char !== '"') cell += char;
      else if (line[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else quoted = false;
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      cells.push(cell);
      cell = "";
    } else {
      cell += char;
    }
  }
  cells.push(cell);
  return cells;
}

function frameOf(cell: string): number | null {
  const text = cell.trim();
  if (!/^\d+$/.test(text)) return null;
  const value = Number(text);
  return Number.isSafeInteger(value) ? value : null;
}

/** Rows of one question's CSV. Problems are reported, never guessed past. */
export function parseAnswerCsv(
  text: string,
  questionId: string,
  kind: QuestionKind,
): { rows: Omit<SubmissionRow, "id">[]; problems: string[] } {
  const rows: Omit<SubmissionRow, "id">[] = [];
  const problems: string[] = [];
  const lines = text.replace(/^﻿/, "").split(/\r\n|\n|\r/);

  lines.forEach((line, index) => {
    if (!line.trim()) return; // a trailing newline is not a row
    const where = `${questionId} dòng ${index + 1}`;
    const cells = parseCsvLine(line);
    const [videoCell, ...rest] = cells;
    const videoId = (videoCell ?? "").trim();
    if (!videoId) {
      problems.push(`${where}: thiếu tên video`);
      return;
    }
    // In Q&A the last cell is the answer, whatever it looks like; everything
    // between the video and it is the frame.
    const answer = kind === "qa" ? (rest.pop() ?? "").trim() : "";
    const frames: number[] = [];
    for (const cell of rest) {
      const frame = frameOf(cell);
      if (frame === null) {
        problems.push(`${where}: "${cell.trim()}" không phải frame hợp lệ`);
        return;
      }
      frames.push(frame);
    }
    if (!frames.length) {
      problems.push(`${where}: không có frame nào`);
      return;
    }
    rows.push({
      questionId,
      videoId,
      frames,
      answer,
      // Somebody's earlier work, so the answer generator must treat it the way
      // it treats a hand-picked row: keep it, never replace it.
      source: "manual",
      keyframeIds: frames.map(() => null),
      ptsTimes: frames.map(() => null),
    });
  });

  if (rows.length > MAX_ROWS_PER_QUESTION) {
    problems.push(
      `${questionId}: ${rows.length} dòng > ${MAX_ROWS_PER_QUESTION}, chỉ lấy ${MAX_ROWS_PER_QUESTION} dòng đầu`,
    );
    rows.length = MAX_ROWS_PER_QUESTION;
  }
  return { rows, problems };
}

/** Kind of a question id, from the pack if it is there and from the organiser's
 *  file-name suffix if it is not. */
function kindOf(questionId: string, byId: Map<string, ImportedQuestion>): QuestionKind | null {
  const known = byId.get(questionId);
  if (known) return known.kind;
  const suffix = KIND_SUFFIX_RE.exec(questionId);
  if (!suffix) return null;
  const raw = suffix[1].toLowerCase();
  // `-vkis` is answered exactly like textual KIS; the CSV has no separate shape.
  return raw === "vkis" ? "kis" : (raw as QuestionKind);
}

export function parseSubmissionPack(
  entries: ZipTextEntry[],
  questions: ImportedQuestion[],
): ImportedSubmission {
  const byId = new Map(questions.map((question) => [question.id, question]));
  const files: ImportedAnswerFile[] = [];
  const problems: string[] = [];
  const orphanIds: string[] = [];

  for (const entry of entries) {
    // Zips made on macOS carry a parallel `__MACOSX/._name` resource fork.
    if (entry.name.includes("__MACOSX/") || entry.name.split("/").pop()?.startsWith("._")) {
      continue;
    }
    const match = CSV_NAME_RE.exec(entry.name);
    if (!match) continue;
    const questionId = match[1];
    const kind = kindOf(questionId, byId);
    if (!kind) {
      problems.push(
        `${entry.name}: không biết đây là câu KIS/QA hay TRAKE (không có trong gói câu hỏi, ` +
          "tên file cũng không kết thúc bằng -kis/-qa/-trake)",
      );
      continue;
    }
    const parsed = parseAnswerCsv(entry.text, questionId, kind);
    const inPack = byId.has(questionId);
    if (!inPack) orphanIds.push(questionId);
    files.push({ questionId, kind, inPack, rows: parsed.rows, problems: parsed.problems });
  }

  if (!files.length) {
    // A question pack is a zip too, and both live next to each other on disk.
    const looksLikeQuestionPack = entries.some((entry) => /\.txt$/i.test(entry.name));
    throw new ZipError(
      looksLikeQuestionPack
        ? "File này là gói CÂU HỎI (.txt), không phải gói đáp án. Dùng nút “⭳ Import câu hỏi”."
        : "Không tìm thấy file .csv nào trong zip — gói đáp án phải có submission/<tên câu>.csv",
    );
  }

  files.sort((a, b) => a.questionId.localeCompare(b.questionId));
  return {
    files,
    totalRows: files.reduce((total, file) => total + file.rows.length, 0),
    orphanIds,
    problems,
  };
}

export type ImportMode = "replace" | "fill";

/** Which questions an import would actually write, given what is on screen.
 *
 *  `replace` overwrites a question's rows outright; `fill` only touches questions
 *  that hold nothing yet, so an operator restoring a backup cannot wipe answers
 *  a teammate added in the meantime. */
export function planImport(
  parsed: ImportedSubmission,
  existing: SubmissionRow[],
  mode: ImportMode,
): { write: ImportedAnswerFile[]; skipped: ImportedAnswerFile[]; replacedRows: number } {
  const counts = new Map<string, number>();
  for (const row of existing) {
    counts.set(row.questionId, (counts.get(row.questionId) ?? 0) + 1);
  }
  const write: ImportedAnswerFile[] = [];
  const skipped: ImportedAnswerFile[] = [];
  let replacedRows = 0;
  for (const file of parsed.files) {
    const held = counts.get(file.questionId) ?? 0;
    if (!file.rows.length || (mode === "fill" && held > 0)) {
      skipped.push(file);
      continue;
    }
    replacedRows += held;
    write.push(file);
  }
  return { write, skipped, replacedRows };
}
