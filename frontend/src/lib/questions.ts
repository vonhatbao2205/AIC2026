/** The organiser's query pack: one .txt per question, type encoded in the name.
 *
 *   query-p1-1-kis.txt      -> T-KIS
 *   query-p1-15-qa.txt      -> QA
 *   query-p1-4-trake.txt    -> TRAKE, event count read from the E1..EN lines
 *
 * The file stem is the question's identity everywhere in the app, because the
 * submission pack has to name each CSV after its query (`query-p1-1-kis.csv`).
 */
import type { QueryType } from "../api/types";
import type { ZipTextEntry } from "./zip";

export type QuestionKind = "kis" | "qa" | "trake";

export const KIND_QUERY_TYPE: Record<QuestionKind, QueryType> = {
  kis: "T-KIS",
  qa: "QA",
  trake: "TRAKE",
};

/** V-KIS answers the same way textual KIS does — one video plus one frame — and
 *  the submission spec has no separate shape for it, so it maps onto `kis`. */
export function kindForQueryType(queryType: QueryType): QuestionKind {
  if (queryType === "QA") return "qa";
  if (queryType === "TRAKE") return "trake";
  return "kis";
}

export interface ImportedQuestion {
  /** File stem, e.g. "query-p1-15-qa" — also the CSV name on export. */
  id: string;
  /** Numeric part of the name, used to keep the pack in the organiser's order. */
  order: number;
  kind: QuestionKind;
  queryType: QueryType;
  text: string;
  /** TRAKE only: how many frames one submitted row must carry. */
  eventCount: number | null;
}

const NAME_RE = /^(?:.*\/)?([^/]*?-(\d+)-(kis|qa|trake|vkis))\.txt$/i;

/** How many frames one TRAKE row must carry — i.e. how many moments the
 *  statement asks the operator to locate.
 *
 *  Counts the marked DESCRIPTIONS, not the distinct numbers written on them.
 *  Statements are typed by hand and the numbering is not reliable: a real pack
 *  shipped `E1 / E2 / E2 / E4` for four events. Counting distinct indices gave 3
 *  there, so every correct four-frame row was rejected at export as the wrong
 *  width. Taking the highest index instead would be wrong the other way round,
 *  on a statement that merely skips a number.
 *
 *  Markers are the backend's (`app/trake_events.py`), which splits one
 *  retrieval per event: this validates the row width, and the two disagreeing
 *  means generated rows get rejected by the very export they were generated for.
 *
 *  Only E1, E2, … mark events, and only where a marker stands: at the start of
 *  the statement, a line or a sentence, or followed by a colon. A number in the
 *  prose ("(1)", "(4)", "cột số 4.") or an event mentioned in it ("như ở E1")
 *  is text: counting those widened every row. The colon is optional: a real
 *  organiser pack ships `E1 Khoảnh khắc…` with nothing but a space. */
export function trakeEventCount(text: string): number | null {
  const marker = /(?<![\p{L}\p{N}])E\d{1,2}(\s*:|\s*[.)\]–—-]|(?=\s)|$)/giu;
  let count = 0;
  for (const match of text.matchAll(marker)) {
    const before = text.slice(0, match.index).replace(/[ \t]+$/, "");
    if (match[1].includes(":") || !before || "\n.!?;,".includes(before[before.length - 1])) count += 1;
  }
  return count >= 2 ? count : null;
}

export function parseQuestionEntry(entry: ZipTextEntry): ImportedQuestion | null {
  // Zips made on macOS carry a parallel `__MACOSX/._name` resource fork per file.
  if (entry.name.includes("__MACOSX/") || entry.name.split("/").pop()?.startsWith("._")) {
    return null;
  }
  const match = NAME_RE.exec(entry.name);
  if (!match) return null;
  const [, id, order, rawKind] = match;
  const kind: QuestionKind = rawKind.toLowerCase() === "vkis" ? "kis" : (rawKind.toLowerCase() as QuestionKind);
  const text = entry.text.replace(/^﻿/, "").trim();
  return {
    id,
    order: Number(order),
    kind,
    // A `-vkis` file is still answered visually, so the tab opens as V-KIS even
    // though its CSV shape is the KIS one.
    queryType: rawKind.toLowerCase() === "vkis" ? "V-KIS" : KIND_QUERY_TYPE[kind],
    text,
    eventCount: kind === "trake" ? trakeEventCount(text) : null,
  };
}

export function parseQuestionPack(entries: ZipTextEntry[]): ImportedQuestion[] {
  const questions: ImportedQuestion[] = [];
  const seen = new Set<string>();
  for (const entry of entries) {
    const question = parseQuestionEntry(entry);
    if (!question || seen.has(question.id)) continue;
    seen.add(question.id);
    questions.push(question);
  }
  questions.sort((a, b) => a.order - b.order || a.id.localeCompare(b.id));
  return questions;
}

/** Short label for the tab rail / selector, e.g. "query-p1-15-qa" -> "15 · qa". */
export function questionLabel(question: ImportedQuestion): string {
  return `${question.order} · ${question.kind}`;
}
