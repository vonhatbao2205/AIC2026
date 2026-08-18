/** The question pack as shared state: canonical form, fingerprint, wire shape.
 *
 *  Kept free of React and of Supabase so the thing five machines compare
 *  themselves against can be tested directly.
 */
import type { QueryType } from "../api/types";
import type { ImportedQuestion, QuestionKind } from "./questions";

export interface SessionInfo {
  id: string;
  room: string;
  name: string;
  packHash: string;
  questionCount: number;
  publishedBy: string;
  createdAt: string;
}

export interface QuestionRecord {
  session_id?: string;
  question_id: string;
  ordinal: number;
  kind: string;
  query_type: string;
  text: string;
  event_count: number | null;
}

/** Field-for-field canonical form. Order and shape must not depend on how the
 *  pack was produced, or two machines holding the same questions would compute
 *  different fingerprints and chase each other's "updates" forever. */
export function canonicalQuestions(questions: ImportedQuestion[]): QuestionRecord[] {
  return [...questions]
    .sort((a, b) => a.order - b.order || a.id.localeCompare(b.id))
    .map((question) => ({
      question_id: question.id,
      ordinal: question.order,
      kind: question.kind,
      query_type: question.queryType,
      text: question.text,
      event_count: question.eventCount,
    }));
}

/** Content fingerprint used to answer one question: is my cached pack the one
 *  the team is using?
 *
 *  Deliberately a plain FNV-1a over the canonical JSON rather than SHA-256 via
 *  `crypto.subtle`: this only has to be stable and identical on every client,
 *  it is not a security boundary, and subtle's async, secure-context-only API
 *  is absent in enough environments that a fallback would risk two machines
 *  hashing the same pack differently — the one failure mode that matters here. */
export function packHash(questions: ImportedQuestion[]): string {
  const canonical = JSON.stringify(canonicalQuestions(questions));
  let hash = 0x811c9dc5;
  for (let i = 0; i < canonical.length; i += 1) {
    hash ^= canonical.charCodeAt(i);
    // 32-bit FNV prime multiply, kept in range without BigInt.
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  // Fold the length in so two packs differing only by a truncation still differ.
  const salted = (hash ^ Math.imul(canonical.length, 0x9e3779b1)) >>> 0;
  return salted.toString(16).padStart(8, "0");
}

export function recordToQuestion(record: QuestionRecord): ImportedQuestion {
  return {
    id: record.question_id,
    order: Number(record.ordinal),
    kind: record.kind as QuestionKind,
    queryType: record.query_type as QueryType,
    text: record.text ?? "",
    eventCount: record.event_count ?? null,
  };
}

export function recordsToQuestions(records: QuestionRecord[]): ImportedQuestion[] {
  return records.map(recordToQuestion).sort((a, b) => a.order - b.order || a.id.localeCompare(b.id));
}

/** Counts shown in the publish preview, so nobody pushes the wrong pack. */
export function packSummary(questions: ImportedQuestion[]): {
  total: number;
  byKind: Record<QuestionKind, number>;
} {
  const byKind: Record<QuestionKind, number> = { kis: 0, qa: 0, trake: 0 };
  for (const question of questions) byKind[question.kind] += 1;
  return { total: questions.length, byKind };
}

/** Default session name: the pack's own prefix plus the day it was published. */
export function defaultSessionName(questions: ImportedQuestion[]): string {
  const prefix = questions[0]?.id.replace(/-\d+-(kis|qa|trake)$/i, "") || "pack";
  return `${prefix} · ${new Date().toISOString().slice(0, 10)}`;
}
