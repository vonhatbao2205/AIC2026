// Which DRES evaluation run belongs to which query type.
//
// The backend resolves this authoritatively at submit time (`resolve_evaluation`
// in submit_service.py); the same rule runs here so the console can SHOW the
// operator which run the next submit will go to, before they press Enter.
import type { DresEvaluation, QueryType } from "../api/types";

const KEYWORDS: Record<QueryType, RegExp[]> = {
  "T-KIS": [/t-?kis/, /textual/],
  "V-KIS": [/v-?kis/, /visual/],
  QA: [/\bqa\b/, /question/],
  TRAKE: [/trake/, /multi-?event/, /sequence/],
};

function haystack(evaluation: DresEvaluation): string {
  const templates = (evaluation.task_templates ?? [])
    .map((t) => `${t.taskGroup ?? ""} ${t.taskType ?? ""} ${t.name ?? ""}`)
    .join(" ");
  return `${evaluation.name ?? ""} ${templates}`.toLowerCase();
}

export function matchesQueryType(evaluation: DresEvaluation, queryType: QueryType): boolean {
  const text = haystack(evaluation);
  return (KEYWORDS[queryType] ?? []).some((re) => re.test(text));
}

/** The run a submit auto-routes to, or null when it is ambiguous. */
export function autoEvaluationId(
  evaluations: DresEvaluation[],
  queryType: QueryType,
): string | null {
  const active = evaluations.filter((e) => (e.status ?? "").toUpperCase() === "ACTIVE");
  const pool = active.length ? active : evaluations;
  const matching = pool.filter((e) => matchesQueryType(e, queryType));
  // A run with an open task beats one that is merely active.
  const running = matching.filter((e) => e.current_task);
  for (const candidates of [running, matching, pool]) {
    if (candidates.length === 1) return candidates[0].id;
  }
  return null;
}
