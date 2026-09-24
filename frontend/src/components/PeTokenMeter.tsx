import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { PeLiveReport } from "../api/types";

/** Long enough to skip keystrokes, short enough to feel live. */
const DEBOUNCE_MS = 300;

export function clipText(text: string, max = 140): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

interface Props {
  query: string;
  /** Accumulated hints: the search joins them in front of the query. */
  hints?: string[];
  translate: boolean;
  /** Only PE-Core has a short text window; hidden when PE is not searched. */
  active: boolean;
  /** The LLM parser rewrites the visual query, so only the search can count it exactly. */
  llm?: boolean;
  /** TRAKE encodes each event on its own. */
  perEvent?: boolean;
}

/** How much of the query PE-Core will read.
 *
 *  PE keeps 70 text tokens and drops the rest without an error, so the end of a
 *  long description silently never reaches the vector. The count is exact when
 *  PE reads the text as typed and estimated when the search translates it first
 *  (the backend does not translate while the operator types); the search then
 *  reports the exact count of what it sent. */
export function PeTokenMeter({ query, hints = [], translate, active, llm = false, perEvent = false }: Props) {
  const [report, setReport] = useState<PeLiveReport | null>(null);
  const hintKey = hints.join("\u0001");

  useEffect(() => {
    if (!active || !(query.trim() || hints.length)) {
      setReport(null);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      api
        .peTokens({ query, previous_hints: hints, translate }, controller.signal)
        .then(setReport)
        .catch(() => {
          /* aborted by the next keystroke, or the backend is down: keep the last count */
        });
    }, DEBOUNCE_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
    // `hintKey` stands in for `hints`, whose identity changes on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, hintKey, translate, active]);

  if (!active || !report || report.status === "empty") return null;
  const range = report.range ? ` (${report.range[0]}–${report.range[1]})` : "";
  let detail = "";
  if (report.status === "near") detail = "near the limit";
  else if (report.status === "may_exceed")
    detail = report.at_risk
      ? `may exceed after translation — at risk: “${clipText(report.at_risk)}”`
      : "may exceed after translation";
  else if (report.status === "over")
    detail = `PE ${report.exact ? "ignores" : "likely ignores"}: “${clipText(report.dropped)}”`;
  const notes: string[] = [];
  if (!report.exact) notes.push("estimated for the English translation");
  if (llm) notes.push("the LLM rewrites the visual query — checked after Search");
  if (perEvent) notes.push("each TRAKE event is encoded separately — checked after Search");

  return (
    <div
      className={`pe-meter ${report.status}`}
      data-testid="pe-meter"
      title="PE-Core reads 70 text tokens (72 with its start/end markers) and silently drops the rest."
    >
      <span className="mono">
        PE {report.exact ? "" : "≈"}
        {report.tokens}/{report.limit} tokens{range}
      </span>
      {detail && <span> · {detail}</span>}
      {notes.length > 0 && <span className="pe-meter-note"> · {notes.join("; ")}</span>}
    </div>
  );
}
