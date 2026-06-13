import type { HealthResponse, LatencyBreakdown, QueryType } from "../api/types";
import { formatTime } from "../lib/media";
import { ThemeToggle } from "./ThemeToggle";

const TYPES: QueryType[] = ["T-KIS", "QA", "V-KIS", "TRAKE"];

interface Props {
  queryType: QueryType;
  onQueryType: (t: QueryType) => void;
  elapsed: number;
  penalties: number;
  latency: LatencyBreakdown | null;
  health: HealthResponse | null;
  onSimpleMode: () => void;
  onShowKeymap: () => void;
}

export function TopBar({ queryType, onQueryType, elapsed, penalties, latency, health, onSimpleMode, onShowKeymap }: Props) {
  const dot = !health ? "warn" : health.ok ? "ok" : health.warnings.length ? "warn" : "bad";
  return (
    <div className="topbar">
      <div className="brand">
        AIC<span>26</span> Console
      </div>
      <div className="seg" role="tablist" aria-label="Query type">
        {TYPES.map((t) => (
          <button
            key={t}
            role="tab"
            aria-selected={queryType === t}
            className={queryType === t ? "active" : ""}
            onClick={() => onQueryType(t)}
          >
            {t}
          </button>
        ))}
      </div>
      <button className="btn sm ghost" onClick={onSimpleMode} title="Switch to the minimal vector-search view">
        Simple ⤴
      </button>
      <div className="spacer" />
      {latency && (
        <span className="latency-mini" title="parse / fusion / total ms">
          ⏱ {latency.parse_ms ?? 0} / {latency.fusion_ms ?? 0} / {latency.total_ms ?? 0} ms
        </span>
      )}
      <div className="metric">
        <span className="k">Timer</span>
        <span className="v mono">{formatTime(elapsed)}</span>
      </div>
      <div className="metric">
        <span className="k">Penalty</span>
        <span className={`v mono ${penalties > 0 ? "bad" : ""}`}>{penalties}</span>
      </div>
      <span
        className={`mode-pill ${health?.mode ?? ""}`}
        title={health?.warnings.join("; ") || "all services reachable"}
      >
        <span className={`health-dot ${dot}`} /> {health?.mode ?? "…"}
      </span>
      <button
        className="icon-btn"
        onClick={onShowKeymap}
        aria-label="Keyboard shortcuts"
        title="Keyboard shortcuts (Ctrl+/)"
        data-testid="keymap-btn"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="2" y="6" width="20" height="12" rx="2" />
          <path d="M6 10h0M10 10h0M14 10h0M18 10h0M6 14h0M18 14h0M9 14h6" />
        </svg>
      </button>
      <ThemeToggle />
    </div>
  );
}
