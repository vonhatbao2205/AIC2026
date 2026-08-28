import type { HealthResponse, LatencyBreakdown, QueryType, RetrievalDatabase } from "../api/types";
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
  retrievalDatabase: RetrievalDatabase;
  onRetrievalDatabase: (database: RetrievalDatabase) => void;
  onSimpleMode: () => void;
  onShowKeymap: () => void;
  onShowSettings: () => void;
}

export function TopBar({ queryType, onQueryType, elapsed, penalties, latency, health, retrievalDatabase, onRetrievalDatabase, onSimpleMode, onShowKeymap, onShowSettings }: Props) {
  const dot = !health ? "warn" : health.ok ? "ok" : health.warnings.length ? "warn" : "bad";
  return (
    <div className="topbar">
      <div className="brand">
        AIC<span>26</span> Console
      </div>
      <select
        className="btn sm ghost"
        aria-label="Retrieval database"
        data-testid="retrieval-database"
        value={retrievalDatabase}
        onChange={(event) => onRetrievalDatabase(event.target.value as RetrievalDatabase)}
        title={
          retrievalDatabase === "btc"
            ? "BTC: đầy đủ mọi kênh"
            : "InfoShot++: chọn PE Core, Qwen3-VL Embedding hoặc fuse cả hai bằng RRF; V-KIS canvas chưa có"
        }
      >
        <option value="btc">BTC · đầy đủ</option>
        <option value="infoshotpp">InfoShot++ · PE / Qwen3-VL</option>
      </select>
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
      {/* Counts the WRONG verdicts DRES returned this session — the number the
          penalty gradient is actually charged on. */}
      <div className="metric" title="Wrong verdicts returned by DRES this session">
        <span className="k">Wrong</span>
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
      <button
        className="icon-btn"
        onClick={onShowSettings}
        aria-label="Configuration"
        title="Cấu hình — import .env"
        data-testid="settings-btn"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </svg>
      </button>
      <ThemeToggle />
    </div>
  );
}
