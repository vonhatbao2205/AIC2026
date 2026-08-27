import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api/client";
import type {
  ResolvedScope,
  RetrievalDatabase,
  ScopeCatalogue,
  ScopeMode,
  RerankerReport,
  SimpleResult,
} from "./api/types";
import { ScopeFilter } from "./components/ScopeFilter";
import { DEFAULT_SCOPE_MODE, orderCategories, scopeRequest } from "./lib/scope";
import { ThemeToggle } from "./components/ThemeToggle";

// Minimal vector-only view: one query box + slider + keyframe grid.
export default function SimpleSearch({
  onFullMode,
  onShowSettings,
  retrievalDatabase,
  onRetrievalDatabase,
}: {
  onFullMode: () => void;
  onShowSettings: () => void;
  retrievalDatabase: RetrievalDatabase;
  onRetrievalDatabase: (database: RetrievalDatabase) => void;
}) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(40);
  const [results, setResults] = useState<SimpleResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<string>("");
  const [latency, setLatency] = useState<number | null>(null);
  const [translated, setTranslated] = useState<string | null>(null);
  // Qwen3-VL reranking, default off — same opt-in contract as the console.
  const [rerank, setRerank] = useState(false);
  const [rerankAvailable, setRerankAvailable] = useState(false);
  const [rerankReport, setRerankReport] = useState<RerankerReport | null>(null);
  const [scopeMode, setScopeMode] = useState<ScopeMode>(DEFAULT_SCOPE_MODE);
  const [scopeSelection, setScopeSelection] = useState<string[]>([]);
  const [scopeCatalogue, setScopeCatalogue] = useState<ScopeCatalogue | null>(null);
  const [scopeError, setScopeError] = useState<string | null>(null);
  const [appliedScope, setAppliedScope] = useState<ResolvedScope | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // The folder catalogue is per profile; switching profile also drops folders
  // the new one does not hold, which would otherwise filter every search to zero.
  useEffect(() => {
    let cancelled = false;
    setScopeError(null);
    setAppliedScope(null);
    api.searchScope(retrievalDatabase).then(
      (body) => {
        if (cancelled) return;
        setScopeCatalogue(body);
        const held = new Set(body.categories.map((item) => item.category));
        setScopeSelection((current) => {
          const kept = current.filter((category) => held.has(category));
          return kept.length === current.length ? current : kept;
        });
      },
      () => {
        if (!cancelled) setScopeError("Không tải được danh sách thư mục — tạm thời tìm toàn bộ.");
      },
    );
    return () => {
      cancelled = true;
    };
  }, [retrievalDatabase]);

  // The tick box only appears when a rerank worker can actually answer; a
  // control that silently does nothing is worse than no control.
  useEffect(() => {
    let cancelled = false;
    api.health(retrievalDatabase).then(
      (body) => {
        if (cancelled) return;
        const available = Boolean(body.capabilities.visual_rerank);
        setRerankAvailable(available);
        if (!available) setRerank(false);
      },
      () => {
        if (!cancelled) setRerankAvailable(false);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [retrievalDatabase]);

  async function runSearch() {
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.simpleSearch(
        q, topK, retrievalDatabase, scopeRequest(scopeMode, scopeSelection), rerank,
      );
      setResults(res.results);
      setMode(res.mode);
      setLatency(res.latency_ms);
      setTranslated(res.translated_query ?? null);
      setAppliedScope(res.scope ?? null);
      setRerankReport(res.reranker ?? null);
    } catch (e) {
      let msg = "Search failed — backend không chạy? (kiểm tra http://localhost:8000/api/health)";
      if (e instanceof ApiError) {
        msg = typeof e.detail === "string" ? e.detail : `Search failed (HTTP ${e.status})`;
      }
      setError(msg);
      setResults([]);
      setRerankReport(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="simple-app">
      <header className="simple-head">
        <div className="simple-title">
          AIC<span>26</span> · Vector Search
          {mode && <span className={`mode-pill ${mode}`} style={{ marginLeft: 8 }}>{mode}</span>}
          <button className="btn sm ghost" style={{ marginLeft: 12 }} onClick={onFullMode}>
            Console ⤴
          </button>
          <select
            className="btn sm ghost"
            aria-label="Retrieval database"
            data-testid="retrieval-database"
            value={retrievalDatabase}
            onChange={(event) => onRetrievalDatabase(event.target.value as RetrievalDatabase)}
          >
            <option value="btc">BTC · đầy đủ</option>
            <option value="infoshotpp">InfoShot++ · PE only</option>
          </select>
          <span style={{ flex: 1 }} />
          <button className="btn sm ghost" onClick={onShowSettings} title="Cấu hình — import .env">
            ⚙ Cấu hình
          </button>
          <ThemeToggle />
        </div>

        <div className="simple-bar">
          <input
            ref={inputRef}
            className="simple-input"
            data-testid="query-input"
            placeholder="Nhập mô tả cảnh cần tìm (tiếng Anh cho kết quả tốt nhất)…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") runSearch();
            }}
          />
          <button className="btn primary" data-testid="search-btn" onClick={runSearch} disabled={loading}>
            {loading ? "Đang tìm…" : "Search"}
          </button>
        </div>

        <div className="simple-controls">
          <label className="slider-label">
            Số keyframe: <b data-testid="topk-value">{topK}</b>
          </label>
          <input
            type="range"
            min={10}
            max={200}
            step={10}
            value={topK}
            data-testid="topk-slider"
            onChange={(e) => setTopK(Number(e.target.value))}
            className="slider"
          />
          <ScopeFilter
            catalogue={scopeCatalogue}
            mode={scopeMode}
            onMode={setScopeMode}
            selected={scopeSelection}
            onSelected={(categories) => setScopeSelection(orderCategories(categories, scopeCatalogue))}
            applied={appliedScope}
            error={scopeError}
          />
          {rerankAvailable && (
            <label
              className="check-toggle"
              data-testid="rerank-toggle"
              title="Qwen3-VL chấm lại từng cặp (query, keyframe) trên tập ứng viên PE mở rộng. Chính xác hơn nhưng chậm hơn vài giây; worker lỗi thì giữ nguyên thứ tự PE."
            >
              <input
                type="checkbox"
                checked={rerank}
                data-testid="rerank-checkbox"
                onChange={(event) => setRerank(event.target.checked)}
              />
              <span>Rerank</span>
            </label>
          )}
          {rerankReport && (
            <span
              className={`latency-mini ${rerankReport.ok ? "" : "warn"}`}
              data-testid="rerank-report"
            >
              {rerankReport.ok
                ? `· ↕ rerank ${rerankReport.reranked}/${rerankReport.candidates} · ${Math.round(rerankReport.ms)} ms`
                : "· ⚠ rerank lỗi, giữ thứ tự PE"}
            </span>
          )}
          {latency != null && <span className="latency-mini">{latency} ms</span>}
          {results.length > 0 && <span className="latency-mini">· {results.length} kết quả</span>}
          {translated && (
            <span className="latency-mini" title="Query đã được dịch sang tiếng Anh cho PE encoder">
              🌐 EN: <b style={{ color: "var(--ch-vector)" }}>{translated}</b>
            </span>
          )}
        </div>
      </header>

      {mode === "mock" && (
        <div className="simple-warn" data-testid="mock-banner">
          ⚠ Backend đang ở <b>mock mode</b> — kết quả là dữ liệu giả cố định, không phụ thuộc query.
          Tắt mock (<span className="mono">AIC26_MOCK_MODE=false</span> trong <span className="mono">backend/.env</span>) và khởi động lại backend.
        </div>
      )}
      {error && <div className="simple-error" data-testid="error">{error}</div>}

      <main className="kf-grid" data-testid="results">
        {results.length === 0 && !loading && !error && (
          <div className="kf-empty">Nhập query và nhấn Enter để tìm keyframe.</div>
        )}
        {results.map((r, i) => (
          <figure key={r.submit_keyframe_id} className="kf-card" data-testid="kf-card">
            <div className="kf-rank">#{i + 1}</div>
            <img src={r.keyframe_url} alt={r.submit_keyframe_id} loading="lazy" />
            <figcaption>
              <span className="kf-id mono">{r.submit_keyframe_id}</span>
              <span className="kf-score mono">{r.score.toFixed(4)}</span>
            </figcaption>
          </figure>
        ))}
      </main>
    </div>
  );
}
