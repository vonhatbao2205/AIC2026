import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api/client";
import type { SimpleResult } from "./api/types";
import { ThemeToggle } from "./components/ThemeToggle";

// Minimal vector-only view: one query box + slider + keyframe grid.
export default function SimpleSearch({ onFullMode }: { onFullMode: () => void }) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(40);
  const [results, setResults] = useState<SimpleResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<string>("");
  const [latency, setLatency] = useState<number | null>(null);
  const [translated, setTranslated] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  async function runSearch() {
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.simpleSearch(q, topK);
      setResults(res.results);
      setMode(res.mode);
      setLatency(res.latency_ms);
      setTranslated(res.translated_query ?? null);
    } catch (e) {
      let msg = "Search failed — backend không chạy? (kiểm tra http://localhost:8000/api/health)";
      if (e instanceof ApiError) {
        msg = typeof e.detail === "string" ? e.detail : `Search failed (HTTP ${e.status})`;
      }
      setError(msg);
      setResults([]);
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
          <span style={{ flex: 1 }} />
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
