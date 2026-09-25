import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api/client";
import type {
  ImageEmbeddingModel,
  ResolvedScope,
  RetrievalDatabase,
  ScopeCatalogue,
  ScopeMode,
  RerankerReport,
  SimpleResult,
} from "./api/types";
import { ScopeFilter } from "./components/ScopeFilter";
import { DEFAULT_SCOPE_MODE, orderCategories, scopeItems, scopeRequest } from "./lib/scope";
import { ThemeToggle } from "./components/ThemeToggle";
import { ImageModelSelector } from "./components/ImageModelSelector";
import { PeTokenMeter } from "./components/PeTokenMeter";
import { DEFAULT_IMAGE_MODELS, imageModelsForSearch } from "./lib/imageModels";

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
  // VI→EN translation, default on: PE-Core is English-centric, so an
  // untranslated Vietnamese query returns frames that look plausible and are
  // not. Off searches with the words as typed.
  const [translate, setTranslate] = useState(true);
  // InfoShot++ can search either image index or RRF-fuse both. The component
  // prevents an empty selection; the request helper is a second safety net.
  const [imageModels, setImageModels] = useState<ImageEmbeddingModel[]>(() => [...DEFAULT_IMAGE_MODELS]);
  const [rerankAvailable, setRerankAvailable] = useState(false);
  const [rerankReport, setRerankReport] = useState<RerankerReport | null>(null);
  // A dead encoder degrades the search instead of emptying it, so the operator
  // has to be told which index actually answered before trusting the ranking.
  const [warnings, setWarnings] = useState<string[]>([]);
  const [answeredModels, setAnsweredModels] = useState<ImageEmbeddingModel[]>([]);
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
        const held = new Set(scopeItems(body));
        setScopeSelection((current) => {
          const kept = current.filter((category) => held.has(category));
          return kept.length === current.length ? current : kept;
        });
      },
      () => {
        if (!cancelled) setScopeError("Could not load folders; searching the entire dataset for now.");
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
        q,
        topK,
        retrievalDatabase,
        scopeRequest(scopeMode, scopeSelection),
        rerank,
        imageModelsForSearch(retrievalDatabase, imageModels),
        translate,
      );
      setResults(res.results);
      setMode(res.mode);
      setLatency(res.latency_ms);
      setTranslated(res.translated_query ?? null);
      setAppliedScope(res.scope ?? null);
      setRerankReport(res.reranker ?? null);
      const cut = res.pe_tokens?.queries.find((q) => q.truncated);
      setWarnings([
        ...(res.warnings ?? []),
        ...(cut
          ? [`PE-Core read only ${cut.limit} of ${cut.tokens} tokens; ignored: “${cut.dropped.slice(0, 160)}”`]
          : []),
      ]);
      setAnsweredModels(res.image_models ?? [...DEFAULT_IMAGE_MODELS]);
    } catch (e) {
      let msg = "Search failed — is the backend running? (check http://localhost:8000/api/health)";
      if (e instanceof ApiError) {
        msg = typeof e.detail === "string" ? e.detail : `Search failed (HTTP ${e.status})`;
      }
      setError(msg);
      setResults([]);
      setRerankReport(null);
      setWarnings([]);
      setAnsweredModels([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="simple-app">
      <header className="simple-head">
        <div className="simple-title">
          <div title="From clues to moments.">Clue<span>Scope</span> · Vector Search</div>
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
            <option value="btc">BTC · all channels</option>
            <option value="infoshotpp">InfoShot++ · select embedding</option>
          </select>
          <span style={{ flex: 1 }} />
          <button className="btn sm ghost" onClick={onShowSettings} title="Settings — import .env">
            ⚙ Settings
          </button>
          <ThemeToggle />
        </div>

        <div className="simple-bar">
          <input
            ref={inputRef}
            className="simple-input"
            data-testid="query-input"
            placeholder="Describe the scene to find (English works best)…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") runSearch();
            }}
          />
          <button className="btn primary" data-testid="search-btn" onClick={runSearch} disabled={loading}>
            {loading ? "Searching…" : "Search"}
          </button>
        </div>
        <PeTokenMeter
          query={query}
          translate={translate}
          active={imageModelsForSearch(retrievalDatabase, imageModels).includes("pe")}
        />

        <div className="simple-controls">
          <label className="slider-label">
            Keyframes: <b data-testid="topk-value">{topK}</b>
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
          <ImageModelSelector
            retrievalDatabase={retrievalDatabase}
            value={imageModels}
            onChange={setImageModels}
          />
          <label
            className="check-toggle"
            data-testid="translate-toggle"
            title="Translate the query to English for PE-Core. Disable for English queries, proper names or incorrect translations to search the original text."
          >
            <input
              type="checkbox"
              checked={translate}
              data-testid="translate-checkbox"
              onChange={(event) => setTranslate(event.target.checked)}
            />
            <span>Translate VI→EN</span>
          </label>
          {rerankAvailable && (
            <label
              className="check-toggle"
              data-testid="rerank-toggle"
              title="Pool candidates from selected image indices (PE + Qwen3-VL), then rerank query–keyframe pairs with Qwen3-VL. Adds latency; worker failures preserve retrieval order."
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
                : "· ⚠ reranking failed; preserving retrieval order"}
            </span>
          )}
          {latency != null && <span className="latency-mini">{latency} ms</span>}
          {results.length > 0 && <span className="latency-mini">· {results.length} results</span>}
          {translated && (
            <span className="latency-mini" title="Query translated to English for the PE encoder">
              🌐 EN: <b style={{ color: "var(--ch-vector)" }}>{translated}</b>
            </span>
          )}
        </div>
      </header>

      {mode === "mock" && (
        <div className="simple-warn" data-testid="mock-banner">
          ⚠ Backend is in <b>mock mode</b>; results are fixed fixtures and do not depend on the query.
          Disable mock mode (<span className="mono">AIC26_MOCK_MODE=false</span> in <span className="mono">backend/.env</span>) and restart the backend.
        </div>
      )}
      {error && <div className="simple-error" data-testid="error">{error}</div>}
      {warnings.map((warning) => (
        <div className="simple-warn" data-testid="search-warning" key={warning}>⚠ {warning}</div>
      ))}

      <main className="kf-grid" data-testid="results">
        {results.length === 0 && !loading && !error && (
          <div className="kf-empty">Enter a query and press Enter to find keyframes.</div>
        )}
        {results.map((r, i) => (
          <figure key={r.submit_keyframe_id} className="kf-card" data-testid="kf-card">
            <div className="kf-rank">#{i + 1}</div>
            <img src={r.keyframe_url} alt={r.submit_keyframe_id} loading="lazy" />
            <figcaption>
              <span className="kf-id mono">{r.submit_keyframe_id}</span>
              {answeredModels.length > 1 && (
                <span className="kf-models">
                  {(r.models ?? []).map((model) => (
                    <span
                      className={`badge ${model === "pe" ? "image_pe" : "image_qwen"}`}
                      key={model}
                      title={`cosine ${(r.per_model_score?.[model] ?? 0).toFixed(4)}`}
                    >
                      {model === "pe" ? "PE" : "Qwen"}
                    </span>
                  ))}
                </span>
              )}
              <span className="kf-score mono">{r.score.toFixed(4)}</span>
            </figcaption>
          </figure>
        ))}
      </main>
    </div>
  );
}
