import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { ProgressiveConfig, ProgressiveHint, ProgressiveSnapshot } from "../api/types";
import "./progressive.css";

interface Props {
  config: ProgressiveConfig;
  initialText: string;
  onSnapshot: (snapshot: ProgressiveSnapshot) => void;
  onBusy: (busy: boolean) => void;
}

/** Mounted separately per console tab/configuration. Never stored in localStorage. */
export function ProgressivePanel({ config, initialText, onSnapshot, onBusy }: Props) {
  const [ledger, setLedger] = useState<ProgressiveHint[]>([]);
  const [draft, setDraft] = useState(initialText);
  const [mode, setMode] = useState<"delta" | "cumulative">("delta");
  const [snapshot, setSnapshot] = useState<ProgressiveSnapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dirty, setDirty] = useState(false);
  const session = useRef<{ id: string; revision: number } | null>(null);
  const pending = useRef<{ expected_revision: number; client_request_id: string; hints: ProgressiveHint[] } | null>(null);
  const generation = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const running = useRef(false);
  const callbacks = useRef({ onSnapshot, onBusy });
  callbacks.current = { onSnapshot, onBusy };

  useEffect(() => () => {
    generation.current++;
    controller.current?.abort();
    if (session.current) void api.closeProgressive(session.current.id).catch(() => {});
    callbacks.current.onBusy(false);
  }, []);

  async function run(retry = false) {
    if (running.current) return;
    const next = retry ? pending.current?.hints : draft.trim()
      ? [...ledger, { hint_id: crypto.randomUUID(), raw_text: draft.trim(), input_mode: mode, enabled: true }]
      : ledger;
    if (!next || (!retry && !next.length && !session.current)) return;
    const owner = ++generation.current;
    const abort = new AbortController();
    controller.current = abort;
    running.current = true;
    setBusy(true);
    callbacks.current.onBusy(true);
    setError("");
    try {
      if (!session.current) {
        // Let creation finish so an unmounted tab can close the returned ID.
        const created = await api.createProgressive(config);
        if (owner !== generation.current) {
          void api.closeProgressive(created.session_id).catch(() => {});
          return;
        }
        session.current = { id: created.session_id, revision: created.revision };
      }
      if (!retry || !pending.current) {
        pending.current = { expected_revision: session.current.revision, client_request_id: crypto.randomUUID(), hints: next };
      }
      const result = await api.updateProgressive(session.current.id, pending.current, abort.signal);
      if (owner !== generation.current || result.session_id !== session.current.id
          || result.committed_revision !== result.revision) return;
      session.current.revision = result.revision;
      setLedger(next);
      setDraft("");
      setDirty(false);
      setSnapshot(result);
      pending.current = null;
      callbacks.current.onSnapshot(result);
    } catch (cause) {
      if (owner !== generation.current) return;
      const detail = cause instanceof ApiError && typeof cause.detail === "string" ? cause.detail : "Không nhận được kết quả. Retry dùng lại cùng request ID.";
      setError(detail);
      // A definite server failure consumes a revision; resync before allowing a
      // new attempt. An ambiguous network failure must retry the identical body.
      if (cause instanceof ApiError && session.current) {
        try {
          const current = await api.getProgressive(session.current.id, abort.signal);
          if (owner !== generation.current) return;
          session.current.revision = current.revision;
          pending.current = null;
        } catch (syncError) {
          if (owner !== generation.current) return;
          if (syncError instanceof ApiError && syncError.status === 410) {
            session.current = null;
            pending.current = null;
            setError("Session đã hết hạn. Chạy lại để tạo session và replay các hint hiện hành.");
          }
        }
      }
    } finally {
      if (owner === generation.current) {
        running.current = false;
        setBusy(false);
        callbacks.current.onBusy(false);
      }
    }
  }

  const locked = busy || pending.current !== null;
  return <section className="phm-panel" aria-label="Progressive Hint Memory">
    <strong>Progressive Hint Memory</strong>
    <p>Toàn corpus · {config.image_models.join(" + ")} · 200 hits/model/view · {config.hybrid ? "hybrid" : "visual"}. Parser heuristic; expansion, reranker và TARA tắt.</p>
    {ledger.map((hint, i) => <div className="phm-hint" key={hint.hint_id}>
      <label><input type="checkbox" checked={hint.enabled} disabled={locked} onChange={e => {
        setLedger(ledger.map(h => h.hint_id === hint.hint_id ? { ...h, enabled: e.target.checked } : h)); setDirty(true);
      }} /> H{i + 1} · {hint.input_mode}</label>
      <textarea aria-label={`Nội dung hint ${i + 1}`} value={hint.raw_text} disabled={locked} onChange={e => {
        setLedger(ledger.map(h => h.hint_id === hint.hint_id ? { ...h, raw_text: e.target.value } : h)); setDirty(true);
      }} />
    </div>)}
    <label>Hint đã công bố
      <select aria-label="Kiểu hint" value={mode} disabled={locked} onChange={e => setMode(e.target.value as typeof mode)}>
        <option value="delta">Phần mô tả mới</option><option value="cumulative">Mô tả tích lũy</option>
      </select>
      <textarea aria-label="Hint mới" value={draft} disabled={locked} onChange={e => setDraft(e.target.value)} placeholder="Nhập hint đã được tiết lộ…" />
    </label>
    <button className="btn primary" disabled={locked || ledger.some(h => !h.raw_text.trim())} onClick={() => void run()}> {busy ? "Đang truy hồi…" : "Áp dụng hint và tìm kiếm"}</button>
    {pending.current && !busy && <button className="btn" onClick={() => void run(true)}>Retry request</button>}
    {dirty && <p role="status">Ledger đã sửa; bấm áp dụng để replay. Kết quả đang xem thuộc revision trước.</p>}
    {error && <p role="alert">{error}</p>}
    {snapshot && <div aria-live="polite">
      <p>{snapshot.mode} · lượt {snapshot.turn} · revision {snapshot.committed_revision}{snapshot.degraded ? " · degraded" : ""}</p>
      <p>{snapshot.hint_ledger?.length ?? 0} hint độc lập có hiệu lực · {snapshot.revision_latency_ms ?? snapshot.latency_ms?.total_ms ?? 0} ms{snapshot.no_op ? " · không có hint mới" : ""}</p>
      <p>Top-1 streak: {snapshot.stability?.top1_streak ?? 0} · giao top-10: {snapshot.stability?.top10_jaccard?.toFixed(2) ?? "—"}. Ổn định không đồng nghĩa đúng.</p>
      <p>Revision: {snapshot.revision_budget?.query_vectors ?? 0} query vectors · {snapshot.revision_budget?.index_calls ?? 0} index calls. Lượt cuối: {snapshot.budget?.admitted_pairs ?? 0}/16 cặp rescue/backfill.</p>
      {snapshot.warnings?.map(w => <p key={w}>{w}</p>)}
    </div>}
  </section>;
}
