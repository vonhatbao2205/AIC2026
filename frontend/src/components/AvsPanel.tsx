import { useEffect, useRef, useState } from "react";
import { api, ApiError, type SubmitBody } from "../api/client";
import type { FrameResult, ImageEmbeddingModel, RetrievalDatabase, SubmitPreview, VideoGroup } from "../api/types";

interface AcceptedMoment { frame: FrameResult; verdict?: string; submitted?: boolean }
interface Props {
  groups: VideoGroup[];
  query: string;
  database: RetrievalDatabase;
  imageModels: ImageEmbeddingModel[];
  scopeId: string;
  dresEnabled: boolean;
  evaluationId: string | null;
  taskName: string;
  selected: FrameResult | null;
  requested?: { frame: FrameResult; id: number } | null;
  onInspect: (frame: FrameResult) => void;
}
const keyOf = (f: FrameResult) => `${f.video_id}:${f.frame_idx}`;
export function avsConflict(frame: FrameResult, accepted: AcceptedMoment[]): "exact" | "near" | null {
  if (accepted.some(a => keyOf(a.frame) === keyOf(frame))) return "exact";
  if (frame.pts_time != null && accepted.some(a => a.frame.video_id === frame.video_id
    && a.frame.pts_time != null && Math.abs(a.frame.pts_time - frame.pts_time!) <= 2)) return "near";
  return null;
}
function message(e: unknown) {
  if (e instanceof ApiError) return (e.detail as { message?: string })?.message ?? e.message;
  return e instanceof Error ? e.message : "Request failed.";
}

/** Mounted with a key per collection/task. Refining a query preserves its basket. */
export function AvsPanel(props: Props) {
  const storageKey = `aic26_avs:${props.database}:${props.scopeId}`;
  const [accepted, setAccepted] = useState<AcceptedMoment[]>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) ?? "[]");
      return Array.isArray(saved) ? saved.filter(a => a?.frame?.video_id && Number.isInteger(a.frame.frame_idx)) : [];
    } catch { return []; }
  });
  const [queue, setQueue] = useState<FrameResult[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [near, setNear] = useState<FrameResult | null>(null);
  const [review, setReview] = useState<FrameResult | null>(null);
  const [preview, setPreview] = useState<{ frame: FrameResult; body: SubmitBody; data: SubmitPreview } | null>(null);
  const request = useRef(0);
  const sending = useRef(false);
  useEffect(() => {
    try { localStorage.setItem(storageKey, JSON.stringify(accepted)); }
    catch { setError("Could not persist the basket in this browser. Keep this tab open."); }
  }, [accepted, storageKey]);
  useEffect(() => {
    request.current += 1; setQueue([]); setBusy(false); setPreview(null); setNear(null);
    return () => { request.current += 1; };
  }, [props.groups, props.evaluationId, props.taskName]);
  useEffect(() => {
    if (props.requested) accept(props.requested.frame);
  }, [props.requested]);

  function accept(frame: FrameResult, allowNear = false) {
    if (frame.frame_idx == null || frame.pts_time == null) {
      setError("Inspect a moment with a source frame and timestamp before accepting it."); return;
    }
    const conflict = avsConflict(frame, accepted);
    if (conflict === "exact") { setError("Duplicate: this moment is already in the basket."); return; }
    if (conflict === "near" && !allowNear) { setNear(frame); setError("Nearby moment: this video already has an accepted result within 2 seconds."); return; }
    setAccepted(a => [...a, { frame }]); setNear(null); setError("");
  }
  async function prepareQueue() {
    const revision = ++request.current;
    setBusy(true); setError("");
    try {
      const result = await api.generateAnswers({ retrieval_database: props.database, image_models: props.imageModels,
        query: props.query, query_type_hint: "AVS", groups: props.groups, limit: 100,
        taken: accepted.map(a => ({ video_id: a.frame.video_id, frames: [a.frame.frame_idx!] })) });
      if (request.current !== revision) return;
      const frames = props.groups.flatMap(g => g.frames);
      setQueue(result.answers.flatMap(a => {
        const frame = frames.find(f => f.video_id === a.video_id && f.frame_idx === a.frames[0]);
        return frame ? [frame] : [];
      }));
    } catch (e) { if (request.current === revision) setError(message(e)); }
    finally { if (request.current === revision) setBusy(false); }
  }
  async function prepareSubmit(frame: FrameResult) {
    const revision = ++request.current;
    const body: SubmitBody = { query_type: "AVS", evaluation_id: props.evaluationId,
      task_name: props.taskName || null, answer_mode: "temporal", segment_pad_ms: 0,
      payload: { video_id: frame.video_id, frame_idx: frame.frame_idx, timestamp: frame.pts_time,
        fps: frame.fps, submit_keyframe_id: frame.submit_keyframe_id } };
    setBusy(true); setError(""); setPreview(null);
    try {
      const data = await api.submitPreview(body);
      if (request.current === revision) setPreview({ frame, body, data });
    } catch (e) { if (request.current === revision) setError(message(e)); }
    finally { if (request.current === revision) setBusy(false); }
  }
  async function submit() {
    if (!preview || sending.current || !props.dresEnabled || preview.data.duplicate_key || preview.data.answer_mode_mismatch) return;
    sending.current = true; setBusy(true); setError("");
    try {
      const entry = await api.submit({ ...preview.body, evaluation_id: preview.data.evaluation_id,
        task_name: preview.data.task_name, expected_task_name: preview.data.task_name, require_dres: true });
      if (entry.status !== "dres_ok") throw new Error(entry.dres?.error ?? "DRES did not acknowledge the submission. Check history before retrying.");
      setAccepted(rows => rows.map(a => keyOf(a.frame) === keyOf(preview.frame)
        ? { ...a, submitted: true, verdict: entry.verdict ?? "Acknowledged" } : a));
      setPreview(null);
    } catch (e) { setError(message(e)); }
    finally { sending.current = false; setBusy(false); }
  }
  function inspect(frame: FrameResult) { setReview(frame); setNear(null); setError(""); props.onInspect(frame); }
  const label = (frame: FrameResult) => `${frame.video_id} · ${frame.pts_time?.toFixed(2) ?? "?"} s`;
  return <section className="panel avs-panel" aria-label="AVS collection">
    <h3>AVS · collect relevant moments</h3>
    <p>Inspect candidates, then accept verified moments. The basket stays with this task while you refine the query.</p>
    <div className="avs-actions">
      <button className="btn sm" disabled={busy || !props.groups.length} onClick={prepareQueue}>Prepare review queue</button>
      <button className="btn sm" disabled={!props.selected} onClick={() => props.selected && accept(props.selected)}>Accept selected frame</button>
    </div>
    {error && !preview && <p role="alert">{error}</p>}
    {near && <button className="btn sm" onClick={() => accept(near, true)}>Accept nearby moment anyway</button>}
    {review && <div className="avs-review">
      <strong>{label(review)}</strong>
      <video key={keyOf(review)} controls preload="metadata" poster={review.keyframe_url}
        src={`${review.video_url}#t=${review.pts_time ?? 0}`} aria-label="AVS verification playback" />
      <button className="btn sm" onClick={() => accept(review)}>Accept verified moment</button>
      <button className="btn sm ghost" onClick={() => setReview(null)}>Close playback</button>
    </div>}
    <div className="avs-columns">
      <div><h4>Review queue · {queue.length}</h4><div className="avs-list">
        {queue.map(frame => <div className="avs-row" key={keyOf(frame)}>
          <img src={frame.keyframe_url} alt={label(frame)} />
          <span>{label(frame)}{avsConflict(frame, accepted) && <small>Already accepted or nearby</small>}</span>
          <button className="btn sm" onClick={() => inspect(frame)}>Inspect</button>
        </div>)}
        {!queue.length && <p>Prepare candidates from the current search. Only retrieved moments are proposed.</p>}
      </div></div>
      <div><h4>Accepted basket · {accepted.length}</h4><div className="avs-list">
        {accepted.map(a => <div className="avs-row" key={keyOf(a.frame)}>
          <span>{label(a.frame)}<small>{a.submitted ? `DRES: ${a.verdict}` : "Verified by operator · not submitted"}</small></span>
          <button className="btn sm" onClick={() => inspect(a.frame)}>Inspect</button>
          <button className="btn sm" disabled={busy || !props.dresEnabled || !props.evaluationId || !props.taskName || a.submitted}
            onClick={() => prepareSubmit(a.frame)}>Preview DRES</button>
          <button className="btn sm ghost" disabled={busy} onClick={() => setAccepted(rows => rows.filter(r => keyOf(r.frame) !== keyOf(a.frame)))}>Remove</button>
        </div>)}
      </div></div>
    </div>
    {!props.dresEnabled && <p>DRES is not configured. Accepted moments are saved in this browser.</p>}
    {preview && <div className="modal-backdrop"><div className="modal" role="dialog" aria-label="AVS submission preview">
      <div className="modal-head"><h3>Review AVS submission · {preview.data.task_name}</h3></div>
      <div className="modal-body">
        <p>{label(preview.frame)}</p>{error && <p role="alert">{error}</p>}<pre>{JSON.stringify(preview.data.body, null, 2)}</pre>
        {preview.data.warnings.map(w => <p key={w}>{w}</p>)}
        {preview.data.duplicate_key && <p role="alert">Duplicate: already submitted for this task.</p>}
      </div>
      <div className="modal-foot">
        <button className="btn" disabled={busy} onClick={() => setPreview(null)}>Cancel</button>
        <button className="btn primary" disabled={busy || !props.dresEnabled || !preview.data.evaluation_id || !preview.data.task_name
          || !!preview.data.duplicate_key || preview.data.answer_mode_mismatch} onClick={submit}>Confirm AVS submission</button>
      </div>
    </div></div>}
  </section>;
}
