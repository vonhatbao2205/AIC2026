import type { Evidence, FrameResult, QueryType } from "../api/types";
import { formatTime } from "../lib/media";
import { ChannelBadge } from "./Badges";

interface Props {
  frame: FrameResult | null;
  queryType: QueryType;
  onSubmit: () => void;
  onCopyId: (id: string) => void;
}

function EvidenceCard({ e }: { e: Evidence }) {
  const time = e.start != null ? `${formatTime(e.start)}${e.end != null ? `–${formatTime(e.end)}` : ""}` : null;
  const extra: string[] = [];
  if (e.top1_label) extra.push(String(e.top1_label));
  if (e.caption_quality) extra.push(`q:${e.caption_quality}`);
  if (e.confidence_bucket) extra.push(String(e.confidence_bucket));
  if (e.segment_role) extra.push(String(e.segment_role));
  if (e.clock) extra.push(`🕑${e.clock}`);
  const isGlap = e.type === "audio" && (e as { match?: string }).match === "glap-vector";
  return (
    <div className="evidence" data-testid="evidence-card">
      <div className="e-head">
        <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
          <ChannelBadge channel={e.type} />
          {isGlap && <span className="badge glap" title="GLAP audio-vector match">GLAP</span>}
        </span>
        <span className="e-time">{time} · {e.score.toFixed(3)}</span>
      </div>
      {e.text && <div className="e-text">{e.text}</div>}
      {extra.length > 0 && <div className="e-time">{extra.join(" · ")}</div>}
    </div>
  );
}

// Always describes the selected result keyframe. A captured raw frame lives in
// PausedFramePanel and is submitted from there, so this panel never changes
// identity underneath the operator.
export function DetailPanel({ frame, queryType, onSubmit, onCopyId }: Props) {
  if (!frame) {
    return (
      <div className="panel">
        <h3>Detail</h3>
        <div className="empty">Select a frame (←/→) to inspect evidence.</div>
      </div>
    );
  }
  return (
    <div className="panel" data-testid="detail-panel">
      <h3>Detail</h3>
      <div className="detail-preview">
        <img src={frame.keyframe_url} alt={frame.submit_keyframe_id} />
      </div>
      <div className="kv">
        <span className="k">submit id</span>
        <span className="v id-copy" title="click to copy" onClick={() => onCopyId(frame.submit_keyframe_id)} data-testid="submit-id">
          {frame.submit_keyframe_id}
        </span>
        <span className="k">video</span>
        <span className="v">{frame.video_id}</span>
        <span className="k">keyframe</span>
        <span className="v">#{frame.keyframe_n} · {formatTime(frame.pts_time)}</span>
        <span className="k">fused score</span>
        <span className="v">{frame.score.toFixed(5)}</span>
        <span className="k">channels</span>
        <span className="v">
          {Object.entries(frame.per_channel_score).map(([c, s]) => `${c}:${s.toFixed(2)}`).join("  ") || "—"}
        </span>
      </div>
      <div className="row" style={{ marginBottom: 8 }}>
        <button className="btn primary" data-testid="open-submit" onClick={onSubmit}>
          {queryType === "TRAKE" ? "Submit sequence" : "Submit result frame"}{" "}
          <span className="kbd">↵</span>
        </button>
      </div>
      <h3>Evidence · explain match</h3>
      {frame.evidence.length ? (
        frame.evidence.map((e, i) => <EvidenceCard key={i} e={e} />)
      ) : (
        <div className="empty" style={{ padding: 8 }}>No text evidence (vector-only match).</div>
      )}
    </div>
  );
}
