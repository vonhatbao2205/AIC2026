import type { Evidence, FrameResult, QueryType } from "../api/types";
import { boxStyle, layoutEvidenceOf, outlineFor } from "../lib/canvas";
import { formatTime } from "../lib/media";
import { CHANNEL_LABEL } from "../lib/constants";
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
  // A merged visual hit shows one badge, so the retrievers behind it have to be
  // readable somewhere: an operator checking a frame needs to know whether both
  // embedding spaces agreed on it or only one of them found it at all.
  if (Array.isArray(e.models)) extra.push((e.models as string[]).map((m) => CHANNEL_LABEL[m === "pe" ? "image_pe" : "image_qwen"]).join(" + "));
  if (typeof e.rerank_score === "number") extra.push(`rerank ${e.rerank_score.toFixed(3)}`);
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

/** Draws the detections that matched the canvas over the keyframe.
 *
 * A single fused score cannot tell an operator whether the frame came back for
 * the right reason. Showing which drawn object landed on which real detection —
 * in the same colour as the box they drew — is what turns a ranked list into
 * something verifiable before a submit. */
function CanvasOverlay({ frame }: { frame: FrameResult }) {
  const layout = layoutEvidenceOf(frame);
  if (!layout?.matches?.length) return null;
  return (
    <>
      {layout.matches.map((match) => {
        const outline = outlineFor((Number(match.object_id.slice(1)) || 1) - 1);
        return (
          <div
            key={match.object_id}
            className="overlay-box"
            style={{ ...boxStyle(match.bbox_norm), borderColor: outline }}
            data-testid={`overlay-${match.object_id}`}
          >
            <span className="overlay-tag" style={{ background: outline }}>
              {match.label}
              {match.dominant_color ? ` · ${match.dominant_color}` : ""} {match.score.toFixed(2)}
            </span>
          </div>
        );
      })}
    </>
  );
}

function CanvasMatchSummary({ frame }: { frame: FrameResult }) {
  const layout = layoutEvidenceOf(frame);
  if (!layout) return null;
  return (
    <div className="evidence" data-testid="canvas-match-summary">
      <div className="e-head">
        <ChannelBadge channel="object_layout" />
        <span className="e-time">
          {Math.round(layout.coverage * 100)}% object · {layout.score.toFixed(3)}
        </span>
      </div>
      {layout.matches.map((match) => (
        <div key={match.object_id} className="e-text">
          {match.object_id} · {match.label}
          {match.dominant_color ? ` (${match.dominant_color}${match.color_ok === false ? " ≠ vẽ" : ""})` : ""}
          {match.position ? ` @${match.position}` : ""} · conf {match.conf.toFixed(2)}
        </div>
      ))}
      {layout.missing.length > 0 && (
        <div className="e-time">Không tìm thấy: {layout.missing.join(", ")}</div>
      )}
      {layout.excluded_hits.length > 0 && (
        <div className="e-time">Có object bị loại trừ: {layout.excluded_hits.join(", ")}</div>
      )}
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
        <CanvasOverlay frame={frame} />
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
      <CanvasMatchSummary frame={frame} />
      {frame.evidence.length ? (
        frame.evidence
          .filter((e) => e.type !== "object_layout")
          .map((e, i) => <EvidenceCard key={i} e={e} />)
      ) : (
        <div className="empty" style={{ padding: 8 }}>No text evidence (vector-only match).</div>
      )}
    </div>
  );
}
