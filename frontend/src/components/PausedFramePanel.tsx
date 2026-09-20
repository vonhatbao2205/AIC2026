import type { QueryType } from "../api/types";
import { formatTime } from "../lib/media";

// Exact raw frame captured from the video element when playback is paused.
// It is intentionally independent from BTC keyframes returned by retrieval.
export interface PausedFrame {
  video_id: string;
  frame_idx: number;
  pts_time: number;
  fps: number;
  thumbnail?: string | null;
}

interface Props {
  frame: PausedFrame | null;
  queryType: QueryType;
  activeTrakeSlot: number;
  onSubmitPaused: () => void;
  onAddToSticky: () => void;
  onAssignToTrake: () => void;
  onClear: () => void;
}

export function PausedFramePanel(props: Props) {
  const { frame, queryType } = props;
  const isTrake = queryType === "TRAKE";

  return (
    <div className="panel" data-testid="paused-frame-panel">
      <div className="paused-frame-head">
        <h3>Paused frame · global</h3>
        {frame && !isTrake && <span className="badge paused-active">raw frame</span>}
      </div>

      {!frame ? (
        <div className="paused-frame-empty">
          Open the video with <span className="kbd">V</span>, then pause at the frame to
          submit. Available for T-KIS, QA, V-KIS and TRAKE.
        </div>
      ) : (
        <div
          className={`frame-chip global-frame-chip ${isTrake ? "draggable" : ""}`}
          data-testid="paused-frame-chip"
          draggable={isTrake}
          onDragStart={(event) => {
            if (isTrake) {
              event.dataTransfer.setData("text/x-paused-frame", String(frame.frame_idx));
            }
          }}
        >
          <div className="row">
            {frame.thumbnail ? (
              <img
                src={frame.thumbnail}
                className="paused-frame-thumb"
                alt={`Paused frame ${frame.frame_idx}`}
              />
            ) : (
              <div className="paused-frame-placeholder">no preview</div>
            )}
            <div className="paused-frame-meta">
              <div className="mono">
                {frame.video_id} · frame <b>{frame.frame_idx}</b>
              </div>
              <div className="mono paused-frame-time">
                {formatTime(frame.pts_time)} · {frame.pts_time.toFixed(3)}s ·{" "}
                {frame.fps.toFixed(3)} fps
              </div>
              <div className="paused-frame-formula">
                exact raw frame = round(mediaTime × fps)
              </div>
            </div>
          </div>

          <div className="row paused-frame-actions">
            {isTrake ? (
              <button
                className="btn sm primary"
                data-testid="assign-paused-frame"
                onClick={props.onAssignToTrake}
              >
                Assign to E{props.activeTrakeSlot + 1} <span className="kbd">↵</span>
              </button>
            ) : (
              <button
                className="btn sm primary"
                data-testid="submit-paused-frame"
                onClick={props.onSubmitPaused}
              >
                Submit paused frame <span className="kbd">⇧↵</span>
              </button>
            )}
            {!isTrake && (
              <button className="btn sm" data-testid="sticky-add-paused" onClick={props.onAddToSticky}>
                + Sticky
              </button>
            )}
            <button className="btn sm ghost" onClick={props.onClear}>
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
