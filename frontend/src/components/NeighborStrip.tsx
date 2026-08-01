import type { TimelineKeyframe } from "../api/types";
import { formatTime } from "../lib/media";

/** Keyframes shown on each side of the centre. The full video is browsable —
 *  only this window is mounted, so a 400-keyframe video never loads 400 images. */
export const NEIGHBOR_RADIUS = 6;

interface Props {
  keyframes: TimelineKeyframe[];
  centerIndex: number;
  anchorId: string | null;
  videoLinked: boolean;
  onPick: (index: number) => void;
}

export function NeighborStrip({ keyframes, centerIndex, anchorId, videoLinked, onPick }: Props) {
  if (keyframes.length === 0) {
    return (
      <div className="neighbor-strip" data-testid="neighbor-strip">
        <div className="empty" style={{ padding: 8 }}>Loading keyframes…</div>
      </div>
    );
  }

  const center = Math.min(Math.max(centerIndex, 0), keyframes.length - 1);
  const from = Math.max(0, center - NEIGHBOR_RADIUS);
  const to = Math.min(keyframes.length, center + NEIGHBOR_RADIUS + 1);
  const window = keyframes.slice(from, to);

  return (
    <div className="neighbor-strip" data-testid="neighbor-strip">
      <div className="neighbor-head">
        <span className="mono" data-testid="neighbor-position">
          keyframe {center + 1} / {keyframes.length}
        </span>
        <span className="neighbor-hint">
          <span className="kbd">←</span> <span className="kbd">→</span> browse
          {videoLinked ? " · video follows" : ""}
        </span>
      </div>
      <div className="neighbor-track">
        {from > 0 && <div className="neighbor-more" title={`${from} earlier`}>+{from}</div>}
        {window.map((kf, offset) => {
          const index = from + offset;
          const isCenter = index === center;
          const isAnchor = kf.submit_keyframe_id === anchorId;
          return (
            <button
              key={kf.submit_keyframe_id}
              className={`neighbor-cell${isCenter ? " center" : ""}${isAnchor ? " anchor" : ""}`}
              data-testid={isCenter ? "neighbor-center" : "neighbor-cell"}
              title={`${kf.submit_keyframe_id}${isAnchor ? " · selected result frame" : ""}`}
              onClick={() => onPick(index)}
            >
              <img src={kf.keyframe_url} alt={kf.submit_keyframe_id} loading="lazy" />
              <span className="neighbor-meta mono">
                #{kf.keyframe_n} · {formatTime(kf.pts_time)}
              </span>
            </button>
          );
        })}
        {to < keyframes.length && (
          <div className="neighbor-more" title={`${keyframes.length - to} later`}>
            +{keyframes.length - to}
          </div>
        )}
      </div>
    </div>
  );
}
