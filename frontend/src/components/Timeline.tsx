import type { Timeline as TimelineData } from "../api/types";
import { eventColor } from "../lib/constants";
import { formatTime } from "../lib/media";

interface EventMarker {
  eventIndex: number;
  pts_time: number;
}

interface Props {
  data: TimelineData;
  playhead: number;
  selectedPts: number | null;
  eventMarkers?: EventMarker[];
  onSeek: (t: number) => void;
}

// Lean keyframe filmstrip timeline (OCR/speech/audio/heatmap tracks removed for speed).
export function Timeline({ data, playhead, selectedPts, eventMarkers = [], onSeek }: Props) {
  const duration = data.duration || data.keyframes.at(-1)?.pts_time || 1;
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / duration) * 100))}%`;
  const stride = Math.ceil((data.keyframes.length || 1) / 60) || 1;

  return (
    <div className="timeline" data-testid="timeline">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 11, color: "var(--fg-dim)" }} className="mono">
          {data.video_id} · {formatTime(playhead)} / {formatTime(duration)} · {data.fps}fps · {data.keyframes.length} kf
        </span>
      </div>

      <div className="tl-row">
        <span className="tl-label">frames</span>
        <div className="tl-area" style={{ height: 40 }}>
          <div className="tl-track film" />
          {data.keyframes.filter((_, i) => i % stride === 0).map((kf) => (
            <img
              key={kf.submit_keyframe_id}
              className="tl-film-frame"
              src={kf.keyframe_url}
              style={{ left: pct(kf.pts_time ?? 0) }}
              title={`#${kf.keyframe_n} ${formatTime(kf.pts_time)}`}
              onClick={() => onSeek(kf.pts_time ?? 0)}
              loading="lazy"
            />
          ))}
          {selectedPts != null && (
            <div className="tl-playhead" style={{ left: pct(selectedPts), background: "var(--accent)" }} />
          )}
          <div className="tl-playhead" style={{ left: pct(playhead) }} />
          {eventMarkers.map((m, i) => (
            <div
              key={i}
              className="tl-marker ev"
              style={{ left: pct(m.pts_time), background: eventColor(m.eventIndex - 1) }}
              title={`E${m.eventIndex} @ ${formatTime(m.pts_time)}`}
              onClick={() => onSeek(m.pts_time)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
