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
  qaHotspots?: { submit_keyframe_id: string; pts_time: number; relevance: number }[];
  onSeek: (t: number) => void;
}

/** Scrub bar with the markers that matter — playhead, selection, TRAKE events,
 *  QA hotspots — and NO keyframe filmstrip.
 *
 *  The filmstrip sampled the keyframes down to ~60 thumbnails, which sounds
 *  cheap and is not: ~60 x 150 KB is roughly 9 MB fetched every time a video is
 *  opened, all of it inside the viewport so `loading="lazy"` defers nothing.
 *  Over a tunnel that was the visible stutter. Everything the strip was actually
 *  used for survives: click the bar to seek, and the markers still say where the
 *  events and hotspots are. */
export function Timeline({ data, playhead, selectedPts, eventMarkers = [], qaHotspots = [], onSeek }: Props) {
  const duration = data.duration || data.keyframes.at(-1)?.pts_time || 1;
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / duration) * 100))}%`;

  // Seeking used to mean clicking a thumbnail. With the strip gone the bar
  // itself has to take the click, or removing the images would quietly remove
  // the ability to jump anywhere in the clip.
  function seekFromClick(event: React.MouseEvent<HTMLDivElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    if (box.width <= 0) return;
    const ratio = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    onSeek(ratio * duration);
  }

  return (
    <div className="timeline" data-testid="timeline">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 11, color: "var(--fg-dim)" }} className="mono">
          {data.video_id} · {formatTime(playhead)} / {formatTime(duration)} · {data.fps}fps · {data.keyframes.length} kf
        </span>
      </div>

      <div className="tl-row">
        <span className="tl-label">seek</span>
        <div
          className="tl-area tl-seekable"
          data-testid="timeline-seek"
          onClick={seekFromClick}
          title="Bấm để nhảy tới vị trí đó"
        >
          <div className="tl-track" />
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
          {qaHotspots.map((hotspot, index) => (
            <div
              key={hotspot.submit_keyframe_id}
              className="tl-marker qa"
              data-testid="qa-timeline-marker"
              style={{ left: pct(hotspot.pts_time) }}
              title={`QA hotspot ${index + 1} · ${Math.round(hotspot.relevance * 100)}% · ${formatTime(hotspot.pts_time)}`}
              onClick={() => onSeek(hotspot.pts_time)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
