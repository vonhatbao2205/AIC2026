import type { TrakeEventEvidence, TrakeHeatPeak } from "../api/types";
import { eventColor } from "../lib/constants";
import { formatTime } from "../lib/media";
import { setPeakDrag } from "../lib/trakeDrag";

interface Props {
  videoId: string;
  events: TrakeEventEvidence[];
  /** Seconds the rows span — the video's real length when the backend resolved
   *  it, so "everything happens in the first third" stays visible. */
  duration: number;
  activeEventIndex: number | null;
  /** Which frame each event is currently showing, so the heat row can mark it. */
  chosenByEvent: Map<number, string>;
  onPick: (peak: TrakeHeatPeak) => void;
}

/** Where E1..En fire along one video — as the frames themselves, one row per event.
 *
 *  Two jobs at once. The POSITIONS answer the question the DP cannot: it only
 *  knows its chain is orderable, not whether it is plausible, so a clean diagonal
 *  (E1 early, E2 after it, E3 after that) reads as a real sequence at a glance
 *  while E1 sitting minutes from a tight E2-E3-E4 cluster is worth checking. The
 *  IMAGES answer "is this actually the moment?" without spending a video load, a
 *  seek and a pause per candidate — which was the whole cost of looking at an
 *  alternative before.
 *
 *  Every frame here is draggable: onto an event slot to submit it, or onto the
 *  event strip above to put it in the chain. */
export function TrakeHeatmap(props: Props) {
  const { videoId, events, duration, activeEventIndex, chosenByEvent } = props;
  const span = duration > 0 ? duration : 1;

  return (
    <div className="trake-heatmap" data-testid="trake-heatmap">
      {events.map((event) => {
        const i = event.event_index - 1;
        const active = activeEventIndex === event.event_index;
        const chosen = chosenByEvent.get(event.event_index);
        return (
          <div
            key={event.event_index}
            className={`trake-heat-row ${active ? "active" : ""}`}
            data-testid={`trake-heat-row-${event.event_index}`}
          >
            <span className="trake-heat-label" style={{ color: eventColor(i) }}>
              E{event.event_index}
            </span>
            <div className="trake-heat-track">
              {event.peaks.length === 0 && (
                <span className="trake-heat-empty">không có ứng viên</span>
              )}
              {event.peaks.map((peak) => {
                const isChosen = peak.submit_keyframe_id === chosen;
                return (
                  <button
                    key={peak.submit_keyframe_id}
                    type="button"
                    className={`trake-heat-peak ${isChosen ? "picked" : ""} ${peak.via_fill ? "fill" : ""}`}
                    data-testid="trake-heat-peak"
                    aria-label={`E${event.event_index} tại ${formatTime(peak.pts_time)}`}
                    title={`E${event.event_index} · ${formatTime(peak.pts_time)} · ${Math.round(peak.strength * 100)}%${peak.via_fill ? " · in-video fill" : ""}${isChosen ? " · đang dùng cho chuỗi" : ""}\nBấm để tua · kéo vào ô event để dùng`}
                    style={{
                      left: `${Math.min(99, Math.max(1, (peak.pts_time / span) * 100))}%`,
                      borderColor: eventColor(i),
                      // Strength fades the border, never the image: a weak hit is
                      // still a frame the operator has to be able to read.
                      opacity: 0.55 + 0.45 * peak.strength,
                    }}
                    draggable
                    onDragStart={(e) => setPeakDrag(e, { videoId, peak })}
                    onClick={(e) => {
                      e.stopPropagation();
                      props.onPick(peak);
                    }}
                  >
                    <img src={peak.keyframe_url} alt="" loading="lazy" decoding="async" />
                    <span className="trake-heat-peak-meta">
                      {formatTime(peak.pts_time)}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
      <div className="trake-heat-axis">
        <span>0:00</span>
        <span>{formatTime(span / 2)}</span>
        <span>{formatTime(span)}</span>
      </div>
    </div>
  );
}
