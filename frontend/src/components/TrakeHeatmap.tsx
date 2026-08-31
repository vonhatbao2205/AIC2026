import type { TrakeEventEvidence, TrakeHeatPeak } from "../api/types";
import { eventColor } from "../lib/constants";
import { formatTime } from "../lib/media";

interface Props {
  events: TrakeEventEvidence[];
  /** Seconds the rows span — the video's real length when the backend resolved
   *  it, so "everything happens in the first third" stays visible. */
  duration: number;
  activeEventIndex: number | null;
  onPick: (peak: TrakeHeatPeak) => void;
}

/** Where E1..En fire along one video, one row per event.
 *
 *  This is the whole point of ranking by video: the DP only knows that its chain
 *  is orderable, not whether it is plausible. A clean diagonal (E1 early, E2
 *  after it, E3 after that) reads as a real sequence at a glance, while E1
 *  sitting minutes away from a tight E2-E3-E4 cluster is worth checking before
 *  submitting. Neither is visible in a list of thumbnails.
 *
 *  Peaks are drawn from the sparse points the backend sends — no image is
 *  rendered server-side — and each one is clickable, so verifying an alternative
 *  moment costs a seek instead of another search. They are deliberately far
 *  wider than the mark they draw: this is a click target on a laptop screen
 *  under a competition clock, not a chart. */
export function TrakeHeatmap(props: Props) {
  const { events, duration, activeEventIndex } = props;
  const span = duration > 0 ? duration : 1;

  return (
    <div className="trake-heatmap" data-testid="trake-heatmap">
      {events.map((event) => {
        const i = event.event_index - 1;
        const active = activeEventIndex === event.event_index;
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
              {event.peaks.length === 0 && <span className="trake-heat-empty">không có ứng viên</span>}
              {event.peaks.map((peak) => (
                <button
                  key={peak.submit_keyframe_id}
                  type="button"
                  className={`trake-heat-peak ${peak.selected_by_dp ? "picked" : ""} ${peak.via_fill ? "fill" : ""}`}
                  data-testid="trake-heat-peak"
                  aria-label={`E${event.event_index} tại ${formatTime(peak.pts_time)}`}
                  title={`E${event.event_index} · ${formatTime(peak.pts_time)} · ${Math.round(peak.strength * 100)}%${peak.via_fill ? " · in-video fill" : ""}${peak.selected_by_dp ? " · chuỗi chọn frame này" : ""}`}
                  style={{
                    left: `${Math.min(99, Math.max(0, (peak.pts_time / span) * 100))}%`,
                    // Strength drives height and opacity so a weak hit stays
                    // visible (it is evidence too) without reading as a strong
                    // one. The click target is a fixed, generous overlay — it
                    // must not shrink with the score.
                    height: `${40 + 55 * peak.strength}%`,
                    opacity: 0.45 + 0.55 * peak.strength,
                    background: eventColor(i),
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    props.onPick(peak);
                  }}
                />
              ))}
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
