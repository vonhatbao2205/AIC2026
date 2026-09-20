import { useEffect, useRef, useState } from "react";
import type { TrakeEventEvidence, TrakeHeatPeak } from "../api/types";
import { eventColor } from "../lib/constants";
import { formatTime } from "../lib/media";
import { setPeakDrag } from "../lib/trakeDrag";
import {
  FALLBACK_TRACK_W,
  HEAT_LANE_H,
  HEAT_THUMB_H,
  HEAT_THUMB_W,
  HEAT_TRACK_PAD,
  layoutHeatPeaks,
} from "../lib/trakeHeatLayout";

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
 *  Crowded candidates stack into a second lane rather than burying each other
 *  (see `layoutHeatPeaks`); a tick at the true timestamp keeps the row honest
 *  about where each one really is. Every frame is draggable: onto an event slot
 *  to submit it, or onto the event strip above to put it in the chain. */
export function TrakeHeatmap(props: Props) {
  const { videoId, events, duration, activeEventIndex, chosenByEvent } = props;
  const span = duration > 0 ? duration : 1;

  // Placement is in pixels, so it needs the real track width. All rows are
  // siblings of the same width, so one measurement serves the whole map.
  const trackRef = useRef<HTMLDivElement>(null);
  const [trackWidth, setTrackWidth] = useState(0);
  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const measure = () => setTrackWidth(el.clientWidth);
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [events.length]);
  const width = trackWidth || FALLBACK_TRACK_W;

  return (
    <div className="trake-heatmap" data-testid="trake-heatmap">
      {events.map((event, rowIndex) => {
        const i = event.event_index - 1;
        const color = eventColor(i);
        const active = activeEventIndex === event.event_index;
        const chosen = chosenByEvent.get(event.event_index);
        const { placed, trackHeight } = layoutHeatPeaks(event.peaks, span, width);
        return (
          <div
            key={event.event_index}
            className={`trake-heat-row ${active ? "active" : ""}`}
            data-testid={`trake-heat-row-${event.event_index}`}
          >
            <span className="trake-heat-label" style={{ color }}>
              E{event.event_index}
            </span>
            <div
              className="trake-heat-track"
              ref={rowIndex === 0 ? trackRef : undefined}
              style={{ height: trackHeight }}
            >
              {event.peaks.length === 0 && (
                <span className="trake-heat-empty">no candidates</span>
              )}
              {/* Ticks first, behind the frames: this is the exact timeline. */}
              {placed.map((item) => (
                <span
                  key={`tick-${item.peak.submit_keyframe_id}`}
                  className={`trake-heat-tick ${item.shifted ? "shifted" : ""}`}
                  style={{ left: item.trueX, background: color }}
                  aria-hidden
                />
              ))}
              {placed.map((item) => {
                const peak = item.peak;
                const isChosen = peak.submit_keyframe_id === chosen;
                return (
                  <button
                    key={peak.submit_keyframe_id}
                    type="button"
                    className={`trake-heat-peak ${isChosen ? "picked" : ""} ${peak.via_fill ? "fill" : ""}`}
                    data-testid="trake-heat-peak"
                    aria-label={`E${event.event_index} at ${formatTime(peak.pts_time)}`}
                    title={`E${event.event_index} · ${formatTime(peak.pts_time)} · ${Math.round(peak.strength * 100)}%${peak.via_fill ? " · in-video fill" : ""}${isChosen ? " · selected for sequence" : ""}\nClick to select · drag to an event slot to use · V to open video`}
                    style={{
                      left: item.x,
                      top: item.lane * HEAT_LANE_H + HEAT_TRACK_PAD / 2,
                      width: HEAT_THUMB_W,
                      height: HEAT_THUMB_H,
                      borderColor: color,
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
                    <span className="trake-heat-peak-meta">{formatTime(peak.pts_time)}</span>
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
