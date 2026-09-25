import { useState, type ReactNode } from "react";
import type { TrakeEventEvidence, TrakeHeatPeak, TrakeVideoResult } from "../api/types";
import { eventColor } from "../lib/constants";
import { formatTime } from "../lib/media";
import { hasPeakDrag, readPeakDrag, setPeakDrag, type TrakePeakDrag } from "../lib/trakeDrag";
import { SearchThinking } from "./SearchThinking";
import { TrakeHeatmap } from "./TrakeHeatmap";

/** Frames the operator has put into a video's chain by hand, replacing the DP's
 *  pick for that event. Keyed `video_id -> event_index`. */
export type TrakeChainOverrides = Record<string, Record<number, TrakeHeatPeak>>;

interface Props {
  videos: TrakeVideoResult[];
  eventCount: number;
  selectedVideoId: string | null;
  /** Which event slot the panel is filling; its representative is highlighted. */
  activeEventIndex: number | null;
  overrides: TrakeChainOverrides;
  loading: boolean;
  onSelectVideo: (videoId: string) => void;
  /** Click a representative or a heat peak: select the video, seek to that
   *  moment, and arm that event's slot. */
  onPickMoment: (videoId: string, peak: TrakeHeatPeak) => void;
  /** Drop a frame onto event Ei of this video: it replaces the chain's pick. */
  onOverrideEvent: (videoId: string, eventIndex: number, peak: TrakeHeatPeak) => void;
  onClearOverride: (videoId: string, eventIndex: number) => void;
  onQuickSubmit: (videoId: string) => void;
  videoSlot?: ReactNode;
  videoSlotVideoId?: string | null;
}

/** TRAKE result list — VIDEOS, not keyframes.
 *
 *  A TRAKE answer is one video carrying an ordered chain, so squeezing the
 *  result into the generic video group (best chain in, every other candidate
 *  thrown away before the UI ever saw it) hid exactly what the operator has to
 *  judge: whether each event really fires in this video, and whether the moments
 *  line up in the right order. Each card therefore shows the chain across the
 *  top and every candidate frame, in place on the video's timeline, below it. */
export function TrakeVideoResults(props: Props) {
  const { videos, eventCount, selectedVideoId, loading } = props;
  if (loading) return <SearchThinking />;
  if (!videos.length) return <div className="empty">GPT-6 Astra is ready. Enter a query to begin.</div>;

  return (
    <div className="trake-videos" data-testid="trake-videos">
      {videos.map((video, rank) => {
        const selected = video.video_id === selectedVideoId;
        const confident = video.confident_coverage >= eventCount;
        const full = video.coverage >= eventCount;
        const overrides = props.overrides[video.video_id] ?? {};
        // What each event is ACTUALLY showing: the operator's frame if they put
        // one there, otherwise the DP's.
        const chosen = new Map<number, TrakeHeatPeak>();
        for (const event of video.events) {
          const pick = overrides[event.event_index] ?? event.representative;
          if (pick) chosen.set(event.event_index, pick);
        }
        const outOfOrder = orderViolations(video.events, chosen);
        return (
          <div
            key={video.video_id}
            className={`trake-video ${selected ? "selected" : ""}`}
            data-testid="trake-video-card"
            onClick={() => props.onSelectVideo(video.video_id)}
          >
            <div className="trake-video-head">
              <span className="rank mono">#{rank + 1}</span>
              <span className="vid">{video.video_id}</span>
              <span className="score mono" title="TRAKE video score: coverage first, then chain quality">
                ▮{video.trake_video_score.toFixed(3)}
              </span>
              <span
                className={`badge ${confident ? "speech" : "warn"}`}
                title={
                  confident
                    ? "Every event covered by real retrieval"
                    : full
                      ? `All events covered, but ${video.filled_events} filled by in-video search — verify before submitting`
                      : "Some events have no orderable candidate — pause the video and pick them by hand"
                }
                data-testid="trake-coverage-badge"
              >
                {video.coverage}/{eventCount} events
              </span>
              {video.filled_events > 0 && (
                <span className="badge warn" title="Events filled by the pass-2 in-video search (lower confidence)">
                  +{video.filled_events} in-video
                </span>
              )}
              <span className="score mono" title="Chain quality: 0.70·mean + 0.30·min of the per-event strengths">
                q{video.chain_quality.toFixed(2)}
              </span>
              {video.min_quality < 0.35 && video.confident_coverage > 0 && (
                <span className="badge warn" title={`Weakest event in the chain is only ${Math.round(video.min_quality * 100)}% as strong as that event's best hit anywhere`}>
                  ⚠ weak {Math.round(video.min_quality * 100)}%
                </span>
              )}
              {video.min_event_gap != null && video.min_event_gap < 1 && (
                <span
                  className="badge warn"
                  data-testid="trake-compressed-warning"
                  title="Two events land within a second of each other — often one moment matched twice rather than two events"
                >
                  ⚠ {video.min_event_gap.toFixed(1)}s between events
                </span>
              )}
              {outOfOrder.length > 0 && (
                <span
                  className="badge bad"
                  data-testid="trake-order-warning"
                  title="TRAKE events must be in chronological order; out-of-order submissions are rejected"
                >
                  ⚠ out of order: E{outOfOrder.join(", E")}
                </span>
              )}
              {/* No prioritise/deprioritise here on purpose: `/api/search/trake`
                  takes no feedback, so the buttons re-ran the search and changed
                  nothing while telling the operator they had. A control with no
                  effect is worse than no control under a clock. */}
              {full && (
                <button
                  className="btn sm primary"
                  style={{ marginLeft: "auto", padding: "3px 10px" }}
                  title="Fill all event slots from this chain and open the submit guard"
                  data-testid="trake-quick-submit"
                  onClick={(e) => { e.stopPropagation(); props.onQuickSubmit(video.video_id); }}
                >
                  Submit sequence ↵
                </button>
              )}
            </div>

            <div className="trake-event-strip">
              {video.events.map((event) => (
                <EventCard
                  key={event.event_index}
                  videoId={video.video_id}
                  event={event}
                  shown={chosen.get(event.event_index) ?? null}
                  overridden={overrides[event.event_index] != null}
                  active={props.activeEventIndex === event.event_index}
                  outOfOrder={outOfOrder.includes(event.event_index)}
                  onPick={(peak) => props.onPickMoment(video.video_id, peak)}
                  onDropPeak={(peak) => props.onOverrideEvent(video.video_id, event.event_index, peak)}
                  onClearOverride={() => props.onClearOverride(video.video_id, event.event_index)}
                />
              ))}
            </div>

            <TrakeHeatmap
              videoId={video.video_id}
              events={video.events}
              duration={video.duration_s || videoSpan(video)}
              activeEventIndex={props.activeEventIndex}
              chosenByEvent={
                new Map([...chosen].map(([index, peak]) => [index, peak.submit_keyframe_id]))
              }
              onPick={(peak) => props.onPickMoment(video.video_id, peak)}
            />

            {props.videoSlot && props.videoSlotVideoId === video.video_id && (
              <div className="inline-video-panel">{props.videoSlot}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function EventCard(props: {
  videoId: string;
  event: TrakeEventEvidence;
  shown: TrakeHeatPeak | null;
  overridden: boolean;
  active: boolean;
  outOfOrder: boolean;
  onPick: (peak: TrakeHeatPeak) => void;
  onDropPeak: (peak: TrakeHeatPeak) => void;
  onClearOverride: () => void;
}) {
  const { event, shown, active } = props;
  const [over, setOver] = useState(false);
  const color = eventColor(event.event_index - 1);
  // An override IS in the chain by the operator's decision, whatever the DP said.
  const loose = !props.overridden && !event.in_chain;

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setOver(false);
    const payload = readPeakDrag(e);
    // A frame from another video cannot join this video's chain — every TRAKE
    // event has to come from the one video that is being submitted.
    if (payload && payload.videoId === props.videoId) props.onDropPeak(payload.peak);
  }

  return (
    <div
      className={`trake-event ${active ? "active" : ""} ${loose ? "loose" : ""} ${props.overridden ? "overridden" : ""} ${over ? "drop-over" : ""} ${props.outOfOrder ? "out-of-order" : ""}`}
      data-testid={`trake-event-${event.event_index}`}
      style={active ? { boxShadow: `0 0 0 2px ${color}` } : undefined}
      onDragOver={(e) => {
        if (!hasPeakDrag(e)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      draggable={!!shown}
      onDragStart={(e) => {
        if (shown) setPeakDrag(e, { videoId: props.videoId, peak: shown } satisfies TrakePeakDrag);
      }}
      onClick={(e) => { e.stopPropagation(); if (shown) props.onPick(shown); }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && shown) {
          e.preventDefault();
          props.onPick(shown);
        }
      }}
      title="Click to seek · drag a heatmap frame here to replace this sequence frame"
    >
      <span className="trake-event-head">
        <span className="event-dot" style={{ background: color }} /> E{event.event_index}
        {shown?.via_fill && <span className="trake-event-tag">fill</span>}
        {shown?.channels?.includes("tara") && (
          <span className="trake-event-tag tara" data-testid={`trake-event-tara-${event.event_index}`} title="Found by a TARA clip">
            TARA
          </span>
        )}
        {props.overridden && (
          <button
            className="trake-event-undo"
            data-testid={`trake-event-undo-${event.event_index}`}
            title="Restore the automatically selected sequence frame"
            onClick={(e) => { e.stopPropagation(); props.onClearOverride(); }}
          >↺</button>
        )}
      </span>
      {shown ? (
        <>
          <img src={shown.keyframe_url} alt={shown.submit_keyframe_id} loading="lazy" />
          <span className="trake-event-meta mono">
            {formatTime(shown.pts_time)} · {Math.round(shown.strength * 100)}%
          </span>
          {props.overridden && <span className="trake-event-manual">manually selected</span>}
          {loose && (
            <span className="trake-event-warn" title="No orderable position for this event — the strongest candidate is shown so you can judge the video anyway">
              ⚠ outside sequence
            </span>
          )}
        </>
      ) : (
        <span className="trake-event-empty">drag a frame here</span>
      )}
    </div>
  );
}

/** 1-based event indices whose frame is not later than the previous event's.
 *
 *  The organiser's parser rejects a row that is not chronological, and one
 *  rejected row blocks the whole submission — so an override that breaks the
 *  order has to say so on the card, not at submit time. */
function orderViolations(
  events: TrakeEventEvidence[],
  chosen: Map<number, TrakeHeatPeak>,
): number[] {
  const violations: number[] = [];
  let last: number | null = null;
  for (const event of events) {
    const peak = chosen.get(event.event_index);
    if (!peak) continue;
    if (last !== null && peak.pts_time <= last) violations.push(event.event_index);
    last = peak.pts_time;
  }
  return violations;
}

/** Fallback width for the heat rows when the backend could not resolve the
 *  video's duration. The last moment any event fires plus a margin keeps the
 *  shape of the sequence readable, but it silently rescales "all four events
 *  happen in the first third" into "they span the whole video" — which is why
 *  `duration_s` is preferred whenever it is known. */
function videoSpan(video: TrakeVideoResult): number {
  let last = 0;
  for (const event of video.events) {
    for (const peak of event.peaks) if (peak.pts_time > last) last = peak.pts_time;
  }
  return last > 0 ? last * 1.04 : 1;
}
