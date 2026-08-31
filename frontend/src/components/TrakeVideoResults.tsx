import type { ReactNode } from "react";
import type { TrakeEventEvidence, TrakeHeatPeak, TrakeVideoResult } from "../api/types";
import { eventColor } from "../lib/constants";
import { formatTime } from "../lib/media";
import { TrakeHeatmap } from "./TrakeHeatmap";

interface Props {
  videos: TrakeVideoResult[];
  eventCount: number;
  selectedVideoId: string | null;
  /** Which event slot the panel is filling; its representative is highlighted. */
  activeEventIndex: number | null;
  loading: boolean;
  onSelectVideo: (videoId: string) => void;
  /** Click a representative or a heat peak: select the video, seek to that
   *  moment, and arm that event's slot. */
  onPickMoment: (videoId: string, peak: TrakeHeatPeak) => void;
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
 *  line up in the right order. Each card therefore shows one representative per
 *  event plus the full heat row behind it. */
export function TrakeVideoResults(props: Props) {
  const { videos, eventCount, selectedVideoId, loading } = props;
  if (loading) return <div className="empty">Searching…</div>;
  if (!videos.length) return <div className="empty">No results. Enter a query and press Enter.</div>;

  return (
    <div className="trake-videos" data-testid="trake-videos">
      {videos.map((video, rank) => {
        const selected = video.video_id === selectedVideoId;
        const confident = video.confident_coverage >= eventCount;
        const full = video.coverage >= eventCount;
        const span = videoSpan(video);
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
                className={`badge ${confident ? "speech" : full ? "warn" : "warn"}`}
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
                  ⚠ yếu {Math.round(video.min_quality * 100)}%
                </span>
              )}
              {video.min_event_gap != null && video.min_event_gap < 1 && (
                <span
                  className="badge warn"
                  data-testid="trake-compressed-warning"
                  title="Two events land within a second of each other — often one moment matched twice rather than two events"
                >
                  ⚠ {video.min_event_gap.toFixed(1)}s giữa 2 event
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
                  title="Fill all event slots from this video's chain and open the submit guard"
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
                  event={event}
                  active={props.activeEventIndex === event.event_index}
                  onPick={(peak) => props.onPickMoment(video.video_id, peak)}
                />
              ))}
            </div>

            <TrakeHeatmap
              events={video.events}
              duration={video.duration_s || span}
              activeEventIndex={props.activeEventIndex}
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
  event: TrakeEventEvidence;
  active: boolean;
  onPick: (peak: TrakeHeatPeak) => void;
}) {
  const { event, active } = props;
  const rep = event.representative;
  const color = eventColor(event.event_index - 1);
  return (
    <button
      type="button"
      className={`trake-event ${active ? "active" : ""} ${event.in_chain ? "" : "loose"}`}
      data-testid={`trake-event-${event.event_index}`}
      style={active ? { boxShadow: `0 0 0 2px ${color}` } : undefined}
      disabled={!rep}
      onClick={(e) => { e.stopPropagation(); if (rep) props.onPick(rep); }}
    >
      <span className="trake-event-head">
        <span className="event-dot" style={{ background: color }} /> E{event.event_index}
        {rep?.via_fill && <span className="trake-event-tag">fill</span>}
      </span>
      {rep ? (
        <>
          <img src={rep.keyframe_url} alt={rep.submit_keyframe_id} loading="lazy" />
          <span className="trake-event-meta mono">
            {formatTime(rep.pts_time)} · {Math.round(rep.strength * 100)}%
          </span>
          {!event.in_chain && (
            <span className="trake-event-warn" title="No orderable position for this event — the strongest candidate is shown so you can judge the video anyway">
              ⚠ ngoài chuỗi
            </span>
          )}
        </>
      ) : (
        <span className="trake-event-empty">không có ứng viên</span>
      )}
    </button>
  );
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
