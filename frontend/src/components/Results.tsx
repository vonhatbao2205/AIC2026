import type { ReactNode } from "react";
import type { FrameResult, VideoGroup } from "../api/types";
import { formatTime } from "../lib/media";
import { ChannelBadges } from "./Badges";

type ViewMode = "grouped" | "flat";

interface Props {
  groups: VideoGroup[];
  viewMode: ViewMode;
  selectedVideo: number;
  selectedFrame: number;
  expanded: Set<string>;
  onSelectVideo: (i: number) => void;
  onSelectFrame: (videoIdx: number, frameIdx: number) => void;
  onToggleExpand: (videoId: string) => void;
  onFeedback: (frame: FrameResult, kind: "more" | "exclude") => void;
  loading: boolean;
  // Inline video player, rendered directly under the group it belongs to.
  videoSlot?: ReactNode;
  videoSlotVideoId?: string | null;
  // In TRAKE mode, the number of events — to show per-video coverage (cov/total).
  trakeEventCount?: number;
  // TRAKE: fill the event slots from this video's frames and open the submit guard.
  onTrakeQuickSubmit?: (g: VideoGroup) => void;
}

// A frame whose audio evidence came from the GLAP audio-vector search.
function isGlap(f: FrameResult): boolean {
  return (f.evidence || []).some((e) => e.type === "audio" && (e as { match?: string }).match === "glap-vector");
}

export function Results(props: Props) {
  const { groups, viewMode, selectedVideo, selectedFrame, expanded, loading } = props;
  if (loading) return <div className="empty">Searching…</div>;
  if (!groups.length) return <div className="empty">No results. Enter a query and press Enter.</div>;

  if (viewMode === "flat") {
    // Flatten every frame and sort by fused score = the raw top-K ranking.
    const flat = groups
      .flatMap((g, gi) => g.frames.map((f, fi) => ({ f, gi, fi })))
      .sort((a, b) => b.f.score - a.f.score);
    return (
      <div data-testid="results-flat">
        {props.videoSlot && <div className="inline-video-panel">{props.videoSlot}</div>}
        <div className="flat-grid">
        {flat.map(({ f, gi, fi }, rank) => {
          const sel = gi === selectedVideo && fi === selectedFrame;
          return (
            <figure
              key={f.submit_keyframe_id}
              className={`kf-card ${sel ? "sel" : ""}`}
              data-testid="flat-card"
              onClick={() => props.onSelectFrame(gi, fi)}
            >
              <div className="kf-rank">#{rank + 1}</div>
              <div className="chips flat-chips">
                {f.channels.map((c) => (
                  <span key={c} className={`badge ${c}`}>{c[0]}</span>
                ))}
                {isGlap(f) && <span className="badge glap" title="Matched by GLAP audio-vector search">GLAP</span>}
              </div>
              <img src={f.keyframe_url} alt={f.submit_keyframe_id} loading="lazy" />
              <figcaption>
                <span className="kf-id mono">{f.submit_keyframe_id}</span>
                <span className="kf-score mono">{f.score.toFixed(4)}</span>
              </figcaption>
            </figure>
          );
        })}
        </div>
      </div>
    );
  }

  return (
    <div className="results" data-testid="results">
      {groups.map((g, gi) => {
        const isOpen = expanded.has(g.video_id);
        const isSel = gi === selectedVideo;
        const n = props.trakeEventCount;
        const filled = g.trake_filled ?? 0;
        const fullCover = n != null && g.frame_count >= n;
        const confident = fullCover && filled === 0; // every event from real retrieval
        return (
          <div key={g.video_id} className={`vgroup ${isSel ? "selected" : ""}`} data-testid="video-group">
            <div className="vgroup-head" onClick={() => { props.onSelectVideo(gi); props.onToggleExpand(g.video_id); }}>
              <span className="caret">{isOpen ? "▾" : "▸"}</span>
              <span className="vid">{g.video_id}</span>
              <span className="score mono">▮{g.video_score.toFixed(3)}</span>
              {n ? (
                <>
                  <span
                    className={`badge ${confident ? "speech" : "warn"}`}
                    title={
                      confident
                        ? "All events covered by real retrieval"
                        : fullCover
                          ? `All events covered, but ${filled} filled by in-video search — verify before submitting`
                          : "Some events missing — pause the video and pick the missing event manually"
                    }
                  >
                    {g.frame_count}/{n} events
                  </span>
                  {filled > 0 && (
                    <span className="badge warn" title={`${filled} event(s) filled by pass-2 in-video search (lower confidence)`}>
                      +{filled} in-video
                    </span>
                  )}
                  {g.trake_mean != null && (
                    <span className="score mono" title="Mean relevance of real (non-filled) frames">μ{g.trake_mean.toFixed(3)}</span>
                  )}
                </>
              ) : (
                <span className="score">×{g.frame_count}</span>
              )}
              <ChannelBadges channels={g.channels} />
              {!props.trakeEventCount && g.ambiguous && <span className="ambiguous-tag" title="Top frames split into distant time clusters">⚠ ambiguous</span>}
              {props.trakeEventCount && g.frame_count >= props.trakeEventCount && props.onTrakeQuickSubmit && (
                <button
                  className="btn sm primary"
                  style={{ marginLeft: "auto", padding: "3px 10px" }}
                  title="Fill all event slots from this video and open the submit guard"
                  onClick={(e) => { e.stopPropagation(); props.onTrakeQuickSubmit!(g); }}
                  data-testid="trake-quick-submit"
                >
                  Submit sequence ↵
                </button>
              )}
            </div>
            {isOpen && (
              <div className="frames-strip">
                {g.frames.map((f, fi) => {
                  const fsel = isSel && fi === selectedFrame;
                  return (
                    <div
                      key={f.submit_keyframe_id}
                      className={`thumb ${fsel ? "selected" : ""}`}
                      data-testid="frame-thumb"
                      onClick={() => props.onSelectFrame(gi, fi)}
                    >
                      <div className="chips">
                        {f.channels.map((c) => (
                          <span key={c} className={`badge ${c}`}>{c[0]}</span>
                        ))}
                        {isGlap(f) && <span className="badge glap" title="Matched by GLAP audio-vector search">G</span>}
                      </div>
                      <div className="fb">
                        <button title="More like this" onClick={(e) => { e.stopPropagation(); props.onFeedback(f, "more"); }}>＋</button>
                        <button title="Exclude this" onClick={(e) => { e.stopPropagation(); props.onFeedback(f, "exclude"); }}>✕</button>
                      </div>
                      <img src={f.keyframe_url} alt={f.submit_keyframe_id} loading="lazy" />
                      <div className="meta">
                        <div className="kf">#{f.keyframe_n} · {formatTime(f.pts_time)}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
            {props.videoSlot && props.videoSlotVideoId === g.video_id && (
              <div className="inline-video-panel">{props.videoSlot}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}
