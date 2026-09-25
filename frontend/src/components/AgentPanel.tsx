import { useEffect, useMemo, useState, type Ref } from "react";
import type { AgentCandidate, AgentName, AgentRunSnapshot, AgentState, QueryType } from "../api/types";
import {
  AGENT_LABELS,
  AGENT_NAMES,
  buildSequences,
  candidateTime,
  findConsensus,
  isAgentWorking,
  sortCandidates,
  type AgentSequence,
} from "../lib/agent";
import { eventColor } from "../lib/constants";
import { formatTime } from "../lib/media";
import type { PausedFrame } from "./PausedFramePanel";
import { VideoViewer, type VideoViewerHandle } from "./VideoViewer";

interface Props {
  run: AgentRunSnapshot | null;
  starting: boolean;
  error: string | null;
  /** The tab's task type: it decides how a frame is submitted, not the run's. */
  queryType: QueryType;
  /** The query currently in the box, to flag a run that belongs to an older one. */
  currentQuery: string;
  /** TRAKE: how many events the answer needs (the width of every sequence row). */
  eventCount: number;
  /** TRAKE: the event slots the paused frame can be assigned to, and the armed one. */
  trakeSlotCount: number;
  activeTrakeSlot: number;
  /** Videos present in the main results, so a candidate can jump there. */
  resultVideoIds: Set<string>;
  /** The console's captured raw frame; shown under the player when it is this video's. */
  pausedFrame: PausedFrame | null;
  onStop: () => void;
  onDismiss: () => void;
  /** Select the candidate's video in the main results and play it there. */
  onLocate: (candidate: AgentCandidate) => void;
  /** Open the submit guard on this frame (T-KIS / V-KIS / QA), or send it to AVS. */
  onSubmit: (candidate: AgentCandidate) => void;
  onSticky: (candidate: AgentCandidate) => void;
  /** QA: put the agent's answer into the answer box. */
  onUseAnswer: (answer: string) => void;
  /** A frame paused in this panel's own player: the console's paused frame. */
  onPaused: (frame: PausedFrame) => void;
  onSubmitPaused: () => void;
  onStickyPaused: () => void;
  onAssignPaused: (slotIndex: number) => void;
  /** TRAKE: put a sequence into the event slots, or do that and open the guard. */
  onLoadSequence: (sequence: AgentSequence) => void;
  onSubmitSequence: (sequence: AgentSequence) => void;
  onStickySequence: (sequence: AgentSequence) => void;
  /** The same video under the fallback origin, when one is configured. */
  videoFallback: (url: string) => string | null;
  /** The strip's player, so the console's scrub keys (a/d, arrows, Space) can drive it. */
  playerRef?: Ref<VideoViewerHandle>;
  /** Called when the strip's player opens or closes. */
  onPlayerChange?: (open: boolean) => void;
  /** A click in the player: it takes the scrub keys back from the main player. */
  onPlayerFocus?: () => void;
}

const STATUS_LABEL: Record<AgentState["status"], string> = {
  pending: "starting",
  queued: "queued",
  running: "searching",
  done: "done",
  failed: "failed",
  timeout: "timed out",
  cancelled: "stopped",
  unavailable: "not installed",
};

function agentError(state: AgentState): string | null {
  if (!state.error) return null;
  if (state.error_kind === "quota") return `Quota / usage limit reached — switch account, the next search uses it. (${state.error})`;
  if (state.error_kind === "auth") return `Not logged in — run the CLI's login on this machine. (${state.error})`;
  if (state.error_kind === "update") return `CLI too old for ${state.model} — update it on the backend machine. (${state.error})`;
  return state.error;
}

function scopeLabel(run: AgentRunSnapshot): string | null {
  const scope = run.scope;
  if (!scope?.active) return null;
  const shown = scope.categories.slice(0, 6).join(", ");
  const more = scope.categories.length > 6 ? ` … (${scope.categories.length})` : "";
  return `${scope.mode === "manual" ? "Filter" : "Auto filter"}: ${shown}${more}`;
}

function AgentBox({ state }: { state: AgentState }) {
  const [open, setOpen] = useState(false);
  const working = isAgentWorking(state);
  const last = state.steps[state.steps.length - 1];
  const error = agentError(state);
  return (
    <div className={`agent-row ${state.status}`} data-testid={`agent-row-${state.name}`}>
      <button className="agent-row-head" onClick={() => setOpen((v) => !v)} title="Show this agent's steps">
        <span className={`agent-dot ${working ? "live" : state.status}`} />
        <b>{AGENT_LABELS[state.name]}</b>
        <span className="agent-model">{state.model} · {state.effort}{state.fast ? " · fast" : ""}</span>
        <span className="agent-status">{STATUS_LABEL[state.status]}</span>
        <span className="agent-meta">
          {state.elapsed_s != null && `${Math.round(state.elapsed_s)} s`}
          {state.tool_calls > 0 && ` · ${state.tool_calls} tools`}
          {state.cost_usd != null && ` · $${state.cost_usd.toFixed(2)}`}
        </span>
      </button>
      {working && last && <div className="agent-last">{last.text}</div>}
      {!working && state.summary && <div className="agent-summary">{state.summary}</div>}
      {error && <div className="agent-error">{error}</div>}
      {open && state.steps.length > 0 && (
        <ol className="agent-steps">
          {state.steps.map((step, index) => (
            <li key={index} className={step.kind}>
              <span className="mono">{Math.round(step.at)}s</span> {step.text}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

interface CardProps {
  candidate: AgentCandidate;
  queryType: QueryType;
  agreed: boolean;
  inResults: boolean;
  previewing: boolean;
  onPlay: () => void;
  onSubmit: () => void;
  onSticky: () => void;
  onLocate: () => void;
  onUseAnswer: (answer: string) => void;
}

function CandidateCard(props: CardProps) {
  const { candidate, queryType } = props;
  const time = candidateTime(candidate);
  return (
    <article
      className={`agent-card ${props.agreed ? "agreed" : ""} ${candidate.outside_scope ? "outside" : ""} ${props.previewing ? "previewing" : ""}`}
      data-testid="agent-candidate"
    >
      <button className="agent-card-thumb" onClick={props.onPlay} title="Play here from this moment">
        <img src={candidate.keyframe_url} alt={candidate.submit_keyframe_id} loading="lazy" />
        <span className="agent-card-time mono">{formatTime(time)}</span>
      </button>
      <div className="agent-card-title">
        <span className={`agent-tag ${candidate.agent}`}>{AGENT_LABELS[candidate.agent]}</span>
        <b className="mono">{candidate.video_id}</b>
        {candidate.event != null && <span className="badge">E{candidate.event}</span>}
        <span className="agent-conf" title="Agent's own confidence">{Math.round(candidate.confidence * 100)}%</span>
      </div>
      {candidate.outside_scope && (
        <div className="agent-outside" data-testid="agent-outside">Outside the folder filter</div>
      )}
      {candidate.answer && (
        <div className="agent-answer">
          Answer: <b>{candidate.answer}</b>
          {queryType === "QA" && (
            <button className="btn sm ghost" onClick={() => props.onUseAnswer(candidate.answer as string)}>
              Use
            </button>
          )}
        </div>
      )}
      {candidate.reason && <div className="agent-reason" title={candidate.reason}>{candidate.reason}</div>}
      <div className="agent-actions">
        <button className="btn sm primary" onClick={props.onSubmit} data-testid="agent-submit">
          {queryType === "AVS" ? "Add to AVS" : "Submit"}
        </button>
        <button className="btn sm" onClick={props.onPlay} data-testid="agent-play">
          {props.previewing ? "Close" : "▶ Play"}
        </button>
        {queryType !== "AVS" && (
          <button className="btn sm" onClick={props.onSticky} data-testid="agent-sticky">+ Sticky</button>
        )}
        {props.inResults && (
          <button className="btn sm ghost" onClick={props.onLocate} data-testid="agent-locate">
            Show in results
          </button>
        )}
      </div>
      <div className="agent-kf mono">{candidate.submit_keyframe_id}</div>
    </article>
  );
}

interface SequenceProps {
  sequence: AgentSequence;
  eventCount: number;
  previewId: string | null;
  onPlay: (candidate: AgentCandidate) => void;
  onLoad: () => void;
  onSubmit: () => void;
  onSticky: () => void;
}

function SequenceRow(props: SequenceProps) {
  const { sequence } = props;
  const complete = sequence.found >= props.eventCount;
  const first = sequence.events.find((c): c is AgentCandidate => c !== null) ?? sequence.extras[0] ?? null;
  return (
    <article
      className={`agent-seq ${sequence.agreed ? "agreed" : ""} ${sequence.outside ? "outside" : ""}`}
      data-testid="agent-sequence"
    >
      <div className="agent-seq-head">
        <span className={`agent-tag ${sequence.agent}`}>{AGENT_LABELS[sequence.agent]}</span>
        <b className="mono">{sequence.video_id}</b>
        <span className={`badge ${complete ? "speech" : "warn"}`}>
          {sequence.found}/{props.eventCount} events
        </span>
        {sequence.found > 1 && (
          <span className={`badge ${sequence.inOrder ? "speech" : "bad"}`}>
            {sequence.inOrder ? "in order" : "out of order"}
          </span>
        )}
        <span className="agent-conf" title="Mean confidence of the chosen frames">
          {Math.round(sequence.confidence * 100)}%
        </span>
        {sequence.agreed && <span className="badge speech">both agents</span>}
        {sequence.outside && <span className="agent-outside">Outside the folder filter</span>}
        <span className="agent-seq-actions">
          <button
            className="btn sm primary"
            onClick={props.onSubmit}
            disabled={!sequence.found}
            title={complete ? "Load E1..En and open the submit guard" : "Some events are missing; the guard will say what is wrong"}
            data-testid="agent-submit-sequence"
          >
            Submit sequence
          </button>
          <button className="btn sm" onClick={props.onLoad} disabled={!sequence.found} data-testid="agent-load-sequence">
            Load into E1..E{props.eventCount}
          </button>
          <button className="btn sm" onClick={props.onSticky} disabled={!sequence.found} data-testid="agent-sticky-sequence">
            + Sticky
          </button>
          {first && (
            <button className="btn sm ghost" onClick={() => props.onPlay(first)}>▶ Play</button>
          )}
        </span>
      </div>
      <div className="agent-seq-strip">
        {sequence.events.map((candidate, index) => (
          <SequenceTile
            key={index}
            label={`E${index + 1}`}
            color={eventColor(index)}
            candidate={candidate}
            previewing={candidate != null && props.previewId === candidate.id}
            onPlay={props.onPlay}
          />
        ))}
        {sequence.extras.map((candidate) => (
          <SequenceTile
            key={candidate.id}
            label={candidate.event != null ? `E${candidate.event} alt` : "E?"}
            color="var(--fg-faint)"
            candidate={candidate}
            extra
            previewing={props.previewId === candidate.id}
            onPlay={props.onPlay}
          />
        ))}
      </div>
    </article>
  );
}

function SequenceTile(props: {
  label: string;
  color: string;
  candidate: AgentCandidate | null;
  extra?: boolean;
  previewing: boolean;
  onPlay: (candidate: AgentCandidate) => void;
}) {
  const { candidate } = props;
  if (!candidate) {
    return (
      <div className="agent-tile empty" data-testid="agent-tile-empty">
        <span className="agent-tile-head">
          <span className="event-dot" style={{ background: props.color }} /> {props.label}
        </span>
        <span className="agent-tile-missing">not reported</span>
      </div>
    );
  }
  return (
    <button
      className={`agent-tile ${props.extra ? "extra" : ""} ${props.previewing ? "previewing" : ""}`}
      onClick={() => props.onPlay(candidate)}
      title={candidate.reason || candidate.submit_keyframe_id}
      data-testid="agent-tile"
    >
      <span className="agent-tile-head">
        <span className="event-dot" style={{ background: props.color }} /> {props.label}
        <span className="agent-tile-conf mono">{Math.round(candidate.confidence * 100)}%</span>
      </span>
      <img src={candidate.keyframe_url} alt={candidate.submit_keyframe_id} loading="lazy" />
      <span className="agent-tile-meta mono">
        {formatTime(candidateTime(candidate))} · {candidate.submit_keyframe_id.split("/").pop()}
      </span>
    </button>
  );
}

export function AgentPanel(props: Props) {
  const { run, starting, error, queryType, currentQuery, resultVideoIds, pausedFrame } = props;
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);
  const isTrake = queryType === "TRAKE";
  const candidates = useMemo(() => sortCandidates(run?.candidates ?? []), [run?.candidates]);
  const consensus = useMemo(() => (isTrake ? [] : findConsensus(run?.candidates ?? [])), [isTrake, run?.candidates]);
  const agreed = useMemo(() => new Set(consensus.flatMap((c) => c.candidateIds)), [consensus]);
  const sequences = useMemo(
    () => (isTrake ? buildSequences(run?.candidates ?? [], props.eventCount) : []),
    [isTrake, run?.candidates, props.eventCount],
  );

  const preview = candidates.find((c) => c.id === previewId) ?? null;
  const playerOpen = preview !== null;
  const { onPlayerChange } = props;
  useEffect(() => {
    onPlayerChange?.(playerOpen);
  }, [playerOpen, onPlayerChange]);

  if (!run && !starting && !error) return null;
  const working = Boolean(run && !run.finished);
  const stale = Boolean(run && currentQuery.trim() && run.query.trim() !== currentQuery.trim());
  const togglePreview = (candidate: AgentCandidate) =>
    setPreviewId((current) => (current === candidate.id ? null : candidate.id));
  const scope = run ? scopeLabel(run) : null;
  const pausedHere = preview && pausedFrame && pausedFrame.video_id === preview.video_id ? pausedFrame : null;
  const agreedVideos = [...new Set(sequences.filter((s) => s.agreed).map((s) => s.video_id))];

  return (
    <section className="agent-strip" data-testid="agent-panel">
      <div className="agent-head">
        <button
          className="agent-collapse"
          onClick={() => setCollapsed((v) => !v)}
          aria-expanded={!collapsed}
          title={collapsed ? "Show the agents' answers" : "Hide the agents' answers"}
        >
          {collapsed ? "▸" : "▾"}
        </button>
        <h3>Agents · Codex + Claude</h3>
        {run && (
          <span className="agent-count">
            {isTrake
              ? `${sequences.length} sequence${sequences.length === 1 ? "" : "s"}`
              : `${candidates.length} frame${candidates.length === 1 ? "" : "s"}`}
          </span>
        )}
        {scope && (
          <span className="agent-scope" title={run?.scope?.reason} data-testid="agent-scope">{scope}</span>
        )}
        <span className="agent-head-note">Separate from the main ranking — it never changes the results list.</span>
        {working ? (
          <button className="btn sm danger" onClick={props.onStop} data-testid="agent-stop">Stop</button>
        ) : (
          <button className="btn sm ghost" onClick={props.onDismiss} title="Hide this run">✕</button>
        )}
      </div>

      {!collapsed && (
        <>
          {starting && !run && <div className="agent-note">Starting agents…</div>}
          {error && <div className="agent-error">{error}</div>}
          {run && stale && (
            <div className="agent-stale" data-testid="agent-stale">
              For an earlier query: “{run.query.length > 120 ? `${run.query.slice(0, 119)}…` : run.query}”
            </div>
          )}
          {run && (
            <div className="agent-agents">
              {AGENT_NAMES.map((name: AgentName) => {
                const state = run.agents[name];
                return state ? <AgentBox key={name} state={state} /> : null;
              })}
            </div>
          )}

          {consensus.map((item) => (
            <div className="agent-consensus" key={item.candidateIds.join("+")} data-testid="agent-consensus">
              ✓ Codex and Claude agree: <b>{item.video_id}</b> ~{formatTime(item.time)}
              {item.event != null && ` · E${item.event}`}
            </div>
          ))}
          {agreedVideos.map((videoId) => (
            <div className="agent-consensus" key={videoId} data-testid="agent-consensus">
              ✓ Codex and Claude both chose <b>{videoId}</b>
            </div>
          ))}

          {run && candidates.length === 0 && (
            <div className="agent-note">{working ? "No candidate yet." : "No candidate reported."}</div>
          )}

          {isTrake ? (
            <div className="agent-seqs">
              {sequences.map((sequence) => (
                <SequenceRow
                  key={sequence.key}
                  sequence={sequence}
                  eventCount={Math.max(props.eventCount, sequence.events.length)}
                  previewId={previewId}
                  onPlay={togglePreview}
                  onLoad={() => props.onLoadSequence(sequence)}
                  onSubmit={() => props.onSubmitSequence(sequence)}
                  onSticky={() => props.onStickySequence(sequence)}
                />
              ))}
            </div>
          ) : (
            <div className="agent-cards">
              {candidates.map((candidate) => (
                <CandidateCard
                  key={candidate.id}
                  candidate={candidate}
                  queryType={queryType}
                  agreed={agreed.has(candidate.id)}
                  inResults={resultVideoIds.has(candidate.video_id)}
                  previewing={previewId === candidate.id}
                  onPlay={() => togglePreview(candidate)}
                  onSubmit={() => props.onSubmit(candidate)}
                  onSticky={() => props.onSticky(candidate)}
                  onLocate={() => props.onLocate(candidate)}
                  onUseAnswer={props.onUseAnswer}
                />
              ))}
            </div>
          )}

          {preview && (
            <div className="agent-player" data-testid="agent-player" onPointerDown={props.onPlayerFocus}>
              <div className="agent-player-head">
                <span className={`agent-tag ${preview.agent}`}>{AGENT_LABELS[preview.agent]}</span>
                <b className="mono">{preview.video_id}</b>
                <span className="mono">from {formatTime(candidateTime(preview))}</span>
                {preview.event != null && <span className="badge">E{preview.event}</span>}
                <span className="slot-editor-keys">
                  <span className="kbd">a</span>/<span className="kbd">d</span> ±1s · <span className="kbd">←</span>/
                  <span className="kbd">→</span> ±5s · <span className="kbd">Space</span> play
                </span>
                <button className="btn sm ghost" onClick={() => setPreviewId(null)}>Close</button>
              </div>
              <VideoViewer
                ref={props.playerRef}
                src={preview.video_url}
                fallbackSrc={props.videoFallback(preview.video_url)}
                startTime={candidateTime(preview) ?? 0}
                onPaused={(rawTime, thumbnail) => {
                  const fps = preview.fps ?? 25;
                  props.onPaused({
                    video_id: preview.video_id,
                    frame_idx: Math.round(rawTime * fps),
                    pts_time: rawTime,
                    fps,
                    thumbnail,
                  });
                }}
              />
              <div className="agent-paused" data-testid="agent-paused">
                {pausedHere ? (
                  <>
                    {pausedHere.thumbnail && <img src={pausedHere.thumbnail} alt={`Paused frame ${pausedHere.frame_idx}`} />}
                    <span className="mono">
                      Paused frame <b>{pausedHere.frame_idx}</b> · {formatTime(pausedHere.pts_time)} ·{" "}
                      {pausedHere.pts_time.toFixed(3)}s
                    </span>
                    {isTrake ? (
                      <span className="agent-assign">
                        Assign to
                        {Array.from({ length: props.trakeSlotCount }, (_, index) => (
                          <button
                            key={index}
                            className={`btn sm ${index === props.activeTrakeSlot ? "primary" : ""}`}
                            onClick={() => props.onAssignPaused(index)}
                            data-testid={`agent-assign-${index + 1}`}
                          >
                            <span className="event-dot" style={{ background: eventColor(index) }} /> E{index + 1}
                          </button>
                        ))}
                      </span>
                    ) : (
                      <>
                        <button className="btn sm primary" onClick={props.onSubmitPaused} data-testid="agent-submit-paused">
                          {queryType === "AVS" ? "Add paused frame to AVS" : "Submit paused frame"}
                        </button>
                        {queryType !== "AVS" && (
                          <button className="btn sm" onClick={props.onStickyPaused}>+ Sticky</button>
                        )}
                      </>
                    )}
                  </>
                ) : (
                  <span className="agent-note">
                    Pause the video (or seek while it is paused) to capture the exact frame, then submit it here.
                  </span>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
