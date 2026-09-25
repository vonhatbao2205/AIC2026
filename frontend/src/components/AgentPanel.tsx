import { useMemo, useState } from "react";
import type { AgentCandidate, AgentName, AgentRunSnapshot, AgentState } from "../api/types";
import {
  AGENT_LABELS,
  AGENT_NAMES,
  candidateTime,
  findConsensus,
  isAgentWorking,
  sortCandidates,
} from "../lib/agent";
import { formatTime } from "../lib/media";
import type { PausedFrame } from "./PausedFramePanel";
import { VideoViewer } from "./VideoViewer";

interface Props {
  run: AgentRunSnapshot | null;
  starting: boolean;
  error: string | null;
  /** The query currently in the box, to flag a run that belongs to an older one. */
  currentQuery: string;
  /** Videos present in the main results, so a candidate can jump there. */
  resultVideoIds: Set<string>;
  onStop: () => void;
  onDismiss: () => void;
  /** Select the candidate's video in the main results and play it there. */
  onLocate: (candidate: AgentCandidate) => void;
  /** A frame paused in this panel's own preview, handed to the paused-frame submit path. */
  onPaused: (frame: PausedFrame) => void;
  /** The same video under the fallback origin, when one is configured. */
  videoFallback: (url: string) => string | null;
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

function AgentRow({ state }: { state: AgentState }) {
  const [open, setOpen] = useState(false);
  const working = isAgentWorking(state);
  const last = state.steps[state.steps.length - 1];
  const error = agentError(state);
  return (
    <div className={`agent-row ${state.status}`} data-testid={`agent-row-${state.name}`}>
      <button className="agent-row-head" onClick={() => setOpen((v) => !v)} title="Show this agent's steps">
        <span className={`agent-dot ${working ? "live" : state.status}`} />
        <b>{AGENT_LABELS[state.name]}</b>
        <span className="agent-model">{state.model} · {state.effort}</span>
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

export function AgentPanel(props: Props) {
  const { run, starting, error, currentQuery, resultVideoIds, onStop, onDismiss, onLocate, onPaused, videoFallback } = props;
  const [previewId, setPreviewId] = useState<string | null>(null);
  const candidates = useMemo(() => sortCandidates(run?.candidates ?? []), [run?.candidates]);
  const consensus = useMemo(() => findConsensus(run?.candidates ?? []), [run?.candidates]);
  const agreed = useMemo(() => new Set(consensus.flatMap((c) => c.candidateIds)), [consensus]);

  if (!run && !starting && !error) return null;
  const working = Boolean(run && !run.finished);
  const stale = Boolean(run && currentQuery.trim() && run.query.trim() !== currentQuery.trim());
  const preview = candidates.find((c) => c.id === previewId) ?? null;

  return (
    <div className="panel agent-panel" data-testid="agent-panel">
      <div className="agent-head">
        <h3>Agents · Codex + Claude</h3>
        {working ? (
          <button className="btn sm danger" onClick={onStop} data-testid="agent-stop">Stop</button>
        ) : (
          <button className="btn sm ghost" onClick={onDismiss} title="Hide this run">✕</button>
        )}
      </div>
      <div className="agent-note">
        Separate from the main ranking — candidates here never change the results list.
      </div>
      {starting && !run && <div className="agent-note">Starting agents…</div>}
      {error && <div className="agent-error">{error}</div>}
      {run && stale && (
        <div className="agent-stale" data-testid="agent-stale">
          For an earlier query: “{run.query.length > 80 ? `${run.query.slice(0, 79)}…` : run.query}”
        </div>
      )}
      {run && AGENT_NAMES.map((name: AgentName) => {
        const state = run.agents[name];
        return state ? <AgentRow key={name} state={state} /> : null;
      })}

      {consensus.map((item) => (
        <div className="agent-consensus" key={item.candidateIds.join("+")} data-testid="agent-consensus">
          ✓ Codex and Claude agree: <b>{item.video_id}</b> ~{formatTime(item.time)}
          {item.event != null && ` · E${item.event}`}
        </div>
      ))}

      {run && candidates.length === 0 && (
        <div className="agent-note">{working ? "No candidate yet." : "No candidate reported."}</div>
      )}
      <ul className="agent-candidates">
        {candidates.map((candidate) => {
          const time = candidateTime(candidate);
          const inResults = resultVideoIds.has(candidate.video_id);
          const previewing = previewId === candidate.id;
          return (
            <li
              key={candidate.id}
              className={`agent-candidate ${agreed.has(candidate.id) ? "agreed" : ""}`}
              data-testid="agent-candidate"
            >
              <button
                className="agent-thumb"
                onClick={() => setPreviewId(previewing ? null : candidate.id)}
                title="Play here from this moment"
              >
                <img src={candidate.keyframe_url} alt={candidate.submit_keyframe_id} loading="lazy" />
              </button>
              <div className="agent-candidate-body">
                <div className="agent-candidate-title">
                  <span className={`agent-tag ${candidate.agent}`}>{AGENT_LABELS[candidate.agent]}</span>
                  <b className="mono">{candidate.video_id}</b>
                  <span className="mono">{formatTime(time)}</span>
                  {candidate.event != null && <span className="badge">E{candidate.event}</span>}
                  <span className="agent-conf" title="Agent's own confidence">{Math.round(candidate.confidence * 100)}%</span>
                </div>
                {candidate.answer && <div className="agent-answer">Answer: <b>{candidate.answer}</b></div>}
                {candidate.reason && <div className="agent-reason">{candidate.reason}</div>}
                <div className="agent-actions">
                  <button className="btn sm" onClick={() => setPreviewId(previewing ? null : candidate.id)}>
                    {previewing ? "Close" : "▶ Play"}
                  </button>
                  {inResults && (
                    <button className="btn sm" onClick={() => onLocate(candidate)} data-testid="agent-locate">
                      Show in results
                    </button>
                  )}
                  <span className="agent-kf mono">{candidate.submit_keyframe_id}</span>
                </div>
              </div>
              {previewing && preview && (
                <div className="agent-preview">
                  <VideoViewer
                    src={preview.video_url}
                    fallbackSrc={videoFallback(preview.video_url)}
                    startTime={candidateTime(preview) ?? 0}
                    onPaused={(rawTime, thumbnail) => {
                      const fps = preview.fps ?? 25;
                      onPaused({
                        video_id: preview.video_id,
                        frame_idx: Math.round(rawTime * fps),
                        pts_time: rawTime,
                        fps,
                        thumbnail,
                      });
                    }}
                  />
                  <div className="agent-note">Pause on the exact frame, then submit it from “Paused frame”.</div>
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
