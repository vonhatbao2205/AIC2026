// Agent sidecar search (Codex CLI / Claude Code CLI): pure helpers for the panel.
import type { AgentCandidate, AgentName, AgentState } from "../api/types";

export const AGENT_NAMES: AgentName[] = ["codex", "claude"];

export const AGENT_LABELS: Record<AgentName, string> = { codex: "Codex", claude: "Claude" };

/** Two agents naming the same video within this many seconds found the same moment. */
export const CONSENSUS_WINDOW_S = 3;

const PREFERENCE_KEY = "aic26_agent";

/** The AGENT button is ON unless the operator turned it off on this browser. */
export function readAgentPreference(): boolean {
  try {
    return localStorage.getItem(PREFERENCE_KEY) !== "off";
  } catch {
    return true;
  }
}

export function writeAgentPreference(on: boolean): void {
  try {
    localStorage.setItem(PREFERENCE_KEY, on ? "on" : "off");
  } catch {
    /* private mode / blocked storage: the button still works for this session */
  }
}

export function isAgentWorking(state: AgentState | undefined): boolean {
  return state?.status === "pending" || state?.status === "queued" || state?.status === "running";
}

export function candidateTime(candidate: AgentCandidate): number | null {
  return candidate.time ?? candidate.pts_time;
}

/** Best first: confidence, then whoever reported it earlier. */
export function sortCandidates(candidates: AgentCandidate[]): AgentCandidate[] {
  return [...candidates].sort(
    (a, b) => b.confidence - a.confidence || a.found_at_s - b.found_at_s,
  );
}

export interface AgentConsensus {
  video_id: string;
  /** Mean of the two agents' times. */
  time: number;
  event: number | null;
  candidateIds: string[];
  confidence: number;
}

/** Moments BOTH agents reported on their own: same video (and TRAKE event),
 *  within `CONSENSUS_WINDOW_S`. Two independent searches converging is a strong
 *  signal, but it is shown as one, never submitted automatically. */
export function findConsensus(candidates: AgentCandidate[]): AgentConsensus[] {
  const codex = candidates.filter((c) => c.agent === "codex");
  const claude = candidates.filter((c) => c.agent === "claude");
  const found: AgentConsensus[] = [];
  const used = new Set<string>();
  for (const a of codex) {
    const ta = candidateTime(a);
    if (ta == null) continue;
    let best: AgentCandidate | null = null;
    let bestDelta = Infinity;
    for (const b of claude) {
      const tb = candidateTime(b);
      if (used.has(b.id) || tb == null || b.video_id !== a.video_id) continue;
      if ((a.event ?? null) !== (b.event ?? null)) continue;
      const delta = Math.abs(ta - tb);
      if (delta <= CONSENSUS_WINDOW_S && delta < bestDelta) {
        best = b;
        bestDelta = delta;
      }
    }
    if (!best) continue;
    used.add(best.id);
    found.push({
      video_id: a.video_id,
      time: (ta + (candidateTime(best) as number)) / 2,
      event: a.event ?? null,
      candidateIds: [a.id, best.id],
      confidence: Math.max(a.confidence, best.confidence),
    });
  }
  return found.sort((x, y) => y.confidence - x.confidence);
}
