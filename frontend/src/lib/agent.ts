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

/** The raw frame number a candidate stands for: the keyframe map's, else pts × fps. */
export function candidateFrameIdx(candidate: AgentCandidate): number {
  if (candidate.frame_idx != null) return candidate.frame_idx;
  return Math.round((candidate.pts_time ?? candidate.time ?? 0) * (candidate.fps ?? 25));
}

/** One agent's TRAKE answer inside one video: a frame per event, E1..En. */
export interface AgentSequence {
  key: string;
  agent: AgentName;
  video_id: string;
  /** Index i is event i+1; null where the agent reported nothing for it. */
  events: (AgentCandidate | null)[];
  /** Frames the agent sent without an event number, or lost to a better pick. */
  extras: AgentCandidate[];
  found: number;
  /** The reported frames run forward in time, E1 first. */
  inOrder: boolean;
  /** Mean confidence of the chosen frames. */
  confidence: number;
  outside: boolean;
  /** The other agent chose the same video. */
  agreed: boolean;
}

/** Group TRAKE candidates into one sequence per (agent, video).
 *
 *  `eventCount` is the tab's number of events, so every row is as wide as the
 *  answer has to be; an agent that numbered more events widens its own row.
 *  Where an agent reported several frames for one event, the most confident
 *  is the one in the chain and the rest are kept as extras. */
export function buildSequences(candidates: AgentCandidate[], eventCount: number): AgentSequence[] {
  const byKey = new Map<string, AgentCandidate[]>();
  for (const candidate of candidates) {
    const key = `${candidate.agent}|${candidate.video_id}`;
    byKey.set(key, [...(byKey.get(key) ?? []), candidate]);
  }
  const sequences: AgentSequence[] = [];
  for (const [key, items] of byKey) {
    const numbered = items.filter((c) => c.event != null && c.event >= 1);
    const width = Math.max(eventCount, 1, ...numbered.map((c) => c.event as number));
    const events: (AgentCandidate | null)[] = Array(width).fill(null);
    const extras: AgentCandidate[] = items.filter((c) => c.event == null || c.event < 1);
    for (const candidate of sortCandidates(numbered)) {
      const slot = (candidate.event as number) - 1;
      if (events[slot] === null) events[slot] = candidate;
      else extras.push(candidate);
    }
    const chosen = events.filter((c): c is AgentCandidate => c !== null);
    const times = chosen.map((c) => candidateFrameIdx(c));
    sequences.push({
      key,
      agent: items[0].agent,
      video_id: items[0].video_id,
      events,
      extras: extras.sort((a, b) => (candidateTime(a) ?? 0) - (candidateTime(b) ?? 0)),
      found: chosen.length,
      inOrder: times.every((t, i) => i === 0 || t > times[i - 1]),
      confidence: chosen.length ? chosen.reduce((sum, c) => sum + c.confidence, 0) / chosen.length : 0,
      outside: items.some((c) => c.outside_scope),
      agreed: false,
    });
  }
  for (const sequence of sequences) {
    sequence.agreed = sequences.some(
      (other) => other.agent !== sequence.agent && other.video_id === sequence.video_id && other.found > 0,
    ) && sequence.found > 0;
  }
  return sequences.sort((a, b) => b.found - a.found || b.confidence - a.confidence);
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
