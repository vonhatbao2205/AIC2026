/**
 * Agent sidecar search: Codex + Claude run next to the main search.
 *
 * The contract that matters most is isolation: the AGENT button (on by default)
 * adds a background run on an operator's Search press, and nothing else — the
 * main `/api/search` request is identical, never waits, and the agents' frames
 * live in their own panel.
 */
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { AgentCandidate, AgentRunSnapshot, AgentState } from "../api/types";
import { AgentPanel } from "../components/AgentPanel";
import { buildSequences, findConsensus, readAgentPreference, sortCandidates } from "../lib/agent";

function candidate(overrides: Partial<AgentCandidate>): AgentCandidate {
  return {
    id: "codex-1",
    agent: "codex",
    video_id: "L21_V001",
    submit_keyframe_id: "L21/L21_V001/010",
    keyframe_n: 10,
    frame_idx: 2250,
    fps: 25,
    pts_time: 90,
    time: 90,
    keyframe_url: "https://media.test/k.jpg",
    video_url: "https://media.test/v.mp4",
    confidence: 0.5,
    reason: "red shirt in front of a shop",
    answer: null,
    event: null,
    found_at_s: 10,
    ...overrides,
  };
}

function agentState(overrides: Partial<AgentState>): AgentState {
  return {
    name: "codex",
    status: "running",
    model: "gpt-6-sol",
    effort: "high",
    elapsed_s: 12,
    tool_calls: 3,
    summary: "",
    error: null,
    error_kind: null,
    cost_usd: null,
    steps: [{ at: 11, kind: "tool", text: "search [visual] “man in red shirt”" }],
    ...overrides,
  };
}

function snapshot(overrides: Partial<AgentRunSnapshot> = {}): AgentRunSnapshot {
  return {
    run_id: "run1",
    query: "người đàn ông áo đỏ",
    query_type: "T-KIS",
    retrieval_database: "btc",
    created_at: 0,
    elapsed_s: 12,
    finished: false,
    agents: {
      codex: agentState({ fast: true }),
      claude: agentState({ name: "claude", model: "claude-opus-5-5" }),
    },
    candidates: [],
    ...overrides,
  };
}

describe("agent helpers", () => {
  it("finds a consensus only for the same video within 3 s", () => {
    const found = findConsensus([
      candidate({ id: "codex-1", agent: "codex", time: 90 }),
      candidate({ id: "claude-1", agent: "claude", time: 92, confidence: 0.8 }),
      candidate({ id: "claude-2", agent: "claude", video_id: "L21_V002", time: 90 }),
    ]);
    expect(found).toHaveLength(1);
    expect(found[0].candidateIds).toEqual(["codex-1", "claude-1"]);
    expect(found[0].time).toBe(91);
    expect(findConsensus([
      candidate({ id: "codex-1", agent: "codex", time: 90 }),
      candidate({ id: "claude-1", agent: "claude", time: 95 }),
    ])).toHaveLength(0);
  });

  it("never pairs different TRAKE events", () => {
    expect(findConsensus([
      candidate({ id: "codex-1", agent: "codex", event: 1 }),
      candidate({ id: "claude-1", agent: "claude", event: 2 }),
    ])).toHaveLength(0);
  });

  it("orders candidates by confidence", () => {
    const sorted = sortCandidates([
      candidate({ id: "a", confidence: 0.3 }),
      candidate({ id: "b", confidence: 0.9 }),
    ]);
    expect(sorted.map((c) => c.id)).toEqual(["b", "a"]);
  });

  it("is on unless turned off", () => {
    localStorage.clear();
    expect(readAgentPreference()).toBe(true);
    localStorage.setItem("aic26_agent", "off");
    expect(readAgentPreference()).toBe(false);
    localStorage.clear();
  });
});

describe("AgentPanel", () => {
  const noop = () => {};
  const base = {
    starting: false,
    error: null,
    queryType: "T-KIS" as const,
    currentQuery: "người đàn ông áo đỏ",
    eventCount: 0,
    trakeSlotCount: 2,
    activeTrakeSlot: 0,
    resultVideoIds: new Set<string>(["L21_V001"]),
    pausedFrame: null,
    onStop: noop,
    onDismiss: noop,
    onLocate: noop,
    onSubmit: noop,
    onSticky: noop,
    onUseAnswer: noop,
    onPaused: noop,
    onSubmitPaused: noop,
    onStickyPaused: noop,
    onAssignPaused: noop,
    onLoadSequence: noop,
    onSubmitSequence: noop,
    onStickySequence: noop,
    videoFallback: () => null,
  };

  it("renders nothing before any run", () => {
    const { container } = render(<AgentPanel {...base} run={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows both agents, their progress, candidates and the consensus", async () => {
    const onLocate = vi.fn();
    render(
      <AgentPanel
        {...base}
        onLocate={onLocate}
        run={snapshot({
          candidates: [
            candidate({ id: "codex-1", agent: "codex", confidence: 0.6 }),
            candidate({ id: "claude-1", agent: "claude", confidence: 0.9, time: 91 }),
          ],
        })}
      />,
    );
    expect(screen.getByTestId("agent-row-codex")).toHaveTextContent("gpt-6-sol · high · fast");
    expect(screen.getByTestId("agent-row-claude")).not.toHaveTextContent("fast");
    expect(screen.getByTestId("agent-row-codex")).toHaveTextContent("search [visual] “man in red shirt”");
    expect(screen.getByTestId("agent-row-claude")).toHaveTextContent("claude-opus-5-5");
    expect(screen.getByTestId("agent-consensus")).toHaveTextContent("Codex and Claude agree: L21_V001");
    const items = screen.getAllByTestId("agent-candidate");
    expect(items[0]).toHaveTextContent("Claude");
    expect(items[0]).toHaveTextContent("90%");
    await userEvent.click(within(items[0]).getByTestId("agent-locate"));
    expect(onLocate).toHaveBeenCalledWith(expect.objectContaining({ id: "claude-1" }));
    expect(screen.getByTestId("agent-stop")).toBeInTheDocument();
  });

  it("explains a quota failure and flags a run for an older query", () => {
    render(
      <AgentPanel
        {...base}
        currentQuery="something else"
        run={snapshot({
          finished: true,
          agents: {
            codex: agentState({ status: "failed", error: "usage limit", error_kind: "quota" }),
          },
        })}
      />,
    );
    expect(screen.getByTestId("agent-row-codex")).toHaveTextContent("Quota / usage limit reached");
    expect(screen.getByTestId("agent-stale")).toHaveTextContent("For an earlier query");
    expect(screen.queryByTestId("agent-stop")).not.toBeInTheDocument();
  });

  it("submits, pins and answers from a frame card, and flags one outside the filter", async () => {
    const onSubmit = vi.fn();
    const onSticky = vi.fn();
    const onUseAnswer = vi.fn();
    render(
      <AgentPanel
        {...base}
        queryType="QA"
        onSubmit={onSubmit}
        onSticky={onSticky}
        onUseAnswer={onUseAnswer}
        run={snapshot({
          scope: { mode: "manual", active: true, categories: ["N003"], reason: "Manual: 1 folders (N003)." },
          candidates: [candidate({ id: "codex-1", answer: "5", outside_scope: true })],
        })}
      />,
    );
    expect(screen.getByTestId("agent-scope")).toHaveTextContent("Filter: N003");
    const card = screen.getByTestId("agent-candidate");
    expect(within(card).getByTestId("agent-outside")).toHaveTextContent("Outside the folder filter");
    await userEvent.click(within(card).getByTestId("agent-submit"));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ id: "codex-1" }));
    await userEvent.click(within(card).getByTestId("agent-sticky"));
    expect(onSticky).toHaveBeenCalledWith(expect.objectContaining({ id: "codex-1" }));
    await userEvent.click(within(card).getByRole("button", { name: "Use" }));
    expect(onUseAnswer).toHaveBeenCalledWith("5");
  });

  it("captures the paused frame of its own player and submits it there", async () => {
    const onSubmitPaused = vi.fn();
    const { rerender } = render(
      <AgentPanel {...base} onSubmitPaused={onSubmitPaused} run={snapshot({ candidates: [candidate({})] })} />,
    );
    await userEvent.click(screen.getByTestId("agent-play"));
    expect(screen.getByTestId("agent-player")).toBeInTheDocument();
    expect(screen.getByTestId("agent-paused")).toHaveTextContent("Pause the video");
    rerender(
      <AgentPanel
        {...base}
        onSubmitPaused={onSubmitPaused}
        pausedFrame={{ video_id: "L21_V001", frame_idx: 2263, pts_time: 90.52, fps: 25 }}
        run={snapshot({ candidates: [candidate({})] })}
      />,
    );
    expect(screen.getByTestId("agent-paused")).toHaveTextContent("Paused frame 2263");
    await userEvent.click(screen.getByTestId("agent-submit-paused"));
    expect(onSubmitPaused).toHaveBeenCalled();
  });

  it("shows TRAKE answers as one row of E1..En per agent and video", async () => {
    const onSubmitSequence = vi.fn();
    const onLoadSequence = vi.fn();
    render(
      <AgentPanel
        {...base}
        queryType="TRAKE"
        eventCount={3}
        trakeSlotCount={3}
        onSubmitSequence={onSubmitSequence}
        onLoadSequence={onLoadSequence}
        run={snapshot({
          query_type: "TRAKE",
          candidates: [
            candidate({ id: "codex-1", event: 1, frame_idx: 100, time: 4, confidence: 0.8 }),
            candidate({ id: "codex-2", event: 2, frame_idx: 300, time: 12, confidence: 0.6 }),
            candidate({ id: "claude-1", agent: "claude", event: 1, frame_idx: 110, time: 4.4 }),
          ],
        })}
      />,
    );
    const rows = screen.getAllByTestId("agent-sequence");
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent("Codex");
    expect(rows[0]).toHaveTextContent("2/3 events");
    expect(rows[0]).toHaveTextContent("in order");
    expect(within(rows[0]).getAllByTestId("agent-tile")).toHaveLength(2);
    expect(within(rows[0]).getByTestId("agent-tile-empty")).toHaveTextContent("E3");
    expect(screen.getByTestId("agent-consensus")).toHaveTextContent("both chose L21_V001");
    await userEvent.click(within(rows[0]).getByTestId("agent-submit-sequence"));
    const submitted = onSubmitSequence.mock.calls[0][0];
    expect(submitted.events.map((c: AgentCandidate | null) => c?.id ?? null)).toEqual(["codex-1", "codex-2", null]);
    await userEvent.click(within(rows[0]).getByTestId("agent-load-sequence"));
    expect(onLoadSequence).toHaveBeenCalled();
  });
});

describe("buildSequences", () => {
  it("keeps the most confident frame per event and the rest as extras", () => {
    const [sequence] = buildSequences([
      candidate({ id: "a", event: 1, frame_idx: 500, confidence: 0.4 }),
      candidate({ id: "b", event: 1, frame_idx: 520, confidence: 0.9 }),
      candidate({ id: "c", event: 2, frame_idx: 100, confidence: 0.7 }),
      candidate({ id: "d", event: null, frame_idx: 900 }),
    ], 2);
    expect(sequence.events.map((c) => c?.id)).toEqual(["b", "c"]);
    expect(sequence.extras.map((c) => c.id).sort()).toEqual(["a", "d"]);
    expect(sequence.inOrder).toBe(false);
    expect(sequence.found).toBe(2);
  });

  it("widens a row to the highest event an agent numbered", () => {
    const [sequence] = buildSequences([candidate({ event: 4 })], 2);
    expect(sequence.events).toHaveLength(4);
  });
});

// ---- console wiring ------------------------------------------------------------
const PARSED = {
  query_type: "T-KIS",
  confidence: 0.4,
  original_query: "người đàn ông áo đỏ",
  normalized_vi: "người đàn ông áo đỏ",
  translated_en_visual: "a man in a red shirt",
  channels: { image_pe: { enabled: true, weight: 1 } },
  filters: {},
  trake: { enabled: false, events: [] },
  qa: { enabled: false },
  ui_hints: {},
  _engine: "heuristic",
};

type Recorded = { path: string; method: string; body: Record<string, unknown> };
let requests: Recorded[] = [];
/** What starting a run answers with; a test hands the agents' frames back here. */
let agentStart: (runId: string) => AgentRunSnapshot = (runId) => snapshot({ run_id: runId });

function mockFetch() {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const path = String(url);
    const method = init?.method ?? "GET";
    const json = (body: unknown, status = 200) =>
      ({ ok: status < 400, status, json: async () => body } as Response);
    let body: Record<string, unknown> = {};
    try {
      body = JSON.parse(String(init?.body ?? "{}"));
    } catch {
      /* not JSON */
    }
    if (path.includes("/api/agent/") || path.endsWith("/api/search")) requests.push({ path, method, body });
    if (path.includes("/api/search/scope"))
      return json({ retrieval_database: "btc", categories: [], groups: [], topics: [] });
    if (path.includes("/api/health"))
      return json({
        ok: true, mode: "mock", retrieval_database: "btc", services: {},
        capabilities: { agent_codex: true, agent_claude: true }, warnings: [],
      });
    if (path.endsWith("/api/config"))
      return json({ configured: true, mock_mode: true, missing_required: [], env_path: "", env_exists: true, groups: [] });
    if (path.endsWith("/api/dres/status")) return json({ enabled: false, configured: false, connected: false });
    if (path.includes("/api/submit/history")) return json({ history: [] });
    if (path.endsWith("/api/agent/runs") && method === "POST") return json(agentStart(`run${requests.length}`));
    if (path.includes("/api/agent/runs/")) return json(snapshot({ finished: true }));
    if (path.endsWith("/api/search/trake"))
      return json({
        retrieval_database: "btc", query: "q",
        parsed: { ...PARSED, query_type: "TRAKE", trake: { enabled: true, events: [] } },
        events: [], sequences: [], videos: [], mode: "mock",
      });
    if (path.endsWith("/api/search"))
      return json({
        query: "q", retrieval_database: "btc", parsed: PARSED,
        scope: { mode: "all", categories: [], strict_categories: [], active: false, reason_vi: "", matched_topics: [] },
        groups: [], latency_ms: { total_ms: 1 }, mode: "mock", warnings: [],
      });
    return json({}, 404);
  });
}

beforeEach(() => {
  requests = [];
  agentStart = (runId) => snapshot({ run_id: runId });
  localStorage.clear();
  vi.stubGlobal("fetch", mockFetch());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

async function typeAndSearch(user: ReturnType<typeof userEvent.setup>) {
  await user.type(await screen.findByTestId("query-input"), "người đàn ông áo đỏ");
  await user.click(screen.getByTestId("search-btn"));
  await waitFor(() => expect(requests.some((r) => r.path.endsWith("/api/search"))).toBe(true));
}

describe("AGENT button", () => {
  it("is on by default and starts both agents next to the unchanged main search", async () => {
    const user = userEvent.setup();
    render(<App />);
    const toggle = await screen.findByTestId("agent-toggle");
    await waitFor(() => expect(toggle).toHaveTextContent("Agent on"));
    await typeAndSearch(user);
    await waitFor(() => expect(requests.some((r) => r.path.endsWith("/api/agent/runs"))).toBe(true));
    const start = requests.find((r) => r.path.endsWith("/api/agent/runs"))!;
    expect(start.body).toMatchObject({
      query: "người đàn ông áo đỏ",
      query_type: "T-KIS",
      retrieval_database: "btc",
      agents: ["codex", "claude"],
    });
    const main = requests.find((r) => r.path.endsWith("/api/search"))!;
    expect(main.body).not.toHaveProperty("agents");
    expect(await screen.findByTestId("agent-panel")).toBeInTheDocument();
  });

  it("starts no agent when turned off, and remembers that", async () => {
    const user = userEvent.setup();
    render(<App />);
    const toggle = await screen.findByTestId("agent-toggle");
    await user.click(toggle);
    expect(toggle).toHaveTextContent("Agent off");
    expect(localStorage.getItem("aic26_agent")).toBe("off");
    await typeAndSearch(user);
    expect(requests.some((r) => r.path.endsWith("/api/agent/runs"))).toBe(false);
  });

  it("hands the previous run to the backend to cancel on a new search", async () => {
    const user = userEvent.setup();
    render(<App />);
    await typeAndSearch(user);
    await waitFor(() => expect(requests.filter((r) => r.path.endsWith("/api/agent/runs"))).toHaveLength(1));
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => expect(requests.filter((r) => r.path.endsWith("/api/agent/runs"))).toHaveLength(2));
    const [first, second] = requests.filter((r) => r.path.endsWith("/api/agent/runs"));
    expect(first.body.replaces ?? null).toBeNull();
    expect(second.body.replaces).toMatch(/^run/);
  });

  it("submits an agent's frame through the same guard as a result keyframe", async () => {
    agentStart = (runId) => snapshot({ run_id: runId, finished: true, candidates: [candidate({ id: "codex-1" })] });
    const user = userEvent.setup();
    render(<App />);
    await typeAndSearch(user);
    const card = await screen.findByTestId("agent-candidate");
    await user.click(within(card).getByTestId("agent-submit"));
    expect(await screen.findByTestId("guard-target")).toHaveTextContent("Codex keyframe");
    expect(screen.getByTestId("guard-submit-id")).toHaveTextContent("L21_V001");
    expect(screen.getByTestId("guard-frame-idx")).toHaveTextContent("2250");
  });

  it("loads an agent's TRAKE sequence into the event slots and submits it", async () => {
    agentStart = (runId) => snapshot({
      run_id: runId,
      finished: true,
      query_type: "TRAKE",
      candidates: [
        candidate({ id: "codex-1", event: 1, frame_idx: 100, pts_time: 4, time: 4 }),
        candidate({ id: "codex-2", event: 2, frame_idx: 300, pts_time: 12, time: 12 }),
      ],
    });
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");
    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), "người đàn ông áo đỏ");
    await user.click(screen.getByTestId("search-btn"));
    const row = await screen.findByTestId("agent-sequence");
    expect(row).toHaveTextContent("2/2 events");

    await user.click(within(row).getByTestId("agent-load-sequence"));
    expect(screen.getByTestId("trake-slot-0")).toHaveTextContent("f100");
    expect(screen.getByTestId("trake-slot-1")).toHaveTextContent("f300");

    await user.click(within(row).getByTestId("agent-submit-sequence"));
    expect(await screen.findByTestId("submit-guard")).toBeInTheDocument();
  });

  it("scrubs the Agents strip's player with a/d and the arrows", async () => {
    agentStart = (runId) => snapshot({ run_id: runId, finished: true, candidates: [candidate({ id: "codex-1" })] });
    const user = userEvent.setup();
    render(<App />);
    await typeAndSearch(user);
    await user.click(within(await screen.findByTestId("agent-candidate")).getByTestId("agent-play"));
    const video = within(screen.getByTestId("agent-player")).getByTestId("video-viewer") as HTMLVideoElement;
    video.currentTime = 30;
    (document.activeElement as HTMLElement)?.blur();

    fireEvent.keyDown(window, { key: "d" });
    expect(video.currentTime).toBeCloseTo(31, 3);
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(video.currentTime).toBeCloseTo(36, 3);
    fireEvent.keyDown(window, { key: "a" });
    expect(video.currentTime).toBeCloseTo(35, 3);
  });

  it("re-picks a TRAKE slot's frame: click the slot, v, scrub, Enter", async () => {
    agentStart = (runId) => snapshot({
      run_id: runId,
      finished: true,
      query_type: "TRAKE",
      candidates: [
        candidate({ id: "codex-1", event: 1, frame_idx: 100, pts_time: 4, time: 4 }),
        candidate({ id: "codex-2", event: 2, frame_idx: 300, pts_time: 12, time: 12 }),
      ],
    });
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");
    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), "người đàn ông áo đỏ");
    await user.click(screen.getByTestId("search-btn"));
    await user.click(within(await screen.findByTestId("agent-sequence")).getByTestId("agent-load-sequence"));

    await user.click(screen.getByTestId("trake-slot-1"));
    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    const editor = await screen.findByTestId("trake-slot-editor");
    expect(editor).toHaveTextContent("Editing E2");
    expect(editor).toHaveTextContent("current f300");

    const video = within(editor).getByTestId("video-viewer") as HTMLVideoElement;
    video.currentTime = 20;
    fireEvent.keyDown(window, { key: "d" });
    expect(video.currentTime).toBeCloseTo(21, 3);
    // A seek that lands while the video is paused is the frame pick.
    fireEvent(video, new Event("seeked"));
    // 25 fps, from the slot's own frame number over its time.
    await waitFor(() => expect(screen.getByTestId("slot-editor-bar")).toHaveTextContent("Paused frame 525"));

    fireEvent.keyDown(window, { key: "Enter" });
    await waitFor(() => expect(screen.getByTestId("trake-slot-1")).toHaveTextContent("f525"));
    expect(screen.getByTestId("trake-slot-0")).toHaveTextContent("f100");
    expect(screen.getByTestId("trake-slot-editor")).toHaveTextContent("current f525");
  });
});
