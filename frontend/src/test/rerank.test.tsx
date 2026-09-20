/**
 * The Rerank tick box: an opt-in, per-search refinement.
 *
 * Two properties matter more than the markup. It must default to OFF — the
 * reranker needs a second GPU worker and costs seconds, so a console that turned
 * it on by itself would slow every search an operator did not ask to slow. And
 * it must not appear at all when no worker can answer, because a tick box that
 * silently does nothing is worse than no tick box.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";

const FRAME = {
  image_id: "L26/L26_V001/001",
  submit_keyframe_id: "L26/L26_V001/001",
  video_id: "L26_V001",
  keyframe_n: 1,
  frame_idx: 0,
  fps: 25,
  pts_time: 0,
  score: 0.9,
  channels: ["image_pe"],
  per_channel_score: { image_pe: 0.9 },
  keyframe_url: "https://media.test/Keyframes/Keyframes_L26/L26_V001/001.jpg",
  video_url: "https://media.test/Videos/Videos_L26/L26_V001.mp4",
  evidence: [],
};

const PARSED = {
  query_type: "T-KIS",
  confidence: 0.4,
  translated_en_visual: "a chef cooking",
  channels: {
    image_pe: { enabled: true, weight: 1 },
    ocr: { enabled: false },
    speech: { enabled: false },
    audio: { enabled: false },
  },
  filters: { must_not_include: [] },
  trake: { enabled: false, events: [] },
  qa: { enabled: false },
  ui_hints: {},
  _engine: "heuristic",
};

let searchRequests: any[] = [];
/** What the fake backend reports for `latency_ms.reranker`, if anything. */
let rerankReport: Record<string, unknown> | null = null;

function mockFetch(capabilities: Record<string, boolean>) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const path = String(url);
    const json = (body: unknown, status = 200) =>
      ({ ok: status < 400, status, json: async () => body } as Response);

    if (path.includes("/api/search/scope"))
      return json({
        retrieval_database: "btc",
        categories: [{ category: "L26", label_vi: "Nấu ăn", open_subject: false }],
        groups: [{ id: "L", label_vi: "L", categories: ["L26"] }],
        topics: [],
      });
    if (path.includes("/api/health"))
      return json({
        ok: true,
        mode: "mock",
        retrieval_database: "btc",
        services: {},
        capabilities,
        warnings: [],
      });
    if (path.endsWith("/api/config"))
      return json({
        configured: true,
        mock_mode: true,
        missing_required: [],
        env_path: "/config/.env",
        env_exists: true,
        groups: [],
      });
    if (path.endsWith("/api/dres/status"))
      return json({ enabled: false, configured: false, connected: false });
    if (path.includes("/api/submit/history")) return json({ history: [] });
    if (path.endsWith("/api/search")) {
      const body = JSON.parse((init?.body as string) || "{}");
      searchRequests.push(body);
      return json({
        query: body.query,
        parsed: PARSED,
        scope: {
          mode: "all",
          categories: [],
          strict_categories: [],
          active: false,
          reason_vi: "",
          matched_topics: [],
        },
        groups: [
          {
            video_id: "L26_V001",
            video_score: 1,
            max_score: 0.9,
            mean_top_score: 0.9,
            frame_count: 1,
            timestamp_dispersion: 0,
            ambiguous: false,
            channels: ["image_pe"],
            video_url: FRAME.video_url,
            frames: [FRAME],
          },
        ],
        latency_ms: {
          parse_ms: 1,
          fusion_ms: 0.2,
          total_ms: 5,
          ...(rerankReport ? { reranker: rerankReport } : {}),
        },
        mode: "mock",
        warnings: [],
      });
    }
    return json({}, 404);
  });
}

async function search(user: ReturnType<typeof userEvent.setup>, text: string) {
  const box = screen.getByTestId("query-input");
  await user.clear(box);
  await user.type(box, text);
  const before = searchRequests.length;
  await user.click(screen.getByTestId("search-btn"));
  await waitFor(() => expect(searchRequests.length).toBeGreaterThan(before));
  return searchRequests[searchRequests.length - 1];
}

beforeEach(() => {
  searchRequests = [];
  rerankReport = null;
  globalThis.localStorage?.clear?.();
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("rerank tick box", () => {
  it("is hidden when no rerank worker can answer", async () => {
    vi.stubGlobal("fetch", mockFetch({ visual_rerank: false }));
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");

    await waitFor(() => expect(screen.queryByTestId("search-btn")).toBeInTheDocument());
    expect(screen.queryByTestId("rerank-toggle")).not.toBeInTheDocument();

    // ...and the search still goes out, simply without asking for reranking.
    const body = await search(user, "đầu bếp đang nấu ăn");
    expect(body.rerank).toBe(false);
  });

  it("appears unticked and searches without reranking until it is ticked", async () => {
    vi.stubGlobal("fetch", mockFetch({ visual_rerank: true }));
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");

    const box = await screen.findByTestId("rerank-checkbox");
    expect(box).not.toBeChecked();

    const before = await search(user, "đầu bếp đang nấu ăn");
    expect(before.rerank).toBe(false);
  });

  it("sends rerank=true on the next search once ticked, and back to false when unticked", async () => {
    vi.stubGlobal("fetch", mockFetch({ visual_rerank: true }));
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");

    const box = await screen.findByTestId("rerank-checkbox");
    await user.click(box);
    expect(box).toBeChecked();

    expect((await search(user, "đầu bếp đang nấu ăn")).rerank).toBe(true);

    await user.click(box);
    expect(box).not.toBeChecked();
    expect((await search(user, "đầu bếp đang nấu ăn")).rerank).toBe(false);
  });

  it("reports what the rerank stage actually did", async () => {
    rerankReport = { ok: true, candidates: 200, reranked: 200, ms: 1840 };
    vi.stubGlobal("fetch", mockFetch({ visual_rerank: true }));
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");

    await user.click(await screen.findByTestId("rerank-checkbox"));
    await search(user, "đầu bếp đang nấu ăn");

    const report = await screen.findByTestId("rerank-report");
    expect(report.textContent).toContain("200/200");
    expect(report.textContent).toContain("1840 ms");
  });

  it("says so when the worker failed and the PE order was kept", async () => {
    rerankReport = {
      ok: false,
      candidates: 200,
      reranked: 0,
      ms: 30000,
      error: "Qwen reranker unreachable: timeout",
    };
    vi.stubGlobal("fetch", mockFetch({ visual_rerank: true }));
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");

    await user.click(await screen.findByTestId("rerank-checkbox"));
    await search(user, "đầu bếp đang nấu ăn");

    const report = await screen.findByTestId("rerank-report");
    expect(report.textContent).toContain("preserve retrieval order");
    expect(report.textContent).toContain("timeout");
    expect(report.className).toContain("warn");
    // The results themselves are still on screen — a failed refinement must not
    // read as a failed search.
    expect(screen.getByTestId("query-input")).toBeInTheDocument();
  });
});
