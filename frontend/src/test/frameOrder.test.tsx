/**
 * Sorting the frames inside one video group chronologically.
 *
 * The retrieval ranking answers "which frame is most relevant"; reading a scene
 * needs the opposite — the frames in the order they happen. Each group carries
 * its own sort button and an undo back to the ranking.
 */
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { frameTimeKey, sortFramesByTime } from "../lib/media";

/** One frame of L21_V024, mirroring the console's shape. */
function frame(keyframeN: number, ptsTime: number | null, score: number) {
  const id = `L21/L21_V024/${String(keyframeN).padStart(3, "0")}`;
  return {
    image_id: id,
    submit_keyframe_id: id,
    video_id: "L21_V024",
    keyframe_n: keyframeN,
    frame_idx: ptsTime == null ? null : Math.round(ptsTime * 25),
    fps: 25,
    pts_time: ptsTime,
    score,
    channels: ["image_pe"],
    per_channel_score: { image_pe: score },
    keyframe_url: `https://media.test/Keyframes/Keyframes_L21/L21_V024/${String(keyframeN).padStart(3, "0")}.jpg`,
    video_url: "https://media.test/Videos/Videos_L21/L21_V024.mp4",
    evidence: [],
  };
}

// Deliberately out of chronological order: this is the relevance ranking, which
// is what the backend returns and what the strip shows until the sort is used.
const FRAMES = [
  frame(2967, 822, 0.98), // 13:42
  frame(2924, 808, 0.95), // 13:28
  frame(2972, 823, 0.93), // 13:43
  frame(2825, 780, 0.91), // 13:00
];

const PARSED = {
  query_type: "T-KIS",
  confidence: 0.4,
  translated_en_visual: "astronauts",
  channels: { image_pe: { enabled: true, weight: 1 }, ocr: { enabled: false }, speech: { enabled: false }, audio: { enabled: false } },
  filters: { must_not_include: [] },
  trake: { enabled: false, events: [] },
  qa: { enabled: false },
  ui_hints: {},
  _engine: "heuristic",
};

function searchResponse(frames = FRAMES) {
  return {
    query: "phi hành gia",
    parsed: PARSED,
    groups: [{
      video_id: "L21_V024",
      video_score: 0.998,
      max_score: 0.98,
      mean_top_score: 0.94,
      frame_count: frames.length,
      timestamp_dispersion: 1,
      ambiguous: false,
      channels: ["image_pe"],
      video_url: "https://media.test/Videos/Videos_L21/L21_V024.mp4",
      frames,
    }],
    latency_ms: { parse_ms: 1, fusion_ms: 0.2, total_ms: 5 },
    mode: "mock",
  };
}

let framesForSearch = FRAMES;

function mockFetch() {
  return vi.fn(async (url: string) => {
    const path = String(url);
    const json = (body: unknown, status = 200) =>
      ({ ok: status < 400, status, json: async () => body } as Response);
    if (path.includes("/api/search/scope"))
      return json({ retrieval_database: "btc", categories: [], groups: [], topics: [] });
    if (path.includes("/api/health"))
      return json({ ok: true, mode: "mock", retrieval_database: "btc", services: {}, capabilities: {}, warnings: [] });
    if (path.endsWith("/api/config"))
      return json({ configured: true, mock_mode: true, missing_required: [], env_path: "/c/.env", env_exists: true, groups: [] });
    if (path.endsWith("/api/dres/status")) return json({ enabled: false, configured: false, connected: false });
    if (path.includes("/api/submit/history")) return json({ history: [] });
    if (path.endsWith("/api/search")) return json(searchResponse(framesForSearch));
    if (path.includes("/timeline"))
      return json({ video_id: "L21_V024", fps: 25, duration: 900, video_url: "", keyframes: [], speech_segments: [], ocr_markers: [], audio_windows: [], heatmap: [] });
    return json({}, 404);
  });
}

/** Run a search and wait for the group's thumbnails to render. */
async function search(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByTestId("query-input"), "phi hành gia");
  await user.click(screen.getByTestId("search-btn"));
  await waitFor(() => expect(screen.getAllByTestId("frame-thumb").length).toBeGreaterThan(0));
}

/** The frame ids in the order the strip renders them. */
function stripOrder(): string[] {
  return screen.getAllByTestId("frame-thumb").map((thumb) => {
    const img = thumb.querySelector("img");
    return img?.getAttribute("alt") ?? "";
  });
}

const BY_RELEVANCE = FRAMES.map((f) => f.submit_keyframe_id);
const BY_TIME = [...FRAMES]
  .sort((a, b) => (a.pts_time ?? 0) - (b.pts_time ?? 0))
  .map((f) => f.submit_keyframe_id);

beforeEach(() => {
  globalThis.localStorage?.clear?.();
  framesForSearch = FRAMES;
  vi.stubGlobal("fetch", mockFetch());
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("frame order inside a video group", () => {
  it("shows the retrieval ranking until the sort is used", async () => {
    const user = userEvent.setup();
    render(<App />);
    await search(user);

    expect(stripOrder()).toEqual(BY_RELEVANCE);
    expect(screen.queryByTestId("order-tag")).not.toBeInTheDocument();
  });

  it("sorts the frames of that group chronologically", async () => {
    const user = userEvent.setup();
    render(<App />);
    await search(user);

    await user.click(screen.getByTestId("sort-frames-time"));

    expect(stripOrder()).toEqual(BY_TIME);
    expect(screen.getByTestId("order-tag")).toBeInTheDocument();
  });

  it("undo puts the relevance ranking back", async () => {
    const user = userEvent.setup();
    render(<App />);
    await search(user);
    await user.click(screen.getByTestId("sort-frames-time"));
    await user.click(screen.getByTestId("reset-frames-order"));

    expect(stripOrder()).toEqual(BY_RELEVANCE);
    expect(screen.queryByTestId("order-tag")).not.toBeInTheDocument();
  });

  it("undo is only offered once there is something to undo", async () => {
    const user = userEvent.setup();
    render(<App />);
    await search(user);

    expect(screen.getByTestId("reset-frames-order")).toBeDisabled();
    await user.click(screen.getByTestId("sort-frames-time"));
    expect(screen.getByTestId("reset-frames-order")).toBeEnabled();
  });

  it("keeps the selection on the same frame across a reorder", async () => {
    // The selection is a POSITION in the strip, so reordering under it would
    // leave Detail (and the submit guard) describing a different frame than the
    // one highlighted.
    const user = userEvent.setup();
    render(<App />);
    await search(user);

    // Select the last-ranked frame, which is the FIRST one chronologically.
    await user.click(screen.getAllByTestId("frame-thumb")[3]);
    const detail = screen.getByTestId("detail-panel");
    expect(within(detail).getByTestId("submit-id")).toHaveTextContent("L21/L21_V024/2825");

    await user.click(screen.getByTestId("sort-frames-time"));

    expect(within(screen.getByTestId("detail-panel")).getByTestId("submit-id"))
      .toHaveTextContent("L21/L21_V024/2825");
    // ...and the highlight moved with it, to the front of the sorted strip.
    expect(screen.getAllByTestId("frame-thumb")[0].className).toContain("selected");
  });

  it("a new search drops the sort", async () => {
    const user = userEvent.setup();
    render(<App />);
    await search(user);
    await user.click(screen.getByTestId("sort-frames-time"));
    expect(screen.getByTestId("order-tag")).toBeInTheDocument();

    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => expect(screen.queryByTestId("order-tag")).not.toBeInTheDocument());
    expect(stripOrder()).toEqual(BY_RELEVANCE);
  });

  it("frames with no timing stay put at the end of the strip", async () => {
    // pts_time is filled from the keyframe map, which stops at the retrieval
    // depth cap; an unknown position must not be read as position zero.
    framesForSearch = [frame(2967, 822, 0.98), frame(2900, null, 0.9), frame(2825, 780, 0.8)];
    const user = userEvent.setup();
    render(<App />);
    await search(user);

    await user.click(screen.getByTestId("sort-frames-time"));

    expect(stripOrder()).toEqual([
      "L21/L21_V024/2825",
      "L21/L21_V024/2967",
      "L21/L21_V024/2900",
    ]);
  });
});

describe("sortFramesByTime", () => {
  it("orders by pts_time and never mutates the caller's array", () => {
    const input = [frame(3, 30, 0.9), frame(1, 10, 0.8), frame(2, 20, 0.7)];
    const snapshot = [...input];

    expect(sortFramesByTime(input).map((f) => f.keyframe_n)).toEqual([1, 2, 3]);
    expect(input).toEqual(snapshot);
  });

  it("falls back to frame_idx then keyframe_n when pts_time is missing", () => {
    expect(frameTimeKey({ pts_time: 12.5, frame_idx: 999, keyframe_n: 1 })).toBe(12.5);
    expect(frameTimeKey({ pts_time: null, frame_idx: 300, keyframe_n: 1 })).toBe(300);
    expect(frameTimeKey({ pts_time: null, frame_idx: null, keyframe_n: 7 })).toBe(7);
    expect(frameTimeKey({ pts_time: null, frame_idx: null, keyframe_n: null })).toBeNull();
  });

  it("keeps the relevance order for frames at the same instant", () => {
    const a = frame(1, 10, 0.9);
    const b = frame(2, 10, 0.5);
    expect(sortFramesByTime([a, b]).map((f) => f.keyframe_n)).toEqual([1, 2]);
    expect(sortFramesByTime([b, a]).map((f) => f.keyframe_n)).toEqual([2, 1]);
  });
});
