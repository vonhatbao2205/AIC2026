/**
 * The "Translate VI→EN" tick box.
 *
 * Translation is on by default and must stay that way: PE-Core is an
 * English-centric encoder, so a Vietnamese query it never sees translated comes
 * back with keyframes that look plausible and are not. But it was also the only
 * behaviour available, and it is wrong often enough to matter — a query already
 * in English, a proper noun the translator "helpfully" turns into a common noun,
 * a phrase it mangles. So the box has to genuinely reach every path a query
 * takes: the console search, TRAKE, the simple view, and voice input, which used
 * to rewrite what the operator said into English with no way to stop it.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";

const PARSED = {
  query_type: "T-KIS",
  confidence: 0.4,
  original_query: "người đàn ông đang nấu ăn",
  normalized_vi: "người đàn ông đang nấu ăn",
  translated_en_visual: "a man cooking",
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

const VI = "người đàn ông đang nấu ăn";

type Recorded = { path: string; body: Record<string, unknown> };
let requests: Recorded[] = [];
/** Every URL the transcribe endpoint was called with, query string included. */
let transcribeUrls: string[] = [];

function mockFetch() {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const path = String(url);
    const json = (body: unknown, status = 200) =>
      ({ ok: status < 400, status, json: async () => body } as Response);
    const parseBody = () => {
      try {
        return JSON.parse(String(init?.body ?? "{}"));
      } catch {
        return {};
      }
    };

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
        capabilities: { visual_rerank: true },
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
    if (path.includes("/api/transcribe")) {
      transcribeUrls.push(path);
      const translated = !path.includes("translate=false");
      return json({ text: VI, text_en: translated ? "a man cooking" : null });
    }
    if (path.endsWith("/api/translate")) {
      requests.push({ path, body: parseBody() });
      return json({ text: VI, text_en: "a man cooking" });
    }
    if (path.endsWith("/api/search/simple")) {
      requests.push({ path, body: parseBody() });
      return json({
        query: VI,
        retrieval_database: "btc",
        image_models: ["pe"],
        scope: { mode: "all", categories: [], strict_categories: [], active: false, reason_vi: "", matched_topics: [] },
        translated_query: null,
        results: [],
        mode: "mock",
        latency_ms: 1,
      });
    }
    if (path.endsWith("/api/search/trake")) {
      requests.push({ path, body: parseBody() });
      return json({
        retrieval_database: "btc",
        query: VI,
        parsed: { ...PARSED, query_type: "TRAKE", trake: { enabled: true, events: [] } },
        events: [],
        sequences: [],
        videos: [],
        mode: "mock",
      });
    }
    if (path.endsWith("/api/search")) {
      requests.push({ path, body: parseBody() });
      return json({
        query: VI,
        retrieval_database: "btc",
        parsed: PARSED,
        scope: { mode: "all", categories: [], strict_categories: [], active: false, reason_vi: "", matched_topics: [] },
        groups: [],
        latency_ms: { parse_ms: 1, fusion_ms: 0.2, total_ms: 5 },
        mode: "mock",
        warnings: [],
      });
    }
    return json({}, 404);
  });
}

/** A MediaRecorder that hands the component one chunk and stops on demand. */
class FakeMediaRecorder {
  state = "inactive";
  mimeType = "audio/webm";
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  constructor(public stream: unknown) {}
  start() {
    this.state = "recording";
  }
  stop() {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["audio"], { type: "audio/webm" }) });
    this.onstop?.();
  }
}

async function search(user: ReturnType<typeof userEvent.setup>, path: string) {
  const before = requests.length;
  await user.click(screen.getByTestId("search-btn"));
  await waitFor(() => expect(requests.length).toBeGreaterThan(before));
  return requests.filter((request) => request.path.endsWith(path)).at(-1)!.body;
}

/** Records one utterance and returns once the transcript is in the query box. */
async function dictate(user: ReturnType<typeof userEvent.setup>) {
  const button = screen.getByTestId("voice-btn");
  await user.click(button);
  await waitFor(() => expect(button.textContent).toContain("rec"));
  await user.click(button);
  await waitFor(() => expect(transcribeUrls).toHaveLength(1));
}

beforeEach(() => {
  requests = [];
  transcribeUrls = [];
  globalThis.localStorage?.clear?.();
  vi.stubGlobal("fetch", mockFetch());
  vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: async () => ({ getTracks: () => [] }) },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("translate tick box", () => {
  it("is ticked by default, because an untranslated query silently returns wrong frames", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");

    expect(await screen.findByTestId("translate-checkbox")).toBeChecked();

    await user.type(screen.getByTestId("query-input"), VI);
    expect((await search(user, "/api/search")).translate).toBe(true);
  });

  it("sends translate=false on the next search once unticked, and back on when re-ticked", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");
    await user.type(screen.getByTestId("query-input"), VI);

    const box = await screen.findByTestId("translate-checkbox");
    await user.click(box);
    expect(box).not.toBeChecked();
    expect((await search(user, "/api/search")).translate).toBe(false);

    await user.click(box);
    expect((await search(user, "/api/search")).translate).toBe(true);
  });

  it("carries the choice into a TRAKE search too", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");

    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), VI);
    await user.click(await screen.findByTestId("translate-checkbox"));

    expect((await search(user, "/api/search/trake")).translate).toBe(false);
  });

  it("is offered in the simple vector view as well", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");
    await user.click(screen.getByText("Simple ⤴"));

    const box = await screen.findByTestId("translate-checkbox");
    expect(box).toBeChecked();
    await user.click(box);

    await user.type(screen.getByTestId("query-input"), VI);
    expect((await search(user, "/api/search/simple")).translate).toBe(false);
  });
});

describe("voice input", () => {
  it("translates the dictated sentence while the box is ticked", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");

    await dictate(user);

    expect(transcribeUrls[0]).toContain("translate=true");
    await waitFor(() =>
      expect(screen.getByTestId("query-input")).toHaveValue("a man cooking"),
    );
  });

  it("keeps the operator's own words when the box is unticked", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("query-input");
    await user.click(await screen.findByTestId("translate-checkbox"));

    await dictate(user);

    expect(transcribeUrls[0]).toContain("translate=false");
    await waitFor(() => expect(screen.getByTestId("query-input")).toHaveValue(VI));
    // ...and the search that follows is not translated behind their back either.
    expect((await search(user, "/api/search")).translate).toBe(false);
  });
});
