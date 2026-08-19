/**
 * The folder-scope filter: the checkbox list an operator opens to say which
 * dataset folders may answer, and the auto scope the backend resolves for them.
 */
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { scopeButtonLabel, scopeRequest, toggleCategory } from "../lib/scope";
import type { ScopeCatalogue } from "../api/types";

const L = Array.from({ length: 10 }, (_, i) => `L${21 + i}`);
const K = Array.from({ length: 20 }, (_, i) => `K${String(i + 1).padStart(2, "0")}`);

function catalogueFor(database: string): ScopeCatalogue {
  const cats = database === "infoshotpp" ? L : [...L, ...K];
  return {
    retrieval_database: database as ScopeCatalogue["retrieval_database"],
    categories: cats.map((category) => ({
      category,
      label_vi: category === "L26" ? "Nấu ăn — HTV Online" : `Chương trình ${category}`,
      open_subject: category.startsWith("K") || ["L21", "L22", "L30"].includes(category),
    })),
    groups: [
      { id: "L", label_vi: "L21–L30 · chuyên đề (InfoShot++)", categories: cats.filter((c) => c[0] === "L") },
      ...(database === "infoshotpp"
        ? []
        : [{ id: "K", label_vi: "K01–K20 · Thời sự 60 giây (BTC)", categories: K }]),
    ],
    topics: [{ topic_id: "cooking", label_vi: "Nấu ăn", categories: ["L26"] }],
  };
}

/** The auto scope the backend resolves for a cooking query: L26 narrowed on the
 *  speciality side, with the open-subject folders kept. */
const AUTO_SCOPE = {
  mode: "auto" as const,
  categories: ["L21", "L22", "L26", "L30", ...K],
  strict_categories: ["L26"],
  active: true,
  reason_vi: "Heuristic chủ đề: Nấu ăn (giữ lại thư mục thời sự + L30).",
  matched_topics: [
    { topic_id: "cooking", label_vi: "Nấu ăn", keywords: ["nấu ăn"], categories: ["L26"] },
  ],
};

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
  channels: { image_pe: { enabled: true, weight: 1 }, ocr: { enabled: false }, speech: { enabled: false }, audio: { enabled: false } },
  filters: { must_not_include: [] },
  trake: { enabled: false, events: [] },
  qa: { enabled: false },
  ui_hints: {},
  _engine: "heuristic",
};

let searchRequests: any[] = [];
let scopeRequests: string[] = [];

function mockFetch() {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const path = String(url);
    const json = (body: unknown, status = 200) =>
      ({ ok: status < 400, status, json: async () => body } as Response);

    if (path.includes("/api/search/scope")) {
      scopeRequests.push(path);
      const database = path.includes("infoshotpp") ? "infoshotpp" : "btc";
      return json(catalogueFor(database));
    }
    if (path.includes("/api/health"))
      return json({ ok: true, mode: "mock", retrieval_database: "btc", services: {}, capabilities: {}, warnings: [] });
    if (path.endsWith("/api/config"))
      return json({ configured: true, mock_mode: true, missing_required: [], env_path: "/config/.env", env_exists: true, groups: [] });
    if (path.endsWith("/api/dres/status")) return json({ enabled: false, configured: false, connected: false });
    if (path.includes("/api/submit/history")) return json({ history: [] });
    if (path.endsWith("/api/search")) {
      const body = JSON.parse((init?.body as string) || "{}");
      searchRequests.push(body);
      return json({
        query: body.query,
        parsed: PARSED,
        // Echo an auto scope only when the request asked for one, exactly as the
        // backend does — a manual request gets its own selection back.
        scope:
          body.scope?.mode === "auto"
            ? AUTO_SCOPE
            : body.scope?.mode === "manual"
            ? {
                mode: "manual",
                categories: body.scope.categories,
                strict_categories: body.scope.categories,
                active: body.scope.categories.length > 0,
                reason_vi: "Thủ công",
                matched_topics: [],
              }
            : { mode: "all", categories: [], strict_categories: [], active: false, reason_vi: "", matched_topics: [] },
        groups: [{
          video_id: "L26_V001", video_score: 1, max_score: 0.9, mean_top_score: 0.9,
          frame_count: 1, timestamp_dispersion: 0, ambiguous: false, channels: ["image_pe"],
          video_url: FRAME.video_url, frames: [FRAME],
        }],
        latency_ms: { parse_ms: 1, fusion_ms: 0.2, total_ms: 5 },
        mode: "mock",
      });
    }
    return json({}, 404);
  });
}

/** Type a query and search, then wait for the request to land. */
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
  globalThis.localStorage?.clear?.();
  vi.stubGlobal("fetch", mockFetch());
});
afterEach(() => {
  vi.unstubAllGlobals();
  searchRequests = [];
  scopeRequests = [];
});

describe("search scope filter", () => {
  it("defaults to the topic heuristic, which is the whole profile until a topic matches", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("scope-toggle");

    const body = await search(user, "một người mặc áo trắng");

    expect(body.scope).toEqual({ mode: "auto", categories: [] });
  });

  it("lists every folder of the profile with its programme name", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByTestId("scope-toggle"));

    const menu = screen.getByTestId("scope-menu");
    expect(within(menu).getByTestId("scope-cat-L26")).toBeInTheDocument();
    expect(within(menu).getByTestId("scope-cat-K20")).toBeInTheDocument();
    expect(within(menu).getByTitle("Nấu ăn — HTV Online")).toBeInTheDocument();
  });

  it("ticking folders switches to manual and searches only those", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByTestId("scope-toggle"));
    await user.click(screen.getByTestId("scope-clear"));
    await user.click(screen.getByTestId("scope-cat-L26"));
    await user.click(screen.getByTestId("scope-cat-L23"));

    expect(screen.getByTestId("scope-mode-manual")).toHaveAttribute("aria-selected", "true");

    const body = await search(user, "đầu bếp");

    // Ordered by the catalogue, not by the order they were clicked.
    expect(body.scope).toEqual({ mode: "manual", categories: ["L23", "L26"] });
  });

  it("a whole group can be ticked at once", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByTestId("scope-toggle"));
    await user.click(screen.getByTestId("scope-clear"));
    await user.click(screen.getByTestId("scope-group-K"));

    const body = await search(user, "bản tin");

    expect(body.scope.categories).toEqual(K);
  });

  it("reports the folders the auto heuristic actually picked", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("scope-toggle");
    await search(user, "cảnh đầu bếp nấu ăn");

    // The button names the topic, so the operator can see the scope narrowed
    // without opening the menu.
    expect(screen.getByTestId("scope-toggle")).toHaveTextContent("Nấu ăn");
    await user.click(screen.getByTestId("scope-toggle"));
    expect(screen.getByTestId("scope-reason")).toHaveTextContent("Heuristic chủ đề: Nấu ăn");
  });

  it("offers the strict topic folders as one click", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("scope-toggle");
    await search(user, "cảnh đầu bếp nấu ăn");
    await user.click(screen.getByTestId("scope-toggle"));
    await user.click(screen.getByTestId("scope-strict"));

    const body = await search(user, "cảnh đầu bếp nấu ăn");

    expect(body.scope).toEqual({ mode: "manual", categories: ["L26"] });
  });

  it("drops folders the other profile does not hold when switching", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByTestId("scope-toggle"));
    await user.click(screen.getByTestId("scope-clear"));
    await user.click(screen.getByTestId("scope-cat-L26"));
    await user.click(screen.getByTestId("scope-cat-K01"));

    // InfoShot++ has no K-side, so keeping K01 would filter every search to zero.
    fireEvent.change(screen.getAllByTestId("retrieval-database")[0], {
      target: { value: "infoshotpp" },
    });
    await waitFor(() => expect(scopeRequests.some((p) => p.includes("infoshotpp"))).toBe(true));

    const body = await search(user, "đầu bếp");

    expect(body.scope).toEqual({ mode: "manual", categories: ["L26"] });
  });

  it("does not re-run the search while the operator is ticking boxes", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("scope-toggle");
    await search(user, "đầu bếp");
    const after = searchRequests.length;

    await user.click(screen.getByTestId("scope-toggle"));
    await user.click(screen.getByTestId("scope-cat-L23"));
    await user.click(screen.getByTestId("scope-cat-L24"));

    expect(searchRequests).toHaveLength(after);
  });
});

describe("scope helpers", () => {
  const catalogue = catalogueFor("btc");

  it("only sends folders in manual mode", () => {
    expect(scopeRequest("manual", ["L26"])).toEqual({ mode: "manual", categories: ["L26"] });
    // A stale list must not ride along on a mode that ignores it.
    expect(scopeRequest("auto", ["L26"])).toEqual({ mode: "auto", categories: [] });
    expect(scopeRequest("all", ["L26"])).toEqual({ mode: "all", categories: [] });
  });

  it("keeps the catalogue's order when toggling", () => {
    let selected = toggleCategory([], "K01", catalogue);
    selected = toggleCategory(selected, "L26", catalogue);
    expect(selected).toEqual(["L26", "K01"]);
    expect(toggleCategory(selected, "K01", catalogue)).toEqual(["L26"]);
  });

  it("labels the button by what will actually be searched", () => {
    expect(scopeButtonLabel("all", [], catalogue, null)).toBe("Tất cả (30)");
    expect(scopeButtonLabel("manual", ["L26", "L23"], catalogue, null)).toBe("L26, L23");
    expect(scopeButtonLabel("manual", L, catalogue, null)).toBe("10 thư mục");
    // A full manual selection is the same as no filter, and says so.
    expect(scopeButtonLabel("manual", [...L, ...K], catalogue, null)).toBe("Tất cả (30)");
    expect(scopeButtonLabel("auto", [], catalogue, null)).toBe("Tự động");
    expect(scopeButtonLabel("auto", [], catalogue, AUTO_SCOPE)).toBe("Tự động · Nấu ăn");
  });
});
