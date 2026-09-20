import { useState } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { ImageEmbeddingModel, QueryType, RetrievalDatabase } from "../api/types";
import { ChannelBadge } from "../components/Badges";
import { ImageModelSelector } from "../components/ImageModelSelector";
import { CHANNEL_LABEL } from "../lib/constants";
import { imageModelsForSearch } from "../lib/imageModels";

type RecordedRequest = { path: string; body: Record<string, unknown> };
let requests: RecordedRequest[] = [];
/** Lets one test stage a degraded simple search (a dead encoder) without a second harness. */
let simpleSearchExtras: Record<string, unknown> = {};
let healthCapabilities: Record<string, boolean> = {};

function parsed(queryType: QueryType) {
  return {
    query_type: queryType,
    confidence: 1,
    original_query: "street scene",
    normalized_vi: "street scene",
    translated_en_visual: "street scene",
    operator_summary_vi: "",
    channels: {
      image_pe: { enabled: true, weight: 1 },
      image_qwen: { enabled: true, weight: 1 },
      ocr: { enabled: false, weight: 0 },
      speech: { enabled: false, weight: 0 },
      audio: { enabled: false, weight: 0 },
    },
    filters: { video_ids: [], categories: [], must_include: [], must_not_include: [] },
    trake: { enabled: queryType === "TRAKE", events: [], ordering_rule: "", fallback_policy: "" },
    qa: { enabled: queryType === "QA", question_vi: "", draft_answer_allowed_from_text_evidence_only: false },
    ui_hints: {
      show_timeline: false,
      show_ocr_snippets: false,
      show_speech_snippets: false,
      show_audio_badges: false,
      warning_vi: "",
    },
    _engine: "test",
  };
}

function mockFetch() {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const path = String(url);
    const json = (body: unknown, status = 200) =>
      ({ ok: status < 400, status, json: async () => body } as Response);
    const body = init?.body ? JSON.parse(String(init.body)) : {};

    if (path.endsWith("/api/config")) {
      return json({ configured: true, mock_mode: true, missing_required: [], env_path: "/tmp/.env", env_exists: true, groups: [] });
    }
    if (path.includes("/api/health")) {
      return json({
        retrieval_database: path.includes("infoshotpp") ? "infoshotpp" : "btc",
        ok: true,
        mode: "mock",
        services: {},
        capabilities: healthCapabilities,
        warnings: [],
      });
    }
    if (path.includes("/api/search/scope")) {
      return json({
        retrieval_database: path.includes("infoshotpp") ? "infoshotpp" : "btc",
        categories: [],
        groups: [],
        topics: [],
      });
    }
    if (path.endsWith("/api/dres/status")) {
      return json({ enabled: false, configured: false, connected: false });
    }
    if (path.includes("/api/submit/history")) return json({ history: [] });

    if (path.endsWith("/api/search/simple")) {
      requests.push({ path, body });
      const models = (body.image_models as ImageEmbeddingModel[]) ?? ["pe"];
      return json({
        retrieval_database: body.retrieval_database,
        query: body.query,
        image_models: models,
        results: [
          {
            image_id: "L21/L21_V001/001",
            submit_keyframe_id: "L21/L21_V001/001",
            video_id: "L21_V001",
            keyframe_n: 1,
            score: 0.0328,
            models,
            per_model_score: Object.fromEntries(models.map((model) => [model, 0.5])),
            keyframe_url: "https://media.test/a.jpg",
            video_url: "https://media.test/a.mp4",
          },
        ],
        mode: "mock",
        latency_ms: 1,
        ...simpleSearchExtras,
      });
    }
    if (path.endsWith("/api/search/trake")) {
      requests.push({ path, body });
      return json({
        retrieval_database: body.retrieval_database,
        query: body.query,
        parsed: parsed("TRAKE"),
        events: [],
        sequences: [],
        mode: "mock",
      });
    }
    if (path.endsWith("/api/search")) {
      requests.push({ path, body });
      return json({
        retrieval_database: body.retrieval_database,
        query: body.query,
        parsed: parsed((body.query_type_hint as QueryType) || "T-KIS"),
        groups: [],
        latency_ms: { parse_ms: 0, fusion_ms: 0, total_ms: 1 },
        mode: "mock",
      });
    }
    return json({}, 404);
  });
}

function SelectorHarness({
  database = "infoshotpp",
  initial = ["pe"],
}: {
  database?: RetrievalDatabase;
  initial?: ImageEmbeddingModel[];
}) {
  const [models, setModels] = useState<ImageEmbeddingModel[]>(initial);
  return (
    <ImageModelSelector
      retrievalDatabase={database}
      value={models}
      onChange={setModels}
    />
  );
}

function requestBodies(path: string) {
  return requests.filter((request) => request.path.endsWith(path)).map((request) => request.body);
}

async function submitSearch(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByTestId("search-btn"));
  await waitFor(() => expect(requests.length).toBeGreaterThan(0));
}

beforeEach(() => {
  globalThis.localStorage?.clear?.();
  requests = [];
  simpleSearchExtras = {};
  healthCapabilities = {};
  vi.stubGlobal("fetch", mockFetch());
});

afterEach(() => {
  vi.unstubAllGlobals();
  requests = [];
  simpleSearchExtras = {};
  healthCapabilities = {};
});

async function openSimpleSearchOnInfoshotpp(user: ReturnType<typeof userEvent.setup>) {
  render(<App />);
  await user.click(screen.getByText("Simple ⤴"));
  await user.selectOptions(screen.getByTestId("retrieval-database"), "infoshotpp");
  await screen.findByTestId("image-model-selector");
}

describe("InfoShot++ image model selector", () => {
  it("supports PE-only, both/RRF and Qwen-only while preventing an empty selection", async () => {
    const user = userEvent.setup();
    render(<SelectorHarness />);

    const pe = screen.getByTestId("image-model-pe") as HTMLInputElement;
    const qwen = screen.getByTestId("image-model-qwen3_vl") as HTMLInputElement;
    expect(pe).toBeChecked();
    expect(pe).toBeDisabled();
    expect(qwen).not.toBeChecked();
    expect(screen.queryByTestId("image-model-fusion")).not.toBeInTheDocument();

    await user.click(qwen);
    expect(pe).toBeChecked();
    expect(pe).not.toBeDisabled();
    expect(qwen).toBeChecked();
    expect(screen.getByTestId("image-model-fusion")).toHaveTextContent("RRF");

    await user.click(pe);
    expect(pe).not.toBeChecked();
    expect(qwen).toBeChecked();
    expect(qwen).toBeDisabled();
    expect(screen.queryByTestId("image-model-fusion")).not.toBeInTheDocument();
  });

  it("is hidden for BTC and normalizes every BTC request to PE", () => {
    render(<SelectorHarness database="btc" initial={["qwen3_vl"]} />);
    expect(screen.queryByTestId("image-model-selector")).not.toBeInTheDocument();
    expect(imageModelsForSearch("btc", ["qwen3_vl"])).toEqual(["pe"]);
    expect(imageModelsForSearch("infoshotpp", [])).toEqual(["pe"]);
  });

  it("renders the backend's Qwen evidence channel distinctly", () => {
    render(<ChannelBadge channel="image_qwen" />);
    expect(screen.getByText("Qwen3-VL")).toHaveClass("image_qwen");
  });

  it("shows TARA beside the keyframe models and writes its channel override", async () => {
    const user = userEvent.setup();
    healthCapabilities = { tara_clip_search: true };
    render(<App />);
    await user.selectOptions(screen.getByTestId("retrieval-database"), "infoshotpp");

    const tara = await screen.findByTestId("model-tara") as HTMLInputElement;
    expect(tara).toBeChecked();
    expect(tara).not.toBeDisabled();
    expect(screen.getByTestId("tara-video-fusion")).toHaveTextContent("video RRF");

    await user.click(tara);
    expect(tara).not.toBeChecked();
    await user.type(screen.getByTestId("query-input"), "street scene");
    await submitSearch(user);
    expect(requestBodies("/api/search").at(-1)?.manual_overrides).toMatchObject({
      disable_channels: ["tara"],
    });
  });

  it("names the visual switch for what it gates, not for one of the two indices", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByTestId("retrieval-database"), "infoshotpp");
    await screen.findByTestId("image-model-selector");

    // The `image_pe` switch enables BOTH image indices (the backend reads
    // channels.image_pe.enabled for Qwen too), so it must not be labelled
    // "PE Core" — that collides with the embedding picker's own PE box and
    // reads as "PE only", making it look safe to switch off mid-Qwen-search.
    const toggle = screen.getByTestId("channel-image_pe");
    expect(toggle).toHaveTextContent("VISUAL");
    expect(toggle).not.toHaveTextContent("PE Core");
    expect(toggle.getAttribute("title")).toMatch(/PE Core.*Qwen3-VL/);

    // The evidence badge on a result frame still names the index that found it.
    expect(CHANNEL_LABEL.image_pe).toBe("PE Core");
    expect(CHANNEL_LABEL.image_qwen).toBe("Qwen3-VL");
  });
});

describe("image_models search payloads", () => {
  it("sends PE-only, both/RRF, and Qwen-only from the full console", async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(screen.queryByTestId("image-model-selector")).not.toBeInTheDocument();

    await user.selectOptions(screen.getByTestId("retrieval-database"), "infoshotpp");
    await screen.findByTestId("image-model-selector");
    await user.type(screen.getByTestId("query-input"), "street scene");

    await submitSearch(user);
    expect(requestBodies("/api/search").at(-1)?.image_models).toEqual(["pe"]);

    requests = [];
    await user.click(screen.getByTestId("image-model-qwen3_vl"));
    expect(screen.getByTestId("image-model-fusion")).toBeInTheDocument();
    await submitSearch(user);
    expect(requestBodies("/api/search").at(-1)?.image_models).toEqual(["pe", "qwen3_vl"]);

    requests = [];
    await user.click(screen.getByTestId("image-model-pe"));
    await submitSearch(user);
    expect(requestBodies("/api/search").at(-1)?.image_models).toEqual(["qwen3_vl"]);
  });

  it("keeps BTC hidden and PE-only in the full-console request", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "street scene");
    await submitSearch(user);

    expect(screen.queryByTestId("image-model-selector")).not.toBeInTheDocument();
    expect(requestBodies("/api/search").at(-1)?.image_models).toEqual(["pe"]);
  });

  it("passes the selection through TRAKE", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByTestId("retrieval-database"), "infoshotpp");
    await user.click(await screen.findByTestId("image-model-qwen3_vl"));
    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), "event one then event two");
    await submitSearch(user);

    expect(requestBodies("/api/search/trake").at(-1)?.image_models).toEqual(["pe", "qwen3_vl"]);
  });

  it("passes the selection through Simple Search", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByText("Simple ⤴"));
    await user.selectOptions(screen.getByTestId("retrieval-database"), "infoshotpp");
    await user.click(await screen.findByTestId("image-model-qwen3_vl"));
    await user.type(screen.getByTestId("query-input"), "street scene");
    await submitSearch(user);

    expect(requestBodies("/api/search/simple").at(-1)?.image_models).toEqual(["pe", "qwen3_vl"]);
  });
});

describe("Simple Search reports which index answered", () => {
  it("attributes each frame once both indices are searched", async () => {
    const user = userEvent.setup();
    await openSimpleSearchOnInfoshotpp(user);
    await user.type(screen.getByTestId("query-input"), "street scene");

    // One index: no attribution badges, because there is nothing to distinguish.
    await submitSearch(user);
    await waitFor(() => expect(screen.getAllByTestId("kf-card")).toHaveLength(1));
    expect(screen.queryByText("Qwen")).not.toBeInTheDocument();

    requests = [];
    await user.click(screen.getByTestId("image-model-qwen3_vl"));
    await submitSearch(user);

    await waitFor(() => expect(screen.getByText("Qwen")).toBeInTheDocument());
    expect(screen.getByText("PE")).toBeInTheDocument();
  });

  it("shows the operator that a dead encoder degraded the search", async () => {
    const user = userEvent.setup();
    simpleSearchExtras = {
      image_models: ["pe"],
      warnings: ["qwen3_vl image search unavailable: Qwen3-VL encoder did not respond after 120s"],
    };
    await openSimpleSearchOnInfoshotpp(user);
    await user.click(screen.getByTestId("image-model-qwen3_vl"));
    await user.type(screen.getByTestId("query-input"), "street scene");
    await submitSearch(user);

    const warning = await screen.findByTestId("search-warning");
    expect(warning).toHaveTextContent("qwen3_vl image search unavailable");
    // Degraded, not empty: the surviving index still ranks, and says so.
    expect(screen.getAllByTestId("kf-card")).toHaveLength(1);
    expect(screen.queryByText("Qwen")).not.toBeInTheDocument();
  });
});
