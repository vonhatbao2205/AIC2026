import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";

const FRAME1 = {
  image_id: "K01/K01_V001/001",
  submit_keyframe_id: "K01/K01_V001/001",
  video_id: "K01_V001",
  keyframe_n: 1,
  frame_idx: 0,
  fps: 25,
  pts_time: 0,
  score: 0.9,
  channels: ["image_pe", "ocr"],
  per_channel_score: { image_pe: 0.9, ocr: 5 },
  keyframe_url: "https://media.test/Keyframes/Keyframes_K01/K01_V001/001.jpg",
  video_url: "https://media.test/Videos/Videos_K01/K01_V001.mp4",
  evidence: [{ type: "ocr", score: 5, text: "Bản tin thời sự HTV7" }],
};
const FRAME2 = { ...FRAME1, submit_keyframe_id: "K01/K01_V001/006", image_id: "K01/K01_V001/006", keyframe_n: 6, frame_idx: 450, pts_time: 18 };

const SEARCH_RESPONSE = {
  query: "thời sự",
  parsed: {
    query_type: "T-KIS",
    confidence: 0.4,
    translated_en_visual: "news broadcast",
    channels: { image_pe: { enabled: true, weight: 1 }, ocr: { enabled: true, weight: 0.8 }, speech: { enabled: false }, audio: { enabled: false } },
    filters: { must_not_include: [] },
    trake: { enabled: false, events: [] },
    qa: { enabled: false },
    ui_hints: {},
    _engine: "heuristic",
  },
  groups: [
    {
      video_id: "K01_V001",
      video_score: 1.23,
      max_score: 0.9,
      mean_top_score: 0.5,
      frame_count: 2,
      timestamp_dispersion: 1,
      ambiguous: false,
      channels: ["image_pe", "ocr"],
      video_url: FRAME1.video_url,
      frames: [FRAME1, FRAME2],
    },
  ],
  latency_ms: { parse_ms: 1, fusion_ms: 0.2, total_ms: 5 },
  mode: "mock",
};

const TIMELINE = {
  video_id: "K01_V001",
  fps: 25,
  duration: 30,
  video_url: FRAME1.video_url,
  keyframes: Array.from({ length: 40 }, (_, i) => ({
    submit_keyframe_id: `K01/K01_V001/${String(i + 1).padStart(3, "0")}`,
    keyframe_n: i + 1,
    frame_idx: i * 25,
    pts_time: i,
    keyframe_url: `https://media.test/Keyframes/Keyframes_K01/K01_V001/${String(i + 1).padStart(3, "0")}.jpg`,
  })),
  speech_segments: [],
  ocr_markers: [],
  audio_windows: [],
  heatmap: [],
};

const TRAKE_RESPONSE = {
  query: "trake",
  parsed: {
    query_type: "TRAKE",
    confidence: 0.5,
    translated_en_visual: "a then b",
    channels: { image_pe: { enabled: true, weight: 1 }, ocr: { enabled: false }, speech: { enabled: false }, audio: { enabled: false } },
    filters: { must_not_include: [] },
    trake: { enabled: true, events: [{ event_index: 1 }, { event_index: 2 }] },
    qa: { enabled: false },
    ui_hints: {},
    _engine: "heuristic",
  },
  events: [{ event_index: 1, candidate_count: 5 }, { event_index: 2, candidate_count: 5 }],
  sequences: [
    {
      video_id: "K19_V028",
      score: 0.06,
      complete: true,
      coverage: 2,
      warning: null,
      video_url: "https://media.test/Videos/Videos_K19/K19_V028.mp4",
      frames: [
        { event_index: 1, submit_keyframe_id: "K19/K19_V028/203", keyframe_n: 203, frame_idx: 15926, pts_time: 530, score: 0.03, keyframe_url: "https://media.test/Keyframes/Keyframes_K19/K19_V028/203.jpg" },
        { event_index: 2, submit_keyframe_id: "K19/K19_V028/301", keyframe_n: 301, frame_idx: 21030, pts_time: 842, score: 0.03, keyframe_url: "https://media.test/Keyframes/Keyframes_K19/K19_V028/301.jpg" },
      ],
    },
  ],
  mode: "mock",
};

const QA_ANALYSIS_RESPONSE = {
  question: "Bản tin trên màn hình tên gì?",
  model: "Efficient-Large-Model/NVILA-8B-hf",
  mode: "live",
  answerable: true,
  best_answer: "Bản tin thời sự HTV7",
  best_candidate_id: "C01",
  best_submit_keyframe_id: FRAME1.submit_keyframe_id,
  candidate_answers: [
    {
      answer: "Bản tin thời sự HTV7",
      confidence: 0.92,
      supporting_candidate_ids: ["C01"],
      supporting_frames: [{
        candidate_id: "C01",
        submit_keyframe_id: FRAME1.submit_keyframe_id,
        video_id: FRAME1.video_id,
        frame_idx: FRAME1.frame_idx,
        pts_time: FRAME1.pts_time,
        keyframe_url: FRAME1.keyframe_url,
      }],
      reason: "The title is visible in the candidate frame.",
      source: "hybrid",
      web_sources: [{ title: "HTV", url: "https://www.htv.com.vn/" }],
      visual_verification: {
        answer: "Bản tin thời sự HTV7",
        status: "supported",
        visual_confidence: 0.94,
        supporting_candidate_ids: ["C01"],
        reason: "The programme title remains visually consistent.",
      },
    },
    {
      answer: "HTV7",
      confidence: 0.81,
      supporting_candidate_ids: ["C02"],
      supporting_frames: [{
        candidate_id: "C02",
        submit_keyframe_id: FRAME2.submit_keyframe_id,
        video_id: FRAME2.video_id,
        frame_idx: FRAME2.frame_idx,
        pts_time: FRAME2.pts_time,
        keyframe_url: FRAME2.keyframe_url,
      }],
      reason: "A shorter grounded alternative.",
      source: "web",
      web_sources: [{ title: "HTV", url: "https://www.htv.com.vn/" }],
      visual_verification: {
        answer: "HTV7",
        status: "insufficient",
        visual_confidence: 0.55,
        supporting_candidate_ids: ["C02"],
        reason: "The shorter form is plausible but not fully visible.",
      },
    },
  ],
  hotspots: [{
    candidate_id: "C01",
    submit_keyframe_id: FRAME1.submit_keyframe_id,
    video_id: FRAME1.video_id,
    keyframe_n: FRAME1.keyframe_n,
    frame_idx: FRAME1.frame_idx,
    pts_time: FRAME1.pts_time,
    keyframe_url: FRAME1.keyframe_url,
    relevance: 0.95,
    answer_support: "Visible program title",
  }],
  uncertainty: "Verify the OCR spelling.",
  candidate_count: 2,
  latency_ms: 1234,
  total_latency_ms: 1880,
  cached: false,
  warnings: [],
  web_grounding: {
    requested: "auto",
    available: true,
    attempted: true,
    used: true,
    model: "deepseek-v4-flash",
    queries: ["HTV7 news programme"],
    sources: [{ title: "HTV", url: "https://www.htv.com.vn/" }],
    summary: "Web-grounded entity resolution.",
    latency_ms: 236,
    search_suggestions_html: "",
    visual_verification: {
      attempted: true,
      used: true,
      model: "Efficient-Large-Model/NVILA-8B-hf",
      latency_ms: 410,
      verdicts: [],
      rejected_answers: [],
      uncertainty: "Verify the exact broadcast title manually.",
    },
  },
};

const CANVAS_PALETTE = {
  colors: [
    { name: "red", hex: "#dc1414" },
    { name: "blue", hex: "#1e50dc" },
  ],
  labels: [
    { label: "car", label_vi: "ô tô", colorable: true },
    { label: "person", label_vi: "người", colorable: true },
    { label: "crowd", label_vi: "đám đông", colorable: false },
  ],
  modes: ["rough", "precise"],
  color_palette_version: "aic16-lab-v1",
};

const CANVAS_FRAME = {
  ...FRAME1,
  channels: ["object_layout", "image_pe"],
  per_channel_score: { object_layout: 0.82, image_pe: 0.6 },
  evidence: [
    {
      type: "object_layout",
      score: 0.82,
      text: "car · red @bottom_left",
      matches: [
        {
          object_id: "q1",
          label: "car",
          score: 0.86,
          detection_label: "car",
          conf: 0.91,
          bbox_norm: { x1: 0.05, y1: 0.5, x2: 0.35, y2: 0.86 },
          position: "bottom_left",
          dominant_color: "red",
          color_reliable: true,
          color_ok: true,
        },
      ],
      missing: ["q2"],
      coverage: 0.5,
      excluded_hits: [],
    },
  ],
};

const CANVAS_RESPONSE = {
  canvas: {
    mode: "rough",
    objects: [{ id: "q1", label: "car", bbox: [0.05, 0.52, 0.35, 0.86], color: "red", required: true }],
    exclude_labels: [],
    action_text: "",
    queries_en: ["A video frame showing a red car in the lower left."],
  },
  groups: [{ ...SEARCH_RESPONSE.groups[0], channels: ["object_layout"], frames: [CANVAS_FRAME] }],
  latency_ms: { fusion_ms: 0.3, total_ms: 12 },
  warnings: [],
  mode: "mock",
};

// ---- DRES (official evaluation server) ------------------------------------
// One run per task type, exactly like the BTC server exposes them.
const DRES_STATUS = {
  configured: true,
  base_url: "http://dres.test",
  logged_in: true,
  username: "fourier1",
  user: { id: "u1", username: "fourier1", role: "PARTICIPANT" },
  pinned_evaluation_id: null,
  segment_pad_ms: 500,
  error: null,
};

const DRES_EVALUATIONS = [
  {
    id: "e-tkis", name: "tkis", status: "ACTIVE", type: "SYNCHRONOUS", teams: ["Fourier"],
    task_templates: [{ name: "tkis-00", taskGroup: "T-KIS Group", taskType: "Textual KIS", duration: null }],
    current_task: { name: "tkis-00", taskGroup: "T-KIS Group", taskType: "Textual KIS", duration: null },
    state: { evaluationStatus: "ACTIVE", taskStatus: "RUNNING", taskTemplateId: "t-tkis", timeLeft: 300, timeElapsed: 42 },
  },
  {
    id: "e-qa", name: "qa", status: "ACTIVE", type: "SYNCHRONOUS", teams: ["Fourier"],
    task_templates: [{ name: "qa-00", taskGroup: "QA Group", taskType: "Question Answering", duration: null }],
    current_task: { name: "qa-00", taskGroup: "QA Group", taskType: "Question Answering", duration: null },
    state: { evaluationStatus: "ACTIVE", taskStatus: "RUNNING", taskTemplateId: "t-qa", timeLeft: 240, timeElapsed: 60 },
  },
];

const TKIS_SCOPE = "e-tkis/tkis-00";

const TKIS_STATEMENT = "Đoạn phim bắt đầu bằng hình ảnh một chiếc máy ảnh đặt trên quyển sách.";
const QA_STATEMENT = "Người phụ nữ áo tím lấy một cuốn sách tranh. Hỏi tác giả cuốn sách đó là ai?";

function taskHintFor(path: string) {
  const isQa = path.includes("query_type=QA");
  const text = isQa ? QA_STATEMENT : TKIS_STATEMENT;
  return {
    evaluation_id: isQa ? "e-qa" : "e-tkis",
    task_template_id: isQa ? "t-qa" : "t-tkis",
    task_name: isQa ? "qa-00" : "tkis-00",
    task_type: isQa ? "Question Answering" : "Textual KIS",
    task_status: "RUNNING",
    text,
    elements: [{ content_type: "TEXT", content: text, offset: 0 }],
    loop: false,
    warnings: [],
  };
}

/** Mirrors the backend's answer building, so the tests assert the real shape. */
function dresAnswers(body: any) {
  const p = body.payload ?? {};
  const mode = !body.answer_mode || body.answer_mode === "auto"
    ? (body.query_type === "QA" ? "temporal_text" : "temporal")
    : body.answer_mode;
  if (mode === "text") return [{ text: p.answer ?? "" }];
  if (mode === "item") return [{ mediaItemName: p.video_id }];
  const pad = body.segment_pad_ms ?? 0;
  const ms = (pts?: number | null, frameIdx?: number | null) =>
    Math.round((pts != null ? pts : (frameIdx ?? 0) / (p.fps || 25)) * 1000);
  const window = (center: number) => ({ start: Math.max(0, center - pad), end: center + pad });
  if (p.events?.length) {
    return p.events.map((e: any) => ({ mediaItemName: p.video_id, ...window(ms(e.pts_time, e.frame_idx)) }));
  }
  const segment = { mediaItemName: p.video_id, ...window(ms(p.timestamp, p.frame_idx)) };
  // QA: the segment AND the answer travel in one answer object.
  return [mode === "temporal_text" ? { ...segment, text: p.answer ?? "" } : segment];
}

/** The backend refuses a body DRES could not judge; the mock does the same. */
function formatError(body: any): string | null {
  const p = body.payload ?? {};
  const mode = !body.answer_mode || body.answer_mode === "auto"
    ? (body.query_type === "QA" ? "temporal_text" : "temporal")
    : body.answer_mode;
  if (mode === "text" || mode === "temporal_text") {
    if (!(p.answer ?? "").trim()) return "Task QA cần `text` (đáp án) — ô answer đang trống.";
  }
  if (mode !== "text" && !p.video_id) return "Thiếu `mediaItemName` (video_id) cho task KIS/QA/TRAKE.";
  return null;
}

function dedupKey(body: any) {
  const p = body.payload ?? {};
  if (p.events?.length) return `${p.video_id}|${p.events.map((e: any) => e.frame_idx).join(",")}`;
  const frameKey = `${p.video_id}:${p.frame_idx}`;
  if (body.query_type === "QA") return `${frameKey}|text:${(p.answer ?? "").trim().toLowerCase()}`;
  return frameKey;
}

/** QA sent as a bare segment drops the answer — the backend flags it. */
function mismatched(body: any) {
  return body.query_type === "QA" && (body.answer_mode === "temporal" || body.answer_mode === "item");
}

function taskScope(body: any) {
  const evaluation = DRES_EVALUATIONS.find((e) => e.id === body.evaluation_id);
  const name = body.task_name || evaluation?.current_task?.name;
  return name ? `${body.evaluation_id}/${name}` : "unknown-task";
}

let historyStore: any[] = [];
let canvasRequests: any[] = [];

function mockFetch(historySeed: any[] = []) {
  historyStore = historySeed;
  return vi.fn(async (url: string, init?: RequestInit) => {
    const path = String(url);
    const json = (body: unknown, status = 200) =>
      ({ ok: status < 400, status, json: async () => body } as Response);

    if (path.includes("/api/health"))
      return json({ ok: true, mode: "mock", retrieval_database: path.includes("infoshotpp") ? "infoshotpp" : "btc", services: {}, capabilities: { qa_nvila: true, qa_web_grounding: true, qa_visual_verification: true, canvas_object_search: true }, warnings: [] });
    if (path.endsWith("/api/config"))
      return json({ configured: true, mock_mode: true, missing_required: [], env_path: "/config/.env", env_exists: true, groups: [] });
    if (path.endsWith("/api/dres/status")) return json(DRES_STATUS);
    if (path.includes("/api/dres/task-hint")) return json(taskHintFor(path));
    if (path.includes("/api/dres/evaluations"))
      return json({ evaluations: DRES_EVALUATIONS, server_time: 1, pinned_evaluation_id: null });
    if (path.endsWith("/api/submit/preview")) {
      const body = JSON.parse((init?.body as string) || "{}");
      const invalid = formatError(body);
      if (invalid) return json({ detail: { error: "invalid_format", message: invalid } }, 400);
      const scope = taskScope(body);
      const key = dedupKey(body);
      const evaluation = DRES_EVALUATIONS.find((e) => e.id === body.evaluation_id);
      const taskName = body.task_name || evaluation?.current_task?.name || null;
      return json({
        evaluation_id: body.evaluation_id ?? null,
        task_name: taskName,
        task_id: scope,
        answer_mode: body.query_type === "QA" ? "temporal_text" : "temporal",
        answer_mode_mismatch: mismatched(body),
        task_type: body.query_type === "QA" ? "Question Answering" : "Textual KIS",
        segment_pad_ms: body.segment_pad_ms ?? 500,
        body: { answerSets: [{ ...(taskName ? { taskName } : {}), answers: dresAnswers(body) }] },
        duplicate_key: historyStore.some((h) => h.task_id === scope && (h.dedup_keys || []).includes(key))
          ? key
          : null,
        url: `http://dres.test/api/v2/submit/${body.evaluation_id}`,
        warnings: mismatched(body)
          ? ["Task DRES là “Question Answering” (cần `text`) nhưng payload đang gửi “temporal” — đáp án bạn gõ SẼ BỊ BỎ."]
          : [],
      });
    }
    if (path.endsWith("/api/canvas/palette")) return json(CANVAS_PALETTE);
    if (path.endsWith("/api/search/canvas")) {
      canvasRequests.push(JSON.parse((init?.body as string) || "{}"));
      return json(CANVAS_RESPONSE);
    }
    if (path.includes("/api/submit/history")) {
      if (init?.method === "DELETE") {
        const params = new URL(path, "http://x").searchParams;
        const taskId = params.get("task_id");
        const ids = params.get("ids")?.split(",").filter(Boolean);
        const before = historyStore.length;
        historyStore = ids
          ? historyStore.filter((h) => !ids.includes(h.id))
          : taskId
          ? historyStore.filter((h) => h.task_id !== taskId)
          : [];
        return json({ deleted: before - historyStore.length, remaining: historyStore.length, backup: "h.1.bak.json" });
      }
      return json({ history: historyStore });
    }
    if (path.endsWith("/api/search/trake")) return json(TRAKE_RESPONSE);
    if (path.endsWith("/api/search")) return json(SEARCH_RESPONSE);
    if (path.endsWith("/api/query/parse")) return json(SEARCH_RESPONSE.parsed);
    if (path.endsWith("/api/qa/analyze")) return json(QA_ANALYSIS_RESPONSE);
    if (path.includes("/timeline")) return json(TIMELINE);
    if (path.endsWith("/api/submit")) {
      const body = JSON.parse((init?.body as string) || "{}");
      const invalid = formatError(body);
      if (invalid) return json({ detail: { error: "invalid_format", message: invalid } }, 400);
      const key = dedupKey(body);
      const scope = taskScope(body);
      if (historyStore.some((h) => h.task_id === scope && (h.dedup_keys || []).includes(key)) && !body.allow_duplicate) {
        return json({ detail: { error: "duplicate_submit", submit_keyframe_id: key } }, 409);
      }
      const entry = {
        id: "x", ts: Date.now() / 1000, task_id: scope, evaluation_id: body.evaluation_id,
        task_name: taskScope(body).split("/").pop(), query_type: body.query_type, payload: body.payload,
        answer_sets: [{ answers: dresAnswers(body) }], dedup_keys: [key],
        status: "dres_ok", verdict: "CORRECT", was_duplicate: false,
      };
      historyStore.push(entry);
      return json(entry);
    }
    return json({}, 404);
  });
}

beforeEach(() => {
  // App defaults to the full console view when no saved preference exists.
  globalThis.localStorage?.clear?.();
  vi.stubGlobal("fetch", mockFetch());
});
afterEach(() => {
  vi.unstubAllGlobals();
  historyStore = [];
  canvasRequests = [];
});

describe("V-KIS canvas", () => {
  async function openCanvas(user: ReturnType<typeof userEvent.setup>) {
    render(<App />);
    await user.click(screen.getByRole("tab", { name: "V-KIS" }));
    return screen.findByTestId("canvas-panel");
  }

  it("is offered for V-KIS only", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(screen.queryByTestId("canvas-panel")).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "V-KIS" }));

    expect(await screen.findByTestId("canvas-panel")).toBeInTheDocument();
  });

  it("sends the drawn objects as structured JSON, not only a picture", async () => {
    const user = userEvent.setup();
    await openCanvas(user);

    await user.click(await screen.findByTestId("canvas-add-car"));
    await user.click(screen.getByTestId("canvas-search"));

    await waitFor(() => expect(canvasRequests).toHaveLength(1));
    const [object] = canvasRequests[0].canvas.objects;
    expect(object.label).toBe("car");
    expect(object.bbox).toHaveLength(4);
    expect(object.required).toBe(true);
    expect(canvasRequests[0].canvas.mode).toBe("rough");
  });

  it("omits the raster entirely when the PE image channel is switched off", async () => {
    const user = userEvent.setup();
    await openCanvas(user);

    await user.click(await screen.findByTestId("canvas-add-car"));
    await user.click(screen.getByTestId("canvas-use-raster"));
    await user.click(screen.getByTestId("canvas-search"));

    await waitFor(() => expect(canvasRequests).toHaveLength(1));
    expect(canvasRequests[0].canvas.image).toBeNull();
    expect(canvasRequests[0].canvas.objects).toHaveLength(1);
  });

  it("switches the matching mode sent to the backend", async () => {
    const user = userEvent.setup();
    await openCanvas(user);

    await user.click(await screen.findByTestId("canvas-add-person"));
    await user.click(screen.getByTestId("canvas-mode-precise"));
    await user.click(screen.getByTestId("canvas-search"));

    await waitFor(() => expect(canvasRequests).toHaveLength(1));
    expect(canvasRequests[0].canvas.mode).toBe("precise");
  });

  it("hides the colour picker for a class OD extracts no colour for", async () => {
    const user = userEvent.setup();
    await openCanvas(user);

    await user.click(await screen.findByTestId("canvas-add-crowd"));
    expect(await screen.findByTestId("canvas-editor")).toBeInTheDocument();
    expect(screen.queryByTestId("canvas-color-red")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("canvas-add-car"));
    expect(await screen.findByTestId("canvas-color-red")).toBeInTheDocument();
  });

  it("shows the generated PE text and draws matched detections over the frame", async () => {
    const user = userEvent.setup();
    await openCanvas(user);

    await user.click(await screen.findByTestId("canvas-add-car"));
    await user.click(screen.getByTestId("canvas-search"));

    expect(await screen.findByTestId("canvas-queries")).toHaveTextContent("red car in the lower left");
    await waitFor(() => expect(screen.getByTestId("detail-panel")).toBeInTheDocument());
    expect(screen.getByTestId("overlay-q1")).toBeInTheDocument();
    expect(screen.getByTestId("canvas-match-summary")).toHaveTextContent("q1 · car");
  });
});

describe("AIC26 retrieval console (full)", () => {
  it("renders the search form", async () => {
    render(<App />);
    expect(screen.getByTestId("query-input")).toBeInTheDocument();
    expect(screen.getByTestId("search-btn")).toBeInTheDocument();
    await waitFor(() => expect(fetch).toHaveBeenCalled());
  });

  it("displays grouped results with thumbnails and evidence", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));

    await waitFor(() => expect(screen.getByTestId("results")).toBeInTheDocument());
    expect(within(screen.getByTestId("results")).getByText("K01_V001")).toBeInTheDocument();

    await waitFor(() => {
      const thumbs = screen.getAllByTestId("frame-thumb");
      expect(thumbs.length).toBe(2);
    });
    const imgs = screen.getAllByRole("img");
    expect(imgs.some((i) => (i as HTMLImageElement).src.includes("/Keyframes_K01/K01_V001/001.jpg"))).toBe(true);

    const detail = screen.getByTestId("detail-panel");
    expect(within(detail).getByTestId("submit-id")).toHaveTextContent("K01/K01_V001/001");
    expect(screen.getByText(/Bản tin thời sự HTV7/)).toBeInTheDocument();
  });

  it("QA: shows the full candidate pack and waits for the operator to choose an answer", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: "QA" }));
    await user.type(screen.getByTestId("query-input"), "Bản tin trên màn hình tên gì?");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => expect(screen.getByTestId("qa-assist-panel")).toBeInTheDocument());
    expect(within(screen.getByTestId("qa-input-candidates")).getAllByRole("button")).toHaveLength(2);

    await user.click(screen.getByTestId("qa-analyze"));
    await waitFor(() => expect(screen.getByTestId("qa-analysis")).toBeInTheDocument());
    expect(screen.getAllByTestId("qa-answer-option")).toHaveLength(2);
    expect(screen.getAllByTestId("qa-answer-option")[0]).toHaveTextContent("Bản tin thời sự HTV7");
    expect(screen.getByTestId("qa-hotspot")).toHaveTextContent("95%");
    expect(screen.getByTestId("qa-web-grounding")).toHaveTextContent("HTV7 news programme");
    expect(screen.getByTestId("qa-pass3-verification")).toHaveTextContent("NVILA pass 3 · checked");
    expect(screen.getAllByTestId("qa-answer-option")[0]).toHaveTextContent("NVILA visually consistent");
    expect(screen.queryByTestId("video-viewer")).not.toBeInTheDocument();

    const analyzeCall = vi.mocked(fetch).mock.calls.find(
      ([url]) => String(url).endsWith("/api/qa/analyze"),
    );
    const body = JSON.parse(String(analyzeCall?.[1]?.body));
    expect(body.candidates).toHaveLength(2);
    expect(body.candidates[0]).toMatchObject({
      submit_keyframe_id: FRAME1.submit_keyframe_id,
      retrieval_score: FRAME1.score,
    });
    expect(body.web_grounding).toBe("auto");

    await user.click(screen.getAllByTestId("qa-answer-option")[0]);
    expect(await screen.findByTestId("video-viewer")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId("qa-timeline-marker")).toBeInTheDocument());

    await user.click(screen.getByTestId("open-submit"));
    expect(await screen.findByTestId("qa-answer")).toHaveValue("Bản tin thời sự HTV7");
  });

  it("supports keyboard frame selection (ArrowRight)", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getAllByTestId("frame-thumb"));

    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "ArrowRight" });

    await waitFor(() => {
      const detail = screen.getByTestId("detail-panel");
      expect(within(detail).getByTestId("submit-id")).toHaveTextContent("K01/K01_V001/006");
    });
  });

  it("submit guard blocks a duplicate submit", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetch([
        { id: "old", ts: Date.now() / 1000, task_id: TKIS_SCOPE, query_type: "T-KIS", payload: { video_id: "K01_V001", frame_idx: 0 }, dedup_keys: ["K01_V001:0"], status: "dres_ok", verdict: "WRONG", was_duplicate: false },
      ]),
    );
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));

    await user.click(screen.getByTestId("open-submit"));
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());
    expect(screen.getByTestId("dup-warn")).toBeInTheDocument();
    expect(screen.getByTestId("confirm-submit")).toHaveTextContent(/Submit anyway/i);
  });

  it("switches between grouped and flat top-K views", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => expect(screen.getByTestId("results")).toBeInTheDocument());

    await user.click(screen.getByTestId("view-flat"));
    await waitFor(() => expect(screen.getByTestId("results-flat")).toBeInTheDocument());
    // both frames shown as flat cards, ranked
    expect(screen.getAllByTestId("flat-card")).toHaveLength(2);

    await user.click(screen.getByTestId("view-grouped"));
    await waitFor(() => expect(screen.getByTestId("results")).toBeInTheDocument());
  });

  it("'v' shows the inline video under the group and hides on a second press", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));

    // no video until 'v'
    expect(screen.queryByTestId("video-viewer")).not.toBeInTheDocument();

    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    await waitFor(() => {
      const v = screen.getByTestId("video-viewer") as HTMLVideoElement;
      expect(v.getAttribute("src")).toContain("/Videos/Videos_K01/K01_V001.mp4");
    });

    fireEvent.keyDown(window, { key: "v" });
    await waitFor(() => expect(screen.queryByTestId("video-viewer")).not.toBeInTheDocument());
  });

  it.each(["T-KIS", "QA", "V-KIS"] as const)(
    "%s: the paused panel submits the exact raw frame from its own button",
    async (queryType) => {
      const user = userEvent.setup();
      render(<App />);
      if (queryType !== "T-KIS") {
        await user.click(screen.getByRole("tab", { name: queryType }));
      }
      await user.type(screen.getByTestId("query-input"), "thời sự");
      await user.click(screen.getByTestId("search-btn"));
      await waitFor(() => screen.getByTestId("detail-panel"));

      (document.activeElement as HTMLElement)?.blur();
      fireEvent.keyDown(window, { key: "v" });
      const video = await screen.findByTestId("video-viewer") as HTMLVideoElement;
      Object.defineProperty(video, "currentTime", {
        configurable: true,
        writable: true,
        value: 5.2,
      });
      fireEvent.pause(video);

      const chip = await screen.findByTestId("paused-frame-chip");
      expect(chip).toHaveTextContent("frame 130");
      // Capturing a raw frame must not re-point the detail panel.
      expect(screen.getByTestId("submit-id")).toHaveTextContent("K01/K01_V001/001");
      expect(screen.getByTestId("open-submit")).toHaveTextContent("Submit result frame");

      await user.click(screen.getByTestId("submit-paused-frame"));
      expect(await screen.findByTestId("guard-frame-idx")).toHaveTextContent("130");
      expect(screen.getByTestId("guard-target")).toHaveTextContent("paused raw frame");
      expect(screen.getByText(/exact paused/)).toBeInTheDocument();
      if (queryType === "QA") {
        await user.type(screen.getByTestId("qa-answer"), "HTV7");
      }
      await user.click(screen.getByTestId("confirm-submit"));

      await waitFor(() => {
        const submitCall = vi.mocked(fetch).mock.calls.find(
          ([url]) => String(url).endsWith("/api/submit"),
        );
        expect(submitCall).toBeDefined();
        const body = JSON.parse(String(submitCall?.[1]?.body));
        expect(body.query_type).toBe(queryType);
        expect(body.payload).toMatchObject({
          video_id: "K01_V001",
          frame_idx: 130,
          timestamp: 5.2,
        });
        expect(body.payload.submit_keyframe_id).toBeUndefined();
      });
    },
  );

  it("runs the duplicate pre-check against the frame the guard was opened for", async () => {
    // Only the PAUSED frame (130) was submitted before; the result keyframe was not.
    vi.stubGlobal(
      "fetch",
      mockFetch([
        { id: "old", ts: Date.now() / 1000, task_id: TKIS_SCOPE, query_type: "T-KIS", payload: { video_id: "K01_V001", frame_idx: 130 }, dedup_keys: ["K01_V001:130"], status: "dres_ok", verdict: "WRONG", was_duplicate: false },
      ]),
    );
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));

    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    const video = await screen.findByTestId("video-viewer") as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 5.2 });
    fireEvent.pause(video);
    await screen.findByTestId("paused-frame-chip");

    // Result target: not a duplicate, even though a paused frame is captured.
    fireEvent.keyDown(window, { key: "Enter" });
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());
    expect(screen.queryByTestId("dup-warn")).not.toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });

    // Paused target: the warning must fire on the very first open, not one late.
    fireEvent.keyDown(window, { key: "Enter", shiftKey: true });
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());
    expect(screen.getByTestId("dup-warn")).toHaveTextContent("K01_V001:130");
  });

  it("keeps Detail on the result keyframe while a paused frame is captured", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));

    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    const video = await screen.findByTestId("video-viewer") as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 5.2 });
    fireEvent.pause(video);
    await screen.findByTestId("paused-frame-chip");

    // Plain Enter stays on the result keyframe even though a raw frame exists.
    fireEvent.keyDown(window, { key: "Enter" });
    expect(await screen.findByTestId("guard-target")).toHaveTextContent("result keyframe");
    expect(screen.getByTestId("guard-frame-idx")).not.toHaveTextContent("130");
    await user.click(screen.getByTestId("confirm-submit"));

    await waitFor(() => {
      const call = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith("/api/submit"));
      expect(call).toBeDefined();
      const body = JSON.parse(String(call?.[1]?.body));
      expect(body.payload.submit_keyframe_id).toBe("K01/K01_V001/001");
      expect(body.payload.frame_idx).not.toBe(130);
    });
  });

  it("Shift+Enter submits the paused raw frame instead of the result keyframe", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));

    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    const video = await screen.findByTestId("video-viewer") as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 5.2 });
    fireEvent.pause(video);
    await screen.findByTestId("paused-frame-chip");

    fireEvent.keyDown(window, { key: "Enter", shiftKey: true });
    expect(await screen.findByTestId("guard-target")).toHaveTextContent("paused raw frame");
    expect(screen.getByTestId("guard-frame-idx")).toHaveTextContent("130");
  });

  it("TRAKE: a paused raw frame can be assigned to the active event slot", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), "a then b");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("trake-panel"));

    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    const video = await screen.findByTestId("video-viewer") as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", {
      configurable: true,
      writable: true,
      value: 5.2,
    });
    fireEvent.pause(video);

    expect(await screen.findByTestId("paused-frame-chip")).toHaveTextContent("frame 130");
    await user.click(screen.getByTestId("assign-paused-frame"));
    expect(screen.getByTestId("trake-slot-0")).toHaveTextContent("f130");
    expect(screen.queryByTestId("paused-frame-chip")).not.toBeInTheDocument();
  });

  it("Ctrl+/ toggles the keyboard shortcuts help", async () => {
    render(<App />);
    await waitFor(() => expect(fetch).toHaveBeenCalled());
    expect(screen.queryByTestId("shortcuts-modal")).not.toBeInTheDocument();

    fireEvent.keyDown(window, { key: "/", ctrlKey: true });
    await waitFor(() => expect(screen.getByTestId("shortcuts-modal")).toBeInTheDocument());
    expect(screen.getByText("Keyboard shortcuts")).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(screen.queryByTestId("shortcuts-modal")).not.toBeInTheDocument());
  });

  it("TRAKE: a full-coverage video can be submitted directly from the result list", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), "a then b");
    await user.click(screen.getByTestId("search-btn"));

    // 2/2 coverage video shows a quick-submit button
    await waitFor(() => expect(screen.getByTestId("trake-quick-submit")).toBeInTheDocument());
    await user.click(screen.getByTestId("trake-quick-submit"));

    // guard opens with the ordered frame_idx list filled from the video
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());
    expect(screen.getByTestId("trake-ordered-ids")).toHaveTextContent("frame 15926");
    expect(screen.getByTestId("trake-ordered-ids")).toHaveTextContent("frame 21030");
  });

  it("can switch to the simple view", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByText("Simple ⤴"));
    expect(screen.getByTestId("topk-slider")).toBeInTheDocument();
  });

  it("sends the selected InfoShot++ profile and disables unavailable channels", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByTestId("retrieval-database"), "infoshotpp");

    expect(screen.getByTestId("channel-ocr")).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByTestId("channel-speech")).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByTestId("channel-audio")).toHaveAttribute("aria-disabled", "true");

    await user.type(screen.getByTestId("query-input"), "street scene");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => {
      const call = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith("/api/search"));
      expect(call).toBeTruthy();
      expect(JSON.parse(String(call?.[1]?.body)).retrieval_database).toBe("infoshotpp");
    });
  });

  // ---- relevance feedback -------------------------------------------------
  async function searchAndFeedback(user: any, query = "thoi su") {
    render(<App />);
    await user.type(screen.getByTestId("query-input"), query);
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));
    return () => vi.mocked(fetch).mock.calls
      .filter(([u]) => String(u).endsWith("/api/search"))
      .map(([, i]) => JSON.parse(String(i?.body)));
  }

  it("re-runs the search when feedback changes instead of silently doing nothing", async () => {
    const user = userEvent.setup();
    const bodies = await searchAndFeedback(user);
    const before = bodies().length;

    await user.click(screen.getAllByTitle(/More like this frame/)[0]);

    await waitFor(() => expect(bodies().length).toBe(before + 1));
    expect(bodies()[before].feedback.positive_frames).toEqual(["K01/K01_V001/001"]);
  });

  it("sends 'more like this' as a FRAME seed, not a whole-video boost", async () => {
    const user = userEvent.setup();
    const bodies = await searchAndFeedback(user);

    await user.click(screen.getAllByTitle(/More like this frame/)[0]);
    await waitFor(() => expect(bodies().length).toBeGreaterThan(1));
    const fb = bodies()[bodies().length - 1].feedback;

    expect(fb.positive_frames).toEqual(["K01/K01_V001/001"]);
    expect(fb.positive_videos).toEqual([]);
  });

  it("prioritizes a video from the group header, replacing any deprioritize", async () => {
    const user = userEvent.setup();
    const bodies = await searchAndFeedback(user);

    await user.click(screen.getAllByTestId("deprioritize-video")[0]);
    await waitFor(() => expect(bodies().length).toBeGreaterThan(1));
    expect(bodies()[bodies().length - 1].feedback.negative_videos).toEqual(["K01_V001"]);

    await user.click(screen.getAllByTestId("prioritize-video")[0]);
    await waitFor(() =>
      expect(bodies()[bodies().length - 1].feedback.positive_videos).toEqual(["K01_V001"]));
    expect(bodies()[bodies().length - 1].feedback.negative_videos).toEqual([]);
  });

  it("lists every active feedback item and clears them all", async () => {
    const user = userEvent.setup();
    const bodies = await searchAndFeedback(user);
    expect(screen.queryByTestId("feedback-bar")).not.toBeInTheDocument();

    await user.click(screen.getAllByTitle(/More like this frame/)[0]);
    await user.click(screen.getAllByTitle("Exclude this frame")[1]);
    await user.click(screen.getAllByTestId("prioritize-video")[0]);

    const chips = await screen.findAllByTestId("feedback-chip");
    expect(chips.map((c) => c.textContent)).toEqual(expect.arrayContaining([
      expect.stringContaining("K01/K01_V001/001"),
      expect.stringContaining("K01/K01_V001/006"),
      expect.stringContaining("K01_V001"),
    ]));

    await user.click(screen.getByTestId("feedback-clear"));
    await waitFor(() => expect(screen.queryByTestId("feedback-bar")).not.toBeInTheDocument());
    await waitFor(() => {
      const fb = bodies()[bodies().length - 1].feedback;
      expect(fb).toEqual({ positive_videos: [], negative_videos: [], positive_frames: [], negative_frames: [] });
    });
  });

  it("does not leak feedback into the next question", async () => {
    const user = userEvent.setup();
    const bodies = await searchAndFeedback(user, "query one");
    await user.click(screen.getAllByTitle(/More like this frame/)[0]);
    await waitFor(() => expect(bodies().length).toBeGreaterThan(1));

    const input = screen.getByTestId("query-input");
    await user.clear(input);
    await user.type(input, "cau hoi khac han");
    await user.click(screen.getByTestId("search-btn"));

    await waitFor(() => {
      const last = bodies()[bodies().length - 1];
      expect(last.query).toBe("cau hoi khac han");
      expect(last.feedback.positive_frames).toEqual([]);
      expect(last.feedback.positive_videos).toEqual([]);
    });
  });

  // ---- neighbour keyframe browser ---------------------------------------
  async function openResults(user: any) {
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thoi su");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));
    (document.activeElement as HTMLElement)?.blur();
  }

  it("K opens a windowed neighbour strip instead of every keyframe", async () => {
    const user = userEvent.setup();
    await openResults(user);
    expect(screen.queryByTestId("neighbor-strip")).not.toBeInTheDocument();

    fireEvent.keyDown(window, { key: "k" });

    await screen.findByTestId("neighbor-strip");
    // 40 keyframes exist; only the window around the centre is mounted.
    const cells = screen.getAllByTestId(/neighbor-(cell|center)/);
    expect(cells.length).toBeLessThan(40);
    expect(screen.getByTestId("neighbor-position")).toHaveTextContent("keyframe 1 / 40");

    fireEvent.keyDown(window, { key: "k" });
    await waitFor(() => expect(screen.queryByTestId("neighbor-strip")).not.toBeInTheDocument());
  });

  it("arrows page the strip and pull in later keyframes", async () => {
    const user = userEvent.setup();
    await openResults(user);
    fireEvent.keyDown(window, { key: "k" });
    await screen.findByTestId("neighbor-strip");

    const shown = () => screen.getAllByTestId(/neighbor-(cell|center)/)
      .map((el) => el.getAttribute("title")?.split(" ")[0]);
    expect(shown()).not.toContain("K01/K01_V001/020");

    for (let i = 0; i < 15; i += 1) fireEvent.keyDown(window, { key: "ArrowRight" });

    await waitFor(() => expect(screen.getByTestId("neighbor-position")).toHaveTextContent("keyframe 16 / 40"));
    expect(shown()).toContain("K01/K01_V001/020");
  });

  it("keeps the video above the strip and seeks it while paging", async () => {
    const user = userEvent.setup();
    await openResults(user);
    fireEvent.keyDown(window, { key: "v" });
    const video = await screen.findByTestId("video-viewer") as HTMLVideoElement;
    fireEvent.keyDown(window, { key: "k" });
    const strip = await screen.findByTestId("neighbor-strip");

    // Order in the DOM: player first, neighbour strip after it.
    expect(video.compareDocumentPosition(strip) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    fireEvent.keyDown(window, { key: "ArrowRight" });
    fireEvent.keyDown(window, { key: "ArrowRight" });
    // pts_time == index, so paging twice from keyframe 1 seeks to t=2.
    await waitFor(() => expect(video.currentTime).toBe(2));
  });

  it("arrows still move between result frames when the strip is closed", async () => {
    const user = userEvent.setup();
    await openResults(user);
    expect(screen.getByTestId("submit-id")).toHaveTextContent("K01/K01_V001/001");

    fireEvent.keyDown(window, { key: "ArrowRight" });

    await waitFor(() => expect(screen.getByTestId("submit-id")).toHaveTextContent("K01/K01_V001/006"));
  });
});

describe("DRES submission", () => {
  async function search(user: ReturnType<typeof userEvent.setup>, queryType?: "QA") {
    render(<App />);
    if (queryType) await user.click(screen.getByRole("tab", { name: queryType }));
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));
  }

  const lastSubmit = () => {
    const call = vi.mocked(fetch).mock.calls.filter(([url]) => String(url).endsWith("/api/submit")).pop();
    return JSON.parse(String(call?.[1]?.body));
  };

  it("fetches the task statement and prefills the empty query box", async () => {
    render(<App />);

    expect(await screen.findByTestId("task-hint-text")).toHaveTextContent(TKIS_STATEMENT);
    await waitFor(() => expect(screen.getByTestId("query-input")).toHaveValue(TKIS_STATEMENT));
  });

  it("never overwrites a query the operator wrote, but offers the statement", async () => {
    const user = userEvent.setup();
    render(<App />);
    // Written before the statement lands (fireEvent is synchronous, so this
    // deterministically wins the race the operator would otherwise lose).
    fireEvent.change(screen.getByTestId("query-input"), { target: { value: "máy ảnh trên quyển sách" } });
    await screen.findByTestId("task-hint-text");

    expect(screen.getByTestId("query-input")).toHaveValue("máy ảnh trên quyển sách");

    await user.click(screen.getByTestId("task-hint-use"));
    expect(screen.getByTestId("query-input")).toHaveValue(TKIS_STATEMENT);
  });

  it("loads the statement of the run the query type routes to", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("task-hint-text");

    await user.click(screen.getByRole("tab", { name: "QA" }));

    await waitFor(() => expect(screen.getByTestId("task-hint-text")).toHaveTextContent(QA_STATEMENT));
  });

  it("clears the submit log only after confirming, and can scope it to the task", async () => {
    const other = {
      id: "old2", ts: Date.now() / 1000, task_id: "e-qa/qa-00", query_type: "QA",
      payload: { video_id: "K01_V001", frame_idx: 0, answer: "HTV7" },
      dedup_keys: ["K01_V001:0|text:htv7"], status: "dres_ok", verdict: "CORRECT", was_duplicate: false,
    };
    vi.stubGlobal("fetch", mockFetch([
      {
        id: "old1", ts: Date.now() / 1000, task_id: TKIS_SCOPE, query_type: "T-KIS",
        payload: { video_id: "K01_V001", frame_idx: 0 }, dedup_keys: ["K01_V001:0"],
        status: "dres_ok", verdict: "WRONG", was_duplicate: false,
      },
      other,
    ]));
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("history-sidebar");
    expect(await screen.findByText(/Submit history \(2\)/)).toBeInTheDocument();

    // One click only arms the action — nothing is deleted yet.
    await user.click(screen.getByTestId("history-clear"));
    expect(screen.getByTestId("history-clear-confirm")).toBeInTheDocument();
    expect(screen.getByText(/Submit history \(2\)/)).toBeInTheDocument();

    // Scoped clear keeps the other task's entries.
    await user.click(screen.getByTestId("history-clear-task"));
    await waitFor(() => expect(screen.getByText(/Submit history \(1\)/)).toBeInTheDocument());
    expect(historyStore).toEqual([other]);

    await user.click(screen.getByTestId("history-clear"));
    await user.click(screen.getByTestId("history-clear-all"));
    await waitFor(() => expect(screen.getByText(/Submit history \(0\)/)).toBeInTheDocument());
    expect(historyStore).toEqual([]);
  });

  it("deletes only the submissions the operator picked", async () => {
    const entry = (id: string, frame: number) => ({
      id, ts: Date.now() / 1000, task_id: TKIS_SCOPE, query_type: "T-KIS",
      payload: { video_id: "K01_V001", frame_idx: frame },
      dedup_keys: [`K01_V001:${frame}`], status: "dres_ok", verdict: "WRONG", was_duplicate: false,
    });
    vi.stubGlobal("fetch", mockFetch([entry("h1", 100), entry("h2", 200), entry("h3", 300)]));
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTestId("history-sidebar");

    // Multi-select two of the three, then delete just those.
    await user.click(screen.getByTestId("history-pick-h1"));
    await user.click(screen.getByTestId("history-pick-h3"));
    expect(screen.getByTestId("history-selection")).toHaveTextContent("2 mục đã chọn");
    await user.click(screen.getByTestId("history-delete-selected"));

    await waitFor(() => expect(historyStore.map((h) => h.id)).toEqual(["h2"]));
    expect(screen.getByText(/Submit history \(1\)/)).toBeInTheDocument();

    // The per-row ✕ deletes a single entry.
    await user.click(screen.getByTestId("history-delete-h2"));
    await waitFor(() => expect(historyStore).toEqual([]));
  });

  it("shows the open task, its type and the countdown", async () => {
    render(<App />);
    expect(await screen.findByTestId("dres-task")).toHaveTextContent("tkis-00");
    expect(screen.getByTestId("dres-task-status")).toHaveTextContent("RUNNING");
    expect(screen.getByTestId("dres-clock")).toHaveTextContent("5:00 left");
  });

  it("routes the submit to the evaluation run that matches the query type", async () => {
    const user = userEvent.setup();
    await search(user);

    await user.click(screen.getByTestId("open-submit"));
    expect(await screen.findByTestId("guard-evaluation")).toHaveTextContent("tkis");
    await waitFor(() => expect(screen.getByTestId("guard-task-auto")).toHaveTextContent("tkis-00"));
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    await user.click(screen.getByRole("tab", { name: "QA" }));
    await waitFor(() => expect(screen.getByTestId("dres-task")).toHaveTextContent("qa-00"));
  });

  it("KIS submits a media item with a millisecond window, never text", async () => {
    const user = userEvent.setup();
    await search(user);

    await user.click(screen.getByTestId("open-submit"));
    // The guard shows the exact body before it is sent.
    await waitFor(() =>
      expect(screen.getByTestId("guard-json")).toHaveTextContent(/"mediaItemName": "K01_V001"/),
    );
    expect(screen.getByTestId("guard-answer-summary")).toHaveTextContent("0–500 ms");

    await user.click(screen.getByTestId("confirm-submit"));

    await waitFor(() => {
      const body = lastSubmit();
      expect(body.evaluation_id).toBe("e-tkis");
      expect(body.query_type).toBe("T-KIS");
      expect(body.answer_mode).toBe("auto");
      expect(body.segment_pad_ms).toBe(500);
      // pts_time + fps let the backend build start/end in ms.
      expect(body.payload).toMatchObject({ video_id: "K01_V001", timestamp: 0, fps: 25 });
      expect(body.payload.answer).toBeUndefined();
    });
  });

  it("a manual answer shape never survives a query-type switch", async () => {
    // Regression: `temporal` picked on a KIS task used to stick, so the next QA
    // submit silently sent a media segment and dropped the typed answer.
    const user = userEvent.setup();
    await search(user);

    await user.click(screen.getByTestId("open-submit"));
    await user.selectOptions(await screen.findByTestId("guard-answer-mode"), "temporal");
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    await user.click(screen.getByRole("tab", { name: "QA" }));
    await user.type(screen.getByTestId("query-input"), "tác giả");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));
    await user.click(screen.getByTestId("open-submit"));

    expect(await screen.findByTestId("guard-answer-mode")).toHaveValue("auto");
    await user.type(screen.getByTestId("qa-answer"), "Nguyễn Thắm");
    await waitFor(() =>
      expect(screen.getByTestId("guard-json")).toHaveTextContent(/"text": "Nguyễn Thắm"/),
    );
  });

  it("warns when the answer shape contradicts the DRES task type", async () => {
    const user = userEvent.setup();
    await search(user, "QA");

    await user.click(screen.getByTestId("open-submit"));
    await user.type(await screen.findByTestId("qa-answer"), "Nguyễn Thắm");
    await user.selectOptions(screen.getByTestId("guard-answer-mode"), "temporal");

    await waitFor(() =>
      expect(screen.getByTestId("submit-guard")).toHaveTextContent(/SẼ BỊ BỎ/),
    );
  });

  it("QA submits the answer as text", async () => {
    const user = userEvent.setup();
    await search(user, "QA");

    await user.click(screen.getByTestId("open-submit"));
    await user.type(await screen.findByTestId("qa-answer"), "HTV7");

    await waitFor(() => expect(screen.getByTestId("guard-json")).toHaveTextContent(/"text": "HTV7"/));
    await user.click(screen.getByTestId("confirm-submit"));

    await waitFor(() => {
      const body = lastSubmit();
      expect(body.evaluation_id).toBe("e-qa");
      expect(body.payload.answer).toBe("HTV7");
    });
  });

  it("QA duplicate check covers the segment and the answer text", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetch([
        {
          id: "old", ts: Date.now() / 1000, task_id: "e-qa/qa-00", query_type: "QA",
          payload: { video_id: "K01_V001", frame_idx: 0, answer: "HTV7" },
          dedup_keys: ["K01_V001:0|text:htv7"], status: "dres_ok", verdict: "WRONG", was_duplicate: false,
        },
      ]),
    );
    const user = userEvent.setup();
    await search(user, "QA");

    await user.click(screen.getByTestId("open-submit"));
    await user.type(await screen.findByTestId("qa-answer"), "HTV7");
    expect(await screen.findByTestId("dup-warn")).toHaveTextContent("K01_V001:0|text:htv7");

    // A different answer from the same frame is a new attempt, not a duplicate.
    await user.clear(screen.getByTestId("qa-answer"));
    await user.type(screen.getByTestId("qa-answer"), "HTV9");
    await waitFor(() => expect(screen.queryByTestId("dup-warn")).not.toBeInTheDocument());
  });

  it("blocks the submit when the payload cannot be built", async () => {
    const user = userEvent.setup();
    await search(user, "QA");

    // QA with an empty answer has no `text` — DRES would reject it.
    await user.click(screen.getByTestId("open-submit"));
    await waitFor(() => expect(screen.getByTestId("confirm-submit")).toBeDisabled());
  });

  it("keeps working with DRES offline: the bar says so and the guard still submits", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      const path = String(url);
      if (path.endsWith("/api/dres/status") || path.includes("/api/dres/evaluations")) {
        return { ok: false, status: 502, json: async () => ({ detail: "DRES unreachable" }) } as Response;
      }
      return mockFetch()(url, init);
    }));
    const user = userEvent.setup();
    await search(user);

    expect(await screen.findByTestId("dres-offline")).toBeInTheDocument();
    await user.click(screen.getByTestId("open-submit"));
    await waitFor(() => expect(screen.getByTestId("confirm-submit")).toBeEnabled());
  });
});
