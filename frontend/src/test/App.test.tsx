import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { buildZipBytes } from "../lib/zip";
import { buildSubmissionFiles } from "../lib/submission";
import { extractVideoThumbnails } from "../lib/videoThumbnail";

vi.mock("../lib/videoThumbnail", async () => {
  const actual = await vi.importActual<typeof import("../lib/videoThumbnail")>("../lib/videoThumbnail");
  return {
    ...actual,
    extractVideoThumbnails: vi.fn(async (_url: string, times: readonly number[]) =>
      times.map((time) => `data:image/jpeg;base64,raw-${time}`),
    ),
  };
});

/** A query pack shaped like the organiser's: one .txt per question, type in the name. */
const PACK = [
  { name: "query-p1-1-kis.txt", text: "bản tin thời sự buổi tối" },
  { name: "query-p1-2-qa.txt", text: "xã này tên là gì?" },
  { name: "query-p1-3-trake.txt", text: "E1: bắt đầu chạy.\nE2: về đích." },
];

/** Import a pack through the real file input (STORED zip: no inflate in jsdom).
 *
 *  Parsing no longer applies the pack — publishing one replaces what the whole
 *  team answers — so the helper confirms it the way an operator would. With
 *  Supabase disabled in tests, publishing applies the pack to this client. */
async function importPack(files = PACK) {
  const bytes = buildZipBytes(files);
  const file = new File([bytes as BlobPart], "query-p1-groupA.zip", { type: "application/zip" });
  fireEvent.change(screen.getByTestId("import-input"), { target: { files: [file] } });
  await screen.findByTestId("pack-preview");
  fireEvent.click(screen.getByTestId("pack-publish"));
  await waitFor(() =>
    expect(screen.getAllByTestId(/^rail-tab-\d+$/)).toHaveLength(files.length),
  );
}

/** Open the tab the rail shows at `index` (tabs come back in pack order). */
async function openTab(user: ReturnType<typeof userEvent.setup>, index: number) {
  await user.click(screen.getByTestId(`rail-tab-${index}`));
}

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

const kfUrl = (n: number, videoId = "K19_V028") =>
  `https://media.test/Keyframes/Keyframes_K19/${videoId}/${String(n).padStart(3, "0")}.jpg`;
/** One TRAKE heat peak, the shape `/api/search/trake` returns per event. */
const peak = (
  eventIndex: number,
  keyframeN: number,
  pts: number,
  frameIdx: number,
  strength: number,
  picked = false,
  videoId = "K19_V028",
) => ({
  event_index: eventIndex,
  submit_keyframe_id: `K19/${videoId}/${String(keyframeN).padStart(3, "0")}`,
  keyframe_n: keyframeN,
  frame_idx: frameIdx,
  pts_time: pts,
  score: 0.03,
  strength,
  via_fill: false,
  selected_by_dp: picked,
  keyframe_url: kfUrl(keyframeN, videoId),
});

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
  videos: [
    {
      video_id: "K19_V028",
      video_url: "https://media.test/Videos/Videos_K19/K19_V028.mp4",
      trake_video_score: 0.981,
      coverage: 2,
      confident_coverage: 2,
      filled_events: 0,
      evidence_coverage: 2,
      chain_quality: 0.9,
      mean_quality: 0.93,
      min_quality: 0.83,
      complete: true,
      warning: null,
      min_event_gap: 312,
      duration_s: 900,
      preliminary_rank: 1,
      events: [
        {
          event_index: 1,
          has_candidate: true,
          in_chain: true,
          representative: peak(1, 203, 530, 15926, 0.95, true),
          peaks: [peak(1, 12, 120, 3600, 0.42), peak(1, 203, 530, 15926, 0.95, true)],
        },
        {
          event_index: 2,
          has_candidate: true,
          in_chain: true,
          representative: peak(2, 301, 842, 21030, 0.83, true),
          peaks: [peak(2, 301, 842, 21030, 0.83, true)],
        },
      ],
      best_chain: [
        { event_index: 1, submit_keyframe_id: "K19/K19_V028/203", keyframe_n: 203, frame_idx: 15926, pts_time: 530, score: 0.03, via_fill: false, keyframe_url: kfUrl(203) },
        { event_index: 2, submit_keyframe_id: "K19/K19_V028/301", keyframe_n: 301, frame_idx: 21030, pts_time: 842, score: 0.03, via_fill: false, keyframe_url: kfUrl(301) },
      ],
    },
    {
      video_id: "K19_V099",
      video_url: "https://media.test/Videos/Videos_K19/K19_V099.mp4",
      trake_video_score: 0.541,
      coverage: 1,
      confident_coverage: 1,
      filled_events: 0,
      evidence_coverage: 2,
      chain_quality: 0.6,
      mean_quality: 0.6,
      min_quality: 0.6,
      complete: false,
      warning: "Partial sequence: events 2 have no orderable candidate.",
      min_event_gap: null,
      duration_s: 600,
      preliminary_rank: 2,
      events: [
        {
          event_index: 1,
          has_candidate: true,
          in_chain: true,
          representative: peak(1, 400, 300, 9000, 0.6, true, "K19_V099"),
          peaks: [peak(1, 400, 300, 9000, 0.6, true, "K19_V099")],
        },
        {
          event_index: 2,
          has_candidate: true,
          in_chain: false,
          representative: peak(2, 20, 100, 3000, 0.5, false, "K19_V099"),
          peaks: [peak(2, 20, 100, 3000, 0.5, false, "K19_V099")],
        },
      ],
      best_chain: [
        { event_index: 1, submit_keyframe_id: "K19/K19_V099/400", keyframe_n: 400, frame_idx: 9000, pts_time: 300, score: 0.03, via_fill: false, keyframe_url: kfUrl(400, "K19_V099") },
      ],
    },
  ],
  pass2: { targets: ["K19_V099"], candidates: 2 },
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
    if (!(p.answer ?? "").trim()) return "QA tasks require `text` (answer); the answer field is empty.";
  }
  if (mode !== "text" && !p.video_id) return "Missing `mediaItemName` (video_id) for KIS/QA/TRAKE tasks.";
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

let keyframeLookups: string[] = [];
let historyStore: any[] = [];
let canvasRequests: any[] = [];

let answerGenRequests: any[] = [];
/** DRES is the default submission path whenever the server is configured, so
 *  the CSV-pack tests run with it unavailable; the DRES tests switch it on. */
let dresConfigured = false;
let inFlightSearches = 0;
let maxConcurrentSearches = 0;

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
    if (path.endsWith("/api/dres/status")) return json({ ...DRES_STATUS, configured: dresConfigured });
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
    if (path.endsWith("/api/answers/generate")) {
      const body = JSON.parse((init?.body as string) || "{}");
      answerGenRequests.push(body);
      const limit = body.limit ?? 100;
      const events = body.query_type_hint === "TRAKE" ? 2 : 1;
      return json({
        query: body.query,
        query_type: body.query_type_hint,
        retrieval_database: body.retrieval_database,
        reused_results: false,
        scope: null,
        params: { pool_depth: 100 },
        answers: Array.from({ length: limit }, (_, i) => ({
          rank: i + 1,
          video_id: "K01_V001",
          frames: Array.from({ length: events }, (_, e) => 1000 + i * 10 + e * 500),
          keyframe_ids: Array.from({ length: events }, () => (i === 0 ? "K01/K01_V001/001" : null)),
          pts_times: Array.from({ length: events }, () => null),
          answer: body.answer_text ?? "",
          kind: i === 0 ? "anchor" : "offset",
          offset_frames: i * 10,
        })),
        diagnostics: { n_videos_in_pool: 1 },
        warnings: [],
        latency_ms: { answer_gen_ms: 3 },
        mode: "mock",
      });
    }
    if (path.endsWith("/api/search/trake")) return json(TRAKE_RESPONSE);
    if (path.endsWith("/api/search")) {
      // Held open for a macrotask so a parallel fan-out would overlap here.
      inFlightSearches += 1;
      maxConcurrentSearches = Math.max(maxConcurrentSearches, inFlightSearches);
      await new Promise((resolve) => setTimeout(resolve, 0));
      inFlightSearches -= 1;
      return json(SEARCH_RESPONSE);
    }
    if (path.endsWith("/api/query/parse")) return json(SEARCH_RESPONSE.parsed);
    if (path.endsWith("/api/qa/analyze")) return json(QA_ANALYSIS_RESPONSE);
    if (path.includes("/timeline")) return json(TIMELINE);
    if (path.includes("/api/keyframes/")) {
      const id = decodeURIComponent(path.split("/api/keyframes/")[1].split("?")[0]);
      keyframeLookups.push(id);
      return json({
        image_id: id, submit_keyframe_id: id, video_id: "K01_V001", keyframe_n: 1,
        pts_time: 0, fps: 25, frame_idx: 0, retrieval_database: "btc",
        keyframe_url: `https://media.test/Keyframes/${id}.jpg`,
        video_url: FRAME1.video_url, found: true,
      });
    }
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
  keyframeLookups = [];
  answerGenRequests = [];
  inFlightSearches = 0;
  maxConcurrentSearches = 0;
  dresConfigured = false;
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
    // Timeline vẫn còn, chỉ bỏ dải ảnh keyframe: marker hotspot là thứ nói cho
    // operator biết bằng chứng nằm ở đâu trong clip, và nó không tốn byte nào.
    await waitFor(() => expect(screen.getByTestId("qa-timeline-marker")).toBeInTheDocument());
    // ...còn dải ảnh phải biến mất — chính nó là ~9 MB mỗi lần mở video.
    expect(document.querySelectorAll(".tl-film-frame")).toHaveLength(0);

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
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await waitFor(() => screen.getByTestId("detail-panel"));

    // First submit of this frame writes a row…
    await user.click(screen.getByTestId("open-submit"));
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());
    expect(screen.queryByTestId("dup-warn")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("confirm-submit"));

    // …so the second one is a repeat, and the guard says so before it lands.
    await user.click(screen.getByTestId("open-submit"));
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());
    expect(screen.getByTestId("dup-warn")).toHaveTextContent("K01_V001,0");
    expect(screen.getByTestId("confirm-submit")).toHaveTextContent(/despite duplicate/i);
  });

  it("retrieves 100 keyframes by default and sends the slider value on the next search", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));

    const bodies = () =>
      vi.mocked(fetch).mock.calls
        .filter(([url]) => String(url).endsWith("/api/search"))
        .map(([, init]) => JSON.parse(String(init?.body)));
    expect(bodies()[0].top_k).toBe(100);
    expect(screen.getByTestId("topk-value")).toHaveTextContent("100");

    fireEvent.change(screen.getByTestId("topk-slider"), { target: { value: "600" } });
    await user.click(screen.getByTestId("search-btn"));

    await waitFor(() => expect(bodies()).toHaveLength(2));
    expect(bodies()[1].top_k).toBe(600);
    // The 50-video cap would otherwise trim a deep search back to roughly the
    // old frame count, so it scales with the depth that was asked for.
    expect(bodies()[1].max_videos).toBe(300);
  });

  it("does not re-run the search while the depth slider is being dragged", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));

    const searchCount = () =>
      vi.mocked(fetch).mock.calls.filter(([url]) => String(url).endsWith("/api/search")).length;
    expect(searchCount()).toBe(1);

    for (const value of ["200", "400", "800", "1000"]) {
      fireEvent.change(screen.getByTestId("topk-slider"), { target: { value } });
    }
    // Every channel would be re-queried on each drag step otherwise.
    expect(searchCount()).toBe(1);
    expect(screen.getByTestId("topk-value")).toHaveTextContent("1000");
    // ...and the operator is told the results on screen are not the new depth.
    expect(screen.getByTestId("topk-dirty")).toHaveTextContent("100");
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

  // ---- scrubbing the open video ------------------------------------------
  /** Search, select a frame, open the inline video with 'v'. */
  async function openVideo(user: ReturnType<typeof userEvent.setup>) {
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));
    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    const video = (await screen.findByTestId("video-viewer")) as HTMLVideoElement;
    video.currentTime = 30;
    return video;
  }

  it("scrubs the open video: arrows by 5s, a/d by 1s", async () => {
    const user = userEvent.setup();
    const video = await openVideo(user);

    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(video.currentTime).toBeCloseTo(35, 3);
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(video.currentTime).toBeCloseTo(30, 3);

    fireEvent.keyDown(window, { key: "d" });
    expect(video.currentTime).toBeCloseTo(31, 3);
    fireEvent.keyDown(window, { key: "a" });
    expect(video.currentTime).toBeCloseTo(30, 3);

    // Shift-free capitals reach the same handler (Caps Lock is not a mode here).
    fireEvent.keyDown(window, { key: "D" });
    expect(video.currentTime).toBeCloseTo(31, 3);
    fireEvent.keyDown(window, { key: "A" });
    expect(video.currentTime).toBeCloseTo(30, 3);
  });

  it("keeps the arrow keys away from the video's own media controls", async () => {
    const user = userEvent.setup();
    const video = await openVideo(user);
    // Chrome's shadow-DOM controls seek on their own when the element sees an
    // arrow key — the focused progress slider steps ~1% of duration, which
    // stacked on top of our 5s and moved the playhead ~16s per press. Our
    // handler must consume the key before the element is ever reached.
    const reachedVideo = vi.fn();
    video.addEventListener("keydown", reachedVideo);

    fireEvent.keyDown(video, { key: "ArrowRight" });
    fireEvent.keyDown(video, { key: "ArrowLeft" });
    fireEvent.keyDown(video, { key: "a" });
    fireEvent.keyDown(video, { key: "d" });

    expect(reachedVideo).not.toHaveBeenCalled();

    // Space stays the video's own play/pause, so it must still get through.
    fireEvent.keyDown(video, { key: " " });
    expect(reachedVideo).toHaveBeenCalledTimes(1);

    video.removeEventListener("keydown", reachedVideo);
  });

  it("rewinding stops at the start instead of going negative", async () => {
    const user = userEvent.setup();
    const video = await openVideo(user);
    video.currentTime = 2;

    fireEvent.keyDown(window, { key: "ArrowLeft" });

    expect(video.currentTime).toBe(0);
  });

  it("leaves the arrows on the result list while no video is open", async () => {
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

  it("gives the arrows back to the neighbour strip when it is open", async () => {
    const user = userEvent.setup();
    const video = await openVideo(user);
    fireEvent.keyDown(window, { key: "k" });
    await screen.findByTestId("neighbor-strip");
    const before = video.currentTime;

    fireEvent.keyDown(window, { key: "ArrowRight" });

    // The strip pages and drives the video to that keyframe's pts — which is a
    // seek to an absolute time, never `before + 5`.
    await waitFor(() =>
      expect(screen.getByTestId("neighbor-position")).toHaveTextContent("keyframe 2 / 40"),
    );
    expect(video.currentTime).not.toBeCloseTo(before + 5, 3);
    // 'a'/'d' have no such contention and still nudge the video.
    const paged = video.currentTime;
    fireEvent.keyDown(window, { key: "d" });
    expect(video.currentTime).toBeCloseTo(paged + 1, 3);
  });

  it("never scrubs while the operator is typing", async () => {
    const user = userEvent.setup();
    const video = await openVideo(user);
    const input = screen.getByTestId("query-input");
    input.focus();
    const before = video.currentTime;

    // 'a' and 'd' are ordinary letters: typing them must reach the query box,
    // not the playhead. Arrows move the caret for the same reason.
    fireEvent.keyDown(input, { key: "a" });
    fireEvent.keyDown(input, { key: "d" });
    fireEvent.keyDown(input, { key: "ArrowRight" });
    fireEvent.keyDown(input, { key: "ArrowLeft" });

    expect(video.currentTime).toBe(before);
  });

  it.each(["T-KIS", "QA", "V-KIS"] as const)(
    "%s: the paused panel submits the exact raw frame from its own button",
    async (queryType) => {
      const user = userEvent.setup();
      render(<App />);
      await importPack();
      if (queryType === "QA") await openTab(user, 1);
      if (queryType !== "T-KIS" && queryType !== "QA") {
        await user.click(screen.getByRole("tab", { name: queryType }));
      }
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
      // The row the guard is about to write carries the RAW frame, not the
      // result keyframe the detail panel is still showing.
      expect(screen.getByTestId("guard-csv-line")).toHaveTextContent(
        queryType === "QA" ? "K01_V001,130,HTV7" : "K01_V001,130",
      );
      await user.click(screen.getByTestId("confirm-submit"));

      await user.click(screen.getByTestId("open-submission"));
      const panel = await screen.findByTestId("submission-panel");
      const expectedFile = queryType === "QA" ? "query-p1-2-qa" : "query-p1-1-kis";
      expect(within(panel).getByTestId(`csv-${expectedFile}`)).toHaveTextContent(
        queryType === "QA" ? "K01_V001,130,HTV7" : "K01_V001,130",
      );
    },
  );

  it("runs the duplicate pre-check against the frame the guard was opened for", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await waitFor(() => screen.getByTestId("detail-panel"));

    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    const video = await screen.findByTestId("video-viewer") as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 5.2 });
    fireEvent.pause(video);
    await screen.findByTestId("paused-frame-chip");

    // Submit ONLY the paused frame (130); the result keyframe (0) stays unused.
    fireEvent.keyDown(window, { key: "Enter", shiftKey: true });
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());
    await user.click(screen.getByTestId("confirm-submit"));

    // Result target: not a duplicate, even though a paused frame is captured.
    fireEvent.keyDown(window, { key: "Enter" });
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());
    expect(screen.queryByTestId("dup-warn")).not.toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });

    // Paused target: the warning must fire on the very first open, not one late.
    fireEvent.keyDown(window, { key: "Enter", shiftKey: true });
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());
    expect(screen.getByTestId("dup-warn")).toHaveTextContent("K01_V001,130");
  });

  it("keeps Detail on the result keyframe while a paused frame is captured", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
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
    // The row follows the guard's target, so it holds the result keyframe (0).
    expect(screen.getByTestId("guard-csv-line")).toHaveTextContent("K01_V001,0");
    await user.click(screen.getByTestId("confirm-submit"));

    await user.click(screen.getByTestId("open-submission"));
    const panel = await screen.findByTestId("submission-panel");
    expect(within(panel).getByTestId("csv-query-p1-1-kis")).toHaveTextContent("K01_V001,0");
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

  it("scrolls the player into view when 'v' opens it below the fold", async () => {
    const scrollIntoView = vi.spyOn(Element.prototype, "scrollIntoView");
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "bản tin");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => expect(screen.getByTestId("results")).toBeInTheDocument());
    scrollIntoView.mockClear();

    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    await screen.findByTestId("video-viewer");

    // The player is rendered inside the result card it belongs to, so a group
    // holding hundreds of keyframes opens it far below the viewport.
    expect(scrollIntoView).toHaveBeenCalled();
    expect(scrollIntoView.mock.calls[0][0]).toMatchObject({ block: "center" });
    scrollIntoView.mockRestore();
  });

  it("timeline seeks by click and carries no keyframe images", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "bản tin");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => expect(screen.getByTestId("results")).toBeInTheDocument());
    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    const video = (await screen.findByTestId("video-viewer")) as HTMLVideoElement;
    await waitFor(() => expect(screen.getByTestId("timeline-seek")).toBeInTheDocument());

    // Cái đắt đỏ đã biến mất: ~60 thumbnail x ~150 KB nạp cùng lúc mỗi lần mở
    // một video, tất cả nằm trong khung nhìn nên loading="lazy" không hoãn được.
    expect(document.querySelectorAll(".tl-film-frame")).toHaveLength(0);

    // ...nhưng khả năng nhảy tới một điểm bất kỳ thì không được mất theo. Trước
    // đây seek là bấm vào một thumbnail; giờ cả dải nhận click.
    const bar = screen.getByTestId("timeline-seek");
    bar.getBoundingClientRect = () => ({
      left: 0, width: 1000, top: 0, height: 22, right: 1000, bottom: 22, x: 0, y: 0, toJSON: () => ({}),
    }) as DOMRect;
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 0 });

    fireEvent.click(bar, { clientX: 250 });   // 25% của dải

    // fixture TIMELINE.duration = 30 -> 25% = 7,5s
    expect(video.currentTime).toBe(7.5);
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

  it("TRAKE: results are VIDEOS with per-event representatives and a heat row", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), "a then b");
    await user.click(screen.getByTestId("search-btn"));

    await waitFor(() => expect(screen.getByTestId("trake-videos")).toBeInTheDocument());
    const cards = screen.getAllByTestId("trake-video-card");
    expect(cards).toHaveLength(2);
    // Ranked by TRAKE video score: coverage first, so the 2/2 video leads.
    expect(cards[0]).toHaveTextContent("K19_V028");
    expect(cards[0]).toHaveTextContent("2/2 events");
    expect(cards[1]).toHaveTextContent("K19_V099");
    expect(cards[1]).toHaveTextContent("1/2 events");
    // One representative per event, and the whole heat row behind it.
    expect(within(cards[0]).getByTestId("trake-event-1")).toBeInTheDocument();
    expect(within(cards[0]).getByTestId("trake-event-2")).toBeInTheDocument();
    expect(within(cards[0]).getAllByTestId("trake-heat-peak")).toHaveLength(3);
    // An event the chain could not place still shows its best candidate, badged.
    expect(within(cards[1]).getByTestId("trake-event-2")).toHaveTextContent("outside sequence");
  });

  it("TRAKE: clicking an event arms its slot without pulling the video down", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), "a then b");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => expect(screen.getByTestId("trake-videos")).toBeInTheDocument());

    // E1 is the active slot after a search.
    expect(screen.getByTestId("trake-slot-0").className).toContain("active");
    const card = screen.getAllByTestId("trake-video-card")[0];
    await user.click(within(card).getByTestId("trake-event-2"));

    expect(screen.getByTestId("trake-slot-1").className).toContain("active");
    expect(screen.getByTestId("trake-slot-0").className).not.toContain("active");
    // Choosing a moment and watching it are separate decisions: the card already
    // shows the frame, so a click must not cost a video fetch and a layout jump.
    expect(screen.queryByTestId("video-viewer")).not.toBeInTheDocument();

    // 'v' opens it, parked on the moment that was picked.
    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    const video = (await screen.findByTestId("video-viewer")) as HTMLVideoElement;
    expect(video.src).toContain("K19_V028.mp4");
  });

  it("TRAKE: clicking an alternative heat peak arms its event without a new search", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), "a then b");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => expect(screen.getByTestId("trake-videos")).toBeInTheDocument());
    const searchesBefore = vi.mocked(fetch).mock.calls
      .filter(([url]) => String(url).endsWith("/api/search/trake")).length;

    const card = screen.getAllByTestId("trake-video-card")[0];
    // E1's row carries the chain's own peak plus one alternative moment; the
    // alternative is exactly what the old sequence-only result threw away.
    const e1Peaks = within(within(card).getByTestId("trake-heat-row-1")).getAllByTestId("trake-heat-peak");
    expect(e1Peaks).toHaveLength(2);
    await user.click(e1Peaks[0]);

    expect(screen.getByTestId("trake-slot-0").className).toContain("active");
    // Verifying an alternative costs a click, not another retrieval round.
    expect(
      vi.mocked(fetch).mock.calls.filter(([url]) => String(url).endsWith("/api/search/trake")).length,
    ).toBe(searchesBefore);
  });

  it("TRAKE: verifying a heat peak points the detail panel at that peak, not the chain frame", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), "a then b");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => expect(screen.getByTestId("trake-videos")).toBeInTheDocument());

    const card = screen.getAllByTestId("trake-video-card")[0];
    const e1Row = within(card).getByTestId("trake-heat-row-1");
    // E1's alternative moment at 120s — a candidate the chain did NOT take, so
    // it exists nowhere in the result frame strip.
    const alternative = within(e1Row).getAllByTestId("trake-heat-peak")[0];
    await user.click(alternative);

    // Everything reading the selection has to describe the SAME moment, or the
    // operator verifies one frame while the panel documents another.
    const detail = screen.getByTestId("detail-panel");
    expect(detail).toHaveTextContent("K19/K19_V028/012");

    // And the player, once opened, starts on that same alternative.
    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    const video = (await screen.findByTestId("video-viewer")) as HTMLVideoElement;
    expect(video.src).toContain("K19_V028.mp4");
  });

  it("TRAKE: no prioritize/deprioritize controls — the endpoint takes no feedback", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), "a then b");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => expect(screen.getByTestId("trake-videos")).toBeInTheDocument());

    // A control that re-runs the search and changes nothing is worse than none.
    expect(screen.queryByTestId("prioritize-video")).not.toBeInTheDocument();
    expect(screen.queryByTestId("deprioritize-video")).not.toBeInTheDocument();
  });

  it("TRAKE: switching the retrieval database drops the previous result", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), "a then b");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => expect(screen.getByTestId("trake-videos")).toBeInTheDocument());

    await user.selectOptions(screen.getByTestId("retrieval-database"), "infoshotpp");
    // Those cards describe videos from a corpus that is no longer being searched.
    await waitFor(() => expect(screen.queryByTestId("trake-videos")).not.toBeInTheDocument());
  });

  /** jsdom ships no working DataTransfer, so drag tests carry their own. */
  function dataTransfer() {
    const store: Record<string, string> = {};
    return {
      dropEffect: "",
      effectAllowed: "",
      setData(type: string, value: string) { store[type] = value; },
      getData(type: string) { return store[type] ?? ""; },
      get types() { return Object.keys(store); },
    };
  }

  async function openTrakeResults(user: ReturnType<typeof userEvent.setup>) {
    render(<App />);
    await user.click(screen.getByRole("tab", { name: "TRAKE" }));
    await user.type(screen.getByTestId("query-input"), "a then b");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => expect(screen.getByTestId("trake-videos")).toBeInTheDocument());
    return screen.getAllByTestId("trake-video-card");
  }

  it("TRAKE: heat rows show the candidate frames themselves, not just marks", async () => {
    const user = userEvent.setup();
    const cards = await openTrakeResults(user);
    const e1Row = within(cards[0]).getByTestId("trake-heat-row-1");
    const peaks = within(e1Row).getAllByTestId("trake-heat-peak");

    // Every candidate carries its own keyframe image, so judging an alternative
    // costs a glance instead of a video load, a seek and a pause.
    expect(peaks).toHaveLength(2);
    for (const p of peaks) {
      expect(within(p).getByRole("presentation")).toHaveAttribute(
        "src",
        expect.stringContaining("/Keyframes/Keyframes_K19/K19_V028/"),
      );
    }
    expect(peaks[0]).toHaveTextContent("02:00"); // the alternative moment
    expect(peaks[1]).toHaveTextContent("08:50"); // the one the chain took
  });

  it("TRAKE: a frame dragged from the heat row lands in an event slot", async () => {
    const user = userEvent.setup();
    const cards = await openTrakeResults(user);
    const e1Row = within(cards[0]).getByTestId("trake-heat-row-1");
    const alternative = within(e1Row).getAllByTestId("trake-heat-peak")[0];

    const dt = dataTransfer();
    fireEvent.dragStart(alternative, { dataTransfer: dt });
    fireEvent.drop(screen.getByTestId("trake-slot-1"), { dataTransfer: dt });

    // frame_idx 3600 @ 120s, straight off the card — no video was ever loaded.
    expect(screen.getByTestId("trake-slot-1")).toHaveTextContent("f3600");
    expect(screen.getByTestId("trake-slot-1")).toHaveTextContent("02:00");
  });

  it("TRAKE: a frame dropped on an event replaces the chain's pick, and ↺ gives it back", async () => {
    const user = userEvent.setup();
    const cards = await openTrakeResults(user);
    const e1Card = within(cards[0]).getByTestId("trake-event-1");
    expect(e1Card).toHaveTextContent("08:50"); // the DP's choice

    const e1Row = within(cards[0]).getByTestId("trake-heat-row-1");
    const dt = dataTransfer();
    fireEvent.dragStart(within(e1Row).getAllByTestId("trake-heat-peak")[0], { dataTransfer: dt });
    fireEvent.drop(e1Card, { dataTransfer: dt });

    expect(within(cards[0]).getByTestId("trake-event-1")).toHaveTextContent("02:00");
    expect(within(cards[0]).getByTestId("trake-event-1")).toHaveTextContent("manually selected");

    // The backend result is never mutated, so the DP's answer is always recoverable.
    await user.click(within(cards[0]).getByTestId("trake-event-undo-1"));
    expect(within(cards[0]).getByTestId("trake-event-1")).toHaveTextContent("08:50");
    expect(within(cards[0]).queryByTestId("trake-event-undo-1")).not.toBeInTheDocument();
  });

  it("TRAKE: an override that breaks chronological order is called out", async () => {
    const user = userEvent.setup();
    const cards = await openTrakeResults(user);
    expect(within(cards[0]).queryByTestId("trake-order-warning")).not.toBeInTheDocument();

    // Put E1's 02:00 frame into E2, which leaves E1 at 08:50 ahead of it. The
    // organiser's parser rejects a row that is not chronological, and one
    // rejected row blocks the whole submission — so it has to show here.
    const e1Row = within(cards[0]).getByTestId("trake-heat-row-1");
    const dt = dataTransfer();
    fireEvent.dragStart(within(e1Row).getAllByTestId("trake-heat-peak")[0], { dataTransfer: dt });
    fireEvent.drop(within(cards[0]).getByTestId("trake-event-2"), { dataTransfer: dt });

    expect(within(cards[0]).getByTestId("trake-order-warning")).toHaveTextContent("E2");
  });

  it("TRAKE: a frame from another video cannot join this video's chain", async () => {
    const user = userEvent.setup();
    const cards = await openTrakeResults(user);
    const otherRow = within(cards[1]).getByTestId("trake-heat-row-1");
    const foreign = within(otherRow).getAllByTestId("trake-heat-peak")[0];

    const dt = dataTransfer();
    fireEvent.dragStart(foreign, { dataTransfer: dt });
    fireEvent.drop(within(cards[0]).getByTestId("trake-event-1"), { dataTransfer: dt });

    // Every TRAKE event has to come from the one video being submitted.
    expect(within(cards[0]).getByTestId("trake-event-1")).toHaveTextContent("08:50");
    expect(within(cards[0]).getByTestId("trake-event-1")).not.toHaveTextContent("manually selected");
  });

  it("TRAKE: quick submit sends the operator's frame, not the one it replaced", async () => {
    const user = userEvent.setup();
    const cards = await openTrakeResults(user);
    const e1Row = within(cards[0]).getByTestId("trake-heat-row-1");
    const dt = dataTransfer();
    fireEvent.dragStart(within(e1Row).getAllByTestId("trake-heat-peak")[0], { dataTransfer: dt });
    fireEvent.drop(within(cards[0]).getByTestId("trake-event-1"), { dataTransfer: dt });

    await user.click(within(cards[0]).getByTestId("trake-quick-submit"));
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());
    const ordered = screen.getByTestId("trake-ordered-ids");
    expect(ordered).toHaveTextContent("frame 3600");   // the dragged frame
    expect(ordered).not.toHaveTextContent("frame 15926"); // the DP's, replaced
    expect(ordered).toHaveTextContent("frame 21030");  // E2 untouched
  });

  it("TRAKE: seeking a paused video captures the frame with no play/pause dance", async () => {
    const user = userEvent.setup();
    const cards = await openTrakeResults(user);
    const e1Row = within(cards[0]).getByTestId("trake-heat-row-1");
    await user.click(within(e1Row).getAllByTestId("trake-heat-peak")[0]);
    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });

    const video = (await screen.findByTestId("video-viewer")) as HTMLVideoElement;
    // It opens parked on the frame instead of playing away from it.
    expect(video).not.toHaveAttribute("autoplay");
    expect(screen.queryByTestId("paused-frame-chip")).not.toBeInTheDocument();

    // The seek settling IS the frame pick: no play, no pause.
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 120 });
    fireEvent.seeked(video);

    // 120s at the 25fps fallback -> frame 3000, ready for Enter into the slot.
    expect(await screen.findByTestId("paused-frame-chip")).toHaveTextContent("frame 3000");
    await user.click(screen.getByTestId("assign-paused-frame"));
    expect(screen.getByTestId("trake-slot-0")).toHaveTextContent("f3000");
  });

  it("TRAKE: a slot refuses a frame from a different video", async () => {
    const user = userEvent.setup();
    const cards = await openTrakeResults(user);

    // E1 from the first video...
    const own = within(within(cards[0]).getByTestId("trake-heat-row-1")).getAllByTestId("trake-heat-peak")[0];
    const dt1 = dataTransfer();
    fireEvent.dragStart(own, { dataTransfer: dt1 });
    fireEvent.drop(screen.getByTestId("trake-slot-0"), { dataTransfer: dt1 });
    expect(screen.getByTestId("trake-slot-0")).toHaveTextContent("f3600");

    // ...then E2 from a DIFFERENT one. The row names one video and lists frame
    // indices under it, so this frame would be submitted as if it came from the
    // first video — well-formed and wrong.
    const foreign = within(within(cards[1]).getByTestId("trake-heat-row-1")).getAllByTestId("trake-heat-peak")[0];
    const dt2 = dataTransfer();
    fireEvent.dragStart(foreign, { dataTransfer: dt2 });
    fireEvent.drop(screen.getByTestId("trake-slot-1"), { dataTransfer: dt2 });

    expect(screen.getByTestId("trake-slot-1")).toHaveTextContent("empty");
    expect(await screen.findByText(/Sequence uses K19_V028/)).toBeInTheDocument();
  });

  it("TRAKE: the guard blocks a chain that is not chronological", async () => {
    const user = userEvent.setup();
    const cards = await openTrakeResults(user);

    // Put E1's 02:00 frame into E2, leaving E1 at 08:50 ahead of it.
    const e1Row = within(cards[0]).getByTestId("trake-heat-row-1");
    const dt = dataTransfer();
    fireEvent.dragStart(within(e1Row).getAllByTestId("trake-heat-peak")[0], { dataTransfer: dt });
    fireEvent.drop(within(cards[0]).getByTestId("trake-event-2"), { dataTransfer: dt });

    await user.click(within(cards[0]).getByTestId("trake-quick-submit"));
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());

    // The parser rejects a row out of order, and one rejected row blocks the
    // whole submission — so the guard must refuse, not merely warn.
    expect(screen.getByTestId("guard-order-violation")).toBeInTheDocument();
    expect(screen.getByTestId("confirm-submit")).toBeDisabled();
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

  it("sends the selected InfoShot++ profile and keeps the v2 metadata channels usable", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByTestId("retrieval-database"), "infoshotpp");

    // OCR / speech / audio are served from the InfoShot++ v2 indices, so they
    // must stay toggleable on this profile (only the OD channels are locked).
    expect(screen.getByTestId("channel-ocr")).toHaveAttribute("aria-disabled", "false");
    expect(screen.getByTestId("channel-speech")).toHaveAttribute("aria-disabled", "false");
    expect(screen.getByTestId("channel-audio")).toHaveAttribute("aria-disabled", "false");

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

describe("query pack + submission table", () => {
  it("toggles a per-question sticky note with the Backquote shortcut", async () => {
    render(<App />);
    expect(screen.queryByTestId("sticky-note")).not.toBeInTheDocument();

    fireEvent.keyDown(window, { key: "`", code: "Backquote" });
    expect(await screen.findByTestId("sticky-note")).toHaveTextContent("no question selected");

    fireEvent.keyDown(window, { key: "`", code: "Backquote" });
    await waitFor(() => expect(screen.queryByTestId("sticky-note")).not.toBeInTheDocument());
  });

  it("pushes only sticky candidates added since the previous push", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await screen.findByTestId("detail-panel");

    await user.click(screen.getByTestId("sticky-add-result"));
    expect(screen.queryByTestId("sticky-note")).not.toBeInTheDocument();
    expect(await screen.findByText("Pinned candidate to query-p1-1-kis.")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "`", code: "Backquote" });
    const note = await screen.findByTestId("sticky-note");
    expect(within(note).getAllByTestId(/^sticky-row-\d+$/)).toHaveLength(1);
    expect(within(note).getByTestId("sticky-push")).toHaveTextContent("Push 1");

    await user.click(within(note).getByTestId("sticky-push"));
    await waitFor(() => expect(within(note).getByTestId("sticky-push")).toBeDisabled());
    expect(within(note).queryByText("✓")).not.toBeInTheDocument();

    await user.click(within(note).getByTestId("sticky-close"));
    await user.click(screen.getByTestId("open-submission"));
    const firstBlock = await screen.findByTestId("submission-query-p1-1-kis");
    expect(within(firstBlock).getAllByTestId(/^row-query-p1-1-kis-\d+$/)).toHaveLength(1);
    expect(screen.getByTestId("csv-query-p1-1-kis")).toHaveTextContent("K01_V001,0");

    await user.click(screen.getByTestId("submission-back"));
    await user.click(screen.getAllByTestId("frame-thumb")[1]);
    await user.click(screen.getByTestId("sticky-add-result"));
    expect(screen.queryByTestId("sticky-note")).not.toBeInTheDocument();
    fireEvent.keyDown(window, { key: "`", code: "Backquote" });
    const reopened = await screen.findByTestId("sticky-note");
    expect(within(reopened).getAllByTestId(/^sticky-row-\d+$/)).toHaveLength(2);
    // The first candidate remains as history, but is not part of this push.
    expect(within(reopened).getByTestId("sticky-push")).toHaveTextContent("Push 1");

    await user.click(within(reopened).getByTestId("sticky-push"));
    await user.click(within(reopened).getByTestId("sticky-close"));
    await user.click(screen.getByTestId("open-submission"));
    const finalBlock = await screen.findByTestId("submission-query-p1-1-kis");
    expect(within(finalBlock).getAllByTestId(/^row-query-p1-1-kis-\d+$/)).toHaveLength(2);
    expect(screen.getByTestId("csv-query-p1-1-kis").textContent).toContain("K01_V001,450");
  });

  it("restores a Sticky TRAKE sequence with keyframe previews, including a raw event", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await openTab(user, 2);
    const card = (await screen.findAllByTestId("trake-video-card"))[0];

    // Start with the retrieved chain, then replace E1 by an exact paused raw
    // frame. Sticky stores raw provenance as keyframeIds[0] === null.
    await user.click(within(card).getByTestId("trake-quick-submit"));
    await user.click(within(screen.getByTestId("submit-guard")).getByRole("button", { name: "Cancel" }));
    const firstSlot = screen.getByTestId("trake-slot-0");
    await user.click(within(firstSlot).getByText("✕"));
    await user.click(firstSlot);
    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    const video = await screen.findByTestId("video-viewer") as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 120 });
    fireEvent.seeked(video);
    await user.click(await screen.findByTestId("assign-paused-frame"));
    expect(firstSlot).toHaveTextContent("f3000");

    await user.click(screen.getByTestId("sticky-add-result"));
    for (const index of [0, 1]) {
      await user.click(within(screen.getByTestId(`trake-slot-${index}`)).getByText("✕"));
    }
    fireEvent.keyDown(window, { key: "`", code: "Backquote" });
    const note = await screen.findByTestId("sticky-note");
    await user.click(within(note).getByTestId("sticky-restore-trake-0"));

    await waitFor(() => expect(screen.queryByTestId("sticky-note")).not.toBeInTheDocument());
    const restoredRaw = screen.getByTestId("trake-slot-0");
    const restoredKeyframe = screen.getByTestId("trake-slot-1");
    expect(restoredRaw).toHaveTextContent("f3000");
    expect(restoredKeyframe).toHaveTextContent("f21030");
    await waitFor(() => expect(within(restoredRaw).getByRole("img")).toBeInTheDocument());
    expect(within(restoredRaw).getByText("raw exact")).toHaveAttribute(
      "title",
      "Image extracted from the video at the exact raw-frame timestamp",
    );
    expect(within(restoredRaw).getByRole("img")).toHaveAttribute(
      "src",
      "data:image/jpeg;base64,raw-120",
    );
    expect(within(restoredKeyframe).getByRole("img")).toHaveAttribute(
      "src",
      expect.stringContaining("/Keyframes/"),
    );
    expect(extractVideoThumbnails).toHaveBeenCalledWith(expect.stringContaining(".mp4"), [120]);
  });

  it("opens one tab per question, on the right query type, and searches them all", async () => {
    render(<App />);
    await importPack();

    const rail = screen.getByTestId("tab-rail");
    expect(within(rail).getByTestId("rail-tab-0")).toHaveTextContent("T-KIS");
    expect(within(rail).getByTestId("rail-tab-1")).toHaveTextContent("QA");
    expect(within(rail).getByTestId("rail-tab-2")).toHaveTextContent("TRAKE");

    // The statement is loaded into the query box of its own tab…
    await waitFor(() => expect(screen.getByTestId("query-input")).toHaveValue(PACK[0].text));
    // …and every tab ran its search without anyone pressing the button.
    await waitFor(() => {
      const searched = vi.mocked(fetch).mock.calls
        .filter(([url]) => String(url).endsWith("/api/search"))
        .map(([, init]) => JSON.parse(String(init?.body)).query);
      expect(searched).toContain(PACK[0].text);
      expect(searched).toContain(PACK[1].text);
    });
    await waitFor(() => {
      const trake = vi.mocked(fetch).mock.calls
        .filter(([url]) => String(url).endsWith("/api/search/trake"))
        .map(([, init]) => JSON.parse(String(init?.body)).query);
      expect(trake).toContain(PACK[2].text);
    });
  });

  it("runs an imported pack one search at a time", async () => {
    render(<App />);
    await importPack();

    // Every question still gets searched...
    await waitFor(() => {
      const searched = vi.mocked(fetch).mock.calls
        .filter(([url]) => String(url).endsWith("/api/search"))
        .map(([, init]) => JSON.parse(String(init?.body)).query);
      expect(searched).toContain(PACK[0].text);
      expect(searched).toContain(PACK[1].text);
    });

    // ...but never two at once. Arming every tab together sent one search per
    // question simultaneously, each hitting two encoders plus Milvus and
    // Elastic; the services answered ConnectTimeout and the operator got
    // nothing back. The answer generator already runs questions one at a time.
    expect(maxConcurrentSearches).toBe(1);
  });

  it("asks for health once for the tab on screen, not once per tab", async () => {
    render(<App />);
    const before = vi.mocked(fetch).mock.calls.filter(([url]) =>
      String(url).includes("/api/health"),
    ).length;

    await importPack();
    await waitFor(() => expect(screen.getAllByTestId(/^rail-tab-\d+$/)).toHaveLength(PACK.length));

    const after = vi.mocked(fetch).mock.calls.filter(([url]) =>
      String(url).includes("/api/health"),
    ).length;
    // Every tab of a pack stays mounted. Ungated, a 25-question import fired 25
    // health checks at once, each fanning out to every upstream service.
    expect(after - before).toBeLessThanOrEqual(1);
  });

  it("renames the tab in real time when the query type changes inside it", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();

    expect(screen.getByTestId("rail-tab-0")).toHaveTextContent("T-KIS");
    await user.click(screen.getByRole("tab", { name: "V-KIS" }));
    expect(screen.getByTestId("rail-tab-0")).toHaveTextContent("V-KIS");
  });

  it("still shows a console on a second mount", async () => {
    // Regression: tab ids come from a module-level counter, so a hardcoded
    // initial `active` of "tab-1" matched nothing from the second mount on and
    // the workspace rendered an empty body.
    const first = render(<App />);
    expect(screen.getByTestId("query-input")).toBeInTheDocument();
    first.unmount();

    render(<App />);
    expect(screen.getByTestId("query-input")).toBeInTheDocument();
    expect(screen.getAllByTestId(/^rail-tab-\d+$/)).toHaveLength(1);
  });

  it("fills the whole pack from the answer generator", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByTestId("autogen-toggle"));

    // Small limit so the assertions read as counts rather than "100 of them".
    fireEvent.change(screen.getByTestId("autogen-limit"), { target: { value: "3" } });
    await user.click(screen.getByTestId("autogen-start"));

    await waitFor(() => expect(answerGenRequests).toHaveLength(PACK.length));
    expect(answerGenRequests.map((r) => r.query)).toEqual(PACK.map((p) => p.text));
    expect(answerGenRequests.map((r) => r.query_type_hint)).toEqual(["T-KIS", "QA", "TRAKE"]);
    expect(answerGenRequests.every((r) => r.limit === 3)).toBe(true);
    // The statement decides how many frames a TRAKE row carries, so the width
    // travels with the request: the backend uses it to refuse chains that
    // located fewer moments, which would be written as rows the export rejects.
    expect(answerGenRequests[2].event_count).toBe(2);
    expect(answerGenRequests[0].event_count).toBeUndefined();

    // The rows land in the table, in the order the generator produced them.
    const block = await screen.findByTestId("submission-query-p1-1-kis");
    await waitFor(() =>
      expect(within(block).getAllByTestId(/^row-query-p1-1-kis-\d+$/)).toHaveLength(3),
    );
    expect(screen.getByTestId("csv-query-p1-1-kis")).toHaveTextContent("K01_V001,1000");
    // A TRAKE row carries one frame per event.
    expect(screen.getByTestId("csv-query-p1-3-trake")).toHaveTextContent("K01_V001,1000,1500");
    // BTC has no Qwen vectors, so the generator must stay on PE there.
    expect(answerGenRequests.every((r) => JSON.stringify(r.image_models) === '["pe"]')).toBe(true);
  });

  it("generates from every image index the profile has, not PE alone", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByTestId("retrieval-database"), "infoshotpp");
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByTestId("autogen-toggle"));
    fireEvent.change(screen.getByTestId("autogen-limit"), { target: { value: "3" } });
    await user.click(screen.getByTestId("autogen-start"));

    await waitFor(() => expect(answerGenRequests).toHaveLength(PACK.length));
    // The generated list is what gets graded, so it is built on the same
    // evidence a two-model console search would surface. It used to run the
    // generator's own retrieval on PE alone whatever was ticked in the console.
    for (const request of answerGenRequests) {
      expect(request.image_models).toEqual(["pe", "qwen3_vl"]);
    }
  });

  it("keeps a hand-picked answer at the top and only fills what is missing", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));

    // Hand-pick one answer first.
    await user.click(screen.getByText("query-p1-1-kis +"));
    const row = await screen.findByTestId("row-query-p1-1-kis-0");
    await user.type(within(row).getAllByRole("textbox")[0], "L26_V001");
    // Enter sends the whole row once, rather than a write per keystroke.
    await user.type(within(row).getAllByRole("textbox")[1], "6400{Enter}");

    await user.click(screen.getByTestId("autogen-toggle"));
    fireEvent.change(screen.getByTestId("autogen-limit"), { target: { value: "3" } });
    await user.click(screen.getByTestId("autogen-start"));

    // The question is no longer skipped — it still wants its insurance tail —
    // and the row already there costs a position rather than being destroyed.
    await waitFor(() => expect(answerGenRequests).toHaveLength(PACK.length));
    expect(answerGenRequests.map((r) => r.query)).toContain(PACK[0].text);

    const block = await screen.findByTestId("submission-query-p1-1-kis");
    await waitFor(() =>
      expect(within(block).getAllByTestId(/^row-query-p1-1-kis-\d+$/)).toHaveLength(3),
    );
    // The hand-picked row survived AND still leads: at rank 1 a verified frame
    // scores the whole query, at rank 100 only a fifth of it.
    expect(within(await screen.findByTestId("row-query-p1-1-kis-0")).getAllByRole("textbox")[0])
      .toHaveValue("L26_V001");

    // It also travels WITH the request, so the generator neither repeats it nor
    // spends positions re-covering the instant it already covers — and so the
    // band schedule continues after it instead of restarting at rank 1.
    const request = answerGenRequests.find((r) => r.query === PACK[0].text);
    expect(request.taken).toEqual([{ video_id: "L26_V001", frames: [6400] }]);
    expect(request.limit).toBe(2); // 3 tổng − 1 rows đã có
  });

  it("replaces only its own rows when run a second time", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByText("query-p1-1-kis +"));
    const row = await screen.findByTestId("row-query-p1-1-kis-0");
    await user.type(within(row).getAllByRole("textbox")[0], "L26_V001");

    await user.click(screen.getByTestId("autogen-toggle"));
    fireEvent.change(screen.getByTestId("autogen-limit"), { target: { value: "3" } });
    await user.click(screen.getByTestId("autogen-start"));
    const block = await screen.findByTestId("submission-query-p1-1-kis");
    await waitFor(() =>
      expect(within(block).getAllByTestId(/^row-query-p1-1-kis-\d+$/)).toHaveLength(3),
    );

    // Run it again on that one question: the generated pair is swapped out, the
    // hand-picked row is not, and the question does not grow past its budget.
    await user.click(screen.getByTestId("autogen-one-query-p1-1-kis"));
    await waitFor(() => expect(answerGenRequests.length).toBeGreaterThan(PACK.length));
    await waitFor(() =>
      expect(
        within(screen.getByTestId("submission-query-p1-1-kis")).getAllByTestId(
          /^row-query-p1-1-kis-\d+$/,
        ),
      ).toHaveLength(3),
    );
    expect(within(await screen.findByTestId("row-query-p1-1-kis-0")).getAllByRole("textbox")[0])
      .toHaveValue("L26_V001");
  });

  it("says what it will keep and what it will replace before running", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByText("query-p1-1-kis +"));

    await user.click(screen.getByTestId("autogen-toggle"));
    const panel = screen.getByTestId("autogen-panel");
    expect(panel).toHaveTextContent("keep 1 manual rows");
    // Nothing generated yet, so there is nothing to replace.
    expect(screen.queryByTestId("autogen-regenerate-note")).not.toBeInTheDocument();

    fireEvent.change(screen.getByTestId("autogen-limit"), { target: { value: "2" } });
    await user.click(screen.getByTestId("autogen-start"));
    await waitFor(() => expect(answerGenRequests).toHaveLength(PACK.length));

    // Now every question holds generated rows, so a second run says so — and
    // says the hand-picked row is not part of what gets replaced.
    await user.click(screen.getByTestId("autogen-only-empty"));
    await waitFor(() =>
      expect(screen.getByTestId("autogen-regenerate-note")).toHaveTextContent("3 questions"),
    );
    expect(screen.getByTestId("autogen-regenerate-note")).toHaveTextContent("preserved");
  });

  it("regenerates one question without touching the rest", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByText("query-p1-2-qa +"));

    const row = await screen.findByTestId("row-query-p1-2-qa-0");
    await user.type(within(row).getAllByRole("textbox")[2], "năm người{Enter}");
    await user.click(screen.getByTestId("autogen-one-query-p1-2-qa"));

    await waitFor(() => expect(answerGenRequests).toHaveLength(1));
    expect(answerGenRequests[0].query).toBe(PACK[1].text);
    // The Q&A text the operator wrote is carried into every generated row —
    // the generator ranks where to look, it does not invent the answer.
    expect(answerGenRequests[0].answer_text).toBe("năm người");
    await waitFor(() =>
      expect(screen.getByTestId("csv-query-p1-2-qa")).toHaveTextContent("K01_V001,1000,năm người"),
    );
  });

  it("fills every generated Q&A row from one answer box", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByTestId("autogen-toggle"));
    fireEvent.change(screen.getByTestId("autogen-limit"), { target: { value: "3" } });
    await user.click(screen.getByTestId("autogen-start"));

    const block = await screen.findByTestId("submission-query-p1-2-qa");
    await waitFor(() =>
      expect(within(block).getAllByTestId(/^row-query-p1-2-qa-\d+$/)).toHaveLength(3),
    );
    // Generated frames carry no answer: the generator ranks where to look, the
    // text is a human judgement. One question-level error, not one per row.
    expect(within(block).getByTestId("qa-needs-answer-query-p1-2-qa")).toBeInTheDocument();
    expect(screen.getByTestId("export-blocked")).toHaveTextContent("missing answer");
    expect(screen.getByTestId("export-blocked")).toHaveTextContent("all 3 rows will be blocked");

    await user.type(within(block).getByTestId("qa-answer-input"), "năm người");
    await user.click(within(block).getByTestId("qa-answer-apply"));

    await waitFor(() =>
      expect(screen.getByTestId("csv-query-p1-2-qa")).toHaveTextContent("K01_V001,1000,năm người"),
    );
    // All three rows, and the export is no longer blocked by this question.
    expect(screen.getByTestId("csv-query-p1-2-qa").textContent?.match(/năm người/g)).toHaveLength(3);
    expect(within(block).queryByTestId("qa-needs-answer-query-p1-2-qa")).not.toBeInTheDocument();
  });

  it("imports a previously exported submission.zip back into the table", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));

    // A zip shaped exactly like the one the export button writes.
    const files = buildSubmissionFiles(
      [
        { id: "query-p1-1-kis", order: 1, kind: "kis", queryType: "T-KIS", text: "", eventCount: null },
        { id: "query-p1-2-qa", order: 2, kind: "qa", queryType: "QA", text: "", eventCount: null },
      ],
      [
        { id: "a", questionId: "query-p1-1-kis", videoId: "L26_V056", frames: [6400], answer: "" },
        { id: "b", questionId: "query-p1-2-qa", videoId: "L30_V072", frames: [1776], answer: "Giang Ly" },
      ],
    );
    const zip = new File(
      [buildZipBytes(files) as BlobPart],
      "submission.zip",
      { type: "application/zip" },
    );
    fireEvent.change(screen.getByTestId("answer-import-input"), { target: { files: [zip] } });

    // Nothing is written until the operator has seen the plan.
    const preview = await screen.findByTestId("answer-import-preview");
    expect(preview).toHaveTextContent("2 questions");
    expect(screen.getByTestId("answer-import-plan")).toHaveTextContent("Will save 2 rows");
    expect(screen.queryByTestId("row-query-p1-1-kis-0")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("answer-import-apply"));
    await waitFor(() =>
      expect(screen.getByTestId("csv-query-p1-1-kis")).toHaveTextContent("L26_V056,6400"),
    );
    expect(screen.getByTestId("csv-query-p1-2-qa")).toHaveTextContent("L30_V072,1776,Giang Ly");
  });

  it("refuses to destroy answers on import without being told to", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));

    // One answer already on screen for the question the archive also covers.
    await user.click(screen.getByText("query-p1-1-kis +"));
    const row = await screen.findByTestId("row-query-p1-1-kis-0");
    await user.type(within(row).getAllByRole("textbox")[0], "L30_V999");

    const files = buildSubmissionFiles(
      [{ id: "query-p1-1-kis", order: 1, kind: "kis", queryType: "T-KIS", text: "", eventCount: null }],
      [{ id: "a", questionId: "query-p1-1-kis", videoId: "L26_V056", frames: [6400], answer: "" }],
    );
    const zip = new File([buildZipBytes(files) as BlobPart], "submission.zip", { type: "application/zip" });
    fireEvent.change(screen.getByTestId("answer-import-input"), { target: { files: [zip] } });
    await screen.findByTestId("answer-import-preview");

    // Default mode leaves an answered question alone, so there is nothing to write.
    expect(screen.getByTestId("answer-import-plan")).toHaveTextContent("Will save 0 rows");
    expect(screen.getByTestId("answer-import-apply")).toBeDisabled();
    expect(screen.queryByTestId("answer-import-replace-warning")).not.toBeInTheDocument();

    // Choosing to replace states the cost first, in rows.
    await user.click(screen.getByTestId("answer-import-mode-replace"));
    expect(screen.getByTestId("answer-import-replace-warning")).toHaveTextContent("1 existing answer rows");
    await user.click(screen.getByTestId("answer-import-apply"));
    await waitFor(() =>
      expect(screen.getByTestId("csv-query-p1-1-kis")).toHaveTextContent("L26_V056,6400"),
    );
  });

  it("says when the zip is a question pack rather than answers", async () => {
    render(<App />);
    await importPack();
    fireEvent.click(screen.getByTestId("open-submission"));
    const zip = new File([buildZipBytes(PACK) as BlobPart], "pack.zip", { type: "application/zip" });
    fireEvent.change(screen.getByTestId("answer-import-input"), { target: { files: [zip] } });
    expect(await screen.findByTestId("answer-import-error")).toHaveTextContent("QUESTION pack");
  });

  it("exports generated answers in the rank the generator chose", async () => {
    // The whole algorithm is an ORDER — rank 1 scores the query, rank 100 scores
    // a fifth of it. A batch used to share one timestamp, so the CSV came out in
    // random-uuid order and the ranking was thrown away before anyone saw it.
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByTestId("autogen-toggle"));
    fireEvent.change(screen.getByTestId("autogen-limit"), { target: { value: "12" } });
    await user.click(screen.getByTestId("autogen-start"));

    const block = await screen.findByTestId("submission-query-p1-1-kis");
    // Only the first ten rows are on screen; the CSV below carries all twelve.
    await waitFor(() =>
      expect(within(block).getAllByTestId(/^row-query-p1-1-kis-\d+$/)).toHaveLength(10),
    );
    // The stub answers frame 1000 + rank*10, so rank order is frame order.
    const frames = screen
      .getByTestId("csv-query-p1-1-kis")
      .textContent!.trim()
      .split(/\r?\n/)
      .map((line) => Number(line.split(",")[1]));
    expect(frames).toEqual([...frames].sort((a, b) => a - b));
    expect(frames[0]).toBe(1000);
  });

  it("shows ten rows per question until asked for the rest", async () => {
    // 25 questions × 100 rows is one undifferentiated scroll: finding the next
    // question means paging past a hundred rows of the previous one.
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByTestId("autogen-toggle"));
    fireEvent.change(screen.getByTestId("autogen-limit"), { target: { value: "12" } });
    await user.click(screen.getByTestId("autogen-start"));

    const block = await screen.findByTestId("submission-query-p1-1-kis");
    const rows = () => within(block).getAllByTestId(/^row-query-p1-1-kis-\d+$/);
    await waitFor(() => expect(rows()).toHaveLength(10));

    const toggle = screen.getByTestId("expand-query-p1-1-kis");
    expect(toggle).toHaveTextContent("show all 12 rows");
    await user.click(toggle);
    expect(rows()).toHaveLength(12);

    await user.click(screen.getByTestId("expand-query-p1-1-kis"));
    expect(rows()).toHaveLength(10);
  });

  it("undoes a whole generation with one press, not one question at a time", async () => {
    // Regenerating is a delete plus an insert across every question in the pack.
    // Undo has to reverse the pair as one step, or the table keeps both sets.
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByTestId("autogen-toggle"));
    fireEvent.change(screen.getByTestId("autogen-limit"), { target: { value: "12" } });
    await user.click(screen.getByTestId("autogen-start"));

    const block = await screen.findByTestId("submission-query-p1-1-kis");
    await waitFor(() =>
      expect(within(block).getAllByTestId(/^row-query-p1-1-kis-\d+$/)).toHaveLength(10),
    );

    const undo = screen.getByTestId("submission-undo");
    expect(undo).toBeEnabled();
    await user.click(undo);

    await waitFor(() =>
      expect(screen.queryByTestId("submission-query-p1-1-kis")).not.toBeInTheDocument(),
    );
    expect(screen.getByTestId("undo-toast")).toHaveTextContent("regenerate");
    // Nothing left to undo, so the button goes back to being unavailable.
    expect(screen.getByTestId("submission-undo")).toBeDisabled();
  });

  it("does not leave the selection on a row that collapsing hid", async () => {
    // P / V / Delete all act on the selection, and Delete would drop a row the
    // operator can no longer see.
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByTestId("autogen-toggle"));
    fireEvent.change(screen.getByTestId("autogen-limit"), { target: { value: "12" } });
    await user.click(screen.getByTestId("autogen-start"));

    await screen.findByTestId("submission-query-p1-1-kis");
    await user.click(screen.getByTestId("expand-query-p1-1-kis"));
    const last = await screen.findByTestId("row-query-p1-1-kis-11");
    await user.click(last);
    expect(last.className).toContain("selected-row");

    await user.click(screen.getByTestId("expand-query-p1-1-kis"));
    expect(screen.queryByTestId("row-query-p1-1-kis-11")).not.toBeInTheDocument();
    expect(
      within(screen.getByTestId("submission-query-p1-1-kis")).queryAllByRole("row", {
        // nothing on screen is still marked as the selection
      }).filter((row) => row.className.includes("selected-row")),
    ).toHaveLength(0);
  });

  it("keeps a comma while a multi-event TRAKE row is being typed", async () => {
    // Regression: the frame cell used to be controlled straight off number[],
    // so "1200," re-rendered as "1200" and a second event could never be typed.
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByText("query-p1-3-trake +"));

    const row = await screen.findByTestId("row-query-p1-3-trake-0");
    const frameCell = within(row).getAllByRole("textbox")[1];
    await user.type(frameCell, "1200, 1850");

    expect(frameCell).toHaveValue("1200, 1850");
    // Nothing has been sent yet: typing edits a local draft, and the table the
    // team shares still holds the previous value.
    expect(screen.getByTestId("csv-query-p1-3-trake")).not.toHaveTextContent(",1200,1850");
    expect(screen.getByTestId("unsent-banner")).toBeInTheDocument();

    await user.type(frameCell, "{Enter}");
    expect(screen.getByTestId("csv-query-p1-3-trake")).toHaveTextContent(",1200,1850");
    expect(screen.queryByTestId("unsent-banner")).not.toBeInTheDocument();
  });

  it("sends a hand-typed row once, on confirm, not on every keystroke", async () => {
    // Five operators share this table. Writing through on each character made
    // "L30_V046" eight writes, and because the cell was driven off the shared
    // row, anything arriving from Postgres mid-word replaced what was on screen
    // and ate the rest of it.
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByText("query-p1-1-kis +"));

    const row = await screen.findByTestId("row-query-p1-1-kis-0");
    await user.type(within(row).getAllByRole("textbox")[0], "L30_V046");
    await user.type(within(row).getAllByRole("textbox")[1], "6644");

    // Still local: the shared table has neither value.
    const csv = () => screen.getByTestId("csv-query-p1-1-kis").textContent ?? "";
    expect(csv()).not.toContain("L30_V046");
    expect(screen.getByTestId("unsent-banner")).toHaveTextContent("1 rows");

    // One confirm carries the whole row — video id and frames together, rather
    // than racing each other into the table as separate writes.
    await user.click(screen.getByTestId("commit-query-p1-1-kis-0"));
    expect(csv()).toContain("L30_V046,6644");
    expect(screen.queryByTestId("unsent-banner")).not.toBeInTheDocument();
  });

  it("throws the draft away on Escape", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByText("query-p1-1-kis +"));

    const row = await screen.findByTestId("row-query-p1-1-kis-0");
    const videoCell = within(row).getAllByRole("textbox")[0];
    await user.type(videoCell, "L30_V04");
    expect(screen.getByTestId("unsent-banner")).toBeInTheDocument();

    await user.type(videoCell, "{Escape}");
    expect(videoCell).toHaveValue("");
    expect(screen.queryByTestId("unsent-banner")).not.toBeInTheDocument();
  });

  it("confirms every pending row from the banner", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByText("query-p1-1-kis +"));
    // The "+" shortcut only exists while a question is unanswered; a second row
    // comes from the block's own button.
    const block = await screen.findByTestId("submission-query-p1-1-kis");
    await user.click(within(block).getByText("+ row"));

    const first = await screen.findByTestId("row-query-p1-1-kis-0");
    const second = await screen.findByTestId("row-query-p1-1-kis-1");
    await user.type(within(first).getAllByRole("textbox")[0], "L30_V046");
    await user.type(within(second).getAllByRole("textbox")[0], "L21_V003");
    expect(screen.getByTestId("unsent-banner")).toHaveTextContent("2 rows");

    await user.click(screen.getByTestId("commit-all"));
    const csv = screen.getByTestId("csv-query-p1-1-kis").textContent ?? "";
    expect(csv).toContain("L30_V046");
    expect(csv).toContain("L21_V003");
    expect(screen.queryByTestId("unsent-banner")).not.toBeInTheDocument();
  });

  it("undo after a regeneration gives the hand-typed rows back", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByText("query-p1-1-kis +"));
    const block = await screen.findByTestId("submission-query-p1-1-kis");
    await user.click(within(block).getByText("+ row"));

    const first = await screen.findByTestId("row-query-p1-1-kis-0");
    const second = await screen.findByTestId("row-query-p1-1-kis-1");
    await user.type(within(first).getAllByRole("textbox")[0], "L21_V005");
    await user.type(within(first).getAllByRole("textbox")[1], "2315{Enter}");
    await user.type(within(second).getAllByRole("textbox")[0], "L30_V046");
    await user.type(within(second).getAllByRole("textbox")[1], "6644{Enter}");
    expect(screen.getByTestId("csv-query-p1-1-kis")).toHaveTextContent("L21_V005,2315");

    await user.click(screen.getByTestId("autogen-one-query-p1-1-kis"));
    await waitFor(() => expect(answerGenRequests).toHaveLength(1));
    await waitFor(() =>
      expect(within(block).getAllByTestId(/^row-query-p1-1-kis-\d+$/).length).toBeGreaterThan(2),
    );

    await user.click(screen.getByTestId("submission-undo"));

    const csv = screen.getByTestId("csv-query-p1-1-kis").textContent ?? "";
    expect(csv).toContain("L21_V005,2315");
    expect(csv).toContain("L30_V046,6644");
    expect(csv.trim().split(/\r?\n/)).toHaveLength(2);
  });

  it("a regeneration that produces nothing must not let Ctrl+Z hit an older action", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByText("query-p1-1-kis +"));
    const block = await screen.findByTestId("submission-query-p1-1-kis");
    await user.click(within(block).getByText("+ row"));

    const first = await screen.findByTestId("row-query-p1-1-kis-0");
    const second = await screen.findByTestId("row-query-p1-1-kis-1");
    await user.type(within(first).getAllByRole("textbox")[0], "L21_V005");
    await user.type(within(first).getAllByRole("textbox")[1], "2315{Enter}");
    await user.type(within(second).getAllByRole("textbox")[0], "L30_V046");
    await user.type(within(second).getAllByRole("textbox")[1], "6644{Enter}");

    // Budget = limit - kept = 0, so the generator is skipped and the question
    // ends the run exactly as it started.
    await user.click(screen.getByTestId("autogen-toggle"));
    fireEvent.change(screen.getByTestId("autogen-limit"), { target: { value: "2" } });
    await user.click(screen.getByTestId("autogen-one-query-p1-1-kis"));
    await waitFor(() =>
      expect(within(block).getAllByTestId(/^row-query-p1-1-kis-\d+$/)).toHaveLength(2),
    );

    await user.click(screen.getByTestId("submission-undo"));

    // The hand-typed values must still be there: the press cannot silently
    // reach past the action the operator just took and blank a row instead.
    const csv = screen.getByTestId("csv-query-p1-1-kis").textContent ?? "";
    expect(csv).toContain("L21_V005,2315");
    expect(csv).toContain("L30_V046,6644");
    // And it says why nothing came back, rather than looking like a dud press.
    expect(screen.getByTestId("undo-toast")).toHaveTextContent("made no changes");
  });

  it("can get back to the search UI and import from the submission view", async () => {
    // The bar holding "Import questions" lives inside the console, which the
    // submission view does not render — so that view needs its own way out and
    // its own import button, or it is a dead end on first launch.
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByTestId("open-submission"));
    expect(await screen.findByTestId("submission-panel")).toBeInTheDocument();
    expect(screen.queryByTestId("query-input")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("submission-back"));
    expect(screen.getByTestId("query-input")).toBeInTheDocument();

    // …and the empty state's instruction points at a button that is really there.
    await user.click(screen.getByTestId("open-submission"));
    const bytes = buildZipBytes(PACK);
    const file = new File([bytes as BlobPart], "pack.zip", { type: "application/zip" });
    fireEvent.change(screen.getByTestId("submission-import-input"), { target: { files: [file] } });
    await screen.findByTestId("pack-preview");
    fireEvent.click(screen.getByTestId("pack-publish"));
    await waitFor(() => expect(screen.getAllByTestId(/^rail-tab-\d+$/)).toHaveLength(PACK.length));
  });

  it("returns to the tab the operator came from, not always the first", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await openTab(user, 2); // the TRAKE tab
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByTestId("submission-back"));

    expect(screen.getByRole("tab", { name: "TRAKE", selected: true })).toBeInTheDocument();
  });

  it("keeps the rail addressable while it is collapsed", async () => {
    // The rail collapses to a two-letter code and expands on hover, so every tab
    // carries a spoken name of its own (the visible code is aria-hidden) and the
    // widening panel is a separate layer that can overlay the console.
    const user = userEvent.setup();
    render(<App />);
    await importPack();

    expect(document.querySelector(".tab-rail-inner")).toBeInTheDocument();
    const trakeTab = screen.getByTestId("rail-tab-2");
    expect(trakeTab).toHaveAccessibleName("Tab 3 · TRAKE · question 3");
    expect(within(trakeTab).getByText("TR")).toHaveAttribute("aria-hidden", "true");

    // Collapsed or not, clicking still switches tabs.
    await user.click(trakeTab);
    expect(trakeTab).toHaveAttribute("aria-selected", "true");
  });

  it("keeps the table textual and only loads a picture when P is pressed", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await waitFor(() => screen.getByTestId("detail-panel"));
    await user.click(screen.getByTestId("open-submit"));
    await user.click(screen.getByTestId("confirm-submit"));
    await user.click(screen.getByTestId("open-submission"));

    // Five clients syncing text must not turn into five clients fetching media:
    // the table shows ids only until somebody asks for a frame.
    const panel = await screen.findByTestId("submission-panel");
    expect(within(panel).queryByRole("img")).not.toBeInTheDocument();
    expect(keyframeLookups).toEqual([]);

    await user.click(screen.getByTestId("row-query-p1-1-kis-0"));
    fireEvent.keyDown(window, { key: "P" });

    const preview = await screen.findByTestId("frame-preview");
    await waitFor(() => expect(keyframeLookups).toEqual(["K01/K01_V001/001"]));
    expect(within(preview).getByRole("img")).toHaveAttribute(
      "src",
      "https://media.test/Keyframes/K01/K01_V001/001.jpg",
    );

    fireEvent.keyDown(window, { key: "P" });
    await waitFor(() => expect(screen.queryByTestId("frame-preview")).not.toBeInTheDocument());
  });

  it("says so instead of faking a picture for a raw video frame", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await waitFor(() => screen.getByTestId("detail-panel"));

    // Submit the paused RAW frame: it has no extracted keyframe behind it.
    (document.activeElement as HTMLElement)?.blur();
    fireEvent.keyDown(window, { key: "v" });
    const video = (await screen.findByTestId("video-viewer")) as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 5.2 });
    fireEvent.pause(video);
    await screen.findByTestId("paused-frame-chip");
    await user.click(screen.getByTestId("submit-paused-frame"));
    await user.click(screen.getByTestId("confirm-submit"));

    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByTestId("row-query-p1-1-kis-0"));
    fireEvent.keyDown(window, { key: "P" });

    expect(await screen.findByTestId("preview-raw")).toHaveTextContent("RAW VIDEO FRAME");
    expect(keyframeLookups).toEqual([]);
  });

  it("re-picks a frame from the video and writes it back to the row", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await waitFor(() => screen.getByTestId("detail-panel"));
    await user.click(screen.getByTestId("open-submit"));
    await user.click(screen.getByTestId("confirm-submit"));
    await user.click(screen.getByTestId("open-submission"));
    expect(await screen.findByTestId("csv-query-p1-1-kis")).toHaveTextContent("K01_V001,0");

    await user.click(screen.getByTestId("row-query-p1-1-kis-0"));
    fireEvent.keyDown(window, { key: "V" });
    const editor = await screen.findByTestId("frame-editor");

    const video = (await within(editor).findByTestId("video-viewer")) as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 5.2 });
    fireEvent.pause(video);

    // The draft stays local until the commit button — otherwise every scrub
    // would jump the row on four other screens.
    expect(within(editor).getByTestId("editor-candidate")).toHaveTextContent("frame 130");
    expect(screen.getByTestId("csv-query-p1-1-kis")).toHaveTextContent("K01_V001,0");

    await user.click(within(editor).getByTestId("editor-commit"));
    await waitFor(() =>
      expect(screen.getByTestId("csv-query-p1-1-kis")).toHaveTextContent("K01_V001,130"),
    );
  });

  it("scrubs the editor video with the same keys as the search console", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await waitFor(() => screen.getByTestId("detail-panel"));
    await user.click(screen.getByTestId("open-submit"));
    await user.click(screen.getByTestId("confirm-submit"));
    await user.click(screen.getByTestId("open-submission"));
    await screen.findByTestId("csv-query-p1-1-kis");
    await user.click(screen.getByTestId("row-query-p1-1-kis-0"));
    fireEvent.keyDown(window, { key: "V" });

    const editor = await screen.findByTestId("frame-editor");
    const video = (await within(editor).findByTestId("video-viewer")) as HTMLVideoElement;
    video.currentTime = 30;

    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(video.currentTime).toBeCloseTo(35, 3);
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(video.currentTime).toBeCloseTo(30, 3);
    fireEvent.keyDown(window, { key: "d" });
    expect(video.currentTime).toBeCloseTo(31, 3);
    fireEvent.keyDown(window, { key: "a" });
    expect(video.currentTime).toBeCloseTo(30, 3);

    video.currentTime = 2;
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(video.currentTime).toBe(0);

    // The modal's player has the same shadow-DOM controls, so the key must not
    // reach it here either — otherwise the browser adds its own seek to ours.
    const reachedVideo = vi.fn();
    video.addEventListener("keydown", reachedVideo);
    fireEvent.keyDown(video, { key: "ArrowRight" });
    fireEvent.keyDown(video, { key: "d" });
    expect(reachedVideo).not.toHaveBeenCalled();
    video.removeEventListener("keydown", reachedVideo);

    // Closing the editor hands the arrows back to the row list, which is what
    // they mean everywhere else in the submission tab.
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(screen.queryByTestId("frame-editor")).not.toBeInTheDocument());
    expect(() => fireEvent.keyDown(window, { key: "ArrowDown" })).not.toThrow();
  });

  it("can keep the original answer and add the re-picked frame as a second row", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await waitFor(() => screen.getByTestId("detail-panel"));
    await user.click(screen.getByTestId("open-submit"));
    await user.click(screen.getByTestId("confirm-submit"));
    await user.click(screen.getByTestId("open-submission"));

    await user.click(screen.getByTestId("row-query-p1-1-kis-0"));
    fireEvent.keyDown(window, { key: "V" });
    const editor = await screen.findByTestId("frame-editor");
    const video = (await within(editor).findByTestId("video-viewer")) as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 5.2 });
    fireEvent.pause(video);

    await user.click(within(editor).getByTestId("editor-commit-new"));

    // Both guesses survive: the original stays put, the new frame is appended.
    await waitFor(() =>
      expect(screen.getByTestId("csv-query-p1-1-kis")).toHaveTextContent("K01_V001,0"),
    );
    expect(screen.getByTestId("csv-query-p1-1-kis")).toHaveTextContent("K01_V001,130");
    expect(screen.getByTestId("row-query-p1-1-kis-1")).toBeInTheDocument();
  });

  it("searches every question on demand, for a machine that only received the pack", async () => {
    // The machine that imports a pack gets its tabs opened and searched as part
    // of applying it. A machine that received the same pack over realtime never
    // ran that step, so it needs a way to reach the same state.
    const user = userEvent.setup();
    render(<App />);
    await importPack();

    // Collapse back to a single idle tab, the way a sync-only machine starts.
    await user.click(screen.getByTestId("rail-close-2"));
    await user.click(screen.getByTestId("rail-close-1"));
    expect(screen.getAllByTestId(/^rail-tab-\d+$/)).toHaveLength(1);
    const before = vi.mocked(fetch).mock.calls.filter(([url]) =>
      String(url).endsWith("/api/search"),
    ).length;

    await user.click(screen.getByTestId("search-all"));

    await waitFor(() => expect(screen.getAllByTestId(/^rail-tab-\d+$/)).toHaveLength(PACK.length));
    await waitFor(() => {
      const searched = vi.mocked(fetch).mock.calls
        .filter(([url]) => String(url).endsWith("/api/search"))
        .map(([, init]) => JSON.parse(String(init?.body)).query);
      expect(searched).toContain(PACK[0].text);
      expect(searched).toContain(PACK[1].text);
      expect(searched.length).toBeGreaterThan(before);
    });
  });

  it("offers nothing to search before a pack exists", async () => {
    render(<App />);
    expect(screen.getByTestId("search-all")).toBeDisabled();
    expect(screen.getByTestId("search-all")).toHaveTextContent("(0)");
  });

  it("offers only publish or cancel after parsing a pack", async () => {
    // A per-machine import is gone on purpose: it was the way two machines
    // could end up answering different questions under the same question_id.
    render(<App />);
    const bytes = buildZipBytes(PACK);
    const file = new File([bytes as BlobPart], "pack.zip", { type: "application/zip" });
    fireEvent.change(screen.getByTestId("import-input"), { target: { files: [file] } });

    const preview = await screen.findByTestId("pack-preview");
    expect(within(preview).getByTestId("pack-publish")).toBeEnabled();
    expect(within(preview).getByTestId("pack-cancel")).toBeInTheDocument();
    expect(screen.queryByTestId("pack-local")).not.toBeInTheDocument();
    expect(preview).toHaveTextContent("3 questions");

    // Cancelling leaves the previous pack (here: none) untouched.
    fireEvent.click(within(preview).getByTestId("pack-cancel"));
    await waitFor(() => expect(screen.queryByTestId("pack-preview")).not.toBeInTheDocument());
    expect(screen.getAllByTestId(/^rail-tab-\d+$/)).toHaveLength(1);
  });

  it("adds and closes tabs", async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(screen.getAllByTestId(/^rail-tab-\d+$/)).toHaveLength(1);
    await user.click(screen.getByTestId("rail-add"));
    expect(screen.getAllByTestId(/^rail-tab-\d+$/)).toHaveLength(2);
    await user.click(screen.getByTestId("rail-close-1"));
    expect(screen.getAllByTestId(/^rail-tab-\d+$/)).toHaveLength(1);
  });

  it("submits live to DRES by default once the server is configured", async () => {
    dresConfigured = true;
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByTestId("dres-bar")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /direct dres submission/i })).toBeChecked();
    expect(screen.queryByTestId("question-bar")).not.toBeInTheDocument();

    // Unticking it brings the preliminary round's query-pack workflow back.
    await user.click(screen.getByRole("checkbox", { name: /direct dres submission/i }));
    expect(await screen.findByTestId("question-bar")).toBeInTheDocument();
    expect(screen.queryByTestId("dres-bar")).not.toBeInTheDocument();
  });

  it("never posts to DRES once unticked: a submit writes a row into the submission table", async () => {
    dresConfigured = true;
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    // The switch belongs to each search tab; untick it on the question's tab.
    await user.click(await screen.findByRole("checkbox", { name: /direct dres submission/i }));
    await waitFor(() => screen.getByTestId("detail-panel"));

    await user.click(screen.getByTestId("open-submit"));
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());
    expect(screen.getByTestId("guard-question")).toHaveTextContent("query-p1-1-kis.csv");
    expect(screen.getByTestId("guard-csv-line")).toHaveTextContent("K01_V001,0");
    await user.click(screen.getByTestId("confirm-submit"));

    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith("/api/submit"))).toBe(false);
    await user.click(screen.getByTestId("open-submission"));
    expect(await screen.findByTestId("csv-query-p1-1-kis")).toHaveTextContent("K01_V001,0");
  });

  it("blocks a submit from a tab with no question bound", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("query-input"), "thời sự");
    await user.click(screen.getByTestId("search-btn"));
    await waitFor(() => screen.getByTestId("detail-panel"));

    await user.click(screen.getByTestId("open-submit"));
    await waitFor(() => expect(screen.getByTestId("submit-guard")).toBeInTheDocument());
    expect(screen.getByTestId("guard-format-error")).toHaveTextContent(/no question assigned/i);
    expect(screen.getByTestId("confirm-submit")).toBeDisabled();
  });

  it("keeps several answers per question and exports them as one CSV", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await waitFor(() => screen.getByTestId("detail-panel"));

    await user.click(screen.getByTestId("open-submit"));
    await user.click(screen.getByTestId("confirm-submit"));
    // Move to the second retrieved frame and submit that one too.
    await user.click(screen.getAllByTestId("frame-thumb")[1]);
    await user.click(screen.getByTestId("open-submit"));
    await user.click(screen.getByTestId("confirm-submit"));

    await user.click(screen.getByTestId("open-submission"));
    const csv = await screen.findByTestId("csv-query-p1-1-kis");
    expect(csv.textContent).toBe("K01_V001,0\r\nK01_V001,450");
  });

  it("keeps a bad cell editable but refuses to export it", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await waitFor(() => screen.getByTestId("detail-panel"));
    await user.click(screen.getByTestId("open-submit"));
    await user.click(screen.getByTestId("confirm-submit"));
    await user.click(screen.getByTestId("open-submission"));

    const row = await screen.findByTestId("row-query-p1-1-kis-0");
    const videoCell = within(row).getAllByRole("textbox")[0];
    await user.clear(videoCell);
    await user.type(videoCell, "L01_V028.mp4{Enter}");

    // The cell still accepts what was typed — nothing is locked mid-edit…
    expect(await screen.findByTestId("csv-query-p1-1-kis")).toHaveTextContent("L01_V028.mp4,0");
    expect(screen.getByTestId("errors-query-p1-1-kis")).toBeInTheDocument();
    // …but a name the organiser's parser rejects must never reach a zip, and a
    // rejected upload still costs one of the three attempts.
    expect(screen.getByTestId("export-blocked")).toHaveTextContent(".mp4");
    expect(screen.getByTestId("export-submission")).toBeDisabled();
  });

  it("checks the TRAKE event count of a row against its question", async () => {
    const user = userEvent.setup();
    render(<App />);
    await importPack();
    await openTab(user, 2); // the 2-event TRAKE question
    await user.click(screen.getByTestId("open-submission"));

    // A blank row starts with no frames at all, which is already the wrong count.
    await user.click(screen.getByText("query-p1-3-trake +"));
    const row = await screen.findByTestId("row-query-p1-3-trake-0");
    const frameCell = within(row).getAllByRole("textbox")[1];
    await user.type(frameCell, "1200, 900{Enter}");

    expect(screen.getByTestId("errors-query-p1-3-trake")).toBeInTheDocument();
    expect(screen.getByTestId("submission-query-p1-3-trake")).toHaveTextContent(/increasing chronological order/);
    expect(screen.getByTestId("export-submission")).toBeDisabled();
  });

  it("exports a zip with the CSVs under submission/", async () => {
    const user = userEvent.setup();
    const created: Blob[] = [];
    const createObjectURL = vi.fn((blob: Blob) => {
      created.push(blob);
      return "blob:submission";
    });
    vi.stubGlobal("URL", { ...URL, createObjectURL, revokeObjectURL: vi.fn() });

    render(<App />);
    await importPack();
    await waitFor(() => screen.getByTestId("detail-panel"));
    await user.click(screen.getByTestId("open-submit"));
    await user.click(screen.getByTestId("confirm-submit"));
    await user.click(screen.getByTestId("open-submission"));
    await user.click(screen.getByTestId("export-submission"));

    expect(created).toHaveLength(1);
    const text = new TextDecoder().decode(await created[0].arrayBuffer());
    expect(text).toContain("submission/query-p1-1-kis.csv");
    expect(text).toContain("K01_V001,0");
  });
});
