// Types mirroring the backend API contract (see backend/README and main.py).

export type Channel =
  | "image_pe" | "ocr" | "speech" | "audio" | "similar" | "object_layout" | "canvas_image";
export type QueryType = "T-KIS" | "QA" | "V-KIS" | "TRAKE";
export type QueryTypeHint = "auto" | QueryType;

export interface Evidence {
  type: Channel;
  score: number;
  text?: string;
  start?: number;
  end?: number;
  [k: string]: unknown;
}

export interface FrameResult {
  image_id: string;
  submit_keyframe_id: string;
  video_id: string;
  keyframe_n: number;
  frame_idx: number | null;
  fps: number | null;
  pts_time: number | null;
  score: number;
  channels: Channel[];
  per_channel_score: Record<string, number>;
  keyframe_url: string;
  video_url: string;
  evidence: Evidence[];
}

export interface VideoGroup {
  video_id: string;
  video_score: number;
  max_score: number;
  mean_top_score: number;
  frame_count: number;
  timestamp_dispersion: number;
  ambiguous: boolean;
  channels: Channel[];
  video_url: string;
  frames: FrameResult[];
  // TRAKE-only (set when a sequence is mapped into a group): events backed by real
  // retrieval vs filled by pass-2 in-video search, and mean relevance of real frames.
  trake_confident?: number;
  trake_filled?: number;
  trake_mean?: number;
}

export interface ChannelConfig {
  enabled: boolean;
  weight: number;
  reason?: string;
  queries_en?: string[];
  queries_vi?: string[];
  queries_folded?: string[];
  sound_labels_en?: string[];
  time_filters?: { hour: number | null; clock: string | null };
}

export interface ParsedQuery {
  query_type: QueryType;
  confidence: number;
  original_query: string;
  normalized_vi: string;
  translated_en_visual: string;
  operator_summary_vi: string;
  channels: Record<Channel, ChannelConfig>;
  filters: {
    video_ids: string[];
    categories: string[];
    must_include: string[];
    must_not_include: string[];
  };
  trake: {
    enabled: boolean;
    events: TrakeEventSpec[];
    ordering_rule: string;
    fallback_policy: string;
  };
  qa: {
    enabled: boolean;
    question_vi: string;
    draft_answer_allowed_from_text_evidence_only: boolean;
  };
  ui_hints: {
    show_timeline: boolean;
    show_ocr_snippets: boolean;
    show_speech_snippets: boolean;
    show_audio_badges: boolean;
    warning_vi: string;
  };
  _engine?: string;
}

export interface TrakeEventSpec {
  event_index: number;
  description_vi: string;
  description_en_visual?: string;
  image_pe_queries_en?: string[];
}

export interface LatencyBreakdown {
  parse_ms?: number;
  fusion_ms?: number;
  total_ms?: number;
  channels?: Record<string, number>;
}

export interface SearchResponse {
  query: string;
  parsed: ParsedQuery;
  groups: VideoGroup[];
  latency_ms: LatencyBreakdown;
  warnings?: string[];
  mode: "mock" | "live";
}

// ---- V-KIS canvas ----------------------------------------------------------

export interface CanvasObjectSpec {
  id: string;
  label: string;
  /** normalized [x1, y1, x2, y2] on the canvas */
  bbox: [number, number, number, number];
  color: string | null;
  required: boolean;
}

export type CanvasMode = "rough" | "precise";

export interface CanvasSpec {
  objects: CanvasObjectSpec[];
  exclude_labels?: string[];
  action_text?: string;
  mode: CanvasMode;
  /** Rendered drawing as a PNG data URL, for the low-weight PE image channel. */
  image?: string | null;
}

/** One drawn object paired with the detection it matched, for the overlay. */
export interface CanvasMatch {
  object_id: string;
  label: string;
  score: number;
  detection_label: string | null;
  conf: number;
  bbox_norm: { x1: number; y1: number; x2: number; y2: number };
  position: string | null;
  dominant_color: string | null;
  color_reliable: boolean;
  color_ok: boolean | null;
}

/** `object_layout` evidence carried on a frame returned by the canvas search. */
export interface CanvasLayoutEvidence extends Evidence {
  type: "object_layout";
  matches: CanvasMatch[];
  missing: string[];
  coverage: number;
  excluded_hits: string[];
}

export interface CanvasSearchResponse {
  canvas: Omit<CanvasSpec, "image"> & { queries_en: string[]; has_image: boolean };
  groups: VideoGroup[];
  latency_ms: LatencyBreakdown;
  warnings: string[];
  mode: "mock" | "live";
}

export interface CanvasPalette {
  colors: { name: string; hex: string }[];
  labels: { label: string; label_vi: string | null; colorable: boolean }[];
  modes: CanvasMode[];
  color_palette_version: string;
}

export interface QaAnalyzeCandidate {
  submit_keyframe_id: string;
  frame_idx: number | null;
  pts_time: number | null;
  retrieval_score: number;
  evidence: Evidence[];
}

export interface QaSupportingFrame {
  candidate_id: string;
  submit_keyframe_id: string;
  video_id: string;
  frame_idx: number | null;
  pts_time: number | null;
  keyframe_url: string;
}

export interface QaCandidateAnswer {
  answer: string;
  confidence: number;
  supporting_candidate_ids: string[];
  supporting_frames: QaSupportingFrame[];
  reason: string;
  source: "nvila" | "web" | "hybrid" | string;
  web_sources: QaWebSource[];
  visual_verification?: QaVisualVerificationVerdict;
}

export interface QaWebSource {
  title: string;
  url: string;
}

export interface QaVisualVerificationVerdict {
  answer: string;
  status: "supported" | "contradicted" | "insufficient" | "unverified";
  visual_confidence: number;
  supporting_candidate_ids: string[];
  reason: string;
}

export interface QaWebVisualVerification {
  attempted: boolean;
  used: boolean;
  model: string | null;
  latency_ms: number;
  verdicts: QaVisualVerificationVerdict[];
  rejected_answers: string[];
  uncertainty: string;
}

export interface QaWebGrounding {
  requested: "auto" | "on" | "off";
  available: boolean;
  attempted: boolean;
  used: boolean;
  model: string | null;
  queries: string[];
  sources: QaWebSource[];
  summary: string;
  latency_ms: number;
  search_suggestions_html: string;
  visual_verification?: QaWebVisualVerification;
}

export interface QaHotspot extends QaSupportingFrame {
  keyframe_n: number;
  relevance: number;
  answer_support: string;
}

export interface QaAnalysisResponse {
  question: string;
  model: string;
  mode: "mock" | "live" | string;
  answerable: boolean;
  best_answer: string;
  best_candidate_id: string | null;
  best_submit_keyframe_id: string | null;
  candidate_answers: QaCandidateAnswer[];
  hotspots: QaHotspot[];
  uncertainty: string;
  candidate_count: number;
  latency_ms: number;
  total_latency_ms?: number;
  cached: boolean;
  warnings: string[];
  web_grounding?: QaWebGrounding;
}

export interface TrakeSequenceFrame {
  event_index: number;
  submit_keyframe_id: string;
  keyframe_n: number;
  frame_idx: number | null;
  pts_time: number;
  score: number;
  keyframe_url: string;
  via_fill?: boolean;
}

export interface TrakeSequence {
  video_id: string;
  score: number;
  complete: boolean;
  coverage: number;
  confident_coverage?: number;
  filled_events?: number;
  mean_score?: number;
  min_score?: number;
  warning: string | null;
  frames: TrakeSequenceFrame[];
  video_url: string;
}

export interface TrakeSearchResponse {
  query: string;
  parsed: ParsedQuery;
  events: { event_index: number; description_vi: string; candidate_count: number }[];
  sequences: TrakeSequence[];
  mode: string;
}

export interface TimelineKeyframe {
  submit_keyframe_id: string;
  keyframe_n: number;
  frame_idx: number | null;
  pts_time: number | null;
  keyframe_url: string;
}

export interface Timeline {
  video_id: string;
  fps: number;
  duration: number | null;
  video_url: string;
  keyframes: TimelineKeyframe[];
}

export interface SnapResult {
  video_id: string;
  fps: number;
  submit_keyframe_id: string;
  keyframe_n: number;
  frame_idx: number;
  pts_time: number;
  raw_time: number;
  raw_frame_idx: number;
  delta_frames: number;
  delta_seconds: number;
  far: boolean;
  keyframe_url: string;
}

export interface SimpleResult {
  image_id: string;
  submit_keyframe_id: string;
  video_id: string;
  keyframe_n: number;
  score: number;
  keyframe_url: string;
  video_url: string;
}

export interface SimpleSearchResponse {
  query: string;
  translated_query?: string | null;
  results: SimpleResult[];
  mode: "mock" | "live";
  latency_ms: number;
}

export interface HealthResponse {
  ok: boolean;
  mode: "mock" | "live";
  services: Record<string, { ok: boolean; error?: string }>;
  capabilities: Record<string, boolean>;
  warnings: string[];
}

// ---- DRES (official evaluation server, Client API v2) ----------------------

/** One answer as DRES v2 wants it: text (QA) or media item + ms window (KIS). */
export interface DresAnswer {
  text?: string;
  mediaItemName?: string;
  mediaItemCollectionName?: string;
  start?: number;
  end?: number;
}

export interface DresAnswerSet {
  taskName?: string;
  answers: DresAnswer[];
}

export interface DresTaskTemplate {
  name: string;
  taskGroup: string;
  taskType: string;
  duration: number | null;
}

export interface DresEvaluationState {
  evaluationStatus: string;
  taskStatus: "NO_TASK" | "CREATED" | "PREPARING" | "RUNNING" | "ENDED" | "IGNORED";
  /** Identifies the open task; changes when the organisers move to the next one. */
  taskTemplateId?: string | null;
  /** Seconds left in the open task; null for open-ended tasks. */
  timeLeft: number | null;
  timeElapsed: number;
}

/** The task statement DRES shows the team: text (T-KIS/QA) and/or media (V-KIS). */
export interface DresTaskHint {
  evaluation_id: string;
  task_template_id: string | null;
  task_name: string | null;
  task_type: string | null;
  task_status: string | null;
  text: string;
  elements: { content_type: "TEXT" | "IMAGE" | "VIDEO"; content: string; offset: number }[];
  loop?: boolean;
  warnings: string[];
}

export interface DresEvaluation {
  id: string;
  name: string;
  status: string;
  type: string;
  teams: string[];
  task_templates: DresTaskTemplate[];
  current_task: DresTaskTemplate | null;
  state: DresEvaluationState | null;
}

export interface DresStatus {
  configured: boolean;
  base_url: string;
  logged_in: boolean;
  username: string | null;
  user: { id?: string; username?: string; role?: string } | null;
  pinned_evaluation_id: string | null;
  segment_pad_ms: number;
  error: string | null;
}

export type AnswerMode = "auto" | "temporal" | "temporal_text" | "text" | "item";

/** The exact request the backend would send to DRES for the current pick. */
export interface SubmitPreview {
  evaluation_id: string | null;
  task_name: string | null;
  /** DRES's own task type ("Textual KIS", "Question Answering", …). */
  task_type: string | null;
  task_id: string;
  answer_mode: Exclude<AnswerMode, "auto">;
  /** True when the answer shape contradicts the task type — a silent wrong submit. */
  answer_mode_mismatch: boolean;
  segment_pad_ms: number;
  body: { answerSets: DresAnswerSet[] };
  duplicate_key: string | null;
  url: string | null;
  warnings: string[];
}

export interface SubmitEntry {
  id: string;
  ts: number;
  task_id: string;
  evaluation_id?: string | null;
  task_name?: string | null;
  query_type: QueryType;
  payload: {
    video_id?: string;
    frame_idx?: number | null;
    events?: { event_index: number; frame_idx: number }[];
    answer?: string;
    submit_keyframe_id?: string;
  };
  answer_sets?: DresAnswerSet[];
  dedup_keys: string[];
  status: string;
  /** DRES verdict when the server judged it: CORRECT | WRONG | INDETERMINATE | UNDECIDABLE. */
  verdict?: string | null;
  dres?: { description?: string; error?: string } | null;
  was_duplicate: boolean;
}

export interface FeedbackState {
  /** Soft multiplier on the aggregate video score. */
  positive_videos: string[];
  negative_videos: string[];
  /** Seeds the image-to-image `similar` retrieval channel. */
  positive_frames: string[];
  negative_frames: string[];
}
