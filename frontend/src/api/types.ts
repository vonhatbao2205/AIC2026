// Types mirroring the backend API contract (see backend/README and main.py).

export type Channel =
  | "image_pe" | "image_qwen" | "image_visual" | "tara" | "ocr" | "speech" | "audio" | "similar"
  | "canvas_image";
export type QueryType = "T-KIS" | "QA" | "V-KIS" | "TRAKE" | "AVS";
export type QueryTypeHint = "auto" | QueryType;
export type RetrievalDatabase = "btc" | "infoshotpp";
/** Image-vector indices that can answer an InfoShot++ search. BTC remains PE-only. */
export type ImageEmbeddingModel = "pe" | "qwen3_vl";

// ---- search scope (which dataset folders a search may return) --------------

/** `all` = whole profile, `manual` = exactly the ticked folders, `auto` = let the
 *  backend's topic heuristic read them off the query. */
export type ScopeMode = "all" | "auto" | "manual";

export interface SearchScope {
  mode: ScopeMode;
  categories: string[];
}

export interface ScopeTopicMatch {
  topic_id: string;
  label_vi: string;
  label_en?: string;
  keywords: string[];
  categories: string[];
}

/** What a search actually applied, echoed back on every search response. */
export interface ResolvedScope {
  mode: ScopeMode;
  categories: string[];
  /** The narrower set the matched topics alone imply, offered as one click. */
  strict_categories: string[];
  active: boolean;
  reason_vi: string;
  reason_en?: string;
  matched_topics: ScopeTopicMatch[];
}

// ---- PE-Core text window ------------------------------------------------------

/** One text as PE-Core's tokenizer sees it. PE reads `limit` (70) tokens and
 *  drops the rest without an error. */
export interface PeTokenCount {
  text: string;
  tokens: number;
  limit: number;
  truncated: boolean;
  kept: string;
  dropped: string;
}

/** Exact counts of every query a search sent to the PE text encoder. */
export interface PeQueryReport {
  context_length: number;
  limit: number;
  queries: PeTokenCount[];
  truncated: number;
}

/** The search box's live count: exact when PE will read the text as typed,
 *  estimated (with a range) when the search translates it first. */
export interface PeLiveReport {
  context_length: number;
  limit: number;
  exact: boolean;
  text: string;
  tokens: number;
  range: [number, number] | null;
  kept: string;
  dropped: string;
  at_risk: string;
  status: "empty" | "ok" | "near" | "may_exceed" | "over";
}

// ---- traffic-camera / race-stage filter --------------------------------------

/** `auto` narrows the traffic cameras (N) and the cycling race (S01) to the
 *  junction, date, time or stage the query names; `off` searches them unfiltered. */
export type TrafficFilterMode = "auto" | "off";

/** What the query's camera / date / time / stage cues restricted. Only the N
 *  cameras and the S01 race are ever narrowed; every other folder is untouched. */
export interface TrafficFilterInfo {
  mode: TrafficFilterMode;
  active: boolean;
  streets: string[];
  cameras: { id: string; label: string; banner: string; videos: string[] }[];
  dates: string[];
  time: { from: string; to: string; text: string } | null;
  race_stage: number | null;
  videos: number;
  keyframes: number;
  warnings: string[];
  reason_en: string;
}

/** What the camera banner or race HUD printed on a traffic-camera / S01 frame. */
export interface FrameOverlay {
  camera?: string;
  banner_camera?: string;
  banner_date?: string;
  clock?: string;
  race_stage?: number;
  race_time?: string;
}

export interface ScopeCategory {
  category: string;
  label_vi: string;
  label_en?: string;
  /** True for programmes with no fixed subject (news bulletins, L30 shorts). */
  open_subject: boolean;
  /** Single videos the scope can narrow this folder to (S01: one per race stage). */
  videos?: ScopeVideo[];
}

export interface ScopeVideo {
  video_id: string;
  label_vi: string;
  label_en?: string;
  summary?: string | null;
}

export interface ScopeCatalogue {
  retrieval_database: RetrievalDatabase;
  categories: ScopeCategory[];
  /** `collapsed` groups show only their header until the operator unfolds them. */
  groups: { id: string; label_vi: string; label_en?: string; categories: string[]; collapsed?: boolean }[];
  topics: { topic_id: string; label_vi: string; label_en?: string; categories: string[] }[];
}

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
  /** Traffic-camera junction/date/clock or race stage, when the frame has one. */
  overlay?: FrameOverlay;
  video_url: string;
  evidence: Evidence[];
}

export interface VideoGroup {
  progressive?: {
    memory_score: number;
    cumulative_rank: number | null;
    trajectory: (number | null)[];
    dispersed: boolean;
    moment_source: "cumulative" | "historical_evidence" | "delta" | "latest";
    hint_evidence: { hint_id: string; status: string; rank_evidence?: number; channels: Record<string, string>; frames: FrameResult[] }[];
  };
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
  best_clip?: {
    clip_id: string;
    scale: "event" | "sequence" | "scene";
    start_time: number;
    end_time: number;
    center_frame_idx: number;
    fps: number;
    score: number;
  } | null;
  // TRAKE-only (set when a sequence is mapped into a group): events backed by real
  // retrieval vs filled by pass-2 in-video search, and mean relevance of real frames.
  trake_confident?: number;
  trake_filled?: number;
  trake_mean?: number;
}

export interface ProgressiveHint {
  hint_id: string;
  raw_text: string;
  input_mode: "delta" | "cumulative";
  enabled: boolean;
  source?: string;
  revealed_at?: number | null;
  delta_text?: string;
  cumulative_text?: string;
}

export interface ProgressiveConfig {
  retrieval_database: RetrievalDatabase;
  task_id: string;
  image_models: ImageEmbeddingModel[];
  scope: SearchScope;
  translate: boolean;
  hybrid: boolean;
  top_k: number;
}

export interface ProgressiveSnapshot {
  query?: string;
  session_id: string;
  revision: number;
  committed_revision: number;
  pending_revision: number | null;
  turn: number;
  groups: VideoGroup[];
  hint_ledger?: ProgressiveHint[];
  parsed?: ParsedQuery;
  scope?: ResolvedScope;
  budget?: Record<string, number>;
  revision_budget?: Record<string, number>;
  revision_latency_ms?: number;
  no_op?: boolean;
  warnings?: string[];
  degraded?: boolean;
  mode?: "mock" | "live";
  stability?: { top1_changed: boolean; top1_streak: number; top10_jaccard: number | null };
  latency_ms?: { total_ms: number };
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

/** What the Qwen3-VL rerank stage did for one search.
 *  `ok: false` means the worker was unreachable and the PE order was kept. */
export interface RerankerReport {
  ok: boolean;
  candidates: number;
  reranked: number;
  ms: number;
  error?: string;
}

export interface LatencyBreakdown {
  parse_ms?: number;
  fusion_ms?: number;
  total_ms?: number;
  channels?: Record<string, number>;
  reranker?: RerankerReport;
}

export interface SearchResponse {
  retrieval_database: RetrievalDatabase;
  query: string;
  parsed: ParsedQuery;
  scope?: ResolvedScope;
  traffic?: TrafficFilterInfo;
  /** Null when PE was not searched. */
  pe_tokens?: PeQueryReport | null;
  groups: VideoGroup[];
  latency_ms: LatencyBreakdown;
  warnings?: string[];
  mode: "mock" | "live";
}

// ---- answer generation (the ordered 100-answer list) -----------------------

/** Tuning knobs of the rank-budget allocator. Every field is optional: the
 *  backend fills in the values fitted on the L21-L30 ground-truth queries and
 *  clamps whatever is sent, so a partial object is always valid. */
export interface AnswerGenParams {
  pool_depth?: number;
  max_videos?: number;
  dedup_radius_s?: number;
  max_anchors_per_video?: number;
  anchor_support_weight?: number;
  temperatures?: number[];
  offsets?: number[];
  offset_decay?: number;
  novelty_sigma_s?: number;
  novelty_floor?: number;
  novelty_mode?: "product" | "min";
  band_anchors?: number[];
  band_offsets?: number[];
  ambiguous_anchor_bonus?: number;
  ambiguous_offset_penalty?: number;
  min_answer_gap_frames?: number;
  tail_margin_frames?: number;
  snap_offsets?: boolean;
  snap_radius_frames?: number;
}

export interface GeneratedAnswer {
  rank: number;
  video_id: string;
  /** One frame for KIS/QA, one per event for TRAKE. This is what the CSV holds. */
  frames: number[];
  keyframe_ids: (string | null)[];
  pts_times: (number | null)[];
  answer: string;
  /** `anchor`/`sequence` = a retrieved frame; `offset` = a +/-eps probe around it. */
  kind: "anchor" | "offset" | "sequence";
  offset_frames: number;
  /** Why this answer sits at this rank; debugging only, never submitted. */
  utility?: number;
  video_weight?: number;
  evidence?: number;
  novelty?: number;
  keyframe_url?: string | null;
  video_url?: string;
}

export interface AnswerGenerateResponse {
  query: string;
  query_type: QueryType;
  retrieval_database: RetrievalDatabase;
  /** True when the backend ranked a result the client passed in rather than searching. */
  reused_results: boolean;
  scope?: ResolvedScope | null;
  params: Required<AnswerGenParams>;
  answers: GeneratedAnswer[];
  diagnostics: Record<string, number>;
  warnings: string[];
  latency_ms: Record<string, number>;
  mode: "mock" | "live";
}

// ---- V-KIS sketch canvas --------------------------------------------------

export interface CanvasSpec {
  /** The drawing flattened onto its background, as a PNG data URL. */
  image: string;
}

export interface CanvasSearchResponse {
  canvas: { has_image: boolean; suppress_blank?: boolean };
  retrieval_database?: RetrievalDatabase;
  scope?: ResolvedScope;
  groups: VideoGroup[];
  latency_ms: LatencyBreakdown;
  warnings: string[];
  mode: "mock" | "live";
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
  /** "visual": found by looking at the frames ("nvila" from older backends). */
  source: "visual" | "nvila" | "web" | "knowledge" | "hybrid" | string;
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

/** One distinct moment an event fires at inside a video.
 *
 *  `strength` is normalized PER EVENT against that event's best hit anywhere in
 *  the pool, so E1's and E2's heat rows are comparable even though their raw
 *  score distributions are not. */
export interface TrakeHeatPeak {
  event_index: number;
  submit_keyframe_id: string;
  keyframe_n: number;
  frame_idx: number | null;
  pts_time: number;
  score: number;
  strength: number;
  via_fill: boolean;
  selected_by_dp: boolean;
  keyframe_url: string;
}

/** What one video has to say about one event. `in_chain` is false when the DP
 *  could not place the event in time — the strongest peak is still shown, so the
 *  operator can judge the video instead of only seeing a gap. */
export interface TrakeEventEvidence {
  event_index: number;
  /** This video has something to show for the event — NOT that the chain covers
   *  it. Only `in_chain` means that. */
  has_candidate: boolean;
  in_chain: boolean;
  representative: TrakeHeatPeak | null;
  peaks: TrakeHeatPeak[];
}

/** A TRAKE answer candidate as the console ranks it: one VIDEO, its per-event
 *  evidence, and the coverage-dominant score the list sorts on. */
export interface TrakeVideoResult {
  video_id: string;
  video_url: string;
  trake_video_score: number;
  coverage: number;
  confident_coverage: number;
  filled_events: number;
  /** Events pass-1 retrieval found any candidate for — what pass 2 could fix. */
  evidence_coverage: number;
  chain_quality: number;
  mean_quality: number;
  min_quality: number;
  complete: boolean;
  warning: string | null;
  /** Smallest gap between two chain frames; a warning sign, never a ranking term. */
  min_event_gap: number | null;
  /** Video length in seconds when the keyframe map could resolve it — the time
   *  axis the heat rows are drawn against. */
  duration_s: number | null;
  /** Where the video stood before the pass-2 fill ran. */
  preliminary_rank: number | null;
  events: TrakeEventEvidence[];
  best_chain: TrakeSequenceFrame[];
}

export interface TrakeSearchResponse {
  retrieval_database: RetrievalDatabase;
  query: string;
  parsed: ParsedQuery;
  scope?: ResolvedScope;
  traffic?: TrafficFilterInfo;
  /** One entry per event query; null when PE was not searched. */
  pe_tokens?: PeQueryReport | null;
  events: { event_index: number; description_vi: string; candidate_count: number }[];
  /** The video-centric result the console renders. */
  videos?: TrakeVideoResult[];
  /** The same assembly as flat chains — what the answer generator consumes. */
  sequences: TrakeSequence[];
  pass2?: { targets: string[]; candidates: number };
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
  retrieval_database: RetrievalDatabase;
  video_id: string;
  fps: number;
  duration: number | null;
  video_url: string;
  keyframes: TimelineKeyframe[];
}

/** `GET /api/keyframes/{submit_keyframe_id}` — resolves one keyframe's media
 *  URLs without loading a whole timeline. Used by the Submission tab's preview. */
export interface KeyframeInfo {
  image_id: string;
  submit_keyframe_id: string;
  video_id: string;
  keyframe_n: number;
  pts_time: number | null;
  fps: number | null;
  frame_idx: number | null;
  retrieval_database: RetrievalDatabase;
  keyframe_url: string;
  video_url: string;
  found: boolean;
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
  /** Cosine for one model; the RRF score when two models answered. */
  score: number;
  /** Which image indices returned this frame, in request order. */
  models: ImageEmbeddingModel[];
  /** Each model's own cosine — never summed across models, only shown. */
  per_model_score: Partial<Record<ImageEmbeddingModel, number>>;
  /** Qwen3-VL relevance, present only when the search was reranked. */
  rerank_score?: number | null;
  keyframe_url: string;
  video_url: string;
}

export interface SimpleSearchResponse {
  retrieval_database: RetrievalDatabase;
  query: string;
  /** Which image indices the backend actually searched. */
  image_models: ImageEmbeddingModel[];
  translated_query?: string | null;
  scope?: ResolvedScope;
  pe_tokens?: PeQueryReport | null;
  results: SimpleResult[];
  mode: "mock" | "live";
  latency_ms: number;
  /** Per-model wall clock, so a slow encoder is visible next to a fast one. */
  model_latency_ms?: Partial<Record<ImageEmbeddingModel, number>>;
  reranker?: RerankerReport;
  /** Degraded-search notices: a dead encoder, a failed VI→EN translation. */
  warnings?: string[];
}

export interface HealthResponse {
  retrieval_database: RetrievalDatabase;
  ok: boolean;
  mode: "mock" | "live";
  services: Record<string, { ok: boolean; error?: string }>;
  capabilities: Record<string, boolean>;
  /** Media origins for this profile. A fallback is empty when no second origin
   *  is configured; failed assets are retried under it when present. */
  media?: {
    keyframe_base_url: string;
    keyframe_fallback_base_url: string;
    video_base_url: string;
    video_fallback_base_url: string;
  };
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

/** `qa_text` / `trake_text` are the final-round text forms (HD-ChungKet-2026). */
export type AnswerMode = "auto" | "temporal" | "qa_text" | "trake_text" | "temporal_text" | "text" | "item";

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

// ---- Configuration (in-app .env import) ------------------------------------

export interface ConfigKeyStatus {
  key: string;
  label: string;
  secret: boolean;
  required: boolean;
  set: boolean;
  /** Masked for secrets; the real value for everything else. */
  preview: string;
  /** Set from the process environment, so importing a file will not change it. */
  from_process_env: boolean;
}

export interface ConfigGroupStatus {
  name: string;
  summary: string;
  keys: ConfigKeyStatus[];
}

export interface ConfigStatus {
  configured: boolean;
  mock_mode: boolean;
  missing_required: string[];
  env_path: string;
  env_exists: boolean;
  groups: ConfigGroupStatus[];
}

export interface ConfigImportResult {
  env_path: string;
  applied: string[];
  unknown: string[];
  rejected: string[];
  ignored_blank: string[];
  replaced: boolean;
  status: ConfigStatus;
}

// ---- Agent sidecar search (Codex CLI / Claude Code CLI) ---------------------

export type AgentName = "codex" | "claude";

/** `pending`/`queued` wait for a slot; the last five are final. */
export type AgentStatus =
  | "pending"
  | "queued"
  | "running"
  | "done"
  | "failed"
  | "timeout"
  | "cancelled"
  | "unavailable";

export interface AgentStep {
  /** Seconds since the run started. */
  at: number;
  kind: "tool" | "thought" | "message" | "error" | "info";
  text: string;
}

export interface AgentState {
  name: AgentName;
  status: AgentStatus;
  model: string;
  effort: string;
  /** Codex priority tier / Claude Code fast mode. */
  fast?: boolean;
  elapsed_s: number | null;
  tool_calls: number;
  summary: string;
  error: string | null;
  /** `quota` / `auth` when the account, not the search, is the problem;
   *  `update` when the installed CLI is too old for the configured model. */
  error_kind: "quota" | "auth" | "update" | "timeout" | "cli" | null;
  cost_usd: number | null;
  steps: AgentStep[];
}

/** A frame an agent reported, resolved to the profile's keyframe map. */
export interface AgentCandidate {
  id: string;
  agent: AgentName;
  video_id: string;
  submit_keyframe_id: string;
  keyframe_n: number;
  frame_idx: number | null;
  fps: number | null;
  pts_time: number | null;
  /** The instant the agent named (its keyframe's pts when it gave an id). */
  time: number | null;
  keyframe_url: string;
  video_url: string;
  confidence: number;
  reason: string;
  answer: string | null;
  /** TRAKE: which event this frame is (1-based). */
  event: number | null;
  /** The frame's folder is not in the operator's folder filter. */
  outside_scope?: boolean;
  found_at_s: number;
  updated_at_s?: number;
}

export interface AgentRunSnapshot {
  run_id: string;
  query: string;
  query_type: QueryType;
  retrieval_database: RetrievalDatabase;
  created_at: number;
  elapsed_s: number;
  finished: boolean;
  agents: Partial<Record<AgentName, AgentState>>;
  candidates: AgentCandidate[];
  /** The folder filter the agents were told to search first. */
  scope?: AgentRunScope;
}

export interface AgentRunScope {
  mode: "all" | "auto" | "manual";
  active: boolean;
  categories: string[];
  reason: string;
}
