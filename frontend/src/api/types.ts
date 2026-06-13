// Types mirroring the backend API contract (see backend/README and main.py).

export type Channel = "image_pe" | "ocr" | "speech" | "audio";
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

export interface SubmitEntry {
  id: string;
  ts: number;
  task_id: string;
  query_type: QueryType;
  payload: {
    video_id?: string;
    frame_idx?: number | null;
    events?: { event_index: number; frame_idx: number }[];
    answer?: string;
    submit_keyframe_id?: string;
  };
  dedup_keys: string[];
  status: string;
  was_duplicate: boolean;
}

export interface FeedbackState {
  positive_videos: string[];
  negative_videos: string[];
  negative_frames: string[];
}
