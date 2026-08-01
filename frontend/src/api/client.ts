// Thin fetch wrapper around the backend. The frontend only ever talks to the
// backend — it never holds Elastic/Milvus/NVIDIA credentials.
import type {
  FeedbackState,
  HealthResponse,
  ParsedQuery,
  QaAnalysisResponse,
  QaAnalyzeCandidate,
  QueryTypeHint,
  SearchResponse,
  SimpleSearchResponse,
  SnapResult,
  SubmitEntry,
  Timeline,
  TrakeSearchResponse,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE_URL || "";

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    let detail: unknown = null;
    try {
      detail = (await resp.json()).detail ?? null;
    } catch {
      /* ignore */
    }
    throw new ApiError(resp.status, detail, `${resp.status} ${path}`);
  }
  return resp.json() as Promise<T>;
}

export interface ManualOverrides {
  force_channels: string[];
  disable_channels: string[];
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),

  simpleSearch: (query: string, top_k: number) =>
    request<SimpleSearchResponse>("/api/search/simple", {
      method: "POST",
      body: JSON.stringify({ query, top_k }),
    }),

  parse: (query: string, hint: QueryTypeHint, previous_hints: string[], overrides: ManualOverrides, use_llm = false) =>
    request<ParsedQuery>("/api/query/parse", {
      method: "POST",
      body: JSON.stringify({ query, query_type_hint: hint, previous_hints, manual_overrides: overrides, use_llm }),
    }),

  search: (body: {
    query: string;
    query_type_hint: QueryTypeHint;
    previous_hints: string[];
    manual_overrides: ManualOverrides;
    parsed?: ParsedQuery | null;
    feedback?: FeedbackState;
    use_llm?: boolean;
    expand?: boolean;
  }) =>
    request<SearchResponse>("/api/search", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  searchTrake: (body: { query: string; previous_hints: string[]; manual_overrides: ManualOverrides; use_llm?: boolean; expand?: boolean }) =>
    request<TrakeSearchResponse>("/api/search/trake", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  qaAnalyze: (body: {
    question: string;
    candidates: QaAnalyzeCandidate[];
    max_answers?: number;
    web_grounding?: "auto" | "on" | "off";
  }) =>
    request<QaAnalysisResponse>("/api/qa/analyze", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  timeline: (videoId: string) => request<Timeline>(`/api/videos/${videoId}/timeline`),

  snap: (videoId: string, rawTime: number, fps?: number) =>
    request<SnapResult>(`/api/videos/${videoId}/snap`, {
      method: "POST",
      body: JSON.stringify({ raw_time: rawTime, fps }),
    }),

  submit: (body: {
    task_id: string;
    query_type: string;
    payload: {
      video_id?: string;
      frame_idx?: number | null;
      timestamp?: number | null;
      events?: { event_index: number; frame_idx: number; pts_time?: number | null; submit_keyframe_id?: string }[];
      answer?: string;
      submit_keyframe_id?: string;
    };
    allow_duplicate?: boolean;
  }) =>
    request<SubmitEntry>("/api/submit", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  history: (taskId?: string) =>
    request<{ history: SubmitEntry[] }>(`/api/submit/history${taskId ? `?task_id=${taskId}` : ""}`),

  translate: (text: string) =>
    request<{ text: string; text_en: string }>("/api/translate", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  transcribe: async (audio: Blob): Promise<{ text: string; text_en: string | null }> => {
    const form = new FormData();
    form.append("audio", audio, "voice.webm");
    const resp = await fetch(`${BASE}/api/transcribe`, { method: "POST", body: form });
    if (!resp.ok) {
      let detail: unknown = null;
      try {
        detail = (await resp.json()).detail ?? null;
      } catch {
        /* ignore */
      }
      throw new ApiError(resp.status, detail, `${resp.status} /api/transcribe`);
    }
    return resp.json();
  },
};
