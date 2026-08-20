// Thin fetch wrapper around the backend. The frontend only ever talks to the
// backend — it never holds Elastic/Milvus/NVIDIA credentials.
import type {
  AnswerGenerateResponse,
  AnswerGenParams,
  AnswerMode,
  CanvasPalette,
  CanvasSearchResponse,
  CanvasSpec,
  ConfigImportResult,
  ConfigStatus,
  DresEvaluation,
  DresStatus,
  DresTaskHint,
  FeedbackState,
  HealthResponse,
  KeyframeInfo,
  ParsedQuery,
  QaAnalysisResponse,
  QaAnalyzeCandidate,
  QueryTypeHint,
  RetrievalDatabase,
  ScopeCatalogue,
  SearchResponse,
  SearchScope,
  SimpleSearchResponse,
  SnapResult,
  SubmitEntry,
  SubmitPreview,
  Timeline,
  TrakeSearchResponse,
  TrakeSequence,
  VideoGroup,
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

/** What /api/submit and /api/submit/preview take. `evaluation_id` / `task_name`
 *  are resolved from the live DRES run when omitted; `start_ms`/`end_ms` override
 *  the ms window the backend derives from `timestamp` (or `frame_idx` + `fps`). */
export interface SubmitBody {
  task_id?: string;
  evaluation_id?: string | null;
  task_name?: string | null;
  query_type: string;
  payload: {
    video_id?: string;
    frame_idx?: number | null;
    timestamp?: number | null;
    fps?: number | null;
    start_ms?: number | null;
    end_ms?: number | null;
    events?: {
      event_index: number;
      frame_idx: number;
      pts_time?: number | null;
      video_id?: string;
      submit_keyframe_id?: string;
    }[];
    answer?: string;
    submit_keyframe_id?: string;
  };
  answer_mode?: AnswerMode;
  segment_pad_ms?: number | null;
  allow_duplicate?: boolean;
}

export const api = {
  health: (database: RetrievalDatabase = "btc") =>
    request<HealthResponse>(`/api/health?retrieval_database=${database}`),

  config: () => request<ConfigStatus>("/api/config"),

  /** URL of the documented `.env` template — used as a plain download link. */
  configTemplateUrl: () => `${BASE}/api/config/template`,

  importConfig: async (file: File, replace: boolean): Promise<ConfigImportResult> => {
    const form = new FormData();
    form.append("file", file, ".env");
    form.append("replace", String(replace));
    const resp = await fetch(`${BASE}/api/config/import`, { method: "POST", body: form });
    if (!resp.ok) {
      let detail: unknown = null;
      try {
        detail = (await resp.json()).detail ?? null;
      } catch {
        /* ignore */
      }
      throw new ApiError(resp.status, detail, `${resp.status} /api/config/import`);
    }
    return resp.json();
  },

  reloadConfig: () => request<ConfigStatus>("/api/config/reload", { method: "POST" }),

  /** The folder catalogue for one profile (fixed per profile, so fetch once). */
  searchScope: (retrieval_database: RetrievalDatabase = "btc") =>
    request<ScopeCatalogue>(`/api/search/scope?retrieval_database=${retrieval_database}`),

  simpleSearch: (
    query: string,
    top_k: number,
    retrieval_database: RetrievalDatabase = "btc",
    scope?: SearchScope,
  ) =>
    request<SimpleSearchResponse>("/api/search/simple", {
      method: "POST",
      body: JSON.stringify({ query, top_k, retrieval_database, ...(scope ? { scope } : {}) }),
    }),

  parse: (query: string, hint: QueryTypeHint, previous_hints: string[], overrides: ManualOverrides, use_llm = false) =>
    request<ParsedQuery>("/api/query/parse", {
      method: "POST",
      body: JSON.stringify({ query, query_type_hint: hint, previous_hints, manual_overrides: overrides, use_llm }),
    }),

  search: (body: {
    retrieval_database: RetrievalDatabase;
    query: string;
    query_type_hint: QueryTypeHint;
    scope?: SearchScope;
    previous_hints: string[];
    manual_overrides: ManualOverrides;
    parsed?: ParsedQuery | null;
    feedback?: FeedbackState;
    use_llm?: boolean;
    expand?: boolean;
    top_k?: number;
    max_videos?: number;
  }) =>
    request<SearchResponse>("/api/search", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  searchTrake: (body: { retrieval_database: RetrievalDatabase; query: string; scope?: SearchScope; previous_hints: string[]; manual_overrides: ManualOverrides; use_llm?: boolean; expand?: boolean; top_k?: number }) =>
    request<TrakeSearchResponse>("/api/search/trake", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /** The ordered answer list for one query.
   *
   *  Pass `groups`/`sequences` to rank a result already on screen (keeping the
   *  operator's scope, feedback and channel overrides); omit them to let the
   *  backend run its own retrieval, which is what bulk generation does. */
  generateAnswers: (body: {
    retrieval_database: RetrievalDatabase;
    query: string;
    query_type_hint: QueryTypeHint;
    scope?: SearchScope;
    previous_hints?: string[];
    manual_overrides?: ManualOverrides;
    feedback?: FeedbackState;
    use_llm?: boolean;
    expand?: boolean;
    top_k?: number;
    max_videos?: number;
    limit?: number;
    params?: AnswerGenParams;
    answer_text?: string;
    /** TRAKE: frames per row, taken from the statement. */
    event_count?: number;
    groups?: VideoGroup[];
    sequences?: TrakeSequence[];
  }, signal?: AbortSignal) =>
    request<AnswerGenerateResponse>("/api/answers/generate", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),

  canvasPalette: () => request<CanvasPalette>("/api/canvas/palette"),

  searchCanvas: (body: { retrieval_database: RetrievalDatabase; canvas: CanvasSpec; scope?: SearchScope; top_k?: number; candidate_pool?: number }) =>
    request<CanvasSearchResponse>("/api/search/canvas", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  qaAnalyze: (body: {
    retrieval_database: RetrievalDatabase;
    question: string;
    candidates: QaAnalyzeCandidate[];
    max_answers?: number;
    web_grounding?: "auto" | "on" | "off";
  }) =>
    request<QaAnalysisResponse>("/api/qa/analyze", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  timeline: (videoId: string, database: RetrievalDatabase = "btc") =>
    request<Timeline>(`/api/videos/${videoId}/timeline?retrieval_database=${database}`),

  keyframe: (submitKeyframeId: string, database: RetrievalDatabase = "btc") =>
    request<KeyframeInfo>(
      `/api/keyframes/${submitKeyframeId}?retrieval_database=${database}`,
    ),

  snap: (videoId: string, rawTime: number, fps?: number, retrieval_database: RetrievalDatabase = "btc") =>
    request<SnapResult>(`/api/videos/${videoId}/snap`, {
      method: "POST",
      body: JSON.stringify({ raw_time: rawTime, fps, retrieval_database }),
    }),

  // ---- DRES ----------------------------------------------------------------
  dresStatus: () => request<DresStatus>("/api/dres/status"),

  dresLogin: () =>
    request<{ ok: boolean; base_url: string }>("/api/dres/login", { method: "POST" }),

  dresEvaluations: (force = false) =>
    request<{ evaluations: DresEvaluation[]; server_time: number | null; pinned_evaluation_id: string | null }>(
      `/api/dres/evaluations${force ? "?force=true" : ""}`,
    ),

  dresTaskHint: (queryType: string, evaluationId?: string | null, force = false) =>
    request<DresTaskHint>(
      `/api/dres/task-hint?query_type=${encodeURIComponent(queryType)}` +
        `${evaluationId ? `&evaluation_id=${encodeURIComponent(evaluationId)}` : ""}${force ? "&force=true" : ""}`,
    ),

  submitPreview: (body: SubmitBody) =>
    request<SubmitPreview>("/api/submit/preview", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  submit: (body: SubmitBody) =>
    request<SubmitEntry>("/api/submit", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  history: (taskId?: string) =>
    request<{ history: SubmitEntry[] }>(`/api/submit/history${taskId ? `?task_id=${taskId}` : ""}`),

  /** Deletes LOCAL log entries only (and their dedup memory); DRES keeps its record.
   *  `ids` removes specific entries, `taskId` one task, neither clears everything. */
  clearHistory: (opts: { ids?: string[]; taskId?: string } = {}) => {
    const params = new URLSearchParams();
    if (opts.ids) params.set("ids", opts.ids.join(","));
    if (opts.taskId) params.set("task_id", opts.taskId);
    const query = params.toString();
    return request<{ deleted: number; remaining: number; backup: string | null }>(
      `/api/submit/history${query ? `?${query}` : ""}`,
      { method: "DELETE" },
    );
  },

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
