import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type ManualOverrides, type SubmitBody } from "./api/client";
import type {
  AnswerMode,
  CanvasSpec,
  Channel,
  DresEvaluation,
  DresStatus,
  DresTaskHint,
  FeedbackState,
  FrameResult,
  HealthResponse,
  LatencyBreakdown,
  ParsedQuery,
  QaAnalysisResponse,
  QueryType,
  SubmitEntry,
  SubmitPreview,
  Timeline as TimelineData,
  VideoGroup,
} from "./api/types";
import { CanvasPanel } from "./components/CanvasPanel";
import { ChannelControls } from "./components/ChannelControls";
import { DetailPanel } from "./components/DetailPanel";
import { DresBar } from "./components/DresBar";
import { FeedbackBar } from "./components/FeedbackBar";
import { HistorySidebar } from "./components/HistorySidebar";
import { NeighborStrip } from "./components/NeighborStrip";
import { QueryPanel } from "./components/QueryPanel";
import { QueryUnderstanding } from "./components/QueryUnderstanding";
import { QaAssistPanel } from "./components/QaAssistPanel";
import { Results } from "./components/Results";
import { PausedFramePanel, type PausedFrame } from "./components/PausedFramePanel";
import { ShortcutsModal } from "./components/ShortcutsModal";
import { TaskHintPanel } from "./components/TaskHintPanel";
import { SubmitGuard } from "./components/SubmitGuard";
import { Timeline } from "./components/Timeline";
import { TopBar } from "./components/TopBar";
import { TrakePanel, type TrakeSlot } from "./components/TrakePanel";
import { VideoViewer, type VideoViewerHandle } from "./components/VideoViewer";
import { autoEvaluationId } from "./lib/dres";
import { findQaFrame, selectQaCandidateFrames } from "./lib/qa";
import { validateIncreasingOrder } from "./lib/snap";

/** The submit guard acts on exactly one of these; they never shadow each other. */
type GuardTarget = "result" | "paused";

const EMPTY_OVERRIDES: ManualOverrides = { force_channels: [], disable_channels: [] };
const EMPTY_FEEDBACK: FeedbackState = {
  positive_videos: [], negative_videos: [], positive_frames: [], negative_frames: [],
};

export default function FullConsole({ onSimpleMode }: { onSimpleMode: () => void }) {
  const [queryType, setQueryType] = useState<QueryType>("T-KIS");
  const [query, setQuery] = useState("");
  const [hints, setHints] = useState<string[]>([]);
  const [parsed, setParsed] = useState<ParsedQuery | null>(null);
  const [overrides, setOverrides] = useState<ManualOverrides>(EMPTY_OVERRIDES);
  const [useLLM, setUseLLM] = useState(false);
  const [expand, setExpand] = useState(false);
  const [groups, setGroups] = useState<VideoGroup[]>([]);
  const [latency, setLatency] = useState<LatencyBreakdown | null>(null);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [feedback, setFeedback] = useState<FeedbackState>(EMPTY_FEEDBACK);

  const [selectedVideo, setSelectedVideo] = useState(0);
  const [selectedFrame, setSelectedFrame] = useState(0);
  const [viewMode, setViewMode] = useState<"grouped" | "flat">("grouped");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [focusZone, setFocusZone] = useState<"results" | "detail">("results");

  const [timeline, setTimeline] = useState<TimelineData | null>(null);
  const [playhead, setPlayhead] = useState(0);
  const [showTimeline, setShowTimeline] = useState(true);
  // Inline video: rendered under the selected keyframe's group, loaded once per
  // video, hidden automatically when the selection moves to a different video.
  const [videoVisible, setVideoVisible] = useState(false);
  const [activeVideoId, setActiveVideoId] = useState<string | null>(null);
  // Neighbour browser: the selected result keyframe is the anchor and the offset
  // is how far the operator has paged from it. Storing an offset (not an index)
  // keeps the state valid while the timeline is still loading.
  const [neighborsVisible, setNeighborsVisible] = useState(false);
  const [neighborOffset, setNeighborOffset] = useState(0);

  const [trakeSlots, setTrakeSlots] = useState<(TrakeSlot | null)[]>([null, null]);
  const [activeSlot, setActiveSlot] = useState(0);
  const [pausedFrame, setPausedFrame] = useState<PausedFrame | null>(null);
  // Which frame the OPEN guard is about to submit. Capturing a paused frame no
  // longer re-points the detail panel: the two submit paths stay independent and
  // the target is decided by which button (or shortcut) opened the guard.
  const [guardTarget, setGuardTarget] = useState<GuardTarget>("result");
  const [eventMarkers, setEventMarkers] = useState<{ eventIndex: number; pts_time: number }[]>([]);

  const [keymapOpen, setKeymapOpen] = useState(false);
  const [guardOpen, setGuardOpen] = useState(false);
  const [answer, setAnswer] = useState("");

  // ---- DRES (official evaluation server) ----
  // The run and the open task come from the server; `pinnedEvaluationId` is set
  // only when the operator overrides the auto-routing by query type.
  const [dresStatus, setDresStatus] = useState<DresStatus | null>(null);
  const [dresEvaluations, setDresEvaluations] = useState<DresEvaluation[]>([]);
  const [pinnedEvaluationId, setPinnedEvaluationId] = useState<string | null>(null);
  const [dresError, setDresError] = useState<string | null>(null);
  const [dresBusy, setDresBusy] = useState(false);
  const [timeLeft, setTimeLeft] = useState<number | null>(null);
  const [taskNameOverride, setTaskNameOverride] = useState("");
  const [answerMode, setAnswerMode] = useState<AnswerMode>("auto");
  const [segmentPadMs, setSegmentPadMs] = useState(500);
  const [taskHint, setTaskHint] = useState<DresTaskHint | null>(null);
  const [taskHintLoading, setTaskHintLoading] = useState(false);
  const [taskHintError, setTaskHintError] = useState<string | null>(null);
  const [submitPreview, setSubmitPreview] = useState<SubmitPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [qaAnalysis, setQaAnalysis] = useState<QaAnalysisResponse | null>(null);
  const [qaAnalyzing, setQaAnalyzing] = useState(false);
  const [qaAnalysisError, setQaAnalysisError] = useState<string | null>(null);
  const [qaWebGrounding, setQaWebGrounding] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [duplicateId, setDuplicateId] = useState<string | null>(null);

  // V-KIS canvas: the English sentences the backend generated from the drawing,
  // shown back to the operator so the PE side of the search is not a black box.
  const [canvasQueries, setCanvasQueries] = useState<string[]>([]);

  const [history, setHistory] = useState<SubmitEntry[]>([]);
  const [penalties, setPenalties] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [toast, setToast] = useState<{ msg: string; kind: "ok" | "bad" } | null>(null);

  const queryRef = useRef<HTMLTextAreaElement>(null);
  // Set as soon as the operator edits the query themselves; the task statement
  // prefill must never clobber their words.
  const queryTouched = useRef(false);
  const viewerRef = useRef<VideoViewerHandle>(null);
  const timelineCache = useRef<Map<string, TimelineData>>(new Map());

  const selectedGroup = groups[selectedVideo] ?? null;
  const selectedFrameObj: FrameResult | null = selectedGroup?.frames[selectedFrame] ?? null;
  // Only the guard reads this. The detail panel always describes the selected
  // result keyframe, whether or not a raw frame is currently captured.
  const guardPausedFrame =
    queryType !== "TRAKE" && guardTarget === "paused" ? pausedFrame : null;
  const qaCandidateFrames = useMemo(
    () => (queryType === "QA" ? selectQaCandidateFrames(groups, selectedFrameObj) : []),
    [queryType, groups, selectedFrameObj],
  );
  const qaCandidates = useMemo(() => qaCandidateFrames.map((frame) => ({
    submit_keyframe_id: frame.submit_keyframe_id,
    frame_idx: frame.frame_idx,
    pts_time: frame.pts_time,
    retrieval_score: frame.score,
    evidence: frame.evidence,
  })), [qaCandidateFrames]);
  const qaHotspotScores = useMemo(
    () => new Map((qaAnalysis?.hotspots ?? []).map((hotspot) => [hotspot.submit_keyframe_id, hotspot.relevance])),
    [qaAnalysis],
  );

  // ---- neighbour keyframe browser ----
  const neighborKeyframes =
    neighborsVisible && timeline && timeline.video_id === activeVideoId ? timeline.keyframes : [];
  const neighborAnchorId = selectedFrameObj?.submit_keyframe_id ?? null;
  const neighborAnchorIndex = useMemo(() => {
    if (!neighborKeyframes.length) return 0;
    const byId = neighborKeyframes.findIndex((kf) => kf.submit_keyframe_id === neighborAnchorId);
    if (byId >= 0) return byId;
    // A result frame can be missing from the map; fall back to nearest in time.
    const pts = selectedFrameObj?.pts_time;
    if (pts == null) return 0;
    let best = 0;
    let bestDelta = Infinity;
    neighborKeyframes.forEach((kf, index) => {
      if (kf.pts_time == null) return;
      const delta = Math.abs(kf.pts_time - pts);
      if (delta < bestDelta) {
        bestDelta = delta;
        best = index;
      }
    });
    return best;
  }, [neighborKeyframes, neighborAnchorId, selectedFrameObj?.pts_time]);
  const neighborIndex = neighborKeyframes.length
    ? Math.min(Math.max(neighborAnchorIndex + neighborOffset, 0), neighborKeyframes.length - 1)
    : 0;

  function toggleNeighbors() {
    if (!selectedFrameObj || !selectedVideoId) return;
    if (neighborsVisible) {
      setNeighborsVisible(false);
      return;
    }
    // Reuse the same timeline fetch the inline video uses (client-cached).
    setActiveVideoId(selectedVideoId);
    setNeighborOffset(0);
    setNeighborsVisible(true);
  }

  /** Page the strip; the inline video, when open, follows to that keyframe. */
  function moveNeighbor(delta: number) {
    if (!neighborKeyframes.length) return;
    const next = Math.min(Math.max(neighborIndex + delta, 0), neighborKeyframes.length - 1);
    setNeighborOffset(next - neighborAnchorIndex);
    const pts = neighborKeyframes[next]?.pts_time;
    if (videoVisible && pts != null) seekVideo(pts);
  }

  function pickNeighbor(index: number) {
    setNeighborOffset(index - neighborAnchorIndex);
    const pts = neighborKeyframes[index]?.pts_time;
    if (videoVisible && pts != null) seekVideo(pts);
  }


  // ---- bootstrap: health + history + timer ----
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    refreshHistory();
  }, []);

  useEffect(() => {
    const id = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (toast) {
      const id = setTimeout(() => setToast(null), 2600);
      return () => clearTimeout(id);
    }
  }, [toast]);

  const refreshHistory = useCallback(() => {
    api.history().then((r) => setHistory(r.history)).catch(() => {});
  }, []);

  const clearHistory = useCallback(async (opts: { ids?: string[]; taskId?: string } = {}) => {
    try {
      const r = await api.clearHistory(opts);
      setDuplicateId(null);
      refreshHistory();
      setToast({
        msg: r.deleted ? `Đã xoá ${r.deleted} mục (backup ${r.backup})` : "Không có mục nào để xoá",
        kind: "ok",
      });
    } catch {
      setToast({ msg: "Xoá history thất bại", kind: "bad" });
    }
  }, [refreshHistory]);

  // ---- DRES: connection, open task, countdown ----
  const refreshDresEvaluations = useCallback(async (force = false) => {
    try {
      const r = await api.dresEvaluations(force);
      setDresEvaluations(r.evaluations);
      setDresError(null);
    } catch (e) {
      setDresEvaluations([]);
      setDresError(e instanceof ApiError ? String(e.detail ?? e.message) : "DRES unreachable");
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    api.dresStatus()
      .then((s) => {
        if (cancelled) return;
        setDresStatus(s);
        setSegmentPadMs(s.segment_pad_ms);
        if (s.configured) refreshDresEvaluations(true);
        else setDresError(null);
      })
      .catch(() => !cancelled && setDresStatus(null));
    return () => { cancelled = true; };
  }, [refreshDresEvaluations]);

  // The open task changes without warning during a run, so poll it. 5 s is well
  // under the shortest task and the backend caches for 2 s.
  useEffect(() => {
    if (!dresStatus?.configured) return;
    const id = setInterval(() => refreshDresEvaluations(), 5000);
    return () => clearInterval(id);
  }, [dresStatus?.configured, refreshDresEvaluations]);

  const evaluationId = pinnedEvaluationId ?? autoEvaluationId(dresEvaluations, queryType);
  const activeEvaluation = dresEvaluations.find((e) => e.id === evaluationId) ?? null;
  const currentTaskName = activeEvaluation?.current_task?.name ?? null;
  // History and dedup are scoped to the open DRES task (mirrors `_task_scope`
  // in submit_service.py), so answers for one task never mask another's.
  const submitTaskName = taskNameOverride.trim() || currentTaskName;
  const taskScope = submitTaskName
    ? evaluationId
      ? `${evaluationId}/${submitTaskName}`
      : submitTaskName
    : "unknown-task";

  // Countdown between polls: seed from the server, tick locally.
  useEffect(() => {
    setTimeLeft(activeEvaluation?.state?.timeLeft ?? null);
  }, [activeEvaluation?.state?.timeLeft, activeEvaluation?.id]);
  useEffect(() => {
    if (timeLeft == null) return;
    const id = setInterval(() => setTimeLeft((t) => (t == null ? null : Math.max(0, t - 1))), 1000);
    return () => clearInterval(id);
  }, [timeLeft == null]);

  // A new task means a new answer, a fresh dedup scope and no stale overrides.
  useEffect(() => {
    setTaskNameOverride("");
    setAnswerMode("auto");
  }, [currentTaskName]);

  // ---- the task statement (đề bài) ----
  // Keyed on the task template: the hint never changes while a task is open, and
  // a V-KIS media hint is far too big to re-fetch on the 5 s poll.
  const taskTemplateId = activeEvaluation?.state?.taskTemplateId ?? null;
  const fetchTaskHint = useCallback(
    async (force = false) => {
      if (!evaluationId) return;
      setTaskHintLoading(true);
      try {
        const hint = await api.dresTaskHint(queryType, evaluationId, force);
        setTaskHint(hint);
        setTaskHintError(null);
      } catch (e) {
        setTaskHint(null);
        setTaskHintError(e instanceof ApiError ? String(e.detail ?? e.message) : "Không lấy được đề bài");
      } finally {
        setTaskHintLoading(false);
      }
    },
    [evaluationId, queryType],
  );

  useEffect(() => {
    if (!dresStatus?.configured || !evaluationId) {
      setTaskHint(null);
      return;
    }
    fetchTaskHint();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dresStatus?.configured, evaluationId, taskTemplateId]);

  // A statement IS the query for T-KIS/QA, so it prefills the box — but never
  // over the operator's own words, not even when it lands mid-keystroke. The
  // statement of a previous task is fair game: a new task supersedes it.
  const prefilledHint = useRef<string | null>(null);
  useEffect(() => {
    const text = taskHint?.text;
    if (!text || prefilledHint.current === text) return;
    prefilledHint.current = text;
    setQuery((current) => {
      if (queryTouched.current && current.trim()) return current;
      queryTouched.current = false;
      return text;
    });
  }, [taskHint?.text]);

  function useHintAsQuery() {
    if (!taskHint?.text) return;
    queryTouched.current = true;
    setQuery(taskHint.text);
    queryRef.current?.focus();
  }

  const reconnectDres = useCallback(async () => {
    setDresBusy(true);
    try {
      await api.dresLogin();
      setDresStatus(await api.dresStatus());
      await refreshDresEvaluations(true);
      setToast({ msg: "DRES reconnected ✓", kind: "ok" });
    } catch (e) {
      setToast({ msg: e instanceof ApiError ? `DRES login failed: ${String(e.detail ?? "")}` : "DRES login failed", kind: "bad" });
    } finally {
      setDresBusy(false);
    }
  }, [refreshDresEvaluations]);

  // ---- load timeline only for the video actually being shown inline ----
  const selectedVideoId = selectedGroup?.video_id;
  useEffect(() => {
    if (!activeVideoId) {
      setTimeline(null);
      return;
    }
    const cached = timelineCache.current.get(activeVideoId);
    if (cached) {
      setTimeline(cached);
      return;
    }
    let cancelled = false;
    setTimeline(null); // show loading state for the new video
    api.timeline(activeVideoId).then((tl) => {
      timelineCache.current.set(activeVideoId, tl);
      if (!cancelled) setTimeline(tl);
    }).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [activeVideoId]);

  // ---- search ----
  const runSearch = useCallback(async () => {
    if (!query.trim() && hints.length === 0) return;
    setLoading(true);
    setDuplicateId(null);
    setPausedFrame(null);
    setQaAnalysis(null);
    setQaAnalysisError(null);
    setAnswer("");
    // The canvas explanation belongs to the canvas query that produced it.
    setCanvasQueries([]);
    try {
      if (queryType === "TRAKE") {
        const res = await api.searchTrake({ query, previous_hints: hints, manual_overrides: overrides, use_llm: useLLM, expand });
        setParsed(res.parsed);
        const grp: VideoGroup[] = res.sequences.map((s) => ({
          video_id: s.video_id,
          video_score: s.score,
          max_score: s.score,
          mean_top_score: s.score,
          frame_count: s.frames.length,
          timestamp_dispersion: 0,
          ambiguous: !s.complete,
          channels: [],
          video_url: s.video_url,
          trake_confident: s.confident_coverage,
          trake_filled: s.filled_events,
          trake_mean: s.mean_score,
          frames: s.frames.map((f) => ({
            image_id: f.submit_keyframe_id,
            submit_keyframe_id: f.submit_keyframe_id,
            video_id: s.video_id,
            keyframe_n: f.keyframe_n,
            frame_idx: f.frame_idx,
            fps: null,
            pts_time: f.pts_time,
            score: f.score,
            channels: [] as Channel[],
            per_channel_score: {},
            keyframe_url: f.keyframe_url,
            video_url: s.video_url,
            evidence: [],
          })),
        }));
        setGroups(grp);
        setLatency(null);
        const n = Math.max(2, res.parsed.trake?.events?.length || 2);
        setTrakeSlots(Array(n).fill(null));
        setActiveSlot(0);
      } else {
        const res = await api.search({
          query,
          query_type_hint: queryType,
          previous_hints: hints,
          manual_overrides: overrides,
          feedback,
          use_llm: useLLM,
          expand,
        });
        setParsed(res.parsed);
        setGroups(res.groups);
        setLatency(res.latency_ms);
        if (res.warnings?.length) {
          setToast({ msg: res.warnings.join(" · "), kind: "bad" });
        }
      }
      setSelectedVideo(0);
      setSelectedFrame(0);
    } catch (e) {
      const msg = e instanceof ApiError
        ? (typeof e.detail === "string" ? e.detail : `Search failed (${e.status})`)
        : "Search failed — check backend / /api/health";
      setToast({ msg, kind: "bad" });
    } finally {
      setLoading(false);
    }
  }, [query, hints, overrides, queryType, feedback, useLLM, expand]);

  // ---- V-KIS canvas search ----
  // A separate entry point from the text query: the canvas is its own query, so
  // it never silently re-runs when the operator edits the text box (and vice
  // versa). Results land in the same `groups` state, so timeline, detail panel
  // and submit guard behave exactly as they do for any other search.
  const runCanvasSearch = useCallback(async (canvas: CanvasSpec) => {
    setLoading(true);
    setDuplicateId(null);
    setPausedFrame(null);
    try {
      const res = await api.searchCanvas({ canvas });
      setParsed(null);
      setGroups(res.groups);
      setLatency(res.latency_ms);
      setCanvasQueries(res.canvas.queries_en ?? []);
      setSelectedVideo(0);
      setSelectedFrame(0);
      if (res.warnings?.length) setToast({ msg: res.warnings.join(" · "), kind: "bad" });
      else if (!res.groups.length) setToast({ msg: "Không có frame nào khớp bố cục này", kind: "bad" });
    } catch (e) {
      const msg = e instanceof ApiError
        ? (typeof e.detail === "string" ? e.detail : `Canvas search failed (${e.status})`)
        : "Canvas search failed — check backend / /api/health";
      setToast({ msg, kind: "bad" });
    } finally {
      setLoading(false);
    }
  }, []);

  // Moving the selection to a different video auto-hides the old inline video
  // (it will reload only when the operator presses 'v' on the new video).
  useEffect(() => {
    if (videoVisible && activeVideoId && selectedVideoId && selectedVideoId !== activeVideoId) {
      setVideoVisible(false);
      setActiveVideoId(null);
    }
    if (pausedFrame && selectedVideoId && pausedFrame.video_id !== selectedVideoId) {
      setPausedFrame(null);
    }
    if (neighborsVisible && selectedVideoId && selectedVideoId !== activeVideoId) {
      setNeighborsVisible(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedVideoId]);

  // Re-run the search whenever feedback changes, so the ranking the operator is
  // looking at always matches the feedback chips shown above it.
  useEffect(() => {
    if (!feedbackDirty.current) return;
    feedbackDirty.current = false;
    if (groups.length) runSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [feedback]);

  // Feedback belongs to ONE question. Carrying it into the next query (or task
  // type) would silently re-rank an unrelated search for the rest of the session.
  const feedbackScope = useRef(`${queryType} `);
  useEffect(() => {
    const scope = `${queryType} ${query}`;
    if (scope === feedbackScope.current) return;
    feedbackScope.current = scope;
    setFeedback((fb) =>
      fb.positive_videos.length || fb.negative_videos.length
      || fb.positive_frames.length || fb.negative_frames.length
        ? EMPTY_FEEDBACK
        : fb,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, queryType]);

  // Expand every group by default when a new result set lands
  // (operator can still collapse individual videos by clicking the header).
  useEffect(() => {
    if (groups.length) {
      setExpanded(new Set(groups.map((g) => g.video_id)));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [groups]);

  // ---- channel toggle ----
  function toggleChannel(channel: Channel, enabled: boolean) {
    setOverrides((o) => {
      const force = new Set(o.force_channels);
      const disable = new Set(o.disable_channels);
      if (enabled) {
        force.add(channel);
        disable.delete(channel);
      } else {
        disable.add(channel);
        force.delete(channel);
      }
      return { force_channels: [...force], disable_channels: [...disable] };
    });
  }

  // ---- feedback ----
  // Feedback only re-ranks after a fresh search, so every mutation re-runs it —
  // otherwise the toast claims an effect the results do not show.
  const feedbackDirty = useRef(false);
  function mutateFeedback(next: (fb: FeedbackState) => FeedbackState) {
    feedbackDirty.current = true;
    setFeedback(next);
  }

  function onFeedback(frame: FrameResult, kind: "more" | "exclude") {
    const add = (list: string[], value: string) => Array.from(new Set([...list, value]));
    if (kind === "more") {
      // "More like this FRAME" seeds image-to-image kNN — it no longer promotes
      // the whole video, which is a separate, explicit action.
      mutateFeedback((fb) => ({ ...fb, positive_frames: add(fb.positive_frames, frame.submit_keyframe_id) }));
      setToast({ msg: `Similar to ${frame.submit_keyframe_id}`, kind: "ok" });
      return;
    }
    mutateFeedback((fb) => ({ ...fb, negative_frames: add(fb.negative_frames, frame.submit_keyframe_id) }));
    setToast({ msg: `Excluded ${frame.submit_keyframe_id}`, kind: "ok" });
  }

  function onVideoFeedback(videoId: string, kind: "prioritize" | "deprioritize") {
    const bucket = kind === "prioritize" ? "positive_videos" : "negative_videos";
    const opposite = kind === "prioritize" ? "negative_videos" : "positive_videos";
    mutateFeedback((fb) => ({
      ...fb,
      [bucket]: Array.from(new Set([...(fb[bucket] as string[]), videoId])),
      [opposite]: (fb[opposite] as string[]).filter((id) => id !== videoId),
    }));
    setToast({ msg: `${kind === "prioritize" ? "Prioritized" : "Deprioritized"} ${videoId}`, kind: "ok" });
  }

  function removeFeedback(bucket: keyof FeedbackState, value: string) {
    mutateFeedback((fb) => ({ ...fb, [bucket]: (fb[bucket] as string[]).filter((item) => item !== value) }));
  }

  function clearFeedback() {
    mutateFeedback(() => EMPTY_FEEDBACK);
    setToast({ msg: "Feedback cleared", kind: "ok" });
  }

  function focusQaEvidence(submitKeyframeId: string, showVideo = true) {
    const location = findQaFrame(groups, submitKeyframeId);
    if (!location) return;
    const group = groups[location.video];
    setSelectedVideo(location.video);
    setSelectedFrame(location.frame);
    setExpanded((current) => new Set(current).add(group.video_id));
    // A previous raw pause belongs to a different verification decision, so it
    // is discarded when an NVILA hotspot moves the operator to another frame.
    setPausedFrame(null);
    if (showVideo) {
      setActiveVideoId(group.video_id);
      setVideoVisible(true);
      setShowTimeline(true);
    }
  }

  async function runQaAnalysis() {
    if (queryType !== "QA" || qaCandidates.length === 0 || qaAnalyzing) return;
    const question = parsed?.qa?.question_vi || query.trim();
    if (!question) return;
    setQaAnalyzing(true);
    setQaAnalysisError(null);
    try {
      const result = await api.qaAnalyze({
        question,
        candidates: qaCandidates,
        max_answers: 5,
        web_grounding: qaWebGrounding ? "auto" : "off",
      });
      setQaAnalysis(result);
    } catch (error) {
      const message = error instanceof ApiError && typeof error.detail === "string"
        ? error.detail
        : "NVILA analysis failed — check the Colab worker and backend configuration.";
      setQaAnalysisError(message);
      setToast({ msg: message, kind: "bad" });
    } finally {
      setQaAnalyzing(false);
    }
  }

  function chooseQaAnswer(value: string, evidenceId?: string) {
    setAnswer(value);
    if (evidenceId) focusQaEvidence(evidenceId, true);
  }

  /** Every manual edit of the query box goes through here. */
  const editQuery = useCallback((value: string) => {
    queryTouched.current = true;
    setQuery(value);
  }, []);

  // ---- T-KIS append hint ----
  function appendHint() {
    if (!query.trim()) return;
    setHints((h) => [...h, query.trim()]);
    setQuery("");
    queryRef.current?.focus();
  }

  // ---- Global frame-pick (RAW paused frame, no BTC snapping) ----
  const onVideoPaused = useCallback(
    (rawTime: number, thumbnail: string | null) => {
      if (!activeVideoId) return;
      const fps = timeline?.fps ?? selectedFrameObj?.fps ?? 25;
      const frameIdx = Math.round(rawTime * fps); // frame_idx = round(pts * fps)
      setPausedFrame({
        video_id: activeVideoId,
        frame_idx: frameIdx,
        pts_time: rawTime,
        fps,
        thumbnail,
      });
      // Capturing a frame never arms it for submit on its own — T-KIS/QA/V-KIS
      // submit it from the paused panel's own button, TRAKE from an event slot.
    },
    [queryType, activeVideoId, timeline, selectedFrameObj?.fps],
  );

  const assignPausedFrameToSlot = useCallback(
    (slotIdx: number) => {
      if (!pausedFrame) return;
      setTrakeSlots((slots) => {
        const next = [...slots];
        next[slotIdx] = {
          video_id: pausedFrame.video_id,
          frame_idx: pausedFrame.frame_idx,
          pts_time: pausedFrame.pts_time,
          thumbnail: pausedFrame.thumbnail,
        };
        return next;
      });
      setActiveSlot(Math.min(slotIdx + 1, trakeSlots.length - 1));
      setPausedFrame(null);
    },
    [pausedFrame, trakeSlots.length],
  );

  // Effective frame_idx for a result frame: prefer the exact value from the
  // keyframe map; else derive from pts*fps; null only if neither is available.
  function frameIdxOf(f: FrameResult): number | null {
    if (f.frame_idx != null) return f.frame_idx;
    if (f.pts_time == null) return null;
    const fps = f.fps ?? timeline?.fps ?? 25;
    return Math.round(f.pts_time * fps);
  }
  // frame_idx the guard will actually submit, for the target it was opened with.
  const guardFrameIdx = guardPausedFrame
    ? guardPausedFrame.frame_idx
    : selectedFrameObj
      ? frameIdxOf(selectedFrameObj)
      : null;

  function autoFillSlotsFromSelected() {
    if (!selectedGroup) return;
    const n = trakeSlots.length;
    const next: (TrakeSlot | null)[] = Array(n).fill(null);
    selectedGroup.frames.slice(0, n).forEach((f, i) => {
      next[i] = {
        video_id: f.video_id,
        frame_idx: frameIdxOf(f) ?? 0,
        pts_time: f.pts_time ?? 0,
        thumbnail: f.keyframe_url,
        submit_keyframe_id: f.submit_keyframe_id,
      };
    });
    setTrakeSlots(next);
  }

  const orderViolations = useMemo(
    () => validateIncreasingOrder(trakeSlots.map((s) => (s ? s.frame_idx : null))),
    [trakeSlots],
  );

  useEffect(() => {
    setEventMarkers(
      trakeSlots.flatMap((slot, index) =>
        slot ? [{ eventIndex: index + 1, pts_time: slot.pts_time }] : [],
      ),
    );
  }, [trakeSlots]);

  // Swap two TRAKE slots (drag a keyframe onto another slot to reorder events).
  function moveSlot(from: number, to: number) {
    setTrakeSlots((slots) => {
      const next = [...slots];
      [next[from], next[to]] = [next[to], next[from]];
      return next;
    });
  }

  // TRAKE: fill all event slots from a candidate video's frames, then open guard.
  function trakeQuickSubmit(g: VideoGroup) {
    const slots: (TrakeSlot | null)[] = g.frames.map((f) => ({
      video_id: f.video_id,
      frame_idx: f.frame_idx ?? frameIdxOf(f) ?? 0,
      pts_time: f.pts_time ?? 0,
      thumbnail: f.keyframe_url,
      submit_keyframe_id: f.submit_keyframe_id,
    }));
    setTrakeSlots(slots);
    const gi = groups.indexOf(g);
    if (gi >= 0) { setSelectedVideo(gi); setSelectedFrame(0); }
    // Duplicate check against the new slots (state update is async).
    const key = `${slots[0]?.video_id}|${slots.map((s) => s?.frame_idx).join(",")}`;
    const dup = history.some((h) => h.task_id === taskScope && (h.dedup_keys || []).includes(key));
    setDuplicateId(dup ? key : null);
    setGuardOpen(true);
  }

  // ---- duplicate pre-check (mirrors the backend dedup keys) ----
  // The target is passed in rather than read from state: openGuard sets it in the
  // same tick, so the dedup pre-check would otherwise run against the old value.
  // The backend re-checks authoritatively and its answer arrives with the preview.
  function dedupKeyFor(target: GuardTarget): string | null {
    if (queryType === "TRAKE") {
      const filled = trakeSlots.filter((s): s is TrakeSlot => s !== null);
      if (!filled.length) return null;
      return `${filled[0].video_id}|${filled.map((s) => s.frame_idx).join(",")}`;
    }
    // A pure text answer is identified by the text alone.
    if (answerMode === "text" || (queryType === "QA" && answerMode === "auto" && !selectedFrameObj && !pausedFrame)) {
      const text = answer.trim().replace(/\s+/g, " ").toLowerCase();
      return text ? `text:${text}` : null;
    }
    const frameKey =
      target === "paused"
        ? pausedFrame && `${pausedFrame.video_id}:${pausedFrame.frame_idx}`
        : selectedFrameObj && `${selectedFrameObj.video_id}:${frameIdxOf(selectedFrameObj)}`;
    if (!frameKey) return null;
    // QA answers carry the segment AND the text, so both make up the identity:
    // a new answer on the same frame is a new guess, not a repeat.
    const combined = queryType === "QA" ? answerMode === "auto" : answerMode === "temporal_text";
    if (!combined) return frameKey;
    return `${frameKey}|text:${answer.trim().replace(/\s+/g, " ").toLowerCase()}`;
  }
  function computeDuplicate(target: GuardTarget): string | null {
    const key = dedupKeyFor(target);
    if (!key) return null;
    for (const h of history) {
      if (h.task_id !== taskScope) continue;
      if ((h.dedup_keys || []).includes(key)) return key;
    }
    return null;
  }

  function openGuard(target: GuardTarget = "result") {
    if (queryType === "TRAKE") {
      if (trakeSlots.every((s) => s === null)) {
        setToast({ msg: "No TRAKE frames selected", kind: "bad" });
        return;
      }
    } else if (target === "paused") {
      if (!pausedFrame) return;
    } else if (!selectedFrameObj) {
      return;
    }
    setGuardTarget(target);
    setDuplicateId(computeDuplicate(target));
    setGuardOpen(true);
  }

  /** The request behind both the guard's preview and the actual submit.
   *  DRES v2 needs a millisecond window, so `fps` rides along for the frames
   *  whose pts_time is unknown (the backend derives ms = frame_idx / fps). */
  function buildSubmitBody(): SubmitBody | null {
    const payload: SubmitBody["payload"] = {};
    if (queryType === "TRAKE") {
      const video = trakeSlots.find((s): s is TrakeSlot => s !== null)?.video_id;
      payload.video_id = video;
      payload.fps = timeline?.fps ?? undefined;
      payload.events = trakeSlots
        .map((s, i) => (s ? { event_index: i + 1, frame_idx: s.frame_idx, pts_time: s.pts_time, video_id: s.video_id, submit_keyframe_id: s.submit_keyframe_id } : null))
        .filter((e): e is NonNullable<typeof e> => e !== null);
    } else if (guardPausedFrame || selectedFrameObj) {
      const fi = guardPausedFrame?.frame_idx
        ?? (selectedFrameObj ? frameIdxOf(selectedFrameObj) : null);
      if (fi == null) return null;
      payload.video_id = guardPausedFrame?.video_id ?? selectedFrameObj?.video_id;
      payload.frame_idx = fi;
      payload.timestamp = guardPausedFrame?.pts_time
        ?? selectedFrameObj?.pts_time
        ?? undefined;
      payload.fps = guardPausedFrame?.fps ?? selectedFrameObj?.fps ?? timeline?.fps ?? undefined;
      if (!guardPausedFrame && selectedFrameObj) {
        payload.submit_keyframe_id = selectedFrameObj.submit_keyframe_id;
      }
      if (queryType === "QA") payload.answer = answer;
    } else {
      return null;
    }
    return {
      task_id: "",
      evaluation_id: evaluationId,
      task_name: taskNameOverride.trim() || null,
      query_type: queryType,
      payload,
      answer_mode: answerMode,
      segment_pad_ms: segmentPadMs,
    };
  }

  // While the guard is open, ask the backend for the exact DRES body. This is
  // what turns "hope the format is right" into something the operator can read.
  useEffect(() => {
    if (!guardOpen) {
      setSubmitPreview(null);
      setPreviewError(null);
      return;
    }
    const body = buildSubmitBody();
    if (!body) {
      setSubmitPreview(null);
      setPreviewError("Chưa chọn được frame để nộp.");
      return;
    }
    let cancelled = false;
    setPreviewLoading(true);
    const id = setTimeout(() => {
      api.submitPreview(body)
        .then((p) => {
          if (cancelled) return;
          setSubmitPreview(p);
          setPreviewError(null);
          // The server's dedup verdict wins over the local pre-check.
          setDuplicateId(p.duplicate_key);
        })
        .catch((e) => {
          if (cancelled) return;
          setSubmitPreview(null);
          const detail = e instanceof ApiError ? (e.detail as { message?: string } | null) : null;
          setPreviewError(detail?.message ?? "Không dựng được payload DRES.");
        })
        .finally(() => !cancelled && setPreviewLoading(false));
    }, 150); // debounce: the answer / pad inputs change on every keystroke
    return () => { cancelled = true; clearTimeout(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [guardOpen, guardTarget, queryType, answer, answerMode, segmentPadMs, taskNameOverride,
      evaluationId, currentTaskName, selectedFrameObj, pausedFrame, trakeSlots]);

  async function confirmSubmit() {
    const body = buildSubmitBody();
    if (!body) {
      setToast({ msg: "Không xác định được frame_idx cho frame này — không thể nộp.", kind: "bad" });
      return;
    }
    setSubmitting(true);
    try {
      const entry = await api.submit({ ...body, allow_duplicate: duplicateId != null });
      const verdict = entry.verdict ?? null;
      setToast({
        msg: entry.status === "dres_error"
          ? `DRES lỗi: ${entry.dres?.error ?? "?"}`
          : verdict
          ? `DRES: ${verdict}`
          : entry.status === "dres_ok"
          ? "Submitted to DRES ✓"
          : "Saved locally ✓",
        kind: entry.status === "dres_error" || verdict === "WRONG" ? "bad" : "ok",
      });
      if (verdict === "WRONG") setPenalties((p) => p + 1);
      setGuardOpen(false);
      setDuplicateId(null);
      refreshHistory();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        const detail = e.detail as { submit_keyframe_id?: string };
        setDuplicateId(detail?.submit_keyframe_id || "duplicate");
        setToast({ msg: "Blocked: duplicate submit", kind: "bad" });
      } else if (e instanceof ApiError && e.status === 400) {
        const detail = e.detail as { message?: string } | null;
        setPreviewError(detail?.message ?? "Payload không hợp lệ.");
        setToast({ msg: detail?.message ?? "Payload không hợp lệ", kind: "bad" });
      } else {
        setToast({ msg: "Submit failed", kind: "bad" });
      }
    } finally {
      setSubmitting(false);
    }
  }

  // ---- keyboard ----
  useEffect(() => {
    function isTyping() {
      const el = document.activeElement;
      return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA");
    }
    function onKey(e: KeyboardEvent) {
      // Ctrl/Cmd + / toggles the shortcuts help — works anywhere, even while typing.
      if (e.key === "/" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        setKeymapOpen((o) => !o);
        return;
      }
      if (e.key === "Escape") {
        if (keymapOpen) setKeymapOpen(false);
        else if (guardOpen) setGuardOpen(false);
        else if (pausedFrame) setPausedFrame(null);
        return;
      }
      if (e.key === "Enter" && guardOpen) {
        e.preventDefault();
        if (!submitting) confirmSubmit();
        return;
      }
      if (e.key === "/" && !isTyping()) {
        e.preventDefault();
        queryRef.current?.focus();
        return;
      }
      if (isTyping()) return;

      switch (e.key) {
        case "Enter":
          e.preventDefault();
          if (queryType === "TRAKE") {
            if (pausedFrame) assignPausedFrameToSlot(activeSlot);
            else openGuard("result");
          } else if (e.shiftKey) {
            // Submitting the exact raw frame is a deliberate, separate action —
            // plain Enter always stays on the result keyframe shown in Detail.
            if (pausedFrame) openGuard("paused");
          } else {
            openGuard("result");
          }
          break;
        case "ArrowDown":
          e.preventDefault();
          setSelectedVideo((v) => Math.min(v + 1, groups.length - 1));
          setSelectedFrame(0);
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedVideo((v) => Math.max(v - 1, 0));
          setSelectedFrame(0);
          break;
        // While the neighbour strip is open the arrows browse it; otherwise they
        // keep moving between the retrieved frames of the selected video.
        case "ArrowRight":
          e.preventDefault();
          if (neighborsVisible) moveNeighbor(1);
          else setSelectedFrame((f) => Math.min(f + 1, (selectedGroup?.frames.length ?? 1) - 1));
          break;
        case "ArrowLeft":
          e.preventDefault();
          if (neighborsVisible) moveNeighbor(-1);
          else setSelectedFrame((f) => Math.max(f - 1, 0));
          break;
        case "k":
        case "K":
          e.preventDefault();
          toggleNeighbors();
          break;
        case "v":
        case "V":
          e.preventDefault();
          // 'v' shows the inline video under the selected keyframe (seeked to it);
          // pressing it again on the same video hides it. Space is left to the
          // video's native play/pause control.
          if (!selectedFrameObj || !selectedVideoId) break;
          if (videoVisible && activeVideoId === selectedVideoId) {
            setVideoVisible(false);
            setActiveVideoId(null);
          } else {
            setActiveVideoId(selectedVideoId);
            setVideoVisible(true);
          }
          break;
        case "Tab":
          e.preventDefault();
          setFocusZone((z) => (z === "results" ? "detail" : "results"));
          break;
        case "t":
        case "T":
          setShowTimeline((s) => !s);
          break;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [guardOpen, guardTarget, keymapOpen, pausedFrame, queryType, activeSlot, groups.length, selectedGroup, selectedFrameObj, selectedVideoId, videoVisible, activeVideoId, submitting, assignPausedFrameToSlot, neighborsVisible, neighborKeyframes, neighborIndex, neighborAnchorIndex]);

  function seekVideo(t: number) {
    viewerRef.current?.seek(t);
    setPlayhead(t);
  }

  function copyId(id: string) {
    navigator.clipboard?.writeText(id).then(() => setToast({ msg: `Copied ${id}`, kind: "ok" })).catch(() => {});
  }

  return (
    <div className="app">
      <TopBar
        queryType={queryType}
        onQueryType={(t) => {
          setQueryType(t);
          setParsed(null);
          setPausedFrame(null);
          setQaAnalysis(null);
          setQaAnalysisError(null);
          setAnswer("");
          setCanvasQueries([]);
          // The answer shape belongs to the task type, so it must not survive a
          // switch: a `temporal` override carried into QA silently drops the text.
          setAnswerMode("auto");
        }}
        elapsed={elapsed}
        penalties={penalties}
        latency={latency}
        health={health}
        onSimpleMode={onSimpleMode}
        onShowKeymap={() => setKeymapOpen(true)}
      />
      <DresBar
        status={dresStatus}
        evaluations={dresEvaluations}
        evaluationId={evaluationId}
        pinned={pinnedEvaluationId !== null}
        timeLeft={timeLeft}
        error={dresError}
        busy={dresBusy}
        onSelect={setPinnedEvaluationId}
        onRefresh={() => refreshDresEvaluations(true)}
        onReconnect={reconnectDres}
      />
      {health && !health.ok && health.warnings.length > 0 && (
        <div className="warn-banner">⚠ {health.warnings.join(" · ")} — {health.mode === "mock" ? "running in mock mode" : "live retrieval degraded"}</div>
      )}

      <div className="workspace">
        {/* LEFT */}
        <div className="col col-left">
          <TaskHintPanel
            hint={taskHint}
            loading={taskHintLoading}
            error={taskHintError}
            inQuery={Boolean(taskHint?.text) && query.trim() === taskHint?.text.trim()}
            onUseAsQuery={useHintAsQuery}
            onRefresh={() => fetchTaskHint(true)}
          />
          <QueryPanel
            query={query}
            setQuery={editQuery}
            hints={hints}
            onAppendHint={appendHint}
            onClearHints={() => setHints([])}
            onSearch={runSearch}
            loading={loading}
            parsed={parsed}
            queryType={queryType}
            inputRef={queryRef}
            useLLM={useLLM}
            onToggleLLM={() => setUseLLM((v) => !v)}
            expand={expand}
            onToggleExpand={() => setExpand((v) => !v)}
          />
          <ChannelControls parsed={parsed} overrides={overrides} onToggle={toggleChannel} />
          <QueryUnderstanding parsed={parsed} />
        </div>

        {/* CENTER */}
        <div className="col col-center" style={focusZone === "results" ? { boxShadow: "inset 0 2px 0 var(--accent)" } : undefined}>
          {/* The canvas lives in the widest column: V-KIS drawing accuracy is
              limited by how much room the operator has to draw. */}
          {queryType === "V-KIS" && (
            <CanvasPanel
              onSearch={runCanvasSearch}
              loading={loading}
              generatedQueries={canvasQueries}
              objectSearchAvailable={health ? Boolean(health.capabilities.canvas_object_search) : null}
            />
          )}
          <div className="results-toolbar">
            <span className="results-count">
              {groups.length} video{groups.length === 1 ? "" : "s"} ·{" "}
              {groups.reduce((n, g) => n + g.frame_count, 0)} frames
            </span>
            <div className="seg sm">
              <button
                className={viewMode === "grouped" ? "active" : ""}
                onClick={() => setViewMode("grouped")}
                data-testid="view-grouped"
              >
                Group by video
              </button>
              <button
                className={viewMode === "flat" ? "active" : ""}
                onClick={() => setViewMode("flat")}
                data-testid="view-flat"
              >
                Flat top-K
              </button>
            </div>
          </div>
          <FeedbackBar feedback={feedback} onRemove={removeFeedback} onClear={clearFeedback} />
          <Results
            groups={groups}
            viewMode={viewMode}
            trakeEventCount={queryType === "TRAKE" ? parsed?.trake?.events?.length : undefined}
            onTrakeQuickSubmit={queryType === "TRAKE" ? trakeQuickSubmit : undefined}
            qaHotspotScores={queryType === "QA" ? qaHotspotScores : undefined}
            selectedVideo={selectedVideo}
            selectedFrame={selectedFrame}
            expanded={expanded}
            loading={loading}
            onSelectVideo={(i) => { setSelectedVideo(i); setSelectedFrame(0); }}
            onSelectFrame={(vi, fi) => { setSelectedVideo(vi); setSelectedFrame(fi); }}
            onToggleExpand={(vid) =>
              setExpanded((prev) => {
                const n = new Set(prev);
                n.has(vid) ? n.delete(vid) : n.add(vid);
                return n;
              })
            }
            onFeedback={onFeedback}
            onVideoFeedback={onVideoFeedback}
            videoSlotVideoId={videoVisible || neighborsVisible ? activeVideoId : null}
            videoSlot={
              (videoVisible || neighborsVisible) && activeVideoId && selectedGroup
              && selectedGroup.video_id === activeVideoId ? (
                <>
                  {videoVisible && (
                    <>
                      <VideoViewer
                        ref={viewerRef}
                        src={selectedGroup.video_url}
                        fps={timeline?.fps ?? 25}
                        startTime={selectedFrameObj?.pts_time ?? 0}
                        onPaused={onVideoPaused}
                        onTime={setPlayhead}
                      />
                      {showTimeline && timeline && timeline.video_id === activeVideoId && (
                        <Timeline
                          data={timeline}
                          playhead={playhead}
                          selectedPts={selectedFrameObj?.pts_time ?? null}
                          eventMarkers={eventMarkers}
                          qaHotspots={(qaAnalysis?.hotspots ?? [])
                            .filter((hotspot) => hotspot.video_id === activeVideoId && hotspot.pts_time != null)
                            .map((hotspot) => ({
                              submit_keyframe_id: hotspot.submit_keyframe_id,
                              pts_time: hotspot.pts_time as number,
                              relevance: hotspot.relevance,
                            }))}
                          onSeek={seekVideo}
                        />
                      )}
                    </>
                  )}
                  {/* Always below the player, so pressing V and K in either order
                      gives the same layout. */}
                  {neighborsVisible && (
                    <NeighborStrip
                      keyframes={neighborKeyframes}
                      centerIndex={neighborIndex}
                      anchorId={neighborAnchorId}
                      videoLinked={videoVisible}
                      onPick={pickNeighbor}
                    />
                  )}
                </>
              ) : null
            }
          />
        </div>

        {/* RIGHT */}
        <div className="col col-right" style={focusZone === "detail" ? { boxShadow: "inset 0 2px 0 var(--accent)" } : undefined}>
          {queryType === "QA" && (
            <QaAssistPanel
              analysis={qaAnalysis}
              loading={qaAnalyzing}
              error={qaAnalysisError}
              available={health ? Boolean(health.capabilities.qa_nvila) : null}
              candidateCount={qaCandidates.length}
              candidateFrames={qaCandidateFrames}
              selectedAnswer={answer}
              webGrounding={qaWebGrounding}
              webSearchAvailable={health ? Boolean(health.capabilities.qa_web_grounding) : null}
              visualVerificationAvailable={health ? Boolean(health.capabilities.qa_visual_verification) : null}
              onAnalyze={runQaAnalysis}
              onWebGroundingChange={setQaWebGrounding}
              onChooseAnswer={chooseQaAnswer}
              onOpenFrame={(id) => focusQaEvidence(id, true)}
            />
          )}
          <PausedFramePanel
            frame={pausedFrame}
            queryType={queryType}
            activeTrakeSlot={activeSlot}
            onSubmitPaused={() => openGuard("paused")}
            onAssignToTrake={() => assignPausedFrameToSlot(activeSlot)}
            onClear={() => setPausedFrame(null)}
          />
          {queryType === "TRAKE" && (
            <>
              <div className="panel" style={{ paddingBottom: 0 }}>
                <button className="btn sm" onClick={autoFillSlotsFromSelected} disabled={!selectedGroup}>
                  Auto-fill E1..En from selected video
                </button>
              </div>
              <TrakePanel
                slots={trakeSlots}
                activeSlot={activeSlot}
                hasPausedFrame={pausedFrame !== null}
                violations={orderViolations}
                onSetActive={setActiveSlot}
                onAssignChip={assignPausedFrameToSlot}
                onClearSlot={(i) => setTrakeSlots((s) => s.map((x, idx) => (idx === i ? null : x)))}
                onMoveSlot={moveSlot}
                onAddEvent={() => setTrakeSlots((s) => [...s, null])}
                onRemoveEvent={() => setTrakeSlots((s) => (s.length > 1 ? s.slice(0, -1) : s))}
              />
            </>
          )}
          <DetailPanel
            frame={selectedFrameObj}
            queryType={queryType}
            onSubmit={() => openGuard("result")}
            onCopyId={copyId}
          />
          <HistorySidebar
            history={history}
            taskScope={taskScope}
            taskLabel={submitTaskName}
            onDelete={clearHistory}
          />
        </div>
      </div>

      <SubmitGuard
        open={guardOpen}
        queryType={queryType}
        frame={selectedFrameObj}
        pausedFrame={guardPausedFrame}
        frameIdx={guardFrameIdx}
        trakeSlots={trakeSlots}
        answer={answer}
        setAnswer={setAnswer}
        dresConfigured={Boolean(dresStatus?.configured)}
        evaluationName={activeEvaluation?.name ?? null}
        taskNameOverride={taskNameOverride}
        setTaskNameOverride={setTaskNameOverride}
        answerMode={answerMode}
        setAnswerMode={setAnswerMode}
        segmentPadMs={segmentPadMs}
        setSegmentPadMs={setSegmentPadMs}
        preview={submitPreview}
        previewError={previewError}
        previewLoading={previewLoading}
        duplicateId={duplicateId}
        orderViolations={orderViolations}
        submitting={submitting}
        onConfirm={confirmSubmit}
        onCancel={() => setGuardOpen(false)}
      />

      <ShortcutsModal open={keymapOpen} onClose={() => setKeymapOpen(false)} />

      {toast && <div className={`toast ${toast.kind}`}>{toast.msg}</div>}
    </div>
  );
}
