import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type ManualOverrides } from "./api/client";
import type {
  Channel,
  FeedbackState,
  FrameResult,
  HealthResponse,
  LatencyBreakdown,
  ParsedQuery,
  QaAnalysisResponse,
  QueryType,
  SubmitEntry,
  Timeline as TimelineData,
  VideoGroup,
} from "./api/types";
import { ChannelControls } from "./components/ChannelControls";
import { DetailPanel } from "./components/DetailPanel";
import { HistorySidebar } from "./components/HistorySidebar";
import { QueryPanel } from "./components/QueryPanel";
import { QueryUnderstanding } from "./components/QueryUnderstanding";
import { QaAssistPanel } from "./components/QaAssistPanel";
import { Results } from "./components/Results";
import { PausedFramePanel, type PausedFrame } from "./components/PausedFramePanel";
import { ShortcutsModal } from "./components/ShortcutsModal";
import { SubmitGuard } from "./components/SubmitGuard";
import { Timeline } from "./components/Timeline";
import { TopBar } from "./components/TopBar";
import { TrakePanel, type TrakeSlot } from "./components/TrakePanel";
import { VideoViewer, type VideoViewerHandle } from "./components/VideoViewer";
import { findQaFrame, selectQaCandidateFrames } from "./lib/qa";
import { validateIncreasingOrder } from "./lib/snap";

const EMPTY_OVERRIDES: ManualOverrides = { force_channels: [], disable_channels: [] };
const EMPTY_FEEDBACK: FeedbackState = { positive_videos: [], negative_videos: [], negative_frames: [] };

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

  const [trakeSlots, setTrakeSlots] = useState<(TrakeSlot | null)[]>([null, null]);
  const [activeSlot, setActiveSlot] = useState(0);
  const [pausedFrame, setPausedFrame] = useState<PausedFrame | null>(null);
  const [usePausedFrame, setUsePausedFrame] = useState(false);
  const [eventMarkers, setEventMarkers] = useState<{ eventIndex: number; pts_time: number }[]>([]);

  const [keymapOpen, setKeymapOpen] = useState(false);
  const [guardOpen, setGuardOpen] = useState(false);
  const [taskId, setTaskId] = useState("q001");
  const [answer, setAnswer] = useState("");
  const [qaAnalysis, setQaAnalysis] = useState<QaAnalysisResponse | null>(null);
  const [qaAnalyzing, setQaAnalyzing] = useState(false);
  const [qaAnalysisError, setQaAnalysisError] = useState<string | null>(null);
  const [qaWebGrounding, setQaWebGrounding] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [duplicateId, setDuplicateId] = useState<string | null>(null);

  const [history, setHistory] = useState<SubmitEntry[]>([]);
  const [penalties, setPenalties] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [toast, setToast] = useState<{ msg: string; kind: "ok" | "bad" } | null>(null);

  const queryRef = useRef<HTMLTextAreaElement>(null);
  const viewerRef = useRef<VideoViewerHandle>(null);
  const timelineCache = useRef<Map<string, TimelineData>>(new Map());

  const selectedGroup = groups[selectedVideo] ?? null;
  const selectedFrameObj: FrameResult | null = selectedGroup?.frames[selectedFrame] ?? null;
  const pausedSubmitFrame =
    queryType !== "TRAKE"
    && usePausedFrame
    && pausedFrame
    && pausedFrame.video_id === selectedGroup?.video_id
      ? pausedFrame
      : null;
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
    setUsePausedFrame(false);
    setQaAnalysis(null);
    setQaAnalysisError(null);
    setAnswer("");
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

  // Moving the selection to a different video auto-hides the old inline video
  // (it will reload only when the operator presses 'v' on the new video).
  useEffect(() => {
    if (videoVisible && activeVideoId && selectedVideoId && selectedVideoId !== activeVideoId) {
      setVideoVisible(false);
      setActiveVideoId(null);
    }
    if (pausedFrame && selectedVideoId && pausedFrame.video_id !== selectedVideoId) {
      setPausedFrame(null);
      setUsePausedFrame(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedVideoId]);

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
  function onFeedback(frame: FrameResult, kind: "more" | "exclude") {
    setFeedback((fb) => {
      if (kind === "more") {
        return { ...fb, positive_videos: Array.from(new Set([...fb.positive_videos, frame.video_id])) };
      }
      return { ...fb, negative_frames: Array.from(new Set([...fb.negative_frames, frame.submit_keyframe_id])) };
    });
    setToast({ msg: kind === "more" ? `Boosted ${frame.video_id}` : `Excluded ${frame.submit_keyframe_id}`, kind: "ok" });
  }

  function focusQaEvidence(submitKeyframeId: string, showVideo = true) {
    const location = findQaFrame(groups, submitKeyframeId);
    if (!location) return;
    const group = groups[location.video];
    setSelectedVideo(location.video);
    setSelectedFrame(location.frame);
    setExpanded((current) => new Set(current).add(group.video_id));
    // A previous raw pause belongs to a different verification decision. Once
    // an NVILA hotspot is opened, make its retrieved frame the explicit target.
    setPausedFrame(null);
    setUsePausedFrame(false);
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
      // For single-frame tasks, pausing explicitly arms this exact raw frame as
      // the submit target. TRAKE still requires assigning it to an event slot.
      setUsePausedFrame(queryType !== "TRAKE");
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
      setUsePausedFrame(false);
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
  const selectedFrameIdx = pausedSubmitFrame
    ? pausedSubmitFrame.frame_idx
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
    const dup = history.some((h) => h.task_id === taskId && (h.dedup_keys || []).includes(key));
    setDuplicateId(dup ? key : null);
    setGuardOpen(true);
  }

  // ---- duplicate pre-check (mirrors backend dedup keys: video_id:frame_idx) ----
  function currentDedupKey(): string | null {
    if (queryType === "TRAKE") {
      const filled = trakeSlots.filter((s): s is TrakeSlot => s !== null);
      if (!filled.length) return null;
      return `${filled[0].video_id}|${filled.map((s) => s.frame_idx).join(",")}`;
    }
    if (pausedSubmitFrame) {
      return `${pausedSubmitFrame.video_id}:${pausedSubmitFrame.frame_idx}`;
    }
    if (selectedFrameObj) return `${selectedFrameObj.video_id}:${frameIdxOf(selectedFrameObj)}`;
    return null;
  }
  function computeDuplicate(): string | null {
    const key = currentDedupKey();
    if (!key) return null;
    for (const h of history) {
      if (h.task_id !== taskId) continue;
      if ((h.dedup_keys || []).includes(key)) return key;
    }
    return null;
  }

  function openGuard() {
    if (queryType === "TRAKE") {
      if (trakeSlots.every((s) => s === null)) {
        setToast({ msg: "No TRAKE frames selected", kind: "bad" });
        return;
      }
    } else if (!pausedSubmitFrame && !selectedFrameObj) {
      return;
    }
    setDuplicateId(computeDuplicate());
    setGuardOpen(true);
  }

  async function confirmSubmit() {
    setSubmitting(true);
    try {
      // DRES submits by video_id + frame_idx (frame_idx = round(pts * fps)).
      const payload: Parameters<typeof api.submit>[0]["payload"] = {};
      if (queryType === "TRAKE") {
        const video = trakeSlots.find((s): s is TrakeSlot => s !== null)?.video_id;
        payload.video_id = video;
        payload.events = trakeSlots
          .map((s, i) => (s ? { event_index: i + 1, frame_idx: s.frame_idx, pts_time: s.pts_time, submit_keyframe_id: s.submit_keyframe_id } : null))
          .filter((e): e is NonNullable<typeof e> => e !== null);
      } else if (pausedSubmitFrame || selectedFrameObj) {
        const fi = pausedSubmitFrame?.frame_idx
          ?? (selectedFrameObj ? frameIdxOf(selectedFrameObj) : null);
        if (fi == null) {
          setToast({ msg: "Không xác định được frame_idx cho frame này — không thể nộp.", kind: "bad" });
          setSubmitting(false);
          return;
        }
        payload.video_id = pausedSubmitFrame?.video_id ?? selectedFrameObj?.video_id;
        payload.frame_idx = fi;
        payload.timestamp = pausedSubmitFrame?.pts_time
          ?? selectedFrameObj?.pts_time
          ?? undefined;
        if (!pausedSubmitFrame && selectedFrameObj) {
          payload.submit_keyframe_id = selectedFrameObj.submit_keyframe_id;
        }
        if (queryType === "QA") payload.answer = answer;
      }
      await api.submit({
        task_id: taskId,
        query_type: queryType,
        payload,
        allow_duplicate: duplicateId != null,
      });
      setToast({ msg: "Submitted ✓", kind: "ok" });
      setGuardOpen(false);
      setDuplicateId(null);
      refreshHistory();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        const detail = e.detail as { submit_keyframe_id?: string };
        setDuplicateId(detail?.submit_keyframe_id || "duplicate");
        setPenalties((p) => p + 1);
        setToast({ msg: "Blocked: duplicate submit", kind: "bad" });
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
        else if (pausedFrame) {
          setPausedFrame(null);
          setUsePausedFrame(false);
        }
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
          if (queryType === "TRAKE" && pausedFrame) assignPausedFrameToSlot(activeSlot);
          else openGuard();
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
        case "ArrowRight":
          e.preventDefault();
          setSelectedFrame((f) => Math.min(f + 1, (selectedGroup?.frames.length ?? 1) - 1));
          break;
        case "ArrowLeft":
          e.preventDefault();
          setSelectedFrame((f) => Math.max(f - 1, 0));
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
  }, [guardOpen, keymapOpen, pausedFrame, queryType, activeSlot, groups.length, selectedGroup, selectedFrameObj, selectedVideoId, videoVisible, activeVideoId, submitting, assignPausedFrameToSlot]);

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
          setUsePausedFrame(false);
          setQaAnalysis(null);
          setQaAnalysisError(null);
          setAnswer("");
        }}
        elapsed={elapsed}
        penalties={penalties}
        latency={latency}
        health={health}
        onSimpleMode={onSimpleMode}
        onShowKeymap={() => setKeymapOpen(true)}
      />
      {health && !health.ok && health.warnings.length > 0 && (
        <div className="warn-banner">⚠ {health.warnings.join(" · ")} — {health.mode === "mock" ? "running in mock mode" : "live retrieval degraded"}</div>
      )}

      <div className="workspace">
        {/* LEFT */}
        <div className="col col-left">
          <QueryPanel
            query={query}
            setQuery={setQuery}
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
            videoSlotVideoId={videoVisible ? activeVideoId : null}
            videoSlot={
              videoVisible && activeVideoId && selectedGroup && selectedGroup.video_id === activeVideoId ? (
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
              googleAvailable={health ? Boolean(health.capabilities.qa_google_grounding) : null}
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
            activeForSubmit={pausedSubmitFrame !== null}
            activeTrakeSlot={activeSlot}
            onUseForSubmit={() => setUsePausedFrame(true)}
            onUseResultFrame={() => setUsePausedFrame(false)}
            onAssignToTrake={() => assignPausedFrameToSlot(activeSlot)}
            onClear={() => {
              setPausedFrame(null);
              setUsePausedFrame(false);
            }}
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
            usingPausedFrame={pausedSubmitFrame !== null}
            onSubmit={openGuard}
            onCopyId={copyId}
          />
          <HistorySidebar history={history} />
        </div>
      </div>

      <SubmitGuard
        open={guardOpen}
        queryType={queryType}
        frame={selectedFrameObj}
        pausedFrame={pausedSubmitFrame}
        frameIdx={selectedFrameIdx}
        trakeSlots={trakeSlots}
        taskId={taskId}
        setTaskId={setTaskId}
        answer={answer}
        setAnswer={setAnswer}
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
