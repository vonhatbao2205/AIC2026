import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type ManualOverrides, type SubmitBody } from "./api/client";
import type {
  AgentCandidate,
  AnswerMode,
  Channel,
  DresEvaluation,
  DresStatus,
  DresTaskHint,
  FeedbackState,
  FrameResult,
  HealthResponse,
  ImageEmbeddingModel,
  LatencyBreakdown,
  ParsedQuery,
  PeQueryReport,
  ProgressiveSnapshot,
  QaAnalysisResponse,
  QueryType,
  ResolvedScope,
  RetrievalDatabase,
  ScopeCatalogue,
  ScopeMode,
  SubmitEntry,
  SubmitPreview,
  Timeline as TimelineData,
  TrafficFilterInfo,
  TrafficFilterMode,
  TrakeHeatPeak,
  TrakeVideoResult,
  VideoGroup,
} from "./api/types";
import { AgentPanel } from "./components/AgentPanel";
import { AvsPanel } from "./components/AvsPanel";
import { CanvasPanel } from "./components/CanvasPanel";
import { ChannelControls } from "./components/ChannelControls";
import { DetailPanel } from "./components/DetailPanel";
import { DresBar } from "./components/DresBar";
import { QuestionBar } from "./components/QuestionBar";
import { FeedbackBar } from "./components/FeedbackBar";
import { HistorySidebar } from "./components/HistorySidebar";
import { ImageModelSelector } from "./components/ImageModelSelector";
import { NeighborStrip } from "./components/NeighborStrip";
import { ProgressivePanel } from "./components/ProgressivePanel";
import { QueryPanel } from "./components/QueryPanel";
import { QueryUnderstanding } from "./components/QueryUnderstanding";
import { QaAssistPanel } from "./components/QaAssistPanel";
import { Results } from "./components/Results";
import { TrakeVideoResults, type TrakeChainOverrides } from "./components/TrakeVideoResults";
import { ScopeFilter } from "./components/ScopeFilter";
import { PausedFramePanel, type PausedFrame } from "./components/PausedFramePanel";
import { ShortcutsModal } from "./components/ShortcutsModal";
import { TaskHintPanel } from "./components/TaskHintPanel";
import { SubmitGuard } from "./components/SubmitGuard";
import { Timeline } from "./components/Timeline";
import { TopBar } from "./components/TopBar";
import { TrakePanel, type TrakeSlot } from "./components/TrakePanel";
import type { TrakePeakDrag } from "./lib/trakeDrag";
import { VideoViewer, type VideoViewerHandle } from "./components/VideoViewer";
import { useAgentRun } from "./hooks/useAgentRun";
import { AGENT_NAMES, candidateTime, readAgentPreference, writeAgentPreference } from "./lib/agent";
import { autoEvaluationId } from "./lib/dres";
import { sortFramesByTime, swapVideoOrigin } from "./lib/media";
import { findQaFrame, selectQaCandidateFrames } from "./lib/qa";
import { kindForQueryType, type ImportedQuestion } from "./lib/questions";
import { DEFAULT_IMAGE_MODELS, imageModelsForSearch } from "./lib/imageModels";
import { DEFAULT_SCOPE_MODE, orderCategories, scopeRequest } from "./lib/scope";
import { VIDEO_COARSE_STEP_S, VIDEO_FINE_STEP_S, seekDeltaForKey } from "./lib/videoSeek";
import { validateIncreasingOrder } from "./lib/snap";
import { ownsSketchKey } from "./lib/sketch";
import { extractVideoThumbnails } from "./lib/videoThumbnail";
import { MAX_ROWS_PER_QUESTION, findDuplicate, rowToCsvLine, type SubmissionRow } from "./lib/submission";
import type { NoteCandidate } from "./lib/stickyNotes";

/** What a submit hands to the Workspace to become one CSV row. */
export interface SubmissionDraft {
  questionId: string;
  videoId: string;
  frames: number[];
  answer: string;
  // Preview/provenance, positionally parallel to `frames`. A frame captured
  // from a paused video has no extracted keyframe, hence the nulls; `frames`
  // stays the only value the CSV is built from.
  keyframeIds: (string | null)[];
  ptsTimes: (number | null)[];
  retrievalDatabase: RetrievalDatabase;
}

export type StickyAddResult = "added" | "full";

/** A one-shot command from the floating Sticky window back into this Search
 *  tab. The id prevents a parent re-render from restoring the same sequence a
 *  second time. */
export interface StickyTrakeRestoreRequest {
  id: number;
  candidate: NoteCandidate;
}

/** The submit guard acts on exactly one of these; they never shadow each other. */
type GuardTarget = "result" | "paused";

/** Keyframes a search retrieves before fusion, and the slider's starting point. */
const DEFAULT_TOP_K = 100;
/** TRAKE ranks a per-event candidate pool and the assembly needs it wide, so the
 *  slider may raise that pool but never drops it below the working default. */
const TRAKE_MIN_TOP_K = 400;

/** An animated jump is easier to follow than a teleport, but it is exactly the
 *  thing the OS setting exists to switch off. jsdom implements no `matchMedia`,
 *  so the absence of the API is treated as "no preference stated". */
function prefersReducedMotion(): boolean {
  return typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

const EMPTY_OVERRIDES: ManualOverrides = { force_channels: [], disable_channels: [] };
const EMPTY_FEEDBACK: FeedbackState = {
  positive_videos: [], negative_videos: [], positive_frames: [], negative_frames: [],
};

interface ConsoleProps {
  onSimpleMode: () => void;
  onShowSettings: () => void;
  /** Changes when the operator imports a new config; re-reads service health. */
  configVersion?: number;
  retrievalDatabase: RetrievalDatabase;
  onRetrievalDatabase: (database: RetrievalDatabase) => void;
  // ---- multi-tab wiring (owned by Workspace) ----
  /** False while this tab is parked behind another one. */
  active: boolean;
  /** Controlled so the tab rail can label the tab with it in real time. */
  queryType: QueryType;
  onQueryType: (queryType: QueryType) => void;
  questions: ImportedQuestion[];
  question: ImportedQuestion | null;
  onSelectQuestion: (questionId: string | null) => void;
  onImportQuestions: (file: File) => void;
  importing: boolean;
  importError: string | null;
  onOpenSubmission: () => void;
  onSearchAll: () => void;
  searchingAll: boolean;
  /** Rows already collected for this tab's question, for the duplicate check. */
  questionRows: SubmissionRow[];
  onSubmitRow: (draft: SubmissionDraft) => void;
  /** Save locally without touching the shared Submission table. */
  onAddStickyRow: (draft: SubmissionDraft) => StickyAddResult;
  onToggleSticky: () => void;
  stickyOpen: boolean;
  stickyCount: number;
  stickyTrakeRestore: StickyTrakeRestoreRequest | null;
  onStickyTrakeRestoreDone: (id: number) => void;
  /** Bumped by an import to make this tab search on its own. */
  autoRunToken: number;
  onBusyChange: (busy: boolean) => void;
  /** Called once an auto-run has finished, however it ended.
   *
   *  The parent runs these one at a time, and it cannot infer the end from
   *  `onBusyChange`: `runSearch` flips loading true then false around an await,
   *  which React coalesces into a single render when the response is already
   *  in hand, so the `true` is never observed. */
  onAutoRunDone: () => void;
}

export default function FullConsole({
  onSimpleMode, onShowSettings, configVersion = 0, retrievalDatabase, onRetrievalDatabase,
  active, queryType, onQueryType, questions, question, onSelectQuestion, onImportQuestions,
  importing, importError, onOpenSubmission, onSearchAll, searchingAll, questionRows, onSubmitRow,
  onAddStickyRow, onToggleSticky, stickyOpen, stickyCount,
  stickyTrakeRestore, onStickyTrakeRestoreDone,
  autoRunToken, onBusyChange, onAutoRunDone,
}: ConsoleProps) {
  // `runSearch` shadows this with an explicit override, so the state itself is
  // kept under a distinct name and re-exported for every other reader.
  const [queryState, setQuery] = useState("");
  const query = queryState;
  const [progressive, setProgressive] = useState(false);
  const [progressiveHybrid, setProgressiveHybrid] = useState(false);
  const progressiveActive = progressive && queryType === "T-KIS";
  const searchOwner = useRef(0);
  const [hints, setHints] = useState<string[]>([]);
  const [parsed, setParsed] = useState<ParsedQuery | null>(null);
  const [overrides, setOverrides] = useState<ManualOverrides>(EMPTY_OVERRIDES);
  const [useLLM, setUseLLM] = useState(false);
  // Retrieval depth. `topK` is what the slider holds; `appliedTopK` is what the
  // results on screen were fetched with. Keeping them apart is what makes the
  // slider inert until Search is pressed.
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [appliedTopK, setAppliedTopK] = useState<number | null>(null);
  const [expand, setExpand] = useState(false);
  // Qwen3-VL reranking. Default OFF: it needs a second GPU worker running
  // and costs seconds per search, so the operator opts in per query.
  const [rerank, setRerank] = useState(false);
  // VI→EN translation of the visual query. Default ON: PE-Core is an
  // English-centric encoder, so a Vietnamese query it never sees translated
  // returns plausible-looking keyframes that do not match. Off is the escape
  // hatch for the queries where translating is the wrong move — already
  // English, a proper noun, or a phrase the translator keeps mangling — and it
  // governs voice input too, so dictation stops rewriting what was said.
  const [translate, setTranslate] = useState(true);
  // Codex + Claude sidecar agents (the AGENT button). ON by default, remembered
  // per browser. They run next to the main search and report into their own
  // panel: nothing they return is ever written into `groups`.
  const [agentOn, setAgentOn] = useState(readAgentPreference);
  const { run: agentRun, error: agentError, starting: agentStarting,
    start: startAgents, cancel: stopAgents, clear: clearAgents } = useAgentRun();
  // Kept per console tab: two simultaneous tasks may intentionally search
  // different InfoShot++ indices. BTC is normalized to PE-only at request time.
  const [imageModels, setImageModels] = useState<ImageEmbeddingModel[]>(() => [...DEFAULT_IMAGE_MODELS]);
  // Search scope: which dataset folders may answer. `scopeMode` is the policy,
  // `scopeSelection` the hand-picked list it uses in "manual", and `appliedScope`
  // what the last search actually ran on (the backend resolves "auto", so only
  // it can say). Like `topK`, none of them re-run a search on their own.
  const [scopeMode, setScopeMode] = useState<ScopeMode>(DEFAULT_SCOPE_MODE);
  const [scopeSelection, setScopeSelection] = useState<string[]>([]);
  const [scopeCatalogue, setScopeCatalogue] = useState<ScopeCatalogue | null>(null);
  const [scopeError, setScopeError] = useState<string | null>(null);
  const [appliedScope, setAppliedScope] = useState<ResolvedScope | null>(null);
  // Camera / date / time / race-stage cues of the query. `trafficMode` is the
  // policy for the next search, `appliedTraffic` what the last one restricted —
  // the backend reads the cues, so only it can say. Neither re-runs a search.
  const [trafficMode, setTrafficMode] = useState<TrafficFilterMode>("auto");
  const [appliedTraffic, setAppliedTraffic] = useState<TrafficFilterInfo | null>(null);
  // What the last search actually sent to PE-Core, counted exactly (70-token window).
  const [peReport, setPeReport] = useState<PeQueryReport | null>(null);
  const [groups, setGroups] = useState<VideoGroup[]>([]);
  // Videos whose frame strip is shown chronologically instead of by relevance.
  // The retrieval ranking in `groups` is never mutated, so "undo" is just
  // dropping the id — there is no saved copy that could go stale.
  // Frames read chronologically by DEFAULT — the strip is a scene, and a scene
  // out of order is far harder to judge than a ranking out of order. This holds
  // the EXCEPTIONS: videos the operator put back into relevance order.
  const [byRelevance, setByRelevance] = useState<Set<string>>(new Set());
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
  const [clipSeek, setClipSeek] = useState<{ videoId: string; time: number } | null>(null);
  // Neighbour browser: the selected result keyframe is the anchor and the offset
  // is how far the operator has paged from it. Storing an offset (not an index)
  // keeps the state valid while the timeline is still loading.
  const [neighborsVisible, setNeighborsVisible] = useState(false);
  const [neighborOffset, setNeighborOffset] = useState(0);

  const [trakeSlots, setTrakeSlots] = useState<(TrakeSlot | null)[]>([null, null]);
  const [activeSlot, setActiveSlot] = useState(0);
  // The video-centric TRAKE result: every event's candidates per video, not just
  // the one chain per video the generic group shape can carry.
  const [trakeVideos, setTrakeVideos] = useState<TrakeVideoResult[]>([]);
  // The TRAKE moment the operator is verifying. A heat peak is usually an
  // ALTERNATIVE the chain did not take, so no result frame points at it: without
  // this the player went to the peak while the detail panel, the timeline marker
  // and the neighbour anchor still described the chain's frame. The video id
  // travels with it — a pin left over from another video would seek the newly
  // opened one to a timestamp that means nothing there.
  const [trakePeak, setTrakePeak] = useState<{ videoId: string; peak: TrakeHeatPeak } | null>(null);
  // Frames the operator dragged into a video's chain, replacing the DP's pick.
  // Kept per video so moving between cards does not lose the work, and kept
  // OUTSIDE the backend result so `↺` can always hand the DP's answer back.
  const [chainOverrides, setChainOverrides] = useState<TrakeChainOverrides>({});
  const [pausedFrame, setPausedFrame] = useState<PausedFrame | null>(null);
  // Which frame the OPEN guard is about to submit. Capturing a paused frame no
  // longer re-points the detail panel: the two submit paths stay independent and
  // the target is decided by which button (or shortcut) opened the guard.
  const [guardTarget, setGuardTarget] = useState<GuardTarget>("result");
  const [eventMarkers, setEventMarkers] = useState<{ eventIndex: number; pts_time: number }[]>([]);

  const [keymapOpen, setKeymapOpen] = useState(false);
  const [guardOpen, setGuardOpen] = useState(false);
  const [answer, setAnswer] = useState("");

  // Restore a whole saved sequence, then hydrate every slot with an image. Raw
  // thumbnails are re-extracted from the video at their exact stored pts_time;
  // the data URLs live only in these slots and never enter Sticky localStorage.
  useEffect(() => {
    if (!stickyTrakeRestore) return;
    const { id, candidate } = stickyTrakeRestore;
    let cancelled = false;
    const expected = question?.eventCount ?? candidate.frames.length;
    if (queryType !== "TRAKE" || question?.kind !== "trake") {
      setToast({ msg: "Candidates can only be loaded into a TRAKE tab.", kind: "bad" });
      onStickyTrakeRestoreDone(id);
      return;
    }
    if (candidate.frames.length !== expected || candidate.frames.some((frame) => !Number.isFinite(frame))) {
      setToast({ msg: `Candidate requires exactly ${expected} TRAKE frames.`, kind: "bad" });
      onStickyTrakeRestoreDone(id);
      return;
    }
    const knownTimes = candidate.ptsTimes.map((value) =>
      typeof value === "number" && Number.isFinite(value) ? value : null,
    );
    const orderValues = candidate.frames.map((frame, index) => knownTimes[index] ?? frame / 25);
    if (validateIncreasingOrder(orderValues).length) {
      setToast({ msg: "Cannot load: the Sticky TRAKE sequence is not in chronological order.", kind: "bad" });
      onStickyTrakeRestoreDone(id);
      return;
    }

    const baseSlots: TrakeSlot[] = candidate.frames.map((frame, index) => ({
      video_id: candidate.videoId,
      frame_idx: frame,
      pts_time: knownTimes[index] ?? frame / 25,
      ...(candidate.keyframeIds[index]
        ? { submit_keyframe_id: candidate.keyframeIds[index] as string }
        : {}),
    }));
    setTrakeSlots(baseSlots);
    setActiveSlot(0);
    setPausedFrame(null);
    setToast({ msg: `Loading ${baseSlots.length} TRAKE images from Sticky…`, kind: "ok" });

    void (async () => {
      let failed = 0;
      const rawIndices = candidate.keyframeIds.flatMap((keyframeId, index) =>
        keyframeId ? [] : [index],
      );
      const timelinePromise = rawIndices.length
        ? api.timeline(candidate.videoId, candidate.retrievalDatabase)
        : null;
      const exactKeyframes = await Promise.all(baseSlots.map(async (slot, index): Promise<TrakeSlot> => {
        const keyframeId = candidate.keyframeIds[index] ?? null;
        if (!keyframeId) return slot;
        try {
          const info = await api.keyframe(keyframeId, candidate.retrievalDatabase);
          return { ...slot, thumbnail: info.keyframe_url, thumbnail_kind: "exact" };
        } catch {
          failed += 1;
          return slot;
        }
      }));
      let hydrated = exactKeyframes;
      if (timelinePromise) {
        try {
          const videoTimeline = await timelinePromise;
          const fps = videoTimeline.fps || 25;
          const rawTimes = rawIndices.map((index) =>
            knownTimes[index] ?? candidate.frames[index] / fps,
          );
          const rawThumbnails = await extractVideoThumbnails(videoTimeline.video_url, rawTimes);
          hydrated = hydrated.map((slot, index) => {
            const rawPosition = rawIndices.indexOf(index);
            if (rawPosition < 0) return slot;
            const thumbnail = rawThumbnails[rawPosition] ?? null;
            if (!thumbnail) failed += 1;
            return {
              ...slot,
              pts_time: rawTimes[rawPosition],
              thumbnail,
              thumbnail_kind: thumbnail ? "exact-raw" : undefined,
            };
          });
        } catch {
          failed += rawIndices.length;
        }
      }
      if (cancelled) return;
      setTrakeSlots((current) => {
        const sameSequence = current.length === baseSlots.length && current.every(
          (slot, index) =>
            slot?.video_id === baseSlots[index].video_id &&
            slot?.frame_idx === baseSlots[index].frame_idx,
        );
        return sameSequence ? hydrated : current;
      });
      setToast({
        msg: failed
          ? `TRAKE loaded; ${failed} preview images could not be loaded.`
          : `Loaded ${hydrated.length} TRAKE events with exact frame images.`,
        kind: failed ? "bad" : "ok",
      });
      onStickyTrakeRestoreDone(id);
    })();

    return () => {
      cancelled = true;
    };
  }, [onStickyTrakeRestoreDone, question?.eventCount, question?.kind, queryType, stickyTrakeRestore]);

  // ---- DRES (official evaluation server) ----
  // The run and the open task come from the server; `pinnedEvaluationId` is set
  // only when the operator overrides the auto-routing by query type.
  const [avsRequested, setAvsRequested] = useState<{ frame: FrameResult; id: number; scope: string } | null>(null);
  const [dresStatus, setDresStatus] = useState<DresStatus | null>(null);
  // On by default: the final round is judged live on DRES. Unticking it falls
  // back to collecting rows for the CSV pack; it only exists when the backend
  // reports DRES configured (DRES_ENABLED + credentials).
  const [dresRequested, setDresRequested] = useState(true);
  const dresEnabled = dresRequested && Boolean(dresStatus?.configured);
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

  const [history, setHistory] = useState<SubmitEntry[]>([]);
  const [penalties, setPenalties] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [toast, setToast] = useState<{ msg: string; kind: "ok" | "bad" } | null>(null);

  const queryRef = useRef<HTMLTextAreaElement>(null);
  // Set as soon as the operator edits the query themselves; the task statement
  // prefill must never clobber their words.
  const queryTouched = useRef(false);
  const viewerRef = useRef<VideoViewerHandle>(null);
  const videoPanelRef = useRef<HTMLDivElement>(null);
  // Set once a video has actually failed on the primary origin. After that every
  // new video starts on the fallback instead of spending a failed request per
  // clip rediscovering that the origin is still down.
  const [videoOriginDown, setVideoOriginDown] = useState(false);
  const timelineCache = useRef<Map<string, TimelineData>>(new Map());

  // What the results column actually shows. Everything that resolves the
  // selection reads THIS, not `groups`: `selectedFrame` is an index into the
  // rendered strip, so a reordered strip and a selection indexing the original
  // ranking would point at two different frames — including at submit time.
  // The player is rendered inside the result card it belongs to, so on a video
  // whose results run past the fold it opened somewhere below the viewport and
  // the operator had to scroll to find it — with 400 keyframes in one group,
  // far enough to cost real seconds under the clock. Bring it into view instead.
  // Also covers the neighbour strip (`k`), which lives in the same panel.
  useEffect(() => {
    if (!videoVisible && !neighborsVisible) return;
    // jsdom has no layout and therefore no scrollIntoView; a missing scroll must
    // never take the console down with it.
    videoPanelRef.current?.scrollIntoView?.({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: "center",
    });
  }, [videoVisible, neighborsVisible, activeVideoId]);

  // TRAKE is excluded outright: there each frame IS event i, and the DP already
  // guarantees increasing time, so sorting the strip could only ever confuse the
  // mapping without changing it.
  const chronological = queryType !== "TRAKE";
  const displayGroups = useMemo(
    () =>
      !chronological
        ? groups
        : groups.map((group) =>
            (group.progressive || byRelevance.has(group.video_id))
              ? group
              : { ...group, frames: sortFramesByTime(group.frames) },
          ),
    [groups, byRelevance, chronological],
  );

  const selectedGroup = displayGroups[selectedVideo] ?? null;
  // A verified TRAKE peak IS the current frame while it is pinned, so everything
  // that reads the selection — player start time, timeline marker, detail panel,
  // neighbour anchor — describes the same moment instead of disagreeing.
  const trakePeakFrame: FrameResult | null =
    trakePeak && selectedGroup && trakePeak.videoId === selectedGroup.video_id
      ? {
          image_id: trakePeak.peak.submit_keyframe_id,
          submit_keyframe_id: trakePeak.peak.submit_keyframe_id,
          video_id: trakePeak.videoId,
          keyframe_n: trakePeak.peak.keyframe_n,
          frame_idx: trakePeak.peak.frame_idx,
          fps: null,
          pts_time: trakePeak.peak.pts_time,
          score: trakePeak.peak.score,
          channels: [],
          per_channel_score: {},
          keyframe_url: trakePeak.peak.keyframe_url,
          video_url: selectedGroup.video_url,
          evidence: [],
        }
      : null;
  const selectedFrameObj: FrameResult | null =
    trakePeakFrame ?? selectedGroup?.frames[selectedFrame] ?? null;
  // How many events the statement asked for; the slots are the fallback when the
  // parser returned none, because a row of the wrong width is a rejected row.
  const trakeEventCount = parsed?.trake?.events?.length ?? 0;
  // The video-centric TRAKE view is used whenever the backend sent one. An older
  // backend (or a search that predates it) still gets the generic group list.
  const trakeVideoView = queryType === "TRAKE" && trakeVideos.length > 0;
  // Only the guard reads this. The detail panel always describes the selected
  // result keyframe, whether or not a raw frame is currently captured.
  const guardPausedFrame =
    queryType !== "TRAKE" && guardTarget === "paused" ? pausedFrame : null;
  const qaCandidateFrames = useMemo(
    () => (queryType === "QA" ? selectQaCandidateFrames(displayGroups, selectedFrameObj) : []),
    [queryType, displayGroups, selectedFrameObj],
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
  // Re-runs on a config import: the backend swapped its whole service stack, so
  // the capability flags this UI hides features behind are stale.
  // Gated on `active`: every tab of an imported pack stays mounted, so without
  // this a 25-question import fired 25 /api/health calls at once, each fanning
  // out to every upstream service. That alone could time out Elastic and the
  // Colab tunnels at the exact moment the searches needed them.
  useEffect(() => {
    if (!active) return;
    api.health(retrievalDatabase).then(setHealth).catch(() => setHealth(null));
    refreshHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, configVersion, retrievalDatabase]);

  // A worker that went away takes its tick box with it; untick so the state the
  // operator can no longer see does not keep riding along on every request.
  useEffect(() => {
    if (health && !health.capabilities.visual_rerank) setRerank(false);
  }, [health]);

  useEffect(() => {
    timelineCache.current.clear();
    setGroups([]);
    setTrakeVideos([]);
    setTrakePeak(null);
    setChainOverrides({});
    setParsed(null);
    setVideoOriginDown(false);
    setTimeline(null);
    setActiveVideoId(null);
    setVideoVisible(false);
    setFeedback(EMPTY_FEEDBACK);
    setOverrides(EMPTY_OVERRIDES);
  }, [retrievalDatabase]);

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
        msg: r.deleted ? `Deleted ${r.deleted} entries (backup ${r.backup})` : "No entries to delete",
        kind: "ok",
      });
    } catch {
      setToast({ msg: "Could not delete history", kind: "bad" });
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
  }, [refreshDresEvaluations, configVersion]);

  // The open task changes without warning during a run, so poll it. 5 s is well
  // under the shortest task and the backend caches for 2 s.
  useEffect(() => {
    if (!dresEnabled || !dresStatus?.configured) return;
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

  const avsScope = `${retrievalDatabase}:${evaluationId ?? "local"}:${currentTaskName || question?.id || "manual"}`;

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
        setTaskHintError(e instanceof ApiError ? String(e.detail ?? e.message) : "Could not retrieve the task description");
      } finally {
        setTaskHintLoading(false);
      }
    },
    [evaluationId, queryType],
  );

  useEffect(() => {
    if (!dresEnabled || !dresStatus?.configured || !evaluationId) {
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

  // ---- the imported question this tab answers ----
  // Assigning a question loads its statement, exactly like a DRES task statement
  // used to. Unlike that path this one always wins: picking a question IS the
  // operator asking for its text, so it is not held back by `queryTouched`.
  const loadedQuestionId = useRef<string | null>(null);
  useEffect(() => {
    if (!question || loadedQuestionId.current === question.id) return;
    loadedQuestionId.current = question.id;
    queryTouched.current = false;
    setQuery(question.text);
    setHints([]);
    setGroups([]);
    setTrakeVideos([]);
    setTrakePeak(null);
    setChainOverrides({});
    setParsed(null);
  }, [question]);

  // Import-driven search: the Workspace bumps the token after it has assigned the
  // question, and the statement rides along so the run does not race the state
  // update that fills the query box.
  const lastAutoRun = useRef(0);
  useEffect(() => {
    if (!autoRunToken || autoRunToken === lastAutoRun.current) return;
    lastAutoRun.current = autoRunToken;
    const text = question?.text?.trim();
    // Report completion even when there is nothing to search, or a question
    // with an empty statement would stall the queue behind it forever.
    if (!text) {
      onAutoRunDone();
      return;
    }
    void runSearch(text).finally(onAutoRunDone);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRunToken, question?.id]);

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
  }, [refreshDresEvaluations, configVersion]);

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
    api.timeline(activeVideoId, retrievalDatabase).then((tl) => {
      if (cancelled) return;
      timelineCache.current.set(activeVideoId, tl);
      setTimeline(tl);
    }).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [activeVideoId, retrievalDatabase]);

  // ---- folder catalogue for the scope filter ----
  // Fixed per profile, so it is fetched once per profile rather than per search.
  // Switching profile also drops folders the new one does not hold (InfoShot++
  // has no K-side), which otherwise would silently filter every search to zero.
  useEffect(() => {
    let cancelled = false;
    setScopeError(null);
    // The applied scope was resolved against the other profile's folders.
    setAppliedScope(null);
    setAppliedTraffic(null);
    api.searchScope(retrievalDatabase).then(
      (body) => {
        if (cancelled) return;
        setScopeCatalogue(body);
        const held = new Set(body.categories.map((item) => item.category));
        setScopeSelection((current) => {
          const kept = current.filter((category) => held.has(category));
          return kept.length === current.length ? current : kept;
        });
      },
      () => {
        if (!cancelled) setScopeError("Could not load folders; searching the entire dataset for now.");
      },
    );
    return () => {
      cancelled = true;
    };
  }, [retrievalDatabase]);

  // Task/config changes own a new session and invalidate every older search.
  const progressiveKey = JSON.stringify([question?.id, queryType, retrievalDatabase, imageModels, translate, progressiveHybrid, configVersion, progressiveActive]);
  useLayoutEffect(() => {
    searchOwner.current++;
    setLoading(false);
    if (progressiveActive) {
      setGroups([]); setParsed(null); setLatency(null);
      setSelectedVideo(0); setSelectedFrame(0); setClipSeek(null);
      setGuardOpen(false); setPausedFrame(null);
    }
  }, [progressiveKey]);

  const acceptProgressive = (res: ProgressiveSnapshot) => {
    const videoId = selectedGroup?.video_id;
    const frameId = selectedFrameObj?.submit_keyframe_id;
    const vi = res.groups.findIndex(g => g.video_id === videoId);
    const fi = vi >= 0 ? res.groups[vi].frames.findIndex(f => f.submit_keyframe_id === frameId) : -1;
    setGroups(res.groups);
    if (res.query) setQuery(res.query);
    setHints([]);
    setParsed(res.parsed ?? null);
    setLatency(null);
    setAppliedScope(res.scope ?? null);
    setSelectedVideo(vi >= 0 ? vi : 0);
    setSelectedFrame(fi >= 0 ? fi : 0);
    setClipSeek(null); setPausedFrame(null); setGuardOpen(false);
    setByRelevance(new Set(res.groups.map(g => g.video_id)));
  };

  // ---- search ----
  const runSearch = useCallback(async (overrideQuery?: string) => {
    if (progressiveActive) return; // PHM uses its explicit ledger Apply action.
    // The override exists for the import-driven run, which fires in the same tick
    // as the setQuery that fills the box and would otherwise read the old value.
    const query = typeof overrideQuery === "string" ? overrideQuery : queryState;
    if (!query.trim() && hints.length === 0) return;
    const owner = ++searchOwner.current;
    setLoading(true);
    setAppliedTopK(topK);
    // A new result set is a new ranking; carrying the old per-video sort into it
    // would silently reorder groups the operator never asked to reorder.
    setByRelevance(new Set());
    setDuplicateId(null);
    setPausedFrame(null);
    setTrakeVideos([]);
    setTrakePeak(null);
    setChainOverrides({});
    setQaAnalysis(null);
    setQaAnalysisError(null);
    setAnswer("");
    try {
      if (queryType === "TRAKE") {
        const res = await api.searchTrake({ retrieval_database: retrievalDatabase, image_models: imageModelsForSearch(retrievalDatabase, imageModels), query, scope: scopeRequest(scopeMode, scopeSelection), traffic: trafficMode, previous_hints: hints, manual_overrides: overrides, use_llm: useLLM, expand, translate, top_k: Math.max(topK, TRAKE_MIN_TOP_K) });
        if (owner !== searchOwner.current) return;
        setParsed(res.parsed);
        setAppliedScope(res.scope ?? null);
        setAppliedTraffic(res.traffic ?? null);
        setPeReport(res.pe_tokens ?? null);
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
        setTrakeVideos(res.videos ?? []);
        setLatency(null);
        const n = Math.max(2, res.parsed.trake?.events?.length || 2);
        setTrakeSlots(Array(n).fill(null));
        setActiveSlot(0);
      } else {
        const res = await api.search({
          retrieval_database: retrievalDatabase,
          image_models: imageModelsForSearch(retrievalDatabase, imageModels),
          query,
          query_type_hint: queryType,
          scope: scopeRequest(scopeMode, scopeSelection),
          traffic: trafficMode,
          previous_hints: hints,
          manual_overrides: overrides,
          feedback,
          use_llm: useLLM,
          expand,
          translate,
          rerank,
          top_k: topK,
          // Results are grouped by video and the backend then keeps only the
          // first `max_videos` groups. Left at its default of 50, a deep search
          // would be silently trimmed back to roughly the old frame count.
          max_videos: Math.min(500, Math.max(50, Math.ceil(topK / 2))),
        });
        if (owner !== searchOwner.current) return;
        setParsed(res.parsed);
        setGroups(res.groups);
        setLatency(res.latency_ms);
        setAppliedScope(res.scope ?? null);
        setAppliedTraffic(res.traffic ?? null);
        setPeReport(res.pe_tokens ?? null);
        if (res.warnings?.length) {
          setToast({ msg: res.warnings.join(" · "), kind: "bad" });
        }
      }
      setSelectedVideo(0);
      setSelectedFrame(0);
    } catch (e) {
      if (owner !== searchOwner.current) return;
      const msg = e instanceof ApiError
        ? (typeof e.detail === "string" ? e.detail : `Search failed (${e.status})`)
        : "Search failed — check backend / /api/health";
      setToast({ msg, kind: "bad" });
    } finally {
      if (owner === searchOwner.current) setLoading(false);
    }
    // `topK` is read here, not watched: nothing re-runs a search when it moves.
  }, [queryState, hints, overrides, queryType, feedback, useLLM, expand, translate, rerank, imageModels, retrievalDatabase, topK, scopeMode, scopeSelection, trafficMode, progressiveActive]);

  // ---- agent sidecar search ----
  // Which CLIs the backend host has; null until health has answered, in which
  // case both are asked for and the backend marks a missing one unavailable.
  const agentAvailability = useMemo(
    () => (health ? AGENT_NAMES.filter((name) => health.capabilities[`agent_${name}`]) : null),
    [health],
  );

  /** Search pressed by the operator: the main search and, with AGENT on, the agents.
   *
   *  Agents start ONLY here, never from `runSearch` itself: that also re-runs on
   *  every feedback click and on a pack import, and restarting two CLIs there
   *  would throw away their progress on the same query (or start fifty of them).
   *  Nothing awaits the agents; the main search goes out in the same tick. */
  const operatorSearch = useCallback(() => {
    const text = queryState.trim();
    const agents = agentAvailability ?? AGENT_NAMES;
    if (agentOn && !progressiveActive && agents.length && (text || hints.length)) {
      void startAgents({
        retrieval_database: retrievalDatabase,
        image_models: imageModelsForSearch(retrievalDatabase, imageModels),
        query: text || hints.join(". "),
        query_type: queryType,
        scope: scopeRequest(scopeMode, scopeSelection),
        previous_hints: text ? hints : [],
        agents,
      });
    }
    void runSearch();
  }, [agentAvailability, agentOn, hints, imageModels, progressiveActive, queryState, queryType,
    retrievalDatabase, runSearch, scopeMode, scopeSelection, startAgents]);

  function toggleAgent() {
    const next = !agentOn;
    setAgentOn(next);
    writeAgentPreference(next);
    // Off means off: agents already searching are stopped too.
    if (!next) void stopAgents();
  }

  // A run belongs to one question on one dataset.
  useEffect(() => {
    clearAgents();
  }, [clearAgents, question?.id, retrievalDatabase]);

  const resultVideoIds = useMemo(
    () => new Set(displayGroups.map((group) => group.video_id)),
    [displayGroups],
  );

  /** Select an agent's candidate in the main results and play it there. */
  function locateAgentCandidate(candidate: AgentCandidate) {
    const videoIndex = displayGroups.findIndex((group) => group.video_id === candidate.video_id);
    if (videoIndex < 0) return;
    const group = displayGroups[videoIndex];
    const time = candidateTime(candidate);
    let frameIndex = group.frames.findIndex(
      (frame) => frame.submit_keyframe_id === candidate.submit_keyframe_id,
    );
    if (frameIndex < 0 && time != null) {
      // Not among this video's result frames: point at the nearest one in time.
      let bestDelta = Infinity;
      group.frames.forEach((frame, index) => {
        if (frame.pts_time == null) return;
        const delta = Math.abs(frame.pts_time - time);
        if (delta < bestDelta) {
          bestDelta = delta;
          frameIndex = index;
        }
      });
    }
    setSelectedVideo(videoIndex);
    setSelectedFrame(Math.max(0, frameIndex));
    setTrakePeak(null);
    setPausedFrame(null);
    setGuardOpen(false);
    setExpanded((current) => new Set(current).add(group.video_id));
    setActiveVideoId(group.video_id);
    setVideoVisible(true);
    setShowTimeline(true);
    if (time != null) {
      setClipSeek({ videoId: group.video_id, time });
      if (activeVideoId === group.video_id) viewerRef.current?.seek(time);
    }
  }

  const agentVideoFallback = useCallback(
    (url: string) => swapVideoOrigin(url, health?.media?.video_base_url, health?.media?.video_fallback_base_url),
    [health?.media?.video_base_url, health?.media?.video_fallback_base_url],
  );

  // The tab rail shows a spinner per tab, so the parent has to know which tabs
  // are still running after an import kicked all of them off at once.
  useEffect(() => { onBusyChange(loading); }, [loading, onBusyChange]);

  // ---- frame order inside one video group ----
  /** Re-point the selection at the SAME frame after a reorder.
   *
   *  `selectedFrame` is a position, so reordering the strip under it would leave
   *  the operator looking at one thumbnail while Detail, the timeline and the
   *  submit guard describe another. Only the selected video can be affected. */
  const keepSelectionThroughReorder = useCallback((videoId: string, byTime: boolean) => {
    const group = groups[selectedVideo];
    if (!group || group.video_id !== videoId) return;
    const current = displayGroups[selectedVideo]?.frames[selectedFrame];
    if (!current) return;
    const nextFrames = byTime ? sortFramesByTime(group.frames) : group.frames;
    const index = nextFrames.findIndex((f) => f.submit_keyframe_id === current.submit_keyframe_id);
    if (index >= 0) setSelectedFrame(index);
  }, [groups, displayGroups, selectedVideo, selectedFrame]);

  const setFrameOrder = useCallback((videoId: string, byTime: boolean) => {
    // Computed outside the state updater: it moves the selection too, and an
    // updater that has side effects runs them twice under StrictMode.
    if (byRelevance.has(videoId) !== byTime) return;
    const next = new Set(byRelevance);
    if (byTime) next.delete(videoId);
    else next.add(videoId);
    keepSelectionThroughReorder(videoId, byTime);
    setByRelevance(next);
  }, [byRelevance, keepSelectionThroughReorder]);

  const sortFramesByTimeFor = useCallback(
    (videoId: string) => setFrameOrder(videoId, true), [setFrameOrder],
  );
  const resetFrameOrderFor = useCallback(
    (videoId: string) => setFrameOrder(videoId, false), [setFrameOrder],
  );

  // ---- V-KIS sketch search ----
  // A separate entry point from the text query: the sketch is its own query, so
  // it never silently re-runs when the operator edits the text box (and vice
  // versa). Results land in the same `groups` state, so timeline, detail panel
  // and submit guard behave exactly as they do for any other search. The
  // retrieval-depth slider applies to it like to any other search.
  const runCanvasSearch = useCallback(async (image: string, { suppressBlank }: { suppressBlank: boolean }) => {
    const owner = ++searchOwner.current;
    setLoading(true);
    setAppliedTopK(topK);
    setByRelevance(new Set());
    setDuplicateId(null);
    setPausedFrame(null);
    try {
      const res = await api.searchCanvas({
        retrieval_database: retrievalDatabase,
        canvas: { image },
        scope: scopeRequest(scopeMode, scopeSelection),
        top_k: topK,
        suppress_blank: suppressBlank,
      });
      if (owner !== searchOwner.current) return;
      setParsed(null);
      setGroups(res.groups);
      setTrakeVideos([]);
      setTrakePeak(null);
      setChainOverrides({});
      setLatency(res.latency_ms);
      setAppliedScope(res.scope ?? null);
      setAppliedTraffic(null);
      setPeReport(null);
      setSelectedVideo(0);
      setSelectedFrame(0);
      if (res.warnings?.length) setToast({ msg: res.warnings.join(" · "), kind: "bad" });
      else if (!res.groups.length) setToast({ msg: "No frames returned for this sketch", kind: "bad" });
    } catch (e) {
      if (owner !== searchOwner.current) return;
      const msg = e instanceof ApiError
        ? (typeof e.detail === "string" ? e.detail : `Sketch search failed (${e.status})`)
        : "Sketch search failed — check backend / /api/health";
      setToast({ msg, kind: "bad" });
    } finally {
      if (owner === searchOwner.current) setLoading(false);
    }
  }, [retrievalDatabase, scopeMode, scopeSelection, topK]);

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
    const location = findQaFrame(displayGroups, submitKeyframeId);
    if (!location) return;
    const group = displayGroups[location.video];
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
        retrieval_database: retrievalDatabase,
        question,
        candidates: qaCandidates,
        max_answers: 5,
        web_grounding: qaWebGrounding ? "auto" : "off",
      });
      setQaAnalysis(result);
    } catch (error) {
      const message = error instanceof ApiError && typeof error.detail === "string"
        ? error.detail
        : "QA analysis failed — check the QA vision backend (DEEPSEEK_API_KEY, or the NVILA worker).";
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
      // Same rule as a dragged frame: one video per row (see `assignPeakToSlot`).
      const held = trakeSlots.find((slot): slot is TrakeSlot => slot !== null);
      if (held && held.video_id !== pausedFrame.video_id) {
        setToast({
          msg: `Sequence uses ${held.video_id}. Clear the slots before selecting frames from ${pausedFrame.video_id}.`,
          kind: "bad",
        });
        return;
      }
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
    [pausedFrame, trakeSlots],
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

  /** Click an event representative or a heat peak on a TRAKE video card.
   *
   *  Selecting a moment and WATCHING it are separate decisions. The click points
   *  the selection, the detail panel and the event slot at the frame, and moves
   *  the player only if it is already open; `v` is what opens it. Opening the
   *  video on every click meant a 10 MB fetch and a layout jump for what is
   *  mostly a glance at a thumbnail the card already shows. */
  const pickTrakeMoment = useCallback(
    (videoId: string, peak: TrakeHeatPeak) => {
      const videoIndex = displayGroups.findIndex((group) => group.video_id === videoId);
      if (videoIndex >= 0) {
        setSelectedVideo(videoIndex);
        // Only when the peak IS a chain frame. An alternative moment is not in
        // the strip at all, and snapping the selection to frame 0 would send the
        // operator back to E1 the moment the pin is dropped.
        const frameIndex = displayGroups[videoIndex].frames.findIndex(
          (frame) => frame.submit_keyframe_id === peak.submit_keyframe_id,
        );
        if (frameIndex >= 0) setSelectedFrame(frameIndex);
      }
      setExpanded((current) => new Set(current).add(videoId));
      // Set even while the player is closed: it warms the timeline fetch and
      // tells `v` which video to open.
      setActiveVideoId(videoId);
      // Both paths are needed: the pin drives `startTime` when the player is
      // opened later (or is being mounted), the imperative seek moves one that
      // is already up, where `startTime` would not have changed.
      setTrakePeak({ videoId, peak });
      if (videoVisible && activeVideoId === videoId) seekVideo(peak.pts_time);
      setActiveSlot(
        Math.min(Math.max(peak.event_index - 1, 0), Math.max(0, trakeSlots.length - 1)),
      );
      // A raw pause belongs to whatever the operator was verifying before.
      setPausedFrame(null);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [displayGroups, trakeSlots.length, videoVisible, activeVideoId],
  );

  const selectTrakeVideo = useCallback(
    (videoId: string) => {
      const videoIndex = displayGroups.findIndex((group) => group.video_id === videoId);
      if (videoIndex < 0) return;
      setSelectedVideo(videoIndex);
      setSelectedFrame(0);
      setTrakePeak(null);
    },
    [displayGroups],
  );

  /** Put a frame from a TRAKE card straight into an event slot.
   *
   *  This is the whole payoff of drawing the candidate frames: the operator can
   *  see a better moment for E3 and drop it into the slot, instead of loading the
   *  video, seeking to it and pausing on it just to find out what it looks like.
   *  `frame_idx` comes from the keyframe map when it is known; deriving it from
   *  `pts × fps` is the fallback, and 25 fps is the same default the rest of the
   *  console uses when no timeline has been loaded for the video. */
  const assignPeakToSlot = useCallback(
    (slotIdx: number, payload: TrakePeakDrag) => {
      const { videoId, peak } = payload;
      // A TRAKE row names ONE video and then lists a frame per event inside it.
      // Mixing videos across slots produces a row that is not merely wrong but
      // wrong in a way nothing downstream can see: the export takes the video id
      // from the first filled slot and writes every frame index under it, so a
      // frame from another video is submitted as if it came from this one.
      const held = trakeSlots.find((slot): slot is TrakeSlot => slot !== null);
      if (held && held.video_id !== videoId) {
        setToast({
          msg: `Sequence uses ${held.video_id}. Clear the slots before selecting frames from ${videoId}.`,
          kind: "bad",
        });
        return;
      }
      const fps = (timeline?.video_id === videoId ? timeline?.fps : null) ?? 25;
      setTrakeSlots((slots) => {
        const next = [...slots];
        next[slotIdx] = {
          video_id: videoId,
          frame_idx: peak.frame_idx ?? Math.round(peak.pts_time * fps),
          pts_time: peak.pts_time,
          thumbnail: peak.keyframe_url,
          submit_keyframe_id: peak.submit_keyframe_id,
        };
        return next;
      });
      setActiveSlot((current) => Math.min(Math.max(current, slotIdx + 1), trakeSlots.length - 1));
    },
    [timeline, trakeSlots],
  );

  /** Drop a frame onto event Ei of a card: it replaces the DP's pick there. */
  const overrideChainEvent = useCallback(
    (videoId: string, eventIndex: number, peak: TrakeHeatPeak) => {
      setChainOverrides((current) => ({
        ...current,
        [videoId]: { ...(current[videoId] ?? {}), [eventIndex]: peak },
      }));
      setTrakePeak({ videoId, peak });
    },
    [],
  );

  const clearChainOverride = useCallback((videoId: string, eventIndex: number) => {
    setChainOverrides((current) => {
      const forVideo = { ...(current[videoId] ?? {}) };
      delete forVideo[eventIndex];
      const next = { ...current };
      if (Object.keys(forVideo).length) next[videoId] = forVideo;
      else delete next[videoId];
      return next;
    });
  }, []);

  /** Fill every slot from a TRAKE card's chain — the operator's frames where they
   *  replaced one, the DP's everywhere else — and open the guard. */
  function trakeQuickSubmitVideo(videoId: string) {
    const video = trakeVideos.find((item) => item.video_id === videoId);
    if (!video) return;
    const overrides = chainOverrides[videoId] ?? {};
    const fps = (timeline?.video_id === videoId ? timeline?.fps : null) ?? 25;
    const slots: (TrakeSlot | null)[] = video.events.map((event) => {
      const peak = overrides[event.event_index] ?? event.representative;
      if (!peak) return null;
      return {
        video_id: videoId,
        frame_idx: peak.frame_idx ?? Math.round(peak.pts_time * fps),
        pts_time: peak.pts_time,
        thumbnail: peak.keyframe_url,
        submit_keyframe_id: peak.submit_keyframe_id,
      };
    });
    setTrakeSlots(slots);
    selectTrakeVideo(videoId);
    const key = `${videoId}|${slots.map((slot) => slot?.frame_idx).join(",")}`;
    const dup = history.some((h) => h.task_id === taskScope && (h.dedup_keys || []).includes(key));
    setDuplicateId(dup ? key : null);
    setGuardOpen(true);
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
    const gi = displayGroups.indexOf(g);
    if (gi >= 0) { setSelectedVideo(gi); setSelectedFrame(0); }
    // Duplicate check against the new slots (state update is async).
    const key = `${slots[0]?.video_id}|${slots.map((s) => s?.frame_idx).join(",")}`;
    const dup = history.some((h) => h.task_id === taskScope && (h.dedup_keys || []).includes(key));
    setDuplicateId(dup ? key : null);
    setGuardOpen(true);
  }

  // ---- submission draft (what a submit writes into the Submission tab) ----
  /** The frames the current selection would contribute, in event order. */
  function draftFrames(target: GuardTarget): number[] | null {
    if (queryType === "TRAKE") {
      const filled = trakeSlots.filter((s): s is TrakeSlot => s !== null);
      return filled.length ? filled.map((slot) => slot.frame_idx) : null;
    }
    const idx =
      target === "paused"
        ? pausedFrame?.frame_idx ?? null
        : selectedFrameObj
          ? frameIdxOf(selectedFrameObj)
          : null;
    return idx == null ? null : [idx];
  }

  /** Keyframe id per frame, or null where the frame came straight off the video. */
  function draftKeyframeIds(target: GuardTarget): (string | null)[] {
    if (queryType === "TRAKE") {
      return trakeSlots
        .filter((s): s is TrakeSlot => s !== null)
        .map((slot) => slot.submit_keyframe_id ?? null);
    }
    if (target === "paused") return [null];
    return [selectedFrameObj?.submit_keyframe_id ?? null];
  }

  function draftPtsTimes(target: GuardTarget): (number | null)[] {
    if (queryType === "TRAKE") {
      return trakeSlots.filter((s): s is TrakeSlot => s !== null).map((slot) => slot.pts_time ?? null);
    }
    if (target === "paused") return [pausedFrame?.pts_time ?? null];
    return [selectedFrameObj?.pts_time ?? null];
  }

  function draftVideoId(target: GuardTarget): string | null {
    if (queryType === "TRAKE") {
      return trakeSlots.find((s): s is TrakeSlot => s !== null)?.video_id ?? null;
    }
    return (target === "paused" ? pausedFrame?.video_id : selectedFrameObj?.video_id) ?? null;
  }

  function buildDraft(target: GuardTarget): SubmissionDraft | null {
    if (!question) return null;
    const frames = draftFrames(target);
    const videoId = draftVideoId(target);
    if (!frames || !videoId) return null;
    return {
      questionId: question.id,
      videoId,
      frames,
      answer: answer.trim(),
      keyframeIds: draftKeyframeIds(target),
      ptsTimes: draftPtsTimes(target),
      retrievalDatabase,
    };
  }

  const guardDraft = guardOpen ? buildDraft(guardTarget) : null;
  const guardCsvLine = guardDraft
    ? rowToCsvLine({ ...guardDraft, id: "" }, question ? question.kind : kindForQueryType(queryType))
    : null;
  const guardCsvError = (() => {
    if (dresEnabled) return null;
    if (!question) return "No question assigned to this tab. Select a question in QUERY PACK before submitting.";
    if (!guardDraft) return "Could not determine video/frame_idx to save.";
    if (question.kind === "qa" && !answer.trim()) return "Q&A questions require an answer.";
    if (question.kind === "trake" && question.eventCount && guardDraft.frames.length !== question.eventCount) {
      return `This TRAKE question requires exactly ${question.eventCount} frames (found ${guardDraft.frames.length}).`;
    }
    return null;
  })();

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
    if (!dresEnabled) {
      // The submission table is the record now, so it is also the dedup scope.
      const draft = buildDraft(target);
      if (!draft || !question) return null;
      const hit = findDuplicate(questionRows, draft, question.kind);
      return hit ? rowToCsvLine(hit, question.kind) : null;
    }
    const key = dedupKeyFor(target);
    if (!key) return null;
    for (const h of history) {
      if (h.task_id !== taskScope) continue;
      if ((h.dedup_keys || []).includes(key)) return key;
    }
    return null;
  }

  function openGuard(target: GuardTarget = "result") {
    if (queryType === "AVS") {
      const frame: FrameResult | null = target === "paused" && pausedFrame ? {
        ...pausedFrame, image_id: `paused:${pausedFrame.video_id}:${pausedFrame.frame_idx}`,
        submit_keyframe_id: "", keyframe_n: 0, score: 0, channels: [], per_channel_score: {}, evidence: [],
        keyframe_url: pausedFrame.thumbnail ?? "", video_url: groups.find(g => g.video_id === pausedFrame.video_id)?.video_url ?? "",
      } : selectedFrameObj;
      if (frame) setAvsRequested({ frame, id: Date.now(), scope: avsScope });
      return;
    }
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

  /** Add the current result/sequence/raw pause to the per-machine scratchpad. */
  function addToSticky(target: GuardTarget = "result") {
    if (!question) {
      setToast({ msg: "Assign a question to this tab before pinning a candidate.", kind: "bad" });
      return;
    }
    const draft = buildDraft(target);
    if (!draft) {
      setToast({ msg: "Could not determine video/frame to pin.", kind: "bad" });
      return;
    }
    if (question.kind === "trake" && question.eventCount && draft.frames.length !== question.eventCount) {
      setToast({
        msg: `TRAKE requires all ${question.eventCount} frames before pinning.`,
        kind: "bad",
      });
      return;
    }
    const result = onAddStickyRow(draft);
    if (result === "full") {
      setToast({ msg: `Sticky note already contains ${MAX_ROWS_PER_QUESTION} candidates.`, kind: "bad" });
      return;
    }
    setToast({ msg: `Pinned candidate to ${question.id}.`, kind: "ok" });
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
    if (!dresEnabled || !guardOpen) {
      setSubmitPreview(null);
      setPreviewError(null);
      return;
    }
    const body = buildSubmitBody();
    if (!body) {
      setSubmitPreview(null);
      setPreviewError("No frame selected for submission.");
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
          setPreviewError(detail?.message ?? "Could not build the DRES payload.");
        })
        .finally(() => !cancelled && setPreviewLoading(false));
    }, 150); // debounce: the answer / pad inputs change on every keystroke
    return () => { cancelled = true; clearTimeout(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dresEnabled, guardOpen, guardTarget, queryType, answer, answerMode, segmentPadMs, taskNameOverride,
      evaluationId, currentTaskName, selectedFrameObj, pausedFrame, trakeSlots]);

  async function confirmSubmit() {
    if (!dresEnabled) {
      const draft = buildDraft(guardTarget);
      if (!draft || guardCsvError) {
        setToast({ msg: guardCsvError ?? "Could not build the submission row.", kind: "bad" });
        return;
      }
      onSubmitRow(draft);
      setToast({ msg: `Saved to ${draft.questionId}.csv`, kind: "ok" });
      setGuardOpen(false);
      setDuplicateId(null);
      return;
    }
    if (!submitPreview || previewLoading || previewError || submitPreview.answer_mode_mismatch || submitting) return;
    const body = buildSubmitBody();
    if (!body) {
      setToast({ msg: "Could not determine frame_idx; this frame cannot be submitted.", kind: "bad" });
      return;
    }
    setSubmitting(true);
    try {
      const entry = await api.submit({ ...body, evaluation_id: submitPreview.evaluation_id,
        task_name: submitPreview.task_name, expected_task_name: submitPreview.task_name,
        require_dres: true, allow_duplicate: duplicateId != null });
      const verdict = entry.verdict ?? null;
      setToast({
        msg: entry.status === "dres_error"
          ? `DRES error: ${entry.dres?.error ?? "?"}`
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
        setPreviewError(detail?.message ?? "Invalid payload.");
        setToast({ msg: detail?.message ?? "Invalid payload", kind: "bad" });
      } else {
        setToast({ msg: "Submit failed", kind: "bad" });
      }
    } finally {
      setSubmitting(false);
    }
  }

  // ---- keyboard ----
  // Every tab stays mounted so an import can search them all at once, so the
  // shortcut handler must belong to exactly one of them: the visible one.
  useEffect(() => {
    if (!active) return;
    function isTyping() {
      const el = document.activeElement;
      return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA");
    }
    // The video only owns the keys while it is actually on screen for the group
    // the operator has selected — the same condition 'v' toggles on.
    const videoScrubbable = videoVisible && !!activeVideoId && activeVideoId === selectedVideoId;

    function onKey(e: KeyboardEvent) {
      // Ctrl/Cmd + / toggles the shortcuts help — works anywhere, even while typing.
      if (e.key === "/" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        setKeymapOpen((o) => !o);
        return;
      }
      // While the floating note is open it owns navigation, preview, video and
      // delete. Letting this hidden handler run too would move/delete a Search
      // selection behind the window.
      if (stickyOpen) return;
      // While focus is inside the V-KIS sketch it owns its tool keys, brush
      // size, undo and Enter (so a stray Enter never opens the submit guard
      // mid-drawing). Arrows, v, k and t still fall through to the console.
      if (ownsSketchKey(e)) return;
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

      // An inline <video controls> brings its own keyboard handling: Chrome's
      // shadow-DOM media controls seek on the arrows, and once the progress
      // slider has focus its step is ~1% of the duration — 11s on a 19-minute
      // clip, which lands on top of our own 5s and moves the playhead 16s.
      // That handling happens on the way UP to window, so preventDefault() here
      // cannot cancel it. This listener is therefore registered in the CAPTURE
      // phase and stops the event before the element is ever reached.
      if (videoVisible && seekDeltaForKey(e.key) !== null) e.stopPropagation();

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
          setTrakePeak(null);
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedVideo((v) => Math.max(v - 1, 0));
          setSelectedFrame(0);
          setTrakePeak(null);
          break;
        // Three claimants on the horizontal arrows, in priority order:
        //   1. the neighbour strip, whose whole purpose is paging keyframes
        //      (and which already drives the video to each one);
        //   2. the open inline video, scrubbed +/-5s — an operator watching a
        //      clip wants the playhead, not the result list, to move;
        //   3. otherwise the retrieved frames of the selected video.
        // 'a'/'d' below stay on the video at +/-1s with no such contention.
        case "ArrowRight":
          e.preventDefault();
          if (neighborsVisible) moveNeighbor(1);
          else if (videoScrubbable) nudgeVideo(VIDEO_COARSE_STEP_S);
          else {
            setSelectedFrame((f) => Math.min(f + 1, (selectedGroup?.frames.length ?? 1) - 1));
            setTrakePeak(null);
          }
          break;
        case "ArrowLeft":
          e.preventDefault();
          if (neighborsVisible) moveNeighbor(-1);
          else if (videoScrubbable) nudgeVideo(-VIDEO_COARSE_STEP_S);
          else {
            setSelectedFrame((f) => Math.max(f - 1, 0));
            setTrakePeak(null);
          }
          break;
        case "d":
        case "D":
          e.preventDefault();
          if (videoScrubbable) nudgeVideo(VIDEO_FINE_STEP_S);
          break;
        case "a":
        case "A":
          e.preventDefault();
          if (videoScrubbable) nudgeVideo(-VIDEO_FINE_STEP_S);
          break;
        case " ":
          // The video no longer autoplays, so play/pause needs a key that works
          // without the element having focus — after clicking a heat peak or
          // pressing 'v', focus is nowhere near it.
          if (!videoScrubbable) break;
          e.preventDefault();
          viewerRef.current?.toggle();
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
          // pressing it again on the same video hides it. It opens PARKED on the
          // selected frame — see VideoViewer — and Space starts playback.
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
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [guardOpen, guardTarget, keymapOpen, pausedFrame, queryType, activeSlot, groups.length, selectedGroup, selectedFrameObj, selectedVideoId, videoVisible, activeVideoId, submitting, assignPausedFrameToSlot, neighborsVisible, neighborKeyframes, neighborIndex, neighborAnchorIndex, stickyOpen]);

  /** Scrub the open inline video by `delta` seconds (clamped by the viewer). */
  function nudgeVideo(delta: number) {
    viewerRef.current?.seekBy(delta);
  }

  function seekVideo(t: number) {
    viewerRef.current?.seek(t);
    setPlayhead(t);
  }

  function copyId(id: string) {
    navigator.clipboard?.writeText(id).then(() => setToast({ msg: `Copied ${id}`, kind: "ok" })).catch(() => {});
  }

  // The inline player + timeline + neighbour strip, rendered under whichever
  // result card owns the active video. Both result views mount the same node,
  // so switching between them never reloads the video.
  // Which URL the player starts on, and which one it retries. They swap once the
  // primary has failed, so the retry always points at the origin NOT in use.
  const primaryVideoUrl = selectedGroup?.video_url ?? "";
  const otherOriginUrl = swapVideoOrigin(
    primaryVideoUrl,
    health?.media?.video_base_url,
    health?.media?.video_fallback_base_url,
  );
  const playerSrc = (videoOriginDown && otherOriginUrl) || primaryVideoUrl;
  const playerFallbackSrc = playerSrc === primaryVideoUrl ? otherOriginUrl : primaryVideoUrl;

  const videoSlot =
    (videoVisible || neighborsVisible) && activeVideoId && selectedGroup
    && selectedGroup.video_id === activeVideoId ? (
      <div ref={videoPanelRef}>
        {videoVisible && (
          <>
            <VideoViewer
              ref={viewerRef}
              src={playerSrc}
              fallbackSrc={playerFallbackSrc}
              onFallback={() => {
                if (videoOriginDown) return;
                setVideoOriginDown(true);
                setToast({ msg: "Primary video source unavailable; switched to the fallback source.", kind: "bad" });
              }}
              startTime={
                clipSeek?.videoId === selectedGroup.video_id
                  ? clipSeek.time
                  : selectedFrameObj?.pts_time ?? selectedGroup.best_clip?.start_time ?? 0
              }
              onPaused={onVideoPaused}
              onTime={setPlayhead}
            />
            {showTimeline && timeline && timeline.video_id === activeVideoId && (
              <Timeline
                data={timeline}
                playhead={playhead}
                selectedPts={selectedFrameObj?.pts_time ?? selectedGroup.best_clip?.start_time ?? null}
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
      </div>
    ) : null;

  // A parked tab stays MOUNTED — that is what lets an imported pack run every
  // tab's search at once and keep each result set — but it renders nothing.
  // Hooks above have all run, so its state and in-flight requests are intact;
  // only the DOM is skipped, which keeps 24 open tabs from meaning 24 live
  // consoles (and keeps `getByTestId` addressing exactly one console).
  // PHM owns its session in the panel: park that subtree instead of unmounting
  // it when another workspace tab is selected. Closing the tab still cleans up.
  if (!active && !progressiveActive) return null;

  const taraAvailable = Boolean(health?.capabilities.tara_clip_search);
  const taraEnabled = taraAvailable
    && !overrides.disable_channels.includes("tara")
    && (overrides.force_channels.includes("tara")
      || (parsed?.channels?.tara?.enabled ?? true));

  return (
    <div className="app" style={active ? undefined : { display: "none" }}>
      <TopBar
        queryType={queryType}
        onQueryType={(t) => {
          onQueryType(t);
          setParsed(null);
          setTrakeVideos([]);
          setTrakePeak(null);
          setChainOverrides({});
          setPausedFrame(null);
          setQaAnalysis(null);
          setQaAnalysisError(null);
          setAnswer("");
          // The answer shape belongs to the task type, so it must not survive a
          // switch: a `temporal` override carried into QA silently drops the text.
          setAnswerMode("auto");
        }}
        elapsed={elapsed}
        penalties={penalties}
        latency={latency}
        health={health}
        retrievalDatabase={retrievalDatabase}
        onRetrievalDatabase={onRetrievalDatabase}
        onSimpleMode={onSimpleMode}
        onShowKeymap={() => setKeymapOpen(true)}
        onShowSettings={onShowSettings}
      />
      {dresStatus?.configured && <label className="phm-toggle">
        <input type="checkbox" checked={dresRequested} onChange={e => { setDresRequested(e.target.checked); setGuardOpen(false); }} />
        Direct DRES submission
      </label>}
      {dresEnabled ? (
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
      ) : (
        <QuestionBar
          questions={questions}
          selectedId={question?.id ?? null}
          onSelect={onSelectQuestion}
          onImport={onImportQuestions}
          importing={importing}
          importError={importError}
          rowCount={questionRows.length}
          stickyCount={stickyCount}
          stickyOpen={stickyOpen}
          onOpenSticky={onToggleSticky}
          onOpenSubmission={onOpenSubmission}
          onSearchAll={onSearchAll}
          searchingAll={searchingAll}
        />
      )}
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
          {queryType === "T-KIS" && <label className="phm-toggle">
            <input type="checkbox" checked={progressive} onChange={e => { setProgressive(e.target.checked); setGroups([]); }} />
            Progressive Hint Memory
          </label>}
          {progressiveActive ? <>
            <ImageModelSelector retrievalDatabase={retrievalDatabase} value={imageModels} onChange={setImageModels} />
            <label className="phm-toggle"><input type="checkbox" checked={translate} onChange={e => setTranslate(e.target.checked)} />Translate VI → EN</label>
            <label className="phm-toggle"><input type="checkbox" checked={progressiveHybrid} onChange={e => setProgressiveHybrid(e.target.checked)} />Include OCR / speech / audio</label>
            <ProgressivePanel key={progressiveKey}
              config={{ retrieval_database: retrievalDatabase, task_id: question?.id ?? "manual", image_models: imageModelsForSearch(retrievalDatabase, imageModels), scope: { mode: "all", categories: [] }, translate, hybrid: progressiveHybrid, top_k: 200 }}
              initialText={query} onSnapshot={acceptProgressive} onBusy={setLoading} />
          </> : <QueryPanel
            query={query}
            setQuery={editQuery}
            hints={hints}
            onAppendHint={appendHint}
            onClearHints={() => setHints([])}
            onSearch={operatorSearch}
            loading={loading}
            parsed={parsed}
            agent={agentOn}
            onToggleAgent={toggleAgent}
            agentAvailable={agentAvailability ? agentAvailability.length > 0 : null}
            queryType={queryType}
            inputRef={queryRef}
            useLLM={useLLM}
            onToggleLLM={() => setUseLLM((v) => !v)}
            expand={expand}
            onToggleExpand={() => setExpand((v) => !v)}
            rerank={rerank}
            onToggleRerank={setRerank}
            translate={translate}
            onToggleTranslate={setTranslate}
            peActive={imageModelsForSearch(retrievalDatabase, imageModels).includes("pe")}
            rerankAvailable={Boolean(health?.capabilities.visual_rerank)}
            rerankReport={latency?.reranker ?? null}
            imageModelSelector={
              <ImageModelSelector
                retrievalDatabase={retrievalDatabase}
                value={imageModels}
                onChange={setImageModels}
                tara={{
                  available: taraAvailable,
                  enabled: taraEnabled,
                  onChange: (enabled) => toggleChannel("tara", enabled),
                }}
              />
            }
            topK={topK}
            onTopK={setTopK}
            appliedTopK={appliedTopK}
            scopeFilter={
              <ScopeFilter
                catalogue={scopeCatalogue}
                mode={scopeMode}
                onMode={setScopeMode}
                selected={scopeSelection}
                onSelected={(categories) =>
                  setScopeSelection(orderCategories(categories, scopeCatalogue))
                }
                applied={appliedScope}
                error={scopeError}
              />
            }
          />
          }
          {!progressiveActive && <ChannelControls retrievalDatabase={retrievalDatabase} parsed={parsed} overrides={overrides} onToggle={toggleChannel} />}
          <QueryUnderstanding
            parsed={parsed}
            traffic={appliedTraffic}
            trafficMode={trafficMode}
            onTrafficMode={setTrafficMode}
            peTokens={peReport}
          />
        </div>

        {/* CENTER */}
        <div className="col col-center" style={focusZone === "results" ? { boxShadow: "inset 0 2px 0 var(--accent)" } : undefined}>
          {/* The canvas lives in the widest column: V-KIS drawing accuracy is
              limited by how much room the operator has to draw. It searches
              the PE index of whichever profile is selected. */}
          {queryType === "V-KIS" && (
            <CanvasPanel
              onSearch={runCanvasSearch}
              loading={loading}
              available={health ? Boolean(health.capabilities.canvas_sketch_search) : null}
              storageKey={question?.id ?? "scratch"}
              indexLabel={retrievalDatabase === "infoshotpp" ? "InfoShot++ PE index" : "BTC PE index"}
            />
          )}
          <div className="results-toolbar">
            <span className="results-count">
              {trakeVideoView ? (
                <>
                  {trakeVideos.length} video{trakeVideos.length === 1 ? "" : "s"} · ranked by TRAKE
                  video score (coverage first, then sequence quality)
                </>
              ) : (
                <>
                  {groups.length} video{groups.length === 1 ? "" : "s"} ·{" "}
                  {groups.reduce((n, g) => n + g.frame_count, 0)} frames
                </>
              )}
            </span>
            {/* A TRAKE answer is a whole video, so there is no flat keyframe
                ranking to switch to — every frame on screen IS event i. */}
            {!trakeVideoView && !progressiveActive && (
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
            )}
          </div>
          {/* Feedback is a `/api/search` concept; the TRAKE endpoint takes none,
              so showing the bar there would promise a re-ranking that cannot happen. */}
          {!trakeVideoView && !progressiveActive && (
            <FeedbackBar feedback={feedback} onRemove={removeFeedback} onClear={clearFeedback} />
          )}
          {queryType === "AVS" && <AvsPanel
            key={avsScope}
            scopeId={avsScope}
            groups={displayGroups} query={query} database={retrievalDatabase} imageModels={imageModels}
            dresEnabled={dresEnabled} evaluationId={evaluationId} taskName={currentTaskName ?? ""}
            selected={selectedFrameObj} requested={avsRequested?.scope === avsScope ? avsRequested : null}
            onInspect={frame => focusQaEvidence(frame.submit_keyframe_id, false)}
          />}
          {trakeVideoView ? (
          <TrakeVideoResults
            videos={trakeVideos}
            eventCount={trakeEventCount || trakeSlots.length}
            selectedVideoId={selectedVideoId}
            activeEventIndex={activeSlot + 1}
            loading={loading}
            overrides={chainOverrides}
            onSelectVideo={selectTrakeVideo}
            onPickMoment={pickTrakeMoment}
            onOverrideEvent={overrideChainEvent}
            onClearOverride={clearChainOverride}
            onQuickSubmit={trakeQuickSubmitVideo}
            videoSlotVideoId={videoVisible || neighborsVisible ? activeVideoId : null}
            videoSlot={videoSlot}
          />
          ) : (
          <Results
            groups={displayGroups}
            viewMode={progressiveActive ? "grouped" : viewMode}
            trakeEventCount={queryType === "TRAKE" ? trakeEventCount : undefined}
            onTrakeQuickSubmit={queryType === "TRAKE" ? trakeQuickSubmit : undefined}
            qaHotspotScores={queryType === "QA" ? qaHotspotScores : undefined}
            selectedVideo={selectedVideo}
            selectedFrame={selectedFrame}
            expanded={expanded}
            loading={loading}
            onSelectVideo={(i) => { setSelectedVideo(i); setSelectedFrame(0); setTrakePeak(null); }}
            onSelectFrame={(vi, fi) => { setSelectedVideo(vi); setSelectedFrame(fi); setTrakePeak(null); setClipSeek(null); }}
            onInspectEvidence={(vi, fi) => {
              const frame = displayGroups[vi]?.frames[fi];
              if (!frame) return;
              setSelectedVideo(vi); setSelectedFrame(fi); setTrakePeak(null);
              setClipSeek(null); setPausedFrame(null); setGuardOpen(false);
              setActiveVideoId(frame.video_id); setVideoVisible(true); setShowTimeline(true);
              if (activeVideoId === frame.video_id && frame.pts_time != null) viewerRef.current?.seek(frame.pts_time);
            }}
            onSelectClip={(vi, time) => {
              const videoId = displayGroups[vi]?.video_id;
              if (!videoId) return;
              setSelectedVideo(vi);
              setSelectedFrame(0);
              setActiveVideoId(videoId);
              setVideoVisible(true);
              setShowTimeline(true);
              setClipSeek({ videoId, time });
              if (activeVideoId === videoId) viewerRef.current?.seek(time);
            }}
            onToggleExpand={(vid) =>
              setExpanded((prev) => {
                const n = new Set(prev);
                n.has(vid) ? n.delete(vid) : n.add(vid);
                return n;
              })
            }
            onFeedback={onFeedback}
            onVideoFeedback={onVideoFeedback}
            relevanceOrdered={byRelevance}
            onSortByTime={sortFramesByTimeFor}
            onResetOrder={resetFrameOrderFor}
            videoSlotVideoId={videoVisible || neighborsVisible ? activeVideoId : null}
            videoSlot={videoSlot}
          />
          )}
        </div>

        {/* RIGHT */}
        <div className="col col-right" style={focusZone === "detail" ? { boxShadow: "inset 0 2px 0 var(--accent)" } : undefined}>
          <AgentPanel
            run={agentRun}
            starting={agentStarting}
            error={agentError}
            currentQuery={query}
            resultVideoIds={resultVideoIds}
            onStop={() => void stopAgents()}
            onDismiss={clearAgents}
            onLocate={locateAgentCandidate}
            onPaused={setPausedFrame}
            videoFallback={agentVideoFallback}
          />
          {queryType === "QA" && (
            <QaAssistPanel
              analysis={qaAnalysis}
              loading={qaAnalyzing}
              error={qaAnalysisError}
              available={health ? Boolean(health.capabilities.qa_nvila) : null}
              visionModel={(health?.services?.qa_vision as { model?: string } | undefined)?.model ?? null}
              visionBackend={(health?.services?.qa_vision as { backend?: string } | undefined)?.backend ?? null}
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
            onAddToSticky={() => addToSticky("paused")}
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
                onAssignPeak={assignPeakToSlot}
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
            onAddToSticky={() => addToSticky("result")}
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
        dresEnabled={dresEnabled}
        questionId={question?.id ?? null}
        csvLine={guardCsvLine}
        csvError={guardCsvError}
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
