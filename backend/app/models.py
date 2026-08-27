"""Pydantic request models for the API. Responses are plain dicts (documented in
the README) built by the services, to keep the routing layer thin."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

QueryTypeHint = Literal["auto", "T-KIS", "QA", "V-KIS", "TRAKE"]
RetrievalDatabase = Literal["btc", "infoshotpp"]


class SearchScope(BaseModel):
    """Which dataset folders a search may return frames from.

    `all` searches the whole profile, `manual` exactly the ticked folders, `auto`
    lets the topic heuristic in `app.scope` pick them from the query text. The
    length cap is the full BTC catalogue (L21-L30 + K01-K20) with room to spare.
    """

    mode: Literal["all", "auto", "manual"] = "all"
    categories: list[str] = Field(default_factory=list, max_length=64)


class ManualOverrides(BaseModel):
    force_channels: list[str] = Field(default_factory=list)
    disable_channels: list[str] = Field(default_factory=list)


class ParseRequest(BaseModel):
    query: str
    query_type_hint: QueryTypeHint = "auto"
    previous_hints: list[str] = Field(default_factory=list)
    manual_overrides: ManualOverrides = Field(default_factory=ManualOverrides)
    use_llm: bool = False


class FeedbackState(BaseModel):
    # Video-level priority is a soft multiplier on the aggregate video score;
    # positive_frames seed the image-to-image `similar` retrieval channel.
    positive_videos: list[str] = Field(default_factory=list)
    negative_videos: list[str] = Field(default_factory=list)
    positive_frames: list[str] = Field(default_factory=list, max_length=12)
    negative_frames: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    retrieval_database: RetrievalDatabase = "btc"
    query: str = ""
    query_type_hint: QueryTypeHint = "auto"
    scope: SearchScope = Field(default_factory=SearchScope)
    previous_hints: list[str] = Field(default_factory=list)
    manual_overrides: ManualOverrides = Field(default_factory=ManualOverrides)
    parsed: dict[str, Any] | None = None  # reuse a prior parse to skip re-parsing
    feedback: FeedbackState | None = None
    use_llm: bool = False
    expand: bool = False  # Nemotron query expansion (extra visual paraphrases)
    # Qwen3-VL rescoring of the PE candidate pool. Off by default: it needs a
    # second GPU worker and costs seconds, so it is the operator's call per
    # search rather than something the console does behind their back.
    rerank: bool = False
    # Bounded so a malformed client cannot ask for an unbounded fusion; 1000 is
    # what the console's retrieval-depth slider tops out at.
    top_k: int = Field(default=100, ge=1, le=1000)
    max_videos: int = Field(default=50, ge=1, le=500)


class SimpleSearchRequest(BaseModel):
    retrieval_database: RetrievalDatabase = "btc"
    query: str = ""
    scope: SearchScope = Field(default_factory=SearchScope)
    top_k: int = 60
    rerank: bool = False


class TranslateRequest(BaseModel):
    text: str = ""


class TrakeSearchRequest(BaseModel):
    retrieval_database: RetrievalDatabase = "btc"
    query: str = ""
    scope: SearchScope = Field(default_factory=SearchScope)
    previous_hints: list[str] = Field(default_factory=list)
    manual_overrides: ManualOverrides = Field(default_factory=ManualOverrides)
    parsed: dict[str, Any] | None = None
    use_llm: bool = False
    expand: bool = False  # Nemotron query expansion per event
    # Wide per-event candidate pool so the video containing all events surfaces a
    # frame for each event (the DP then assembles the ordered sequence).
    top_k: int = Field(default=400, ge=1, le=1000)


class TakenAnswer(BaseModel):
    """One answer already in the list: a video and its frame(s)."""

    video_id: str = Field(min_length=1, max_length=64)
    frames: list[int] = Field(min_length=1, max_length=32)


class AnswerGenerateRequest(BaseModel):
    """Ask for the ordered answer list of ONE query.

    Two ways in. Without `groups`/`sequences` the backend runs the retrieval
    itself, which is what the Submission tab's bulk button does. With them it
    ranks the result the operator already has on screen, so their scope, feedback
    and channel overrides are not silently discarded by a fresh search.
    """

    retrieval_database: RetrievalDatabase = "btc"
    query: str = ""
    query_type_hint: QueryTypeHint = "T-KIS"
    scope: SearchScope = Field(default_factory=SearchScope)
    previous_hints: list[str] = Field(default_factory=list)
    manual_overrides: ManualOverrides = Field(default_factory=ManualOverrides)
    parsed: dict[str, Any] | None = None
    feedback: FeedbackState | None = None
    use_llm: bool = False
    expand: bool = False
    top_k: int = Field(default=400, ge=1, le=1000)
    max_videos: int = Field(default=100, ge=1, le=500)
    #: 100 is the organisers' hard cap on answers per query.
    limit: int = Field(default=100, ge=1, le=100)
    #: Answer-generator parameter overrides; unknown keys are ignored and every
    #: value is clamped, so this cannot put the generator in an invalid state.
    params: dict[str, Any] | None = None
    #: Q&A: the text written into every generated row (the frames are ranked the
    #: same way as T-KIS; only the answer column differs).
    answer_text: str = Field(default="", max_length=200)
    #: Answers already occupying the head of this question's list — the ones a
    #: person picked by hand. The generator does not re-emit them, does not spend
    #: positions re-covering the instants they already cover, and continues the
    #: band schedule after them instead of restarting at rank 1.
    taken: list[TakenAnswer] = Field(default_factory=list, max_length=100)
    #: TRAKE: how many events the statement asks for, so a chain that located
    #: fewer can never become a row. The statement is the authority here, which is
    #: why the client sends it; without it the backend falls back to the number of
    #: events its own parser found.
    event_count: int | None = Field(default=None, ge=1, le=32)
    #: Rank an existing on-screen result instead of searching again. The caps are
    #: the console's own limits, so a legitimate client always fits.
    groups: list[dict[str, Any]] | None = Field(default=None, max_length=500)
    sequences: list[dict[str, Any]] | None = Field(default=None, max_length=500)


class SnapRequest(BaseModel):
    retrieval_database: RetrievalDatabase = "btc"
    raw_time: float
    fps: float | None = None


class CanvasObjectSpec(BaseModel):
    """One object drawn on the V-KIS canvas (normalized bbox, palette colour)."""

    id: str = Field(default="", max_length=32)
    label: str = Field(min_length=1, max_length=64)
    # x1, y1, x2, y2 in [0,1]; validated/clamped in app.canvas.parse_canvas.
    bbox: list[float] = Field(min_length=4, max_length=4)
    color: str | None = Field(default=None, max_length=24)
    required: bool = True


class CanvasSpec(BaseModel):
    objects: list[CanvasObjectSpec] = Field(default_factory=list, max_length=12)
    exclude_labels: list[str] = Field(default_factory=list, max_length=8)
    action_text: str = Field(default="", max_length=300)
    mode: Literal["rough", "precise"] = "rough"
    # Rendered canvas as a base64 PNG data URL for the raster (PE image) channel.
    # ~4 MB caps a 1024×576 PNG with room to spare; validated in app.canvas.
    image: str | None = Field(default=None, max_length=4_000_000)


class CanvasSearchRequest(BaseModel):
    retrieval_database: RetrievalDatabase = "btc"
    canvas: CanvasSpec
    scope: SearchScope = Field(default_factory=SearchScope)
    top_k: int = Field(default=100, ge=1, le=400)
    max_videos: int = Field(default=50, ge=1, le=200)
    candidate_pool: int = Field(default=400, ge=50, le=1000)
    # Channel weights for the RRF fusion; only object_layout / image_pe are used.
    weights: dict[str, float] | None = None


class QaCandidate(BaseModel):
    """A retrieval candidate sent to NVILA.

    The backend rebuilds the image URL from ``submit_keyframe_id``; clients are
    never allowed to make the Colab worker fetch an arbitrary URL.
    """

    submit_keyframe_id: str = Field(min_length=3, max_length=160)
    frame_idx: int | None = None
    pts_time: float | None = None
    retrieval_score: float = 0.0
    evidence: list[dict[str, Any]] = Field(default_factory=list, max_length=12)


class QaAnalyzeRequest(BaseModel):
    retrieval_database: RetrievalDatabase = "btc"
    question: str = Field(min_length=1, max_length=2000)
    candidates: list[QaCandidate] = Field(min_length=1, max_length=24)
    max_answers: int = Field(default=5, ge=1, le=8)
    web_grounding: Literal["auto", "on", "off"] = "auto"


class TrakeEvent(BaseModel):
    event_index: int
    frame_idx: int
    pts_time: float | None = None
    # Explicit DRES window for this event; derived from pts_time/fps when absent.
    start_ms: int | None = None
    end_ms: int | None = None
    video_id: str | None = None  # defaults to the payload video
    submit_keyframe_id: str | None = None  # display only


class SubmitPayload(BaseModel):
    """One picked answer. DRES v2 needs `mediaItemName` + `start`/`end` in ms for
    KIS/TRAKE and `text` for QA; the ms window is derived from `timestamp`
    (pts_time) or `frame_idx`+`fps` unless start_ms/end_ms are given."""

    video_id: str | None = None
    frame_idx: int | None = None
    timestamp: float | None = None  # pts_time in seconds
    fps: float | None = None  # only needed when timestamp is unknown
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    events: list[TrakeEvent] = Field(default_factory=list)  # TRAKE ordered events
    answer: str | None = None
    # Kept for human-readable history / dedup display; NOT used for DRES routing.
    submit_keyframe_id: str | None = None
    submit_keyframe_ids: list[str] = Field(default_factory=list)


class SubmitRequest(BaseModel):
    # Free-text label used only when DRES has no open task (offline practice).
    task_id: str = ""
    # DRES run to submit to, and the exact open task name. Both are resolved
    # from the live server when omitted.
    evaluation_id: str | None = None
    task_name: str | None = None
    query_type: Literal["T-KIS", "QA", "V-KIS", "TRAKE"]
    payload: SubmitPayload
    # auto → temporal for KIS/TRAKE, temporal_text (segment + answer) for QA.
    answer_mode: Literal["auto", "temporal", "temporal_text", "text", "item"] = "auto"
    segment_pad_ms: int | None = Field(default=None, ge=0, le=60_000)
    allow_duplicate: bool = False
