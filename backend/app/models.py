"""Pydantic request models for the API. Responses are plain dicts (documented in
the README) built by the services, to keep the routing layer thin."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

QueryTypeHint = Literal["auto", "T-KIS", "QA", "V-KIS", "TRAKE"]


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
    positive_videos: list[str] = Field(default_factory=list)
    negative_videos: list[str] = Field(default_factory=list)
    negative_frames: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = ""
    query_type_hint: QueryTypeHint = "auto"
    previous_hints: list[str] = Field(default_factory=list)
    manual_overrides: ManualOverrides = Field(default_factory=ManualOverrides)
    parsed: dict[str, Any] | None = None  # reuse a prior parse to skip re-parsing
    feedback: FeedbackState | None = None
    use_llm: bool = False
    expand: bool = False  # Nemotron query expansion (extra visual paraphrases)
    top_k: int = 100
    max_videos: int = 50


class SimpleSearchRequest(BaseModel):
    query: str = ""
    top_k: int = 60


class TranslateRequest(BaseModel):
    text: str = ""


class TrakeSearchRequest(BaseModel):
    query: str = ""
    previous_hints: list[str] = Field(default_factory=list)
    manual_overrides: ManualOverrides = Field(default_factory=ManualOverrides)
    parsed: dict[str, Any] | None = None
    use_llm: bool = False
    expand: bool = False  # Nemotron query expansion per event
    # Wide per-event candidate pool so the video containing all events surfaces a
    # frame for each event (the DP then assembles the ordered sequence).
    top_k: int = 400


class SnapRequest(BaseModel):
    raw_time: float
    fps: float | None = None


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
    question: str = Field(min_length=1, max_length=2000)
    candidates: list[QaCandidate] = Field(min_length=1, max_length=24)
    max_answers: int = Field(default=5, ge=1, le=8)
    web_grounding: Literal["auto", "on", "off"] = "auto"


class TrakeEvent(BaseModel):
    event_index: int
    frame_idx: int
    pts_time: float | None = None
    submit_keyframe_id: str | None = None  # display only


class SubmitPayload(BaseModel):
    # DRES submits by video_id + frame_idx (frame_idx = round(pts_time * fps)).
    video_id: str | None = None
    frame_idx: int | None = None
    timestamp: float | None = None  # pts_time, optional secondary localization
    events: list[TrakeEvent] = Field(default_factory=list)  # TRAKE ordered events
    answer: str | None = None
    # Kept for human-readable history / dedup display; NOT used for DRES routing.
    submit_keyframe_id: str | None = None
    submit_keyframe_ids: list[str] = Field(default_factory=list)


class SubmitRequest(BaseModel):
    task_id: str
    query_type: Literal["T-KIS", "QA", "V-KIS", "TRAKE"]
    payload: SubmitPayload
    allow_duplicate: bool = False
