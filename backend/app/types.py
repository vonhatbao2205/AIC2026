"""Internal dataclasses used by the fusion/grouping/trake logic.

Kept free of pydantic/FastAPI so the core ranking logic is trivially unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# `similar` is the relevance-feedback channel: image-to-image kNN seeded by the
# frames the operator marked "more like this". It is fused by RRF like any other
# channel, which keeps feedback on the same scale instead of a hard-coded boost.
# `canvas_image` is the V-KIS sketch: the operator's drawing encoded by PE's
# image tower and searched against the keyframe vectors (see canvas_service.py).
# `image_visual` is the MERGED visual ranking: the PE and Qwen candidate pools
# unioned and then judged together by the Qwen visual reranker. It replaces the
# two separate `image_pe` / `image_qwen` channels whenever reranking runs, because
# the reranker's verdict is one opinion about the frame, not one per retriever.
# Without a reranker there is nothing to merge the two spaces with, so they stay
# two channels and meet in the global RRF (see `SearchService._run_visual`).
Channel = Literal[
    "image_pe", "image_qwen", "image_visual", "tara", "ocr", "speech", "audio", "similar",
    "canvas_image",
]
ALL_CHANNELS: tuple[Channel, ...] = (
    "image_pe", "image_qwen", "image_visual", "tara", "ocr", "speech", "audio", "similar",
    "canvas_image",
)


@dataclass
class Evidence:
    """One piece of channel evidence attached to a frame."""

    type: Channel
    score: float = 0.0
    text: str | None = None
    start: float | None = None
    end: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "score": round(self.score, 4)}
        if self.text is not None:
            d["text"] = self.text
        if self.start is not None:
            d["start"] = self.start
        if self.end is not None:
            d["end"] = self.end
        d.update(self.extra)
        return d


@dataclass
class ChannelHit:
    """A single hit returned by one retrieval channel, before fusion."""

    channel: Channel
    submit_keyframe_id: str
    video_id: str
    keyframe_n: int
    score: float  # raw channel score (already weighted/demoted by the channel)
    rank: int  # 0-based rank within the channel result list
    pts_time: float | None = None
    evidence: Evidence | None = None


@dataclass
class FusedFrame:
    """A frame after RRF fusion across channels."""

    submit_keyframe_id: str
    video_id: str
    keyframe_n: int
    pts_time: float | None
    score: float  # fused RRF score
    channels: list[Channel] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    per_channel_score: dict[str, float] = field(default_factory=dict)
    frame_idx: int | None = None  # absolute frame index in the video (for DRES submit)
    fps: float | None = None
    via_fill: bool = False  # True if surfaced by TRAKE pass-2 in-video fill (fallback relevance)
    # How good a pass-2 fill was RELATIVE to the best hit that event has anywhere
    # in the index (cosine ratio, or the RRF ratio when two models were combined).
    # `score` compresses it onto a small 0.02 relevance scale, which overlaps
    # real evidence rather than sitting below it (see `FILL_SCORE_SCALE`); this
    # keeps the undistorted value for the TRAKE heatmap, which normalizes every
    # event on its own scale.
    fill_quality: float | None = None
    # What the traffic-camera banner / race HUD printed on this frame (junction,
    # date, clock, stage), for the operator to check before submitting.
    overlay: dict[str, Any] = field(default_factory=dict)

    @property
    def image_id(self) -> str:
        return self.submit_keyframe_id


@dataclass
class VideoGroup:
    """Frames grouped by video, with an aggregate group score."""

    video_id: str
    video_score: float
    max_score: float
    mean_top_score: float
    frame_count: int
    timestamp_dispersion: float
    ambiguous: bool
    channels: list[Channel]
    frames: list[FusedFrame]
    best_clip: dict[str, Any] | None = None
