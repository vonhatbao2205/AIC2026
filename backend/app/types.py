"""Internal dataclasses used by the fusion/grouping/trake logic.

Kept free of pydantic/FastAPI so the core ranking logic is trivially unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Channel = Literal["image_pe", "ocr", "speech", "audio"]
ALL_CHANNELS: tuple[Channel, ...] = ("image_pe", "ocr", "speech", "audio")


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
    via_fill: bool = False  # True if surfaced by TRAKE pass-2 in-video fill (synthetic relevance)

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
