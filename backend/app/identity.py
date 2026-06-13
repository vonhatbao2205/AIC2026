"""Canonical identity handling for keyframes/videos.

Rules (from the handoff docs — these are load-bearing):
- `submit_keyframe_id` is the public id: "<category>/<video_id>/<frame_3_digits>",
  e.g. "K01/K01_V001/001". It doubles as `image_id`.
- Never use `image_path` / `source_path` for routing, joins, media lookup, cache
  keys, or submit. Robust joins use `video_id + keyframe_n`.
- L26 shards (`L26_a`, `L26_b`, ...) normalize to category `L26` for submit/display.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Matches a shard suffix like "_a", "_b", "_a1" appended to an L-style category.
_SHARD_SUFFIX = re.compile(r"^(?P<base>[A-Za-z]+\d+)(?:_[A-Za-z0-9]+)+$")


def normalize_category(raw_category: str) -> str:
    """Normalize a possibly-sharded category to its canonical submit category.

    "L26_a" -> "L26", "L26_b" -> "L26", "K01" -> "K01".
    """
    if not raw_category:
        return raw_category
    m = _SHARD_SUFFIX.match(raw_category.strip())
    if m:
        return m.group("base")
    return raw_category.strip()


def group_from_video_id(video_id: str) -> str:
    """Media group prefix: "K01_V001" -> "K01", "L26_V001" -> "L26"."""
    return video_id.split("_")[0]


def category_from_video_id(video_id: str) -> str:
    """Submit category derived from the video id (already shard-free in our data)."""
    return normalize_category(group_from_video_id(video_id))


def make_submit_keyframe_id(video_id: str, keyframe_n: int) -> str:
    """Build the canonical submit id from a robust (video_id, keyframe_n) join."""
    category = category_from_video_id(video_id)
    return f"{category}/{video_id}/{keyframe_n:03d}"


@dataclass(frozen=True)
class ParsedKeyframeId:
    category: str
    video_id: str
    keyframe_n: int
    keyframe_name: str  # zero-padded, e.g. "001"

    @property
    def submit_keyframe_id(self) -> str:
        return f"{self.category}/{self.video_id}/{self.keyframe_name}"


def parse_submit_keyframe_id(submit_keyframe_id: str) -> ParsedKeyframeId:
    """Parse and normalize a submit_keyframe_id.

    Accepts raw shard categories and normalizes them, so
    "L26_a/L26_V001/14" becomes category "L26", keyframe_n 14, name "014".
    """
    parts = submit_keyframe_id.strip().split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid submit_keyframe_id: {submit_keyframe_id!r}")
    raw_category, video_id, frame = parts
    frame = frame[:-4] if frame.endswith(".jpg") else frame
    if not frame.isdigit():
        raise ValueError(f"Invalid frame number in submit_keyframe_id: {submit_keyframe_id!r}")
    keyframe_n = int(frame)
    category = normalize_category(raw_category)
    return ParsedKeyframeId(
        category=category,
        video_id=video_id,
        keyframe_n=keyframe_n,
        keyframe_name=f"{keyframe_n:03d}",
    )


def canonical_submit_keyframe_id(submit_keyframe_id: str) -> str:
    """Return the normalized submit_keyframe_id (shard-free, zero-padded)."""
    return parse_submit_keyframe_id(submit_keyframe_id).submit_keyframe_id
