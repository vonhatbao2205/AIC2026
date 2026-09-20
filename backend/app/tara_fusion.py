"""TARA scale fusion and clip-to-video aggregation.

Every scale is collapsed to one vote per video before rank fusion. This stops
overlapping 8/24/72-second windows from rewarding long videos repeatedly.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .fusion import DEFAULT_RRF_K, DEPRIORITIZED_VIDEO_MULTIPLIER, PRIORITIZED_VIDEO_MULTIPLIER
from .types import Channel, ChannelHit, VideoGroup

SCALES = ("event", "sequence", "scene")


def fuse_tara_scales(
    hits_by_scale: dict[str, list[dict[str, Any]]], *, k: int = DEFAULT_RRF_K
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    best_clips: dict[str, dict[str, Any]] = {}
    scale_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for scale in SCALES:
        by_video: dict[str, dict[str, Any]] = {}
        for clip in hits_by_scale.get(scale, []):
            video_id = clip["video_id"]
            if video_id not in by_video or clip["score"] > by_video[video_id]["score"]:
                by_video[video_id] = clip
            if video_id not in best_clips or clip["score"] > best_clips[video_id]["score"]:
                best_clips[video_id] = clip
        ranked = sorted(by_video.values(), key=lambda row: (-row["score"], row["video_id"]))
        for rank, clip in enumerate(ranked, 1):
            video_id = clip["video_id"]
            scores[video_id] += 1.0 / (k + rank)
            scale_ranks[video_id][scale] = rank
    return sorted(
        (
            {
                "video_id": video_id,
                "score": score,
                "best_clip": best_clips[video_id],
                "scale_ranks": scale_ranks[video_id],
            }
            for video_id, score in scores.items()
        ),
        key=lambda row: (-row["score"], row["video_id"]),
    )


def fuse_video_rankings(
    groups: list[VideoGroup],
    channel_hits: dict[Channel, list[ChannelHit]],
    tara_videos: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
    k: int = DEFAULT_RRF_K,
    positive_videos: set[str] | None = None,
    negative_videos: set[str] | None = None,
    negative_frames: set[str] | None = None,
) -> list[VideoGroup]:
    """RRF PE/Qwen/text channel video ranks with the TARA video rank."""
    weights = weights or {}
    positive_videos = positive_videos or set()
    negative_videos = negative_videos or set()
    negative_frames = negative_frames or set()
    existing = {group.video_id: group for group in groups}
    scores: dict[str, float] = defaultdict(float)
    for channel, hits in channel_hits.items():
        weight = weights.get(channel, 1.0)
        if weight <= 0:
            continue
        seen: set[str] = set()
        for hit in hits:
            if hit.submit_keyframe_id in negative_frames:
                continue
            if hit.video_id in seen or hit.video_id not in existing:
                continue
            seen.add(hit.video_id)
            scores[hit.video_id] += weight / (k + len(seen))
    for rank, item in enumerate(tara_videos, 1):
        video_id = item["video_id"]
        scores[video_id] += weights.get("tara", 1.0) / (k + rank)
        group = existing.get(video_id)
        if group is None:
            group = VideoGroup(
                video_id=video_id, video_score=0.0, max_score=0.0,
                mean_top_score=0.0, frame_count=0, timestamp_dispersion=0.0,
                ambiguous=False, channels=["tara"], frames=[],
            )
            existing[video_id] = group
        if "tara" not in group.channels:
            group.channels.append("tara")
        group.best_clip = item["best_clip"]
    for video_id, group in existing.items():
        score = scores.get(video_id, 0.0)
        if video_id in positive_videos:
            score *= PRIORITIZED_VIDEO_MULTIPLIER
        if video_id in negative_videos:
            score *= DEPRIORITIZED_VIDEO_MULTIPLIER
        group.video_score = score
    return sorted(existing.values(), key=lambda group: (-group.video_score, group.video_id))
