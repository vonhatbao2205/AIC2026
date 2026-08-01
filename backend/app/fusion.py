"""Reciprocal Rank Fusion + group-by-video aggregation.

RRF (Cormack et al.): score(d) = sum_c weight_c / (k + rank_c(d)).
We fuse over `submit_keyframe_id`, aggregate evidence from every channel, then
group frames by `video_id` and compute a composite video score with an
"ambiguous" flag when a video's top frames split into distant time clusters.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev

from .types import ALL_CHANNELS, Channel, ChannelHit, Evidence, FusedFrame, VideoGroup

DEFAULT_RRF_K = 60

# Operator video priority is a SOFT multiplier on the aggregate score, not a pin:
# a prioritised video climbs, but a rival holding far stronger evidence still wins.
PRIORITIZED_VIDEO_MULTIPLIER = 1.5
DEPRIORITIZED_VIDEO_MULTIPLIER = 0.5


def reciprocal_rank_fusion(
    channel_hits: dict[Channel, list[ChannelHit]],
    *,
    weights: dict[str, float] | None = None,
    k: int = DEFAULT_RRF_K,
    negative_frames: set[str] | None = None,
) -> list[FusedFrame]:
    """Fuse per-channel ranked hits into a single ranked list of frames.

    `weights` scales each channel's contribution.

    Frame-level feedback is expressed here only as an exclusion: a frame the
    operator rejected is dropped. Positive feedback is NOT an additive bonus —
    an RRF score maxes out near `n_channels / (k + 1)` (~0.016 per channel), so
    any constant large enough to be felt also swamps the ranking. "More like
    this frame" is instead a real retrieval channel (`similar`), and video-level
    priority is a multiplier on the aggregate score in `group_by_video`.
    """
    weights = weights or {}
    negative_frames = negative_frames or set()

    fused: dict[str, FusedFrame] = {}

    for channel, hits in channel_hits.items():
        weight = weights.get(channel, 1.0)
        if weight <= 0:
            continue
        for hit in hits:
            kf_id = hit.submit_keyframe_id
            contribution = weight / (k + hit.rank + 1)
            frame = fused.get(kf_id)
            if frame is None:
                frame = FusedFrame(
                    submit_keyframe_id=kf_id,
                    video_id=hit.video_id,
                    keyframe_n=hit.keyframe_n,
                    pts_time=hit.pts_time,
                    score=0.0,
                )
                fused[kf_id] = frame
            frame.score += contribution
            frame.per_channel_score[channel] = (
                frame.per_channel_score.get(channel, 0.0) + hit.score
            )
            if channel not in frame.channels:
                frame.channels.append(channel)
            if hit.evidence is not None:
                frame.evidence.append(hit.evidence)
            if frame.pts_time is None and hit.pts_time is not None:
                frame.pts_time = hit.pts_time

    results: list[FusedFrame] = []
    for kf_id, frame in fused.items():
        if kf_id in negative_frames:
            continue
        # Order channels canonically for stable display.
        frame.channels = [c for c in ALL_CHANNELS if c in frame.channels]
        results.append(frame)

    results.sort(key=lambda f: (-f.score, f.submit_keyframe_id))
    return results


def _timestamp_dispersion(times: list[float]) -> float:
    """Population stddev of frame timestamps (seconds). 0 when <2 frames."""
    valid = [t for t in times if t is not None]
    if len(valid) < 2:
        return 0.0
    return float(pstdev(valid))


def _is_ambiguous(times: list[float], *, gap_threshold: float = 30.0) -> bool:
    """True if sorted top-frame timestamps contain a gap larger than threshold,
    i.e. the matches fall into distant temporal clusters within one video."""
    valid = sorted(t for t in times if t is not None)
    if len(valid) < 2:
        return False
    return any(b - a > gap_threshold for a, b in zip(valid, valid[1:]))


def group_by_video(
    frames: list[FusedFrame],
    *,
    top_n_for_mean: int = 3,
    w_best: float = 0.75,
    w_mean: float = 0.15,
    w_cluster: float = 0.10,
    cluster_window_seconds: float = 30.0,
    cluster_full_support: int = 4,
    ambiguous_gap_seconds: float = 30.0,
    prioritized_videos: set[str] | None = None,
    deprioritized_videos: set[str] | None = None,
) -> list[VideoGroup]:
    """Group fused frames by video and compute a composite video score.

    video_score = w_best * best_frame_norm
                + w_mean * mean_top3_norm
                + w_cluster * cluster_support      (all terms in [0, 1])

    Scores are normalized by the global best frame score so the dominant term is
    the single strongest frame: a video whose best frame is clearly weaker can
    NOT be lifted above a video holding a top frame just because it has many
    (mediocre) frames. `cluster_support` only counts frames temporally close to
    the best frame, and is bounded — it nudges near-ties, it does not bury raw
    top frames. (Replaces the old count-dominated additive formula.)

    Operator video feedback scales this aggregate. Applying it here rather than to
    every frame is deliberate: marking one frame must not lift all 380 frames of
    its video above every other video's best frame.
    """
    prioritized_videos = prioritized_videos or set()
    deprioritized_videos = deprioritized_videos or set()
    by_video: dict[str, list[FusedFrame]] = defaultdict(list)
    for frame in frames:
        by_video[frame.video_id].append(frame)

    all_scores = [f.score for f in frames]
    max_overall = max(all_scores) if all_scores else 1.0
    if max_overall <= 0:
        max_overall = 1.0

    groups: list[VideoGroup] = []
    for video_id, vframes in by_video.items():
        vframes.sort(key=lambda f: (-f.score, f.keyframe_n))
        scores = [f.score for f in vframes]
        top_scores = scores[:top_n_for_mean]
        max_score = max(scores)
        mean_top = mean(top_scores)
        count = len(vframes)
        times = [f.pts_time for f in vframes[: max(top_n_for_mean, 5)]]
        dispersion = _timestamp_dispersion(times)
        ambiguous = _is_ambiguous(times, gap_threshold=ambiguous_gap_seconds)

        # Temporal support: frames clustered near the best frame (not scattered).
        best_time = vframes[0].pts_time
        if best_time is None:
            support_count = count
        else:
            support_count = sum(
                1
                for f in vframes
                if f.pts_time is not None and abs(f.pts_time - best_time) <= cluster_window_seconds
            )
        cluster_support = min(1.0, max(0, support_count - 1) / cluster_full_support)

        video_score = (
            w_best * (max_score / max_overall)
            + w_mean * (mean_top / max_overall)
            + w_cluster * cluster_support
        )
        if video_id in prioritized_videos:
            video_score *= PRIORITIZED_VIDEO_MULTIPLIER
        if video_id in deprioritized_videos:
            video_score *= DEPRIORITIZED_VIDEO_MULTIPLIER

        channels: list[Channel] = []
        for f in vframes:
            for c in f.channels:
                if c not in channels:
                    channels.append(c)
        channels = [c for c in ALL_CHANNELS if c in channels]

        groups.append(
            VideoGroup(
                video_id=video_id,
                video_score=video_score,
                max_score=max_score,
                mean_top_score=mean_top,
                frame_count=count,
                timestamp_dispersion=dispersion,
                ambiguous=ambiguous,
                channels=channels,
                frames=vframes,
            )
        )

    groups.sort(key=lambda g: (-g.video_score, g.video_id))
    return groups
