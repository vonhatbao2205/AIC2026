"""AVS review candidates: retrieved anchors only, no KIS offset expansion.

Reuse answer_gen's temporal NMS and pool construction, but use a separate
coverage policy. Each pass takes one moment per video in retrieval order.
These defaults are interaction heuristics, not parameters fitted on AVS labels.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .answer_gen import AnswerGenParams, build_pools

AVS_GAP_S = 2.0
AVS_PARAMS = AnswerGenParams(
    pool_depth=5000, max_videos=500, dedup_radius_s=AVS_GAP_S,
    anchor_support_weight=0, band_anchors=(64,) * 5,
    offsets=(), band_offsets=(0,) * 5,
)


def generate_avs(
    groups: list[dict[str, Any]], *, limit: int = 100,
    taken: list[tuple[str, int]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    # Never synthesize a time from a fallback fps for a competition candidate.
    usable = [dict(g, frames=[f for f in g.get("frames", [])
              if f.get("pts_time") is not None or (f.get("fps") or 0) > 0]) for g in groups]
    pools = build_pools(usable, replace(AVS_PARAMS).validated())
    queues = []
    for pool in pools:
        accepted = [idx for video, idx in (taken or []) if video == pool.video_id]
        moments = [a for a in pool.anchors if all(
            abs(a.frame_idx - idx) / pool.fps > AVS_GAP_S for idx in accepted
        )]
        queues.append((pool, moments, len(accepted)))
    # Videos with fewer already accepted moments get the first review slots.
    queues.sort(key=lambda item: (item[2], -item[0].video_score, item[0].video_id))
    answers: list[dict[str, Any]] = []
    for offset in range(max((len(q) for _, q, _ in queues), default=0)):
        for pool, queue, _ in queues:
            if offset >= len(queue):
                continue
            if len(answers) >= max(0, min(limit, 100)):
                break
            anchor = queue[offset]
            answers.append({
                "rank": len(answers) + 1, "video_id": pool.video_id,
                "frame_idx": anchor.frame_idx, "pts_time": anchor.pts_time,
                "keyframe_id": anchor.keyframe_id, "kind": "anchor",
                "offset_frames": 0,
            })
    return answers, {"n_videos_in_pool": len(pools),
                     "n_anchors": sum(len(p.anchors) for p in pools),
                     "videos_used": len({a["video_id"] for a in answers})}
