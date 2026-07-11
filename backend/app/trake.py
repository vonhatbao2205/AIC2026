"""TRAKE sequence assembly and BTC keyframe snapping.

TRAKE = locate an ordered set of events E1..En inside ONE video, where the chosen
frames must occur in strictly increasing time. Given each event's ranked
candidate frames, the best per-video assignment is found with an exact
dynamic-programming strictly-increasing chain over retrieved candidates (by
event index AND time): it picks at most one frame per event, in order, maximizing
(events covered, then total relevance). This is globally optimal — unlike a greedy
left-to-right pick it never gets trapped by a locally-good early choice, and it
can skip an event (partial sequence) when no orderable candidate exists.

`snap_to_keyframe` powers the (legacy) pause-frame picker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import FusedFrame


@dataclass
class TrakeCandidateFrame:
    event_index: int
    submit_keyframe_id: str
    keyframe_n: int
    pts_time: float
    score: float
    frame_idx: int | None = None
    via_fill: bool = False  # came from pass-2 in-video fill (small 0.02-scale score)


@dataclass
class TrakeSequence:
    video_id: str
    score: float  # pure relevance (sum of chosen frame scores), coverage excluded
    complete: bool  # every event filled
    frames: list[TrakeCandidateFrame]  # ordered by event_index
    coverage: int = 0  # number of events covered (real + pass-2 fill)
    warning: str | None = None
    # Events backed by genuine pass-1 retrieval (excludes pass-2 in-video fills).
    # This drives ranking so a fill-heavy full-coverage video cannot bury a video
    # with more direct evidence; `filled_events = coverage - confident_coverage`.
    confident_coverage: int = 0
    filled_events: int = 0
    mean_score: float = 0.0  # mean relevance over real (non-filled) frames
    min_score: float = 0.0  # weakest real frame — low value flags a shaky chain

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "score": round(self.score, 4),
            "complete": self.complete,
            "coverage": self.coverage,
            "confident_coverage": self.confident_coverage,
            "filled_events": self.filled_events,
            "mean_score": round(self.mean_score, 4),
            "min_score": round(self.min_score, 4),
            "warning": self.warning,
            "frames": [
                {
                    "event_index": f.event_index,
                    "submit_keyframe_id": f.submit_keyframe_id,
                    "keyframe_n": f.keyframe_n,
                    "frame_idx": f.frame_idx,
                    "pts_time": f.pts_time,
                    "score": round(f.score, 4),
                    "via_fill": f.via_fill,
                }
                for f in self.frames
            ],
        }


def _best_increasing_chain(
    per_event: list[list[TrakeCandidateFrame]], n_events: int
) -> list[TrakeCandidateFrame | None]:
    """Exact DP: lexicographic best chain of candidates with strictly increasing
    (event_index, pts_time). The state value is compared lexicographically as
    (coverage, relevance), so adding one covered event always dominates any
    relevance difference.
    Returns one pick per event (None where that event isn't covered)."""
    nodes: list[TrakeCandidateFrame] = [c for cands in per_event for c in cands]
    if not nodes:
        return [None] * n_events
    # Topological order for the (event, time) DAG: ascending event then time.
    nodes.sort(key=lambda c: (c.event_index, c.pts_time))
    m = len(nodes)
    dp: list[tuple[int, float]] = [(0, 0.0)] * m
    parent = [-1] * m
    best_i, best_val = -1, (-1, float("-inf"))
    for i in range(m):
        ni = nodes[i]
        best_prev, best_j = (0, 0.0), -1
        for j in range(i):
            nj = nodes[j]
            if nj.event_index < ni.event_index and nj.pts_time < ni.pts_time and dp[j] > best_prev:
                best_prev, best_j = dp[j], j
        dp[i] = (best_prev[0] + 1, best_prev[1] + ni.score)
        parent[i] = best_j
        if dp[i] > best_val:
            best_val, best_i = dp[i], i
    picks: list[TrakeCandidateFrame | None] = [None] * n_events
    k = best_i
    while k != -1:
        c = nodes[k]
        picks[c.event_index - 1] = c
        k = parent[k]
    return picks


def assemble_trake_sequences(
    event_frames: list[list[FusedFrame]],
    *,
    fallback_partial: bool = True,
    max_results: int = 50,
    per_video_event_cap: int = 12,
) -> list[TrakeSequence]:
    """Build the best ordered sequence per video from each event's fused frames,
    via the exact DP above. `event_frames[i]` is the ranked frame list for E{i+1}.

    Videos are ranked by (coverage desc, total relevance desc). Per video we keep
    only the top `per_video_event_cap` candidates per event (by score) to bound the
    DP while preserving recall across videos.
    """
    n_events = len(event_frames)
    if n_events == 0:
        return []

    videos: dict[str, list[list[TrakeCandidateFrame]]] = {}
    for ev_idx, frames in enumerate(event_frames):
        for f in frames:
            if f.pts_time is None:
                continue
            slots = videos.setdefault(f.video_id, [[] for _ in range(n_events)])
            slots[ev_idx].append(
                TrakeCandidateFrame(
                    event_index=ev_idx + 1,
                    submit_keyframe_id=f.submit_keyframe_id,
                    keyframe_n=f.keyframe_n,
                    pts_time=f.pts_time,
                    score=f.score,
                    frame_idx=f.frame_idx,
                    via_fill=f.via_fill,
                )
            )

    sequences: list[TrakeSequence] = []
    for video_id, per_event in videos.items():
        capped = [sorted(cs, key=lambda c: -c.score)[:per_video_event_cap] for cs in per_event]
        picks = _best_increasing_chain(capped, n_events)
        filled = [p for p in picks if p is not None]
        if not filled:
            continue
        coverage = len(filled)
        complete = coverage == n_events
        if not complete and not fallback_partial:
            continue
        warning = None
        if not complete:
            missing = [str(i + 1) for i, p in enumerate(picks) if p is None]
            warning = f"Partial sequence: events {', '.join(missing)} have no orderable candidate."
        # Quality is measured over real (pass-1) frames; the synthetic 0.02-scale
        # fill scores would otherwise drag mean/min down and misreport confidence.
        real = [p for p in filled if not p.via_fill]
        quality_src = real or filled
        sequences.append(
            TrakeSequence(
                video_id=video_id,
                score=sum(p.score for p in filled),
                complete=complete,
                frames=filled,
                coverage=coverage,
                warning=warning,
                confident_coverage=len(real),
                filled_events=coverage - len(real),
                mean_score=sum(p.score for p in quality_src) / len(quality_src),
                min_score=min(p.score for p in quality_src),
            )
        )

    # Rank by REAL coverage first so a fill-heavy full-coverage video cannot
    # outrank one with more direct evidence; then total
    # coverage (the extra filled event still helps), then relevance, then id.
    sequences.sort(key=lambda s: (-s.confident_coverage, -s.coverage, -s.score, s.video_id))
    return sequences[:max_results]


def validate_increasing_order(pts_times: list[float | None]) -> list[int]:
    """Return 1-based indices of slots whose pts_time is not greater than the
    previous filled slot (i.e. order violations). Empty list = valid order.

    Used by the submit guard and TRAKE slot UI to warn "E2 earlier than E1".
    """
    violations: list[int] = []
    last: float | None = None
    for i, t in enumerate(pts_times):
        if t is None:
            continue
        if last is not None and t <= last:
            violations.append(i + 1)
        last = t
    return violations


@dataclass
class SnapResult:
    submit_keyframe_id: str
    keyframe_n: int
    frame_idx: int
    pts_time: float
    raw_frame_idx: int
    raw_time: float
    delta_frames: int
    delta_seconds: float
    far: bool  # snapped keyframe is far from the paused frame
    field_count: int = field(default=0, repr=False)


def snap_to_keyframe(
    raw_time: float,
    fps: float,
    keyframes: list[dict[str, Any]],
    *,
    far_threshold_seconds: float = 2.0,
) -> SnapResult | None:
    """Snap a raw paused time to the nearest BTC keyframe.

    Primary key is `frame_idx` (round(raw_time * fps)); falls back to nearest
    `pts_time` when frame_idx is unavailable on the keyframe records.
    `keyframes` items must have submit_keyframe_id, keyframe_n, frame_idx, pts_time.
    """
    if not keyframes:
        return None
    raw_frame_idx = round(raw_time * fps) if fps else 0

    has_frame_idx = all(k.get("frame_idx") is not None for k in keyframes)
    if has_frame_idx:
        nearest = min(keyframes, key=lambda k: abs(int(k["frame_idx"]) - raw_frame_idx))
    else:
        nearest = min(keyframes, key=lambda k: abs(float(k["pts_time"]) - raw_time))

    kf_frame_idx = int(nearest.get("frame_idx") or round(float(nearest["pts_time"]) * fps))
    kf_pts = float(nearest["pts_time"])
    delta_frames = kf_frame_idx - raw_frame_idx
    delta_seconds = kf_pts - raw_time
    return SnapResult(
        submit_keyframe_id=nearest["submit_keyframe_id"],
        keyframe_n=int(nearest["keyframe_n"]),
        frame_idx=kf_frame_idx,
        pts_time=kf_pts,
        raw_frame_idx=raw_frame_idx,
        raw_time=raw_time,
        delta_frames=delta_frames,
        delta_seconds=delta_seconds,
        far=abs(delta_seconds) > far_threshold_seconds,
    )
