"""Rank-budget allocator: turn a group-by-video result into the ordered answer list.

The preliminary round scores one query as

    Final = (R@1 + R@5 + R@20 + R@50 + R@100) / 5,   R@k = max_{i<=k} R-Score(r_i)

so the list is not "the 100 best-scoring frames" — it is a budget spread over the
two ways an answer can be wrong: the wrong video, or the right video at the wrong
instant. Every position is filled with the candidate adding the most NEW chance of
a hit before the next cutoff, scored as

    U_r(V, c) = pi_B(V) x E(c|V) x N(c|A_V)

pi_B  video confidence, a softmax over the group score at the current band's
      temperature (sharp at rank 1, flat by rank 100 — exploitation to hedging);
E     temporal evidence of the candidate inside its own video (an anchor keeps its
      retrieval score, an offset inherits the anchor's, decayed by |eps|);
N     novelty, how much of the video's timeline this candidate adds over the
      answers already picked for it.

Because N drops as a region fills up, a strong video takes the early positions on
its own merit and then yields to the next one without any hard per-video quota.

Two candidate kinds compete in that one ranking:
  * anchors — the distinct temporal hypotheses left after temporal NMS over the
    fused frames ("the event may be somewhere else entirely in this video");
  * offsets — anchor +/- eps ("right region, but the retrieved keyframe sits
    outside the accepted window"). eps is a multi-scale ladder calibrated from
    the retrieval-to-ground-truth error distribution, not a guess.

`ambiguous` (top frames of one video split into distant clusters) does not demote
a video; it says the uncertainty is temporal, so that video opens more anchors and
fewer offsets per band. The fitted default leaves this OFF — see the field docs.

No position is ever left empty. `R@k` is a max, so a wrong answer at rank 90 costs
nothing while an empty rank 90 forfeits whatever it might have caught: when a
band's frontier runs out the gate is dropped for that position, and when the whole
candidate space is too thin the eps ladder densifies until 100 can be filled.

Deterministic: same input + same params => same list, no RNG anywhere.
Parameters live in `AnswerGenParams`; the values shipped as DEFAULT_PARAMS were
fitted on the L21-L30 ground-truth queries (see `benchmarks/run_answer_gen.py`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Sequence

#: The cutoffs Final Score averages over. Position r belongs to the first band
#: whose cutoff it does not exceed.
BANDS: tuple[int, ...] = (1, 5, 20, 50, 100)

MAX_ANSWERS = 100

#: Used only when a frame carries no fps (never seen in the live indices, but the
#: mock fixtures and hand-made payloads can omit it).
FALLBACK_FPS = 25.0


def band_index(rank: int) -> int:
    """0-based band of a 1-based answer position."""
    for i, cutoff in enumerate(BANDS):
        if rank <= cutoff:
            return i
    return len(BANDS) - 1


@dataclass(frozen=True)
class AnswerGenParams:
    """Everything the generator is allowed to be tuned on.

    Kept in one frozen object so a run can record exactly what produced it, and
    so a sweep can `replace()` one field without touching the rest.
    """

    # ---- candidate pool -------------------------------------------------
    #: How deep into the fused frame list the pool is built from. Not the answer
    #: count: it is how many retrieved frames may become hypotheses. Fitted
    #: shallow — past ~100 frames the extra videos only dilute the softmax.
    pool_depth: int = 100
    #: Videos considered at all. Beyond this the softmax mass is noise anyway.
    max_videos: int = 100

    # ---- temporal dedup (anchors) ---------------------------------------
    #: Two frames closer than this (seconds) are the same hypothesis. Fitted
    #: SMALL: InfoShot++ keyframes sit 6-18 frames apart (median 10 at 25 fps),
    #: so 0.3 s still folds away true near-duplicates while a 2 s radius would
    #: collapse five distinct shots into one and cost 0.044 Final.
    dedup_radius_s: float = 0.3
    #: Absolute ceiling on hypotheses per video. How many are actually built is
    #: driven by `band_anchors` (+ the ambiguous bonus) — this only stops a
    #: pathological band setting from turning one video into the whole pool.
    max_anchors_per_video: int = 64
    #: How much the frames an anchor absorbed add to its evidence. 0 = the
    #: anchor is worth exactly its own retrieval score (no count bias); the
    #: fitted 0.25 lets support break near-ties without reintroducing raw count.
    anchor_support_weight: float = 0.25

    # ---- video confidence ------------------------------------------------
    #: Softmax temperature per band; must be non-decreasing (sharp -> flat).
    temperatures: tuple[float, ...] = (0.05, 0.05, 0.10, 0.20, 0.40)

    # ---- coverage candidates (offsets) -----------------------------------
    #: The eps ladder, in FRAMES, smallest first. Calibrated from the retrieval
    #: error percentiles on the dev set: |best anchor - ground truth| has median
    #: ~15 frames, Q75 ~30, and a tail in the hundreds.
    offsets: tuple[int, ...] = (25, 75, 150, 300)
    #: Evidence multiplier applied once per ladder step: level k keeps
    #: decay**(k+1) of its anchor's evidence.
    offset_decay: float = 0.72

    # ---- novelty ----------------------------------------------------------
    #: Width (seconds) of the suppression kernel around an already-picked answer.
    novelty_sigma_s: float = 1.0
    #: Novelty left at zero distance. >0 keeps a re-pick merely bad, not banned,
    #: which matters when a video has run out of everything else.
    novelty_floor: float = 0.25
    #: "product" multiplies the penalty of every picked answer (probabilistic
    #: coverage); "min" only counts the nearest one.
    novelty_mode: Literal["product", "min"] = "product"

    # ---- frontier (how much of a video is exposed at each band) -----------
    #: Anchors per video visible at band 1 / 5 / 20 / 50 / 100.
    band_anchors: tuple[int, ...] = (1, 4, 8, 12, 20)
    #: Offset ladder levels per anchor visible at each band.
    band_offsets: tuple[int, ...] = (1, 2, 3, 4, 5)
    #: An ambiguous video opens this many extra anchors...
    #:
    #: Fitted to 0, i.e. OFF. On the 21 dev queries the flag has no measurable
    #: effect either way (bonus 1 / penalty 1 costs 0.004 Final, bonus 2 costs
    #: 0.006 — both smaller than one query moving one band, which is 0.0095).
    #: The mechanism is kept and remains settable per request rather than
    #: deleted, because "no signal on 21 queries" is not "no signal".
    ambiguous_anchor_bonus: int = 0
    #: ...and this many fewer offset levels, because its uncertainty is "which
    #: region", not "how far off inside one region".
    ambiguous_offset_penalty: int = 0

    # ---- output guards ----------------------------------------------------
    #: How much a video already holding answers is discounted, per answer it
    #: holds: pi'(V) = pi(V) / (1 + penalty * taken(V)).
    #:
    #: Novelty already handles coverage WITHIN a video, but not the video-level
    #: bet. Positions spent on a video only pay off in the world where that video
    #: is right, and in that world the answers already there are likely to have
    #: caught it — so the marginal position is worth less than the first one was.
    #: This is what lets a second video reach the R@5 band when a person has
    #: already claimed rank 1: measured on a real submission, 18 of 24 questions
    #: put all five top positions in one video. 0 disables it.
    taken_video_penalty: float = 0.0
    #: Answers in one video closer than this (frames) are dropped outright.
    #: 0 leaves the job entirely to the novelty term.
    min_answer_gap_frames: int = 0
    #: Offsets may run this many frames past the last keyframe the video showed
    #: (the real end of the video is not in the retrieval payload).
    tail_margin_frames: int = 750
    #: Pull an offset onto the nearest retrieved keyframe when one is close.
    #: Measured on the dev set, half the ground-truth frames ARE an extracted
    #: keyframe (median |nearest retrieved frame - GT| = 0), so an offset that
    #: snaps both probes the accepted window and spends the position on a frame
    #: the retriever actually saw. Temporal NMS folded those neighbours into an
    #: anchor; snapping is how they come back, ranked instead of enumerated.
    snap_offsets: bool = True
    #: How far an offset may be pulled to reach a keyframe. Past this the raw
    #: offset is kept, because the ladder step itself carries the information.
    #: This is the single biggest lever measured: turning snapping off costs
    #: 0.052 Final, more than any other parameter in the grid.
    snap_radius_frames: int = 60

    def validated(self) -> "AnswerGenParams":
        """Clamp to the ranges the maths assumes, rather than trusting a caller."""
        temps = tuple(max(1e-4, float(t)) for t in self.temperatures) or (0.1,)
        temps = temps + temps[-1:] * (len(BANDS) - len(temps))
        anchors = tuple(max(1, int(a)) for a in self.band_anchors) or (1,)
        anchors = anchors + anchors[-1:] * (len(BANDS) - len(anchors))
        offs = tuple(max(0, int(o)) for o in self.band_offsets) or (0,)
        offs = offs + offs[-1:] * (len(BANDS) - len(offs))
        return replace(
            self,
            pool_depth=max(1, int(self.pool_depth)),
            max_videos=max(1, int(self.max_videos)),
            dedup_radius_s=max(0.0, float(self.dedup_radius_s)),
            max_anchors_per_video=max(1, int(self.max_anchors_per_video)),
            anchor_support_weight=max(0.0, float(self.anchor_support_weight)),
            temperatures=temps[: len(BANDS)],
            offsets=tuple(sorted({max(1, int(e)) for e in self.offsets})),
            offset_decay=min(1.0, max(1e-3, float(self.offset_decay))),
            novelty_sigma_s=max(1e-3, float(self.novelty_sigma_s)),
            novelty_floor=min(1.0, max(0.0, float(self.novelty_floor))),
            band_anchors=anchors[: len(BANDS)],
            band_offsets=offs[: len(BANDS)],
            ambiguous_anchor_bonus=max(0, int(self.ambiguous_anchor_bonus)),
            ambiguous_offset_penalty=max(0, int(self.ambiguous_offset_penalty)),
            taken_video_penalty=max(0.0, float(self.taken_video_penalty)),
            min_answer_gap_frames=max(0, int(self.min_answer_gap_frames)),
            tail_margin_frames=max(0, int(self.tail_margin_frames)),
            snap_offsets=bool(self.snap_offsets),
            snap_radius_frames=max(0, int(self.snap_radius_frames)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_depth": self.pool_depth,
            "max_videos": self.max_videos,
            "dedup_radius_s": self.dedup_radius_s,
            "max_anchors_per_video": self.max_anchors_per_video,
            "anchor_support_weight": self.anchor_support_weight,
            "temperatures": list(self.temperatures),
            "offsets": list(self.offsets),
            "offset_decay": self.offset_decay,
            "novelty_sigma_s": self.novelty_sigma_s,
            "novelty_floor": self.novelty_floor,
            "novelty_mode": self.novelty_mode,
            "band_anchors": list(self.band_anchors),
            "band_offsets": list(self.band_offsets),
            "ambiguous_anchor_bonus": self.ambiguous_anchor_bonus,
            "ambiguous_offset_penalty": self.ambiguous_offset_penalty,
            "taken_video_penalty": self.taken_video_penalty,
            "min_answer_gap_frames": self.min_answer_gap_frames,
            "tail_margin_frames": self.tail_margin_frames,
            "snap_offsets": self.snap_offsets,
            "snap_radius_frames": self.snap_radius_frames,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AnswerGenParams":
        """Build from a partial dict; unknown keys are ignored, missing ones keep
        the fitted default. Sequence fields are coerced to tuples so the object
        stays hashable and comparable."""
        base = DEFAULT_PARAMS
        if not data:
            return base
        fields: dict[str, Any] = {}
        for key, value in data.items():
            if not hasattr(base, key) or value is None:
                continue
            current = getattr(base, key)
            if isinstance(current, tuple):
                if not isinstance(value, (list, tuple)):
                    continue
                fields[key] = tuple(value)
            elif isinstance(current, bool):
                fields[key] = bool(value)
            elif isinstance(current, int) and not isinstance(current, bool):
                fields[key] = int(value)
            elif isinstance(current, float):
                fields[key] = float(value)
            else:
                fields[key] = value
        return replace(base, **fields).validated()


#: Fitted on the 21 L21-L30 ground-truth T-KIS queries (InfoShot++ profile) by
#: `benchmarks/run_answer_gen.py --mode sweep`: coordinate descent from 31
#: starting points, all of which converged here.
#:
#: The objective is Final averaged over tolerances {0, 12, 25, 50, 100} frames,
#: not a single one. The organisers score against a window [s, e] whose width
#: nobody knows in advance; fitting at one tolerance picks a strategy for that
#: assumption alone. Measured: fitting on {25..200} only reaches 0.743 at
#: tol 200 but collapses to 0.029 at tol 0, while these parameters hold
#: 0.324 / 0.657 / 0.714 at tol 0 / 25 / 200.
#:
#: In-sample Final (21 queries): 0.324 at tol 0, 0.657 at tol 25, 0.705 at
#: tol 100. Honest generalisation, from 3-fold x 2 cross-validation
#: (`--mode cv`, `benchmarks/runs/answer_gen_cv.json`): tuned 0.515 vs untuned
#: 0.484 on held-out queries — so the tuning is worth about +0.03, not the
#: +0.107 the in-sample number suggests.
DEFAULT_PARAMS = AnswerGenParams()


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


@dataclass
class Anchor:
    """One distinct temporal hypothesis inside a video."""

    frame_idx: int
    pts_time: float
    keyframe_id: str | None
    score: float
    support: float  # summed score of the frames temporal NMS folded into it
    evidence: float = 1.0  # normalized within the video, filled by the builder


@dataclass
class VideoPool:
    video_id: str
    video_score: float
    ambiguous: bool
    fps: float
    min_frame: int
    max_frame: int
    anchors: list[Anchor] = field(default_factory=list)
    #: Every retrieved frame index of this video, sorted — the snap targets.
    keyframes: list[int] = field(default_factory=list)


def _frame_time(frame: dict[str, Any], fps: float) -> float | None:
    pts = frame.get("pts_time")
    if pts is not None:
        return float(pts)
    idx = frame.get("frame_idx")
    if idx is not None and fps > 0:
        return float(idx) / fps
    return None


def _video_fps(frames: Sequence[dict[str, Any]]) -> float:
    for frame in frames:
        fps = frame.get("fps")
        if fps:
            return float(fps)
    return FALLBACK_FPS


def anchor_budget(params: AnswerGenParams) -> int:
    """How many anchors a video can ever need: the deepest band, plus the extra
    an ambiguous video is allowed. Derived rather than tuned separately, so the
    two knobs cannot silently contradict each other."""
    return min(
        params.max_anchors_per_video,
        max(params.band_anchors) + params.ambiguous_anchor_bonus,
    )


def build_anchors(frames: Sequence[dict[str, Any]], params: AnswerGenParams, fps: float) -> list[Anchor]:
    """Temporal NMS over one video's fused frames.

    Greedy from the strongest frame: it becomes an anchor and swallows every
    weaker frame within `dedup_radius_s`, keeping their summed score as support.
    What survives is a set of hypotheses that are actually distinct in time,
    which is the point — 30 near-identical frames of one shot must not be able to
    spend 30 of the 100 positions.
    """
    usable = [
        f
        for f in frames
        if f.get("frame_idx") is not None and _frame_time(f, fps) is not None
    ]
    usable.sort(key=lambda f: (-float(f.get("score") or 0.0), int(f["frame_idx"])))

    anchors: list[Anchor] = []
    taken: list[float] = []
    radius = params.dedup_radius_s
    budget = anchor_budget(params)
    for frame in usable:
        t = _frame_time(frame, fps)
        assert t is not None
        score = float(frame.get("score") or 0.0)
        near = next((i for i, at in enumerate(taken) if abs(at - t) <= radius), None)
        if near is not None:
            anchors[near].support += score
            continue
        if len(anchors) >= budget:
            continue
        anchors.append(
            Anchor(
                frame_idx=int(frame["frame_idx"]),
                pts_time=t,
                keyframe_id=frame.get("submit_keyframe_id") or frame.get("image_id"),
                score=score,
                support=0.0,
            )
        )
        taken.append(t)

    if not anchors:
        return []
    raw = [a.score + params.anchor_support_weight * a.support for a in anchors]
    top = max(raw) or 1.0
    for anchor, value in zip(anchors, raw):
        anchor.evidence = value / top
    # Strongest hypothesis first: the frontier exposes anchors in this order.
    order = sorted(range(len(anchors)), key=lambda i: (-raw[i], anchors[i].frame_idx))
    return [anchors[i] for i in order]


def _snap(target: int, keyframes: Sequence[int], radius: int) -> int:
    """Nearest retrieved frame index within `radius`, else `target` unchanged."""
    if not keyframes or radius <= 0:
        return target
    lo, hi = 0, len(keyframes)
    while lo < hi:
        mid = (lo + hi) // 2
        if keyframes[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    best = target
    best_d = radius + 1
    for i in (lo - 1, lo):
        if 0 <= i < len(keyframes):
            d = abs(keyframes[i] - target)
            if d < best_d:
                best_d, best = d, keyframes[i]
    return best if best_d <= radius else target


def build_pools(groups: Sequence[dict[str, Any]], params: AnswerGenParams) -> list[VideoPool]:
    """Group serialized search output into per-video candidate pools.

    `pool_depth` is applied GLOBALLY over the fused frames, not per video: it is
    the depth of the retrieved list the hypotheses may come from, so a deeper
    pool must be able to add a whole new video rather than only more frames of
    the ones already in.
    """
    flat: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
    for group in groups[: params.max_videos]:
        for frame in group.get("frames") or []:
            if frame.get("frame_idx") is None:
                continue
            flat.append((-float(frame.get("score") or 0.0), int(frame["frame_idx"]), frame, group))
    flat.sort(key=lambda item: (item[0], item[1]))
    kept = flat[: params.pool_depth]

    by_video: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for _, _, frame, group in kept:
        vid = str(group.get("video_id") or frame.get("video_id") or "")
        if not vid:
            continue
        by_video.setdefault(vid, []).append(frame)
        meta.setdefault(vid, group)

    pools: list[VideoPool] = []
    for vid, frames in by_video.items():
        group = meta[vid]
        fps = _video_fps(frames)
        anchors = build_anchors(frames, params, fps)
        if not anchors:
            continue
        indices = sorted({int(f["frame_idx"]) for f in frames})
        pools.append(
            VideoPool(
                video_id=vid,
                video_score=float(group.get("video_score") or 0.0),
                ambiguous=bool(group.get("ambiguous")),
                fps=fps,
                min_frame=indices[0],
                max_frame=indices[-1],
                anchors=anchors,
                keyframes=indices,
            )
        )
    pools.sort(key=lambda p: (-p.video_score, p.video_id))
    return pools


#: How far the eps ladder may be grown to fill the answer list, and by how much
#: per step. Bounded so a pathologically thin pool cannot generate forever. The
#: bound can be generous because growth only runs while the candidate space is
#: too thin to fill 100 — a rich result never reaches it.
MAX_LADDER_LEVELS = 64
LADDER_GROWTH = 1.8
#: A rung wider than this names a frame no video in the corpus has (an hour at
#: 25 fps). Growth stops there rather than emitting arithmetic that is not a frame.
MAX_OFFSET_FRAMES = 90_000


#: Rungs closer together than this probe the same instant twice.
MIN_LADDER_STEP = 3


def _grow_ladder(params: AnswerGenParams) -> AnswerGenParams | None:
    """A denser eps ladder — or None when it cannot usefully get denser.

    Unused positions are wasted, not saved: `R@k` is a max over the first k
    answers, so a wrong answer at rank 90 costs nothing while an empty rank 90
    forfeits whatever it might have caught. When a thin result (one video and a
    handful of keyframes, or two TRAKE chains) cannot fill 100 with the fitted
    ladder, this is what makes up the difference.

    It DENSIFIES before it extends: halving the gaps closes the "no answer lands
    in the accepted window" holes the design note warns about, whereas marching
    outward past the end of the video only names frames that do not exist. Only
    once every gap is down to `MIN_LADDER_STEP` does it add a wider rung.
    """
    if len(params.offsets) >= MAX_LADDER_LEVELS or not params.offsets:
        return None
    rungs = list(params.offsets)
    midpoints = sorted(
        {
            (a + b) // 2
            for a, b in zip([0, *rungs], rungs)
            if b - a >= 2 * MIN_LADDER_STEP
        }
        - set(rungs)
    )
    if midpoints and len(rungs) + len(midpoints) <= MAX_LADDER_LEVELS:
        # Never truncate: dropping the widest rungs would shrink the range the
        # ladder covers, which is the opposite of what growing it is for.
        offsets = tuple(sorted(set(rungs) | set(midpoints)))
    elif rungs[-1] < MAX_OFFSET_FRAMES:
        offsets = (*rungs, max(rungs[-1] + 1, int(round(rungs[-1] * LADDER_GROWTH))))
    else:
        return None
    # The deepest band has to be able to see the rungs that were just added.
    band_offsets = (*params.band_offsets[:-1], max(params.band_offsets[-1], len(offsets)))
    return replace(params, offsets=offsets, band_offsets=band_offsets)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class GeneratedAnswer:
    rank: int
    video_id: str
    frame_idx: int
    keyframe_id: str | None
    pts_time: float | None
    kind: Literal["anchor", "offset"]
    anchor_index: int
    offset_frames: int
    utility: float
    video_weight: float
    evidence: float
    novelty: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "keyframe_id": self.keyframe_id,
            "pts_time": None if self.pts_time is None else round(self.pts_time, 3),
            "kind": self.kind,
            "anchor_index": self.anchor_index,
            "offset_frames": self.offset_frames,
            # Kept for debugging and parameter fitting; never submitted.
            "utility": round(self.utility, 9),
            "video_weight": round(self.video_weight, 6),
            "evidence": round(self.evidence, 6),
            "novelty": round(self.novelty, 6),
        }


def _softmax(scores: Sequence[float], temperature: float) -> list[float]:
    if not scores:
        return []
    top = max(scores)
    exps = [math.exp((s - top) / temperature) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _flatten(
    pools: Sequence[VideoPool], params: AnswerGenParams
) -> tuple[list[int], list[int], list[int], list[int], list[float], list[float], list[int]]:
    """Every candidate of every video as parallel arrays.

    Parallel arrays rather than objects: the selection loop touches all of them
    once per answer and this is the hot path of a UI action.
    """
    c_pool: list[int] = []       # index into pools
    c_anchor: list[int] = []     # index into that pool's anchors
    c_level: list[int] = []      # 0 = the anchor itself, k>=1 = eps ladder rung k
    c_frame: list[int] = []
    c_time: list[float] = []
    c_evidence: list[float] = []
    c_offset: list[int] = []

    margin = params.tail_margin_frames
    for p_i, pool in enumerate(pools):
        lo = max(0, pool.min_frame - margin)
        hi = pool.max_frame + margin
        seen_frames: set[int] = set()
        for a_i, anchor in enumerate(pool.anchors):
            variants: list[tuple[int, int, float]] = [(0, anchor.frame_idx, anchor.evidence)]
            for level, eps in enumerate(params.offsets, start=1):
                evidence_at = anchor.evidence * (params.offset_decay ** level)
                for sign in (-1, 1):
                    variants.append((level, anchor.frame_idx + sign * eps, evidence_at))
            for level, frame_idx, ev in variants:
                if level and params.snap_offsets:
                    frame_idx = _snap(frame_idx, pool.keyframes, params.snap_radius_frames)
                if frame_idx < lo or frame_idx > hi or frame_idx < 0:
                    continue
                if frame_idx in seen_frames:
                    continue
                seen_frames.add(frame_idx)
                c_pool.append(p_i)
                c_anchor.append(a_i)
                c_level.append(level)
                c_frame.append(frame_idx)
                c_time.append(frame_idx / pool.fps if pool.fps else 0.0)
                c_evidence.append(ev)
                c_offset.append(frame_idx - anchor.frame_idx)
    return c_pool, c_anchor, c_level, c_frame, c_time, c_evidence, c_offset


def generate_answers(
    groups: Sequence[dict[str, Any]],
    params: AnswerGenParams | None = None,
    *,
    limit: int = MAX_ANSWERS,
    taken: Sequence[tuple[str, int]] = (),
) -> list[GeneratedAnswer]:
    """The ordered answer list. Position order IS the result — it decides R@k."""
    params = (params or DEFAULT_PARAMS).validated()
    return generate_from_pools(
        build_pools(groups, params), params, limit=limit, taken=taken
    )


def generate_from_pools(
    pools: Sequence[VideoPool],
    params: AnswerGenParams | None = None,
    *,
    limit: int = MAX_ANSWERS,
    taken: Sequence[tuple[str, int]] = (),
) -> list[GeneratedAnswer]:
    """`generate_answers` on an already-built pool, for callers that also want to
    report on the pool itself and should not pay to build it twice.

    `taken` is the answers that already occupy the head of the list — the ones a
    person picked by hand. They are not re-generated, and they are not ignored
    either: the list is scored as a whole, so a position already spent on an
    instant is coverage the generator must account for rather than duplicate,
    and the band schedule continues from where they end. The first answer after
    six hand-picked ones is rank 7 — deep enough that its band should already be
    diversifying — not rank 1, where the policy is to bet everything on the single
    strongest hypothesis.
    """
    params = (params or DEFAULT_PARAMS).validated()
    limit = max(0, min(int(limit), MAX_ANSWERS))
    if not pools or limit == 0:
        return []

    # Always size the candidate space for the full 100, never for `limit`. That
    # makes a shorter list an exact prefix of a longer one: asking for 20 answers
    # and asking for 100 agree on the first 20, which they would not if the
    # ladder depended on how many were requested.
    candidates = _flatten(pools, params)
    while len(candidates[0]) < MAX_ANSWERS:
        grown = _grow_ladder(params)
        if grown is None:
            break
        wider = _flatten(pools, grown)
        if len(wider[0]) <= len(candidates[0]):
            break  # the rung produced nothing new; wider ones will not either
        params, candidates = grown, wider
    c_pool, c_anchor, c_level, c_frame, c_time, c_evidence, c_offset = candidates
    n = len(c_pool)
    if not n:
        return []
    c_novelty = [1.0] * n
    c_alive = [True] * n

    # ---- fold the answers already in the list into the coverage state -----
    by_video = {pool.video_id: i for i, pool in enumerate(pools)}
    taken_count: dict[int, int] = {}
    for video_id, frame_idx in taken:
        p_i = by_video.get(video_id)
        if p_i is None:
            # A video nothing retrieved: it still spent a position, so it still
            # counts against the budget, but there is no candidate to suppress.
            continue
        taken_count[p_i] = taken_count.get(p_i, 0) + 1
        _suppress(
            p_i,
            int(frame_idx),
            int(frame_idx) / (pools[p_i].fps or FALLBACK_FPS),
            params,
            c_pool, c_frame, c_time, c_novelty, c_alive,
        )

    video_scores = [p.video_score for p in pools]
    # pi is constant inside a band, so it is computed five times, not 100.
    pi_by_band = [_softmax(video_scores, t) for t in params.temperatures]
    if params.taken_video_penalty and taken_count:
        # A per-video rescale, so it genuinely changes which video wins a
        # position; a uniform one would cancel out of the argmax.
        pi_by_band = [
            [
                weight / (1.0 + params.taken_video_penalty * taken_count.get(i, 0))
                for i, weight in enumerate(band)
            ]
            for band in pi_by_band
        ]

    # Frontier limits per (pool, band): anchors opened, and eps levels opened.
    anchors_open: list[list[int]] = []
    offsets_open: list[list[int]] = []
    for pool in pools:
        a_bonus = params.ambiguous_anchor_bonus if pool.ambiguous else 0
        o_penalty = params.ambiguous_offset_penalty if pool.ambiguous else 0
        anchors_open.append([min(len(pool.anchors), a + a_bonus) for a in params.band_anchors])
        offsets_open.append([max(0, o - o_penalty) for o in params.band_offsets])

    picked: list[GeneratedAnswer] = []
    last_band = -1
    # Positions the caller already spent; the bands are defined on the position
    # in the FINAL list, not on how many answers this call happens to produce.
    offset = len(taken)
    pi: list[float] = pi_by_band[band_index(min(MAX_ANSWERS, offset + 1))]

    for rank in range(offset + 1, min(MAX_ANSWERS, offset + limit) + 1):
        band = band_index(rank)
        if band != last_band:
            pi = pi_by_band[band]
            last_band = band

        # `gated` narrows the frontier to what this band is meant to consider.
        # Dropping it is the fallback, not the rule: if the band's own frontier is
        # empty this position would otherwise be left blank, and a blank rank is
        # strictly worse than a weak answer — R@k is a max, so a wrong answer at
        # rank 14 costs nothing while an empty rank 14 forfeits its chance.
        best = -1.0
        best_i = -1
        for gated in (True, False):
            for i in range(n):
                if not c_alive[i]:
                    continue
                p_i = c_pool[i]
                if gated and c_anchor[i] >= anchors_open[p_i][band]:
                    continue
                if gated and c_level[i] > offsets_open[p_i][band]:
                    continue
                u = pi[p_i] * c_evidence[i] * c_novelty[i]
                if u > best:
                    best = u
                    best_i = i
            if best_i >= 0:
                break

        if best_i < 0:
            break

        p_i = c_pool[best_i]
        pool = pools[p_i]
        anchor = pool.anchors[c_anchor[best_i]]
        picked.append(
            GeneratedAnswer(
                rank=rank,
                video_id=pool.video_id,
                frame_idx=c_frame[best_i],
                keyframe_id=anchor.keyframe_id if c_level[best_i] == 0 else None,
                pts_time=c_time[best_i],
                kind="anchor" if c_level[best_i] == 0 else "offset",
                anchor_index=c_anchor[best_i],
                offset_frames=c_offset[best_i],
                utility=best,
                video_weight=pi[p_i],
                evidence=c_evidence[best_i],
                novelty=c_novelty[best_i],
            )
        )
        c_alive[best_i] = False

        # Only the chosen video's candidates change; everything else keeps its
        # novelty, which is what makes the loop linear rather than quadratic.
        _suppress(
            p_i, c_frame[best_i], c_time[best_i], params,
            c_pool, c_frame, c_time, c_novelty, c_alive,
        )

    return picked


def _suppress(
    p_i: int,
    frame_idx: int,
    at_time: float,
    params: AnswerGenParams,
    c_pool: list[int],
    c_frame: list[int],
    c_time: list[float],
    c_novelty: list[float],
    c_alive: list[bool],
) -> None:
    """Mark one instant of one video as covered.

    Shared by the selection loop and by the pre-seeding of `taken`, on purpose:
    an answer a person put at rank 1 has to age the candidate pool exactly the
    way an answer the generator put there does, or the two halves of the list
    would be reasoning about different states.
    """
    sigma = params.novelty_sigma_s
    floor = params.novelty_floor
    use_min = params.novelty_mode == "min"
    gap = params.min_answer_gap_frames
    for i in range(len(c_pool)):
        if not c_alive[i] or c_pool[i] != p_i:
            continue
        if c_frame[i] == frame_idx or (gap and abs(c_frame[i] - frame_idx) < gap):
            c_alive[i] = False
            continue
        d = (c_time[i] - at_time) / sigma
        penalty = 1.0 - (1.0 - floor) * math.exp(-d * d)
        c_novelty[i] = min(c_novelty[i], penalty) if use_min else c_novelty[i] * penalty


@dataclass
class GeneratedSequence:
    """One TRAKE answer row: a video plus one frame per event, in event order."""

    rank: int
    video_id: str
    frames: list[int]
    keyframe_ids: list[str | None]
    pts_times: list[float | None]
    kind: Literal["sequence", "offset"]
    offset_frames: int
    utility: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "video_id": self.video_id,
            "frames": list(self.frames),
            "keyframe_ids": list(self.keyframe_ids),
            "pts_times": [None if t is None else round(t, 3) for t in self.pts_times],
            "kind": self.kind,
            "offset_frames": self.offset_frames,
            "utility": round(self.utility, 9),
        }


def complete_sequences(
    sequences: Sequence[dict[str, Any]], event_count: int
) -> list[dict[str, Any]]:
    """Only chains that answer EVERY event can become answers.

    A TRAKE row is `<video>,<frame_1>,...,<frame_N>` with N fixed by the
    statement, so a chain that located 3 of 4 moments is not a shorter answer —
    it is not a row at all. The assembler emits partial chains deliberately (they
    are worth looking at in the console) and ranks by confident coverage, so a
    3-of-4 chain routinely outranks a complete one: on the dev queries 45 of 50
    chains were partial, and taking them wrote up to 55 rows of the wrong width
    per query. The organiser's parser rejects those, and one rejected row blocks
    the whole submission export.

    A missing `frame_idx` disqualifies a chain for the same reason: it used to be
    coerced to frame 0, which is a real position in the CSV and always wrong.
    """
    if event_count <= 0:
        return []
    wanted = list(range(1, event_count + 1))
    out: list[dict[str, Any]] = []
    for seq in sequences:
        frames = seq.get("frames") or []
        if len(frames) != event_count:
            continue
        if sorted(int(f.get("event_index") or 0) for f in frames) != wanted:
            continue
        if any(f.get("frame_idx") is None for f in frames):
            continue
        out.append(seq)
    return out


def _strictly_increasing(frames: Sequence[int]) -> bool:
    """The organisers require the events in chronological order, and the export
    refuses a row that is not — so a candidate that is not one is not a candidate."""
    return all(b > a for a, b in zip(frames, frames[1:]))


def _flatten_sequences(
    base: Sequence[list[int]], params: AnswerGenParams
) -> list[tuple[int, tuple[int, ...], float, int, int]]:
    """Chain candidates: (sequence index, frames, evidence, ladder rung, shift)."""
    cand: list[tuple[int, tuple[int, ...], float, int, int]] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()

    def add(i: int, frames: tuple[int, ...], evidence: float, level: int, shift: int) -> None:
        # Shifting one event on its own, or clamping a whole-chain shift at 0, can
        # push a frame past its neighbour. Such a chain is out of chronological
        # order and the export rejects it, so it never becomes a candidate.
        if (i, frames) in seen or not _strictly_increasing(frames):
            return
        seen.add((i, frames))
        cand.append((i, frames, evidence, level, shift))

    for i, frames in enumerate(base):
        if not frames:
            continue
        add(i, tuple(frames), 1.0, 0, 0)
        for level, eps in enumerate(params.offsets, start=1):
            decay = params.offset_decay ** level
            for sign in (-1, 1):
                shift = sign * eps
                add(i, tuple(max(0, f + shift) for f in frames), decay, level, shift)
                # One event off on its own is the commoner failure than the whole
                # chain sliding, but it is also a narrower bet — hence the small
                # extra discount.
                for j in range(len(frames)):
                    one = list(frames)
                    one[j] = max(0, one[j] + shift)
                    add(i, tuple(one), decay * 0.9, level, shift)
    return cand


def generate_trake_answers(
    sequences: Sequence[dict[str, Any]],
    params: AnswerGenParams | None = None,
    *,
    limit: int = MAX_ANSWERS,
    event_count: int | None = None,
    taken: Sequence[tuple[str, tuple[int, ...]]] = (),
) -> list[GeneratedSequence]:
    """The same rank-budget idea, with a whole ordered chain as the candidate.

    TRAKE scores per event (`R = matched events / N`, and 0 outright on the wrong
    video), so the two failure modes are the same two: wrong video, or right video
    with events off in time. pi_B still allocates positions across videos; the eps
    ladder shifts events — the whole chain together (a constant clock offset) and
    one event at a time (a single mis-localised moment) — and novelty is measured
    on mean per-event displacement from the rows already chosen.

    The accepted window per event is narrow (the organisers say typically under 10
    frames), so the small end of the ladder is what carries this task.
    """
    params = (params or DEFAULT_PARAMS).validated()
    limit = max(0, min(int(limit), MAX_ANSWERS))
    # N comes from the statement wherever the caller knows it. Inferring it from
    # the chains themselves is the last resort and is deliberately the WIDEST one:
    # if every chain is partial, the widest is the only evidence of how many
    # events were asked for, and guessing lower would emit rows of a width the
    # organiser never asked for.
    if not event_count or event_count <= 0:
        event_count = max((len(s.get("frames") or []) for s in sequences), default=0)
    seqs = complete_sequences(sequences, event_count)[: params.max_videos]
    if not seqs or limit == 0:
        return []

    base: list[list[int]] = []
    meta: list[tuple[list[str | None], list[float | None]]] = []
    for seq in seqs:
        frames = sorted(seq.get("frames") or [], key=lambda f: int(f.get("event_index") or 0))
        base.append([int(f.get("frame_idx") or 0) for f in frames])
        meta.append(
            (
                [f.get("submit_keyframe_id") for f in frames],
                [f.get("pts_time") for f in frames],
            )
        )

    scores = [float(s.get("score") or 0.0) for s in seqs]
    top = max(scores) or 1.0
    norm = [s / top for s in scores]

    # Sized for the full 100 regardless of `limit`, for the same prefix-stability
    # reason as the frame generator.
    cand = _flatten_sequences(base, params)
    while len(cand) < MAX_ANSWERS:
        grown = _grow_ladder(params)
        if grown is None:
            break
        wider = _flatten_sequences(base, grown)
        if len(wider) <= len(cand):
            break
        params, cand = grown, wider

    alive = [True] * len(cand)
    novelty = [1.0] * len(cand)
    pi_by_band = [_softmax(norm, t) for t in params.temperatures]
    # Novelty for a chain is measured in frames, not seconds: TRAKE windows are
    # frame-tight and a sequence carries no single timestamp.
    sigma_frames = max(1.0, params.novelty_sigma_s * FALLBACK_FPS)

    # `_flatten_sequences` already made every (sequence, frames) pair unique, so
    # one pass per position needs no second dedup here.
    already = {(video, tuple(frames)) for video, frames in taken}
    if already:
        for i, (s_i, frames, _, _, _) in enumerate(cand):
            if (str(seqs[s_i].get("video_id") or ""), frames) in already:
                alive[i] = False

    picked: list[GeneratedSequence] = []
    last_band = -1
    offset = len(taken)  # see the frame generator: bands run on the final list
    pi = pi_by_band[band_index(min(MAX_ANSWERS, offset + 1))]
    for rank in range(offset + 1, min(MAX_ANSWERS, offset + limit) + 1):
        band = band_index(rank)
        if band != last_band:
            pi = pi_by_band[band]
            last_band = band
        best, best_i = -1.0, -1
        for gated in (True, False):  # see the frame generator: never leave a hole
            for i, ok in enumerate(alive):
                if not ok:
                    continue
                s_i, _, evidence, level, _ = cand[i]
                if gated and level > params.band_offsets[band]:
                    continue
                u = pi[s_i] * evidence * novelty[i]
                if u > best:
                    best, best_i = u, i
            if best_i >= 0:
                break
        if best_i < 0:
            break
        s_i, frames, _, level, shift = cand[best_i]
        alive[best_i] = False
        keyframe_ids, pts = meta[s_i]
        picked.append(
            GeneratedSequence(
                rank=rank,
                video_id=str(seqs[s_i].get("video_id") or ""),
                frames=list(frames),
                keyframe_ids=list(keyframe_ids) if level == 0 else [None] * len(frames),
                pts_times=list(pts) if level == 0 else [None] * len(frames),
                kind="sequence" if level == 0 else "offset",
                offset_frames=shift,
                utility=best,
            )
        )
        for i, ok in enumerate(alive):
            if not ok or cand[i][0] != s_i:
                continue
            other = cand[i][1]
            d = sum(abs(a - b) for a, b in zip(other, frames)) / max(1, len(frames))
            x = d / sigma_frames
            novelty[i] *= 1.0 - (1.0 - params.novelty_floor) * math.exp(-x * x)
    return picked


def generate_trake_rows(
    sequences: Sequence[dict[str, Any]],
    params: AnswerGenParams | None = None,
    *,
    limit: int = MAX_ANSWERS,
    event_count: int | None = None,
    taken: Sequence[tuple[str, tuple[int, ...]]] = (),
) -> list[dict[str, Any]]:
    return [
        s.to_dict()
        for s in generate_trake_answers(
            sequences, params, limit=limit, event_count=event_count, taken=taken
        )
    ]


def generate_answer_rows(
    groups: Sequence[dict[str, Any]],
    params: AnswerGenParams | None = None,
    *,
    limit: int = MAX_ANSWERS,
) -> list[dict[str, Any]]:
    """`generate_answers` as plain dicts, the shape the API returns."""
    return [a.to_dict() for a in generate_answers(groups, params, limit=limit)]


def retrieval_error_percentiles(
    errors: Iterable[int], percentiles: Sequence[float] = (25, 50, 75, 90, 97)
) -> list[int]:
    """The eps ladder, read off |retrieved frame - ground truth| on a dev set.

    Section 4.3 of the design note: the probe spacing has to come from the measured
    error distribution, because the width of the accepted window is not knowable in
    advance and a ladder that is too sparse leaves gaps no answer falls into.
    """
    values = sorted(int(e) for e in errors)
    if not values:
        return []
    out: list[int] = []
    for p in percentiles:
        pos = (len(values) - 1) * (p / 100.0)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        value = values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)
        out.append(int(round(value)))
    return out
