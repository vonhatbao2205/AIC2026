"""TRAKE sequence assembly and BTC keyframe snapping.

TRAKE = locate an ordered set of events E1..En inside ONE video, where the chosen
frames must occur in strictly increasing time. The unit of a TRAKE answer is
therefore the VIDEO, not the keyframe, so the intermediate representation here is
video-centric: every event's ranked candidates are grouped per video into a
`TrakeVideoResult` that carries the event evidence (heat peaks), the best ordered
chain, and one coverage-dominant `trake_video_score`.

Per video and event the candidates first go through a temporal NMS
(`temporal_diversify`): twelve frames of the same two-second moment are one
hypothesis, and keeping them would evict the genuinely different moment later in
the video that the chain may actually need. The chain itself is then an exact
dynamic-programming strictly-increasing chain over the surviving candidates (by
event index AND time): it picks at most one frame per event, in order,
maximizing `(real evidence, events covered, normalized quality)` — the same
priority, term for term, that `trake_video_score` gives the video it belongs to.
This is globally optimal — unlike a greedy left-to-right pick it never gets
trapped by a locally-good early choice, and it can skip an event (partial
sequence) when no orderable candidate exists.

`snap_to_keyframe` powers the (legacy) pause-frame picker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import FusedFrame

# Two candidates closer together than this are the same moment, not two
# hypotheses. Used by the per-event temporal NMS below.
DEFAULT_MIN_PEAK_GAP_SECONDS = 2.0
# Distinct moments kept per (video, event). These are what the operator sees as
# the event's heat peaks and what the DP gets to choose an alternative from.
DEFAULT_PEAKS_PER_EVENT = 8
# Total candidates per (video, event) handed to the DP: the distinct peaks plus,
# if room is left, the next-best near-duplicates. A near-duplicate is worth
# keeping only as chain slack (0.3 s earlier can be the difference between an
# orderable chain and none), never at the price of a distinct peak.
DEFAULT_PER_VIDEO_EVENT_CAP = 12
# Pass-2 fill scores are put on a small 0.02 relevance scale; dividing by it
# recovers the fill quality when the field is missing.
#
# It does NOT keep fills below real evidence, and nothing may assume it does: a
# frame found by a single channel at rank 0 scores `1/(60+1) = 0.0164`, so the
# two ranges overlap. Real-vs-fill preference is enforced explicitly, by the DP
# objective and by `confident_coverage` in the ranking.
FILL_SCORE_SCALE = 0.02


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


@dataclass
class TrakeCandidateFrame:
    event_index: int
    submit_keyframe_id: str
    keyframe_n: int
    pts_time: float
    score: float
    frame_idx: int | None = None
    via_fill: bool = False  # came from pass-2 in-video fill (small 0.02-scale score)
    fill_quality: float | None = None  # fill strength on its own [0,1] scale
    # `score` normalized against this event's best hit anywhere (see
    # `candidate_strength`). Filled in before assembly and used as the DP's
    # quality term, so chain selection and video ranking optimise one thing.
    strength: float = 0.0


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


@dataclass
class TrakeHeatPeak:
    """One distinct moment where an event fires inside a video.

    `strength` is normalized PER EVENT (see `candidate_strength`) so E1's and
    E2's heat rows can be read against each other even when their raw score
    distributions are nothing alike."""

    event_index: int
    submit_keyframe_id: str
    keyframe_n: int
    pts_time: float
    score: float
    strength: float
    frame_idx: int | None = None
    via_fill: bool = False
    selected_by_dp: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "submit_keyframe_id": self.submit_keyframe_id,
            "keyframe_n": self.keyframe_n,
            "frame_idx": self.frame_idx,
            "pts_time": round(self.pts_time, 3),
            "score": round(self.score, 4),
            "strength": round(self.strength, 4),
            "via_fill": self.via_fill,
            "selected_by_dp": self.selected_by_dp,
        }


@dataclass
class TrakeEventEvidence:
    """What one video has to say about one event: its distinct peaks and the one
    frame to show the operator.

    The representative is the DP's pick whenever the chain covers the event —
    that is the frame the system claims belongs to the globally ordered sequence.
    When the chain cannot place the event, the strongest peak is shown anyway
    (`in_chain=False`): the operator still needs to see the best the video has
    before writing the video off."""

    event_index: int
    peaks: list[TrakeHeatPeak] = field(default_factory=list)
    representative: TrakeHeatPeak | None = None
    in_chain: bool = False

    @property
    def has_candidate(self) -> bool:
        """This video has SOMETHING to show for the event — which is not the same
        as the chain covering it. Only `in_chain` means that."""
        return self.representative is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "has_candidate": self.has_candidate,
            "in_chain": self.in_chain,
            "representative": self.representative.to_dict() if self.representative else None,
            "peaks": [p.to_dict() for p in self.peaks],
        }


@dataclass
class TrakeVideoResult:
    """One video as a TRAKE answer candidate: evidence per event, the best
    ordered chain through it, and the single score the UI sorts on."""

    video_id: str
    trake_video_score: float
    coverage: int
    confident_coverage: int
    filled_events: int
    # Events pass-1 retrieval found ANY candidate for in this video, orderable or
    # not. It is what a pass-2 fill can still change, so it — not the chain
    # coverage — is what tiers the pass-2 budget.
    evidence_coverage: int
    mean_quality: float
    min_quality: float
    chain_quality: float
    events: list[TrakeEventEvidence]
    sequence: TrakeSequence
    warning: str | None = None
    # Smallest gap between two consecutive chain frames. A chain squeezed into a
    # fraction of a second is usually one moment matched n times, not n events —
    # a diagnostic for the operator, deliberately NOT part of the ranking.
    min_event_gap: float | None = None
    # Length of the video in seconds, when known. The heat rows are drawn against
    # it; without it they would be drawn against the last peak, which silently
    # rescales "all four events happen in the first third" into "they span the
    # whole video".
    duration_s: float | None = None

    @property
    def unplaced_events(self) -> list[int]:
        """0-based indices of the events the chain could NOT place.

        This — not "the event has no candidate" — is what pass 2 has to go after.
        An event whose only candidates sit before the previous event is just as
        absent from the answer as one with no candidate at all, and it is the
        commoner failure: retrieval found the right action in the wrong part of
        the video."""
        return [e.event_index - 1 for e in self.events if not e.in_chain]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "trake_video_score": round(self.trake_video_score, 4),
            "coverage": self.coverage,
            "confident_coverage": self.confident_coverage,
            "filled_events": self.filled_events,
            "evidence_coverage": self.evidence_coverage,
            "chain_quality": round(self.chain_quality, 4),
            "mean_quality": round(self.mean_quality, 4),
            "min_quality": round(self.min_quality, 4),
            "complete": self.sequence.complete,
            "warning": self.warning,
            "min_event_gap": None if self.min_event_gap is None else round(self.min_event_gap, 3),
            "duration_s": None if self.duration_s is None else round(self.duration_s, 3),
            "events": [e.to_dict() for e in self.events],
            "best_chain": self.sequence.to_dict()["frames"],
        }


def _best_increasing_chain(
    per_event: list[list[TrakeCandidateFrame]], n_events: int
) -> list[TrakeCandidateFrame | None]:
    """Exact DP: lexicographic best chain of candidates with strictly increasing
    (event_index, pts_time).

    The state is compared as `(real evidence, coverage, quality)` — term for term
    the priority `trake_video_score` encodes, so the chain the DP hands over is
    the chain the ranking is about to reward.

    The order matters, not just the terms. `(coverage, real, ...)` looks close
    enough and is not: with E1/E4/E5 found for real and E2/E3 only fillable after
    E4, it prefers `E1 + fill E2 + fill E3 + E5` (coverage 4, real 2) over
    `E1 + E4 + E5` (coverage 3, real 3) — and the ranking then scores that video
    on `confident_coverage = 2`, marking it down for the trade the DP made on its
    behalf. Real evidence dominating fill coverage is the policy across videos;
    it has to be the policy inside one too.

    `real evidence` also has to be its own term rather than an emergent property
    of the scores. A pass-2 fill scores `0.02 x quality`, while a frame found by a
    single channel at rank 0 scores `1/(60+1) = 0.0164`: a strong fill genuinely
    outweighs real evidence on a raw sum.

    Quality is per-event normalized `strength`, not raw score, so an event whose
    scores happen to run high cannot dominate the sum for events where they run
    low. Returns one pick per event (None where that event isn't covered)."""
    nodes: list[TrakeCandidateFrame] = [c for cands in per_event for c in cands]
    if not nodes:
        return [None] * n_events
    # Topological order for the (event, time) DAG: ascending event then time.
    nodes.sort(key=lambda c: (c.event_index, c.pts_time))
    m = len(nodes)
    dp: list[tuple[int, int, float]] = [(0, 0, 0.0)] * m
    parent = [-1] * m
    best_i, best_val = -1, (-1, -1, float("-inf"))
    for i in range(m):
        ni = nodes[i]
        best_prev, best_j = (0, 0, 0.0), -1
        for j in range(i):
            nj = nodes[j]
            if nj.event_index < ni.event_index and nj.pts_time < ni.pts_time and dp[j] > best_prev:
                best_prev, best_j = dp[j], j
        dp[i] = (
            best_prev[0] + (0 if ni.via_fill else 1),
            best_prev[1] + 1,
            best_prev[2] + ni.strength,
        )
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


def build_video_event_map(
    event_frames: list[list[FusedFrame]],
) -> dict[str, list[list[TrakeCandidateFrame]]]:
    """The video-centric view of a TRAKE retrieval: `video_id -> per-event
    candidates`. This is the representation everything downstream reads —
    pass-2 targeting, the DP, the heatmap and the ranking — instead of each of
    them grouping the flat event lists by video again on its own.

    Frames with no `pts_time` are dropped: without a timestamp a frame cannot be
    ordered, so it can be neither a chain link nor a heat peak."""
    n_events = len(event_frames)
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
                    fill_quality=f.fill_quality,
                )
            )
    return videos


def event_reference_scores(event_frames: list[list[FusedFrame]]) -> list[float]:
    """`G_e`: the best pass-1 score each event reached ANYWHERE in the pool.

    Heat intensity is normalized against this rather than against a global best
    across events: each event is a different query with its own channel mix and
    its own difficulty, so a 0.020 hit can be event 2's near-perfect match while
    it would be a weak one for event 1."""
    return [max((f.score for f in frames if not f.via_fill), default=0.0) for frames in event_frames]


def candidate_strength(candidate: TrakeCandidateFrame, reference: float) -> float:
    """Heat intensity of one candidate in [0, 1], normalized per event."""
    if candidate.via_fill:
        # A fill's own quality is already a ratio against the best hit that event
        # has in the whole index. The RRF reference is a different scale, so
        # dividing by it would report a number that means nothing.
        quality = (
            candidate.fill_quality
            if candidate.fill_quality is not None
            else candidate.score / FILL_SCORE_SCALE
        )
        return _clip01(quality)
    if reference <= 0:
        return 0.0
    return _clip01(candidate.score / reference)


def temporal_diversify(
    candidates: list[TrakeCandidateFrame],
    *,
    min_gap_seconds: float = DEFAULT_MIN_PEAK_GAP_SECONDS,
    peak_cap: int = DEFAULT_PEAKS_PER_EVENT,
    total_cap: int = DEFAULT_PER_VIDEO_EVENT_CAP,
) -> tuple[list[TrakeCandidateFrame], list[TrakeCandidateFrame]]:
    """Split one event's candidates inside one video into `(for_dp, peaks)`.

    `peaks` are distinct moments, chosen by greedy temporal NMS: highest score
    first, then every candidate closer than `min_gap_seconds` to an already
    chosen one is suppressed as the same moment. Taking the top-N by score alone
    routinely spent the whole budget on one burst — twelve frames 0.2 s apart —
    and dropped the alternative scene later in the video that the chain needed.

    `for_dp` is those peaks plus, while budget is left, the best suppressed
    near-duplicates: a near-duplicate is worthless as a hypothesis but can still
    be the frame that makes a chain orderable (0.3 s earlier fits before the next
    event where the peak does not)."""
    ordered = sorted(candidates, key=lambda c: (-c.score, c.pts_time, c.submit_keyframe_id))
    peaks: list[TrakeCandidateFrame] = []
    suppressed: list[TrakeCandidateFrame] = []
    for c in ordered:
        if len(peaks) < peak_cap and all(
            abs(c.pts_time - p.pts_time) >= min_gap_seconds for p in peaks
        ):
            peaks.append(c)
        else:
            suppressed.append(c)
    for_dp = peaks + suppressed[: max(0, total_cap - len(peaks))]
    return for_dp, peaks


def trake_video_score(
    *,
    n_events: int,
    confident_coverage: int,
    coverage: int,
    mean_quality: float,
    min_quality: float,
) -> float:
    """The lexicographic TRAKE ranking `(confident coverage, coverage, quality)`
    encoded as one number in [0, 1] so the UI can sort and display it.

    With `B = n + 1`, one more confidently covered event is worth `B²` and one
    more covered event `B`, while chain quality contributes at most 1 — so no
    amount of cosine can buy a covered event, which is the whole difference
    between TRAKE ranking and T-KIS ranking. Quality itself is
    `0.70·mean + 0.30·min`: the minimum has to matter on its own, or a chain with
    three superb events and one hopeless one outranks four solid ones."""
    if n_events <= 0:
        return 0.0
    b = n_events + 1
    quality = 0.70 * _clip01(mean_quality) + 0.30 * _clip01(min_quality)
    raw = confident_coverage * b * b + coverage * b + quality
    return raw / (n_events * b * b + n_events * b + 1.0)


def _assemble_one_video(
    video_id: str,
    per_event: list[list[TrakeCandidateFrame]],
    references: list[float],
    *,
    per_video_event_cap: int,
    peaks_per_event: int,
    min_peak_gap_seconds: float,
) -> TrakeVideoResult | None:
    """Diversify, run the exact DP, and describe the result of one video."""
    n_events = len(per_event)
    # Normalize once, here: the DP, the heat peaks and the video score all read
    # the same number, so none of them can drift onto its own scale.
    for idx, candidates in enumerate(per_event):
        reference = references[idx] if idx < len(references) else 0.0
        for candidate in candidates:
            candidate.strength = candidate_strength(candidate, reference)

    for_dp: list[list[TrakeCandidateFrame]] = []
    peaks_per_slot: list[list[TrakeCandidateFrame]] = []
    for candidates in per_event:
        dp_cands, peaks = temporal_diversify(
            candidates,
            min_gap_seconds=min_peak_gap_seconds,
            peak_cap=peaks_per_event,
            total_cap=per_video_event_cap,
        )
        for_dp.append(dp_cands)
        peaks_per_slot.append(peaks)

    picks = _best_increasing_chain(for_dp, n_events)
    chain = [p for p in picks if p is not None]
    if not chain:
        return None

    coverage = len(chain)
    complete = coverage == n_events
    warning = None
    if not complete:
        missing = [str(i + 1) for i, p in enumerate(picks) if p is None]
        warning = f"Partial sequence: events {', '.join(missing)} have no orderable candidate."
    # Quality is measured over real (pass-1) frames; the synthetic 0.02-scale
    # fill scores would otherwise drag mean/min down and misreport confidence.
    real = [p for p in chain if not p.via_fill]
    quality_src = real or chain

    events: list[TrakeEventEvidence] = []
    for idx in range(n_events):
        pick = picks[idx]
        heat = [
            TrakeHeatPeak(
                event_index=idx + 1,
                submit_keyframe_id=c.submit_keyframe_id,
                keyframe_n=c.keyframe_n,
                pts_time=c.pts_time,
                score=c.score,
                strength=c.strength,
                frame_idx=c.frame_idx,
                via_fill=c.via_fill,
                selected_by_dp=pick is not None and c.submit_keyframe_id == pick.submit_keyframe_id,
            )
            for c in peaks_per_slot[idx]
        ]
        # The DP can settle on a near-duplicate that NMS suppressed; it is still
        # the chain's frame, so it belongs on the heat row.
        if pick is not None and not any(p.selected_by_dp for p in heat):
            heat.append(
                TrakeHeatPeak(
                    event_index=idx + 1,
                    submit_keyframe_id=pick.submit_keyframe_id,
                    keyframe_n=pick.keyframe_n,
                    pts_time=pick.pts_time,
                    score=pick.score,
                    strength=pick.strength,
                    frame_idx=pick.frame_idx,
                    via_fill=pick.via_fill,
                    selected_by_dp=True,
                )
            )
        heat.sort(key=lambda p: p.pts_time)
        representative = next((p for p in heat if p.selected_by_dp), None)
        in_chain = representative is not None
        if representative is None and heat:
            representative = max(heat, key=lambda p: (p.strength, -p.pts_time))
        events.append(
            TrakeEventEvidence(
                event_index=idx + 1,
                peaks=heat,
                representative=representative,
                in_chain=in_chain,
            )
        )

    # Chain quality covers EVERY link, fills included — unlike the sequence's
    # mean/min relevance, which is a statement about real evidence only.
    # `confident_coverage` already dominates the ranking, so excluding fills here
    # too just threw away the one signal that separates two videos with identical
    # coverage: a chain closed by a convincing fill is worth more than the same
    # chain closed by one that barely cleared the acceptance floor.
    strengths = [p.strength for p in chain]
    mean_quality = sum(strengths) / len(strengths) if strengths else 0.0
    min_quality = min(strengths) if strengths else 0.0
    chain_times = sorted(p.pts_time for p in chain)
    min_event_gap = (
        min((b - a) for a, b in zip(chain_times, chain_times[1:])) if len(chain_times) > 1 else None
    )
    sequence = TrakeSequence(
        video_id=video_id,
        score=sum(p.score for p in chain),
        complete=complete,
        frames=chain,
        coverage=coverage,
        warning=warning,
        confident_coverage=len(real),
        filled_events=coverage - len(real),
        mean_score=sum(p.score for p in quality_src) / len(quality_src),
        min_score=min(p.score for p in quality_src),
    )
    return TrakeVideoResult(
        video_id=video_id,
        trake_video_score=trake_video_score(
            n_events=n_events,
            confident_coverage=len(real),
            coverage=coverage,
            mean_quality=mean_quality,
            min_quality=min_quality,
        ),
        coverage=coverage,
        confident_coverage=len(real),
        filled_events=coverage - len(real),
        evidence_coverage=sum(
            1 for candidates in per_event if any(not c.via_fill for c in candidates)
        ),
        mean_quality=mean_quality,
        min_quality=min_quality,
        chain_quality=0.70 * mean_quality + 0.30 * min_quality,
        events=events,
        sequence=sequence,
        warning=warning,
        min_event_gap=min_event_gap,
    )


def build_trake_videos(
    event_frames: list[list[FusedFrame]],
    *,
    fallback_partial: bool = True,
    max_results: int | None = 50,
    per_video_event_cap: int = DEFAULT_PER_VIDEO_EVENT_CAP,
    peaks_per_event: int = DEFAULT_PEAKS_PER_EVENT,
    min_peak_gap_seconds: float = DEFAULT_MIN_PEAK_GAP_SECONDS,
) -> list[TrakeVideoResult]:
    """Every candidate video of a TRAKE query, ranked by `trake_video_score`.

    `event_frames[i]` is the ranked frame list for E{i+1}. Ranking is
    coverage-dominant by construction of the score, so this order is exactly the
    old lexicographic `(confident coverage, coverage, quality)` — one number the
    console can also print."""
    n_events = len(event_frames)
    if n_events == 0:
        return []
    references = event_reference_scores(event_frames)
    videos = build_video_event_map(event_frames)

    results: list[TrakeVideoResult] = []
    for video_id, per_event in videos.items():
        result = _assemble_one_video(
            video_id,
            per_event,
            references,
            per_video_event_cap=per_video_event_cap,
            peaks_per_event=peaks_per_event,
            min_peak_gap_seconds=min_peak_gap_seconds,
        )
        if result is None:
            continue
        if not result.sequence.complete and not fallback_partial:
            continue
        results.append(result)

    results.sort(key=lambda v: (-v.trake_video_score, v.video_id))
    return results if max_results is None else results[:max_results]


def select_pass2_gaps(
    videos: list[TrakeVideoResult], n_events: int, *, budget: int = 15
) -> dict[str, list[int]]:
    """What pass 2 should search for, per video: `video_id -> missing event indices`.

    Two decisions live here, and both used to be made on the wrong quantity.

    WHICH EVENTS. A gap is an event the preliminary DP could not place
    (`unplaced_events`), not one with no candidate at all. A video whose E1 only
    fires at 90 s and whose E2 only fires at 10 s has a candidate for every event
    and still answers nothing — the chain is 1 of 2. That is the commoner failure
    (retrieval found the right action in the wrong part of the video), and asking
    "does this event have a candidate?" made pass 2 blind to it: the video looked
    fully covered, so it was never re-searched.

    WHICH VIDEOS. Tier by how many events the CHAIN is still missing — one short
    is a single query away from a full chain, so it goes first — and inside a
    tier let the preliminary `trake_video_score` decide. The old rule ranked on
    the count of events with a candidate, which treats "covers E1-E3 barely" and
    "covers E1-E3 convincingly" as the same bet. Videos missing many events fall
    to the end and therefore run only while the budget is not yet spent.

    Returned in priority order, so a caller that truncates keeps the best."""
    eligible = [v for v in videos if v.coverage < n_events]
    eligible.sort(key=lambda v: (n_events - v.coverage, -v.trake_video_score, v.video_id))
    return {v.video_id: v.unplaced_events for v in eligible[:budget]}


def assemble_trake_sequences(
    event_frames: list[list[FusedFrame]],
    *,
    fallback_partial: bool = True,
    max_results: int = 50,
    per_video_event_cap: int = DEFAULT_PER_VIDEO_EVENT_CAP,
) -> list[TrakeSequence]:
    """The best ordered sequence per video, ranked exactly as `build_trake_videos`
    ranks the videos they were taken from.

    Kept as the flat view of the same assembly: the answer generator and the
    benchmarks consume chains, while the console consumes whole videos."""
    return [
        v.sequence
        for v in build_trake_videos(
            event_frames,
            fallback_partial=fallback_partial,
            max_results=max_results,
            per_video_event_cap=per_video_event_cap,
        )
    ]


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
