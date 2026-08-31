from app.trake import (
    DEFAULT_PER_VIDEO_EVENT_CAP,
    assemble_trake_sequences,
    build_trake_videos,
    build_video_event_map,
    event_reference_scores,
    select_pass2_videos,
    snap_to_keyframe,
    temporal_diversify,
    trake_video_score,
    validate_increasing_order,
)
from app.types import FusedFrame


def _frame(kf_id, video_id, n, pts, score, via_fill=False, fill_quality=None):
    return FusedFrame(
        submit_keyframe_id=kf_id,
        video_id=video_id,
        keyframe_n=n,
        pts_time=pts,
        score=score,
        via_fill=via_fill,
        fill_quality=fill_quality,
    )


def _video(videos, video_id):
    return next(v for v in videos if v.video_id == video_id)


def test_assemble_complete_ordered_sequence():
    e1 = [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.9)]
    e2 = [_frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.8)]
    seqs = assemble_trake_sequences([e1, e2])
    assert seqs[0].complete is True
    assert [f.pts_time for f in seqs[0].frames] == [10.0, 50.0]


def test_assemble_rejects_non_orderable_when_no_fallback():
    # Event2 candidate is earlier than event1's only candidate -> cannot order.
    e1 = [_frame("K01/K01_V001/050", "K01_V001", 50, 90.0, 0.9)]
    e2 = [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.8)]
    seqs = assemble_trake_sequences([e1, e2], fallback_partial=False)
    # E1 fills at 90s, E2 has no candidate after 90s -> incomplete -> dropped.
    assert all(s.complete for s in seqs)


def test_assemble_partial_allowed_with_warning():
    e1 = [_frame("K01/K01_V001/050", "K01_V001", 50, 90.0, 0.9)]
    e2 = [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.8)]
    seqs = assemble_trake_sequences([e1, e2], fallback_partial=True)
    assert len(seqs) == 1
    assert seqs[0].complete is False
    assert seqs[0].warning is not None


def test_complete_sequences_rank_above_partial():
    # Video A complete, video B partial.
    a1 = [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.5)]
    a2 = [_frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.5)]
    b1 = [_frame("K02/K02_V001/050", "K02_V001", 50, 90.0, 0.99)]
    b2 = [_frame("K02/K02_V001/001", "K02_V001", 1, 10.0, 0.99)]
    seqs = assemble_trake_sequences(
        [a1 + b1, a2 + b2], fallback_partial=True
    )
    assert seqs[0].complete is True


def test_dp_finds_global_optimum_where_greedy_fails():
    """Greedy picks E1's highest-scoring frame (B@50s) and then can't order E2
    (C@30s < 50s) -> coverage 1. The DP instead picks A@10s for E1 so E2's C@30s
    fits -> full coverage."""
    e1 = [
        _frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.5),   # A (lower score, early)
        _frame("K01/K01_V001/050", "K01_V001", 50, 50.0, 0.9),  # B (higher score, late)
    ]
    e2 = [_frame("K01/K01_V001/030", "K01_V001", 30, 30.0, 0.9)]  # C @30s
    seqs = assemble_trake_sequences([e1, e2])
    assert seqs[0].complete is True
    assert seqs[0].coverage == 2
    assert [f.pts_time for f in seqs[0].frames] == [10.0, 30.0]  # A then C, time-ordered


def test_dp_completes_controlled_three_event_trace_that_greedy_blocks():
    """The paper's controlled TRAKE trace: local greedy choices leave E3 behind
    the selected time, whereas lexicographic DP takes earlier E1/E2 candidates
    and covers all three events."""
    e1 = [
        _frame("K01/K01_V001/010", "K01_V001", 10, 10.0, 0.80),
        _frame("K01/K01_V001/050", "K01_V001", 50, 50.0, 0.95),
    ]
    e2 = [
        _frame("K01/K01_V001/030", "K01_V001", 30, 30.0, 0.83),
        _frame("K01/K01_V001/070", "K01_V001", 70, 70.0, 0.93),
    ]
    e3 = [_frame("K01/K01_V001/060", "K01_V001", 60, 60.0, 0.90)]

    last_time = float("-inf")
    greedy = []
    for candidates in (e1, e2, e3):
        feasible = [candidate for candidate in candidates if candidate.pts_time > last_time]
        pick = max(feasible, key=lambda candidate: candidate.score, default=None)
        greedy.append(pick)
        if pick is not None:
            last_time = pick.pts_time
    assert [pick.pts_time if pick else None for pick in greedy] == [50.0, 70.0, None]

    sequence = assemble_trake_sequences([e1, e2, e3])[0]
    assert sequence.complete is True
    assert sequence.coverage == 3
    assert [frame.pts_time for frame in sequence.frames] == [10.0, 30.0, 60.0]


def test_coverage_dominates_ranking_over_raw_score():
    # Video A covers both events (modest scores); video B covers only one (huge score).
    a1 = [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.10)]
    a2 = [_frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.10)]
    b1 = [_frame("K02/K02_V001/001", "K02_V001", 1, 10.0, 5.0)]  # only E1, very high
    seqs = assemble_trake_sequences([a1 + b1, a2])
    assert seqs[0].video_id == "K01_V001"  # coverage 2 beats a single high-score frame
    assert seqs[0].coverage == 2


def test_via_fill_excluded_from_confidence_metrics():
    # E2 came from pass-2 in-video fill (small 0.02 score); it counts toward
    # coverage but not toward confidence, and must not drag mean/min relevance.
    e1 = [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.80)]
    e2 = [_frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.02, via_fill=True)]
    s = assemble_trake_sequences([e1, e2])[0]
    assert s.complete is True
    assert s.coverage == 2
    assert s.confident_coverage == 1
    assert s.filled_events == 1
    assert abs(s.mean_score - 0.80) < 1e-9  # real frame only, not the 0.02 fill
    assert abs(s.min_score - 0.80) < 1e-9
    d = s.to_dict()
    assert d["confident_coverage"] == 1 and d["filled_events"] == 1
    assert d["frames"][1]["via_fill"] is True


def test_real_coverage_outranks_fill_heavy_coverage():
    # Fill-heavy video: 1 genuine event + 2 in-video fills -> coverage 3 but confident 1.
    fake_e1 = [_frame("K02/K02_V001/001", "K02_V001", 1, 10.0, 0.90)]
    fake_e2 = [_frame("K02/K02_V001/005", "K02_V001", 5, 20.0, 0.02, via_fill=True)]
    fake_e3 = [_frame("K02/K02_V001/010", "K02_V001", 10, 30.0, 0.02, via_fill=True)]
    # TRUE video: 2 genuine events, E3 genuinely absent -> coverage 2, confident 2.
    true_e1 = [_frame("K01/K01_V001/001", "K01_V001", 1, 5.0, 0.50)]
    true_e2 = [_frame("K01/K01_V001/004", "K01_V001", 4, 15.0, 0.50)]
    seqs = assemble_trake_sequences(
        [fake_e1 + true_e1, fake_e2 + true_e2, fake_e3], fallback_partial=True
    )
    # Old total-coverage ranking would float the fill-heavy 3/3 to the top;
    # real-coverage-first puts the genuinely-supported 2/3 video first.
    assert seqs[0].video_id == "K01_V001"
    assert seqs[0].confident_coverage == 2
    assert seqs[0].filled_events == 0


def test_clean_full_coverage_still_tops_filled_full_coverage():
    # Two complete videos; the one with NO fills should rank first.
    clean = [
        [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.40)],
        [_frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.40)],
    ]
    filled = [
        [_frame("K02/K02_V001/001", "K02_V001", 1, 10.0, 0.99)],
        [_frame("K02/K02_V001/010", "K02_V001", 10, 50.0, 0.02, via_fill=True)],
    ]
    seqs = assemble_trake_sequences([clean[0] + filled[0], clean[1] + filled[1]])
    assert seqs[0].video_id == "K01_V001"  # confident 2 beats confident 1 + fill
    assert seqs[0].filled_events == 0



# ---- video-centric assembly: map, NMS, heat, score -------------------------


def test_video_event_map_groups_candidates_per_video_and_event():
    e1 = [
        _frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.30),
        _frame("K02/K02_V001/001", "K02_V001", 1, 11.0, 0.20),
    ]
    e2 = [_frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.25)]
    vmap = build_video_event_map([e1, e2])
    assert set(vmap) == {"K01_V001", "K02_V001"}
    assert [len(slot) for slot in vmap["K01_V001"]] == [1, 1]
    assert [len(slot) for slot in vmap["K02_V001"]] == [1, 0]
    # A frame with no timestamp cannot be ordered, so it is not a candidate.
    ptsless = _frame("K03/K03_V001/001", "K03_V001", 1, 10.0, 0.3)
    ptsless.pts_time = None
    assert "K03_V001" not in build_video_event_map([[ptsless], []])


def test_temporal_nms_keeps_a_distinct_alternate_moment():
    # One burst of 12 near-identical frames plus one genuinely different scene
    # ranked below all of them. Top-by-score alone would drop the alternative.
    burst = [
        _frame(f"K01/K01_V001/{i:03d}", "K01_V001", i, 31.0 + 0.2 * i, 0.040 - 0.001 * i)
        for i in range(12)
    ]
    alternative = _frame("K01/K01_V001/200", "K01_V001", 200, 78.3, 0.025)
    from app.trake import TrakeCandidateFrame

    def _cand(f):
        return TrakeCandidateFrame(
            event_index=1,
            submit_keyframe_id=f.submit_keyframe_id,
            keyframe_n=f.keyframe_n,
            pts_time=f.pts_time,
            score=f.score,
        )

    for_dp, peaks = temporal_diversify([_cand(f) for f in [*burst, alternative]])
    times = {round(p.pts_time, 1) for p in peaks}
    assert 78.3 in times  # the distinct moment survived
    assert len(for_dp) == DEFAULT_PER_VIDEO_EVENT_CAP
    assert 78.3 in {round(c.pts_time, 1) for c in for_dp}
    # Peaks are at least the minimum gap apart — no two describe one moment.
    ordered = sorted(p.pts_time for p in peaks)
    assert all(b - a >= 2.0 for a, b in zip(ordered, ordered[1:]))


def test_duplicate_burst_cannot_evict_a_later_distinct_peak_in_assembly():
    # E2's burst sits before E1's frame; only the late alternative can be ordered
    # after it. The old top-12-by-score cap dropped that alternative, which cost
    # the chain a whole event.
    e1 = [_frame("K01/K01_V001/100", "K01_V001", 100, 40.0, 0.050)]
    e2 = [
        _frame(f"K01/K01_V001/{i:03d}", "K01_V001", i, 10.0 + 0.2 * i, 0.040 - 0.001 * i)
        for i in range(14)
    ] + [_frame("K01/K01_V001/200", "K01_V001", 200, 78.3, 0.010)]
    videos = build_trake_videos([e1, e2])
    chain = _video(videos, "K01_V001").sequence
    assert chain.complete is True
    assert [f.pts_time for f in chain.frames] == [40.0, 78.3]


def test_heat_strength_is_normalized_per_event():
    # E1 and E2 live on completely different score scales. Each video candidate
    # is ~90% of its OWN event's global best, so both must read ~0.9.
    e1 = [
        _frame("K02/K02_V001/001", "K02_V001", 1, 5.0, 0.040),  # global best of E1
        _frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.036),
    ]
    e2 = [
        _frame("K02/K02_V001/010", "K02_V001", 10, 90.0, 0.022),  # global best of E2
        _frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.020),
    ]
    assert event_reference_scores([e1, e2]) == [0.040, 0.022]
    video = _video(build_trake_videos([e1, e2]), "K01_V001")
    s1 = video.events[0].representative.strength
    s2 = video.events[1].representative.strength
    assert abs(s1 - 0.90) < 1e-6
    assert abs(s2 - 0.909091) < 1e-5
    # Raw scores differ by ~2x; normalized strengths do not.
    assert abs(s1 - s2) < 0.02


def test_fill_strength_uses_its_own_quality_scale():
    # A fill's 0.02-scale score means nothing against the RRF reference; its
    # `fill_quality` (cosine ratio to the event's best hit) is the honest number.
    e1 = [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.040)]
    e2 = [_frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.016, via_fill=True, fill_quality=0.8)]
    video = _video(build_trake_videos([e1, e2]), "K01_V001")
    peak = video.events[1].representative
    assert peak.via_fill is True
    assert abs(peak.strength - 0.8) < 1e-9


def test_representative_is_the_dp_pick_when_the_chain_covers_the_event():
    # E1's highest-scoring frame is at 50s, but only the 10s one lets E2 order.
    e1 = [
        _frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.020),
        _frame("K01/K01_V001/050", "K01_V001", 50, 50.0, 0.040),
    ]
    e2 = [_frame("K01/K01_V001/030", "K01_V001", 30, 30.0, 0.040)]
    video = _video(build_trake_videos([e1, e2]), "K01_V001")
    rep = video.events[0].representative
    assert video.events[0].in_chain is True
    assert rep.pts_time == 10.0  # the DP's choice, not the strongest peak
    assert rep.selected_by_dp is True
    assert {p.pts_time for p in video.events[0].peaks} == {10.0, 50.0}  # both still offered


def test_uncovered_event_still_exposes_its_best_alternative():
    # E2 only fires BEFORE E1, so no chain can place it — but the operator still
    # needs to see the frame, badged as not part of the chain.
    e1 = [_frame("K01/K01_V001/050", "K01_V001", 50, 90.0, 0.040)]
    e2 = [
        _frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.030),
        _frame("K01/K01_V001/005", "K01_V001", 5, 20.0, 0.010),
    ]
    video = _video(build_trake_videos([e1, e2]), "K01_V001")
    assert video.sequence.complete is False
    evidence = video.events[1]
    assert evidence.in_chain is False
    assert evidence.covered is True  # a representative is still shown
    assert evidence.representative.pts_time == 10.0  # the strongest alternative
    assert evidence.representative.selected_by_dp is False


def test_video_score_never_trades_a_confident_event_for_quality():
    # 2 confident events at terrible quality vs 1 confident event at perfect quality.
    worse_coverage = trake_video_score(
        n_events=4, confident_coverage=1, coverage=4, mean_quality=1.0, min_quality=1.0
    )
    better_coverage = trake_video_score(
        n_events=4, confident_coverage=2, coverage=2, mean_quality=0.01, min_quality=0.0
    )
    assert better_coverage > worse_coverage


def test_video_score_prefers_more_total_coverage_at_equal_confidence():
    thin = trake_video_score(
        n_events=4, confident_coverage=3, coverage=3, mean_quality=1.0, min_quality=1.0
    )
    filled = trake_video_score(
        n_events=4, confident_coverage=3, coverage=4, mean_quality=0.05, min_quality=0.0
    )
    assert filled > thin


def test_video_score_penalizes_one_hopeless_event_in_the_chain():
    # Same mean-ish chains; the one with a collapsed weakest event scores lower.
    steady = trake_video_score(
        n_events=4, confident_coverage=4, coverage=4, mean_quality=0.9125, min_quality=0.88
    )
    shaky = trake_video_score(
        n_events=4, confident_coverage=4, coverage=4, mean_quality=0.79, min_quality=0.22
    )
    assert steady > shaky
    assert trake_video_score(n_events=0, confident_coverage=0, coverage=0,
                             mean_quality=1.0, min_quality=1.0) == 0.0


def test_video_score_ranking_matches_the_lexicographic_chain_ranking():
    videos = build_trake_videos(
        [
            [
                _frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.40),
                _frame("K02/K02_V001/001", "K02_V001", 1, 10.0, 0.99),
                _frame("K03/K03_V001/001", "K03_V001", 1, 10.0, 0.99),
            ],
            [
                _frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.40),
                _frame("K02/K02_V001/010", "K02_V001", 10, 50.0, 0.02, via_fill=True),
            ],
        ]
    )
    assert [v.video_id for v in videos] == ["K01_V001", "K02_V001", "K03_V001"]
    assert [
        (-v.confident_coverage, -v.coverage) for v in videos
    ] == sorted((-v.confident_coverage, -v.coverage) for v in videos)
    assert [v.trake_video_score for v in videos] == sorted(
        (v.trake_video_score for v in videos), reverse=True
    )


def test_min_event_gap_flags_a_chain_squeezed_into_one_moment():
    e1 = [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.04)]
    e2 = [_frame("K01/K01_V001/002", "K01_V001", 2, 10.2, 0.04)]
    e3 = [_frame("K01/K01_V001/003", "K01_V001", 3, 10.4, 0.04)]
    video = _video(build_trake_videos([e1, e2, e3]), "K01_V001")
    assert video.min_event_gap is not None
    assert abs(video.min_event_gap - 0.2) < 1e-6
    # Diagnostic only: it must not have changed the ranking or the chain.
    assert video.sequence.complete is True


def test_pass2_targets_prefer_the_video_that_is_one_event_short():
    # A is missing 1 event; B is missing 2 but every hit it has is stronger.
    a = [
        [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.030)],
        [_frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.030)],
        [],
    ]
    b = [
        [_frame("K02/K02_V001/001", "K02_V001", 1, 10.0, 0.040)],
        [],
        [],
    ]
    videos = build_trake_videos([a[i] + b[i] for i in range(3)])
    assert select_pass2_videos(videos, 3) == ["K01_V001", "K02_V001"]


def test_pass2_targets_break_a_tier_tie_on_preliminary_score():
    # Both videos are missing exactly E3; the stronger evidence goes first.
    weak = [
        [_frame("K02/K02_V001/001", "K02_V001", 1, 10.0, 0.005)],
        [_frame("K02/K02_V001/010", "K02_V001", 10, 50.0, 0.005)],
        [],
    ]
    strong = [
        [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.040)],
        [_frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.040)],
        [],
    ]
    videos = build_trake_videos([weak[i] + strong[i] for i in range(3)])
    assert select_pass2_videos(videos, 3) == ["K01_V001", "K02_V001"]
    # The old rule ranked on covered-event COUNT alone, which is a tie here.
    assert {v.evidence_coverage for v in videos} == {2}
    assert select_pass2_videos(videos, 3, budget=1) == ["K01_V001"]


def test_pass2_targets_exclude_videos_that_need_no_fill():
    complete = [
        [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.040)],
        [_frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.040)],
    ]
    videos = build_trake_videos(complete)
    assert select_pass2_videos(videos, 2) == []


def test_validate_increasing_order():
    assert validate_increasing_order([10.0, 20.0, 30.0]) == []
    assert validate_increasing_order([10.0, 5.0, 30.0]) == [2]
    assert validate_increasing_order([10.0, 10.0]) == [2]
    assert validate_increasing_order([10.0, None, 30.0]) == []


KEYFRAMES = [
    {"submit_keyframe_id": "K01/K01_V001/001", "keyframe_n": 1, "frame_idx": 0, "pts_time": 0.0},
    {"submit_keyframe_id": "K01/K01_V001/002", "keyframe_n": 2, "frame_idx": 150, "pts_time": 6.0},
    {"submit_keyframe_id": "K01/K01_V001/003", "keyframe_n": 3, "frame_idx": 300, "pts_time": 12.0},
]


def test_snap_picks_nearest_frame_idx():
    # raw time 5.0s @25fps -> frame 125, nearest keyframe frame_idx 150 (kf 2)
    res = snap_to_keyframe(5.0, 25.0, KEYFRAMES)
    assert res.submit_keyframe_id == "K01/K01_V001/002"
    assert res.raw_frame_idx == 125
    assert res.delta_frames == 25
    assert abs(res.delta_seconds - 1.0) < 1e-6
    assert res.far is False


def test_snap_far_flag():
    res = snap_to_keyframe(0.0, 25.0, KEYFRAMES)
    assert res.submit_keyframe_id == "K01/K01_V001/001"
    assert res.far is False
    # A raw time with no nearby keyframe -> far
    res2 = snap_to_keyframe(9.0, 25.0, KEYFRAMES)  # frame 225 -> nearest kf2(150) delta 3s
    assert res2.far is True


def test_snap_fallback_pts_time_when_no_frame_idx():
    kfs = [{"submit_keyframe_id": "K01/K01_V001/001", "keyframe_n": 1, "frame_idx": None, "pts_time": 0.0},
           {"submit_keyframe_id": "K01/K01_V001/002", "keyframe_n": 2, "frame_idx": None, "pts_time": 6.0}]
    res = snap_to_keyframe(5.0, 25.0, kfs)
    assert res.submit_keyframe_id == "K01/K01_V001/002"


def test_snap_empty_returns_none():
    assert snap_to_keyframe(5.0, 25.0, []) is None
