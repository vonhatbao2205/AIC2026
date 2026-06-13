from app.trake import (
    assemble_trake_sequences,
    snap_to_keyframe,
    validate_increasing_order,
)
from app.types import FusedFrame


def _frame(kf_id, video_id, n, pts, score, via_fill=False):
    return FusedFrame(
        submit_keyframe_id=kf_id,
        video_id=video_id,
        keyframe_n=n,
        pts_time=pts,
        score=score,
        via_fill=via_fill,
    )


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


def test_coverage_dominates_ranking_over_raw_score():
    # Video A covers both events (modest scores); video B covers only one (huge score).
    a1 = [_frame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.10)]
    a2 = [_frame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.10)]
    b1 = [_frame("K02/K02_V001/001", "K02_V001", 1, 10.0, 5.0)]  # only E1, very high
    seqs = assemble_trake_sequences([a1 + b1, a2])
    assert seqs[0].video_id == "K01_V001"  # coverage 2 beats a single high-score frame
    assert seqs[0].coverage == 2


def test_via_fill_excluded_from_confidence_metrics():
    # E2 came from pass-2 in-video fill (synthetic 0.02 score); it counts toward
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


def test_real_coverage_outranks_fabricated_fill_coverage():
    # FAKE video: 1 genuine event + 2 in-video fills -> coverage 3 but confident 1.
    fake_e1 = [_frame("K02/K02_V001/001", "K02_V001", 1, 10.0, 0.90)]
    fake_e2 = [_frame("K02/K02_V001/005", "K02_V001", 5, 20.0, 0.02, via_fill=True)]
    fake_e3 = [_frame("K02/K02_V001/010", "K02_V001", 10, 30.0, 0.02, via_fill=True)]
    # TRUE video: 2 genuine events, E3 genuinely absent -> coverage 2, confident 2.
    true_e1 = [_frame("K01/K01_V001/001", "K01_V001", 1, 5.0, 0.50)]
    true_e2 = [_frame("K01/K01_V001/004", "K01_V001", 4, 15.0, 0.50)]
    seqs = assemble_trake_sequences(
        [fake_e1 + true_e1, fake_e2 + true_e2, fake_e3], fallback_partial=True
    )
    # Old (coverage-first) ranking would float the fabricated 3/3 to the top;
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
