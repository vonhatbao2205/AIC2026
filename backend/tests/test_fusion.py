from app.fusion import DEFAULT_RRF_K, group_by_video, reciprocal_rank_fusion
from app.types import ChannelHit, Evidence


def _hit(channel, kf_id, video_id, n, rank, score=1.0, pts=None):
    return ChannelHit(
        channel=channel,
        submit_keyframe_id=kf_id,
        video_id=video_id,
        keyframe_n=n,
        score=score,
        rank=rank,
        pts_time=pts,
        evidence=Evidence(type=channel, score=score, text=f"{channel}-{kf_id}"),
    )


def test_rrf_combines_channels_and_ranks_shared_hit_first():
    hits = {
        "image_pe": [
            _hit("image_pe", "K01/K01_V001/001", "K01_V001", 1, 0, pts=0.0),
            _hit("image_pe", "K01/K01_V001/002", "K01_V001", 2, 1, pts=6.0),
        ],
        "ocr": [
            _hit("ocr", "K01/K01_V001/001", "K01_V001", 1, 0, pts=0.0),
        ],
    }
    fused = reciprocal_rank_fusion(hits, k=DEFAULT_RRF_K)
    assert fused[0].submit_keyframe_id == "K01/K01_V001/001"
    # shared frame has both channels and two evidence items
    assert set(fused[0].channels) == {"image_pe", "ocr"}
    assert len(fused[0].evidence) == 2
    expected = 1 / (DEFAULT_RRF_K + 1) + 1 / (DEFAULT_RRF_K + 1)
    assert abs(fused[0].score - expected) < 1e-9


def test_rrf_channel_weight_zero_skips_channel():
    hits = {
        "image_pe": [_hit("image_pe", "K01/K01_V001/001", "K01_V001", 1, 0)],
        "ocr": [_hit("ocr", "K01/K01_V001/099", "K01_V001", 99, 0)],
    }
    fused = reciprocal_rank_fusion(hits, weights={"ocr": 0.0})
    ids = {f.submit_keyframe_id for f in fused}
    assert "K01/K01_V001/099" not in ids


def test_rrf_drops_only_the_excluded_frame():
    hits = {
        "image_pe": [
            _hit("image_pe", "K01/K01_V001/001", "K01_V001", 1, 0),
            _hit("image_pe", "K01/K01_V001/002", "K01_V001", 2, 1),
            _hit("image_pe", "K02/K02_V001/001", "K02_V001", 1, 2),
        ]
    }
    fused = reciprocal_rank_fusion(hits, negative_frames={"K01/K01_V001/001"})
    ids = [f.submit_keyframe_id for f in fused]
    # Excluding a frame must not touch its siblings or its video.
    assert ids == ["K01/K01_V001/002", "K02/K02_V001/001"]


def _two_video_hits(weak_start_rank: int, weak_count: int = 59):
    return {
        "image_pe": [
            _hit("image_pe", f"K01/K01_V001/{n:03d}", "K01_V001", n, n - 1, pts=float(n))
            for n in range(1, 4)
        ]
        + [
            _hit(
                "image_pe",
                f"K03/K03_V003/{n:03d}",
                "K03_V003",
                n,
                n + weak_start_rank,
                pts=float(n),
            )
            for n in range(1, weak_count + 1)
        ]
    }


def test_video_priority_never_touches_individual_frame_scores():
    """The old additive boost lifted all 380 frames of a video above every rival's
    best frame. Priority must live on the aggregate only."""
    fused = reciprocal_rank_fusion(_two_video_hits(40))
    plain = {f.submit_keyframe_id: f.score for f in fused}

    groups = group_by_video(fused, prioritized_videos={"K03_V003"})

    assert {f.submit_keyframe_id: f.score for g in groups for f in g.frames} == plain


def test_video_priority_flips_a_near_tie_but_not_a_clear_winner():
    """`soft` means exactly this: the multiplier decides close calls, and a video
    holding genuinely stronger evidence still cannot be displaced."""
    near_tie = reciprocal_rank_fusion(_two_video_hits(40))
    assert group_by_video(near_tie)[0].video_id == "K01_V001"
    assert group_by_video(near_tie, prioritized_videos={"K03_V003"})[0].video_id == "K03_V003"

    clear_gap = reciprocal_rank_fusion(_two_video_hits(400))
    assert group_by_video(clear_gap)[0].video_id == "K01_V001"
    assert group_by_video(clear_gap, prioritized_videos={"K03_V003"})[0].video_id == "K01_V001"


def test_deprioritized_video_is_demoted_without_losing_its_frames():
    hits = {
        "image_pe": [
            _hit("image_pe", "K01/K01_V001/001", "K01_V001", 1, 0, pts=1.0),
            _hit("image_pe", "K02/K02_V001/001", "K02_V001", 1, 1, pts=1.0),
        ]
    }
    fused = reciprocal_rank_fusion(hits)
    groups = group_by_video(fused, deprioritized_videos={"K01_V001"})
    assert groups[0].video_id == "K02_V001"
    assert {g.video_id for g in groups} == {"K01_V001", "K02_V001"}


def test_group_by_video_scores_and_orders():
    hits = {
        "image_pe": [
            _hit("image_pe", "K01/K01_V001/001", "K01_V001", 1, 0, pts=1.0),
            _hit("image_pe", "K01/K01_V001/002", "K01_V001", 2, 1, pts=2.0),
            _hit("image_pe", "K02/K02_V005/010", "K02_V005", 10, 2, pts=5.0),
        ]
    }
    fused = reciprocal_rank_fusion(hits)
    groups = group_by_video(fused)
    assert groups[0].video_id == "K01_V001"
    assert groups[0].frame_count == 2
    assert groups[0].max_score >= groups[0].mean_top_score - 1e-9


def test_many_mediocre_frames_do_not_bury_strong_single_frame():
    """Regression: the old additive count_bonus let a video with many mediocre
    frames outrank a video holding the single best frame. The normalized formula
    must keep the strong-best-frame video on top."""
    from app.types import FusedFrame

    strong = [FusedFrame("X/X_V001/001", "X_V001", 1, 1.0, 0.020)]  # the clear best
    weak = [
        FusedFrame(f"Y/Y_V001/{n:03d}", "Y_V001", n, float(n * 5), 0.013)  # 0.65 of best
        for n in range(1, 31)  # 30 spread-out mediocre frames
    ]
    groups = group_by_video(strong + weak)
    assert groups[0].video_id == "X_V001"
    # video_score must be in a sane [0, ~1] range, not dominated by count
    assert 0.0 <= groups[0].video_score <= 1.0001


def test_raw_top_two_different_videos_both_near_top():
    """Top-1 and top-2 in different videos must both stay visible at the top."""
    from app.types import FusedFrame

    a = [FusedFrame("X/X_V001/001", "X_V001", 1, 0.0, 0.020)]
    b = [FusedFrame("Y/Y_V002/001", "Y_V002", 1, 0.0, 0.0195)]
    groups = group_by_video(a + b)
    top_two = {groups[0].video_id, groups[1].video_id}
    assert top_two == {"X_V001", "Y_V002"}


def test_ambiguous_flag_on_distant_clusters():
    hits = {
        "image_pe": [
            _hit("image_pe", "K01/K01_V001/001", "K01_V001", 1, 0, pts=1.0),
            _hit("image_pe", "K01/K01_V001/050", "K01_V001", 50, 1, pts=400.0),
        ]
    }
    fused = reciprocal_rank_fusion(hits)
    groups = group_by_video(fused, ambiguous_gap_seconds=30.0)
    assert groups[0].ambiguous is True


def test_not_ambiguous_when_close():
    hits = {
        "image_pe": [
            _hit("image_pe", "K01/K01_V001/001", "K01_V001", 1, 0, pts=1.0),
            _hit("image_pe", "K01/K01_V001/002", "K01_V001", 2, 1, pts=4.0),
        ]
    }
    fused = reciprocal_rank_fusion(hits)
    groups = group_by_video(fused, ambiguous_gap_seconds=30.0)
    assert groups[0].ambiguous is False
