"""Answer-list generator: the properties the submission CSV depends on."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.answer_gen import (
    BANDS,
    AnswerGenParams,
    anchor_budget,
    band_index,
    build_pools,
    generate_answers,
    generate_trake_rows,
    retrieval_error_percentiles,
)
from app.main import app

FPS = 25.0


def frame(video_id: str, keyframe_n: int, frame_idx: int, score: float) -> dict:
    category = video_id.split("_")[0]
    return {
        "submit_keyframe_id": f"{category}/{video_id}/{keyframe_n}",
        "image_id": f"{category}/{video_id}/{keyframe_n}",
        "video_id": video_id,
        "keyframe_n": keyframe_n,
        "frame_idx": frame_idx,
        "fps": FPS,
        "pts_time": frame_idx / FPS,
        "score": score,
        "channels": ["image_pe"],
    }


def group(video_id: str, video_score: float, frames: list[dict], *, ambiguous: bool = False) -> dict:
    return {
        "video_id": video_id,
        "video_score": video_score,
        "ambiguous": ambiguous,
        "channels": ["image_pe"],
        "frames": frames,
    }


def dense_group(video_id: str, video_score: float, start: int, *, n: int = 40, step: int = 12,
                ambiguous: bool = False) -> dict:
    """A video whose retrieved frames are a run of near-consecutive keyframes."""
    frames = [
        frame(video_id, i + 1, start + i * step, 0.02 - i * 0.0002) for i in range(n)
    ]
    return group(video_id, video_score, frames, ambiguous=ambiguous)


@pytest.fixture
def groups() -> list[dict]:
    """Three videos: a clear winner, a close rival, and a distant third."""
    return [
        dense_group("L26_V001", 1.0, 5000),
        dense_group("L26_V002", 0.85, 12000),
        dense_group("L30_V003", 0.40, 800),
    ]


# ---- band mapping ---------------------------------------------------------


def test_band_index_matches_the_scoring_cutoffs():
    assert [band_index(r) for r in (1, 2, 5, 6, 20, 21, 50, 51, 100)] == [0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert band_index(1000) == len(BANDS) - 1


# ---- core invariants of the produced list ---------------------------------


def test_produces_exactly_the_requested_count_without_repeats(groups):
    answers = generate_answers(groups, limit=100)
    assert len(answers) == 100
    keys = [(a.video_id, a.frame_idx) for a in answers]
    assert len(set(keys)) == len(keys)
    assert [a.rank for a in answers] == list(range(1, 101))


def test_is_deterministic(groups):
    first = [(a.video_id, a.frame_idx) for a in generate_answers(groups, limit=100)]
    second = [(a.video_id, a.frame_idx) for a in generate_answers(groups, limit=100)]
    assert first == second


def test_rank_one_is_the_strongest_anchor_of_the_strongest_video(groups):
    answers = generate_answers(groups, limit=5)
    assert answers[0].video_id == "L26_V001"
    assert answers[0].kind == "anchor"
    assert answers[0].frame_idx == 5000


def test_frames_are_non_negative_and_stay_near_the_retrieved_range(groups):
    params = AnswerGenParams(tail_margin_frames=100).validated()
    bounds = {
        g["video_id"]: (
            min(f["frame_idx"] for f in g["frames"]),
            max(f["frame_idx"] for f in g["frames"]),
        )
        for g in groups
    }
    for answer in generate_answers(groups, params, limit=100):
        lo, hi = bounds[answer.video_id]
        assert answer.frame_idx >= 0
        assert lo - 100 <= answer.frame_idx <= hi + 100


def test_empty_input_yields_no_answers():
    assert generate_answers([], limit=100) == []
    assert generate_answers([group("L26_V001", 1.0, [])], limit=100) == []


def test_frames_without_an_index_cannot_become_answers():
    """A frame with no `frame_idx` has nothing to write into the CSV."""
    frames = [frame("L26_V001", 1, 500, 0.02), frame("L26_V001", 2, 600, 0.01)]
    frames[1]["frame_idx"] = None
    answers = generate_answers([group("L26_V001", 1.0, frames)], limit=10)
    assert {a.frame_idx for a in answers if a.kind == "anchor"} == {500}


# ---- the three factors of the utility ------------------------------------


def test_the_list_spreads_across_videos_rather_than_draining_one(groups):
    """Once a video's regions are covered its novelty falls and a rival wins.

    The early positions still belong to the strongest video — but only a majority
    of them, not all: the frontier deliberately opens few anchors that early, so a
    top video with nothing new left to say yields rather than repeating itself.
    """
    answers = generate_answers(groups, limit=100)
    assert len({a.video_id for a in answers}) == 3
    early = [a.video_id for a in answers[:5]]
    assert early[0] == "L26_V001"
    assert early.count("L26_V001") > len(early) / 2


def test_a_flat_video_ranking_diversifies_earlier_than_a_peaked_one():
    peaked = [dense_group("L26_V001", 1.0, 5000), dense_group("L26_V002", 0.2, 12000)]
    flat = [dense_group("L26_V001", 1.0, 5000), dense_group("L26_V002", 0.98, 12000)]
    first_rival = lambda gs: next(  # noqa: E731
        (a.rank for a in generate_answers(gs, limit=100) if a.video_id == "L26_V002"), None
    )
    assert first_rival(flat) < first_rival(peaked)


def test_novelty_keeps_consecutive_answers_off_the_same_instant(groups):
    answers = generate_answers(groups, limit=20)
    same_video = [a for a in answers if a.video_id == answers[0].video_id]
    gaps = [
        abs(a.frame_idx - b.frame_idx) for a, b in zip(same_video, same_video[1:])
    ]
    assert all(gap > 0 for gap in gaps)


def test_min_answer_gap_is_enforced_when_asked_for(groups):
    params = AnswerGenParams(min_answer_gap_frames=40).validated()
    answers = generate_answers(groups, params, limit=60)
    by_video: dict[str, list[int]] = {}
    for answer in answers:
        by_video.setdefault(answer.video_id, []).append(answer.frame_idx)
    for frames in by_video.values():
        ordered = sorted(frames)
        assert all(b - a >= 40 for a, b in zip(ordered, ordered[1:]))


# ---- anchors, ambiguity and the frontier ---------------------------------


def test_temporal_dedup_collapses_one_shot_into_one_hypothesis():
    """40 keyframes of a single continuous run are one temporal hypothesis."""
    params = AnswerGenParams(dedup_radius_s=30.0).validated()
    pools = build_pools([dense_group("L26_V001", 1.0, 5000, n=40, step=12)], params)
    assert len(pools[0].anchors) == 1


def test_distant_clusters_stay_separate_hypotheses():
    frames = [frame("L26_V001", i + 1, 1000 + i * 10, 0.02 - i * 0.001) for i in range(5)]
    frames += [frame("L26_V001", 50 + i, 40000 + i * 10, 0.015 - i * 0.001) for i in range(5)]
    params = AnswerGenParams(dedup_radius_s=5.0).validated()
    pools = build_pools([group("L26_V001", 1.0, frames)], params)
    anchor_frames = sorted(a.frame_idx for a in pools[0].anchors)
    assert anchor_frames[0] < 2000 and anchor_frames[-1] > 39000


def first_rank(answers, predicate) -> int:
    """Rank of the first answer satisfying `predicate`, or a sentinel past the end."""
    return next((a.rank for a in answers if predicate(a)), 10_000)


def test_ambiguous_video_reaches_its_other_regions_sooner():
    """The flag says the uncertainty is WHERE in the video, not whether.

    It shifts the ORDER, not what is reachable: every candidate stays available
    either way, because a position left empty would score nothing at all. So the
    ambiguous video spends its early ranks walking to other regions while the
    plain one spends them probing around the region it already has.
    """
    params = AnswerGenParams(
        band_anchors=(1, 1, 2, 3, 4),
        band_offsets=(2, 2, 2, 2, 2),
        ambiguous_anchor_bonus=3,
        ambiguous_offset_penalty=2,
        dedup_radius_s=5.0,
    ).validated()
    frames = [
        frame("L26_V001", i + 1, 1000 + i * 1000, 0.02 - i * 0.001) for i in range(6)
    ]
    # A rival video has to be present, otherwise the band gate never binds: with
    # one video its frontier empties immediately and the no-empty-position
    # fallback exposes everything from rank 2 on.
    rival = dense_group("L26_V002", 0.7, 30000)
    plain = generate_answers([group("L26_V001", 1.0, frames), rival], params, limit=40)
    amb = generate_answers(
        [group("L26_V001", 1.0, frames, ambiguous=True), rival], params, limit=40
    )

    target = lambda answers, pred: first_rank(  # noqa: E731
        [a for a in answers if a.video_id == "L26_V001"], pred
    )
    # The fourth distinct hypothesis of the target arrives earlier when it is
    # ambiguous…
    assert target(amb, lambda a: a.anchor_index >= 3) < target(
        plain, lambda a: a.anchor_index >= 3
    )
    # …and all six of its regions are reached earlier overall, rather than the
    # positions between them going to +/-eps probes around region one.
    anchor_ranks = lambda answers: sum(  # noqa: E731
        a.rank for a in answers if a.video_id == "L26_V001" and a.kind == "anchor"
    )
    assert anchor_ranks(amb) < anchor_ranks(plain)


def test_anchor_budget_follows_the_bands_not_the_ceiling():
    params = AnswerGenParams(band_anchors=(1, 2, 3, 4, 6), ambiguous_anchor_bonus=2).validated()
    assert anchor_budget(params) == 8
    assert anchor_budget(AnswerGenParams(max_anchors_per_video=3).validated()) == 3


# ---- offsets --------------------------------------------------------------


def test_offsets_snap_onto_a_retrieved_keyframe_when_one_is_close():
    """Half the ground-truth frames ARE extracted keyframes, so an offset that can
    land on one should. 5031 is a real keyframe temporal NMS folded away; the
    +25 probe off the anchor at 5000 is what brings it back."""
    frames = [frame("L26_V001", 1, 5000, 0.02), frame("L26_V001", 2, 5031, 0.001)]
    base = dict(
        offsets=(25,), band_anchors=(1, 1, 1, 1, 1), band_offsets=(1, 1, 1, 1, 1),
        dedup_radius_s=30.0, snap_radius_frames=10,
    )
    snapped = generate_answers(
        [group("L26_V001", 1.0, frames)],
        AnswerGenParams(**base, snap_offsets=True).validated(),
        limit=3,
    )
    assert 5031 in {a.frame_idx for a in snapped}

    unsnapped = generate_answers(
        [group("L26_V001", 1.0, frames)],
        AnswerGenParams(**base, snap_offsets=False).validated(),
        limit=20,
    )
    # Without snapping the probe stays where arithmetic put it, and the real
    # keyframe 31 frames away is never named.
    assert 5031 not in {a.frame_idx for a in unsnapped}


def test_the_band_gate_delays_offsets_behind_anchors():
    """The gate orders the two candidate kinds; it does not forbid either."""
    late = AnswerGenParams(band_offsets=(0, 0, 1, 2, 3)).validated()
    early = AnswerGenParams(band_offsets=(3, 3, 3, 3, 3)).validated()
    pool = [dense_group("L26_V001", 1.0, 5000), dense_group("L26_V002", 0.9, 12000)]
    assert first_rank(generate_answers(pool, late, limit=100), lambda a: a.kind == "offset") > (
        first_rank(generate_answers(pool, early, limit=100), lambda a: a.kind == "offset")
    )


def test_epsilon_ladder_is_read_off_the_error_distribution():
    errors = [0, 0, 4, 5, 9, 13, 18, 25, 30, 45, 86, 279]
    ladder = retrieval_error_percentiles(errors, (25, 50, 75, 90))
    assert ladder == sorted(ladder)
    assert ladder[0] < ladder[-1]
    assert retrieval_error_percentiles([]) == []


# ---- parameters -----------------------------------------------------------


def test_params_from_dict_ignores_junk_and_clamps():
    params = AnswerGenParams.from_dict(
        {"pool_depth": -5, "temperatures": [0.0], "novelty_floor": 9.0, "nonsense": 1}
    )
    assert params.pool_depth == 1
    assert len(params.temperatures) == len(BANDS)
    assert all(t > 0 for t in params.temperatures)
    assert params.novelty_floor == 1.0
    assert not hasattr(params, "nonsense")


def test_pool_depth_bounds_the_hypotheses_globally(groups):
    shallow = build_pools(groups, AnswerGenParams(pool_depth=5).validated())
    deep = build_pools(groups, AnswerGenParams(pool_depth=400).validated())
    assert sum(len(p.anchors) for p in shallow) <= sum(len(p.anchors) for p in deep)
    assert len(shallow) <= len(deep)


# ---- TRAKE ----------------------------------------------------------------


def sequence(video_id: str, score: float, frames: list[int]) -> dict:
    return {
        "video_id": video_id,
        "score": score,
        "complete": True,
        "frames": [
            {
                "event_index": i + 1,
                "submit_keyframe_id": f"L26/{video_id}/{i + 1}",
                "keyframe_n": i + 1,
                "frame_idx": f,
                "pts_time": f / FPS,
                "score": score,
            }
            for i, f in enumerate(frames)
        ],
    }


def partial(video_id: str, score: float, frames: dict[int, int]) -> dict:
    """A chain the assembler could only fill for SOME events, keyed by event index."""
    return {
        "video_id": video_id,
        "score": score,
        "complete": False,
        "frames": [
            {
                "event_index": i,
                "submit_keyframe_id": f"L26/{video_id}/{i}",
                "keyframe_n": i,
                "frame_idx": f,
                "pts_time": f / FPS,
                "score": score,
            }
            for i, f in sorted(frames.items())
        ],
    }


def test_a_chain_missing_an_event_never_becomes_an_answer():
    """A TRAKE row is `<video>,<f1>..<fN>`; 3 of 4 moments is not a shorter answer,
    it is not a row. The assembler ranks by confident coverage, so partial chains
    routinely outrank complete ones and used to be written out as short rows."""
    seqs = [
        partial("L26_V900", 0.09, {1: 100, 2: 200, 3: 300}),   # highest score, 3 of 4
        partial("L26_V901", 0.08, {1: 100, 3: 300, 4: 400}),   # a gap in the middle
        sequence("L26_V194", 0.05, [4707, 5120, 5425, 5870]),  # the only usable chain
    ]
    rows = generate_trake_rows(seqs, limit=100, event_count=4)
    assert rows, "the complete chain still has to be answered"
    assert {r["video_id"] for r in rows} == {"L26_V194"}
    assert all(len(r["frames"]) == 4 for r in rows)


def test_no_complete_chain_means_no_answers_rather_than_broken_ones():
    """Zero rows is recoverable — the operator sees the question unanswered. One
    malformed row is not: the organiser's parser rejects the whole submission."""
    seqs = [partial("L26_V900", 0.09, {1: 100, 2: 200, 3: 300})]
    assert generate_trake_rows(seqs, limit=100, event_count=4) == []


def test_the_statement_decides_the_width_not_the_widest_chain():
    """With every chain partial, inferring N from the chains would happily emit
    3-frame rows for a 4-event question."""
    seqs = [partial("L26_V900", 0.09, {1: 100, 2: 200, 3: 300})]
    assert generate_trake_rows(seqs, limit=10, event_count=4) == []
    # Told it is a 3-event question, the same chain is a perfectly good answer.
    assert len(generate_trake_rows(seqs, limit=10, event_count=3)) == 10


def test_a_chain_with_no_frame_index_is_dropped_not_coerced_to_zero():
    seq = sequence("L26_V194", 0.05, [4707, 5120, 5425, 5870])
    seq["frames"][2]["frame_idx"] = None
    assert generate_trake_rows([seq], limit=10, event_count=4) == []


def test_every_row_stays_in_chronological_order():
    """Shifting one event on its own can push it past its neighbour. The export
    refuses a row that is not strictly increasing, so such a chain is not a
    candidate — before this, 13 of 100 rows on a dev query were out of order."""
    seqs = [sequence("L26_V194", 0.05, [4707, 4720, 4735, 4750])]  # tightly spaced
    rows = generate_trake_rows(seqs, limit=100, event_count=4)
    assert rows
    for row in rows:
        frames = row["frames"]
        assert len(frames) == 4
        assert all(b > a for a, b in zip(frames, frames[1:])), frames
        assert all(f >= 0 for f in frames)


def test_trake_rows_keep_the_event_count_and_lead_with_the_best_chain():
    seqs = [sequence("L26_V194", 0.06, [4707, 5120, 5425, 5870]),
            sequence("L26_V072", 0.04, [2466, 3133, 3429, 3800])]
    rows = generate_trake_rows(seqs, limit=100, event_count=4)
    assert len(rows) == 100
    assert all(len(r["frames"]) == 4 for r in rows)
    assert rows[0]["video_id"] == "L26_V194"
    assert rows[0]["frames"] == [4707, 5120, 5425, 5870]
    assert len({(r["video_id"], tuple(r["frames"])) for r in rows}) == len(rows)
    assert all(f >= 0 for r in rows for f in r["frames"])


def test_trake_without_sequences_yields_nothing():
    assert generate_trake_rows([], limit=100) == []


# ---- filling the list when the candidate space is thin --------------------


def test_a_thin_result_still_fills_every_position():
    """One video, four keyframes. `R@k` is a max, so an empty rank 90 forfeits a
    chance while a wrong one costs nothing — the ladder grows instead of stopping."""
    frames = [frame("L26_V001", i + 1, 1000 + i * 200, 0.02 - i * 0.001) for i in range(4)]
    answers = generate_answers([group("L26_V001", 1.0, frames)], limit=100)
    assert len(answers) == 100
    assert len({(a.video_id, a.frame_idx) for a in answers}) == 100


def test_a_two_chain_trake_result_still_fills_every_position():
    seqs = [sequence("L26_V194", 0.06, [4707, 5120, 5425, 5870]),
            sequence("L26_V072", 0.04, [2466, 3133, 3429, 3800])]
    rows = generate_trake_rows(seqs, limit=100, event_count=4)
    assert len(rows) == 100
    assert all(len(r["frames"]) == 4 for r in rows)


def test_growing_the_ladder_does_not_disturb_the_early_answers(groups):
    """The extension only ever adds tail candidates, so a rich result is
    unaffected and a thin one keeps the ranking it would have had."""
    frames = [frame("L26_V001", i + 1, 1000 + i * 200, 0.02 - i * 0.001) for i in range(4)]
    thin = [group("L26_V001", 1.0, frames)]
    short = generate_answers(thin, limit=20)
    long = generate_answers(thin, limit=100)
    assert [(a.video_id, a.frame_idx) for a in long[:20]] == [
        (a.video_id, a.frame_idx) for a in short
    ]
    assert len(generate_answers(groups, limit=100)) == 100


# ---- API ------------------------------------------------------------------


def test_generate_endpoint_returns_an_ordered_list():
    client = TestClient(app)
    response = client.post(
        "/api/answers/generate",
        json={"retrieval_database": "btc", "query": "thủ tướng", "query_type_hint": "T-KIS", "limit": 20},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query_type"] == "T-KIS"
    assert body["reused_results"] is False
    assert [a["rank"] for a in body["answers"]] == list(range(1, len(body["answers"]) + 1))
    assert all(len(a["frames"]) == 1 for a in body["answers"])
    assert "pool_depth" in body["params"]


def test_generate_endpoint_ranks_a_result_handed_to_it(groups):
    """Passing the on-screen result keeps the operator's scope/feedback work."""
    client = TestClient(app)
    response = client.post(
        "/api/answers/generate",
        json={"retrieval_database": "btc", "query": "x", "query_type_hint": "T-KIS",
              "limit": 10, "groups": groups},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reused_results"] is True
    assert body["answers"][0]["video_id"] == "L26_V001"
    assert body["answers"][0]["frames"] == [5000]


def test_generate_endpoint_writes_the_qa_answer_into_every_row():
    client = TestClient(app)
    response = client.post(
        "/api/answers/generate",
        json={"retrieval_database": "btc", "query": "mấy người?", "query_type_hint": "QA",
              "limit": 5, "answer_text": "5"},
    )
    assert response.status_code == 200
    assert all(a["answer"] == "5" for a in response.json()["answers"])


def test_generate_endpoint_says_why_a_trake_question_got_no_answers():
    """Silence would read as "the generator is broken". The operator has to know
    the chains were too short, because that is fixed by searching, not retrying."""
    client = TestClient(app)
    response = client.post(
        "/api/answers/generate",
        json={
            "retrieval_database": "btc", "query": "E1: a. E2: b.", "query_type_hint": "TRAKE",
            "limit": 10, "event_count": 4,
            "sequences": [partial("L26_V900", 0.09, {1: 100, 2: 200, 3: 300})],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answers"] == []
    assert body["diagnostics"] == {"n_sequences": 1, "n_complete": 0, "event_count": 4}
    assert any("4 sự kiện" in w for w in body["warnings"])


def test_generate_endpoint_honours_the_width_the_client_sends():
    client = TestClient(app)
    body = {
        "retrieval_database": "btc", "query": "x", "query_type_hint": "TRAKE", "limit": 5,
        "sequences": [
            partial("L26_V900", 0.09, {1: 100, 2: 200, 3: 300}),
            sequence("L26_V194", 0.05, [4707, 5120, 5425, 5870]),
        ],
    }
    four = client.post("/api/answers/generate", json={**body, "event_count": 4}).json()
    assert {a["video_id"] for a in four["answers"]} == {"L26_V194"}
    three = client.post("/api/answers/generate", json={**body, "event_count": 3}).json()
    assert {a["video_id"] for a in three["answers"]} == {"L26_V900"}


def test_generate_endpoint_rejects_more_than_a_hundred_answers():
    client = TestClient(app)
    response = client.post(
        "/api/answers/generate",
        json={"retrieval_database": "btc", "query": "x", "limit": 500},
    )
    assert response.status_code == 422
