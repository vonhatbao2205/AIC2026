import pytest

from app.services.search_service import SearchService
from app.services.trake_service import TrakeService


@pytest.mark.asyncio
async def test_search_groups_by_video_with_media_urls(settings):
    svc = SearchService(settings)
    res = await svc.search({"query": "bản tin thời sự thủ tướng", "query_type_hint": "T-KIS"})
    assert res["mode"] == "mock"
    assert res["groups"], "expected at least one video group"
    g = res["groups"][0]
    assert "video_score" in g and "ambiguous" in g
    frame = g["frames"][0]
    assert frame["image_id"] == frame["submit_keyframe_id"]
    assert frame["keyframe_url"].startswith("https://media.test/Keyframes/")
    assert frame["video_url"].startswith("https://media.test/Videos/")
    assert frame["pts_time"] is not None
    assert frame["channels"]
    # latency breakdown present
    assert "total_ms" in res["latency_ms"]


@pytest.mark.asyncio
async def test_search_never_emits_image_path(settings):
    svc = SearchService(settings)
    res = await svc.search({"query": "xe máy dưới mưa"})
    blob = str(res)
    assert "image_path" not in blob
    assert "/kaggle/" not in blob


@pytest.mark.asyncio
async def test_infoshotpp_profile_runs_text_channels_and_splits_media_origins(settings):
    settings.milvus_endpoint_2 = "https://milvus-2.test"
    settings.milvus_token_2 = "token-2"
    settings.keyframe_media_base_url_2 = "https://hf.test/resolve"
    settings.video_media_base_url_2 = "https://hf.test/resolve"
    svc = SearchService(settings.for_retrieval_database("infoshotpp"))
    parsed = {
        "channels": {
            "image_pe": {"enabled": True, "weight": 1.0, "queries_en": ["a street"]},
            "ocr": {"enabled": True, "weight": 1.0, "queries_vi": ["giá vàng"]},
            "speech": {"enabled": True, "weight": 1.0, "queries_vi": ["giá vàng"]},
            "audio": {"enabled": True, "weight": 1.0, "queries_en": ["music"]},
        },
        "filters": {},
        "rerank_policy": {"rrf_k": 60},
    }

    res = await svc.search({"query": "x", "parsed": parsed})

    assert res["retrieval_database"] == "infoshotpp"
    # Since the v2 metadata upload every Elastic channel is served for InfoShot++ too.
    assert set(res["latency_ms"]["channels"]) == {"image_pe", "ocr", "speech", "audio"}
    # ...but the operator is told which categories the v2 OCR index does not cover.
    assert any("L26" in warning for warning in res["warnings"])
    frame = res["groups"][0]["frames"][0]
    # Both media kinds come from the Hugging Face bucket on this profile; the
    # BTC profile keeps its own origins (see test_retrieval_profiles.py).
    assert frame["keyframe_url"].startswith("https://hf.test/resolve/Keyframes/")
    assert frame["video_url"].startswith("https://hf.test/resolve/Videos/")


@pytest.mark.asyncio
async def test_infoshotpp_profile_selects_the_v2_metadata_indices(settings):
    profile = settings.for_retrieval_database("infoshotpp")

    assert profile.idx_ocr == settings.idx_ocr_2
    assert profile.idx_speech == settings.idx_speech_2
    assert profile.idx_audio == settings.idx_audio_2
    # The V-KIS sketch searches InfoShot++'s own PE collection, so nothing is missing.
    assert profile.unsupported_channels == frozenset()
    # The BTC profile must keep pointing at the untouched v1 indices.
    btc = settings.for_retrieval_database("btc")
    assert (btc.idx_ocr, btc.idx_speech, btc.idx_audio) == (
        settings.idx_ocr_1,
        settings.idx_speech_1,
        settings.idx_audio_1,
    )
    assert btc.unsupported_channels == frozenset()


@pytest.mark.asyncio
async def test_ocr_runner_forwards_structured_numbers(settings, monkeypatch):
    svc = SearchService(settings)
    captured = {}

    async def fake_search_ocr(queries_vi, queries_folded, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(svc.elastic, "search_ocr", fake_search_ocr)
    await svc._run_ocr(
        {
            "queries_vi": ["THPT 2021"],
            "queries_folded": ["thpt 2021"],
            "numbers": ["2021"],
            "time_filters": {},
        },
        50,
    )

    assert captured["numbers"] == ["2021"]


@pytest.mark.asyncio
async def test_search_feedback_negative_frame_excluded(settings):
    svc = SearchService(settings)
    base = await svc.search({"query": "thời sự"})
    target = base["groups"][0]["frames"][0]["submit_keyframe_id"]
    res = await svc.search(
        {"query": "thời sự", "feedback": {"negative_frames": [target]}}
    )
    all_ids = {f["submit_keyframe_id"] for g in res["groups"] for f in g["frames"]}
    assert target not in all_ids


@pytest.mark.asyncio
async def test_image_query_expansion_dedups_variants(settings):
    """Multiple image_pe query variants are searched and fused by max score; the
    same frame must not appear twice (union, not concatenation)."""
    svc = SearchService(settings)
    parsed = {
        "channels": {
            "image_pe": {"enabled": True, "weight": 1.0, "queries_en": ["a man", "a person walking", "someone"]},
            "ocr": {"enabled": False},
            "speech": {"enabled": False},
            "audio": {"enabled": False},
        },
        "filters": {},
        "rerank_policy": {"rrf_k": 60},
    }
    res = await svc.search({"query": "x", "parsed": parsed})
    ids = [f["submit_keyframe_id"] for g in res["groups"] for f in g["frames"]]
    assert len(ids) == len(set(ids))  # no duplicate frames across fused variants
    assert ids  # still returns results


@pytest.mark.asyncio
async def test_trake_search_orders_events(settings):
    svc = SearchService(settings)
    trake = TrakeService(settings, svc)
    res = await trake.search_trake(
        {"query": "người nói chuyện sau đó có tiếng nhạc", "top_k": 50}
    )
    assert "sequences" in res
    assert "events" in res
    # any returned sequence must have non-decreasing pts_time
    for seq in res["sequences"]:
        times = [f["pts_time"] for f in seq["frames"]]
        assert times == sorted(times)


# ---- Pass-2 in-video fill: order-aware selection (TRAKE weakness A) ----


class _StubPe:
    async def encode_text(self, texts):
        return [[0.1, 0.2] for _ in texts]


class _StubElastic:
    def __init__(self, recs):
        self._recs = recs

    async def get_keyframes_by_ids(self, ids):
        return {i: self._recs[i] for i in ids if i in self._recs}


class _StubMilvus:
    """Global ref search (video_id=None) scores 1.0; in-video search returns `hits`."""

    def __init__(self, hits):
        self._hits = hits

    def search_image(self, vector, top_k=10, video_id=None):
        if video_id is None:
            return [{"submit_keyframe_id": "ref", "video_id": "X", "keyframe_n": 1, "score": 1.0}]
        return self._hits


def _wire(settings, hits, recs):
    trake = TrakeService(settings)
    trake.search.pe = _StubPe()
    trake.search.milvus = _StubMilvus(hits)
    trake.search.elastic = _StubElastic(recs)
    return trake


@pytest.mark.asyncio
async def test_fill_lets_the_assembly_dp_pick_the_orderable_frame(settings):
    from app.trake import build_trake_videos
    from app.types import FusedFrame

    # E2 is missing in K01_V001 (covers E1@10s, E3@50s). In-video search yields a
    # higher-scoring frame AFTER E3 (90s, unorderable) and a lower-scoring one
    # that slots between E1 and E3 (30s). Both are offered; the DP takes the 30s
    # frame because that is the one that completes the chain.
    hits = [
        {"submit_keyframe_id": "K01/K01_V001/090", "video_id": "K01_V001", "keyframe_n": 90, "score": 0.90},
        {"submit_keyframe_id": "K01/K01_V001/030", "video_id": "K01_V001", "keyframe_n": 30, "score": 0.70},
    ]
    recs = {
        "K01/K01_V001/090": {"pts_time": 90.0, "frame_idx": 2250, "fps": 25.0},
        "K01/K01_V001/030": {"pts_time": 30.0, "frame_idx": 750, "fps": 25.0},
    }
    trake = _wire(settings, hits, recs)
    events = [{"event_index": j + 1, "image_pe_queries_en": ["q"]} for j in range(3)]
    ef = [
        [FusedFrame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.9)],
        [],
        [FusedFrame("K01/K01_V001/050", "K01_V001", 50, 50.0, 0.9)],
    ]
    await trake._fill_missing_events(events, ef)
    assert all(f.via_fill for f in ef[1])
    chain = build_trake_videos(ef)[0].sequence
    assert [f.pts_time for f in chain.frames] == [10.0, 30.0, 50.0]


@pytest.mark.asyncio
async def test_fill_hands_the_dp_alternatives_it_can_actually_combine(settings):
    """Two gaps at once, where each gap's BEST hit is unorderable next to the
    other's.

    E2's strongest is at 80 s and E3's at 20 s: committing to the best of each,
    one gap at a time, produces 80 -> 20 and a dead chain. The second-best of
    each — 30 s and 70 s — completes E1@10 -> 30 -> 70 -> E4@100. Retrieval had
    already paid for both, so the choice belongs to the DP, which is the only
    thing here that can see the gaps together."""
    from app.trake import build_trake_videos
    from app.types import FusedFrame

    per_event_hits = {
        1: [
            {"submit_keyframe_id": "K01/K01_V001/080", "video_id": "K01_V001", "keyframe_n": 80, "score": 0.90},
            {"submit_keyframe_id": "K01/K01_V001/030", "video_id": "K01_V001", "keyframe_n": 30, "score": 0.70},
        ],
        2: [
            {"submit_keyframe_id": "K01/K01_V001/020", "video_id": "K01_V001", "keyframe_n": 20, "score": 0.90},
            {"submit_keyframe_id": "K01/K01_V001/070", "video_id": "K01_V001", "keyframe_n": 70, "score": 0.70},
        ],
    }
    recs = {
        "K01/K01_V001/080": {"pts_time": 80.0, "frame_idx": 2000, "fps": 25.0},
        "K01/K01_V001/030": {"pts_time": 30.0, "frame_idx": 750, "fps": 25.0},
        "K01/K01_V001/020": {"pts_time": 20.0, "frame_idx": 500, "fps": 25.0},
        "K01/K01_V001/070": {"pts_time": 70.0, "frame_idx": 1750, "fps": 25.0},
    }

    class _EventVectorEncoder:
        """One distinguishable vector per event, so the search stub can answer by
        EVENT rather than by call order — the gap searches run under
        `asyncio.gather`, whose completion order is not the submission order."""

        async def encode_text(self, texts):
            return [[float(index), 0.2] for index, _ in enumerate(texts)]

    class _PerGapMilvus(_StubMilvus):
        def search_image(self, vector, top_k=10, video_id=None):
            if video_id is None:
                return [{"submit_keyframe_id": "ref", "video_id": "X", "keyframe_n": 1, "score": 1.0}]
            return per_event_hits[int(vector[0])]

    trake = _wire(settings, [], recs)
    trake.search.pe = _EventVectorEncoder()
    trake.search.milvus = _PerGapMilvus([])
    events = [{"event_index": j + 1, "image_pe_queries_en": ["q"]} for j in range(4)]
    ef = [
        [FusedFrame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.9)],
        [],
        [],
        [FusedFrame("K01/K01_V001/100", "K01_V001", 100, 100.0, 0.9)],
    ]
    await trake._fill_missing_events(events, ef, gaps={"K01_V001": [1, 2]})

    chain = build_trake_videos(ef)[0].sequence
    assert chain.coverage == 4
    assert [f.pts_time for f in chain.frames] == [10.0, 30.0, 70.0, 100.0]


@pytest.mark.asyncio
async def test_fill_falls_back_when_no_feasible_frame(settings):
    from app.types import FusedFrame

    # Only an infeasible (after-E3) candidate exists -> fall back to it rather than
    # adding nothing (no regression vs old behaviour; the DP drops it later).
    hits = [
        {"submit_keyframe_id": "K01/K01_V001/090", "video_id": "K01_V001", "keyframe_n": 90, "score": 0.90},
    ]
    recs = {"K01/K01_V001/090": {"pts_time": 90.0, "frame_idx": 2250, "fps": 25.0}}
    trake = _wire(settings, hits, recs)
    events = [{"event_index": j + 1, "image_pe_queries_en": ["q"]} for j in range(3)]
    ef = [
        [FusedFrame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.9)],
        [],
        [FusedFrame("K01/K01_V001/050", "K01_V001", 50, 50.0, 0.9)],
    ]
    await trake._fill_missing_events(events, ef)
    assert [f.pts_time for f in ef[1]] == [90.0]


@pytest.mark.asyncio
async def test_pass2_rescues_an_event_whose_only_candidate_is_out_of_order(settings):
    """The whole point of making pass 2 DP-aware.

    Pass 1 finds E1 at 90 s and E2 at 10 s: every event has a candidate, so the
    old "gap = event with no candidate" rule saw a fully covered video and never
    went back in. The chain was still 1 of 2, because E2 cannot follow E1. Pass 2
    now re-searches E2 inside the video, finds it at 100 s, and the chain closes.
    """
    from app.trake import build_trake_videos, select_pass2_gaps
    from app.types import FusedFrame

    hits = [
        {"submit_keyframe_id": "K01/K01_V001/100", "video_id": "K01_V001", "keyframe_n": 100, "score": 0.80},
    ]
    recs = {"K01/K01_V001/100": {"pts_time": 100.0, "frame_idx": 2500, "fps": 25.0}}
    trake = _wire(settings, hits, recs)
    events = [{"event_index": j + 1, "image_pe_queries_en": ["q"]} for j in range(2)]
    ef = [
        [FusedFrame("K01/K01_V001/050", "K01_V001", 50, 90.0, 0.040)],
        [FusedFrame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.030)],
    ]

    before = build_trake_videos([list(ef[0]), list(ef[1])])[0]
    assert before.coverage == 1 and before.evidence_coverage == 2

    gaps = select_pass2_gaps([before], 2)
    assert gaps == {"K01_V001": [1]}
    await trake._fill_missing_events(events, ef, gaps=gaps)

    after = build_trake_videos(ef)[0]
    assert after.coverage == 2
    assert [f.pts_time for f in after.sequence.frames] == [90.0, 100.0]
    assert after.trake_video_score > before.trake_video_score


@pytest.mark.asyncio
async def test_pass2_does_not_re_add_an_unorderable_frame_pass1_already_had(settings):
    """An event that HAS candidates and simply could not be placed gets no
    consolation frame: another unorderable hit is exactly what the DP rejected,
    so spending a slot on it would only pad the candidate list."""
    from app.types import FusedFrame

    # The only in-video hit for E2 is the same early frame pass 1 already offered.
    hits = [
        {"submit_keyframe_id": "K01/K01_V001/001", "video_id": "K01_V001", "keyframe_n": 1, "score": 0.80},
    ]
    recs = {"K01/K01_V001/001": {"pts_time": 10.0, "frame_idx": 250, "fps": 25.0}}
    trake = _wire(settings, hits, recs)
    events = [{"event_index": j + 1, "image_pe_queries_en": ["q"]} for j in range(2)]
    ef = [
        [FusedFrame("K01/K01_V001/050", "K01_V001", 50, 90.0, 0.040)],
        [FusedFrame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.030)],
    ]
    await trake._fill_missing_events(events, ef, gaps={"K01_V001": [1]})
    assert len(ef[1]) == 1  # nothing added
    assert not any(f.via_fill for f in ef[1])


@pytest.mark.asyncio
async def test_trake_response_carries_video_centric_results(settings):
    """The console ranks VIDEOS, so the response leads with them; `sequences` is
    the same assembly flattened for the answer generator and the benchmarks."""
    svc = SearchService(settings)
    trake = TrakeService(settings, svc)
    res = await trake.search_trake(
        {"query": "người nói chuyện sau đó có tiếng nhạc", "top_k": 50}
    )
    videos = res["videos"]
    assert videos
    # One assembly, two views: same videos, same order.
    assert [v["video_id"] for v in videos] == [s["video_id"] for s in res["sequences"]]
    scores = [v["trake_video_score"] for v in videos]
    assert scores == sorted(scores, reverse=True)
    top = videos[0]
    assert 0.0 <= top["trake_video_score"] <= 1.0
    assert top["video_url"].endswith(f"{top['video_id']}.mp4")
    assert top["preliminary_rank"] is not None
    assert len(top["events"]) == len(res["events"])
    for event in top["events"]:
        assert event["peaks"], "every event must offer its peaks for the heatmap"
        strengths = [p["strength"] for p in event["peaks"]]
        assert all(0.0 <= s <= 1.0 for s in strengths)
        rep = event["representative"]
        assert rep is not None and rep["keyframe_url"].endswith(f"{rep['keyframe_n']:03d}.jpg")
        if event["in_chain"]:
            assert rep["selected_by_dp"] is True
    chain_times = [f["pts_time"] for f in top["best_chain"]]
    assert chain_times == sorted(chain_times)


@pytest.mark.asyncio
async def test_pass2_only_queries_the_videos_the_preliminary_score_chose(settings):
    from app.types import FusedFrame

    class _RecordingMilvus(_StubMilvus):
        def __init__(self):
            super().__init__([])
            self.searched: list[str] = []

        def search_image(self, vector, top_k=10, video_id=None):
            if video_id is not None:
                self.searched.append(video_id)
            return super().search_image(vector, top_k=top_k, video_id=video_id)

    trake = _wire(settings, [], {})
    milvus = _RecordingMilvus()
    trake.search.milvus = milvus
    events = [{"event_index": j + 1, "image_pe_queries_en": ["q"]} for j in range(3)]
    ef = [
        [
            FusedFrame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.9),
            FusedFrame("K02/K02_V001/001", "K02_V001", 1, 10.0, 0.9),
        ],
        [FusedFrame("K01/K01_V001/010", "K01_V001", 10, 50.0, 0.9)],
        [],
    ]
    await trake._fill_missing_events(events, ef, gaps={"K02_V001": [2]})
    # K01_V001 is the stronger, 2-of-3 video — but the plan said K02_V001, and
    # pass 2 does not get to second-guess it.
    assert set(milvus.searched) == {"K02_V001"}


@pytest.mark.asyncio
async def test_pass2_skips_a_target_pass1_left_completely_empty(settings):
    from app.types import FusedFrame

    class _RecordingMilvus(_StubMilvus):
        def __init__(self):
            super().__init__([])
            self.searched: list[str] = []

        def search_image(self, vector, top_k=10, video_id=None):
            if video_id is not None:
                self.searched.append(video_id)
            return super().search_image(vector, top_k=top_k, video_id=video_id)

    trake = _wire(settings, [], {})
    milvus = _RecordingMilvus()
    trake.search.milvus = milvus
    events = [{"event_index": j + 1, "image_pe_queries_en": ["q"]} for j in range(2)]
    ef = [[FusedFrame("K01/K01_V001/001", "K01_V001", 1, 10.0, 0.9)], []]
    await trake._fill_missing_events(events, ef, gaps={"K09_V999": [0, 1], "K01_V001": [1]})
    # Nothing in K09_V999 to order a fill against, so its queries are not spent.
    assert set(milvus.searched) == {"K01_V001"}


@pytest.mark.asyncio
async def test_marked_frames_add_a_similar_channel_seeded_by_their_embeddings(settings):
    """'More like this frame' is a retrieval channel, not a score bonus: it must
    appear in the latency breakdown and attribute frames like any other channel."""
    svc = SearchService(settings)
    res = await svc.search({
        "query": "bản tin thời sự",
        "feedback": {"positive_frames": ["K01/K01_V001/001"]},
    })

    assert "similar" in res["latency_ms"]["channels"]
    attributed = {c for g in res["groups"] for f in g["frames"] for c in f["channels"]}
    assert "similar" in attributed


@pytest.mark.asyncio
async def test_no_similar_channel_without_marked_frames(settings):
    svc = SearchService(settings)
    res = await svc.search({"query": "bản tin thời sự"})
    assert "similar" not in res["latency_ms"]["channels"]


@pytest.mark.asyncio
async def test_video_priority_does_not_alter_frame_scores(settings):
    """Regression for the old +0.15 per-frame boost, which was ~9x the maximum
    achievable RRF score and therefore pinned every frame of the marked video."""
    svc = SearchService(settings)
    plain = await svc.search({"query": "bản tin thời sự"})
    target = plain["groups"][0]["video_id"]
    boosted = await svc.search({
        "query": "bản tin thời sự",
        "feedback": {"positive_videos": [target]},
    })

    def frame_scores(res):
        return {f["submit_keyframe_id"]: f["score"] for g in res["groups"] for f in g["frames"]}

    assert frame_scores(boosted) == frame_scores(plain)
