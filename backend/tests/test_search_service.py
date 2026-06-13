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
async def test_fill_prefers_temporally_feasible_frame(settings):
    from app.types import FusedFrame

    # E2 is missing in K01_V001 (covers E1@10s, E3@50s). In-video search yields a
    # higher-scoring frame AFTER E3 (90s, infeasible) and a lower-scoring one that
    # slots between E1 and E3 (30s). Order-aware fill must take the 30s frame.
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
    pts = [f.pts_time for f in ef[1]]
    assert 30.0 in pts
    assert 90.0 not in pts
    assert ef[1] and all(f.via_fill for f in ef[1])


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
