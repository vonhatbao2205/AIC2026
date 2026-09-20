"""TARA's temporal clips must join the existing frame retrieval by video."""
from __future__ import annotations

import pytest

from app.config import Settings
from app.adapters.elastic_client import ElasticClient
from app.adapters.milvus_client import MilvusClient
from app.services.search_service import SearchService
from app.tara_fusion import fuse_tara_scales, fuse_video_rankings
from app.types import ChannelHit, VideoGroup


def clip(video_id: str, scale: str, score: float, start: float) -> dict:
    return {
        "clip_id": f"{video_id}@{scale}@{start}", "video_id": video_id,
        "category": video_id.split("_")[0], "scale": scale,
        "start_time": start, "end_time": start + 8,
        "fps": 25.0, "center_frame_idx": int((start + 4) * 25),
        "score": score,
    }


def test_overlapping_clips_vote_once_per_scale() -> None:
    rows = fuse_tara_scales({
        "event": [clip("L26_V001", "event", .9, 0), clip("L26_V001", "event", .8, 4),
                  clip("L26_V002", "event", .7, 0)],
        "sequence": [clip("L26_V002", "sequence", .9, 0)],
    })
    assert rows[0]["video_id"] == "L26_V002"
    assert rows[1]["score"] == pytest.approx(1 / 61)
    assert rows[1]["best_clip"]["start_time"] == 0


def test_tara_only_video_is_visible_and_ranked() -> None:
    old = VideoGroup(
        "L26_V001", 1.0, 1.0, 1.0, 1, 0.0, False, ["image_pe"], []
    )
    hits = {"image_pe": [ChannelHit("image_pe", "L26/L26_V001/1", "L26_V001", 1, .8, 0)]}
    tara = fuse_tara_scales({"event": [clip("L26_V002", "event", .9, 12)]})
    groups = fuse_video_rankings([old], hits, tara)
    assert {g.video_id for g in groups} == {"L26_V001", "L26_V002"}
    only = next(g for g in groups if g.video_id == "L26_V002")
    assert only.best_clip["start_time"] == 12
    assert only.channels == ["tara"] and only.frames == []


@pytest.mark.asyncio
async def test_search_fuses_tara_with_existing_frame_channel(monkeypatch) -> None:
    settings = Settings(
        retrieval_database="infoshotpp", mock_mode=True, tara_enabled=True,
        milvus_endpoint="http://mock", milvus_token="mock",
        tara_encoder_url="http://mock", tara_encoder_token="mock",
    )
    service = SearchService(settings)

    async def visual(*args, **kwargs):
        return {
            "image_pe": [ChannelHit("image_pe", "L26/L26_V001/1", "L26_V001", 1, .8, 0)]
        }, {"image_pe": 1.0}, []

    monkeypatch.setattr(service, "_run_visual", visual)
    monkeypatch.setattr(
        service.milvus, "search_tara_clips",
        lambda vector, *, scale, top_k, categories: (
            [clip("L26_V002", "event", .9, 12)] if scale == "event" else []
        ),
    )
    parsed = {
        "query_type": "T-KIS", "translated_en_visual": "chef cuts shrimp then plates it",
        "channels": {
            "image_pe": {"enabled": True, "weight": 1.0, "queries_en": ["chef"]},
            "tara": {"enabled": True, "weight": 1.0, "queries_en": ["chef cuts shrimp then plates it"]},
        },
        "filters": {},
    }
    groups, latency = await service.retrieve(parsed, categories=("L26",))
    assert {g.video_id for g in groups} == {"L26_V001", "L26_V002"}
    assert next(g for g in groups if g.video_id == "L26_V002").best_clip["clip_id"]
    assert "tara" in latency["channels"]


def test_milvus_tara_filter_and_clip_metadata() -> None:
    settings = Settings(
        retrieval_database="infoshotpp", milvus_endpoint="http://mock", milvus_token="mock",
    )
    adapter = MilvusClient(settings)

    class FakeClient:
        def search(self, **kwargs):
            assert kwargs["collection_name"] == settings.milvus_tara_collection
            assert 'scale == "event"' in kwargs["filter"]
            assert 'video_id like "L26_%"' in kwargs["filter"]
            return [[{"distance": .76, "entity": clip("L26_V001", "event", .76, 12)}]]

    adapter._client = FakeClient()
    rows = adapter.search_tara_clips([0.0] * 3584, scale="event", categories=("L26",))
    assert rows[0]["start_time"] == 12
    assert rows[0]["center_frame_idx"] == 400


@pytest.mark.asyncio
async def test_tara_midpoint_maps_to_nearest_real_keyframe(monkeypatch) -> None:
    settings = Settings(elastic_endpoint="http://mock", elastic_api_key="mock")
    adapter = ElasticClient(settings)
    before = {"submit_keyframe_id": "L26/L26_V001/4", "video_id": "L26_V001", "pts_time": 12.0}
    after = {"submit_keyframe_id": "L26/L26_V001/5", "video_id": "L26_V001", "pts_time": 16.0}

    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"responses": [
                {"hits": {"hits": [{"_source": before}]}},
                {"hits": {"hits": [{"_source": after}]}},
            ]}

    class Http:
        async def post(self, url, **kwargs):
            assert url.endswith("/_msearch")
            assert kwargs["headers"]["Content-Type"] == "application/x-ndjson"
            assert kwargs["content"].count(b"\n") == 4
            return Response()

    monkeypatch.setattr(adapter._http, "get", lambda: Http())
    rows = await adapter.nearest_keyframes_by_time([("L26_V001", 15.0)])
    assert rows[0]["submit_keyframe_id"] == after["submit_keyframe_id"]
