"""TARA's temporal clips must join the existing frame retrieval by video."""
from __future__ import annotations

import pytest

from app.config import Settings
from app.scope import FrameFilter
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


@pytest.mark.asyncio
async def test_tara_keeps_to_the_cameras_the_query_named(monkeypatch) -> None:
    settings = Settings(
        retrieval_database="infoshotpp", mock_mode=True, tara_enabled=True,
        milvus_endpoint="http://mock", milvus_token="mock",
        tara_encoder_url="http://mock", tara_encoder_token="mock",
    )
    service = SearchService(settings)
    frames = FrameFilter(families=("N",), windows=(("N042-V002", ()),))
    pushed: list[FrameFilter] = []

    async def visual(*args, **kwargs):
        return {}, {}, []

    def search(vector, *, scale, top_k, categories, frames):
        # A camera outside the filter still comes back, as it would in mock mode.
        pushed.append(frames)
        return [clip("N042-V002", scale, .8, 6), clip("N007-V001", scale, .9, 6),
                clip("M01_V003", scale, .7, 6)]

    monkeypatch.setattr(service, "_run_visual", visual)
    monkeypatch.setattr(service.milvus, "search_tara_clips", search)
    parsed = {
        "query_type": "T-KIS", "translated_en_visual": "a car turns left",
        "channels": {"tara": {"enabled": True, "weight": 1.0, "queries_en": ["a car turns left"]}},
        "filters": {},
    }
    groups, _ = await service.retrieve(parsed, frames=frames)
    assert pushed == [frames] * 3
    assert {g.video_id for g in groups} == {"N042-V002", "M01_V003"}


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


def test_milvus_tara_serves_batch2_cameras_through_the_frame_filter() -> None:
    settings = Settings(
        retrieval_database="infoshotpp", milvus_endpoint="http://mock", milvus_token="mock",
    )
    adapter = MilvusClient(settings)
    frames = FrameFilter(families=("N",), windows=(("N042-V002", ((3, 9),)),))

    class FakeClient:
        def search(self, **kwargs):
            assert 'video_id like "N042-%"' in kwargs["filter"]
            assert 'video_id in ["N042-V002"]' in kwargs["filter"]
            assert "keyframe_n" not in kwargs["filter"]
            return [[{"distance": .7, "entity": clip("N042-V002", "event", .7, 6)}]]

    adapter._client = FakeClient()
    rows = adapter.search_tara_clips(
        [0.0] * 3584, scale="event", categories=("N042",), frames=frames,
    )
    assert rows[0]["video_id"] == "N042-V002"

    filters: list[str] = []

    class RecordingClient:
        def search(self, **kwargs):
            filters.append(kwargs["filter"])
            return [[]]

    adapter._client = RecordingClient()
    for video_id in ("L21_V001", "M10_V029", "N100-V004", "S01-V007"):
        adapter.search_tara_clips([0.0] * 3584, scale="event", video_id=video_id)
    assert [f.split('video_id == "')[1].split('"')[0] for f in filters] == [
        "L21_V001", "M10_V029", "N100-V004", "S01-V007",
    ]
    for bad in ("K01_V001", "N001_V001", 'S01-V001" or 1==1'):
        with pytest.raises(ValueError):
            adapter.search_tara_clips([0.0] * 3584, scale="event", video_id=bad)


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


@pytest.mark.asyncio
async def test_tara_encoder_needs_only_its_url(monkeypatch) -> None:
    from app.adapters.tara_encoder import MODEL_ID, MODEL_REVISION, SEMANTIC_FINGERPRINT, TaraEncoderClient

    settings = Settings(
        retrieval_database="infoshotpp", tara_enabled=True,
        milvus_endpoint="http://mock", milvus_token="mock",
        tara_encoder_url="https://tara.baoencoder.site", tara_encoder_token=None,
    )
    assert settings.has_tara_search
    client = TaraEncoderClient(settings)
    contract = {"model": MODEL_ID, "revision": MODEL_REVISION, "semantic_fingerprint": SEMANTIC_FINGERPRINT,
                "dim": 3584, "l2_normalized": True}
    sent: list[dict] = []

    class Response:
        def __init__(self, payload): self.payload = payload
        def raise_for_status(self): pass
        def json(self): return self.payload

    class Http:
        async def post(self, url, **kwargs):
            sent.append(kwargs["headers"])
            assert url == "https://tara.baoencoder.site/encode-text"
            return Response({**contract, "vectors": [[1.0] + [0.0] * 3583]})

    monkeypatch.setattr(client._http, "get", lambda: Http())
    vectors = await client.encode_text(["a car turns left"])
    assert len(vectors[0]) == 3584
    assert "Authorization" not in sent[0]
    no_url = TaraEncoderClient(Settings(retrieval_database="infoshotpp", tara_enabled=True))
    assert (await no_url.health())["reason"] == "TARA_ENCODER_URL missing"


@pytest.mark.asyncio
async def test_operator_can_search_with_tara_alone() -> None:
    """Unticking every keyframe model disables the visual channel; TARA must not
    bring PE back through the all-channels-off fallback."""
    from app.query_parser import QueryParser

    settings = Settings(
        retrieval_database="infoshotpp", mock_mode=True, tara_enabled=True,
        milvus_endpoint="http://mock", milvus_token="mock",
        tara_encoder_url="https://tara.baoencoder.site",
    )
    parsed = await QueryParser(settings).parse(
        "chiếc xe buýt màu xanh quẹo phải", "T-KIS",
        manual_overrides={"force_channels": ["tara"], "disable_channels": ["image_pe"]},
        translate=False,
    )
    channels = parsed["channels"]
    assert channels["image_pe"]["enabled"] is False
    assert channels["tara"]["enabled"] is True
    assert channels["tara"]["queries_en"]


class _NoPe:
    async def encode_text(self, texts):
        raise AssertionError("a TARA-only TRAKE must not encode with PE")


@pytest.mark.asyncio
async def test_trake_pass2_fills_a_missing_event_with_tara_alone() -> None:
    from app.services.trake_service import TrakeService
    from app.trake import build_trake_videos
    from app.types import FusedFrame

    settings = Settings(retrieval_database="infoshotpp", mock_mode=True)
    trake = TrakeService(settings)
    trake.search.pe = _NoPe()
    trake.search.qwen3_vl = _NoPe()

    class Milvus:
        def search_tara_clips(self, vector, *, scale, top_k, video_id=None):
            assert scale == "event"
            if video_id is None:  # the event's reference search
                return [clip("N042-V002", "event", .40, 0)]
            # Two clips land on the same keyframe; one sits after E3 and cannot be ordered.
            return [clip(video_id, "event", .36, 26), clip(video_id, "event", .30, 27),
                    clip(video_id, "event", .38, 86)]

    class Elastic:
        async def nearest_keyframes_by_time(self, positions):
            # One keyframe every 10 s: the clips at 30 s and 31 s share keyframe 030.
            return [{"submit_keyframe_id": f"N042/{video}/{int(t) // 10 * 10:03d}", "video_id": video,
                     "keyframe_n": int(t) // 10 * 10, "pts_time": float(int(t) // 10 * 10)}
                    for video, t in positions]

        async def get_keyframes_by_ids(self, ids):
            return {i: {"pts_time": float(i.rsplit("/", 1)[1]), "frame_idx": 1, "fps": 25.0} for i in ids}

    trake.search.milvus = Milvus()
    trake.search.elastic = Elastic()
    events = [{"event_index": j + 1, "image_pe_queries_en": ["q"]} for j in range(3)]
    ef = [
        [FusedFrame("N042/N042-V002/010", "N042-V002", 10, 10.0, 0.9, channels=["tara"])],
        [],
        [FusedFrame("N042/N042-V002/050", "N042-V002", 50, 50.0, 0.9, channels=["tara"])],
    ]
    await trake._fill_missing_events(
        events, ef, gaps={"N042-V002": [1]}, image_models=(),
        tara_vectors={0: [1.0], 1: [1.0], 2: [1.0]},
    )
    assert all(f.via_fill and f.channels == ["tara"] for f in ef[1])
    assert sorted(f.submit_keyframe_id for f in ef[1]) == ["N042/N042-V002/030", "N042/N042-V002/090"]
    chain = build_trake_videos(ef)[0].sequence
    assert [f.pts_time for f in chain.frames] == [10.0, 30.0, 50.0]
    assert chain.to_dict()["frames"][1]["channels"] == ["tara"]


def _trake_request(disable: list[str]) -> dict:
    return {
        "query": "xe buýt quẹo phải rồi dừng lại",
        "manual_overrides": {"force_channels": [], "disable_channels": disable},
        "parsed": {
            "channels": {"image_pe": {"enabled": True, "queries_en": ["bus"]},
                         "tara": {"enabled": True, "queries_en": ["bus"]}},
            "filters": {},
            "trake": {"events": [
                {"event_index": 1, "description_en_visual": "a bus turns right", "image_pe_queries_en": ["bus turns"]},
                {"event_index": 2, "description_en_visual": "the bus stops", "image_pe_queries_en": ["bus stops"]},
            ], "fallback_policy": "allow_partial"},
        },
    }


def _stub_trake(trake, monkeypatch) -> dict:
    from app.types import FusedFrame

    seen: dict = {"visual": [], "fill": None}

    async def retrieve(parsed, **kwargs):
        seen["visual"].append(parsed["channels"]["image_pe"]["enabled"])
        return [], {}

    async def tara(events, event_frames, categories, parsed, frames=None):
        for i in range(len(events)):
            event_frames[i].append(FusedFrame(
                f"N042/N042-V002/{(i + 1) * 10:03d}", "N042-V002", (i + 1) * 10,
                float((i + 1) * 10), 0.02, channels=["tara"],
            ))
        return {i: [1.0] for i in range(len(events))}

    async def fill(events, event_frames, **kwargs):
        seen["fill"] = kwargs

    monkeypatch.setattr(trake.search, "retrieve", retrieve)
    monkeypatch.setattr(trake, "_add_tara_event_candidates", tara)
    monkeypatch.setattr(trake, "_fill_missing_events", fill)
    return seen


@pytest.mark.asyncio
async def test_trake_runs_tara_only_when_the_keyframe_models_are_off(monkeypatch) -> None:
    from app.services.trake_service import TrakeService

    trake = TrakeService(Settings(
        retrieval_database="infoshotpp", mock_mode=True, tara_enabled=True,
        milvus_endpoint="http://mock", milvus_token="mock",
        tara_encoder_url="https://tara.baoencoder.site",
    ))
    seen = _stub_trake(trake, monkeypatch)
    res = await trake.search_trake(_trake_request(["image_pe"]))
    assert seen["visual"] == [False, False]
    assert seen["fill"]["image_models"] == () and seen["fill"]["tara_vectors"]
    assert res["tara_only"] is True and res["image_models"] == [] and res["pe_tokens"] is None
    assert res["sequences"][0]["frames"][0]["channels"] == ["tara"]


@pytest.mark.asyncio
async def test_trake_keeps_pe_when_tara_cannot_search(monkeypatch) -> None:
    from app.services.trake_service import TrakeService

    trake = TrakeService(Settings(retrieval_database="infoshotpp", mock_mode=True))
    seen = _stub_trake(trake, monkeypatch)
    res = await trake.search_trake(_trake_request(["image_pe"]))
    assert seen["visual"] == [True, True]
    assert seen["fill"]["image_models"] == ("pe",) and not seen["fill"]["tara_vectors"]
    assert res["tara_only"] is False
