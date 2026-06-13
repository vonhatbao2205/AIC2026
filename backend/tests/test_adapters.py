"""Mocked adapter tests — exercise the mock-mode paths (no live services)."""
import pytest

from app.adapters.elastic_client import ElasticClient
from app.adapters.milvus_client import MilvusClient
from app.adapters.pe_encoder import GLAP_DIM, PE_DIM, GlapEncoderClient, PeEncoderClient


@pytest.mark.asyncio
async def test_elastic_mock_ocr_search(settings):
    client = ElasticClient(settings)
    assert client.mock is True
    hits = await client.search_ocr(["thời sự"], [])
    assert any("K01_V001" in h["video_id"] for h in hits)
    for h in hits:
        assert "image_path" not in h  # never leak path
        assert h["submit_keyframe_id"].count("/") == 2


@pytest.mark.asyncio
async def test_elastic_empty_query_returns_nothing(settings):
    """Empty query must never fall through to match_all (the 'same results for
    every query' bug)."""
    client = ElasticClient(settings)
    assert await client.search_ocr([], []) == []
    assert await client.search_speech([]) == []
    assert await client.search_audio([], []) == []


@pytest.mark.asyncio
async def test_elastic_mock_speech_demotes_low(settings):
    client = ElasticClient(settings)
    hits = await client.search_speech(["thủ tướng nhật bản"])
    assert hits
    # results sorted by demoted score descending
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_elastic_mock_audio_drops_stoplisted(settings):
    client = ElasticClient(settings)
    hits = await client.search_audio(["music"], ["Music"])
    for h in hits:
        assert h["score"] > 0


@pytest.mark.asyncio
async def test_milvus_mock_image_returns_submit_ids(settings):
    client = MilvusClient(settings)
    assert client.mock is True
    hits = client.search_image([0.0] * PE_DIM, top_k=10)
    assert len(hits) == 10
    for h in hits:
        assert h["submit_keyframe_id"].count("/") == 2
        assert 0.0 <= h["score"] <= 1.0


@pytest.mark.asyncio
async def test_glap_encoder_mock_dim(settings):
    client = GlapEncoderClient(settings)
    vecs = await client.encode_text(["piano music"])
    assert len(vecs[0]) == GLAP_DIM  # 1024-d GLAP space


@pytest.mark.asyncio
async def test_milvus_audio_vector_mock(settings):
    client = MilvusClient(settings)
    hits = client.search_audio([0.0] * GLAP_DIM, top_k=6)
    assert hits and all(h["submit_keyframe_id"].count("/") == 2 for h in hits)
    assert all("top1_label" in h for h in hits)


@pytest.mark.asyncio
async def test_pe_encoder_mock_unit_norm(settings):
    client = PeEncoderClient(settings)
    vecs = await client.encode_text(["a flooded street"])
    assert len(vecs) == 1
    assert len(vecs[0]) == PE_DIM
    norm = sum(v * v for v in vecs[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_health_reports_mock(settings):
    assert (await ElasticClient(settings).health())["ok"] is True
    assert (await MilvusClient(settings).health())["ok"] is True
    assert (await PeEncoderClient(settings).health())["ok"] is True
