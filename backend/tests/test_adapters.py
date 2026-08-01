"""Mocked adapter tests — exercise the mock-mode paths (no live services)."""
import httpx
import pytest

import app.adapters.gemini_grounding as gemini_grounding_module
from app.adapters.elastic_client import ElasticClient
from app.adapters.gemini_grounding import (
    GeminiGroundingClient,
    _model_candidates,
    _parse_generate_content,
)
from app.adapters.milvus_client import MilvusClient, _sanitize_float16_vector
from app.adapters.nvila_client import NvilaQaClient
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


def test_milvus_audio_vector_zeros_float16_underflow():
    vector = [0.0, 5.6401743e-05, -5.0e-05, 6.103515625e-05, 0.25]
    assert _sanitize_float16_vector(vector) == [0.0, 0.0, 0.0, 6.103515625e-05, 0.25]


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
    nvila_health = await NvilaQaClient(settings).health()
    assert nvila_health["mode"] == "mock"
    assert nvila_health["visual_verification_pass"] is True
    assert (await GeminiGroundingClient(settings).health())["mode"] == "mock"


@pytest.mark.asyncio
async def test_nvila_qa_mock_returns_grounded_ids(settings):
    client = NvilaQaClient(settings)
    candidates = [{"candidate_id": "C01", "submit_keyframe_id": "K01/K01_V001/001"}]
    result = await client.analyze("Đây là bản tin gì?", candidates)
    assert result["best_candidate_id"] == "C01"
    assert result["candidate_answers"]


@pytest.mark.asyncio
async def test_nvila_mock_verifies_google_option_against_candidate(settings):
    client = NvilaQaClient(settings)
    result = await client.verify_grounded(
        "Tên là gì?",
        [{"candidate_id": "C03"}],
        [{"answer": "WALTDISNEY", "supporting_candidate_ids": ["C03"]}],
        hotspot_ids=["C03"],
    )
    assert result["verdicts"] == [{
        "answer": "WALTDISNEY",
        "status": "supported",
        "visual_confidence": 0.90,
        "supporting_candidate_ids": ["C03"],
        "reason": "Mock NVILA confirms consistency with the supplied visual clue.",
    }]


@pytest.mark.asyncio
async def test_gemini_grounding_mock_keeps_candidate_id(settings):
    client = GeminiGroundingClient(settings)
    candidates = [{"candidate_id": "C07", "video_id": "K20_V013", "evidence": []}]
    result = await client.ground("Tên cửa hàng là gì?", candidates, {"candidate_answers": []})
    assert result["best_answer"] == "WALTDISNEY"
    assert result["candidate_answers"][0]["supporting_candidate_ids"] == ["C07"]


def test_gemini_response_parser_preserves_grounding_provenance():
    parsed = _parse_generate_content({
        "candidates": [{
            "content": {"parts": [{"text": '<grounded_json>{"answerable": true}</grounded_json>'}]},
            "groundingMetadata": {
                "webSearchQueries": ["Disney official full name"],
                "groundingChunks": [
                    {"web": {"title": "The Walt Disney Company", "uri": "https://thewaltdisneycompany.com/"}},
                    {"web": {"title": "Duplicate", "uri": "https://thewaltdisneycompany.com/"}},
                    {"web": {"title": "Unsafe", "uri": "javascript:alert(1)"}},
                ],
                "searchEntryPoint": {"renderedContent": "<div>Google suggestions</div>"},
            },
        }],
    })
    assert parsed["queries"] == ["Disney official full name"]
    assert parsed["sources"] == [{
        "title": "The Walt Disney Company",
        "url": "https://thewaltdisneycompany.com/",
    }]
    assert parsed["search_suggestions_html"] == "<div>Google suggestions</div>"


@pytest.mark.asyncio
async def test_gemini_grounding_falls_back_when_primary_model_is_retired(monkeypatch):
    calls = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, *, json, headers):
            calls.append((url, json))
            request = httpx.Request("POST", url)
            if "gemini-2.5-flash:" in url:
                return httpx.Response(404, request=request, json={"error": {"message": "retired"}})
            return httpx.Response(200, request=request, json={
                "candidates": [{
                    "content": {"parts": [{"text": (
                        '<grounded_json>{"answerable":true,"best_answer":"WALTDISNEY",'
                        '"candidate_answers":[{"answer":"WALTDISNEY","confidence":0.9,'
                        '"supporting_candidate_ids":["C01"],"reason":"canonical name"}],'
                        '"uncertainty":""}</grounded_json>'
                    )}]},
                    "groundingMetadata": {
                        "webSearchQueries": ["Disney canonical name"],
                        "groundingChunks": [{"web": {
                            "title": "Disney",
                            "uri": "https://thewaltdisneycompany.com/",
                        }}],
                    },
                }],
            })

    monkeypatch.setattr(gemini_grounding_module.httpx, "AsyncClient", FakeAsyncClient)
    settings = gemini_grounding_module.Settings(
        gemini_api_key="test-key",
        gemini_grounding_model="models/gemini-2.5-flash",
        gemini_grounding_fallback_models=["gemini-3.6-flash"],
    )
    assert _model_candidates(settings) == ["gemini-2.5-flash", "gemini-3.6-flash"]

    result = await GeminiGroundingClient(settings).ground(
        "Tên cửa hàng là gì?",
        [{
            "candidate_id": "C01",
            "video_id": "K20_V013",
            "pts_time": 609,
            "evidence": [{"type": "speech", "text": "logo Disney"}],
        }],
        {"candidate_answers": []},
    )

    assert result["model"] == "gemini-3.6-flash"
    assert len(calls) == 2
    assert calls[1][1]["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"
    assert "temperature" not in calls[1][1]["generationConfig"]
