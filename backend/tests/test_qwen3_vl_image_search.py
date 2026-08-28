"""Qwen3-VL-Embedding-8B image retrieval.

Two independent embedding spaces answer the same InfoShot++ keyframes: PE-Core's
1280-d collection and Qwen's native 4096-d one. The rules under test are the ones
that make that safe — a Qwen vector never reaches the PE collection, a BTC request
never claims a Qwen index it has no vectors for, the two are combined by rank and
never by adding their incomparable cosine scores, and the encoder never invents a
vector in live mode (a fabricated query looks healthy while returning noise).
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from app.adapters.milvus_client import MilvusClient
from app.adapters.qwen3_vl_encoder import (
    QWEN3_VL_DIM,
    Qwen3VlEncoderClient,
    Qwen3VlEncoderUnavailable,
)
from app.main import app
from app.services.search_service import SearchService
from app.services.trake_service import TrakeService

client = TestClient(app)


def infoshotpp(settings):
    """InfoShot++ settings with both image indices and the Qwen worker configured."""
    settings.milvus_endpoint_2 = "https://infoshot.milvus.test"
    settings.milvus_token_2 = "infoshot-token"
    settings.qwen3_vl_encoder_url = "https://qwen.tunnel.test"
    settings.qwen3_vl_encoder_token = "qwen-token"
    return settings.for_retrieval_database("infoshotpp")


def live_encoder(settings, handler):
    """A non-mock Qwen encoder client answered by an in-process transport."""
    encoder = Qwen3VlEncoderClient(infoshotpp(settings))
    encoder.mock = False
    transport = httpx.MockTransport(handler)
    encoder._http.get = lambda: httpx.AsyncClient(transport=transport)  # type: ignore[method-assign]
    return encoder


VISUAL_PARSE = {
    "channels": {"image_pe": {"enabled": True, "weight": 1.0, "queries_en": ["a street"]}},
    "filters": {},
    "rerank_policy": {"rrf_k": 60},
}


# ---- selection contract ------------------------------------------------------


def test_btc_requests_cannot_select_an_index_that_has_no_vectors():
    """BTC keyframes were never encoded with Qwen, so the API refuses rather than
    silently answering from PE while the UI shows a ticked Qwen box."""
    for route, body in (
        ("/api/search", {"query": "x", "image_models": ["qwen3_vl"]}),
        ("/api/search/simple", {"query": "x", "image_models": ["qwen3_vl"]}),
        ("/api/search/trake", {"query": "x", "image_models": ["pe", "qwen3_vl"]}),
    ):
        response = client.post(route, json={"retrieval_database": "btc", **body})
        assert response.status_code == 422, route


def test_image_model_selection_is_validated_at_the_api_boundary():
    for body in (
        {"image_models": []},
        {"image_models": ["pe", "pe"]},
        {"image_models": ["siglip"]},
        {"image_models": ["pe", "qwen3_vl", "pe"]},
    ):
        response = client.post("/api/search", json={"query": "x", **body})
        assert response.status_code == 422, body


def test_search_defaults_to_pe_when_no_model_is_named():
    body = client.post("/api/search", json={"query": "bản tin thời sự"}).json()
    assert body["image_models"] == ["pe"]
    assert "image_qwen" not in body["latency_ms"]["channels"]


@pytest.mark.asyncio
async def test_service_level_callers_get_the_same_rule_as_the_api(settings):
    """`resolve_image_models` is the single gate, so answer generation and TRAKE
    cannot bypass the check by calling the service directly."""
    btc = SearchService(settings.for_retrieval_database("btc"))

    assert btc.resolve_image_models(None) == ("pe",)
    with pytest.raises(ValueError):
        btc.resolve_image_models(["qwen3_vl"])
    with pytest.raises(ValueError):
        btc.resolve_image_models([])
    with pytest.raises(ValueError):
        btc.resolve_image_models(["pe", "pe"])

    svc = SearchService(infoshotpp(settings))
    assert svc.resolve_image_models(["qwen3_vl", "pe"]) == ("qwen3_vl", "pe")


# ---- collection isolation ----------------------------------------------------


@pytest.mark.asyncio
async def test_each_model_only_ever_reads_its_own_collection(settings, monkeypatch):
    profile = infoshotpp(settings)
    profile.milvus_image_collection = "infoshot-pe"
    profile.milvus_qwen3_vl_image_collection = "infoshot-qwen"
    milvus = MilvusClient(profile)
    seen: list[tuple[str, int]] = []

    def spy(collection_name, vector, **kwargs):
        seen.append((collection_name, len(vector)))
        return []

    monkeypatch.setattr(milvus, "_search_image_collection", spy)
    milvus.search_image([0.0] * 1280)
    milvus.search_qwen_image([0.0] * QWEN3_VL_DIM)

    assert seen == [("infoshot-pe", 1280), ("infoshot-qwen", QWEN3_VL_DIM)]


def test_btc_profile_has_no_qwen_collection_to_search(settings):
    btc = MilvusClient(settings.for_retrieval_database("btc"))

    with pytest.raises(RuntimeError):
        btc.search_qwen_image([0.0] * QWEN3_VL_DIM)
    with pytest.raises(RuntimeError):
        btc.get_qwen_image_vectors(["K01/K01_V001/001"])


@pytest.mark.asyncio
async def test_qwen_only_search_never_calls_the_pe_encoder(settings, monkeypatch):
    svc = SearchService(infoshotpp(settings))

    async def forbidden(*args, **kwargs):
        raise AssertionError("PE must not be encoded for a Qwen-only search")

    monkeypatch.setattr(svc.pe, "encode_text", forbidden)
    body = await svc.search(
        {"query": "x", "parsed": VISUAL_PARSE, "image_models": ["qwen3_vl"]}
    )

    assert body["image_models"] == ["qwen3_vl"]
    assert set(body["latency_ms"]["channels"]) == {"image_qwen"}
    assert body["groups"][0]["frames"][0]["channels"] == ["image_qwen"]


# ---- fusion ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_models_run_as_two_channels_and_meet_in_rrf(settings):
    svc = SearchService(infoshotpp(settings))

    body = await svc.search(
        {"query": "x", "parsed": VISUAL_PARSE, "image_models": ["pe", "qwen3_vl"]}
    )

    assert body["image_models"] == ["pe", "qwen3_vl"]
    assert set(body["latency_ms"]["channels"]) == {"image_pe", "image_qwen"}
    frame = body["groups"][0]["frames"][0]
    assert frame["channels"] == ["image_pe", "image_qwen"]
    # Each channel keeps its own cosine; the fused score is the RRF sum of two
    # rank-1 contributions, never the sum of two unrelated cosine values.
    assert set(frame["per_channel_score"]) == {"image_pe", "image_qwen"}
    assert frame["score"] == pytest.approx(2 / 61, abs=1e-6)


@pytest.mark.asyncio
async def test_simple_search_reports_which_models_answered(settings):
    svc = SearchService(infoshotpp(settings))

    both = await svc.simple_image_search("a street", top_k=5, image_models=["pe", "qwen3_vl"])
    top = both["results"][0]

    assert both["image_models"] == ["pe", "qwen3_vl"]
    assert top["models"] == ["pe", "qwen3_vl"]
    assert set(top["per_model_score"]) == {"pe", "qwen3_vl"}
    assert top["score"] == pytest.approx(2 / 61, abs=1e-6)
    assert set(both["model_latency_ms"]) == {"pe", "qwen3_vl"}

    # One model keeps its own cosine order rather than being re-scored by RRF.
    only_pe = await svc.simple_image_search("a street", top_k=5, image_models=["pe"])
    assert only_pe["results"][0]["models"] == ["pe"]
    assert only_pe["results"][0]["score"] == pytest.approx(
        only_pe["results"][0]["per_model_score"]["pe"]
    )


@pytest.mark.asyncio
async def test_simple_search_survives_one_dead_model(settings, monkeypatch):
    """A stopped Colab worker must degrade to the other index, not to zero results."""
    svc = SearchService(infoshotpp(settings))

    async def dead(texts):
        raise Qwen3VlEncoderUnavailable("Qwen3-VL encoder không phản hồi sau 120s")

    monkeypatch.setattr(svc.qwen3_vl, "encode_text", dead)
    body = await svc.simple_image_search("a street", top_k=5, image_models=["pe", "qwen3_vl"])

    assert body["results"], "PE results must survive a dead Qwen worker"
    assert body["results"][0]["models"] == ["pe"]
    assert any("qwen3_vl" in warning for warning in body["warnings"])


@pytest.mark.asyncio
async def test_relevance_feedback_reuses_each_model_own_stored_vectors(settings, monkeypatch):
    """`similar` seeds from stored vectors; a PE seed vector searched in the Qwen
    collection would be a dimension error at best and noise at worst."""
    svc = SearchService(infoshotpp(settings))
    fetched: list[str] = []

    def spy_pe(ids):
        fetched.append("pe")
        return {ids[0]: [0.1] * 8}

    def spy_qwen(ids):
        fetched.append("qwen3_vl")
        return {ids[0]: [0.1] * 8}

    monkeypatch.setattr(svc.milvus, "get_image_vectors", spy_pe)
    monkeypatch.setattr(svc.milvus, "get_qwen_image_vectors", spy_qwen)
    groups, _ = await svc.retrieve(
        VISUAL_PARSE,
        top_k=5,
        feedback={"positive_frames": ["L21/L21_V001/001"]},
        image_models=("pe", "qwen3_vl"),
    )

    assert sorted(fetched) == ["pe", "qwen3_vl"]
    evidence = [
        item
        for group in groups
        for frame in group.frames
        for item in frame.evidence
        if item.type == "similar"
    ]
    assert evidence and evidence[0].extra["models"] == ["pe", "qwen3_vl"]
    assert set(evidence[0].extra["per_model_score"]) <= {"pe", "qwen3_vl"}


@pytest.mark.asyncio
async def test_trake_threads_the_selection_through_every_event(settings):
    svc = SearchService(infoshotpp(settings))
    trake = TrakeService(svc.s, svc)

    body = await trake.search_trake(
        {"query": "cầu thủ chạy đà rồi sút bóng", "image_models": ["pe", "qwen3_vl"]}
    )

    assert body["image_models"] == ["pe", "qwen3_vl"]
    assert body["sequences"]


# ---- degraded-search notices -------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("models", "expect_warning"),
    [(["pe"], True), (["pe", "qwen3_vl"], True), (["qwen3_vl"], False)],
    ids=["pe-only", "both", "qwen-only"],
)
async def test_failed_translation_only_warns_about_the_model_that_needs_english(
    settings, monkeypatch, models, expect_warning
):
    """Qwen3-VL is multilingual and is queried in Vietnamese by design, so a
    Qwen-only search loses nothing when VI->EN fails. Warning there would send the
    operator chasing a problem that does not exist."""
    import app.translate as translate_module

    svc = SearchService(infoshotpp(settings))
    svc.s.mock_mode = False  # the translation step is skipped in mock mode
    svc.milvus.mock = True

    async def failed_translation(text, **kwargs):
        return text, False

    monkeypatch.setattr(translate_module, "translate_vi_to_en_status", failed_translation)
    body = await svc.simple_image_search(
        "ngu\u1eddi \u0111\u00e0n \u00f4ng \u0111ang n\u1ea5u \u0103n",
        top_k=5,
        image_models=models,
    )
    notices = [w for w in body.get("warnings", []) if "d\u1ecbch" in w]

    assert bool(notices) is expect_warning
    if expect_warning:
        assert "PE Core" in notices[0]
        assert ("Qwen3-VL kh\u00f4ng b\u1ecb \u1ea3nh h\u01b0\u1edfng" in notices[0]) is (
            "qwen3_vl" in models
        )


# ---- encoder contract --------------------------------------------------------


@pytest.mark.asyncio
async def test_encoder_refuses_to_fabricate_a_vector_when_unconfigured(settings):
    profile = settings.for_retrieval_database("infoshotpp")
    profile.qwen3_vl_encoder_url = None
    profile.qwen3_vl_encoder_token = None
    encoder = Qwen3VlEncoderClient(profile)
    encoder.mock = False

    assert encoder.enabled is False
    with pytest.raises(Qwen3VlEncoderUnavailable):
        await encoder.encode_text(["a street"])


@pytest.mark.asyncio
async def test_encoder_sends_the_bearer_token_and_returns_native_vectors(settings):
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"vectors": [[1.0] + [0.0] * (QWEN3_VL_DIM - 1)]}
        )

    vectors = await live_encoder(settings, handler).encode_text(["a street"])

    assert seen["url"] == "https://qwen.tunnel.test/encode-text"
    assert seen["auth"] == "Bearer qwen-token"
    assert len(vectors[0]) == QWEN3_VL_DIM


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"vectors": [[1.0] + [0.0] * 1279]},  # PE's dimension from a mis-set URL
        {"vectors": [[0.5] + [0.0] * (QWEN3_VL_DIM - 1)]},  # not L2-normalized
        {"vectors": [[float("nan")] + [0.0] * (QWEN3_VL_DIM - 1)]},
        {"vectors": [["not-a-number"] + [0.0] * (QWEN3_VL_DIM - 1)]},
        {"vectors": []},  # fewer vectors than texts
        {"detail": "no vectors at all"},
    ],
    ids=["pe-dimension", "unnormalized", "nan", "non-numeric", "short", "no-vectors"],
)
async def test_encoder_rejects_every_unusable_response(settings, payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(Qwen3VlEncoderUnavailable):
        await live_encoder(settings, handler).encode_text(["a street"])


@pytest.mark.asyncio
async def test_encoder_reports_upstream_failures_with_a_reason(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Invalid bearer token.")

    with pytest.raises(Qwen3VlEncoderUnavailable) as raised:
        await live_encoder(settings, handler).encode_text(["a street"])
    assert "401" in str(raised.value)


@pytest.mark.asyncio
async def test_health_flags_a_worker_serving_the_wrong_dimension(settings):
    """A tunnel pointed at the PE server answers `/health` happily; only the
    dimension gives it away, and it must do so before any search runs."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "dim": 1280})

    health = await live_encoder(settings, handler).health()

    assert health["ok"] is False
    assert health["mode"] == "incompatible"


# ---- health ------------------------------------------------------------------


def test_health_exposes_the_qwen_worker_and_its_collection():
    body = client.get("/api/health", params={"retrieval_database": "infoshotpp"}).json()

    assert body["services"]["qwen3_vl_encoder"]["ok"] is True
    assert body["services"]["qwen3_vl_milvus"]["ok"] is True
    assert body["capabilities"]["qwen3_vl_embedding_search"] is True
    assert body["indices"]["image_qwen"] == "aic26_image_qwen3vl8b_infoshotpp_v3"


def test_btc_health_reports_no_qwen_index():
    body = client.get("/api/health", params={"retrieval_database": "btc"}).json()

    assert body["services"]["qwen3_vl_encoder"]["ok"] is False
    assert body["capabilities"]["qwen3_vl_embedding_search"] is False
    assert body["indices"]["image_qwen"] is None
