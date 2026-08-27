"""Qwen3-VL reranker: the opt-in tick box, over-retrieval, and fail-open.

The load-bearing property here is the last one. `SearchService.retrieve` turns a
channel exception into an empty channel, so a reranker that raised would delete
every PE result the moment the Colab tunnel dropped — a refinement turning into
the outage. Several tests below exist only to hold that line.
"""
import httpx
import pytest

from app.adapters.qwen_reranker import QwenRerankerClient, QwenRerankerUnavailable
from app.services.search_service import SearchService

PARSED = {
    "channels": {"image_pe": {"enabled": True, "weight": 1.0, "queries_en": ["a busy street"]}},
    "filters": {},
    "rerank_policy": {"rrf_k": 60},
}


def _frame_ids(res):
    return [f["submit_keyframe_id"] for g in res["groups"] for f in g["frames"]]


class _StubReranker:
    """Stands in for the Colab worker: records calls, replays a scripted answer."""

    candidate_k = 200

    def __init__(self, *, enabled=True, order=None, error=None):
        self.enabled = enabled
        self._order = order
        self._error = error
        self.calls = []

    async def rerank(self, query, documents, **kwargs):
        self.calls.append({"query": query, "documents": documents, **kwargs})
        if self._error is not None:
            raise self._error
        ids = [d["id"] for d in documents]
        chosen = [i for i in (self._order if self._order is not None else list(reversed(ids)))
                  if i in ids]
        return [{"id": doc_id, "score": 1.0 - n / 100} for n, doc_id in enumerate(chosen)]


# ---- the tick box is genuinely opt-in --------------------------------------


@pytest.mark.asyncio
async def test_search_does_not_rerank_by_default(settings):
    svc = SearchService(settings)
    stub = _StubReranker()
    svc.reranker = stub

    res = await svc.search({"query": "một con phố đông", "parsed": PARSED})

    assert stub.calls == [], "reranking must never happen unless the operator asks"
    assert "reranker" not in res["latency_ms"]


@pytest.mark.asyncio
async def test_rerank_flag_reorders_the_image_channel(settings):
    svc = SearchService(settings)
    baseline = _frame_ids(await svc.search({"query": "một con phố đông", "parsed": PARSED}))
    assert len(baseline) > 1, "fixture must return enough frames to reorder"

    svc.reranker = _StubReranker(order=list(reversed(baseline)))
    res = await svc.search({"query": "một con phố đông", "parsed": PARSED, "rerank": True})

    assert _frame_ids(res)[: len(baseline)] == list(reversed(baseline))
    assert res["latency_ms"]["reranker"]["ok"] is True
    assert res["latency_ms"]["reranker"]["candidates"] >= len(baseline)


@pytest.mark.asyncio
async def test_rerank_flag_is_inert_when_no_worker_is_configured(settings):
    svc = SearchService(settings)
    baseline = _frame_ids(await svc.search({"query": "một con phố đông", "parsed": PARSED}))
    svc.reranker = _StubReranker(enabled=False)

    res = await svc.search({"query": "một con phố đông", "parsed": PARSED, "rerank": True})

    assert svc.reranker.calls == []
    assert _frame_ids(res) == baseline


# ---- over-retrieval ---------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_over_retrieves_before_scoring(settings, monkeypatch):
    """A frame PE ranked 143rd is invisible to a reranker sent only 100."""
    svc = SearchService(settings)
    svc.reranker = _StubReranker()
    seen = []
    original = svc.milvus.search_image

    def spy(vector, *, top_k, categories=()):
        seen.append(top_k)
        return original(vector, top_k=top_k, categories=categories)

    monkeypatch.setattr(svc.milvus, "search_image", spy)
    await svc.search({"query": "phố", "parsed": PARSED, "rerank": True, "top_k": 20})

    assert seen and all(k == _StubReranker.candidate_k for k in seen)


@pytest.mark.asyncio
async def test_result_is_still_cut_back_to_top_k(settings):
    svc = SearchService(settings)
    svc.reranker = _StubReranker()

    res = await svc.search({"query": "phố", "parsed": PARSED, "rerank": True, "top_k": 3})

    assert len(_frame_ids(res)) <= 3


# ---- fail-open --------------------------------------------------------------


@pytest.mark.asyncio
async def test_reranker_failure_keeps_the_pe_result(settings):
    svc = SearchService(settings)
    baseline = _frame_ids(await svc.search({"query": "phố đông", "parsed": PARSED}))
    svc.reranker = _StubReranker(error=QwenRerankerUnavailable("tunnel down"))

    res = await svc.search({"query": "phố đông", "parsed": PARSED, "rerank": True})

    assert _frame_ids(res) == baseline, "a dead reranker must not cost a single frame"
    assert res["latency_ms"]["reranker"]["ok"] is False
    assert "tunnel down" in res["latency_ms"]["reranker"]["error"]
    assert any("Reranker" in warning for warning in res["warnings"])


@pytest.mark.asyncio
async def test_unexpected_reranker_exception_also_fails_open(settings):
    svc = SearchService(settings)
    baseline = _frame_ids(await svc.search({"query": "phố đông", "parsed": PARSED}))
    svc.reranker = _StubReranker(error=RuntimeError("boom"))

    res = await svc.search({"query": "phố đông", "parsed": PARSED, "rerank": True})

    assert _frame_ids(res) == baseline
    assert res["latency_ms"]["reranker"]["ok"] is False


@pytest.mark.asyncio
async def test_frames_the_worker_could_not_score_keep_their_pe_order(settings):
    """A 404 keyframe drops out of `scores`; it must not drop out of the result.

    Asserted on the channel itself rather than through `search()`: group-by-video
    orders frames by aggregate video score, so the channel's own tail ordering is
    not observable from the grouped response.
    """
    svc = SearchService(settings)
    cfg = PARSED["channels"]["image_pe"]
    plain, _ = await svc._run_image_pe(cfg, 50)
    baseline = [h.submit_keyframe_id for h in plain]
    assert len(baseline) >= 3
    # The worker answers about exactly one candidate and ignores the rest.
    svc.reranker = _StubReranker(order=[baseline[-1]])

    hits, _ = await svc._run_image_pe(cfg, 50, rerank=True)
    ids = [h.submit_keyframe_id for h in hits]

    assert ids[0] == baseline[-1]
    assert set(ids) == set(baseline), "unscored frames must survive"
    assert ids[1:] == [i for i in baseline if i != baseline[-1]]
    assert [h.rank for h in hits] == list(range(len(hits))), "RRF fuses on rank"
    assert hits[0].evidence.extra["rerank_score"] == pytest.approx(1.0)
    assert hits[1].evidence.extra == {}, "unscored frames carry no rerank score"


# ---- what is actually sent to the worker ------------------------------------


@pytest.mark.asyncio
async def test_worker_receives_keyframe_urls_and_only_the_main_query(settings):
    svc = SearchService(settings)
    svc.reranker = _StubReranker()
    parsed = {
        **PARSED,
        "channels": {
            "image_pe": {
                "enabled": True,
                "weight": 1.0,
                # Query expansion produced three phrasings; reranking each would
                # triple the GPU cost to answer a question nobody asked.
                "queries_en": ["a busy street", "a crowded road", "traffic downtown"],
            }
        },
    }

    await svc.search({"query": "phố", "parsed": parsed, "rerank": True})

    call = svc.reranker.calls[0]
    assert call["query"] == "a busy street"
    assert len(svc.reranker.calls) == 1
    document = call["documents"][0]
    assert set(document) == {"id", "image"}
    assert document["image"].startswith("https://media.test/Keyframes/")


# ---- the second visual code path -------------------------------------------


@pytest.mark.asyncio
async def test_simple_search_honours_the_flag(settings):
    svc = SearchService(settings)
    plain = await svc.simple_image_search("a busy street", top_k=5)
    baseline = [r["submit_keyframe_id"] for r in plain["results"]]
    assert "reranker" not in plain

    svc.reranker = _StubReranker(order=list(reversed(baseline)))
    ranked = await svc.simple_image_search("a busy street", top_k=5, rerank=True)

    assert [r["submit_keyframe_id"] for r in ranked["results"]] == list(reversed(baseline))
    assert ranked["reranker"]["ok"] is True
    assert ranked["results"][0]["rerank_score"] is not None


@pytest.mark.asyncio
async def test_simple_search_fails_open_too(settings):
    svc = SearchService(settings)
    baseline = [r["submit_keyframe_id"] for r in (
        await svc.simple_image_search("a busy street", top_k=5)
    )["results"]]
    svc.reranker = _StubReranker(error=QwenRerankerUnavailable("down"))

    ranked = await svc.simple_image_search("a busy street", top_k=5, rerank=True)

    assert [r["submit_keyframe_id"] for r in ranked["results"]] == baseline
    assert ranked["reranker"]["ok"] is False


# ---- the client itself ------------------------------------------------------


def test_client_is_disabled_without_a_url_and_token(settings):
    settings.mock_mode = False
    assert QwenRerankerClient(settings).enabled is False

    settings.qwen_reranker_url = "https://worker.test"
    assert QwenRerankerClient(settings).enabled is False, "a URL without a token is not configured"

    settings.qwen_reranker_token = "secret"
    assert QwenRerankerClient(settings).enabled is True

    settings.qwen_reranker_enabled = False
    assert QwenRerankerClient(settings).enabled is False, "the kill switch must win"


@pytest.mark.asyncio
async def test_mock_client_reorders_deterministically(settings):
    client = QwenRerankerClient(settings)
    assert client.enabled is True
    documents = [{"id": f"L21/L21_V001/{n:03d}", "image": "https://x"} for n in range(8)]

    first = await client.rerank("a busy street", documents)
    again = await client.rerank("a busy street", documents)

    assert [i["id"] for i in first] == [i["id"] for i in again]
    assert {i["id"] for i in first} == {d["id"] for d in documents}
    assert [i["score"] for i in first] == sorted((i["score"] for i in first), reverse=True)
    other = await client.rerank("something else entirely", documents)
    assert [i["id"] for i in other] != [i["id"] for i in first]


@pytest.mark.asyncio
async def test_mock_client_returns_nothing_for_no_documents(settings):
    assert await QwenRerankerClient(settings).rerank("q", []) == []


# ---- failure messages must be actionable ------------------------------------


def _live_client(settings, **overrides):
    settings.mock_mode = False
    settings.qwen_reranker_url = "https://worker.test"
    settings.qwen_reranker_token = "secret"
    for key, value in overrides.items():
        setattr(settings, key, value)
    return QwenRerankerClient(settings)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout(""),
        httpx.ConnectTimeout(""),
        httpx.ConnectError(""),
        httpx.RemoteProtocolError(""),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_transport_failures_never_produce_an_empty_reason(settings, monkeypatch, exc):
    """httpx transport errors stringify to "", which shipped a reason reading
    `Qwen reranker unreachable: ` with nothing after the colon — a warning that
    tells the operator nothing at all."""
    client = _live_client(settings)

    class _Boom:
        async def post(self, *args, **kwargs):
            raise exc

    monkeypatch.setattr(client._http, "get", lambda: _Boom())
    with pytest.raises(QwenRerankerUnavailable) as caught:
        await client.rerank("a street", [{"id": "X", "image": "https://x/1.jpg"}])

    message = str(caught.value)
    assert message.strip()
    assert not message.rstrip().endswith(":"), message
    assert type(exc).__name__ in message or "quá" in message, message


@pytest.mark.asyncio
async def test_timeout_names_the_two_knobs_that_fix_it(settings, monkeypatch):
    client = _live_client(settings, qwen_reranker_timeout_seconds=120.0)

    class _Slow:
        async def post(self, *args, **kwargs):
            raise httpx.ReadTimeout("")

    monkeypatch.setattr(client._http, "get", lambda: _Slow())
    documents = [{"id": f"K{n}", "image": "https://x/1.jpg"} for n in range(200)]
    with pytest.raises(QwenRerankerUnavailable) as caught:
        await client.rerank("a street", documents)

    message = str(caught.value)
    assert "120s" in message
    assert "200 keyframe" in message
    assert "QWEN_RERANKER_TIMEOUT_SECONDS" in message
    assert "QWEN_RERANKER_CANDIDATES" in message


@pytest.mark.asyncio
async def test_401_points_at_the_token_rather_than_the_network(settings, monkeypatch):
    client = _live_client(settings)
    request = httpx.Request("POST", "https://worker.test/rerank")
    response = httpx.Response(401, text="Unauthorized", request=request)

    class _Denied:
        async def post(self, *args, **kwargs):
            return response

    monkeypatch.setattr(client._http, "get", lambda: _Denied())
    with pytest.raises(QwenRerankerUnavailable) as caught:
        await client.rerank("a street", [{"id": "X", "image": "https://x/1.jpg"}])

    message = str(caught.value)
    assert "401" in message
    assert "QWEN_RERANKER_TOKEN" in message


@pytest.mark.asyncio
async def test_a_failed_rerank_reports_how_long_it_waited(settings):
    """`ms` is what tells a timeout apart from an instant connection refusal."""
    svc = SearchService(settings)
    svc.reranker = _StubReranker(error=QwenRerankerUnavailable("quá 120s không phản hồi"))

    res = await svc.search({"query": "phố", "parsed": PARSED, "rerank": True})

    report = res["latency_ms"]["reranker"]
    assert report["ok"] is False
    assert isinstance(report["ms"], float)
    assert report["candidates"] > 0
