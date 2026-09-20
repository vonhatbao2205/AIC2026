"""Translation must not fail silently, and must not rest on one endpoint.

Google's free endpoint began answering `429 Too Many Requests`. `translate.py`
caught it and returned the Vietnamese text unchanged, so PE-Core — an
English-centric encoder — was embedding Vietnamese. The console showed
plausible-looking keyframes that did not match the query, with nothing anywhere
saying why. These tests hold both halves of the fix: a second (and third)
provider, and a failure the operator can actually see.
"""
import httpx
import pytest

from app import translate as tr


@pytest.fixture(autouse=True)
def _clear_translation_cache():
    tr._cache.clear()
    yield
    tr._cache.clear()


def _transport(handler):
    return httpx.MockTransport(handler)


@pytest.fixture
def patch_client(monkeypatch):
    """Route every httpx.AsyncClient in translate.py through one handler."""

    # Captured once, before any patching: re-reading it inside `apply` would make
    # a second call wrap the first factory and keep its handler.
    real = httpx.AsyncClient

    def apply(handler):
        def factory(*args, **kwargs):
            kwargs["transport"] = _transport(handler)
            return real(*args, **kwargs)

        monkeypatch.setattr(tr.httpx, "AsyncClient", factory)

    return apply


CLIENTS5_OK = httpx.Response(200, json=[["a man in a blue suit", "vi"]])
GTX_OK = httpx.Response(200, json=[[["a man in a blue suit", "một người...", None, None]]])
RATE_LIMITED = httpx.Response(429, text="<html><title>Sorry...</title></html>")

VI = "một người đàn ông mặc vest xanh"


@pytest.mark.asyncio
async def test_uses_the_working_endpoint(patch_client):
    patch_client(lambda request: CLIENTS5_OK if "clients5" in str(request.url) else RATE_LIMITED)

    text, ok = await tr.translate_vi_to_en_status(VI)

    assert ok is True
    assert text == "a man in a blue suit"


@pytest.mark.asyncio
async def test_falls_through_to_the_second_host_when_the_first_is_rate_limited(patch_client):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return GTX_OK if "translate.googleapis.com" in str(request.url) else RATE_LIMITED

    patch_client(handler)
    text, ok = await tr.translate_vi_to_en_status(VI)

    assert ok is True
    assert text == "a man in a blue suit"
    assert any("clients5" in url for url in seen), "the first provider must still be tried"


@pytest.mark.asyncio
async def test_llm_rescues_the_query_when_both_google_hosts_refuse(patch_client, settings):
    settings.nvidia_api_key = "nim-key"
    settings.nvidia_base_url = "https://nim.test/v1"

    def handler(request):
        if "nim.test" in str(request.url):
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "a man in a blue suit"}}]
            })
        return RATE_LIMITED

    patch_client(handler)
    text, ok = await tr.translate_vi_to_en_status(VI, settings=settings)

    assert ok is True
    assert text == "a man in a blue suit"


@pytest.mark.asyncio
async def test_total_failure_is_reported_not_swallowed(patch_client, settings):
    settings.nvidia_api_key = None  # no LLM to fall back to
    patch_client(lambda request: RATE_LIMITED)

    text, ok = await tr.translate_vi_to_en_status(VI, settings=settings)

    assert ok is False, "a failed translation must be visible to the caller"
    assert text == VI, "...while still returning something searchable"


@pytest.mark.asyncio
async def test_a_failure_is_not_cached(patch_client, settings):
    """A rate limit is temporary; pinning the untranslated text would make one
    bad minute poison every later search for that query."""
    settings.nvidia_api_key = None
    patch_client(lambda request: RATE_LIMITED)
    await tr.translate_vi_to_en_status(VI, settings=settings)
    assert VI not in tr._cache

    patch_client(lambda request: CLIENTS5_OK)
    text, ok = await tr.translate_vi_to_en_status(VI, settings=settings)
    assert (text, ok) == ("a man in a blue suit", True)


@pytest.mark.asyncio
async def test_english_input_is_left_alone_without_any_request(patch_client):
    def handler(request):  # pragma: no cover - must never run
        raise AssertionError("English text must not hit a translation endpoint")

    patch_client(handler)
    text, ok = await tr.translate_vi_to_en_status("a man in a blue suit")

    assert (text, ok) == ("a man in a blue suit", True)


@pytest.mark.asyncio
async def test_search_warns_when_the_visual_query_stayed_vietnamese(settings, monkeypatch):
    """The whole point: the operator is told why the keyframes look wrong."""
    from app.services.search_service import SearchService

    svc = SearchService(settings)
    parsed = {
        "channels": {"image_pe": {"enabled": True, "weight": 1.0, "queries_en": [VI]}},
        "filters": {},
        "rerank_policy": {"rrf_k": 60},
        "translation_failed": True,
    }

    res = await svc.search({"query": VI, "parsed": parsed})

    assert any("Could not translate" in warning for warning in res["warnings"])
