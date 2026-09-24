"""The operator's VI→EN tick box.

Translation is on by default because PE-Core is English-centric, but it is not
always right: an English query, a proper noun, or a phrase Google keeps mangling
("chèo thuyền" → "rowing" when the query is about a place called Chèo) is better
searched exactly as typed. Unticking the box has to mean that literally — no
translation call, and no English rewrite from the LLM parser either — and it must
not leak across the parse cache into the searches that did want it.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.query_parser import QueryParser
from app.services.search_service import SearchService

client = TestClient(app)

VI = "người đàn ông đang nấu ăn"


def live(settings):
    """Settings that reach the translation step (it is skipped in mock mode)."""
    settings.mock_mode = False
    return settings


@pytest.fixture
def translator(monkeypatch):
    """Record every translation call and answer with a fixed English string."""
    import app.translate as translate_module

    calls: list[str] = []

    async def fake_status(text, **kwargs):
        calls.append(text)
        return "a man cooking", True

    async def fake(text, **kwargs):
        calls.append(text)
        return "a man cooking"

    monkeypatch.setattr(translate_module, "translate_vi_to_en_status", fake_status)
    monkeypatch.setattr(translate_module, "translate_vi_to_en", fake)
    return calls


# ---- flat vector search ------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_search_translates_by_default(settings, translator):
    svc = SearchService(live(settings))
    svc.milvus.mock = True

    body = await svc.simple_image_search(VI, top_k=5)

    assert translator == [VI]
    assert body["translated_query"] == "a man cooking"


@pytest.mark.asyncio
async def test_simple_search_leaves_the_query_alone_when_unticked(settings, translator):
    svc = SearchService(live(settings))
    svc.milvus.mock = True

    body = await svc.simple_image_search(VI, top_k=5, translate=False)

    assert translator == [], "unticking the box must not call the translator at all"
    assert body["translated_query"] is None


# ---- parser ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parser_skips_translation_when_unticked(settings, translator):
    parsed = await QueryParser(live(settings)).parse(VI, translate=False)

    assert translator == []
    assert parsed["translated_en_visual"] == VI
    assert parsed["channels"]["image_pe"]["queries_en"] == [VI]


@pytest.mark.asyncio
async def test_the_two_settings_do_not_share_a_cache_entry(settings, translator):
    """One cache key for both would serve whichever search ran first to the other
    — an untranslated query answering a ticked box, or the reverse."""
    parser = QueryParser(live(settings))

    off = await parser.parse(VI, translate=False)
    on = await parser.parse(VI, translate=True)
    off_again = await parser.parse(VI, translate=False)

    assert off["channels"]["image_pe"]["queries_en"] == [VI]
    assert on["channels"]["image_pe"]["queries_en"] == ["a man cooking"]
    assert off_again["channels"]["image_pe"]["queries_en"] == [VI]


@pytest.mark.asyncio
async def test_the_llm_rewrite_is_undone_when_unticked(settings, monkeypatch):
    """The LLM parser translates as part of parsing. Honouring the tick box means
    putting the operator's own words back, not just skipping the fallback call."""
    parser = QueryParser(live(settings))

    async def fake_llm_parse(query, hint, previous_hints):
        return {
            "query_type": "T-KIS",
            "original_query": query,
            "normalized_vi": query,
            "translated_en_visual": "a man cooking",
            "channels": {"image_pe": {"enabled": True, "weight": 1.0, "queries_en": ["a man cooking"]}},
            "trake": {
                "enabled": True,
                "events": [
                    {
                        "event_index": 1,
                        "description_vi": "người đàn ông cầm chảo",
                        "description_en_visual": "a man holding a pan",
                        "image_pe_queries_en": ["a man holding a pan"],
                    }
                ],
            },
        }

    monkeypatch.setattr(parser, "_llm_parse", fake_llm_parse)
    monkeypatch.setattr(parser.s, "query_llm_api_key", "llm-key")

    parsed = await parser.parse(VI, use_llm=True, translate=False)

    assert parsed["translated_en_visual"] == ""
    assert parsed["channels"]["image_pe"]["queries_en"] == [VI]
    assert parsed["trake"]["events"][0]["image_pe_queries_en"] == ["người đàn ông cầm chảo"]


# ---- API wiring --------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("/api/query/parse", {"query": VI}),
        ("/api/search", {"query": VI}),
        ("/api/search/trake", {"query": VI}),
        ("/api/answers/generate", {"query": VI, "limit": 1}),
    ],
)
def test_every_search_route_forwards_the_flag(monkeypatch, route, body):
    """A tick box the console honours on one route and drops on the next is worse
    than no tick box: the operator cannot tell which result came from which."""
    from app.main import search_services

    seen: list[bool] = []
    original = search_services["btc"].parser.parse

    async def spy(*args, translate=True, **kwargs):
        seen.append(translate)
        return await original(*args, translate=translate, **kwargs)

    monkeypatch.setattr(search_services["btc"].parser, "parse", spy)

    assert client.post(route, json={**body, "translate": False}).status_code == 200
    assert seen == [False]


# ---- voice input -------------------------------------------------------------


@pytest.fixture
def whisper(monkeypatch):
    """A Whisper that always hears the same Vietnamese sentence."""
    import app.transcribe as transcribe_module

    monkeypatch.setattr(transcribe_module, "available", lambda: True)
    monkeypatch.setattr(transcribe_module, "transcribe_audio", lambda data, language="vi": VI)


def _post_audio(**params):
    return client.post(
        "/api/transcribe",
        files={"audio": ("voice.webm", b"fake-audio", "audio/webm")},
        params=params,
    )


def test_dictation_is_translated_by_default(whisper, monkeypatch):
    async def fake(text, **kwargs):
        return "a man cooking"

    monkeypatch.setattr("app.main.translate_vi_to_en", fake)

    body = _post_audio().json()

    assert body == {"text": VI, "text_en": "a man cooking"}


def test_dictation_keeps_the_spoken_words_when_unticked(whisper, monkeypatch):
    async def boom(text, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("voice input must not be translated when the box is off")

    monkeypatch.setattr("app.main.translate_vi_to_en", boom)

    body = _post_audio(translate="false").json()

    assert body == {"text": VI, "text_en": None}
