"""A failed VI→EN translation must never outlive the failure that caused it.

`translate.py` deliberately does not cache a failure: a rate limit is temporary
and the next search should try again. The parser caches the whole parse on top of
that, so caching one carrying `translation_failed` pinned the query to its
Vietnamese text until the backend restarted. During a bulk run that saturated the
network, one unlucky question ended up permanently searching PE with Vietnamese
while every other tab was fine, and re-running it changed nothing.
"""
import pytest

import app.translate as translate_module
from app.query_parser import QueryParser

QUERY = "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục."


@pytest.fixture
def parser(settings):
    settings.mock_mode = False
    settings.translate_to_en = True
    return QueryParser(settings)


def _translator(monkeypatch, *, ok: bool, english: str = "a group exercising"):
    async def fake(text, **kwargs):
        return (english, True) if ok else (text, False)

    monkeypatch.setattr(translate_module, "translate_vi_to_en_status", fake)


@pytest.mark.asyncio
async def test_a_transient_translation_failure_is_retried_on_the_next_search(
    parser, monkeypatch
):
    _translator(monkeypatch, ok=False)
    first = await parser.parse(QUERY, use_llm=False)
    assert first["translation_failed"] is True

    # The network comes back and the operator presses Search again.
    _translator(monkeypatch, ok=True)
    second = await parser.parse(QUERY, use_llm=False)

    assert second.get("translation_failed") is None
    assert second["translated_en_visual"] == "a group exercising"
    assert second["channels"]["image_pe"]["queries_en"] == ["a group exercising"]


@pytest.mark.asyncio
async def test_a_successful_parse_is_still_cached(parser, monkeypatch):
    """The retry must not cost a round trip on every search of a working query."""
    calls = 0

    async def counting(text, **kwargs):
        nonlocal calls
        calls += 1
        return "a group exercising", True

    monkeypatch.setattr(translate_module, "translate_vi_to_en_status", counting)
    await parser.parse(QUERY, use_llm=False)
    await parser.parse(QUERY, use_llm=False)

    assert calls == 1


@pytest.mark.asyncio
async def test_one_stuck_query_never_poisons_another(parser, monkeypatch):
    """The symptom that surfaced this: one tab that could never translate while
    every other tab was fine."""
    other = "Người đàn ông đang nấu ăn trong bếp."

    async def selective(text, **kwargs):
        if text.startswith("Cảnh quay"):
            return text, False
        return "a man cooking", True

    monkeypatch.setattr(translate_module, "translate_vi_to_en_status", selective)
    stuck = await parser.parse(QUERY, use_llm=False)
    fine = await parser.parse(other, use_llm=False)

    assert stuck["translation_failed"] is True
    assert fine.get("translation_failed") is None

    _translator(monkeypatch, ok=True)
    assert (await parser.parse(QUERY, use_llm=False)).get("translation_failed") is None
