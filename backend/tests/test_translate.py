import pytest

from app.translate import _looks_english, translate_vi_to_en


def test_looks_english():
    assert _looks_english("a flooded street with motorbikes") is True
    assert _looks_english("một người đàn ông") is False
    assert _looks_english("Ha Long Bay") is True
    assert _looks_english("vịnh hạ long") is False


@pytest.mark.asyncio
async def test_english_passthrough_no_network():
    # English / empty short-circuits before any network call.
    assert await translate_vi_to_en("a man riding a motorbike") == "a man riding a motorbike"
    assert await translate_vi_to_en("") == ""
    assert await translate_vi_to_en("   ") == ""


@pytest.mark.asyncio
async def test_translation_failure_falls_back(monkeypatch):
    # Simulate a network error -> must return the original text, never raise.
    import app.translate as t

    class BoomClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): raise RuntimeError("network down")

    monkeypatch.setattr(t.httpx, "AsyncClient", BoomClient)
    t._cache.clear()
    assert await translate_vi_to_en("vịnh hạ long mùa hè") == "vịnh hạ long mùa hè"


@pytest.mark.asyncio
async def test_parser_does_not_translate_in_mock(settings):
    # In mock mode the parser must not call the network translator.
    from app.query_parser import QueryParser

    parsed = await QueryParser(settings).parse("một người đàn ông đi xe máy")
    # mock mode => translated_en_visual stays the original normalized text
    assert parsed["translated_en_visual"] == "một người đàn ông đi xe máy"
