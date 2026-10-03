"""Shared TRAKE conditions survive every visual retrieval path, offline."""
import copy

import pytest

from app.query_parser import QueryParser, heuristic_parse
from app.query_llm import to_routing, PARSER_SYSTEM_PROMPT
from app.trake_events import shared_context, compose_event_queries, contextual_event_queries
from app.services.search_service import SearchService
from app.services.trake_service import TrakeService

QUERY = 'Một con lân màu đỏ đang biểu diễn. Tìm các khoảnh khắc sau: E1: con lân nhảy lên. E2: nó quay đầu.'
CONTEXT = 'Một con lân màu đỏ đang biểu diễn.'
EN = 'a red lion dance costume performing'


def test_extract_context_without_turning_it_into_an_event():
    assert shared_context(QUERY) == CONTEXT
    assert shared_context('Tìm các sự kiện sau: E1: nhảy. E2: cúi.') == ''
    assert shared_context('E1: nhảy. E2: cúi.') == ''
    assert shared_context('một cảnh không có các marker') == ''
    parsed = heuristic_parse(QUERY, 'TRAKE')
    assert len(parsed['trake']['events']) == 2
    assert parsed['trake']['events'][1]['description_vi'] == 'nó quay đầu.'
    assert all('lân màu đỏ' in e['image_pe_queries_en'][0] for e in parsed['trake']['events'])


def test_compose_every_variant_idempotently_and_never_add_context_only_variant():
    parsed = {'trake': {'shared_context_en_visual': EN, 'events': [
        {'description_vi': 'nó nhảy', 'image_pe_queries_en': ['jumping upward', 'leaping high']}]}}
    compose_event_queries(parsed)
    expected = [EN + '. jumping upward', EN + '. leaping high']
    assert parsed['trake']['events'][0]['image_pe_queries_en'] == expected
    compose_event_queries(parsed)
    assert parsed['trake']['events'][0]['image_pe_queries_en'] == expected
    assert EN not in expected
    assert contextual_event_queries({}, {'image_pe_queries_en': ['jumping']}) == ['jumping']
    assert contextual_event_queries(parsed, {}) == []
    # A global visual summary can contain other events, not shared conditions.
    assert contextual_event_queries({'channels': {'image_pe': {'queries_en': ['jumping then bowing']}}},
                                    {'image_pe_queries_en': ['jumping']}) == ['jumping']


@pytest.mark.asyncio
async def test_heuristic_translation_preserves_context_for_each_event(settings, monkeypatch):
    import app.translate
    settings.mock_mode = False
    settings.translate_to_en = True
    translations = {CONTEXT: EN, 'con lân nhảy lên.': 'jumping upward', 'nó quay đầu.': 'turning its head'}
    calls = []
    async def translate(text, **kwargs):
        assert kwargs['settings'] is settings
        calls.append(text)
        return translations.get(text, 'full event sequence'), True
    monkeypatch.setattr(app.translate, 'translate_vi_to_en_status', translate)
    parser = QueryParser(settings)
    parsed = await parser.parse(QUERY, 'TRAKE', use_llm=False)
    assert parsed['trake']['shared_context_en_visual'] == EN
    assert [e['image_pe_queries_en'] for e in parsed['trake']['events']] == [[EN + '. jumping upward'], [EN + '. turning its head']]
    untranslated = await parser.parse(QUERY, 'TRAKE', use_llm=False, translate=False)
    assert all('lân màu đỏ' in e['image_pe_queries_en'][0] for e in untranslated['trake']['events'])
    assert len(calls) == 4  # Whole statement, shared context, and two events.


@pytest.mark.asyncio
async def test_llm_context_is_merged_even_when_event_phrases_omit_it(settings, monkeypatch):
    query = 'Một người đàn ông mặc áo đỏ ở trong nhà bếp. E1: ông ấy lấy bát. E2: sau đó đổ nguyên liệu vào.'
    context_en = 'a man wearing a red shirt in a kitchen'
    data = {'query_type': 'TRAKE', 'visual_en': ['a man cooking'],
            'trake_context': {'description_vi': 'người đàn ông áo đỏ trong bếp', 'visual_en': [context_en]},
            'trake_events': [
                {'description_vi': 'lấy bát', 'visual_en': ['picking up a bowl', 'holding a bowl']},
                {'description_vi': 'đổ nguyên liệu', 'visual_en': ['pouring ingredients into the bowl']} ]}
    routed = to_routing(data, query=query, hint='TRAKE', previous_hints=[])
    assert routed['trake']['shared_context_vi'] == 'Một người đàn ông mặc áo đỏ ở trong nhà bếp.'
    parser = QueryParser(settings)
    monkeypatch.setattr(type(parser.llm), 'available', property(lambda self: True))
    async def parse(*args):
        return copy.deepcopy(routed)
    monkeypatch.setattr(parser, '_llm_parse', parse)
    parsed = await parser.parse(query, 'TRAKE', use_llm=True)
    variants = [q for e in parsed['trake']['events'] for q in e['image_pe_queries_en']]
    assert len(variants) == 3 and all(context_en in q for q in variants)
    assert 'pouring ingredients into the bowl' in variants[-1]
    assert 'resolve pronouns' in PARSER_SYSTEM_PROMPT
    off = await parser.parse(query, 'TRAKE', use_llm=True, translate=False)
    assert off['trake']['shared_context_en_visual'] == ''
    assert all('mặc áo đỏ' in e['image_pe_queries_en'][0] for e in off['trake']['events'])
    assert 'visual_en' not in off['trake']


@pytest.mark.asyncio
async def test_missing_context_translation_retries_instead_of_caching_failure(settings, monkeypatch):
    import app.translate
    settings.mock_mode = False
    settings.translate_to_en = True
    count = 0
    async def translate(text, **kwargs):
        nonlocal count
        if text == CONTEXT:
            count += 1
            return (text, False) if count == 1 else (EN, True)
        return 'an event', True
    monkeypatch.setattr(app.translate, 'translate_vi_to_en_status', translate)
    parser = QueryParser(settings)
    first = await parser.parse(QUERY, 'TRAKE', use_llm=False)
    assert first['translation_failed']
    assert 'lân màu đỏ' in first['trake']['events'][0]['image_pe_queries_en'][0]
    second = await parser.parse(QUERY, 'TRAKE', use_llm=False)
    assert not second.get('translation_failed') and count == 2
    assert EN in second['trake']['events'][0]['image_pe_queries_en'][0]


@pytest.mark.asyncio
async def test_pe_tara_and_pass2_receive_same_context_complete_events(settings, monkeypatch):
    settings.retrieval_database = 'infoshotpp'
    settings.tara_enabled = True
    settings.tara_encoder_url = 'http://mock'
    settings.milvus_endpoint = 'http://mock'
    settings.milvus_token = 'mock'
    service = TrakeService(settings, SearchService(settings))
    parsed = {'original_query': QUERY, 'channels': {}, 'trake': {
        'enabled': True, 'shared_context_vi': CONTEXT, 'shared_context_en_visual': EN,
        'events': [{'event_index': 1, 'image_pe_queries_en': ['jumping upward', 'leaping high']},
                   {'event_index': 2, 'image_pe_queries_en': ['turning its head']} ]}}
    before = copy.deepcopy(parsed)
    pe, tara, pass2 = [], [], []
    async def retrieve(event, **kwargs):
        pe.extend(event['channels']['image_pe']['queries_en'])
        return [], {}
    async def encode(queries):
        tara.extend(queries)
        return [[0.] for _ in queries]
    async def fill(events, *args, **kwargs):
        pass2.extend(e['image_pe_queries_en'][0] for e in events)
    async def expand(value):
        value['trake']['events'][1]['image_pe_queries_en'].append('looking sideways')
    monkeypatch.setattr(service.search, 'retrieve', retrieve)
    monkeypatch.setattr(service.search.tara, 'encode_text', encode)
    monkeypatch.setattr(service.search.milvus, 'search_tara_clips', lambda *args, **kw: [])
    monkeypatch.setattr(service, '_fill_missing_events', fill)
    monkeypatch.setattr(service.search, 'expand_image_queries', expand)
    result = await service.search_trake({'query': QUERY, 'parsed': parsed, 'scope': {'mode': 'all'}, 'expand': True})
    assert len(pe) == 4 and all(EN in q for q in pe)
    assert result['parsed']['trake']['events'][1]['image_pe_queries_en'][-1] == EN + '. looking sideways'
    assert tara == pass2 == [EN + '. jumping upward', EN + '. turning its head']
    assert parsed == before
