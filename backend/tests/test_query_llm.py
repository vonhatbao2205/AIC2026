"""The query LLM (DeepSeek): request shape, retries, and validation of what it returns."""
import json

import pytest

from app.query_llm import (
    QueryLlm,
    QueryLlmError,
    expand_variants,
    mostly_vietnamese,
    parser_user_message,
    to_routing,
)
from app.query_parser import QueryParser


def live(settings):
    settings.mock_mode = False
    settings.query_llm_api_key = "sk-test"
    settings.query_llm_base_url = "https://api.deepseek.com"
    settings.query_llm_model = "deepseek-flash"
    return settings


class FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply)


def wire(llm: QueryLlm, replies):
    client = FakeClient(replies)
    llm._http.get = lambda: client
    return client


ROUTED = {
    "query_type": "T-KIS",
    "confidence": 0.9,
    "event_vi": "Một người đàn ông mặc áo mưa xanh chạy xe máy qua giao lộ.",
    "question_vi": "",
    "visual_en": ["high-angle CCTV view of a junction at night, man in a blue raincoat on a motorbike", "night street camera, rider in blue rain poncho"],
    "ocr": {"enabled": True, "texts": ["HIGHLANDS COFFEE"], "exact_phrases": [], "numbers": ["69", "abc"], "clock": "19:05"},
    "speech": {"enabled": False, "queries_vi": [], "queries_en": []},
    "audio": {"enabled": False, "labels_en": []},
    "negations": ["no rain"],
    "trake_events": [],
    "note": "Rider and sign decide it.",
}


@pytest.mark.asyncio
async def test_request_is_deepseek_json_mode_without_thinking(settings):
    llm = QueryLlm(live(settings))
    client = wire(llm, [json.dumps({"ok": 1})])
    assert await llm.chat_json("system json", "user", max_tokens=100) == {"ok": 1}
    call = client.calls[0]
    assert call["url"] == "https://api.deepseek.com/chat/completions"
    assert call["json"]["model"] == "deepseek-flash"
    assert call["json"]["thinking"] == {"type": "disabled"}
    assert call["json"]["response_format"] == {"type": "json_object"}
    assert call["headers"]["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_empty_json_reply_is_retried_then_reported(settings):
    llm = QueryLlm(live(settings))
    wire(llm, ["", json.dumps({"ok": 2})])
    assert await llm.chat_json("s", "u", max_tokens=10) == {"ok": 2}
    wire(llm, ["", "not json"])
    with pytest.raises(QueryLlmError, match="not a JSON object"):
        await llm.chat_json("s", "u", max_tokens=10)


def test_routing_is_validated():
    routed = to_routing(ROUTED, query="q", hint="auto", previous_hints=[])
    ocr = routed["channels"]["ocr"]
    assert ocr["enabled"] and ocr["queries_vi"] == ["HIGHLANDS COFFEE"]
    assert ocr["queries_folded"] == ["highlands coffee"]
    assert ocr["numbers"] == ["69"]  # "abc" is not a number
    assert ocr["time_filters"] == {"hour": 19, "clock": "19:05"}
    assert routed["normalized_vi"] == ROUTED["event_vi"]
    assert routed["translated_en_visual"] == ROUTED["visual_en"][0]
    assert routed["filters"]["must_not_include"] == ["no rain"]
    assert routed["channels"]["speech"]["enabled"] is False


def test_operator_hint_wins_and_unusable_fields_are_dropped():
    data = {
        **ROUTED,
        "query_type": "TRAKE",
        "ocr": {"enabled": True, "texts": ["một câu mô tả rất dài không phải chữ trên màn hình mà là lời kể dài dòng"], "clock": "25:99"},
        "audio": {"enabled": True, "labels_en": []},
    }
    routed = to_routing(data, query="q", hint="QA", previous_hints=[])
    assert routed["query_type"] == "QA"
    assert routed["channels"]["ocr"]["enabled"] is False  # a sentence is not on-screen text, 25:99 is not a clock
    assert routed["channels"]["audio"]["enabled"] is False  # enabled without a label
    assert routed["qa"]["question_vi"] == "q"  # no question returned: the query itself


def test_vietnamese_visual_phrases_are_rejected():
    assert mostly_vietnamese("một người đàn ông mặc áo xanh")
    assert not mostly_vietnamese("a shop sign reading Phở Hòa above the door")
    assert to_routing({**ROUTED, "visual_en": ["một người đàn ông"]}, query="q", hint="auto", previous_hints=[]) is None


def test_trake_events_in_order_with_a_fallback():
    data = {
        **ROUTED,
        "query_type": "TRAKE",
        "trake_events": [
            {"description_vi": "E1", "visual_en": ["cyclist breaking away"], "speech_vi": ["bứt phá"]},
            {"description_vi": "E2", "visual_en": ["một câu tiếng việt"]},  # no usable phrase
            {"description_vi": "E3", "visual_en": ["arms raised at the finish line"]},
        ],
    }
    events = to_routing(data, query="q", hint="auto", previous_hints=[])["trake"]["events"]
    assert [e["event_index"] for e in events] == [1, 2]
    assert events[0]["speech_queries_vi"] == ["bứt phá"]
    assert events[1]["image_pe_queries_en"] == ["arms raised at the finish line"]
    single = to_routing({**ROUTED, "trake_events": []}, query="q", hint="TRAKE", previous_hints=[])["trake"]["events"]
    assert len(single) == 1 and single[0]["image_pe_queries_en"] == ROUTED["visual_en"]


def test_progressive_parts_are_numbered_in_order():
    message = parser_user_message("phần hai", "T-KIS", ["phần một"], "infoshotpp")
    assert "1. phần một\n2. phần hai" in message
    assert "InfoShot++" in message and message.endswith("Return the json object.")
    assert "Description: câu" in parser_user_message("câu", "QA", [], "btc")


def test_expand_variants_are_english_new_and_short():
    data = {"variants": ["night CCTV view of a junction", "night CCTV view of a junction", "một câu tiếng việt dài", "word " * 40, "rider in a blue poncho"]}
    assert expand_variants(data, existing=["Night CCTV view of a junction"], n=3) == ["rider in a blue poncho"]


@pytest.mark.asyncio
async def test_parser_uses_the_llm_and_falls_back_with_a_warning(settings):
    parser = QueryParser(live(settings))
    wire(parser.llm, [json.dumps(ROUTED)])
    parsed = await parser.parse("một người đàn ông", use_llm=True)
    assert parsed["_engine"] == "llm:deepseek-flash"
    assert parsed["channels"]["image_pe"]["queries_en"] == ROUTED["visual_en"]

    parser = QueryParser(live(settings))
    wire(parser.llm, [RuntimeError("timeout"), RuntimeError("timeout")])
    parsed = await parser.parse("một người đàn ông khác", use_llm=True, translate=False)
    assert parsed["_engine"] == "heuristic"
    assert "LLM parser unavailable (timeout)" in parsed["ui_hints"]["warning_vi"]


@pytest.mark.asyncio
async def test_expand_is_cached(settings):
    parser = QueryParser(live(settings))
    client = wire(parser.llm, [json.dumps({"variants": ["rider in a blue poncho at night"]})])
    first = await parser.expand_visual("a man in a blue raincoat", original_vi="người đàn ông áo mưa xanh")
    again = await parser.expand_visual("a man in a blue raincoat", original_vi="người đàn ông áo mưa xanh")
    assert first == again == ["rider in a blue poncho at night"]
    assert len(client.calls) == 1
    assert "Original query (Vietnamese, for meaning only)" in client.calls[0]["json"]["messages"][1]["content"]
