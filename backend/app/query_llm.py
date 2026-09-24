"""The query LLM: the routing parser ("LLM" switch) and visual query expansion ("Expand").

Both are one chat-completions call to `settings.query_llm_*`, by default
DeepSeek's `deepseek-flash` (DeepSeek-V4.1-Flash) with thinking switched off —
the operator waits on it under a clock, and routing a query is reading, not
reasoning — and JSON output switched on, so the reply is one parseable object.

The model decides; this module checks. Everything it returns is validated
before it reaches retrieval: a visual phrase that came back in Vietnamese, an
"OCR text" that is really a sentence, a clock that is not a clock, a query type
that contradicts the operator's own choice. A bad field costs its channel, never
the search, and a failed call falls back to the heuristic parser.

The prompt is written for the AIC 2026 final round (HD-ChungKet-2026): Textual
KIS and QA descriptions arrive in parts, TRAKE events arrive together, Video KIS
is the operator's own description of a clip, and a wrong submission costs points.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .adapters.http_pool import PooledHttpClient, failure_reason
from .config import Settings
from .text_normalization import fold_vietnamese

QUERY_TYPES = ("T-KIS", "QA", "V-KIS", "TRAKE", "AVS")

# ---- prompts -----------------------------------------------------------

PARSER_SYSTEM_PROMPT = """You turn one Vietnamese video-retrieval query into search instructions for the AIC 2026 final round. Reply with ONE json object and nothing else.

# The collection (Vietnamese video, searched as keyframes)
- L21, L22: HTV "60 giây" news bulletins. L23: road cycling race (HTV Sports). L24: lion and dragon dance. L25: national high-school exam revision lectures (teacher, slides, formulas). L26: cooking show (chef, ingredients, dishes). L27: Vietnamese culture, craft villages, festivals. L28, L29: Mekong delta life (rivers, boats, floating markets, orchards). L30: short user videos about positive energy, charity, volunteers.
- K01–K20 (BTC profile only): HTV7/HTV9 "60 giây" news bulletins.
- M01–M10 (InfoShot++ only): HTV7 "60 giây" bulletins from 2026: anchor in the studio, field reports, interviews, a red news ticker at the bottom (usually a DIFFERENT story from the picture), HTV7 logo and clock top-right. A few editions are narrated in English.
- N001–N100 (InfoShot++ only): fixed CCTV traffic cameras at Ho Chi Minh City junctions, June 2026: static high-angle view of a street or intersection, motorbikes, cars, buses, trucks, pedestrians, traffic lights, shopfronts; a banner prints the junction name, date and clock. No audio.
- S01 (InfoShot++ only): 2026 HTV Television Cup road cycling race, 12 stages: peloton, breakaways, team cars and escort motorbikes, roadside crowds with flags, finish line, podium and jersey ceremonies; a HUD shows the stage, race clock, speed and distance to go; Vietnamese commentary.

# Task types — "Task type hint" is the operator's choice: keep it unless it is "auto"
- T-KIS: one moment, described in parts revealed over time; later parts add details to the same moment. Search the whole description as one moment.
- V-KIS: the operator describes a clip of at most 20 seconds that they watched, in their own shorthand. If it lists several shots in turn, give one visual phrase per shot: any frame of the clip is a correct answer.
- QA: an event description plus a question to answer from the video. Search for the EVENT. Put the question in question_vi. Whatever the question reveals is on screen (a fire truck, a sign) may go into the search, but never the unknown being asked (its colour, a count, a name).
- TRAKE: several events in time order inside ONE video. Return them in order in trake_events, one entry per event, each with its own short visual phrases. Do not merge or drop events; do not invent events.
- AVS: many different shots of the same kind of scene; describe the shared visual concept.

# Channels
visual_en — ALWAYS 2 or 3 English phrases for image-embedding search (PE-Core reads only ~70 tokens per phrase, so each phrase is at most 20 words).
- Describe only what a camera SEES: people (how many, gender, age, clothing and colours), objects, actions, place, lighting, weather, day or night, shot type (close-up, wide shot, aerial, high-angle CCTV view of a street, TV studio, lecture slide).
- Each phrase restates the WHOLE scene in different words: vary synonyms, word order and which details come first; never split one scene into partial phrases (V-KIS shot lists excepted).
- Use only details the query states or that the source always shows (a CCTV angle for a traffic camera, a studio for an anchor); never invent people, objects or actions.
- Leave out street or junction names, dates, clock times and race-stage numbers (a separate filter handles them), what anyone says, sounds, programme or channel names, what is absent, and the QA question.
- Keep a proper noun only when it is visible (a sign, a logo, a jersey) and pair it with what it looks like.
ocr — text printed on screen: captions, titles, signs, shop names, banners, licence plates, jersey or bib numbers, scoreboards, slides.
- Enable it only when the query says which text is visible (quoted strings, "dòng chữ", "chữ", "biển hiệu", "tiêu đề", "logo", "số áo", "biển số"...).
- texts: the literal strings the query gives, as they would be printed (keep Vietnamese diacritics and case), each at most 8 words, never a description. Never guess what a sign or banner says: if the query mentions one without its words, leave ocr off.
- exact_phrases: strings the query quotes verbatim. numbers: standalone numbers written as digits ("69", "2026").
- clock: a broadcast or banner clock as HH:MM or HH:MM:SS when the query gives one, else null.
speech — what someone SAYS (reporter, interviewee, anchor, teacher, chef, race commentator).
- Enable it only for spoken content ("nói", "phát biểu", "cho biết", "phỏng vấn", "bình luận viên", "thuyết minh", "giới thiệu"...). Traffic cameras have no audio.
- queries_vi: 1 to 3 short phrases in the words a speaker would say, lower-case Vietnamese. The transcript writes numbers as words, so spell them out ("hai nghìn không trăm hai mươi sáu", "ba mươi phần trăm").
- queries_en: only when the narration may be in English (an English-language news edition).
audio — non-speech sounds only (music, applause, car horn, siren, drum, gong, engine, explosion, rain, dog barking). Enable it only when the query itself mentions hearing a sound ("tiếng", "âm thanh", "nhạc"...); never infer a sound from the scene (a lion dance does not imply drums).
- labels_en: AudioSet labels such as "Music", "Applause", "Vehicle horn, car horn, honking", "Siren", "Drum", "Explosion", "Rain", "Dog".

# Output
{"query_type": "T-KIS", "confidence": 0.8, "event_vi": "...", "question_vi": "", "visual_en": ["...", "...", "..."], "ocr": {"enabled": false, "texts": [], "exact_phrases": [], "numbers": [], "clock": null}, "speech": {"enabled": false, "queries_vi": [], "queries_en": []}, "audio": {"enabled": false, "labels_en": []}, "negations": [], "trake_events": [], "note": ""}
- event_vi: the description to search, in Vietnamese, without the QA question and without request words ("Tìm đoạn video...", "Hãy tìm...").
- negations: what the query says is NOT in the scene, in short English.
- trake_events: [] unless query_type is TRAKE; each item is {"description_vi": "...", "visual_en": ["...", "..."], "ocr_texts": [], "speech_vi": []}.
- note: one short English sentence telling the operator which evidence is decisive.
- confidence: 0 to 1, how sure you are of the task type and routing.

# Examples
Task type hint: T-KIS
Description: Camera giao thông tại ngã tư Nguyễn Trãi – Cống Quỳnh lúc khoảng 19 giờ. Một người đàn ông mặc áo mưa màu xanh dương chạy xe máy qua giao lộ, phía sau là quán có biển hiệu Highlands Coffee.
{"query_type": "T-KIS", "confidence": 0.9, "event_vi": "Một người đàn ông mặc áo mưa màu xanh dương chạy xe máy qua giao lộ, phía sau là quán có biển hiệu Highlands Coffee.", "question_vi": "", "visual_en": ["high-angle CCTV view of a city intersection at night, a man in a blue raincoat riding a motorbike", "night street camera: motorbike rider in a blue rain poncho crossing the junction in front of a coffee shop", "traffic camera footage of a crossroads after dark, a man on a scooter wearing a blue raincoat"], "ocr": {"enabled": true, "texts": ["HIGHLANDS COFFEE"], "exact_phrases": [], "numbers": [], "clock": null}, "speech": {"enabled": false, "queries_vi": [], "queries_en": []}, "audio": {"enabled": false, "labels_en": []}, "negations": [], "trake_events": [], "note": "The junction and time go to the camera filter; the rider and the Highlands Coffee sign are the evidence."}

Task type hint: QA
Description: Bản tin 60 giây đưa tin về vụ cháy tại một nhà xưởng, phóng viên đứng trước hiện trường cho biết lửa bùng lên lúc rạng sáng. Hỏi: xe cứu hỏa trong video có màu gì?
{"query_type": "QA", "confidence": 0.85, "event_vi": "Bản tin 60 giây đưa tin về vụ cháy tại một nhà xưởng, phóng viên đứng trước hiện trường cho biết lửa bùng lên lúc rạng sáng.", "question_vi": "Xe cứu hỏa trong video có màu gì?", "visual_en": ["news reporter standing in front of a burning factory with thick smoke", "firefighters and fire trucks at a large warehouse fire", "TV news footage of a factory on fire, flames and black smoke"], "ocr": {"enabled": false, "texts": [], "exact_phrases": [], "numbers": [], "clock": null}, "speech": {"enabled": true, "queries_vi": ["lửa bùng lên lúc rạng sáng", "vụ cháy nhà xưởng"], "queries_en": []}, "audio": {"enabled": false, "labels_en": []}, "negations": [], "trake_events": [], "note": "Find the fire report; the fire truck is visible, its colour is the answer."}

Task type hint: TRAKE
Description: Chặng 6 Cúp Truyền hình. E1: tay đua mặc áo vàng bứt lên khỏi nhóm dẫn đầu. E2: bình luận viên nói tay đua này sắp giành chiến thắng chặng. E3: tay đua giơ hai tay qua vạch đích.
{"query_type": "TRAKE", "confidence": 0.9, "event_vi": "Tay đua áo vàng bứt lên, được bình luận viên dự đoán chiến thắng chặng và giơ hai tay qua vạch đích.", "question_vi": "", "visual_en": ["cyclist in a yellow jersey attacking ahead of the leading group on a road race", "road race cyclist in yellow sprinting away and crossing the finish line with arms raised"], "ocr": {"enabled": false, "texts": [], "exact_phrases": [], "numbers": [], "clock": null}, "speech": {"enabled": true, "queries_vi": ["giành chiến thắng chặng"], "queries_en": []}, "audio": {"enabled": false, "labels_en": []}, "negations": [], "trake_events": [{"description_vi": "Tay đua mặc áo vàng bứt lên khỏi nhóm dẫn đầu.", "visual_en": ["cyclist in a yellow jersey breaking away from the leading group", "rider in yellow attacking ahead of the peloton on a road"], "ocr_texts": [], "speech_vi": []}, {"description_vi": "Bình luận viên nói tay đua sắp giành chiến thắng chặng.", "visual_en": ["cyclist in a yellow jersey riding alone toward the finish of a road race"], "ocr_texts": [], "speech_vi": ["giành chiến thắng chặng"]}, {"description_vi": "Tay đua giơ hai tay qua vạch đích.", "visual_en": ["cyclist raising both arms while crossing the finish line", "road race winner celebrating with arms up at the finish banner"], "ocr_texts": [], "speech_vi": []}], "note": "Stage 6 goes to the race filter; the three moments must come in this order in one video."}"""


EXPAND_SYSTEM_PROMPT = """You write alternative English descriptions of one video keyframe for an image-embedding search (PE-Core and Qwen3-VL) over Vietnamese TV news, fixed street traffic cameras, a road cycling race livestream, cooking, culture and documentary programmes. Reply with ONE json object: {"variants": ["...", "...", "..."]}.
Rules:
- Each variant describes the SAME complete scene as the input, worded differently: synonyms, a different order, the concrete visual details a camera records (people and clothing colours, objects, actions, setting, lighting, day or night, shot type).
- 8 to 20 words each; never more than 25 (the image encoder reads about 70 tokens).
- Visible things only: no street or junction names, dates, clock times, spoken words, sounds, or things said to be absent.
- Pair a named person, brand or character with what it looks like.
- Never contradict the input. You may add the typical look of the source (e.g. "high-angle CCTV view" for a traffic camera, "TV studio" for a news anchor).
Example input: a man in a blue raincoat riding a motorbike through an intersection at night
Example output: {"variants": ["night CCTV view of a junction, motorbike rider wearing a blue rain poncho", "man on a scooter in a blue raincoat crossing a city intersection after dark", "high-angle street camera at night: motorcyclist in blue rainwear passing through traffic"]}"""


# ---- transport ----------------------------------------------------------


class QueryLlmError(RuntimeError):
    """The query LLM could not answer; the caller falls back to heuristics."""


class QueryLlm:
    """One pooled OpenAI-compatible client for the query LLM."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._http = PooledHttpClient()

    @property
    def available(self) -> bool:
        return bool(self.s.has_llm and not self.s.mock_mode)

    async def chat_json(self, system: str, user: str, *, max_tokens: int) -> dict[str, Any]:
        """One JSON-mode completion; retried once, because JSON mode may come back empty."""
        body: dict[str, Any] = {
            "model": self.s.query_llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if "deepseek" in self.s.query_llm_base_url:
            # V4.1 thinks by default; a routing decision does not need it and the
            # console is waiting.
            body["thinking"] = {"type": "disabled"}
        headers = {"Authorization": f"Bearer {self.s.query_llm_api_key}", "Content-Type": "application/json"}
        url = f"{self.s.query_llm_base_url.rstrip('/')}/chat/completions"
        last = "empty reply"
        for _ in range(2):
            try:
                resp = await self._http.get().post(
                    url, json=body, headers=headers, timeout=self.s.query_llm_timeout_seconds
                )
                resp.raise_for_status()
                content = (resp.json()["choices"][0]["message"].get("content") or "").strip()
            except Exception as exc:  # noqa: BLE001 - reported to the caller as one reason
                last = failure_reason(exc)
                continue
            data = extract_json_object(content)
            if data is not None:
                return data
            last = "reply was not a JSON object"
        raise QueryLlmError(last)


def extract_json_object(content: str) -> dict[str, Any] | None:
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content).rstrip("`").rstrip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


# ---- validation ---------------------------------------------------------

_CLOCK = re.compile(r"([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_VI_LETTERS = set("ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ")
#: A literal on-screen string a person types out is short; see OCR_MAX_LITERAL_TOKENS.
OCR_MAX_WORDS = 12
MAX_VISUAL_PHRASES = 4
MAX_TRAKE_EVENTS = 8


def _strings(value: Any, *, limit: int, max_words: int | None = None) -> list[str]:
    items = value if isinstance(value, list) else ([value] if isinstance(value, str) else [])
    out: list[str] = []
    for item in items:
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            continue
        text = " ".join(str(item).split())
        if not text or text in out:
            continue
        if max_words is not None and len(_WORD.findall(text)) > max_words:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def mostly_vietnamese(text: str) -> bool:
    """A phrase the image encoders would read as Vietnamese, not a visible proper noun in English."""
    words = _WORD.findall(text.lower())
    accented = sum(1 for word in words if _VI_LETTERS & set(word))
    return bool(words) and accented / len(words) > 0.3


def english_phrases(value: Any, *, limit: int = MAX_VISUAL_PHRASES) -> list[str]:
    return [phrase for phrase in _strings(value, limit=limit * 2) if not mostly_vietnamese(phrase)][:limit]


def _numbers(value: Any) -> list[str]:
    return [text for text in _strings(value, limit=8) if text.isdigit()]


def _clock(value: Any) -> tuple[str | None, int | None]:
    if not isinstance(value, str):
        return None, None
    match = _CLOCK.fullmatch(value.strip())
    if not match:
        return None, None
    return match.group(0), int(match.group(1))


def _flag(block: dict[str, Any]) -> bool:
    return bool(block.get("enabled"))


def to_routing(
    data: dict[str, Any], *, query: str, hint: str, previous_hints: list[str]
) -> dict[str, Any] | None:
    """The LLM's JSON as the routing dict the rest of the app reads, or None if unusable.

    Returns None when no English visual phrase survived: the visual channel is the
    backbone of every search, and the heuristic path (with translation) does it
    better than an empty one.
    """
    combined = " ".join([*previous_hints, query]).strip()
    query_type = hint if hint in QUERY_TYPES else data.get("query_type")
    if query_type not in QUERY_TYPES:
        query_type = "T-KIS"
    visual = english_phrases(data.get("visual_en"))
    if not visual:
        return None
    event_vi = _strings(data.get("event_vi"), limit=1)
    question_vi = _strings(data.get("question_vi"), limit=1)
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence") or 0.6)))
    except (TypeError, ValueError):
        confidence = 0.6

    ocr = data.get("ocr") if isinstance(data.get("ocr"), dict) else {}
    ocr_texts = _strings(ocr.get("texts"), limit=6, max_words=OCR_MAX_WORDS)
    exact = _strings(ocr.get("exact_phrases"), limit=6, max_words=OCR_MAX_WORDS)
    numbers = _numbers(ocr.get("numbers"))
    clock, hour = _clock(ocr.get("clock"))
    ocr_on = _flag(ocr) and bool(ocr_texts or exact or numbers or clock)

    speech = data.get("speech") if isinstance(data.get("speech"), dict) else {}
    speech_queries = _strings(
        [*(speech.get("queries_vi") or []), *(speech.get("queries_en") or [])], limit=5, max_words=25
    )
    speech_on = _flag(speech) and bool(speech_queries)

    audio = data.get("audio") if isinstance(data.get("audio"), dict) else {}
    labels = english_phrases(audio.get("labels_en"), limit=4)
    audio_on = _flag(audio) and bool(labels)

    events: list[dict[str, Any]] = []
    if query_type == "TRAKE":
        raw_events = data.get("trake_events") if isinstance(data.get("trake_events"), list) else []
        for raw in raw_events[:MAX_TRAKE_EVENTS]:
            if not isinstance(raw, dict):
                continue
            event_visual = english_phrases(raw.get("visual_en"), limit=3)
            if not event_visual:
                continue
            description = _strings(raw.get("description_vi"), limit=1)
            events.append({
                "event_index": len(events) + 1,
                "description_vi": description[0] if description else "",
                "description_en_visual": event_visual[0],
                "image_pe_queries_en": event_visual,
                "ocr_queries_vi": _strings(raw.get("ocr_texts"), limit=3, max_words=OCR_MAX_WORDS),
                "speech_queries_vi": _strings(raw.get("speech_vi"), limit=3, max_words=25),
                "audio_queries_en": [],
                "expected_order_hint": "before" if not events else "after",
                "required": True,
            })
        if not events:
            # A TRAKE statement the model could not split still has to be searched.
            events.append({
                "event_index": 1, "description_vi": event_vi[0] if event_vi else combined,
                "description_en_visual": visual[0], "image_pe_queries_en": visual, "ocr_queries_vi": [],
                "speech_queries_vi": [], "audio_queries_en": [], "expected_order_hint": "before", "required": True,
            })

    def channel(enabled: bool, weight: float, reason: str, **fields: Any) -> dict[str, Any]:
        return {"enabled": enabled, "weight": weight if enabled else 0.0, "reason": reason, **fields}

    ocr_queries = ocr_texts
    note = _strings(data.get("note"), limit=1)
    return {
        "query_type": query_type,
        "confidence": confidence,
        "original_query": query,
        "normalized_vi": event_vi[0] if event_vi else combined,
        "translated_en_visual": visual[0],
        "operator_summary_vi": note[0] if note else "",
        "channels": {
            "image_pe": channel(
                True, 1.0, "Visual scene (LLM).",
                queries_en=visual, positive_concepts_en=[], negative_concepts_en=[],
            ),
            "ocr": channel(
                ocr_on, 1.0 if exact else 0.8, "On-screen text named in the query (LLM)." if ocr_on else "",
                queries_vi=ocr_queries, queries_folded=[fold_vietnamese(text) for text in ocr_queries],
                exact_phrases=exact, numbers=numbers, time_filters={"hour": hour, "clock": clock},
            ),
            "speech": channel(
                speech_on, 0.8, "Spoken content (LLM)." if speech_on else "",
                queries_vi=speech_queries, exact_phrases=[], speaker_or_role_hints=[],
            ),
            "audio": channel(
                audio_on, 0.7, "Non-speech sound (LLM)." if audio_on else "",
                queries_en=labels, sound_labels_en=labels,
            ),
        },
        "filters": {
            "video_ids": [], "categories": [], "time_range_seconds": None, "must_include": [],
            "must_not_include": english_phrases(data.get("negations"), limit=6),
        },
        "trake": {
            "enabled": query_type == "TRAKE",
            "events": events,
            "ordering_rule": "strict_increasing_time",
            "fallback_policy": "allow_partial_confident_events_with_warning",
        },
        "qa": {
            "enabled": query_type == "QA",
            "question_vi": question_vi[0] if question_vi else (combined if query_type == "QA" else ""),
            "answer_evidence_channels": ["ocr", "speech", "audio"],
            "draft_answer_allowed_from_text_evidence_only": True,
        },
        "rerank_policy": {"use_rrf": True, "rrf_k": 60, "group_by_video": True, "prefer_temporal_cooccurrence": True},
        "ui_hints": {
            "show_timeline": query_type == "TRAKE",
            "show_ocr_snippets": ocr_on,
            "show_speech_snippets": speech_on,
            "show_audio_badges": audio_on,
            "require_submit_guard": True,
            "warning_vi": "",
        },
    }


def parser_user_message(query: str, hint: str, previous_hints: list[str], retrieval_database: str) -> str:
    profile = (
        "InfoShot++ (L21–L30, M01–M10, N001–N100, S01)"
        if retrieval_database == "infoshotpp"
        else "BTC (L21–L30, K01–K20)"
    )
    parts = [part for part in [*previous_hints, query] if part and part.strip()]
    if len(parts) <= 1:
        description = f"Description: {parts[0] if parts else ''}"
    else:
        numbered = "\n".join(f"{index}. {part.strip()}" for index, part in enumerate(parts, start=1))
        description = f"Description (parts in the order they were revealed):\n{numbered}"
    return f"Task type hint: {hint}\nProfile: {profile}\n{description}\nReturn the json object."


def expand_user_message(scene_en: str, original_vi: str | None) -> str:
    lines = [f"Scene: {scene_en.strip()}"]
    if original_vi and original_vi.strip() and original_vi.strip() != scene_en.strip():
        lines.append(f"Original query (Vietnamese, for meaning only): {original_vi.strip()}")
    lines.append("Return the json object.")
    return "\n".join(lines)


def expand_variants(data: dict[str, Any], *, existing: list[str], n: int) -> list[str]:
    seen = {phrase.lower() for phrase in existing}
    out: list[str] = []
    for phrase in english_phrases(data.get("variants"), limit=n * 2):
        if len(_WORD.findall(phrase)) > 30 or phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        out.append(phrase)
    return out[:n]
