"""Query understanding + retrieval routing.

Primary path: NVIDIA Nemotron (OpenAI-compatible chat.completions) returns the
strict routing JSON schema. Fallback: deterministic heuristics so the system
works without an LLM. After either path, `manual_overrides` are applied and at
least `image_pe` is guaranteed enabled.

Secrets (NVIDIA_API_KEY) never leave the backend.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any

import httpx

from .config import Settings
from .text_normalization import fold_vietnamese

# ---- Heuristic signal lexicons (Vietnamese) ---------------------------
# Words that say the operator is describing text ON SCREEN. Matched on word
# boundaries: plain `in` would fire "tên" inside unrelated words.
_OCR_HINT_WORDS = [
    "chữ", "dòng chữ", "hàng chữ", "chữ số", "biển", "biển số", "biển hiệu",
    "logo", "thương hiệu", "nhãn hiệu", "tiêu đề", "tên", "số điện thoại",
    "con số", "số hiệu", "hiển thị", "ghi", "đề",
]
_OCR_HINT_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(word) for word in sorted(_OCR_HINT_WORDS, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE | re.UNICODE,
)
# "số 69", "số 7" — a number the operator is naming, not counting with. Bare
# "số" is excluded on purpose: "một số người" means "some people".
_OCR_NUMBER_CUE_RE = re.compile(r"(?<!\w)số\s*\d", re.IGNORECASE | re.UNICODE)


def has_ocr_cue(text: str) -> bool:
    """Does the query claim there is readable text in the frame?"""
    return bool(_OCR_HINT_RE.search(text) or _OCR_NUMBER_CUE_RE.search(text))
_SPEECH_HINT_WORDS = ["nói", "phát biểu", "tuyên bố", "phỏng vấn", "trả lời", "nhắc đến", "thông báo", "đọc", "kể", "phóng viên", "dẫn chương trình"]
_AUDIO_HINT_WORDS = ["tiếng", "âm thanh", "nhạc", "piano", "trống", "còi", "chuông", "nổ", "súng", "vỗ tay", "mưa", "sủa", "động cơ", "trực thăng"]
_AUDIO_LABELS = {
    "nhạc": "Music", "piano": "Piano", "trống": "Drum", "còi": "Vehicle horn",
    "chuông": "Bell", "nổ": "Explosion", "súng": "Gunshot", "vỗ tay": "Applause",
    "mưa": "Rain", "sủa": "Dog", "động cơ": "Engine", "trực thăng": "Helicopter",
    "music": "Music",
}
_TRAKE_CONNECTORS = ["sau đó", "rồi", "tiếp theo", "sau khi", "trước khi", "tiếp đến", "cuối cùng", "đầu tiên", "then", "after that", "next"]
_QA_MARKERS = ["?", "bao nhiêu", "là gì", "ở đâu", "khi nào", "ai", "tại sao", "màu gì", "mấy"]
_CLOCK_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b")
_QUOTED_RE = re.compile(r"[\"“”'`]([^\"“”'`]{2,}?)[\"“”'`]")
_OCR_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# A literal on-screen string a person types out is short: a channel bug, a
# headline fragment, a score, a speed readout. `ElasticClient.search_ocr` turns
# `queries_vi` into an `operator: and` clause, so handing it a descriptive
# sentence asks Elasticsearch for one frame containing every word of it.
# Measured against the InfoShot++ OCR corpus: the 28-token query "Đoạn video
# không chuyển cảnh … 69 km/h" matches 0 of 779,995 frames (best overlap 21/28,
# on an unrelated frame), while its one piece of real screen text — "69" — is
# present on the answer frames. Past this many tokens the query is prose about a
# scene rather than text on screen, so only the structured evidence is searched.
OCR_MAX_LITERAL_TOKENS = 12


def _empty_channel(name: str) -> dict[str, Any]:
    base = {"enabled": False, "weight": 0.0, "reason": ""}
    if name == "image_pe":
        base.update(queries_en=[], positive_concepts_en=[], negative_concepts_en=[])
    elif name == "ocr":
        base.update(queries_vi=[], queries_folded=[], exact_phrases=[], numbers=[], time_filters={"hour": None, "clock": None})
    elif name == "speech":
        base.update(queries_vi=[], exact_phrases=[], speaker_or_role_hints=[])
    elif name == "audio":
        base.update(queries_en=[], sound_labels_en=[])
    return base


def _fold(text: str) -> str:
    return fold_vietnamese(text)


def ocr_evidence(text: str, *, fallback_to_full_text: bool = False) -> dict[str, Any]:
    """Pull out the parts of a query that could plausibly be printed on a frame.

    Returns the OCR channel's evidence fields. Quoted spans, standalone numbers
    and a broadcast clock are things a frame can literally contain; the prose
    around them is not, and forwarding it as `queries_vi` would make the OCR
    search require every word of the sentence (see `OCR_MAX_LITERAL_TOKENS`).

    `fallback_to_full_text` is for a channel the operator forced on: with no
    structured evidence at all, searching the text is still better than a forced
    channel that quietly does nothing.
    """
    text = text.strip()
    clock_match = _CLOCK_RE.search(text)
    clock = clock_match.group(0) if clock_match else None
    hour = int(clock_match.group(1)) if clock_match else None

    exact_phrases: list[str] = []
    for match in _QUOTED_RE.finditer(text):
        phrase = match.group(1).strip()
        if phrase and phrase not in exact_phrases:
            exact_phrases.append(phrase)

    # Clock digits are searched through `time_filters`, and a digit welded to a
    # code ("VTV3") belongs to that code — neither is a number to look for alone.
    clock_digits = set(re.findall(r"\d+", clock or ""))
    numbers: list[str] = []
    for number in re.findall(r"\d+", text):
        if number in clock_digits or number in numbers:
            continue
        if re.search(rf"(?<!\w){re.escape(number)}(?!\w)", text):
            numbers.append(number)

    # The clock is spent through its own filter, so it must not eat the budget
    # that decides whether the rest of the query reads as literal screen text.
    without_clock = text.replace(clock, " ") if clock else text
    literal = len(_OCR_TOKEN_RE.findall(without_clock)) <= OCR_MAX_LITERAL_TOKENS
    if exact_phrases:
        # Quotes already say which span is on screen; the rest is context.
        literal = False
    elif not literal and fallback_to_full_text and not numbers and not clock:
        literal = True

    queries_vi = [text] if literal and text else []
    return {
        "queries_vi": queries_vi,
        "queries_folded": [_fold(query) for query in queries_vi],
        "exact_phrases": exact_phrases,
        "numbers": numbers,
        "time_filters": {"hour": hour, "clock": clock},
    }


def heuristic_parse(
    query: str,
    query_type_hint: str = "auto",
    previous_hints: list[str] | None = None,
) -> dict[str, Any]:
    previous_hints = previous_hints or []
    combined = " ".join([*previous_hints, query]).strip()
    low = combined.lower()

    # A digit alone is NOT evidence of on-screen text. "Có 1 người mặc áo trắng
    # ngồi giữa 2 người mặc áo đen" describes a scene, yet it used to put the
    # two commonest tokens in the whole OCR corpus into a hard filter — measured
    # at 2-16 s per search on a channel that could not contribute anything.
    # OCR now needs an explicit signal: a quoted span, a cue word, a named
    # number ("số 69"), or a broadcast clock.
    ocr_likely = (
        has_ocr_cue(combined)
        or bool(_QUOTED_RE.search(combined))
        or bool(_CLOCK_RE.search(combined))
    )
    speech_likely = any(w in low for w in _SPEECH_HINT_WORDS)
    audio_likely = any(w in low for w in _AUDIO_HINT_WORDS)
    is_trake = query_type_hint == "TRAKE" or any(c in low for c in _TRAKE_CONNECTORS)
    is_qa = query_type_hint == "QA" or any(m in low for m in _QA_MARKERS)

    if query_type_hint in {"T-KIS", "QA", "V-KIS", "TRAKE"}:
        query_type = query_type_hint
    elif is_trake:
        query_type = "TRAKE"
    elif is_qa:
        query_type = "QA"
    else:
        query_type = "T-KIS"

    channels = {name: _empty_channel(name) for name in ["image_pe", "ocr", "speech", "audio"]}

    # image_pe: almost always on.
    channels["image_pe"].update(
        enabled=True,
        weight=1.0,
        reason="Default visual semantic channel (heuristic).",
        queries_en=[combined],
    )

    if ocr_likely:
        evidence = ocr_evidence(combined)
        # A cue word can flag a query as OCR-ish while leaving nothing a frame
        # could actually contain. Enabling the channel then would cost a request
        # and return nothing, so route on the evidence, not on the cue.
        if any(
            (
                evidence["queries_vi"],
                evidence["exact_phrases"],
                evidence["numbers"],
                evidence["time_filters"]["clock"],
            )
        ):
            channels["ocr"].update(
                enabled=True,
                weight=0.8,
                reason="Query has digits/quotes/text cues.",
                **evidence,
            )
    if speech_likely:
        channels["speech"].update(
            enabled=True, weight=0.8, reason="Query refers to spoken content.", queries_vi=[combined]
        )
    if audio_likely:
        labels = sorted({lbl for w, lbl in _AUDIO_LABELS.items() if w in low})
        channels["audio"].update(
            enabled=True,
            weight=0.7,
            reason="Query refers to non-speech sound.",
            queries_en=labels or [combined],
            sound_labels_en=labels,
        )

    trake = {"enabled": False, "events": [], "ordering_rule": "strict_increasing_time", "fallback_policy": "allow_partial_confident_events_with_warning"}
    if query_type == "TRAKE":
        events = _split_trake_events(query)
        trake["enabled"] = True
        trake["events"] = [
            {
                "event_index": i + 1,
                "description_vi": ev,
                "description_en_visual": ev,
                "image_pe_queries_en": [ev],
                "ocr_queries_vi": [],
                "speech_queries_vi": [],
                "audio_queries_en": [],
                "expected_order_hint": "before" if i == 0 else "after",
                "required": True,
            }
            for i, ev in enumerate(events)
        ]

    return {
        "query_type": query_type,
        "confidence": 0.4,
        "original_query": query,
        "normalized_vi": combined,
        "translated_en_visual": combined,
        "operator_summary_vi": f"Heuristic routing cho query: {query}",
        "channels": channels,
        "filters": {"video_ids": [], "categories": [], "time_range_seconds": None, "must_include": [], "must_not_include": []},
        "trake": trake,
        "qa": {
            "enabled": query_type == "QA",
            "question_vi": query if query_type == "QA" else "",
            "answer_evidence_channels": ["ocr", "speech", "audio"],
            "draft_answer_allowed_from_text_evidence_only": True,
        },
        "rerank_policy": {"use_rrf": True, "rrf_k": 60, "group_by_video": True, "prefer_temporal_cooccurrence": True},
        "ui_hints": {
            "show_timeline": query_type == "TRAKE",
            "show_ocr_snippets": channels["ocr"]["enabled"],
            "show_speech_snippets": speech_likely,
            "show_audio_badges": audio_likely,
            "require_submit_guard": True,
            "warning_vi": "",
        },
        "_engine": "heuristic",
    }


_EVENT_MARKER = re.compile(r"(?:^|[\s,;])(?:E\s*\d+|[Ss]ự\s*kiện\s*\d+|[Ee]vent\s*\d+|\d+)\s*[:.)]", re.IGNORECASE)


def _split_trake_events(query: str) -> list[str]:
    # 1) Explicit event markers: "E1: ... E2: ...", "sự kiện 1:", "1) ...".
    marks = list(_EVENT_MARKER.finditer(query))
    if len(marks) >= 2:
        parts: list[str] = []
        for i, m in enumerate(marks):
            start = m.end()
            end = marks[i + 1].start() if i + 1 < len(marks) else len(query)
            txt = query[start:end].strip(" :.)\n\t,;-")
            if txt:
                parts.append(txt)
        if len(parts) >= 2:
            return parts
    # 2) Sequence connectors ("sau đó", "rồi", ...).
    pattern = "|".join(re.escape(c) for c in _TRAKE_CONNECTORS)
    parts = [p.strip(" ,.;") for p in re.split(pattern, query, flags=re.IGNORECASE) if p.strip(" ,.;")]
    return parts if len(parts) >= 2 else [query]


# ---- LLM path ---------------------------------------------------------
SYSTEM_PROMPT = """You are the query understanding and retrieval routing engine for an AIC26 Vietnamese video retrieval system.
Parse a user query, translate it when needed, classify the task type, decide which retrieval channels should run, and produce structured instructions for the backend.
Channels: image_pe (PE-Core-G14 visual semantic, needs English visual query), ocr (Elastic OCR text, Vietnamese + folded), speech (Elastic ASR transcript, Vietnamese), audio (Elastic audio event tags/caption, English sound labels), trake_sequence (ordered events in one video).
Prefer recall: enable image_pe for almost every visual query. Enable OCR only when visible text matters; speech only for spoken content; audio only for non-speech sounds. Parse negations as filters/warnings, do not over-trust them in vector search. Always output valid JSON only, no markdown, no commentary.

QUERY EXPANSION (important for recall): for image_pe.queries_en, output 2-3 SHORT, DIVERSE English visual rephrasings of the same target — describe what is literally VISIBLE (objects, colors, shapes, scene), not named entities alone. A named character must be paired with a visual description (e.g. for "The Thing" use ["The Thing superhero","orange rocky stone-skinned muscular man","brown cracked rock-textured humanoid"]). Avoid one long sentence; use compact phrases. Do the same for each trake event's image_pe_queries_en."""

USER_TEMPLATE = """Parse this AIC26 retrieval query.

User query:
{query}

Query type hint:
{hint}

Previous accumulated hints, if any:
{previous}

Return JSON only using exactly this schema:
{schema}"""

_SCHEMA_HINT = json.dumps(
    {
        "query_type": "T-KIS|QA|V-KIS|TRAKE",
        "confidence": 0.0,
        "original_query": "",
        "normalized_vi": "",
        "translated_en_visual": "",
        "operator_summary_vi": "",
        "channels": {
            "image_pe": {"enabled": True, "weight": 1.0, "reason": "", "queries_en": [], "positive_concepts_en": [], "negative_concepts_en": []},
            "ocr": {"enabled": False, "weight": 0.0, "reason": "", "queries_vi": [], "queries_folded": [], "exact_phrases": [], "numbers": [], "time_filters": {"hour": None, "clock": None}},
            "speech": {"enabled": False, "weight": 0.0, "reason": "", "queries_vi": [], "exact_phrases": [], "speaker_or_role_hints": []},
            "audio": {"enabled": False, "weight": 0.0, "reason": "", "queries_en": [], "sound_labels_en": []},
        },
        "filters": {"video_ids": [], "categories": [], "time_range_seconds": None, "must_include": [], "must_not_include": []},
        "trake": {"enabled": False, "events": [], "ordering_rule": "strict_increasing_time", "fallback_policy": "allow_partial_confident_events_with_warning"},
        "qa": {"enabled": False, "question_vi": "", "answer_evidence_channels": ["ocr", "speech", "audio"], "draft_answer_allowed_from_text_evidence_only": True},
        "rerank_policy": {"use_rrf": True, "rrf_k": 60, "group_by_video": True, "prefer_temporal_cooccurrence": True},
        "ui_hints": {"show_timeline": False, "show_ocr_snippets": False, "show_speech_snippets": False, "show_audio_badges": False, "require_submit_guard": True, "warning_vi": ""},
    },
    ensure_ascii=False,
)

# ---- Slim schema: the LLM emits ONLY the decisions; the backend fills all the
# boilerplate/derivable fields (rerank_policy, ui_hints, qa block, empty arrays).
# Far fewer output tokens => much faster, while every decision-bearing field is
# kept (query_type, per-channel enable/weight/queries, trake events, negations),
# so routing/translation/event-split quality is unchanged. A short per-channel
# `reason` is kept as a light reasoning scaffold.
SLIM_SYSTEM_PROMPT = SYSTEM_PROMPT + """

OUTPUT THE SLIM SCHEMA ONLY (the backend fills the rest). Keep all decisions:
query_type, confidence, translated_en_visual, per-channel enabled/weight/reason/
queries (image_pe.queries_en expanded to 2-3 visual variants), negations.
Rules: enable ocr/speech/audio ONLY when clearly needed — default them to false.
trake_events MUST be [] UNLESS query_type is exactly "TRAKE" (a query describing
several events in chronological order). A single scene/moment is T-KIS or V-KIS,
NOT TRAKE — do not invent events. Do NOT output rerank_policy, ui_hints, or qa."""

_SLIM_SCHEMA_HINT = json.dumps(
    {
        "query_type": "T-KIS|QA|V-KIS|TRAKE",
        "confidence": 0.0,
        "translated_en_visual": "",
        "channels": {
            "image_pe": {"enabled": True, "weight": 1.0, "reason": "", "queries_en": []},
            "ocr": {"enabled": False, "weight": 0.0, "reason": "", "queries_vi": [], "queries_folded": [], "exact_phrases": [], "numbers": [], "hour": None, "clock": None},
            "speech": {"enabled": False, "weight": 0.0, "reason": "", "queries_vi": [], "exact_phrases": []},
            "audio": {"enabled": False, "weight": 0.0, "reason": "", "queries_en": [], "sound_labels_en": []},
        },
        "negations": [],
        "trake_events": [],
    },
    ensure_ascii=False,
)


def slim_to_full(slim: dict[str, Any], query: str, hint: str) -> dict[str, Any]:
    """Expand the slim LLM output into the full routing dict (same shape the rest
    of the app expects). Boilerplate is filled here, not generated by the LLM."""
    ch = slim.get("channels") or {}

    def chan(name: str, extra_keys: tuple[str, ...]) -> dict[str, Any]:
        c = ch.get(name) or {}
        out = _empty_channel(name)
        out["enabled"] = bool(c.get("enabled", out["enabled"]))
        out["weight"] = float(c.get("weight") or (1.0 if out["enabled"] else 0.0))
        out["reason"] = c.get("reason") or ""
        for k in extra_keys:
            if k in c and c[k] is not None:
                out[k] = c[k]
        return out

    image_pe = chan("image_pe", ("queries_en", "positive_concepts_en", "negative_concepts_en"))
    ocr = chan("ocr", ("queries_vi", "queries_folded", "exact_phrases", "numbers"))
    oc = ch.get("ocr") or {}
    ocr["time_filters"] = {"hour": oc.get("hour"), "clock": oc.get("clock")}
    speech = chan("speech", ("queries_vi", "exact_phrases", "speaker_or_role_hints"))
    audio = chan("audio", ("queries_en", "sound_labels_en"))

    qtype = slim.get("query_type") or "T-KIS"
    negations = slim.get("negations") or []

    events = slim.get("trake_events") or []
    trake_events = [
        {
            "event_index": i + 1,
            "description_vi": "",
            "description_en_visual": (ev.get("image_pe_queries_en") or [""])[0],
            "image_pe_queries_en": ev.get("image_pe_queries_en") or [],
            "ocr_queries_vi": ev.get("ocr_queries_vi") or [],
            "speech_queries_vi": ev.get("speech_queries_vi") or [],
            "audio_queries_en": ev.get("audio_queries_en") or [],
            "expected_order_hint": "before" if i == 0 else "after",
            "required": True,
        }
        for i, ev in enumerate(events)
    ]
    trake_enabled = qtype == "TRAKE" and len(trake_events) >= 1

    return {
        "query_type": qtype,
        "confidence": float(slim.get("confidence") or 0.6),
        "original_query": query,
        "normalized_vi": query,
        "translated_en_visual": slim.get("translated_en_visual") or "",
        "operator_summary_vi": "",
        "channels": {"image_pe": image_pe, "ocr": ocr, "speech": speech, "audio": audio},
        "filters": {"video_ids": [], "categories": [], "time_range_seconds": None, "must_include": [], "must_not_include": negations},
        "trake": {
            "enabled": trake_enabled,
            "events": trake_events,
            "ordering_rule": "strict_increasing_time",
            "fallback_policy": "allow_partial_confident_events_with_warning",
        },
        "qa": {
            "enabled": qtype == "QA",
            "question_vi": query if qtype == "QA" else "",
            "answer_evidence_channels": ["ocr", "speech", "audio"],
            "draft_answer_allowed_from_text_evidence_only": True,
        },
        "rerank_policy": {"use_rrf": True, "rrf_k": 60, "group_by_video": True, "prefer_temporal_cooccurrence": True},
        "ui_hints": {
            "show_timeline": trake_enabled,
            "show_ocr_snippets": ocr["enabled"],
            "show_speech_snippets": speech["enabled"],
            "show_audio_badges": audio["enabled"],
            "require_submit_guard": True,
            "warning_vi": "",
        },
        "_engine": "nemotron-slim",
    }


class QueryParser:
    def __init__(self, settings: Settings):
        self.s = settings
        # Cache the pre-override parse so toggling a channel / re-running the same
        # query does not pay the LLM round-trip again.
        self._cache: dict[tuple, dict[str, Any]] = {}

    async def parse(
        self,
        query: str,
        query_type_hint: str = "auto",
        previous_hints: list[str] | None = None,
        manual_overrides: dict[str, Any] | None = None,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        previous_hints = previous_hints or []
        cache_key = (query.strip(), query_type_hint, tuple(previous_hints), use_llm)

        base = self._cache.get(cache_key)
        if base is None:
            parsed: dict[str, Any] | None = None
            if use_llm and self.s.has_llm and not self.s.mock_mode:
                parsed = await self._llm_parse(query, query_type_hint, previous_hints)
            if parsed is None:
                parsed = heuristic_parse(query, query_type_hint, previous_hints)
                # LLM off → translate VI→EN so the PE visual query is English.
                if self.s.translate_to_en and not self.s.mock_mode:
                    await self._apply_translation(parsed)
            base = parsed
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[cache_key] = base

        # Work on a copy so cached base stays clean across override variations.
        result = copy.deepcopy(base)
        result = apply_manual_overrides(result, manual_overrides or {})
        result = ensure_image_pe(result)
        return result

    async def expand_visual(self, text: str, n: int = 3) -> list[str]:
        """Lightweight Nemotron query expansion: turn a visual query into a few
        short, diverse English visual rephrasings (objects/colors/scene). Uses a
        tiny prompt + small max_tokens so it's far faster than the full parser.
        Returns [] on any failure (caller keeps the original query)."""
        text = (text or "").strip()
        if not text or not self.s.has_llm or self.s.mock_mode:
            return []
        sys = (
            "You expand an image-retrieval query into short, diverse English visual "
            "descriptions of what is literally VISIBLE (objects, colors, shapes, scene). "
            "Pair any named entity with a visual description. Output ONLY a JSON array of "
            f"{n} concise strings, no prose."
        )
        body = {
            "model": self.s.nvidia_fast_model,  # fast model — the 550B is too slow for this
            "messages": [{"role": "system", "content": sys}, {"role": "user", "content": text}],
            "temperature": 0.2,  # lower => more reproducible variants run-to-run
            "max_tokens": 220,
        }
        headers = {"Authorization": f"Bearer {self.s.nvidia_api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    f"{self.s.nvidia_base_url}/chat/completions", json=body, headers=headers
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:  # noqa: BLE001
            return []
        # Parse a JSON array; fall back to line splitting.
        variants: list[str] = []
        data = _extract_json_array(content)
        if data is not None:
            variants = [str(x).strip() for x in data if str(x).strip()]
        else:
            variants = [ln.strip(" -*0123456789.\t\"'") for ln in content.splitlines() if ln.strip()]
        # Dedup + drop non-English variants (guard against a model echoing the
        # Vietnamese input — keeps the PE visual query English regardless of model).
        from .translate import _looks_english

        seen, out = set(), []
        for v in variants:
            k = v.lower()
            if v and k not in seen and _looks_english(v):
                seen.add(k)
                out.append(v)
        return out[:n]

    async def _apply_translation(self, parsed: dict[str, Any]) -> None:
        """Translate the Vietnamese query to English for the image_pe channel.
        OCR/speech keep the Vietnamese text; only the visual query is translated."""
        from .translate import translate_vi_to_en, translate_vi_to_en_status

        q_vi = parsed.get("normalized_vi") or parsed.get("original_query") or ""
        if not q_vi.strip():
            return
        en, ok = await translate_vi_to_en_status(q_vi, settings=self.s)
        if en and en.strip() and en.strip().lower() != q_vi.strip().lower():
            parsed["translated_en_visual"] = en
            img = parsed.get("channels", {}).get("image_pe")
            if img is not None:
                img["queries_en"] = [en]
        elif not ok:
            # PE-Core is English-centric: searching with the Vietnamese text
            # returns plausible-looking frames that do not match the query. Say
            # so, rather than letting the operator debug the ranking instead.
            parsed["translation_failed"] = True
        # TRAKE: translate each event's visual query so per-event PE search is English.
        trake = parsed.get("trake") or {}
        if trake.get("enabled"):
            for ev in trake.get("events", []):
                src = ev.get("description_vi") or (ev.get("image_pe_queries_en") or [""])[0]
                if not src or not src.strip():
                    continue
                ev_en = await translate_vi_to_en(src)
                if ev_en and ev_en.strip():
                    ev["description_en_visual"] = ev_en
                    ev["image_pe_queries_en"] = [ev_en]

    async def _llm_parse(self, query, hint, previous_hints) -> dict[str, Any] | None:
        # Default: full schema (accurate routing). Slim schema (opt-in via
        # SLIM_PARSE) is ~2x faster but A/B showed it can misroute query_type on
        # ambiguous queries, so it is not the default.
        slim = self.s.slim_parse
        system = SLIM_SYSTEM_PROMPT if slim else SYSTEM_PROMPT
        prompt = USER_TEMPLATE.format(
            query=query,
            hint=hint,
            previous=json.dumps(previous_hints, ensure_ascii=False),
            schema=_SLIM_SCHEMA_HINT if slim else _SCHEMA_HINT,
        )
        for attempt in range(2):
            extra_instruction = "" if attempt == 0 else "\nReturn valid JSON only."
            try:
                content = await self._chat(system, prompt + extra_instruction)
                data = _extract_json(content)
                if not data:
                    continue
                if slim and data.get("channels"):
                    return slim_to_full(data, query, hint)
                data["_engine"] = "nemotron"
                return data
            except Exception:  # noqa: BLE001 - fall back to heuristics
                break
        return None

    async def _chat(self, system: str, user: str) -> str:
        body = {
            "model": self.s.nvidia_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,  # full schema output is large; slim stops earlier
        }
        headers = {
            "Authorization": f"Bearer {self.s.nvidia_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=40.0) as client:
            resp = await client.post(
                f"{self.s.nvidia_base_url}/chat/completions", json=body, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


def _extract_json(content: str) -> dict[str, Any] | None:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = content.rstrip("`").rstrip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _populate_channel_queries(parsed: dict[str, Any], name: str, channel: dict[str, Any]) -> None:
    """Fill a force-enabled channel's queries from the query text if it has none,
    so forcing e.g. OCR on a query the heuristic didn't route there still searches
    for the actual query (instead of matching the whole index)."""
    q_vi = parsed.get("normalized_vi") or parsed.get("original_query") or ""
    q_en = parsed.get("translated_en_visual") or q_vi
    if name == "image_pe" and not channel.get("queries_en"):
        channel["queries_en"] = [q_en] if q_en else []
    elif name == "ocr" and not any(
        channel.get(key) for key in ("queries_vi", "exact_phrases", "numbers")
    ):
        if q_vi:
            channel.update(ocr_evidence(q_vi, fallback_to_full_text=True))
    elif name == "speech" and not channel.get("queries_vi"):
        channel["queries_vi"] = [q_vi] if q_vi else []
    elif name == "audio" and not channel.get("queries_en"):
        channel["queries_en"] = [q_en] if q_en else []


def _extract_json_array(content: str) -> list | None:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content).rstrip("`").rstrip()
    start, end = content.find("["), content.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(content[start : end + 1])
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None
    return None


def apply_manual_overrides(parsed: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    force = overrides.get("force_channels") or []
    disable = overrides.get("disable_channels") or []
    channels = parsed.get("channels", {})
    for name in force:
        if name in channels:
            channels[name]["enabled"] = True
            if channels[name].get("weight", 0.0) <= 0:
                channels[name]["weight"] = 1.0
            _populate_channel_queries(parsed, name, channels[name])
            channels[name]["reason"] = (channels[name].get("reason") or "") + " [forced by operator]"
    for name in disable:
        if name in channels:
            channels[name]["enabled"] = False
            channels[name]["reason"] = (channels[name].get("reason") or "") + " [disabled by operator]"
    return parsed


def ensure_image_pe(parsed: dict[str, Any]) -> dict[str, Any]:
    channels = parsed.get("channels", {})
    if not any(c.get("enabled") for c in channels.values()):
        img = channels.setdefault("image_pe", _empty_channel("image_pe"))
        img["enabled"] = True
        img["weight"] = img.get("weight") or 1.0
        img["reason"] = "Force-enabled: all channels were disabled."
        if not img.get("queries_en"):
            img["queries_en"] = [parsed.get("translated_en_visual") or parsed.get("original_query") or ""]
    return parsed
