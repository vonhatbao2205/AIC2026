"""Query understanding + retrieval routing.

Primary path (the console's "LLM" switch): the query LLM — DeepSeek
`deepseek-flash` by default — returns a compact routing JSON that
`app.query_llm` validates and expands. Fallback: deterministic heuristics, so the
system works without an LLM or when the call fails. After either path,
`manual_overrides` are applied and at least `image_pe` is guaranteed enabled.

Secrets (QUERY_LLM_API_KEY / DEEPSEEK_API_KEY) never leave the backend.
"""
from __future__ import annotations

import copy
import re
from typing import Any

from .config import Settings
from .query_llm import (
    EXPAND_SYSTEM_PROMPT,
    PARSER_SYSTEM_PROMPT,
    QueryLlm,
    QueryLlmError,
    expand_user_message,
    expand_variants,
    parser_user_message,
    to_routing,
)
from .text_normalization import fold_vietnamese
from .trake_events import compose_event_queries, shared_context, split_marked_events

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

    if query_type_hint in {"T-KIS", "QA", "V-KIS", "TRAKE", "AVS"}:
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
        trake["shared_context_vi"] = shared_context(query)
        trake["shared_context_en_visual"] = ""
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
        compose_event_queries({"trake": trake})

    return {
        "query_type": query_type,
        "confidence": 0.4,
        "original_query": query,
        "normalized_vi": combined,
        "translated_en_visual": combined,
        "operator_summary_vi": f"Heuristic routing for query: {query}",
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


def _split_trake_events(query: str) -> list[str]:
    # 1) The organisers' markers E1, E2, … and nothing else (app.trake_events):
    #    a number in the prose such as "(4)" is not an event.
    marked = split_marked_events(query)
    if marked:
        return marked
    # 2) Sequence connectors ("sau đó", "rồi", ...).
    pattern = "|".join(re.escape(c) for c in _TRAKE_CONNECTORS)
    parts = [p.strip(" ,.;") for p in re.split(pattern, query, flags=re.IGNORECASE) if p.strip(" ,.;")]
    return parts if len(parts) >= 2 else [query]


class QueryParser:
    def __init__(self, settings: Settings):
        self.s = settings
        # Cache the pre-override parse so toggling a channel / re-running the same
        # query does not pay the LLM round-trip again.
        self._cache: dict[tuple, dict[str, Any]] = {}
        self._expand_cache: dict[tuple, list[str]] = {}
        self.llm = QueryLlm(settings)

    async def parse(
        self,
        query: str,
        query_type_hint: str = "auto",
        previous_hints: list[str] | None = None,
        manual_overrides: dict[str, Any] | None = None,
        use_llm: bool = False,
        translate: bool = True,
    ) -> dict[str, Any]:
        """Parse one query. `translate` is the operator's per-search tick box.

        Off means "search with the words I typed": no VI→EN call, and the visual
        query keeps the original text even when the LLM parser handed back an
        English rewrite. `TRANSLATE_TO_EN=false` still disables it globally, so
        the tick box can only ever turn translation off, never force it on where
        the deployment has ruled it out.
        """
        previous_hints = previous_hints or []
        # `translate` belongs in the key: the same query parsed with and without
        # translation are two different results, and sharing one cache entry
        # would serve whichever ran first to both.
        cache_key = (query.strip(), query_type_hint, tuple(previous_hints), use_llm, translate)

        base = self._cache.get(cache_key)
        if base is None:
            translate_on = translate and self.s.translate_to_en and not self.s.mock_mode
            parsed: dict[str, Any] | None = None
            llm_failure: str | None = None
            if use_llm and self.llm.available:
                try:
                    parsed = await self._llm_parse(query, query_type_hint, previous_hints)
                except QueryLlmError as exc:
                    llm_failure = str(exc)
            if parsed is None:
                parsed = heuristic_parse(query, query_type_hint, previous_hints)
                if use_llm and self.llm.available:
                    # Said on screen: an operator who asked for the LLM must not be
                    # left believing a heuristic routing came from it.
                    parsed["ui_hints"]["warning_vi"] = (
                        f"LLM parser unavailable ({llm_failure or 'no usable visual query'}); "
                        "heuristic routing used."
                    )
                # LLM off → translate VI→EN so the PE visual query is English.
                if translate_on:
                    await self._apply_translation(parsed)
            elif not translate:
                # The LLM parser translates as part of parsing, so honouring the
                # tick box here means undoing that rewrite — otherwise "no
                # translation" would still search with English the operator
                # never wrote.
                _drop_translation(parsed)
            else:
                await self._translate_bare_events(parsed)
            compose_event_queries(parsed)
            if query_type_hint == "AVS":
                parsed["query_type"] = "AVS"
                parsed["trake"] = {"enabled": False, "events": []}
            base = parsed
            # A failed VI→EN translation is transient — a rate limit, or the
            # network buckling under a bulk run. `translate.py` refuses to cache
            # one for exactly that reason, but caching the *parse* that carries
            # it defeats that: the query stays pinned to its Vietnamese text on
            # every later search, and only a backend restart clears it. The
            # operator sees one tab that can never translate while the others
            # are fine, and re-running does nothing.
            if not parsed.get("translation_failed"):
                if len(self._cache) > 256:
                    self._cache.clear()
                self._cache[cache_key] = base

        # Work on a copy so cached base stays clean across override variations.
        result = copy.deepcopy(base)
        if self.s.has_tara_search:
            img = (result.get("channels") or {}).get("image_pe") or {}
            query_en = (
                result.get("translated_en_visual")
                or (img.get("queries_en") or [""])[0]
                or result.get("original_query")
                or ""
            )
            result.setdefault("channels", {})["tara"] = {
                "enabled": True, "weight": 1.0,
                "reason": "TARA temporal video clips",
                "queries_en": [query_en] if query_en else [],
            }
        result = apply_manual_overrides(result, manual_overrides or {})
        result = ensure_image_pe(result)
        return result

    async def expand_visual(self, text: str, n: int = 3, *, original_vi: str | None = None) -> list[str]:
        """Up to `n` more English phrasings of a visual query ("Expand").

        Each is searched on its own and a frame keeps its best score, so a frame
        that matches ANY phrasing strongly rises. Returns [] on any failure: the
        search then runs on the query it already had.
        """
        text = (text or "").strip()
        if not text or not self.llm.available:
            return []
        key = (text, original_vi or "", n)
        if key in self._expand_cache:
            return list(self._expand_cache[key])
        try:
            data = await self.llm.chat_json(
                EXPAND_SYSTEM_PROMPT, expand_user_message(text, original_vi), max_tokens=400
            )
        except QueryLlmError:
            return []
        variants = expand_variants(data, existing=[text], n=n)
        if variants:
            if len(self._expand_cache) > 512:
                self._expand_cache.clear()
            self._expand_cache[key] = variants
        return list(variants)

    async def _apply_translation(self, parsed: dict[str, Any]) -> None:
        """Translate the Vietnamese query to English for the image_pe channel.
        OCR/speech keep the Vietnamese text; only the visual query is translated."""
        from .translate import translate_vi_to_en_status

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
            await self._translate_shared_context(parsed)
            for ev in trake.get("events", []):
                src = ev.get("description_vi") or (ev.get("image_pe_queries_en") or [""])[0]
                if not src or not src.strip():
                    continue
                ev_en, event_ok = await translate_vi_to_en_status(src, settings=self.s)
                if not event_ok:
                    parsed["translation_failed"] = True
                if ev_en and ev_en.strip():
                    ev["description_en_visual"] = ev_en
                    ev["image_pe_queries_en"] = [ev_en]

    async def _llm_parse(self, query, hint, previous_hints) -> dict[str, Any] | None:
        """Route through the query LLM; None when its answer has no usable visual query.

        Raises `QueryLlmError` when the call itself fails.
        """
        data = await self.llm.chat_json(
            PARSER_SYSTEM_PROMPT,
            parser_user_message(query, hint, previous_hints, self.s.retrieval_database),
            max_tokens=1600,
        )
        routed = to_routing(data, query=query, hint=hint, previous_hints=previous_hints)
        if routed is not None:
            routed["_engine"] = f"llm:{self.s.query_llm_model}"
        return routed

    async def _translate_shared_context(self, parsed: dict[str, Any]) -> None:
        from .translate import translate_vi_to_en_status

        trake = parsed.get("trake") or {}
        source = trake.get("shared_context_vi")
        if not source or trake.get("shared_context_en_visual") or not self.s.translate_to_en or self.s.mock_mode:
            return
        english, ok = await translate_vi_to_en_status(source, settings=self.s)
        if ok and english:
            trake["shared_context_en_visual"] = english
        else:
            parsed["translation_failed"] = True

    async def _translate_bare_events(self, parsed: dict[str, Any]) -> None:
        """TRAKE events the LLM left without English (rebuilt from the E-markers
        when its count disagreed): each is translated on its own."""
        from .translate import translate_vi_to_en_status

        await self._translate_shared_context(parsed)
        for event in (parsed.get("trake") or {}).get("events", []) or []:
            if event.get("image_pe_queries_en") or not (event.get("description_vi") or "").strip():
                continue
            source = event["description_vi"]
            english, ok = (await translate_vi_to_en_status(source, settings=self.s)
                           if self.s.translate_to_en and not self.s.mock_mode else (source, True))
            if not ok:
                parsed["translation_failed"] = True
            text = english.strip() if english and english.strip() else source
            event["description_en_visual"] = text
            event["image_pe_queries_en"] = [text]


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


def _drop_translation(parsed: dict[str, Any]) -> dict[str, Any]:
    """Put the operator's own words back into the visual query.

    Used when the translate tick box is off but the parse came from the LLM,
    which rewrites the query into English as part of its job. Only the visual
    (image) queries are touched: OCR/speech were always Vietnamese.
    """
    q_vi = parsed.get("normalized_vi") or parsed.get("original_query") or ""
    parsed["translated_en_visual"] = ""
    parsed.pop("translation_failed", None)
    if parsed.get("trake"):
        parsed["trake"]["shared_context_en_visual"] = ""
    if not q_vi.strip():
        return parsed
    img = (parsed.get("channels") or {}).get("image_pe")
    if isinstance(img, dict):
        img["queries_en"] = [q_vi]
    for event in (parsed.get("trake") or {}).get("events", []) or []:
        source = event.get("description_vi") or q_vi
        event["description_en_visual"] = ""
        event["image_pe_queries_en"] = [source]
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
