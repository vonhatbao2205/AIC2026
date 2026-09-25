"""Elastic adapter for OCR / speech / audio / keyframe-map indices.

Search methods return raw scored hits (dicts). The search service converts them
to ranked `ChannelHit`s. Quality demotions from `app.scoring` are applied here so
they influence per-channel ordering (and therefore RRF rank).

In mock mode, returns deterministic matches from `app.mock_data`.
"""
from __future__ import annotations

import json
import re
from typing import Any


from .. import mock_data
from ..config import Settings
from ..scope import FrameFilter, elastic_filter_clause, in_scope
from ..scoring import audio_score_multiplier, speech_score_multiplier
from ..text_normalization import fold_vietnamese
from .http_pool import PooledHttpClient, failure_reason


_OCR_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_OCR_FALLBACK_STRICT_FLOOR = 10

# Batch-2 OCR (M, N, S01) labels every box with the screen region it sits in and
# keeps overlay text out of `text_clean`: the M news ticker (often a different
# story from the picture), the N camera banner (junction/date/clock), the S01 race
# HUD, and channel logos. Scene text stays the strongest evidence; ticker, banner
# and HUD are still found, but only through their own low-boost clauses, and logos
# not at all. The L documents predate regions: they carry no `profile`, keep every
# box, and keep their `text_nfc` clauses, so their ranking is unchanged.
_OVERLAY_REGIONS = ["ticker", "banner", "hud", "logo_clock"]
_OVERLAY_FIELDS = ("text_ticker", "text_banner", "text_hud")
_OVERLAY_PHRASE_BOOST = 3.0
_OVERLAY_EXACT_BOOST = 4.0
_OVERLAY_AND_BOOST = 1.5
_OVERLAY_FOLD_BOOST = 1.5
#: Fields a hit carries so the console can show what the banner / HUD said.
_OCR_OVERLAY_SOURCE = ("banner_camera", "banner_date", "race_stage", "race_time", "text_ticker")
_CLOCK_TEXT = re.compile(r"\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*")


def _scene_boxes_phrase(query: str, boost: float) -> dict[str, Any]:
    """Nested phrase match on one box that is scene text (or predates regions)."""
    return {
        "nested": {
            "path": "boxes",
            "score_mode": "max",
            "query": {
                "bool": {
                    "must": [{"match_phrase": {"boxes.text": {"query": query, "boost": boost}}}],
                    "must_not": [{"terms": {"boxes.region": _OVERLAY_REGIONS}}],
                }
            },
        }
    }


def _l_side_only(clause: dict[str, Any]) -> dict[str, Any]:
    """`text_nfc` holds every box; only documents without regions may score on it."""
    return {"bool": {"must": [clause], "must_not": [{"exists": {"field": "profile"}}]}}


def _overlay_clauses(query: str, *, exact: bool) -> list[dict[str, Any]]:
    phrase_boost = _OVERLAY_EXACT_BOOST if exact else _OVERLAY_PHRASE_BOOST
    clauses: list[dict[str, Any]] = []
    for field in _OVERLAY_FIELDS:
        clauses.append({"match_phrase": {field: {"query": query, "boost": phrase_boost}}})
        if not exact:
            clauses.append({"match": {field: {"query": query, "operator": "and", "boost": _OVERLAY_AND_BOOST}}})
    return clauses


def _time_filters(hour: int | None, clock: str | None) -> list[dict[str, Any]]:
    """Broadcast/banner clock or race clock; the hour alone only when no clock is given.

    Clocks are stored as printed: HTV7 drops the leading zero ("8:30:13"), the
    camera banners keep it ("08:12:03"), so both spellings are asked for. A clock
    without seconds matches every second of that minute, and the S01 race HUD's
    elapsed `race_time` answers a clock too.
    """
    if clock:
        match = _CLOCK_TEXT.fullmatch(clock)
        if not match:
            return [{"term": {"clock": clock}}]
        hour_text = int(match.group(1))
        hours = sorted({f"{hour_text:02d}", str(hour_text)})
        minute, second = match.group(2), match.group(3)
        should: list[dict[str, Any]] = []
        for field in ("clock", "race_time"):
            if second:
                should.append({"terms": {field: [f"{h}:{minute}:{second}" for h in hours]}})
            else:
                should.extend({"prefix": {field: f"{h}:{minute}:"}} for h in hours)
        return [{"bool": {"should": should, "minimum_should_match": 1}}]
    if hour is not None:
        return [{"term": {"hour": hour}}]
    return []


def _unique_text(values: list[str] | None) -> list[str]:
    """Strip and deduplicate query variants without changing their order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values or []:
        clean = str(value or "").strip()
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def _safe_numbers(
    numbers: list[str] | None,
    queries: list[str],
    *,
    hour: int | None,
    clock: str | None,
) -> list[str]:
    """The numbers in a query that are worth searching for on their own.

    Digits embedded in an alphanumeric code (for example the ``3`` in VTV3)
    are already protected by the strict all-token query and must not be queried
    as a separate token.  Clock/hour values are stored outside ``text_clean``
    by OCR post-processing, so their dedicated filters replace text filters.

    A number is dropped only when the query text proves it is embedded: it occurs
    in that text and never occurs standalone there. A number the text does not
    mention at all is left alone — the parser strips descriptive prose before
    calling this, so `queries` can be empty or hold an unrelated phrase while the
    numbers came from the part of the query that was discarded. Vetoing on that
    absence used to delete every number and leave the OCR channel with nothing.
    """
    clock_parts = set(re.findall(r"\d+", clock or ""))
    if hour is not None:
        clock_parts.add(str(hour))
    haystack = [query for query in queries if query and query.strip()]

    safe: list[str] = []
    for raw in numbers or []:
        number = str(raw or "").strip()
        if not number.isdigit() or number in clock_parts or number in safe:
            continue
        standalone = re.compile(rf"(?<!\w){re.escape(number)}(?!\w)", re.UNICODE)
        mentioned = any(number in query for query in haystack)
        if mentioned and not any(standalone.search(query) for query in haystack):
            continue  # only ever appears welded into a code, e.g. the 3 of VTV3
        safe.append(number)
    return safe


def _numeric_filters(numbers: list[str]) -> list[dict[str, Any]]:
    """One hard constraint per number, so every number in the query is required."""
    return [
        {
            "dis_max": {
                "queries": [
                    {"match": {"text_clean": {"query": number, "operator": "and"}}},
                    {"match": {"text_nfc": {"query": number, "operator": "and"}}},
                    {"match": {"text_clean_fold": {"query": number, "operator": "and"}}},
                ],
                "tie_breaker": 0.0,
            }
        }
        for number in numbers
    ]


def _strict_ocr_clauses(
    queries_vi: list[str],
    queries_folded: list[str],
    exact_phrases: list[str],
) -> list[dict[str, Any]]:
    """Return quality-tiered full-coverage clauses for OCR retrieval."""
    clauses: list[dict[str, Any]] = []

    # Explicit phrases carry the strongest intent signal.
    for phrase in exact_phrases:
        clauses.extend(
            [
                _scene_boxes_phrase(phrase, 16.0),
                {"match_phrase": {"text_clean": {"query": phrase, "boost": 12.0}}},
                _l_side_only({"match_phrase": {"text_nfc": {"query": phrase, "boost": 10.0}}}),
                {
                    "match_phrase": {
                        "text_clean_fold": {
                            "query": fold_vietnamese(phrase),
                            "boost": 7.0,
                        }
                    }
                },
                *_overlay_clauses(phrase, exact=True),
            ]
        )

    for query in queries_vi:
        clauses.extend(
            [
                _scene_boxes_phrase(query, 12.0),
                {"match_phrase": {"text_clean": {"query": query, "boost": 8.0}}},
                {
                    "match": {
                        "text_clean": {
                            "query": query,
                            "operator": "and",
                            "boost": 4.0,
                        }
                    }
                },
                _l_side_only(
                    {
                        "match": {
                            "text_nfc": {
                                "query": query,
                                "operator": "and",
                                "boost": 3.0,
                            }
                        }
                    }
                ),
                *_overlay_clauses(query, exact=False),
            ]
        )

    for query in queries_folded:
        clauses.extend(
            [
                {
                    "match_phrase": {
                        "text_clean_fold": {"query": query, "boost": 5.0}
                    }
                },
                {
                    "match": {
                        "text_clean_fold": {
                            "query": query,
                            "operator": "and",
                            "boost": 2.0,
                        }
                    }
                },
                {"match_phrase": {"text_ticker_fold": {"query": query, "boost": _OVERLAY_FOLD_BOOST}}},
            ]
        )
    return clauses


def _is_protected_ocr_token(token: str) -> bool:
    """Numbers, codes and likely acronyms must never receive fuzzy matching."""
    folded = fold_vietnamese(token)
    consonant_code = (
        2 <= len(folded) <= 8
        and folded.isalpha()
        and not any(vowel in folded for vowel in "aeiouy")
    )
    return any(char.isdigit() for char in token) or (
        2 <= len(token) <= 8 and token.isupper()
    ) or consonant_code


def _is_joined_to_numeric_code(
    query: str,
    tokens: list[re.Match[str]],
    index: int,
) -> bool:
    """Protect the alphabetic half of a hyphen/slash/dot joined code."""
    current = tokens[index]
    if not current.group().isalpha():
        return False
    for neighbor_index in (index - 1, index + 1):
        if not 0 <= neighbor_index < len(tokens):
            continue
        neighbor = tokens[neighbor_index]
        if not any(char.isdigit() for char in neighbor.group()):
            continue
        left, right = sorted((current, neighbor), key=lambda match: match.start())
        separator = query[left.end() : right.start()]
        if separator and not any(char.isspace() or char.isalnum() for char in separator):
            return True
    return False


def _fuzzy_ocr_query(queries_vi: list[str], queries_folded: list[str]) -> dict[str, Any] | None:
    """Build an all-token fuzzy fallback while keeping numbers/codes exact."""
    sources = queries_vi or queries_folded
    variants: list[dict[str, Any]] = []
    any_fuzzy = False

    for query in sources:
        tokens = list(_OCR_TOKEN_RE.finditer(query))
        must: list[dict[str, Any]] = []
        variant_has_fuzzy = False
        for index, match in enumerate(tokens):
            token = match.group()
            folded = fold_vietnamese(token)
            if not folded:
                continue
            params: dict[str, Any] = {"query": folded, "operator": "and"}
            if not _is_protected_ocr_token(token) and not _is_joined_to_numeric_code(
                query, tokens, index
            ):
                params.update(fuzziness="AUTO", prefix_length=1, max_expansions=25)
                variant_has_fuzzy = True
                any_fuzzy = True
            must.append({"match": {"text_clean_fold": params}})
        if must and variant_has_fuzzy:
            variants.append({"bool": {"must": must}})

    if not variants or not any_fuzzy:
        return None
    return {"dis_max": {"queries": variants, "tie_breaker": 0.0}}


def _ocr_hits(data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hit in data.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        out.append(
            {
                "submit_keyframe_id": src.get("submit_keyframe_id"),
                "video_id": src.get("video_id"),
                "keyframe_n": src.get("keyframe_n"),
                "score": float(hit.get("_score") or 0.0),
                "text_clean": src.get("text_clean") or src.get("text_nfc") or "",
                "clock": src.get("clock"),
                "hour": src.get("hour"),
                "boxes": src.get("boxes") or [],
                **{name: src.get(name) for name in _OCR_OVERLAY_SOURCE if src.get(name) not in (None, "")},
            }
        )
    return out


def _keep_categories(
    hits: list[dict[str, Any]], categories: tuple[str, ...], frames: FrameFilter | None = None
) -> list[dict[str, Any]]:
    """Mock-mode stand-in for the pushed-down scope and frame filters (same rules)."""
    wanted = set(categories)
    return [
        hit
        for hit in hits
        if (not wanted or in_scope(str(hit.get("video_id") or ""), wanted))
        and (not frames or frames.allows(str(hit.get("video_id") or ""), int(hit.get("keyframe_n") or 0)))
    ]


class ElasticClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.mock = settings.mock_mode or not settings.has_elastic
        self._base = (settings.elastic_endpoint or "").rstrip("/")
        self._http = PooledHttpClient()

    # ---- HTTP plumbing -------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"ApiKey {self.s.elastic_api_key}",
            "Content-Type": "application/json",
        }

    async def _search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base}/{index}/_search"
        resp = await self._http.get().post(url, json=body, headers=self._headers(), timeout=20.0)
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> dict[str, Any]:
        if self.mock:
            return {"ok": True, "mode": "mock"}
        try:
            resp = await self._http.get().get(f"{self._base}/", headers=self._headers(), timeout=10.0)
            resp.raise_for_status()
            info = resp.json()
            return {"ok": True, "cluster": info.get("cluster_name")}
        except Exception as exc:  # noqa: BLE001 - surfaced as health warning
            return {"ok": False, "error": failure_reason(exc)}

    # ---- OCR -----------------------------------------------------------
    async def search_ocr(
        self,
        queries_vi: list[str],
        queries_folded: list[str] | None = None,
        *,
        exact_phrases: list[str] | None = None,
        numbers: list[str] | None = None,
        hour: int | None = None,
        clock: str | None = None,
        categories: tuple[str, ...] = (),
        frames: FrameFilter | None = None,
        size: int = 100,
    ) -> list[dict[str, Any]]:
        queries_vi = _unique_text(queries_vi)
        queries_folded = _unique_text(
            [*(queries_folded or []), *(fold_vietnamese(query) for query in queries_vi)]
        )
        exact_phrases = _unique_text(exact_phrases)
        if not numbers:
            numbers = [
                number
                for query in [*queries_vi, *exact_phrases]
                for number in re.findall(r"\d+", query)
            ]
        # No text, no digits and no time filter => nothing to search (never match_all).
        if not queries_vi and not queries_folded and not exact_phrases and not numbers and hour is None and not clock:
            return []

        if self.mock:
            return _keep_categories(
                self._mock_ocr(
                    queries_vi,
                    queries_folded,
                    exact_phrases=exact_phrases,
                    numbers=numbers or [],
                    hour=hour,
                    clock=clock,
                    size=size,
                ),
                categories,
                frames,
            )

        safe_numbers = _safe_numbers(
            numbers,
            [*queries_vi, *queries_folded, *exact_phrases],
            hour=hour,
            clock=clock,
        )
        numeric = _numeric_filters(safe_numbers)
        strict = _strict_ocr_clauses(queries_vi, queries_folded, exact_phrases)
        if not queries_vi and not queries_folded and safe_numbers:
            # No free-text query, so the digits are first-class evidence and must
            # score, not merely filter. They go through the exact-phrase builder
            # so a frame whose overlay box IS the number — a speed readout, a
            # jersey number — wins on the nested `boxes.text` clause instead of
            # competing on whole-frame text length, where a long ticker
            # mentioning the same digits would bury it.
            #
            # They are added even when `exact_phrases` already produced clauses,
            # because those phrases are guesses about what is printed on screen
            # and a wrong guess must not be able to delete the number. Measured:
            # for "…69 km/h" an LLM proposes the phrase "km/h", but OCR reads the
            # bare readout and never the unit — as the only `must` clause that
            # phrase cut 641 candidates to 1 and lost the answer, while as one
            # option of a dis_max alongside "69" it costs nothing.
            strict += _strict_ocr_clauses([], [], safe_numbers)

        time_filters = _time_filters(hour, clock)
        # The scope constrains WHERE to look, never WHAT counts as a match, so it
        # rides in `filter` (no scoring effect) and applies to the fuzzy pass too.
        scope_clause = elastic_filter_clause(categories, frames)
        scope_filters = [scope_clause] if scope_clause else []

        if strict:
            must: list[dict] = [
                {
                    "dis_max": {
                        "queries": strict,
                        # Multiple fields are representations of the same
                        # OCR evidence.  Only a small secondary-field tie
                        # boost is allowed instead of summing every field.
                        "tie_breaker": 0.1,
                    }
                }
            ]
            # dis_max above only needs ONE clause to match; these keep every
            # number in the query required.
            filt = [*time_filters, *numeric, *scope_filters]
        elif time_filters:
            # A bare broadcast-clock query. Scores are uniform, but the frames it
            # returns are exactly the matching ones and RRF consumes rank anyway.
            # (A scope on its own never reaches here: it narrows a query, it is
            # not one, so an empty query still returns nothing.)
            must = []
            filt = [*time_filters, *scope_filters]
        else:
            return []  # never fall through to match_all on an empty query

        primary_body = {"size": size, "query": {"bool": {"must": must, "filter": filt}}}
        primary_hits = _ocr_hits(await self._search(self.s.idx_ocr, primary_body))

        # Full-coverage matches always form the first quality tier.  Only when
        # that tier is sparse do we pay for a second fuzzy request, whose hits are
        # appended (never allowed to outrank a full-coverage match).
        fallback_floor = min(size, _OCR_FALLBACK_STRICT_FLOOR)
        if len(primary_hits) >= fallback_floor:
            return primary_hits[:size]

        # Fuzzy only ever applies to words. A digits-only query has no fuzzy tier
        # by construction: `_fuzzy_ocr_query` returns None without text sources.
        fuzzy = _fuzzy_ocr_query(queries_vi, queries_folded)
        if fuzzy is None:
            return primary_hits[:size]
        fallback_body = {
            "size": size,
            "query": {"bool": {"must": [fuzzy], "filter": [*time_filters, *numeric, *scope_filters]}},
        }
        fallback_hits = _ocr_hits(await self._search(self.s.idx_ocr, fallback_body))
        seen = {hit["submit_keyframe_id"] for hit in primary_hits}
        for hit in fallback_hits:
            keyframe_id = hit["submit_keyframe_id"]
            if keyframe_id in seen:
                continue
            primary_hits.append(hit)
            seen.add(keyframe_id)
            if len(primary_hits) >= size:
                break
        return primary_hits

    # ---- Speech --------------------------------------------------------
    async def search_speech(
        self,
        queries_vi: list[str],
        *,
        exact_phrases: list[str] | None = None,
        categories: tuple[str, ...] = (),
        frames: FrameFilter | None = None,
        size: int = 100,
    ) -> list[dict[str, Any]]:
        if not queries_vi and not (exact_phrases or []):
            return []
        if self.mock:
            return _keep_categories(self._mock_speech(queries_vi, size=size), categories, frames)

        should: list[dict] = [{"match": {"text": {"query": q}}} for q in queries_vi]
        for p in (exact_phrases or []):
            should.append({"match_phrase": {"text": {"query": p, "boost": 3.0}}})
        if not should:
            return []
        scope_clause = elastic_filter_clause(categories, frames)
        body = {
            "size": size,
            "query": {
                "bool": {
                    "should": should,
                    "minimum_should_match": 1,
                    **({"filter": [scope_clause]} if scope_clause else {}),
                }
            },
        }
        data = await self._search(self.s.idx_speech, body)
        out = []
        for h in data.get("hits", {}).get("hits", []):
            src = h.get("_source", {})
            mult = speech_score_multiplier(src.get("confidence_bucket"), src.get("segment_role"))
            out.append(
                {
                    "submit_keyframe_id": src.get("submit_keyframe_id")
                    or src.get("center_submit_keyframe_id"),
                    "video_id": src.get("video_id"),
                    "keyframe_n": src.get("keyframe_n") or src.get("center_keyframe_n"),
                    "score": float(h.get("_score") or 0.0) * mult,
                    "text": src.get("text") or "",
                    "start": src.get("start"),
                    "end": src.get("end"),
                    "confidence_bucket": src.get("confidence_bucket"),
                    "segment_role": src.get("segment_role"),
                }
            )
        out.sort(key=lambda x: -x["score"])
        return out

    # ---- Audio ---------------------------------------------------------
    async def search_audio(
        self,
        queries_en: list[str],
        sound_labels: list[str] | None = None,
        *,
        categories: tuple[str, ...] = (),
        frames: FrameFilter | None = None,
        size: int = 100,
    ) -> list[dict[str, Any]]:
        if not queries_en and not (sound_labels or []):
            return []
        if self.mock:
            return _keep_categories(
                self._mock_audio(queries_en + (sound_labels or []), size=size), categories, frames
            )

        should: list[dict] = []
        for lbl in (sound_labels or []):
            should.append({"term": {"top1_label": {"value": lbl, "boost": 3.0}}})
            should.append({"terms": {"tag_labels": [lbl]}})
        for q in queries_en:
            should.append({"match": {"caption": {"query": q}}})
            should.append({"match": {"top1_label": {"query": q, "boost": 2.0}}})
            should.append({"match": {"tag_labels": {"query": q}}})
        if not should:
            return []
        scope_clause = elastic_filter_clause(categories, frames)
        body = {
            "size": size,
            "query": {
                "bool": {
                    "should": should,
                    "minimum_should_match": 1,
                    **({"filter": [scope_clause]} if scope_clause else {}),
                }
            },
        }
        data = await self._search(self.s.idx_audio, body)
        wanted = {s.lower() for s in (sound_labels or [])}
        out = []
        for h in data.get("hits", {}).get("hits", []):
            src = h.get("_source", {})
            top1 = (src.get("top1_label") or "")
            matched_top1 = top1.lower() in wanted if wanted else False
            mult = audio_score_multiplier(
                audio_stoplist_hit=bool(src.get("audio_stoplist_hit")),
                top1_is_stoplisted=bool(src.get("top1_is_stoplisted")),
                caption_quality=src.get("caption_quality"),
                matched_on_top1=matched_top1,
            )
            if mult <= 0:
                continue
            out.append(
                {
                    "submit_keyframe_id": src.get("submit_keyframe_id")
                    or src.get("center_submit_keyframe_id"),
                    "video_id": src.get("video_id"),
                    "keyframe_n": src.get("keyframe_n") or src.get("center_keyframe_n"),
                    "score": float(h.get("_score") or 0.0) * mult,
                    "top1_label": top1,
                    "caption": src.get("caption"),
                    "caption_quality": src.get("caption_quality"),
                    "start": src.get("start"),
                    "end": src.get("end"),
                    "window_id": src.get("window_id"),
                }
            )
        out.sort(key=lambda x: -x["score"])
        return out

    # ---- Keyframe map / timeline --------------------------------------
    async def get_keyframe(self, submit_keyframe_id: str) -> dict[str, Any] | None:
        if self.mock:
            return mock_data.keyframe_record(submit_keyframe_id)
        body = {"size": 1, "query": {"term": {"submit_keyframe_id": submit_keyframe_id}}}
        data = await self._search(self.s.idx_keyframe_map, body)
        hits = data.get("hits", {}).get("hits", [])
        return hits[0]["_source"] if hits else None

    async def get_keyframes_by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Batch-fetch keyframe-map docs by submit_keyframe_id (the doc _id) in a
        single _mget request. Returns {submit_keyframe_id: source}."""
        if not ids:
            return {}
        if self.mock:
            out: dict[str, dict[str, Any]] = {}
            for i in ids:
                rec = mock_data.keyframe_record(i)
                if rec:
                    out[i] = rec
            return out
        url = f"{self._base}/{self.s.idx_keyframe_map}/_mget"
        resp = await self._http.get().post(url, json={"ids": ids}, headers=self._headers(), timeout=20.0)
        resp.raise_for_status()
        data = resp.json()
        out2: dict[str, dict[str, Any]] = {}
        for doc in data.get("docs", []):
            if doc.get("found"):
                out2[doc["_id"]] = doc.get("_source", {})
        return out2

    async def get_ocr_overlays(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """What the banner / HUD / clock of these keyframes read, by submit_keyframe_id.

        One `_mget` on the OCR index (whose `_id` is the submit id). Only the
        overlay fields travel; frames without OCR are simply absent.
        """
        if not ids or self.mock:
            return {}
        fields = ["banner_camera", "banner_date", "clock", "race_stage", "race_time"]
        url = f"{self._base}/{self.s.idx_ocr}/_mget"
        resp = await self._http.get().post(
            url,
            params={"_source_includes": ",".join(fields)},
            json={"ids": ids},
            headers=self._headers(),
            timeout=20.0,
        )
        resp.raise_for_status()
        out: dict[str, dict[str, Any]] = {}
        for doc in resp.json().get("docs", []):
            source = doc.get("_source") or {}
            values = {name: source[name] for name in fields if source.get(name) not in (None, "")}
            if doc.get("found") and values:
                out[doc["_id"]] = values
        return out

    async def nearest_keyframes_by_time(
        self, positions: list[tuple[str, float]]
    ) -> list[dict[str, Any] | None]:
        """Resolve TARA clip midpoints to actual submit keyframes in one msearch.

        Both adjacent keyframes are searched, then the closer one is chosen. The
        TARA frame index is not a keyframe ordinal in the InfoShot++ map.
        """
        if not positions:
            return []
        if self.mock:
            result = []
            for video_id, moment in positions:
                frames = mock_data.MOCK_KEYFRAMES.get(video_id, [])
                result.append(min(frames, key=lambda row: abs(float(row["pts_time"]) - moment)) if frames else None)
            return result
        out: list[dict[str, Any] | None] = []
        for offset in range(0, len(positions), 60):
            batch = positions[offset:offset + 60]
            lines: list[str] = []
            for video_id, moment in batch:
                for op, order in (("lte", "desc"), ("gte", "asc")):
                    lines.append("{}")
                    lines.append(json.dumps({
                        "size": 1,
                        "_source": ["submit_keyframe_id", "video_id", "keyframe_n", "frame_idx", "pts_time", "fps"],
                        "query": {"bool": {"filter": [
                            {"term": {"video_id": video_id}},
                            {"range": {"pts_time": {op: moment}}},
                        ]}},
                        "sort": [{"pts_time": order}],
                    }))
            response = await self._http.get().post(
                f"{self._base}/{self.s.idx_keyframe_map}/_msearch",
                content=("\n".join(lines) + "\n").encode(),
                headers={**self._headers(), "Content-Type": "application/x-ndjson"},
                timeout=45.0,
            )
            response.raise_for_status()
            results = response.json().get("responses", [])
            if len(results) != len(batch) * 2:
                raise RuntimeError("Elastic nearest-keyframe msearch returned wrong response count")
            for i, (_, moment) in enumerate(batch):
                options = [
                    hit["_source"]
                    for item in results[2 * i:2 * i + 2]
                    for hit in item.get("hits", {}).get("hits", [])
                ]
                out.append(min(options, key=lambda row: abs(float(row["pts_time"]) - moment)) if options else None)
        return out

    async def video_durations(self, video_ids: list[str]) -> dict[str, float]:
        """Approximate length of each video, as the last keyframe's `pts_time`.

        One terms aggregation for the whole result page rather than one timeline
        fetch per video: the TRAKE heat rows need a time axis for every card on
        screen, not only the one the operator has opened."""
        if not video_ids:
            return {}
        if self.mock:
            return {
                vid: max((float(k["pts_time"]) for k in mock_data.MOCK_KEYFRAMES.get(vid, [])), default=0.0)
                for vid in video_ids
                if mock_data.MOCK_KEYFRAMES.get(vid)
            }
        body = {
            "size": 0,
            "query": {"terms": {"video_id": video_ids}},
            "aggs": {
                "by_video": {
                    "terms": {"field": "video_id", "size": len(video_ids)},
                    "aggs": {"last_pts": {"max": {"field": "pts_time"}}},
                }
            },
        }
        try:
            data = await self._search(self.s.idx_keyframe_map, body)
        except Exception:  # noqa: BLE001
            # A missing time axis costs the heat rows their absolute scale; it
            # must never cost the operator the search result.
            return {}
        buckets = data.get("aggregations", {}).get("by_video", {}).get("buckets", [])
        out: dict[str, float] = {}
        for bucket in buckets:
            value = (bucket.get("last_pts") or {}).get("value")
            if value is not None:
                out[str(bucket["key"])] = float(value)
        return out

    async def get_video_keyframes(self, video_id: str, *, page_size: int = 10_000) -> list[dict[str, Any]]:
        """Every keyframe of one video, in ordinal order.

        One page is not enough: an S01 race video holds up to 75,123 keyframes,
        far past Elastic's 10,000-hit result window, and even L22_V016 (5,152)
        overflowed the old single 5,000-hit page. The map is therefore paged with
        `search_after` on `keyframe_n`, which is unique within a video.
        """
        if self.mock:
            return list(mock_data.MOCK_KEYFRAMES.get(video_id, []))
        out: list[dict[str, Any]] = []
        search_after: list[Any] | None = None
        while True:
            body: dict[str, Any] = {
                "size": page_size,
                "query": {"term": {"video_id": video_id}},
                "sort": [{"keyframe_n": "asc"}],
            }
            if search_after is not None:
                body["search_after"] = search_after
            hits = (await self._search(self.s.idx_keyframe_map, body)).get("hits", {}).get("hits", [])
            out.extend(h["_source"] for h in hits)
            if len(hits) < page_size:
                return out
            search_after = hits[-1]["sort"]

    async def get_video_speech(self, video_id: str, *, size: int = 2000) -> list[dict[str, Any]]:
        if self.mock:
            return list(mock_data.MOCK_SPEECH.get(video_id, []))
        body = {
            "size": size,
            "query": {"term": {"video_id": video_id}},
            "sort": [{"start": "asc"}],
        }
        data = await self._search(self.s.idx_speech, body)
        return [h["_source"] for h in data.get("hits", {}).get("hits", [])]

    async def get_video_ocr(self, video_id: str, *, size: int = 5000) -> list[dict[str, Any]]:
        if self.mock:
            out = []
            for kf_id, rec in mock_data.MOCK_OCR.items():
                if kf_id.split("/")[1] == video_id:
                    out.append({"submit_keyframe_id": kf_id, **rec})
            return out
        body = {
            "size": size,
            "query": {
                "bool": {
                    "must": [{"term": {"video_id": video_id}}],
                    "should": [{"exists": {"field": "text_clean"}}],
                }
            },
        }
        data = await self._search(self.s.idx_ocr, body)
        return [h["_source"] for h in data.get("hits", {}).get("hits", [])]

    async def get_video_speech_window(
        self, video_id: str, start: float, end: float, *, size: int = 200
    ) -> list[dict[str, Any]]:
        """Speech segments of one video overlapping [start, end] seconds, in order."""
        if self.mock:
            return [
                seg for seg in mock_data.MOCK_SPEECH.get(video_id, [])
                if float(seg.get("end", 0)) >= start and float(seg.get("start", 0)) <= end
            ][:size]
        body = {
            "size": size,
            "query": {"bool": {"filter": [
                {"term": {"video_id": video_id}},
                {"range": {"end": {"gte": start}}},
                {"range": {"start": {"lte": end}}},
            ]}},
            "sort": [{"start": "asc"}],
        }
        data = await self._search(self.s.idx_speech, body)
        return [h["_source"] for h in data.get("hits", {}).get("hits", [])]

    async def get_video_ocr_window(
        self, video_id: str, start: float, end: float, *, size: int = 200
    ) -> list[dict[str, Any]]:
        """OCR records of one video's keyframes inside [start, end] seconds, in order.

        Windowed rather than `get_video_ocr`: an S01 race video has tens of
        thousands of OCR'd keyframes, and an agent reading one scene needs a page.
        """
        if self.mock:
            out = []
            for kf_id, rec in mock_data.MOCK_OCR.items():
                if kf_id.split("/")[1] != video_id:
                    continue
                keyframe = mock_data.keyframe_record(kf_id) or {}
                pts = float(keyframe.get("pts_time") or 0.0)
                if start <= pts <= end:
                    out.append({"submit_keyframe_id": kf_id, "pts_time": pts, **rec})
            return sorted(out, key=lambda r: r["pts_time"])[:size]
        body = {
            "size": size,
            "_source": ["submit_keyframe_id", "pts_time", "text_clean", "text_ticker", "text_banner", "text_hud"],
            "query": {"bool": {"filter": [
                {"term": {"video_id": video_id}},
                {"range": {"pts_time": {"gte": start, "lte": end}}},
            ]}},
            "sort": [{"pts_time": "asc"}],
        }
        data = await self._search(self.s.idx_ocr, body)
        return [h["_source"] for h in data.get("hits", {}).get("hits", [])]

    async def list_videos(self, video_id_prefix: str) -> list[dict[str, Any]]:
        """Every video whose id starts with the prefix, with its length and keyframe count.

        One terms aggregation on the keyframe map; sorted by video id. The prefix
        is a folder (`L26_`, `N001-`) or a whole series (`N`).
        """
        if self.mock:
            return [
                {
                    "video_id": video_id,
                    "duration": max((float(k["pts_time"]) for k in frames), default=0.0),
                    "keyframes": len(frames),
                }
                for video_id, frames in sorted(mock_data.MOCK_KEYFRAMES.items())
                if video_id.startswith(video_id_prefix)
            ]
        body = {
            "size": 0,
            "query": {"prefix": {"video_id": video_id_prefix}},
            "aggs": {
                "by_video": {
                    "terms": {"field": "video_id", "size": 5000, "order": {"_key": "asc"}},
                    "aggs": {"last_pts": {"max": {"field": "pts_time"}}},
                }
            },
        }
        data = await self._search(self.s.idx_keyframe_map, body)
        return [
            {
                "video_id": str(bucket["key"]),
                "duration": float((bucket.get("last_pts") or {}).get("value") or 0.0),
                "keyframes": int(bucket.get("doc_count") or 0),
            }
            for bucket in data.get("aggregations", {}).get("by_video", {}).get("buckets", [])
        ]

    async def first_texts_in_windows(
        self, kind: str, windows: list[tuple[str, float, float]], *, per_window: int = 1
    ) -> list[list[dict[str, Any]]]:
        """The first speech segments / OCR'd keyframes of each (video, start, end) window.

        One `_msearch` for the whole list, so a video's minute-by-minute outline or
        a folder's worth of sample lines costs one round trip. `kind` is "speech"
        (segments starting in the window) or "ocr" (keyframes with text in it).
        """
        if not windows:
            return []
        if kind == "speech":
            index, time_field, text_field = self.s.idx_speech, "start", "text"
            fields = ["submit_keyframe_id", "start", "end", "text"]
        elif kind == "ocr":
            index, time_field, text_field = self.s.idx_ocr, "pts_time", "text_clean"
            fields = ["submit_keyframe_id", "pts_time", "text_clean", "text_banner", "text_hud"]
        else:
            raise ValueError(f"Unknown text kind {kind!r}")
        if self.mock:
            out: list[list[dict[str, Any]]] = []
            for video_id, start, end in windows:
                if kind == "speech":
                    rows = [s for s in mock_data.MOCK_SPEECH.get(video_id, []) if start <= float(s["start"]) < end]
                else:
                    rows = []
                    for kf_id, rec in mock_data.MOCK_OCR.items():
                        keyframe = mock_data.keyframe_record(kf_id) or {}
                        if keyframe.get("pts_time") is None:
                            continue
                        pts = float(keyframe["pts_time"])
                        if kf_id.split("/")[1] == video_id and start <= pts < end:
                            rows.append({"submit_keyframe_id": kf_id, "pts_time": pts, **rec})
                out.append(sorted(rows, key=lambda r: float(r[time_field]))[:per_window])
            return out
        results: list[list[dict[str, Any]]] = []
        for offset in range(0, len(windows), 100):
            batch = windows[offset:offset + 100]
            lines: list[str] = []
            for video_id, start, end in batch:
                lines.append("{}")
                lines.append(json.dumps({
                    "size": per_window,
                    "_source": fields,
                    "query": {"bool": {"filter": [
                        {"term": {"video_id": video_id}},
                        {"range": {time_field: {"gte": start, "lt": end}}},
                        {"exists": {"field": text_field}},
                    ]}},
                    "sort": [{time_field: "asc"}],
                }))
            response = await self._http.get().post(
                f"{self._base}/{index}/_msearch",
                content=("\n".join(lines) + "\n").encode(),
                headers={**self._headers(), "Content-Type": "application/x-ndjson"},
                timeout=45.0,
            )
            response.raise_for_status()
            answers = response.json().get("responses", [])
            if len(answers) != len(batch):
                raise RuntimeError("Elastic text-window msearch returned wrong response count")
            for answer in answers:
                results.append([hit["_source"] for hit in answer.get("hits", {}).get("hits", [])])
        return results

    async def get_video_audio(self, video_id: str, *, size: int = 5000) -> list[dict[str, Any]]:
        if self.mock:
            return list(mock_data.MOCK_AUDIO.get(video_id, []))
        body = {
            "size": size,
            "query": {"term": {"video_id": video_id}},
            "sort": [{"start": "asc"}],
        }
        data = await self._search(self.s.idx_audio, body)
        return [h["_source"] for h in data.get("hits", {}).get("hits", [])]

    # ---- Mock helpers --------------------------------------------------
    def _mock_ocr(
        self,
        queries_vi: list[str],
        queries_folded: list[str],
        *,
        exact_phrases: list[str],
        numbers: list[str],
        hour: int | None,
        clock: str | None,
        size: int,
    ) -> list[dict]:
        variants = [
            _OCR_TOKEN_RE.findall(fold_vietnamese(query))
            for query in [*queries_vi, *queries_folded, *exact_phrases]
            if query
        ]
        required_numbers = _safe_numbers(
            numbers,
            [*queries_vi, *queries_folded, *exact_phrases],
            hour=hour,
            clock=clock,
        )
        out = []
        for kf_id, rec in mock_data.MOCK_OCR.items():
            text = fold_vietnamese(rec["text_clean"])
            if hour is not None and rec.get("hour") != hour:
                continue
            if clock and rec.get("clock") != clock:
                continue
            if required_numbers and not all(
                re.search(rf"(?<!\w){re.escape(number)}(?!\w)", text)
                for number in required_numbers
            ):
                continue
            if variants and not any(tokens and all(token in text for token in tokens) for tokens in variants):
                continue

            phrase_match = any(
                fold_vietnamese(phrase) in text for phrase in exact_phrases if phrase
            )
            vid = kf_id.split("/")[1]
            out.append(
                {
                    "submit_keyframe_id": kf_id,
                    "video_id": vid,
                    "keyframe_n": int(kf_id.split("/")[2]),
                    "score": 10.0 if phrase_match else 5.0,
                    "text_clean": rec["text_clean"],
                    "clock": rec.get("clock"),
                    "hour": rec.get("hour"),
                    "boxes": [],
                }
            )
        out.sort(key=lambda item: (-item["score"], item["submit_keyframe_id"]))
        return out[:size]

    def _mock_speech(self, queries: list[str], *, size: int) -> list[dict]:
        terms = [w.lower() for q in queries for w in q.split()]
        out = []
        for video_id, segs in mock_data.MOCK_SPEECH.items():
            for seg in segs:
                text = seg["text"].lower()
                if terms and not any(t in text for t in terms):
                    continue
                kf = mock_data.make_submit_keyframe_id(
                    video_id, _nearest_kf_n(video_id, seg["center_time"])
                )
                mult = speech_score_multiplier(seg.get("confidence_bucket"), seg.get("segment_role"))
                out.append(
                    {
                        "submit_keyframe_id": kf,
                        "video_id": video_id,
                        "keyframe_n": int(kf.split("/")[2]),
                        "score": 4.0 * mult,
                        "text": seg["text"],
                        "start": seg["start"],
                        "end": seg["end"],
                        "confidence_bucket": seg.get("confidence_bucket"),
                        "segment_role": seg.get("segment_role"),
                    }
                )
        out.sort(key=lambda x: -x["score"])
        return out[:size]

    def _mock_audio(self, queries: list[str], *, size: int) -> list[dict]:
        terms = [q.lower() for q in queries if q]
        out = []
        for video_id, windows in mock_data.MOCK_AUDIO.items():
            for w in windows:
                hay = " ".join([w.get("top1_label") or "", *(w.get("tag_labels") or []), w.get("caption") or ""]).lower()
                if terms and not any(t in hay for t in terms):
                    continue
                matched_top1 = any(t in (w.get("top1_label") or "").lower() for t in terms)
                mult = audio_score_multiplier(
                    audio_stoplist_hit=bool(w.get("audio_stoplist_hit")),
                    top1_is_stoplisted=bool(w.get("top1_is_stoplisted")),
                    caption_quality=w.get("caption_quality"),
                    matched_on_top1=matched_top1,
                )
                if mult <= 0:
                    continue
                kf = mock_data.make_submit_keyframe_id(
                    video_id, _nearest_kf_n(video_id, w["center_time"])
                )
                out.append(
                    {
                        "submit_keyframe_id": kf,
                        "video_id": video_id,
                        "keyframe_n": int(kf.split("/")[2]),
                        "score": 4.0 * mult,
                        "top1_label": w.get("top1_label"),
                        "caption": w.get("caption"),
                        "caption_quality": w.get("caption_quality"),
                        "start": w["start"],
                        "end": w["end"],
                        "window_id": w["window_id"],
                    }
                )
        out.sort(key=lambda x: -x["score"])
        return out[:size]


def _nearest_kf_n(video_id: str, t: float) -> int:
    frames = mock_data.MOCK_KEYFRAMES.get(video_id, [])
    if not frames:
        return 1
    nearest = min(frames, key=lambda f: abs(f["pts_time"] - t))
    return nearest["keyframe_n"]
