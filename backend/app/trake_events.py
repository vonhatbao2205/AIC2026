"""TRAKE events as the organisers mark them: E1, E2, E3, …

A statement is split on those markers and on nothing else. A number in an
event's prose — "(1)", "(4)", "cột số 4." — is part of its text: splitting on
bare list indices as well counted those as extra events.

An "E2" is a marker where a marker stands: at the start of the statement, of a
line or of a sentence (after . ! ? ; ,), or followed by a colon. Mentioned in
the prose ("giống như ở E1", "(E1)") it is text. The numbers themselves are not
checked: statements are typed by hand, and a real pack marked four events
E1 / E2 / E2 / E4. The organisers write "E1: …", and a real pack wrote
"E1 Khoảnh khắc…" with only a space, so the colon is optional. The text before
the first marker is the scene's context, not an event.

The console counts events with the same rule (frontend/src/lib/questions.ts).
"""
from __future__ import annotations

import re

_MARKER = re.compile(r"(?<![^\W_])E\d{1,2}(\s*:|\s*[.)\]–—-]|(?=\s)|$)", re.IGNORECASE)
#: Left over from a marker ("E1 : " "E1.-"), before the event's text.
_LEAD = " \t\n:.)]–—-"


def _is_marker(text: str, mark: re.Match) -> bool:
    if ":" in mark.group(1):
        return True
    before = text[:mark.start()].rstrip(" \t")
    return not before or before[-1] in "\n.!?;,"


def split_marked_events(text: str) -> list[str] | None:
    """The events of a statement marked E1, E2, …, or None with fewer than two markers."""
    text = text or ""
    marks = [mark for mark in _MARKER.finditer(text) if _is_marker(text, mark)]
    if len(marks) < 2:
        return None
    events = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        # The end keeps the event's own punctuation: "(3)." is text, not a marker.
        events.append(text[mark.end():end].lstrip(_LEAD).rstrip(" \t\n,;"))
    return events if all(events) else None


def marked_events_block(text: str) -> str:
    """The events spelt out for a model (the query LLM, the Codex/Claude agents),
    or "" when the statement does not mark them."""
    events = split_marked_events(text)
    if not events:
        return ""
    listed = "\n".join(f"E{i}: {event}" for i, event in enumerate(events, start=1))
    return (
        f"The organisers mark the events E1, E2, …: there are exactly {len(events)}, in this order. "
        "A number inside an event's text, such as (1) or (4), is part of that event, not another one.\n"
        f"{listed}"
    )


def shared_context(text: str) -> str:
    """Keep the literal preamble; remove only a trailing event-list instruction."""
    if not split_marked_events(text):
        return ""
    first = next(m for m in _MARKER.finditer(text) if _is_marker(text, m))
    prefix = text[:first.start()].strip()
    prefix = re.sub(
        r"(?:[\s,;:]*)(?:(?:hãy\s+)?tìm\s+(?:các\s+)?(?:khoảnh khắc|hành động|sự kiện)"
        r"|các\s+(?:khoảnh khắc|hành động|sự kiện)|(?:find|locate)\s+(?:the\s+)?(?:following\s+)?(?:events|moments))"
        r"\s*(?:sau(?:\s+đây)?)?\s*[:：.]*\s*$", "", prefix, flags=re.IGNORECASE,
    )
    return prefix.strip(" \t\n,;:")


def contextual_event_queries(parsed: dict, event: dict) -> list[str]:
    """Context AND event in every variant; never a separate MAX-fused query."""
    trake = parsed.get("trake") or {}
    context = (trake.get("shared_context_en_visual") or trake.get("shared_context_vi")
               or shared_context(parsed.get("original_query") or ""))
    context = " ".join(context.split()).strip(" .;")
    variants = event.get("image_pe_queries_en") or [event.get("description_en_visual") or event.get("description_vi") or ""]
    result = []
    for variant in variants:
        text = " ".join(str(variant).split())
        if not text:
            continue
        if context and context.casefold() not in text.casefold():
            text = f"{context}. {text}".strip()
        if text and text not in result:
            result.append(text)
    return result


def compose_event_queries(parsed: dict) -> None:
    for event in (parsed.get("trake") or {}).get("events", []):
        queries = contextual_event_queries(parsed, event)
        event["image_pe_queries_en"] = queries
        event["description_en_visual"] = queries[0] if queries else ""
