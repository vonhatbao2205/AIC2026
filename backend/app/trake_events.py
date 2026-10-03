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
