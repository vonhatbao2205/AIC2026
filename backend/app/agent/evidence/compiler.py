"""Deterministic, lossless query constraints (no extra generative model call).

Keep the full statement as a guard against a clause splitter losing relations.
Clauses are deliberately conservative; this is not an entity extraction model.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ...query_parser import heuristic_parse
from ...trake_events import split_marked_events


@dataclass(frozen=True)
class Constraint:
    id: str
    text: str
    kind: str = "semantic"
    weight: float = 1.0
    event: int | None = None
    requires: tuple[str, ...] = ()
    source_span: tuple[int, int] | None = None

    def to_dict(self):
        return asdict(self)


def compile_constraints(query: str, query_type: str) -> list[Constraint]:
    result = [Constraint("query", query.strip(), "global", 2.0)]
    events = split_marked_events(query) if query_type == "TRAKE" else None
    if events:
        result.extend(Constraint(f"event_{i}", text, "temporal", 1.5, i,
                                 ("query",) if i == 1 else ("query", f"event_{i - 1}"))
                      for i, text in enumerate(events, 1))
    else:
        clauses = [s.strip() for s in re.split(r"[;\n]+|(?<=[.!?])\s+", query) if s.strip()]
        if len(clauses) > 1:
            result.extend(Constraint(f"clause_{i}", text, requires=("query",)) for i, text in enumerate(clauses[:7], 1))
        # Grounded surface spans expose conjunction failures even in a single
        # sentence. Keep the whole statement as a guard: these rules never
        # invent entities, infer synonyms or claim a complete semantic parse.
        markers = re.compile(
            r"(?<!\w)(?P<clothing>mặc\s+(?:áo|quần|đồ|bộ)|wearing|dressed in)\b"
            r"|(?<!\w)(?P<action>đang|repairing|holding|riding|walking|running|cooking|jumping)\b"
            r"|(?<!\w)(?P<relation>bên cạnh|phía sau|phía trước|đối diện|next to|behind|in front of|beside)\b"
            r"|(?<!\w)(?P<location>bên đường|trong phòng|trên đường|trên sân|ở ngoài|at the|inside the)\b"
            r"|(?<!\w)(?P<negation>không có|không mặc|không cầm|without|not wearing|not holding)\b",
            re.IGNORECASE,
        )
        marks = list(markers.finditer(query))
        for i, mark in enumerate(marks[:6], 1):
            end = marks[i].start() if i < len(marks) else len(query)
            punctuation = re.search(r"[,;.!?\n]", query[mark.start():end])
            if punctuation:
                end = mark.start() + punctuation.start()
            text = query[mark.start():end].strip()
            if text and text.casefold() != query.strip().casefold():
                result.append(Constraint(f"detail_{i}", text, mark.lastgroup or "semantic",
                                         requires=("query",), source_span=(mark.start(), end)))
    parsed = heuristic_parse(query, query_type, [])
    phrases = parsed["channels"]["ocr"].get("exact_phrases") or []
    for i, text in enumerate(phrases[:3], 1):
        result.append(Constraint(f"ocr_{i}", f'On-screen text: "{text}"', "ocr", requires=("query",)))
    if query_type == "QA":
        result.append(Constraint("answer", "The proposed answer is directly supported by the evidence and answers the question.", "answer", 2.0, requires=("query",)))
    # Never silently truncate required events: incomplete sequences cannot stop.
    return result
