"""Cumulative-Hint Evaluation: input construction only, never retrieval memory."""
from __future__ import annotations

import copy
import unicodedata

PROTOCOL = {
    "name": "Cumulative-Hint Evaluation",
    "version": 1,
    "construction": "NFC/whitespace-equivalent reviewed hints_vi; newline-joined prefixes; full once",
    "state": "fresh POST per query/level/variant; previous_hints=[]; no retrieval_id or replaces",
    "partial_primary": "video recall",
    "partial_secondary": "moment recall and strict task success against full-query labels",
    "full_primary": "strict task success: moment + accepted QA answer / complete ordered TRAKE sequence",
    "eligible_tasks": ["T-KIS", "QA"],
    "minimum_hints": 2,
}


def normalized_query(text):
    # Preserve case and punctuation: accept formatting changes, not changed facts.
    return " ".join(unicodedata.normalize("NFC", text).split())


def validate_hints(query):
    hints = query.get("hints_vi")
    if hints is None:
        return []
    if not isinstance(hints, list) or any(not isinstance(h, str) or not h.strip() for h in hints):
        raise ValueError(f"{query['query_id']}: hints_vi must be a list of nonempty reviewed strings")
    if hints and normalized_query("\n".join(hints)) != normalized_query(query["query"]):
        raise ValueError(f"{query['query_id']}: hints_vi do not match reviewed query; audit before evaluation")
    return hints


def hint_stages(query, mode="full"):
    hints = validate_hints(query)
    if mode not in {"full", "cumulative"}:
        raise ValueError("hint_mode must be full or cumulative")
    count = len(hints)
    levels = range(1, count + 1) if mode == "cumulative" and count else [count or None]
    result = []
    for level in levels:
        stage = copy.deepcopy(query)
        stage.update(
            base_query_id=query["query_id"],
            hint_stage="full" if level == count or not count else f"h{level}",
            hint_level=level, hint_count=count or None,
            hint_fraction=level / count if count else None,
            is_full_hint=level == count or not count,
            previous_hints=[],
        )
        if mode == "cumulative" and count:
            stage["query"] = "\n".join(hints[:level])
        result.append(stage)
    return result


def build_plan(queries, mode, variants, *, limit=None):
    selected, excluded = [], []
    for query in queries:
        hints = validate_hints(query)
        if mode == "cumulative":
            reason = ("unsupported task for partial-query evaluation" if query.get("query_type", "T-KIS") not in PROTOCOL["eligible_tasks"]
                      else "requires at least two reviewed hints_vi" if len(hints) < 2 else None)
            if reason:
                excluded.append({"query_id": query["query_id"], "reason": reason})
                continue
        selected.extend(hint_stages(query, mode))
    if limit is not None:
        ids = list(dict.fromkeys(q["query_id"] for q in selected))[:limit]
        selected = [q for q in selected if q["query_id"] in ids]
    if not selected:
        raise ValueError("No eligible queries in selected split/hint mode")
    # This file is an offline audit, not input to the backend. No labels are needed.
    fields = ("query_id", "base_query_id", "query", "query_type", "split", "hint_stage",
              "hint_level", "hint_count", "hint_fraction", "is_full_hint")
    plan = {"hint_mode": mode, "protocol": PROTOCOL, "variants": variants, "limit": limit,
            "base_queries": len({q["query_id"] for q in selected}),
            "stage_queries": len(selected), "planned_runs": len(selected) * len(variants),
            "excluded": excluded, "stages": [{k: q[k] for k in fields if k in q} for q in selected]}
    return selected, plan


def run_key(row):
    return row["query_id"], row.get("hint_stage", "full"), row["variant"]


def stage_metadata(query):
    return {key: query[key] for key in ("base_query_id", "hint_stage", "hint_level", "hint_count",
                                       "hint_fraction", "is_full_hint")}
