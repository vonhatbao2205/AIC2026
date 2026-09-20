"""Pure ledger, pool fusion and bounded memory policies."""
from __future__ import annotations

import copy
import math
import unicodedata
from dataclasses import replace
from typing import Any

from ..types import ChannelHit
from .models import HintInput


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def derive_ledger(hints: list[HintInput], elapsed_ms: int | None = None) -> list[dict]:
    """Only return released, enabled, nonduplicate evidence units.

    A non-append cumulative edit replaces the active description. It does not
    retain contradictory evidence from the superseded text. Raw input history
    is journaled only after release, separately from this effective ledger.
    """
    units: list[dict] = []
    cumulative = ""
    for hint in hints:
        if not hint.enabled:
            continue
        if hint.revealed_at is not None and (elapsed_ms is None or hint.revealed_at > elapsed_ms):
            continue
        text = normalize(hint.raw_text)
        if not text:
            continue
        if hint.input_mode == "cumulative":
            if text == cumulative:
                continue
            if cumulative and text.startswith(cumulative + " "):
                delta = text[len(cumulative):].strip()
            else:
                units = []
                delta = text
            cumulative = text
        else:
            if text == cumulative or any(text == h["delta_text"] for h in units):
                continue
            delta = text
            cumulative = normalize(cumulative + " " + delta)
        units.append({**hint.model_dump(), "delta_text": delta, "cumulative_text": cumulative})
    return units


def rank_evidence(rank: int | None, k: int = 60) -> float:
    return (k + 1) / (k + rank) if rank is not None else 0.0


def memory_score(values: list[float], epsilon: float = .1, arithmetic: bool = False) -> float:
    if not values:
        return 0.0
    smoothed = [epsilon + (1 - epsilon) * v for v in values]
    return sum(smoothed) / len(smoothed) if arithmetic else math.exp(
        sum(math.log(v) for v in smoothed) / len(smoothed)
    )


def merge_local(global_hits: list[ChannelHit], local_hits: list[ChannelHit]) -> list[ChannelHit]:
    """Caller supplies ONE model/query/collection. Local ranks have no authority."""
    channels = {h.channel for h in [*global_hits, *local_hits]}
    if len(channels) > 1:
        raise ValueError("Cannot compare raw scores across channels")
    best: dict[str, ChannelHit] = {}
    for hit in [*global_hits, *local_hits]:
        if not math.isfinite(hit.score):
            continue
        old = best.get(hit.submit_keyframe_id)
        if old is None or hit.score > old.score:
            best[hit.submit_keyframe_id] = hit
    return [replace(copy.deepcopy(h), rank=i) for i, h in enumerate(sorted(
        best.values(), key=lambda h: (-h.score, h.submit_keyframe_id)
    ))]


def representatives(frames: list[dict], limit: int = 3, separation_s: float = 2.0) -> list[dict]:
    result: list[dict] = []
    for frame in frames:
        t = frame.get("pts_time")
        if any(frame["submit_keyframe_id"] == f["submit_keyframe_id"] or (
            t is not None and f.get("pts_time") is not None and abs(t - f["pts_time"]) < separation_s
        ) for f in result):
            continue
        result.append(copy.deepcopy(frame))
        if len(result) == limit:
            break
    return result


def bounded_candidates(scores: dict[str, float], current: list[str], previous: list[str], limit: int) -> list[str]:
    protected = set(current[:20]) | set(previous[:20])
    ordered = sorted(scores, key=lambda v: (-scores[v], v))
    keep = set(v for v in ordered if v in protected)
    for video in ordered:
        if len(keep) >= limit:
            break
        keep.add(video)
    return [v for v in ordered if v in keep]


def rescue_pairs(previous: list[str], current: list[str], observed: set[str], old_observed: list[set[str]],
                 survivor_limit: int = 8, newcomer_limit: int = 4, budget: int = 16) -> list[tuple[str, int, str]]:
    """Indices address active hint traces. Allocation never uses relevance labels."""
    latest = len(old_observed)
    pairs = [(v, latest, "rescue") for v in previous if v not in observed][:survivor_limit]
    newcomers = [v for v in current if v not in set(previous)][:newcomer_limit]
    # Round-robin over hint age avoids one newcomer consuming the whole budget.
    for i in range(latest):
        pairs.extend((v, i, "backfill") for v in newcomers if v not in old_observed[i])
    return pairs[:budget]


def stability(previous: list[str], current: list[str], history: list[list[str]]) -> dict[str, Any]:
    top = current[0] if current else None
    streak = 0
    for ranking in reversed([*history, current]):
        if top is None or not ranking or ranking[0] != top:
            break
        streak += 1
    # Explicit top-10 Jaccard, not an approximation mislabeled as RBO.
    a, b = set(previous[:10]), set(current[:10])
    return {"top1_changed": bool(previous and current and previous[0] != current[0]),
            "top1_streak": streak, "top10_jaccard": len(a & b) / len(a | b) if a | b else None}
