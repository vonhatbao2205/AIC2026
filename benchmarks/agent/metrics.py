"""Moment- and video-level metrics; missing telemetry is unknown, never zero."""
from __future__ import annotations

import statistics
import unicodedata


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split())


def moment_matches(frame: dict, target: dict, tolerance_s: float) -> bool:
    if frame.get("video_id") != target["video_id"]:
        return False
    if target.get("submit_keyframe_id"):
        return frame.get("submit_keyframe_id") == target["submit_keyframe_id"]
    if target.get("start_s") is not None:
        time = frame.get("time") if frame.get("time") is not None else frame.get("pts_time")
        return time is not None and target["start_s"] - tolerance_s <= time <= target.get("end_s", target["start_s"]) + tolerance_s
    if target.get("frame_idx") is not None:
        fps = target.get("fps") or frame.get("fps")
        # Frame-index matching still works without guessing an FPS.
        if frame.get("frame_idx") is not None:
            return abs(frame["frame_idx"] - target["frame_idx"]) <= (tolerance_s * fps if fps else 0)
        time = frame.get("time") if frame.get("time") is not None else frame.get("pts_time")
        return bool(fps and time is not None and abs(time - target["frame_idx"] / fps) <= tolerance_s)
    raise ValueError("Ground truth must include a keyframe, frame index or temporal interval")


def correct(frame, query, tolerance_s=1.0):
    for target in query["targets"]:
        if moment_matches(frame, target, tolerance_s):
            if query.get("query_type") != "QA" or normalized(frame.get("answer")) in {
                normalized(a) for a in target.get("answers", [])}:
                return True
    return False


def first_rank(ranking, query, tolerance_s=1.0):
    if query.get("query_type") == "TRAKE":
        # Rank complete videos, with every event in one consistent alternative.
        videos = list(dict.fromkeys(f["video_id"] for f in ranking))
        for rank, video in enumerate(videos, 1):
            for sequence in query.get("sequences", []):
                if any(t["video_id"] != video for t in sequence):
                    continue
                previous = -1.0
                matched = True
                for event, target in enumerate(sequence, 1):
                    chosen = next((f for f in ranking if f["video_id"] == video and f.get("event") == event), None)
                    time = None if chosen is None else (chosen.get("time") if chosen.get("time") is not None else chosen.get("pts_time"))
                    if chosen is None or time is None or time <= previous or not moment_matches(chosen, target, tolerance_s):
                        matched = False
                        break
                    previous = time
                if matched:
                    return rank
        return None
    return next((i for i, frame in enumerate(ranking, 1) if correct(frame, query, tolerance_s)), None)


def video_rank(ranking, query):
    targets = {t["video_id"] for t in query["targets"]}
    return next((i for i, video in enumerate(dict.fromkeys(f["video_id"] for f in ranking), 1) if video in targets), None)


def calibration_metrics(rows: list[tuple[float, int]], bins: int = 10):
    if not rows:
        return {"n": 0, "ece": None, "brier": None}
    ece = 0.0
    for i in range(bins):
        bucket = [(p, y) for p, y in rows if min(bins - 1, int(p * bins)) == i]
        if bucket:
            ece += len(bucket) / len(rows) * abs(statistics.mean(p for p, _ in bucket) - statistics.mean(y for _, y in bucket))
    return {"n": len(rows), "ece": ece, "brier": statistics.mean((p - y) ** 2 for p, y in rows)}


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)


def measure(query, snapshot, trace, *, wall_s, tolerance_s=1.0):
    ranking, baseline = snapshot.get("ranking", []), snapshot.get("baseline_ranking", [])
    rank = first_rank(ranking, query, tolerance_s)
    base_rank = first_rank(baseline, query, tolerance_s)
    correct_at = [row["at_s"] for row in trace if row.get("kind") == "ranking" and first_rank(row["ranking"], query, tolerance_s) == 1]
    agents = snapshot.get("agents", {})
    calls = snapshot.get("metrics", {}).get("agent_calls", {})
    active = [state for name, state in agents.items() if calls.get(name, 0) > 0]
    jev = snapshot.get("controller", {}).get("decision_calls", snapshot.get("controller", {}).get("jev_calls", []))
    costs = [a.get("cost_usd") for a in active] + [a.get("cost_usd") for a in jev]
    tokens = [None if a.get("input_tokens") is None or a.get("output_tokens") is None else a["input_tokens"] + a["output_tokens"] for a in active + jev]
    # One final observation per candidate, so repeated verification cannot inflate N.
    calibration = []
    if query.get("query_type") != "TRAKE":
        calibration = [(f["probability"], int(correct(f, query, tolerance_s))) for f in ranking if f.get("probability") is not None]
    def agent_hit(k):
        if not any(calls.values()) or rank is None or rank > k:
            return False
        if query.get("query_type") == "TRAKE":
            videos = list(dict.fromkeys(f["video_id"] for f in ranking))
            return any(f.get("agent_sources") for f in ranking if f["video_id"] == videos[rank - 1])
        return bool(ranking[rank - 1].get("agent_sources"))

    controller = snapshot.get("controller", {})
    agent_failed = any(a.get("status") in {"failed", "timeout", "cancelled"} for a in active)
    required = {"codex"} if snapshot.get("policy") == "codex" else {"codex", "claude"} if snapshot.get("policy") == "parallel" else set()
    unavailable_required = any(agents.get(name, {}).get("status") == "unavailable" for name in required)
    return {
        "rank": rank, "video_rank": video_rank(ranking, query), "baseline_rank": base_rank,
        "r1": int(rank == 1), "r5": int(rank is not None and rank <= 5), "mrr": 1 / rank if rank else 0.0,
        "rescue1_eligible": base_rank != 1,
        "rescue5_eligible": base_rank is None or base_rank > 5,
        "rescue1": base_rank != 1 and agent_hit(1),
        "rescue5": (base_rank is None or base_rank > 5) and agent_hit(5),
        "improved_at_1": base_rank != 1 and rank == 1,
        "improved_at_5": (base_rank is None or base_rank > 5) and rank is not None and rank <= 5,
        "time_to_first_correct_s": min(correct_at) if correct_at else None,
        "latency_s": wall_s, "tool_calls": snapshot.get("metrics", {}).get("tool_calls", 0),
        "codex_calls": calls.get("codex", 0), "claude_calls": calls.get("claude", 0), "jev_calls": sum(c.get("backend", "jev") == "jev" for c in jev),
        "llm_judge_calls": sum(c.get("backend") == "llm" for c in jev),
        "solved_without_system2": rank == 1 and not any(calls.values()) and not any(c.get("backend") == "llm" for c in jev),
        "cost_usd": sum(costs) if all(c is not None for c in costs) else None,
        "known_cost_usd": sum(c for c in costs if c is not None),
        "tokens": sum(tokens) if all(t is not None for t in tokens) else None,
        "calibration": calibration,
        "agent_disagreement": snapshot.get("metrics", {}).get("agent_disagreement"),
        "verifier": snapshot.get("controller", {}).get("verifier", "jev"),
        "fallback": bool(snapshot.get("controller", {}).get("fallback")),
        "retrieval_failed": bool(controller.get("retrieval_error")) or (not baseline and any(
            "unavailable" in str(w).lower() for event in trace if event.get("kind") == "retrieval"
            for w in event.get("warnings", []))),
        "retrieval_degraded": bool(controller.get("retrieval_degraded")) or any(
            "unavailable" in str(w).lower() for event in trace if event.get("kind") == "retrieval"
            for w in event.get("warnings", [])),
        "agent_failed": agent_failed or unavailable_required,
        "failed": controller.get("status") in {"failed", "timeout", "cancelled"},
    }


def summarize(records: list[dict]):
    groups = {}
    for record in records:
        groups.setdefault(record["variant"], []).append(record["metrics"])
    result = {}
    for name, rows in groups.items():
        n = len(rows)
        out = {"queries": n}
        out["no_cli_rate"] = sum(not r["codex_calls"] and not r["claude_calls"] for r in rows) / n
        out["codex_only_rate"] = sum(bool(r["codex_calls"]) and not r["claude_calls"] for r in rows) / n
        out["claude_only_rate"] = sum(bool(r["claude_calls"]) and not r["codex_calls"] for r in rows) / n
        out["both_agents_rate"] = sum(bool(r["codex_calls"]) and bool(r["claude_calls"]) for r in rows) / n
        for key in ("r1", "r5", "mrr", "tool_calls", "codex_calls", "claude_calls", "jev_calls", "llm_judge_calls", "solved_without_system2", "fallback", "failed", "retrieval_failed", "retrieval_degraded", "agent_failed", "improved_at_1", "improved_at_5"):
            out[key] = statistics.mean(float(r[key]) for r in rows)
        for k in (1, 5):
            denom = sum(r[f"rescue{k}_eligible"] for r in rows)
            out[f"agent_rescue_at_{k}"] = sum(r[f"rescue{k}"] for r in rows) / denom if denom else None
            out[f"rescue{k}_eligible_queries"] = denom
            out[f"video_r{k}"] = sum(r["video_rank"] is not None and r["video_rank"] <= k for r in rows) / n
        out["latency_p50_s"] = percentile([r["latency_s"] for r in rows], .5)
        out["latency_p95_s"] = percentile([r["latency_s"] for r in rows], .95)
        correct_times = [r["time_to_first_correct_s"] for r in rows if r["time_to_first_correct_s"] is not None]
        out["time_to_first_correct_p50_s"] = percentile(correct_times, .5)
        out["time_to_first_correct_observed_queries"] = len(correct_times)
        for key in ("cost_usd", "tokens"):
            known = [r[key] for r in rows if r[key] is not None]
            out[key + "_mean"] = statistics.mean(known) if len(known) == n else None
            out[key + "_coverage"] = len(known) / n
        out["known_cost_usd_mean"] = statistics.mean(r["known_cost_usd"] for r in rows)
        agreements = [r["agent_disagreement"] for r in rows if r.get("agent_disagreement") is not None]
        out["agent_disagreement_rate"] = statistics.mean(agreements) if agreements else None
        out["agent_disagreement_observed_queries"] = len(agreements)
        out["decision_calibration"] = calibration_metrics([pair for r in rows for pair in r["calibration"]])
        out["jev_calibration"] = calibration_metrics([pair for r in rows if r.get("verifier", "jev") == "jev" for pair in r["calibration"]])
        result[name] = out
    return result
