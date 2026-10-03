"""Reports independent cumulative queries; never feeds results into another run."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .metrics import summarize

METHOD_NOTE = (
    "Each cumulative hint level is evaluated as an independent query from a fresh system state; "
    "no cross-hint memory or progressive retrieval mechanism is used."
)


def _dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _paired(rows, variants):
    present = {}
    for row in rows:
        present.setdefault(row["query_id"], set()).add(row["variant"])
    ids = sorted(q for q, found in present.items() if found >= set(variants))
    return {"query_ids": ids, "summary": summarize([r for r in rows if r["query_id"] in ids])}


def _group(rows, key, variants):
    groups = {}
    for row in rows:
        groups.setdefault(str(key(row)), []).append(row)
    return {level: {"summary": summarize(group), "paired": _paired(group, variants)}
            for level, group in sorted(groups.items(), key=lambda item: (1, 0) if item[0] == "full" else (0, int(item[0].removeprefix("h"))))}


def _solve(rows, variants, planned_counts):
    by_query = {}
    for row in rows:
        by_query.setdefault(row["query_id"], {})[(row["variant"], row["hint_level"])] = row
    complete = []
    for qid, count in planned_counts.items():
        expected = {(v, h) for v in variants for h in range(1, count + 1)}
        if set(by_query.get(qid, {})) >= expected:
            complete.append(qid)
    result, observations = {}, []
    for variant in sorted(variants):
        metrics = {}
        for target in ("video", "moment", "strict"):
            values = []
            for qid in sorted(complete):
                count = planned_counts[qid]
                stages = [by_query[qid][variant, h] for h in range(1, count + 1)]
                def success(row):
                    m = row["metrics"]
                    return (m["video_rank"] == 1 if target == "video" else
                            m.get("moment_r1", m["r1"]) == 1 if target == "moment" else m["r1"] == 1)
                outcomes = [success(row) for row in stages]
                first = next((h for h, solved in enumerate(outcomes, 1) if solved), None)
                value = {"base_query_id": qid, "variant": variant, "criterion": target,
                         "hint_count": count, "first_solved_hint": first,
                         "solved_before_full": first is not None and first < count,
                         "hint_saving": (count - first) / (count - 1) if first is not None else 0.0,
                         "full_solved": outcomes[-1],
                         "regressed_after_solve": first is not None and not all(outcomes[first - 1:]),
                         "outcomes": [int(hit) for hit in outcomes]}
                values.append(value)
                observations.append(value)
            n = len(values)
            solved = [v["first_solved_hint"] for v in values if v["first_solved_hint"] is not None]
            metrics[target] = {
                "queries": n, "solved_queries": len(solved), "unsolved_queries": n - len(solved),
                "mean_hints_to_solve_solved_only": statistics.mean(solved) if solved else None,
                "median_hints_to_solve_solved_only": statistics.median(solved) if solved else None,
                "mean_hints_to_solve_penalized": statistics.mean(
                    v["first_solved_hint"] if v["first_solved_hint"] is not None else v["hint_count"] + 1 for v in values) if n else None,
                "early_solve_rate": sum(v["solved_before_full"] for v in values) / n if n else None,
                "mean_hint_saving": statistics.mean(v["hint_saving"] for v in values) if n else None,
                "full_solve_rate": sum(v["full_solved"] for v in values) / n if n else None,
                "regression_rate": sum(v["regressed_after_solve"] for v in values) / n if n else None,
            }
        result[variant] = metrics
    return {"planned_base_queries": len(planned_counts), "complete_base_queries": len(complete),
            "query_ids": sorted(complete), "summary": result, "per_query": observations}


def write_hint_report(output, rows, variants):
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    runtime_mode = manifest.get("config", {}).get("mode", "unspecified")
    plan_path = output / "run_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        counts = {q["query_id"]: q["hint_count"] for q in plan["stages"]}
    else:
        counts = {r["query_id"]: r["hint_count"] for r in rows}
    counts = {q: n for q, n in counts.items() if n and n >= 2}
    levels = _group(rows, lambda r: r["hint_stage"], variants)
    solve = _solve(rows, variants, counts)
    complete_ids = set(solve["query_ids"])
    complete_rows = [r for r in rows if r["query_id"] in complete_ids]
    report = {
        "methodology": METHOD_NOTE,
        "runtime_mode": runtime_mode,
        "partial_primary": "video_r1/video_r5; full-query moment labels are secondary",
        "strict_metric": "r1/r5/mrr require accepted QA answer or complete TRAKE sequence as applicable",
        "levels": levels,
        "complete_cohort_levels": _group(complete_rows, lambda r: r["hint_stage"], variants),
        "by_numeric_hint_level": _group(rows, lambda r: r["hint_level"], variants),
        "by_task": {task: _group([r for r in rows if r.get("query_type", "T-KIS") == task], lambda r: r["hint_stage"], variants)
                    for task in sorted({r.get("query_type", "T-KIS") for r in rows})},
        "by_total_hint_count": {str(n): _group([r for r in rows if r["hint_count"] == n], lambda r: r["hint_level"], variants)
                                for n in sorted(set(counts.values()))},
        "solve": solve,
    }
    _dump(output / "summary_by_hint.json", report)
    _dump(output / "hint_solve.json", solve)
    fields = ["query_id", "variant", "hint_stage", "hint_level", "hint_count", "hint_fraction", "query_type",
              "r1", "r5", "mrr", "moment_r1", "moment_r5", "video_rank", "latency_s", "codex_calls", "claude_calls",
              "jev_calls", "llm_judge_calls", "tool_calls", "cost_usd", "tokens", "failed", "agent_failed", "fallback"]
    with (output / "hint_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, **r["metrics"]} for r in rows)
    lines = ["# Cumulative-Hint Evaluation", "", METHOD_NOTE, "",
             f"Runtime mode: {runtime_mode}. Mock runs validate the harness only and are not paper accuracy results.", "",
             "Partial queries: video recall is primary; moment recall uses the full-query labels and is secondary. "
             "Strict success additionally requires the accepted QA answer. Full uses the existing strict task metric.", "",
             "The table uses a cohort complete across every level and configured variant. Failed attempts remain in the denominator. "
             "Per-level available/paired cohorts, task breakdowns and total-hint-count strata are in summary_by_hint.json.",
             f"Complete base queries: {solve['complete_base_queries']}/{solve['planned_base_queries']}.", "",
             "| Level | Variant | N | Video R@1 | Video R@5 | Moment R@1 | Strict R@1 | Strict R@5 | Agents/query | Tools/query | Jev/query | p50 s | p95 s | $/query |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "unknown" if value is None else str(value) if isinstance(value, int) else f"{value:.4f}"
    for level, group in report["complete_cohort_levels"].items():
        for variant, s in sorted(group["summary"].items()):
            numbers = [s["queries"], s["video_r1"], s["video_r5"], s["moment_r1"], s["r1"], s["r5"],
                       s["codex_calls"] + s["claude_calls"], s["tool_calls"], s["jev_calls"],
                       s["latency_p50_s"], s["latency_p95_s"], s["cost_usd_mean"]]
            lines.append(f"| {level.upper()} | {variant} | " + " | ".join(fmt(v) for v in numbers) + " |")
    lines.extend(["", "H3/other numeric columns can contain fewer queries because the last level is named Full. "
                  "Use the total-hint-count strata for numeric curves with a fixed cohort.", "",
                  "Minimum hints are computed from independent outcomes and do not assume monotonic success. "
                  "Unsolved queries are censored (null h*); conditional means show solved N. "
                  "The penalized mean assigns H+1 to unsolved queries. Early solve and mean hint saving use all complete queries; "
                  "unsolved queries receive zero saving. A regression means a later level failed after an earlier success.", "",
                  "| Criterion | Variant | N | Solved N | Mean h* (solved) | Median h* (solved) | Mean h* (penalized) | Early solve | Hint saving | Full success | Regression |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for variant, criteria in solve["summary"].items():
        for criterion, s in criteria.items():
            values = [s[k] for k in ("queries", "solved_queries", "mean_hints_to_solve_solved_only",
                                     "median_hints_to_solve_solved_only", "mean_hints_to_solve_penalized",
                                     "early_solve_rate", "mean_hint_saving", "full_solve_rate", "regression_rate")]
            lines.append(f"| {criterion} | {variant} | " + " | ".join(fmt(v) for v in values) + " |")
    lines.extend(["", "Cost/tokens cover the agent and decision layers; missing CLI usage stays unknown. "
                  "No empirical improvement is claimed until live runs complete."])
    (output / "CUMULATIVE_HINT_REPORT.md").write_text("\n".join(lines) + "\n")
    return report
