"""Offline paper tables, paired bootstrap intervals and optional exportable figures.

python -m benchmarks.agent.paper_report --run-dir <output> [--plots]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from .hints import run_key
from .metrics import percentile, summarize
from .run_benchmark import write_report


def bootstrap(values, *, samples, rng):
    if not values:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return {"n": len(values), "mean": statistics.mean(values),
            "ci95": [percentile(means, .025), percentile(means, .975)]}


def paired_intervals(rows, variants, *, samples=2000, seed=2026):
    """Resample matched base queries, never treat candidates as independent units."""
    present = {}
    for row in rows:
        present.setdefault(row["query_id"], {})[row["variant"]] = row["metrics"]
    ids = sorted(q for q, systems in present.items() if set(systems) >= set(variants))
    rng = random.Random(seed)
    metrics = ("r1", "r5", "mrr", "video_r1", "video_r5", "moment_r1", "moment_r5",
               "agents_per_query", "tool_calls", "jev_calls", "latency_s")
    def value(row, metric):
        if metric.startswith("video_r"):
            return int(row["video_rank"] is not None and row["video_rank"] <= int(metric[-1]))
        if metric == "agents_per_query":
            return row["codex_calls"] + row["claude_calls"]
        return row.get(metric, row["r" + metric[-1]] if metric.startswith("moment_r") else None)
    summary, differences = {}, {}
    for variant in sorted(variants):
        summary[variant] = {metric: bootstrap([value(present[q][variant], metric) for q in ids], samples=samples, rng=rng)
                            for metric in metrics}
        if "A" in variants and variant != "A":
            differences[f"{variant}-A"] = {
                metric: bootstrap([value(present[q][variant], metric) - value(present[q]["A"], metric) for q in ids], samples=samples, rng=rng)
                for metric in metrics}
    return {"query_ids": ids, "summary": summary, "paired_differences": differences}


def plot_hint_curves(output, rows, variants):
    # Matplotlib is an optional analysis dependency, never imported by the runner.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hint = json.loads((output / "summary_by_hint.json").read_text())
    cohort = set(hint["solve"]["query_ids"])
    rows = [r for r in rows if r["query_id"] in cohort]
    counts = sorted({r["hint_count"] for r in rows})
    if not counts:
        return []
    files = []
    labels = {"video_r1": "Video R@1", "video_r5": "Video R@5", "moment_r1": "Moment R@1",
              "r1": "Strict R@1", "agents_per_query": "Agent calls/query", "tool_calls": "Tool calls/query",
              "jev_calls": "Jev calls/query", "latency_p50_s": "Median latency"}
    for figure, metrics, ylabel in (
        ("hint_recall", ("video_r1", "video_r5", "moment_r1", "r1"), "Recall / strict success"),
        ("hint_computation", ("agents_per_query", "tool_calls", "jev_calls", "latency_p50_s"), "Computation per independent run"),
    ):
        fig, axes = plt.subplots(len(counts), len(metrics), figsize=(3.3 * len(metrics), 2.8 * len(counts)), squeeze=False)
        for i, count in enumerate(counts):
            for j, metric in enumerate(metrics):
                ax = axes[i][j]
                for variant in sorted(variants):
                    ys = []
                    for h in range(1, count + 1):
                        group = [r for r in rows if r["hint_count"] == count and r["hint_level"] == h and r["variant"] == variant]
                        s = summarize(group)[variant]
                        ys.append(s["codex_calls"] + s["claude_calls"] if metric == "agents_per_query" else s[metric])
                    ax.plot(range(1, count + 1), ys, marker="o", label=variant)
                n = len({r["query_id"] for r in rows if r["hint_count"] == count})
                ax.set_title(f"{labels[metric]}: H={count}, N={n}")
                ax.set_xticks(range(1, count + 1), [f"H{h}" if h < count else "Full" for h in range(1, count + 1)])
                ax.set_xlabel("Cumulative input hints (fresh run)")
                ax.set_ylabel("Seconds" if metric == "latency_p50_s" else "Calls" if figure == "hint_computation" else ylabel)
                if figure == "hint_recall":
                    ax.set_ylim(0, 1.02)
                ax.grid(alpha=.2)
                ax.legend()
        mode = hint.get("runtime_mode", "unspecified")
        if mode != "live":
            fig.suptitle(f"Cumulative-Hint Evaluation — {mode.upper()} (harness validation)", fontsize=12)
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            path = output / f"{figure}.{suffix}"
            fig.savefig(path, dpi=240, bbox_inches="tight")
            files.append(path.name)
        plt.close(fig)
    return files


def export_paper_report(output, *, samples=2000, seed=2026, plots=False):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    variants = manifest["config"]["variants"]
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines() if line.strip()]
    if len({run_key(r) for r in rows}) != len(rows):
        raise ValueError("Duplicate query/level/variant results")
    write_report(output, rows)
    full = [r for r in rows if r.get("hint_stage", "full") == "full"]
    runtime_mode = manifest["config"].get("mode", "unspecified")
    intervals = {"method": "percentile paired bootstrap, base-query sampling unit; exploratory 95% intervals",
                 "runtime_mode": runtime_mode,
                 "samples": samples, "seed": seed, "full": paired_intervals(full, variants, samples=samples, seed=seed),
                 "by_task": {task: paired_intervals([r for r in full if r.get("query_type", "T-KIS") == task], variants, samples=samples, seed=seed)
                             for task in sorted({r.get("query_type", "T-KIS") for r in full})}}
    if manifest["config"].get("hint_mode") == "cumulative":
        hint = json.loads((output / "summary_by_hint.json").read_text())
        complete = set(hint["solve"]["query_ids"])
        intervals["hint_levels"] = {
            level: paired_intervals([r for r in rows if r["hint_stage"] == level and r["query_id"] in complete], variants, samples=samples, seed=seed)
            for level in hint["complete_cohort_levels"]}
    (output / "paper_intervals.json").write_text(json.dumps(intervals, indent=2, allow_nan=False) + "\n")
    fields = ["task", "variant", "queries", "r1", "r5", "mrr", "video_r1", "video_r5", "moment_r1",
              "agent_rescue_at_1", "latency_p50_s", "latency_p95_s", "codex_calls", "claude_calls", "jev_calls",
              "tool_calls", "cost_usd_mean", "cost_usd_coverage", "failed", "agent_failed", "fallback"]
    with (output / "paper_tables.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fields, extrasaction="ignore")
        writer.writeheader()
        for task in ["all", *intervals["by_task"]]:
            selected = full if task == "all" else [r for r in full if r.get("query_type", "T-KIS") == task]
            ids = set(intervals["full"]["query_ids"] if task == "all" else intervals["by_task"][task]["query_ids"])
            writer.writerows({"task": task, "variant": v, **s} for v, s in summarize([r for r in selected if r["query_id"] in ids]).items())
    lines = ["# CAD-VR paper tables", "",
             f"Runtime mode: {runtime_mode}. Mock runs validate the harness only and are not paper accuracy results.", "",
             "Only queries complete across all configured variants are included. Failed attempts remain in the denominator.",
             "95% percentile bootstrap intervals resample base queries; differences against A use matched queries. "
             "Intervals are exploratory, with no multiple-comparison correction.", "",
             "| Variant | N | R@1 [95% CI] | R@5 [95% CI] | MRR [95% CI] | ΔR@1 vs A [95% CI] |",
             "|---|---:|---|---|---|---|"]
    def fmt(s):
        if s["mean"] is None:
            return "unknown"
        return f"{s['mean']:.4f} [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]"
    for v, metrics in intervals["full"]["summary"].items():
        delta = intervals["full"]["paired_differences"].get(f"{v}-A", {}).get("r1")
        lines.append(f"| {v} | {metrics['r1']['n']} | {fmt(metrics['r1'])} | {fmt(metrics['r5'])} | {fmt(metrics['mrr'])} | {fmt(delta) if delta else '—'} |")
    artifacts = plot_hint_curves(output, rows, variants) if plots and "hint_levels" in intervals else []
    lines.extend(["", "See paper_tables.csv for matched full/task tables, paper_intervals.json for per-task and per-level intervals, "
                  "and CUMULATIVE_HINT_REPORT.md for hints-to-solve definitions and censoring." if "hint_levels" in intervals else
                  "See paper_tables.csv and paper_intervals.json for matched task tables and intervals."])
    if artifacts:
        lines.extend(["", "Figures: " + ", ".join(artifacts)])
    (output / "PAPER_REPORT.md").write_text("\n".join(lines) + "\n")
    return intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--plots", action="store_true", help="Export PDF/PNG curves; requires matplotlib")
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    export_paper_report(args.run_dir, samples=args.bootstrap_samples, seed=args.seed, plots=args.plots)


if __name__ == "__main__":
    main()
