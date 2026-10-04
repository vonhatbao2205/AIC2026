"""Offline video grouping reports from saved benchmark rankings; no model calls.

Run from the repository root: python -m tools.benchmark_video_report --help.
Kept outside benchmark runtime modules so exporting a report does not change
the code fingerprint used to resume an ongoing benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.agent.metrics import correct, first_rank, video_rank
from benchmarks.agent.run_benchmark import load_queries, quota_interrupted
from benchmarks.agent.hints import run_key
from benchmarks.agent.hint_report import _solve


def grouped_rank(ranking, query, tolerance_s, *, strict=False):
    """Video rank whose returned frame pool contains the required evidence.

    Group order is the first occurrence in the saved frame ranking. QA moment
    coverage ignores the answer; strict coverage also requires an accepted
    answer. TRAKE retains the existing complete ordered sequence semantics.
    """
    groups = defaultdict(list)
    for frame in ranking:
        groups[frame["video_id"]].append(frame)
    for rank, frames in enumerate(groups.values(), 1):
        if query.get("query_type") == "TRAKE":
            hit = first_rank(frames, query, tolerance_s) == 1
        else:
            evaluated = query if strict else {**query, "query_type": "T-KIS"}
            hit = any(correct(frame, evaluated, tolerance_s) for frame in frames)
        if hit:
            return rank
    return None


def score_row(row, query, tolerance_s):
    ranking = row.get("snapshot", {}).get("ranking", [])
    return {
        "query_id": row["query_id"], "variant": row["variant"],
        "query_type": query.get("query_type", "T-KIS"), "hint_stage": row.get("hint_stage", "full"),
        **{key: row.get(key) for key in ("hint_level", "hint_count", "hint_fraction", "is_full_hint")},
        "strict_rank": first_rank(ranking, query, tolerance_s),
        "video_rank": video_rank(ranking, query),
        "group_moment_rank": grouped_rank(ranking, query, tolerance_s),
        "group_strict_rank": grouped_rank(ranking, query, tolerance_s, strict=True),
        "returned_frames": len(ranking), "returned_videos": len({f["video_id"] for f in ranking}),
        "agent_failed": row["metrics"]["agent_failed"], "failed": row["metrics"]["failed"],
    }


def summarize(rows):
    variants = defaultdict(list)
    for row in rows:
        variants[row["variant"]].append(row)
    result = {}
    for variant, records in sorted(variants.items()):
        values = {"queries": len(records), "agent_failed_rate": statistics.mean(r["agent_failed"] for r in records)}
        for metric in ("strict", "video", "group_moment", "group_strict"):
            ranks = [r[metric + "_rank"] for r in records]
            for k in (1, 5):
                values[f"{metric}_r{k}"] = statistics.mean(int(rank is not None and rank <= k) for rank in ranks)
            values[metric + "_mrr"] = statistics.mean(1 / rank if rank else 0 for rank in ranks)
        result[variant] = values
    return result


def paired(rows, variants):
    present = defaultdict(set)
    for row in rows:
        present[row["query_id"]].add(row["variant"])
    ids = sorted(qid for qid, found in present.items() if set(variants) <= found)
    ids_set = set(ids)
    return {"query_ids": ids, "summary": summarize([r for r in rows if r["query_id"] in ids_set])}


def build_levels(rows, variants):
    levels = {}
    for level in sorted({r["hint_stage"] for r in rows}, key=lambda k: (1, 0) if k == 'full' else (0, int(k.removeprefix('h')))):
        selected = [r for r in rows if r["hint_stage"] == level]
        levels[level] = {"available": summarize(selected), "paired": paired(selected, variants),
                         "by_task": {task: paired([r for r in selected if r["query_type"] == task], variants)
                                     for task in sorted({r["query_type"] for r in selected})}}
    return levels


def group_solve(rows, variants, counts):
    # Reuse the existing independent-stage censoring/regression definitions.
    converted = []
    for row in rows:
        converted.append({**row, 'metrics': {'video_rank': row['video_rank'],
                                           'moment_r1': int(row['group_moment_rank'] == 1),
                                           'r1': int(row['group_strict_rank'] == 1)}})
    return _solve(converted, variants, counts)


def plot_group_curves(run_dir):
    """Export fixed-cohort scientific curves for independent hint levels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(run_dir)/'video-report'
    report = json.loads((output/'summary.json').read_text())
    strata = report.get('by_total_hint_count', {})
    strata = {count: levels for count, levels in strata.items() if levels}
    if not strata:
        return []
    metrics = {'video_r1': 'Video R@1', 'group_moment_r1': 'Group moment R@1', 'group_strict_r1': 'Group strict R@1'}
    fig, axes = plt.subplots(len(strata), len(metrics), figsize=(12, 3.2*len(strata)), squeeze=False)
    for i, (count, levels) in enumerate(sorted(strata.items(), key=lambda item: int(item[0]))):
        count = int(count)
        for j, (metric, label) in enumerate(metrics.items()):
            ax = axes[i][j]
            for variant in sorted(report['variants']):
                values = [levels[f'h{h}' if h < count else 'full']['paired']['summary'][variant][metric] for h in range(1, count+1)]
                ax.plot(range(1, count+1), values, marker='o', label=variant)
            n = len(levels['full']['paired']['query_ids'])
            ax.set_title(f'{label}: {count} hints, N={n}')
            ax.set_xticks(range(1,count+1), [f'H{h}' if h < count else 'Full' for h in range(1,count+1)])
            ax.set_ylim(0,1.02)
            ax.set_xlabel('Cumulative query information (fresh run)')
            ax.set_ylabel('Recall')
            ax.grid(alpha=.2)
            ax.legend()
    fig.tight_layout()
    artifacts = []
    for suffix in ['pdf','png']:
        path = output/f'group_hint_recall.{suffix}'
        fig.savefig(path,dpi=240,bbox_inches='tight')
        artifacts.append(path.name)
    plt.close(fig)
    return artifacts


def export(run_dir, dataset, *, tolerance=None, include_legacy_quota=False):
    run_dir, dataset = Path(run_dir), Path(dataset)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    dataset_bytes = dataset.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != manifest["config"]["dataset_sha256"]:
        raise ValueError("Dataset SHA-256 differs from the saved benchmark")
    queries = {q["query_id"]: q for q in load_queries(dataset)}
    tolerance = manifest["config"]["tolerance_s"] if tolerance is None else tolerance
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    path = run_dir / "results.jsonl"
    snapshot_size = path.stat().st_size
    consumed = 0
    digest = hashlib.sha256()
    rows, seen = [], set()
    skipped_quota, incomplete_tail = 0, False
    retained_quota = 0
    with path.open("rb") as file:
        while consumed < snapshot_size:
            line = file.readline(snapshot_size - consumed)
            consumed += len(line)
            digest.update(line)
            if not line.endswith(b"\n"):
                incomplete_tail = True
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if quota_interrupted(row):
                if not include_legacy_quota:
                    skipped_quota += 1
                    continue
                retained_quota += 1
            key = run_key(row)
            if key in seen:
                raise ValueError(f"Duplicate query/level/variant: {key}")
            seen.add(key)
            query = queries[row["query_id"]]
            if query["split"] != row["split"]:
                raise ValueError("Saved split differs from the dataset")
            rows.append(score_row(row, query, tolerance))
    variants = manifest["config"]["variants"]
    levels = build_levels(rows, variants)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "tolerance_s": tolerance,
              "source_directory": str(run_dir.resolve()), "source_fingerprint": manifest["fingerprint"],
              "source_results_snapshot_sha256": digest.hexdigest(), "source_snapshot_bytes": snapshot_size,
              "scored_rows": len(rows), "quota_rows_excluded": skipped_quota,
              "legacy_quota_rows_retained": retained_quota,
              "incomplete_tail_excluded": incomplete_tail, "model_calls": 0, "variants": variants,
              "group_order": "First occurrence of each video in the saved frame ranking; no duplicate video slots",
              "video_recall": "Correct ground-truth video identity only; ignores frame time and QA answer",
              "group_moment_recall": "Any returned matching moment inside the ranked video; ignores QA answer",
              "group_strict_recall": "Any returned matching moment and accepted QA answer inside the ranked video",
              "trake": "Group moment/strict both require the existing complete ordered sequence of selected events",
              "coverage_note": "Grouped coverage scans the full saved candidate list per video, not just its first frame. It is not accuracy of one submitted frame or answer.",
              "levels": levels}
    if manifest['config'].get('hint_mode') == 'cumulative':
        plan = json.loads((run_dir / 'run_plan.json').read_text())
        counts = {s['query_id']: s['hint_count'] for s in plan['stages']}
        report['solve'] = group_solve(rows, variants, counts)
        complete = set(report['solve']['query_ids'])
        complete_rows = [r for r in rows if r['query_id'] in complete]
        report['complete_cohort_levels'] = build_levels(complete_rows, variants)
        report['by_total_hint_count'] = {str(n): build_levels([r for r in complete_rows if r['hint_count'] == n], variants)
                                       for n in sorted(set(counts.values()))}
    output = run_dir / "video-report"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (output / "query_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["query_id", "variant"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Video-group retrieval evaluation", "", f"Moment tolerance: {tolerance:g}s. Runtime mode: {manifest['config'].get('mode', 'unspecified')}.",
             "Offline report from saved rankings; no new model calls. Original results and runtime manifests are preserved.", "",
             "Videos are ordered by their first appearance in the frame ranking. Multiple frames from one video occupy one video slot.",
             "Video R@k checks only video identity. Group moment R@k additionally requires a matching returned frame in a top-k video.",
             "Group strict R@k also requires an accepted QA answer. TRAKE retains complete ordered sequence validation.",
             "Grouped metrics inspect the full returned frame pool inside each video; they measure candidate coverage, not one-frame submission accuracy.",
             "Tables compare only queries complete across all configured variants at each hint level. Failed agent attempts remain in the denominator if present in the source results.", ""]
    if manifest.get('evaluation_note'):
        lines += [manifest['evaluation_note'], '']
    if retained_quota:
        lines += [f'Historical quota-interrupted attempts retained: {retained_quota}. This historical table includes partial agent executions; retry them before treating the combined run as complete without provider interruptions.', '']
    if 'solve' in report:
        lines += [f"Cumulative complete cohort: {report['solve']['complete_base_queries']}/{report['solve']['planned_base_queries']} base queries complete across all hint levels and configured variants.",
                  'Main level tables use this common complete cohort. H3 can have fewer queries because three-hint queries end at Full; compare total-hint-count strata for fixed-cohort curves.', '']
    def table(summary):
        return ["| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|"] + [
                    f"| {v} | {s['queries']} | " + " | ".join(f"{s[k]:.4f}" for k in
                        ("strict_r1", "video_r1", "video_r5", "group_moment_r1", "group_moment_r5", "group_strict_r1", "group_strict_r5")) + " |"
                    for v, s in summary.items()]
    for level, values in report.get('complete_cohort_levels', levels).items():
        lines += [f"## {level.upper()}", "", f"Matched queries: {len(values['paired']['query_ids'])}.", "", *table(values["paired"]["summary"]), ""]
        for task, values in values["by_task"].items():
            lines += [f"### {task}", "", *table(values["summary"]), ""]
    if 'solve' in report:
        lines += ['## Grouped hints-to-solve', '',
                  'Each level is an independent fresh run. Moment/strict success here is computed from the returned frame pool of the top video. Unsolved queries have null first-solved hint; the penalized mean assigns H+1. Regression means a later independent level failed after an earlier success.', '',
                  '| Criterion | Variant | N | Solved N | Mean h* (solved) | Mean h* (penalized) | Early solve | Full success | Regression |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for variant, criteria in report['solve']['summary'].items():
            for criterion, s in criteria.items():
                def fmt(value):
                    return 'unknown' if value is None else str(value) if isinstance(value, int) else f'{value:.4f}'
                lines.append(f'| {criterion} | {variant} | ' + ' | '.join(fmt(s[k]) for k in
                    ['queries','solved_queries','mean_hints_to_solve_solved_only','mean_hints_to_solve_penalized','early_solve_rate','full_solve_rate','regression_rate']) + ' |')
    lines += ["This is a snapshot of the current run. Re-export after new results arrive to update the report.",
              "Per-query ranks and candidate pool sizes are in query_results.csv. JSON includes available and paired cohorts for each hint level."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tolerance", type=float, help="Defaults to the recorded run tolerance")
    parser.add_argument("--include-legacy-quota", action="store_true", help="Historical descriptive report only: retain old interrupted attempts with explicit health notes")
    parser.add_argument("--plots", action="store_true", help="Export fixed-cohort grouped hint recall PDF/PNG; requires matplotlib")
    args = parser.parse_args()
    report = export(args.run_dir, args.dataset, tolerance=args.tolerance, include_legacy_quota=args.include_legacy_quota)
    if args.plots:
        plot_group_curves(args.run_dir)
    print(json.dumps({"output": str(Path(args.run_dir) / 'video-report'), "scored_rows": report['scored_rows'],
                      "matched_full_queries": len(report['levels'].get('full', {}).get('paired', {}).get('query_ids', []))}, indent=2))


if __name__ == "__main__":
    main()
