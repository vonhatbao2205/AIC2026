"""Offline descriptive DEV+TEST report, preserving original split provenance.

Original dataset splits are preserved; an optional evaluation copy labels
the combined cohort TEST. This never resumes/runs an agent benchmark.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.agent.hints import build_plan, run_key
from benchmarks.agent.metrics import measure, summarize
from benchmarks.agent.paper_report import export_paper_report
from benchmarks.agent.run_benchmark import load_queries, quota_interrupted
from tools.benchmark_video_report import score_row, summarize as video_summary, export as export_video_report, plot_group_curves


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def merge(dev_run, test_run, dataset, output, *, treat_dev_as_test=False):
    sources = {"dev": Path(dev_run), "test": Path(test_run)}
    output, dataset = Path(output), Path(dataset)
    if output.exists():
        raise ValueError("Output already exists; choose a new directory to preserve previous results")
    manifests = {split: json.loads((path / "manifest.json").read_text()) for split, path in sources.items()}
    compared = ("dataset_sha256", "variants", "variant_parameters", "hint_mode", "tolerance_s", "status", "seed", "mode")
    for field in compared:
        if manifests["dev"]["config"].get(field) != manifests["test"]["config"].get(field):
            raise ValueError(f"Source configurations differ: {field}")
    for split, manifest in manifests.items():
        if manifest["config"]["split"] != split or manifest["config"]["hint_mode"] not in {'full', 'cumulative'}:
            raise ValueError("Expected matching full/cumulative DEV and TEST runs")
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if dataset_hash != manifests["test"]["config"]["dataset_sha256"]:
        raise ValueError("Dataset SHA-256 differs from source runs")
    queries = load_queries(dataset)
    by_id = {q["query_id"]: q for q in queries}
    variants = manifests["test"]["config"]["variants"]
    tolerance = manifests["test"]["config"]["tolerance_s"]
    mode = manifests['test']['config']['hint_mode']
    original_stages, _ = build_plan(queries, mode, variants)
    stage_by_key = {(q['query_id'], q['hint_stage']): q for q in original_stages}
    expected = {(q["query_id"], q['hint_stage'], v) for q in original_stages for v in variants}
    source_rows = {}
    for split, path in sources.items():
        keys = set()
        digest = hashlib.sha256()
        size = path.joinpath("results.jsonl").stat().st_size
        consumed = 0
        with (path / "results.jsonl").open("rb") as file:
            while consumed < size:
                line = file.readline(size - consumed)
                consumed += len(line)
                digest.update(line)
                if not line.endswith(b"\n"):
                    raise ValueError("Source has an incomplete result row")
                if not line.strip():
                    continue
                row = json.loads(line)
                key = run_key(row)
                if key in keys or key not in expected or row["split"] != split or by_id[row["query_id"]]["split"] != split:
                    raise ValueError(f"Duplicate/unexpected source row: {key}")
                keys.add(key)
        source_expected = {key for key in expected if by_id[key[0]]["split"] == split}
        if keys != source_expected:
            raise ValueError(f"{split} source does not cover every planned query/variant")
        source_rows[split] = {"source_directory": str(path.resolve()), "source_fingerprint": manifests[split]["fingerprint"],
                              "git_head": manifests[split].get("git_head"),
                              "code_sha256": manifests[split]["config"].get("code_sha256"),
                              "results_sha256": digest.hexdigest(), "source_snapshot_bytes": size, "rows": len(keys)}
    evaluation_queries = [{**q, "original_split": q["split"], "split": "test"} for q in queries] if treat_dev_as_test else queries
    _, plan = build_plan(evaluation_queries, mode, variants)
    output.mkdir(parents=True)
    merged_dataset = output / "evaluation-dataset.jsonl"
    merged_dataset.write_text("".join(json.dumps(q, ensure_ascii=False, allow_nan=False) + "\n" for q in evaluation_queries))
    config = copy.deepcopy(manifests["test"]["config"])
    config.update(split="test" if treat_dev_as_test else "dev+test", query_ids=sorted({q['query_id'] for q in original_stages}),
                  dataset_sha256=hashlib.sha256(merged_dataset.read_bytes()).hexdigest(),
                  run_plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest())
    config.pop("code_sha256", None)  # Both source code revisions are recorded separately below.
    timestamp = datetime.now(timezone.utc).isoformat()
    note = ("Descriptive pooled DEV+TEST results, not an independent held-out TEST evaluation. "
            "DEV was used during calibration/policy selection. Original split labels and source results are preserved. "
            "Legacy DEV agent/quota failures remain in the denominator. This directory is report-only; do not use it to resume benchmark runs.")
    if treat_dev_as_test:
        ids = {q['query_id'] for q in original_stages}
        counts = Counter(q['split'] for q in queries if q['query_id'] in ids)
        note = (f"Combined TEST evaluation set: {counts['dev']} queries originally labelled DEV plus {counts['test']} originally labelled TEST. "
                f"Both saved runs use the same raw policy, STOP settings and {tolerance:g}s tolerance, with calibration disabled. "
                "Original source splits are retained as original_split metadata. Original datasets/results are preserved. "
                "Legacy agent/quota failures remain in the denominator. This directory contains offline reports, not new model runs.")
    manifest = {"fingerprint": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "created_at": timestamp, "config": config, "report_only": True, "evaluation_note": note,
                "sources": source_rows, "equal_configuration_fields": list(compared),
                "source_dataset_sha256": dataset_hash, "dataset_path": str(merged_dataset.resolve())}
    dump(output / "manifest.json", manifest)
    dump(output / "run_plan.json", plan)
    light_rows, grouped_rows, failures, seen = [], [], [], set()
    split_rows = defaultdict(list)
    with (output / "results.jsonl").open("w") as dest:
        for split, source in sources.items():
            consumed = 0
            with (source / "results.jsonl").open("rb") as file:
                while consumed < source_rows[split]["source_snapshot_bytes"]:
                    line = file.readline(source_rows[split]["source_snapshot_bytes"] - consumed)
                    consumed += len(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = run_key(row)
                    if key in seen:
                        raise ValueError("Overlapping query/level/variant across splits")
                    seen.add(key)
                    q = stage_by_key[key[:2]]
                    if row.get('evaluated_query', q['query']) != q['query']:
                        raise ValueError(f'Saved stage query differs from audited hints: {key}')
                    computed = measure(q, row.get("snapshot", {}), row.get("trace", {}).get("trace", []),
                                       wall_s=row["metrics"]["latency_s"], tolerance_s=tolerance)
                    for field in ("rank", "video_rank", "r1", "r5", "mrr", "moment_r1", "moment_r5", "calibration"):
                        if json.dumps(computed[field]) != json.dumps(row["metrics"][field]):
                            raise ValueError(f"Source metric cannot be reproduced: {key}, {field}")
                    row["source_directory"] = str(source.resolve())
                    row["source_fingerprint"] = manifests[split]["fingerprint"]
                    if treat_dev_as_test:
                        row["original_split"] = row["split"]
                        row["split"] = "test"
                    row["legacy_quota_interrupted"] = quota_interrupted(row)
                    dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    light = {k: v for k, v in row.items() if k not in {"snapshot", "trace"}}
                    light_rows.append(light)
                    split_rows[split].append(light)
                    grouped_rows.append({**score_row(row, q, tolerance), "split": split})
                    if row["metrics"]["agent_failed"] or row["metrics"]["failed"] or row["metrics"]['fallback'] or row["legacy_quota_interrupted"]:
                        failures.append({"query_id": row["query_id"], "variant": row["variant"], "split": split,
                                         "hint_stage": row.get('hint_stage', 'full'), 'fallback': row['metrics']['fallback'],
                                         "agent_failed": row["metrics"]["agent_failed"], "legacy_quota_interrupted": row["legacy_quota_interrupted"]})
    assert seen == expected
    split_summary = {split: summarize([r for r in rows if r.get('hint_stage', 'full') == 'full']) for split, rows in split_rows.items()}
    split_summary["dev+test"] = summarize([r for r in light_rows if r.get('hint_stage', 'full') == 'full'])
    dump(output / "summary_by_split.json", split_summary)
    dump(output / "MERGE_PROVENANCE.json", {"created_at": timestamp, "sources": source_rows, "evaluation_note": note,
                                           "split_counts": dict(Counter(q['split'] for q in queries)),
                                           "rows": len(light_rows), "model_calls": 0, "failures": failures,
                                           "source_metrics_verified": True, "all_expected_keys_present": True})
    if mode == 'cumulative':
        from benchmarks.agent.hint_report import write_hint_report
        for split, records in split_rows.items():
            directory = output/'source-reports'/split
            directory.mkdir(parents=True)
            dump(directory/'manifest.json', manifests[split])
            (directory/'run_plan.json').write_bytes((sources[split]/'run_plan.json').read_bytes())
            write_hint_report(directory, records, variants)
    grouped = {"tolerance_s": tolerance, "evaluation_note": note, "legacy_failures_retained": failures,
               "group_order": "First occurrence of a video in the saved frame ranking", "summary": video_summary(grouped_rows),
               "by_split": {split: video_summary([r for r in grouped_rows if r["split"] == split]) for split in sources}}
    video_output = output / "video-report"
    video_output.mkdir()
    dump(video_output / "summary.json", grouped)
    with (video_output / "query_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(grouped_rows[0]))
        writer.writeheader()
        writer.writerows(grouped_rows)
    lines = ["# Combined TEST video-group evaluation" if treat_dev_as_test else "# Pooled DEV+TEST video-group evaluation", "", note, "",
             f"Tolerance: {tolerance:g}s. Full saved frame pools are grouped by video; a video occupies one rank slot.",
             "Video recall checks identity; group moment requires a correct returned moment; group strict additionally requires an accepted QA answer. TRAKE retains complete ordered sequence validation.",
             "Group coverage is not accuracy of a single submitted frame or answer.", "",
             "| Variant | N | Strict R@1 | Video R@1 | Group moment R@1 | Group strict R@1 |", "|---|---:|---:|---:|---:|---:|"]
    for v, s in grouped["summary"].items():
        lines.append(f"| {v} | {s['queries']} | {s['strict_r1']:.4f} | {s['video_r1']:.4f} | {s['group_moment_r1']:.4f} | {s['group_strict_r1']:.4f} |")
    (video_output / "REPORT.md").write_text("\n".join(lines) + "\n")
    del grouped_rows, light_rows, split_rows
    export_paper_report(output, plots=mode == 'cumulative')
    if mode == 'cumulative':
        export_video_report(output, merged_dataset, include_legacy_quota=True)
        plot_group_curves(output)
    for name in ("REPORT.md", "PAPER_REPORT.md", *(['CUMULATIVE_HINT_REPORT.md'] if mode == 'cumulative' else [])):
        path = output / name
        text = path.read_text().replace("# CAD-VR benchmark — Full query", "# CAD-VR pooled DEV+TEST — Full query").replace("# CAD-VR paper tables", "# CAD-VR pooled DEV+TEST paper tables")
        if treat_dev_as_test:
            text = text.replace("pooled DEV+TEST", f"combined TEST ({plan['base_queries']} queries)")
        first, rest = text.split("\n", 1)
        path.write_text(first + "\n\n" + note + "\n" + rest)
    return {"output": str(output), "queries": plan['base_queries'], "rows": len(expected), "model_calls": 0,
            "failures": failures, "summary": split_summary["dev+test"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-run", required=True)
    parser.add_argument("--test-run", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--treat-dev-as-test", action="store_true", help="Label the combined evaluation test, preserving original_split metadata")
    args = parser.parse_args()
    result = merge(args.dev_run, args.test_run, args.dataset, args.output, treat_dev_as_test=args.treat_dev_as_test)
    print(json.dumps({k:v for k,v in result.items() if k != 'summary'}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
