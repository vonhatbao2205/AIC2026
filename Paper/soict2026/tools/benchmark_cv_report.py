"""Audit the completed live C/C+V/E experiment and export offline paired results.

Run from the repository root with python -m tools.benchmark_cv_report.
No model, network, or submission calls. Raw traces remain in their original run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from benchmarks.agent.metrics import first_rank, measure, percentile, summarize
from benchmarks.agent.run_benchmark import load_queries, quota_interrupted

VARIANTS = ("C", "C+V", "E")
ERRORS = ("failed", "agent_failed", "retrieval_failed", "retrieval_degraded", "fallback")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched(rows, query_ids):
    by = {q: {} for q in query_ids}
    for r in rows:
        q, v = r["query_id"], r["variant"]
        if q not in by or v not in VARIANTS or v in by[q]:
            raise ValueError("Unexpected or duplicate query/variant")
        if quota_interrupted(r):
            raise ValueError("Quota-interrupted attempt in scored results")
        by[q][v] = r
    if any(set(values) != set(VARIANTS) for values in by.values()):
        raise ValueError("Incomplete matched cohort")
    return by


def paired(a, b, indices):
    delta = [x - y for x, y in zip(a, b)]
    draws = [sum(delta[i] for i in ix) / len(ix) for ix in indices]
    return {"n": len(a), "delta": statistics.mean(delta),
            "ci95": [percentile(draws, .025), percentile(draws, .975)]}


def mcnemar(a, b):
    wins = sum(bool(x) and not y for x, y in zip(a, b))
    losses = sum(bool(y) and not x for x, y in zip(a, b))
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.
    return {"a_only": wins, "b_only": losses, "p_exact": p}


def ranking_metrics(ranking, query, tolerance):
    rank = first_rank(ranking, query, tolerance)
    return {"rank": rank, "r1": int(rank == 1),
            "r5": int(rank is not None and rank <= 5), "mrr": 1 / rank if rank else 0.}


def export(run_dir, dataset, output, *, samples=10000, seed=2026):
    run_dir, dataset, output = map(Path, (run_dir, dataset, output))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = manifest["config"]
    assert config["mode"] == "live" and config["split"] == "test"
    assert config["variants"] == list(VARIANTS) and config["tolerance_s"] == 1
    assert config["dataset_sha256"] == sha(dataset)
    assert manifest["fingerprint"] == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    plan = json.loads((run_dir / "run_plan.json").read_text())
    assert config["run_plan_sha256"] == hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    queries = {q["query_id"]: q for q in load_queries(dataset) if q["split"] == "test"}
    assert set(queries) == set(config["query_ids"])
    ids = sorted(queries)
    initial_dataset = Path("benchmarks/agent/runs/soict-final-dev-test-pooled-tol1-v1/test/main/evaluation-dataset.jsonl")
    initial = {q["query_id"]: q for q in load_queries(initial_dataset)}
    assert initial.keys() == queries.keys()
    for q in ids:
        assert all(initial[q].get(k) == queries[q].get(k) for k in
                   ("query", "query_type", "targets", "sequences", "image_models", "retrieval_database", "scope"))
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    by = matched(rows, ids)
    outcomes, checks = [], {"initial_round_identical_queries_labels_retrieval": len(ids),
                            "recomputed_metrics": 0, "cv_fixed_acquisition_final_verification": 0,
                            "cv_pre_ranking_matches_trace_and_rrf": 0}
    pre_metrics = {}
    for q in ids:
        for v in VARIANTS:
            r = by[q][v]
            snap, trace = r["snapshot"], r["trace"]["trace"]
            assert r["evaluated_query"] == queries[q]["query"] == snap["query"]
            assert r["split"] == "test" and r.get("hint_stage", "full") == "full"
            assert snap["retrieval_database"] == queries[q]["retrieval_database"]
            assert snap["policy"] == config["variant_parameters"][v]["policy"]
            assert snap["finished"] and not snap["controller"]["calibrated"]
            recomputed = measure(queries[q], snap, trace, wall_s=r["metrics"]["latency_s"], tolerance_s=1)
            assert json.loads(json.dumps(recomputed)) == r["metrics"], (q, v, "metric mismatch")
            checks["recomputed_metrics"] += 1
            outcomes.append({"query_id": q, "query_type": queries[q]["query_type"], "variant": v,
                **{k: r["metrics"][k] for k in ("r1", "r5", "mrr", "latency_s", "codex_calls", "claude_calls", *ERRORS)}})
        r = by[q]["C+V"]
        trace, snap = r["trace"]["trace"], r["snapshot"]
        # Validate the intended ablation from observed traces, not only its label.
        kinds = {k: [t for t in trace if t["kind"] == k]
                 for k in ("agent_results", "pre_verification", "verification", "decision", "action")}
        assert all(len(kinds[k]) == 1 for k in ("agent_results", "pre_verification", "verification"))
        assert not kinds["decision"] and not kinds["action"]
        assert set(kinds["agent_results"][0]["agents"]) == {"codex", "claude"}
        assert snap["metrics"]["agent_calls"] == {"codex": 1, "claude": 1}
        assert kinds["agent_results"][0]["at_s"] <= kinds["pre_verification"][0]["at_s"] <= kinds["verification"][0]["at_s"]
        assert snap["controller"]["stop_reason"] == "fixed_verification_complete"
        checks["cv_fixed_acquisition_final_verification"] += 1
        before = snap["pre_verification_ranking"]
        assert before == kinds["pre_verification"][0]["ranking"]
        order = lambda rows: [f["submit_keyframe_id"] for f in rows]
        rrf = sorted(snap["ranking"], key=lambda f: (-f["fusion_score"], f["first_seen_s"], f["submit_keyframe_id"]))
        assert order(before) == order(rrf)
        for a, b in zip(before, rrf):
            assert all(a.get(k) == b.get(k) for k in ("video_id", "frame_idx", "pts_time", "answer", "event"))
        checks["cv_pre_ranking_matches_trace_and_rrf"] += 1
        pre_metrics[q] = ranking_metrics(before, queries[q], 1)
        outcomes.append({"query_id": q, "query_type": queries[q]["query_type"], "variant": "C+V pre",
                         **{k: pre_metrics[q][k] for k in ("r1", "r5", "mrr")}})
    rng = random.Random(seed)
    indices = [[rng.randrange(len(ids)) for _ in ids] for _ in range(samples)]
    contrasts = {}
    for a, b in (("C+V", "C"), ("E", "C+V"), ("E", "C"), ("C+V", "C+V pre")):
        get = lambda q, v, k: pre_metrics[q][k] if v == "C+V pre" else by[q][v]["metrics"][k]
        contrasts[f"{a} - {b}"] = {
            k: paired([get(q, a, k) for q in ids], [get(q, b, k) for q in ids], indices)
            for k in ("r1", "r5", "mrr")}
        contrasts[f"{a} - {b}"]["mcnemar"] = mcnemar([get(q, a, "r1") for q in ids], [get(q, b, "r1") for q in ids])
    quotas_path = run_dir / "quota_attempts.jsonl"
    quotas = [json.loads(x) for x in quotas_path.read_text().splitlines()] if quotas_path.exists() else []
    assert all(quota_interrupted(r) for r in quotas)
    summary = summarize(rows)
    sources = {str(p): sha(p) for p in (dataset, initial_dataset, run_dir / "results.jsonl", run_dir / "manifest.json", run_dir / "run_plan.json")}
    if quotas_path.exists():
        sources[str(quotas_path)] = sha(quotas_path)
    stats = {"protocol": {"cohort": "supplementary TEST rerun; original split; no new DEV accuracy tuning",
                          "samples": samples, "seed": seed, "sampling_unit": "base query",
                          "inference": "exploratory, percentile paired bootstrap and exact McNemar; no multiplicity correction"},
        "sources_sha256": sources, "runtime_code_sha256": config["code_sha256"],
        "query_ids": ids, "checks": checks, "summary": summary,
        "by_task": {t: summarize([r for r in rows if r["query_type"] == t]) for t in sorted({q["query_type"] for q in queries.values()})},
        "contrasts": contrasts,
        "cv_pre_verification": {k: statistics.mean(pre_metrics[q][k] for q in ids) for k in ("r1", "r5", "mrr")},
        "errors": {v: {k: sum(bool(by[q][v]["metrics"][k]) for q in ids) for k in ERRORS} for v in VARIANTS},
        "quota_audit": {"attempts": len(quotas), "by_variant": {v: sum(r["variant"] == v for r in quotas) for v in VARIANTS},
                        "observed_attempt_wall_s": sum(r["metrics"]["latency_s"] for r in quotas),
                        "agent_invocations": sum(r["metrics"]["codex_calls"] + r["metrics"]["claude_calls"] for r in quotas),
                        "pause_time_included": False},
        "query_outcomes": outcomes}
    output.mkdir(parents=True, exist_ok=True)
    (output / "ablation_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "summary_by_task.json").write_text(json.dumps(stats["by_task"], indent=2) + "\n")
    with (output / "query_outcomes.csv").open("w") as f:
        writer = csv.DictWriter(f, list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    lines = ["# Live C/C+V/E ablation — complete TEST cohort", "",
        f"{len(ids)} queries; {len(rows)} completed live runs; tolerance 1 s. Original A–F results remain a separate round.", "",
        "| Variant | R@1 (%) | R@5 (%) | MRR | p50 s | p95 s | Agents/query | Jev calls | kTokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        s = summary[v]
        lines.append(f"| {v} | {100*s['r1']:.1f} | {100*s['r5']:.1f} | {s['mrr']:.3f} | {s['latency_p50_s']:.1f} | {s['latency_p95_s']:.1f} | {s['codex_calls']+s['claude_calls']:.2f} | {s['jev_calls']:.2f} | {s['tokens_mean']/1000:.0f} |")
    lines += ["", "| Paired contrast | ΔR@1, pp [95% CI] | A-only / B-only | Exact McNemar p |",
              "|---|---:|---:|---:|"]
    for name, c in contrasts.items():
        d, m = c["r1"], c["mcnemar"]
        lines.append(f"| {name} | {100*d['delta']:+.1f} [{100*d['ci95'][0]:.1f}, {100*d['ci95'][1]:.1f}] | {m['a_only']} / {m['b_only']} | {m['p_exact']:.4f} |")
    lines += ["", "| Paired contrast | ΔR@5, pp [95% CI] | ΔMRR [95% CI] |",
              "|---|---:|---:|"]
    for name, c in contrasts.items():
        r, m = c["r5"], c["mrr"]
        lines.append(f"| {name} | {100*r['delta']:+.1f} [{100*r['ci95'][0]:.1f}, {100*r['ci95'][1]:.1f}] | {m['delta']:+.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] |")
    lines += ["", "| Task | N | C R@1 (%) | C+V R@1 (%) | E R@1 (%) |",
              "|---|---:|---:|---:|---:|"]
    for task, summaries in stats["by_task"].items():
        lines.append(f"| {task} | {summaries['C']['queries']} | " + " | ".join(f"{100*summaries[v]['r1']:.1f}" for v in VARIANTS) + " |")
    lines += ["", f"All contrasts use {samples:,} paired base-query resamples, seed {seed}. Exploratory, unadjusted for multiplicity.",
        "E exceeds C in this round, but E versus C+V is inconclusive; nonsignificance does not establish equivalence.",
        "C+V versus C contains independently sampled agent trajectories. Only C+V final versus its pre-verification ranking holds acquired evidence fixed.",
        "The controller-package contrast also changes scheduling, intermediate verification and agent context; it does not isolate call allocation.", "",
        f"C+V pre-verification R@1: {100*stats['cv_pre_verification']['r1']:.1f}%. All 86 traces have one final verification pass and no routing decisions.",
        f"Quota audit: {len(quotas)} interrupted attempts, {stats['quota_audit']['observed_attempt_wall_s']:.1f} s recorded wall time and {stats['quota_audit']['agent_invocations']} agent invocations outside completed-run metrics; pause time excluded.",
        "Completed-run errors: " + json.dumps(stats["errors"]), "",
        "Token telemetry is complete; full monetary cost is unavailable because some CLI usage has no cost. Known cost is not total cost.",
        "Per-task results, all R@1/R@5/MRR paired intervals, validation counts and SHA-256 provenance are in ablation_stats.json. Raw traces remain local."]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="benchmarks/agent/runs/soict-cv-20261005-v1/test")
    parser.add_argument("--dataset", default="benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl")
    parser.add_argument("--output", default="benchmarks/agent/results/soict-cv-tol1")
    args = parser.parse_args()
    export(args.run_dir, args.dataset, args.output)
    print(f"Validated C/C+V/E results exported to {args.output}")
