"""Audit raw DEV traces and tune evidence-preserving stopping without model calls.

Recorded-prefix replay is a screening estimate, not a new live experiment:
calibrated probabilities may change subsequent Jev routing. Confirm on fresh DEV
before TEST. Failed agent attempts remain in audit but are excluded from fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from backend.app.agent.decision.calibration import Calibration, fit_temperature
from backend.app.agent.decision.stopping import can_stop
from backend.app.agent.evidence.compiler import Constraint
from backend.app.agent.cli import classify_error
from .metrics import calibration_metrics, first_rank, percentile
from .run_benchmark import load_queries


class RecordedBoard:
    def __init__(self, query_type, constraints, ranking):
        self.query_type, self.constraints, self.ranking = query_type, constraints, ranking

    def ranked(self):
        return self.ranking


def failed(row):
    m = row["metrics"]
    return bool(m.get("failed") or m.get("agent_failed") or m.get("retrieval_failed") or m.get("fallback"))


def checkpoints(row, query, tolerance=1.0):
    trace, invoked, result = row["trace"]["trace"], set(), []
    constraints = [Constraint(**c) for c in row["trace"]["constraints"]]
    for index, event in enumerate(trace):
        if event["kind"] == "agent_results":
            invoked.update(event["agents"])
        if event["kind"] != "verification":
            continue
        # A publish immediately after verification is the board can_stop sees.
        following = trace[index + 1:]
        ranking = next((e["ranking"] for e in following if e["kind"] == "ranking"), [])
        result.append({"at_s": event["at_s"], "agents": sorted(invoked), "ranking": ranking,
                       "constraints": constraints, "strict_rank": first_rank(ranking, query, tolerance),
                       "next_action": next((e["action"] for e in following if e["kind"] == "action"), None)})
    return result


def calibrated_ranking(ranking, calibration):
    result = []
    for frame in ranking:
        frame = dict(frame)
        if frame.get("probability") is not None:
            raw = frame.get("raw_probability")
            frame["probability"] = calibration.apply(frame["probability"] if raw is None else raw)
        result.append(frame)
    return result


def replay(row, query, points, calibration, threshold, margin, tolerance):
    for point in points:
        ranking = calibrated_ranking(point["ranking"], calibration)
        board = RecordedBoard(query["query_type"], point["constraints"], ranking)
        if can_stop(board, threshold, margin, require_constraints=row["variant"] == "F"):
            rank = first_rank(ranking, query, tolerance)
            return {"query_id": row["query_id"], "r1": int(rank == 1), "r5": int(rank is not None and rank <= 5),
                    "agents": len(point["agents"]), "prefix_latency_s": point["at_s"], "early_stop": True}
    return {"query_id": row["query_id"], "r1": row["metrics"]["r1"], "r5": row["metrics"]["r5"],
            "agents": row["metrics"]["codex_calls"] + row["metrics"]["claude_calls"],
            "prefix_latency_s": row["metrics"]["latency_s"], "early_stop": False}


def aggregate(outcomes):
    return {"queries": len(outcomes), "r1": statistics.mean(o["r1"] for o in outcomes),
            "r5": statistics.mean(o["r5"] for o in outcomes),
            "agents_per_query": statistics.mean(o["agents"] for o in outcomes),
            "both_rate": statistics.mean(o["agents"] == 2 for o in outcomes),
            "recorded_prefix_latency_p50_s": percentile([o["prefix_latency_s"] for o in outcomes], .5)}


def video_components(queries):
    groups = {qid: {qid} for qid in queries}
    owners = {}
    for qid, query in queries.items():
        for target in query["targets"]:
            video = target["video_id"]
            if video in owners:
                united = groups[qid] | groups[owners[video]]
                for member in united:
                    groups[member] = united
            owners[video] = qid
    return groups


def tune(dataset, run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = manifest["config"]
    if config["split"] != "dev" or config.get("hint_mode", "full") != "full" or config["mode"] != "live":
        raise ValueError("Tuning requires raw, full-query, live DEV runs")
    if hashlib.sha256(Path(dataset).read_bytes()).hexdigest() != config["dataset_sha256"]:
        raise ValueError("Dataset differs from the DEV manifest")
    queries = {q["query_id"]: q for q in load_queries(dataset) if q["split"] == "dev"}
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    if any(r.get("split") != "dev" or r["query_id"] not in queries for r in rows):
        raise ValueError("TEST or unknown queries cannot be used for tuning")
    if any(r.get("snapshot", {}).get("controller", {}).get("calibrated") for r in rows):
        raise ValueError("Use raw uncalibrated DEV observations")
    if any(r.get("hint_stage", "full") != "full" for r in rows):
        raise ValueError("Main full-query DEV only; do not pool cumulative stages")
    output.mkdir(parents=True, exist_ok=True)
    components = video_components(queries)
    profiles, analysis = {}, {}
    tolerance = config["tolerance_s"]
    thresholds = [.5, .52, .55, .6, .65, .7, .75, .8, .85, .9]
    margins = [0, .05, .1, .15, .2]
    for variant, aggregation in (("E", "holistic"), ("F", "constraints")):
        group = [r for r in rows if r["variant"] == variant]
        if len(group) != len(config["query_ids"]) or len({r["query_id"] for r in group}) != len(group):
            raise ValueError(f"Need complete distinct DEV rows for {variant}")
        healthy = [r for r in group if not failed(r)]
        if len(healthy) < 2:
            raise ValueError("Too few healthy development runs")
        samples = [{"query_id": r["query_id"], "split": "dev", "probability": p, "label": y}
                   for r in healthy for p, y in r["metrics"]["calibration"]]
        fit = fit_temperature(samples)
        models = {c["model"] for r in healthy for c in r["snapshot"]["controller"]["decision_calls"]
                  if c.get("backend") == r["snapshot"]["controller"].get("verifier", "jev")}
        if len(models) != 1:
            raise ValueError("Expected exactly one verifier model")
        fit.update(model=models.pop(), aggregation=aggregation, variant=variant)
        cal = Calibration(fit["temperature"])
        by_query = {r["query_id"]: checkpoints(r, queries[r["query_id"]], tolerance) for r in group}
        crossfit, crossfit_pairs = {}, []
        for r in healthy:
            qid = r["query_id"]
            training = [s for s in samples if s["query_id"] not in components[qid]]
            crossfit[qid] = Calibration(fit_temperature(training)["temperature"])
            crossfit_pairs.extend((crossfit[qid].apply(p), y) for p, y in r["metrics"]["calibration"])
        baseline = aggregate([{"r1": r["metrics"]["r1"], "r5": r["metrics"]["r5"], "agents": r["metrics"]["codex_calls"] + r["metrics"]["claude_calls"],
                               "prefix_latency_s": r["metrics"]["latency_s"]} for r in healthy])
        sweeps = []
        default_threshold = config.get("status", {}).get("controller_config", {}).get("stop_threshold", .9)
        default_margin = config.get("status", {}).get("controller_config", {}).get("stop_margin", .1)
        def simulate(r, calibration, threshold, margin):
            # Two QA and two TRAKE queries do not support lowering their task
            # thresholds. Keep their original settings; screen T-KIS only.
            if queries[r["query_id"]]["query_type"] != "T-KIS":
                threshold, margin = default_threshold, default_margin
            return replay(r, queries[r["query_id"]], by_query[r["query_id"]], calibration, threshold, margin, tolerance)
        for threshold in thresholds:
            for margin in margins:
                outcome = [simulate(r, crossfit[r["query_id"]], threshold, margin) for r in healthy]
                wrong_early = sum(o["agents"] < r["metrics"]["codex_calls"] + r["metrics"]["claude_calls"] and not o["r1"] for r, o in zip(healthy, outcome))
                sweeps.append({"threshold": threshold, "margin": margin,
                               "incorrect_compute_saving_stops": wrong_early, **aggregate(outcome)})
        eligible = [s for s in sweeps if s["r1"] >= baseline["r1"] and s["r5"] >= baseline["r5"] and not s["incorrect_compute_saving_stops"]]
        # Retain both recall metrics; minimize calls then time. For identical
        # outcomes keep default margin and the stricter threshold.
        best = min(eligible, key=lambda s: (s["agents_per_query"], s["recorded_prefix_latency_p50_s"], abs(s["margin"] - .1), -s["threshold"]))
        if best["agents_per_query"] >= baseline["agents_per_query"]:
            best = next(s for s in sweeps if s["threshold"] == default_threshold and s["margin"] == default_margin)
        fit.update(stop_threshold=default_threshold, stop_margin=default_margin,
                   stopping_by_task={"T-KIS": {"threshold": best["threshold"], "margin": best["margin"]}})
        profiles[aggregation] = fit
        diagnosis = []
        for r in group:
            points = by_query[r["query_id"]]
            point = next((p for p in points if len(p["agents"]) == 1), None)
            detail = {"query_id": r["query_id"], "healthy": not failed(r), "final_r1": r["metrics"]["r1"],
                      "stop_overrides": [t["action"] for t in r["trace"]["trace"] if t["kind"] == "stop_overridden"],
                      "direct_call_both": any(t.get("action") == "CALL_BOTH" for t in r["trace"]["trace"])}
            if point:
                board = RecordedBoard(queries[r["query_id"]]["query_type"], point["constraints"], point["ranking"])
                guard = can_stop(board, 0, 0, require_constraints=variant == "F")
                top = point["ranking"][0] if point["ranking"] else {}
                if queries[r["query_id"]]["query_type"] == "TRAKE":
                    competitors = [f for f in point["ranking"][1:] if f["video_id"] != top["video_id"]]
                else:
                    competitors = [f for f in point["ranking"][1:] if f["video_id"] != top["video_id"] or
                                   f.get("pts_time") is None or top.get("pts_time") is None or abs(f["pts_time"] - top["pts_time"]) > 8]
                gap = None if top.get("probability") is None else top["probability"] - max(
                    (f["probability"] for f in competitors if f.get("probability") is not None), default=0)
                detail.update(first_agent=point["agents"], rank_after_agent1=point["strict_rank"],
                              top_probability_after_agent1=top.get("probability"),
                              margin_after_agent1=gap,
                              evidence_guard_passed=guard, next_action=point["next_action"],
                              constraint_scores=top.get("constraint_scores", {}))
            diagnosis.append(detail)
        selected = [simulate(r, cal, best["threshold"], best["margin"]) for r in group]
        raw_pairs = [(s["probability"], s["label"]) for s in samples]
        analysis[variant] = {"healthy_queries": len(healthy), "excluded_failed_query_ids": [r["query_id"] for r in group if failed(r)],
                             "baseline_healthy": baseline, "selected_crossfit_replay_healthy": best,
                             "selected_fitted_replay_all": aggregate(selected), "selected_outcomes": selected,
                             "raw_calibration": calibration_metrics(raw_pairs),
                             "fitted_calibration": calibration_metrics([(cal.apply(p), y) for p, y in raw_pairs]),
                             "leave_video_component_out_calibration": calibration_metrics(crossfit_pairs),
                             "temperature": fit["temperature"], "sweep": sweeps, "diagnosis": diagnosis}
    provenance = {"source_results_sha256": hashlib.sha256((run_dir / "results.jsonl").read_bytes()).hexdigest(),
                  "source_manifest_fingerprint": manifest["fingerprint"], "dataset_sha256": config["dataset_sha256"],
                  "source_code_sha256": config["code_sha256"], "tolerance_s": tolerance,
                  "selection": "Main DEV only; healthy runs for fit; leave-target-video-component-out temperatures for sweep; T-KIS threshold only; retain QA/TRAKE defaults; no R@1/R@5 loss or incorrect compute-saving stops on observed healthy prefixes",
                  "limitations": "Recorded-prefix replay; not a live policy run or unbiased test estimate. Threshold selection reuses DEV. Temperature fit uses final candidate labels, not constraint labels or stage-specific stop labels."}
    bundle = {"version": 2, "split": "dev", "profiles": profiles, "provenance": provenance}
    report = {"provenance": provenance, "variants": analysis}
    failures = []
    for experiment in (run_dir, run_dir.parent / "cumulative-dev"):
        source = experiment / "results.jsonl"
        if not source.exists():
            continue
        for line in source.read_text().splitlines():
            row = json.loads(line)
            if row.get("split") != "dev":
                raise ValueError("Failure audit must not read TEST rows")
            for agent, state in row.get("snapshot", {}).get("agents", {}).items():
                if state.get("status") in {"failed", "timeout", "cancelled"}:
                    failures.append({"experiment": experiment.name, "query_id": row["query_id"],
                                     "hint_stage": row.get("hint_stage", "full"), "variant": row["variant"],
                                     "agent": agent, "status": state["status"], "recorded_kind": state.get("error_kind"),
                                     "diagnosed_kind": classify_error(state.get("error", "")), "error": state.get("error")})
    report["failure_audit"] = failures
    for name, value in (("calibration.json", bundle), ("tuning.json", report)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# DEV stopping audit", "", provenance["selection"], "", provenance["limitations"], "",
             "Evidence guards, QA answer and TRAKE sequence checks remain unchanged. CLI/provider quota failures are excluded from fitting, retained in audit.", "",
             "| Variant | Healthy N | Temperature | T-KIS threshold | T-KIS margin | Raw R@1 | Crossfit prefix R@1 | Raw agents | Prefix agents |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v, a in analysis.items():
        s, b = a["selected_crossfit_replay_healthy"], a["baseline_healthy"]
        lines.append(f"| {v} | {a['healthy_queries']} | {a['temperature']:.4f} | {s['threshold']} | {s['margin']} | {b['r1']:.4f} | {s['r1']:.4f} | {b['agents_per_query']:.4f} | {s['agents_per_query']:.4f} |")
    for v, a in analysis.items():
        first_correct = [d for d in a["diagnosis"] if d.get("rank_after_agent1") == 1]
        blocked = sum(not d["evidence_guard_passed"] for d in first_correct)
        lines.append(f"\n{v}: {len(first_correct)} queries already correct after agent1; {blocked} still blocked by the raw evidence guard.\n")
    lines.extend(["", f"Failure audit: {len(failures)} agent failures across original main/cumulative DEV. "
                  f"{sum(f['recorded_kind'] is None and f['diagnosed_kind'] == 'quota' for f in failures)} session-limit failures previously lacked a quota label. "
                  "See tuning.json failure_audit. Cumulative outcomes are audited, not used to select thresholds."])
    lines.extend(["", "Confirm E/F on a fresh live DEV output before the official TEST run. Do not combine confirmation rows with the original manifest."])
    (output / "TUNING_REPORT.md").write_text("\n".join(lines) + "\n")
    return bundle, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tune(args.dataset, args.run_dir, args.output)


if __name__ == "__main__":
    main()
