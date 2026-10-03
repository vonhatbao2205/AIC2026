"""Run: python -m benchmarks.agent.run_benchmark --help.

Calls only retrieval/agent endpoints. Never submits to DRES or reads PHM state.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .metrics import measure, summarize
from backend.app.trake_events import split_marked_events

VARIANTS = {
    "A": {"policy": "retrieval"},
    "B": {"policy": "codex"},
    "C": {"policy": "parallel"},
    "D": {"policy": "rerank"},
    "E": {"policy": "adaptive"},
    "F": {"policy": "full"},
    "rule": {"policy": "rule"},
    "base_tools": {"policy": "full", "toolset": "base"},
    "compare_only": {"policy": "full", "toolset": "compare"},
    "no_roles": {"policy": "full", "specialization": False},
    "rule_router": {"policy": "full", "router": "rule"},
    "llm_router": {"policy": "full", "router": "llm"},
    "no_verifier": {"policy": "full", "verifier": "none"},
    "claude_verifier": {"policy": "full", "verifier": "llm"},
}


def load_queries(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen = set()
    split_by_text = {}
    for q in rows:
        if not isinstance(q.get("query_id"), str) or not q["query_id"].strip() or q["query_id"] in seen or not isinstance(q.get("query"), str) or not q["query"].strip():
            raise ValueError("Every query must have nonempty text and a unique query_id")
        seen.add(q["query_id"])
        if q.get("split") not in {"dev", "test"} or not q.get("targets"):
            raise ValueError(f"{q['query_id']}: split=dev/test and targets are required")
        text_key = " ".join(unicodedata.normalize("NFC", q["query"]).casefold().split())
        if split_by_text.setdefault(text_key, q["split"]) != q["split"]:
            raise ValueError("Identical query text cannot occur in both dev and test")
        kind = q.get("query_type", "T-KIS")
        events = split_marked_events(q["query"])
        if kind != "TRAKE" and events:
            raise ValueError("Marked temporal queries require TRAKE event-level labels")
        if kind not in {"T-KIS", "QA", "TRAKE", "AVS"}:
            raise ValueError(f"{q['query_id']}: unsupported query_type")
        for target in q["targets"]:
            validate_target(target)
            if kind == "QA" and (not isinstance(target.get("answers"), list) or not target["answers"] or
                                  any(not isinstance(a, str) or not a.strip() for a in target["answers"])):
                raise ValueError("QA targets need exact accepted answers")
        if kind == "TRAKE":
            if not q.get("sequences"):
                raise ValueError("TRAKE requires ground-truth sequence alternatives")
            lengths = set()
            for sequence in q["sequences"]:
                if not isinstance(sequence, list) or len(sequence) < 2:
                    raise ValueError("TRAKE alternatives need at least two events")
                for target in sequence:
                    validate_target(target)
                lengths.add(len(sequence))
                if len({t["video_id"] for t in sequence}) != 1:
                    raise ValueError("TRAKE sequence must stay in one video")
                coordinate = next((key for key in ("frame_idx", "start_s") if all(t.get(key) is not None for t in sequence)), None)
                if coordinate and any(a[coordinate] >= b[coordinate] for a, b in zip(sequence, sequence[1:])):
                    raise ValueError("TRAKE events must have increasing times")
            if len(lengths) != 1:
                raise ValueError("TRAKE alternatives must contain the same number of events")
            if events and len(events) != next(iter(lengths)):
                raise ValueError("TRAKE labels must cover every marked event in order")
    return rows


def validate_target(target):
    if not isinstance(target, dict) or not isinstance(target.get("video_id"), str) or not target["video_id"].strip():
        raise ValueError("Ground truth requires a video_id")
    if not any(target.get(k) is not None for k in ("frame_idx", "start_s", "submit_keyframe_id")):
        raise ValueError("Incomplete moment-level ground truth")
    for field in ("frame_idx", "start_s", "end_s", "fps"):
        value = target.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or
                                  not math.isfinite(value) or value < 0 or (field == "fps" and value == 0)):
            raise ValueError(f"Invalid ground-truth {field}")
    if target.get("frame_idx") is not None and int(target["frame_idx"]) != target["frame_idx"]:
        raise ValueError("frame_idx must be an integer")
    if target.get("start_s") is not None and (target.get("end_s") is None or target["end_s"] < target["start_s"]):
        raise ValueError("Time intervals require end_s >= start_s")
    if target.get("submit_keyframe_id") is not None and (not isinstance(target["submit_keyframe_id"], str) or not target["submit_keyframe_id"].strip()):
        raise ValueError("submit_keyframe_id must be nonempty")


async def run_one(client, query, variant, timeout, poll):
    # The allow-list is intentional: ground truth never leaves the evaluator.
    payload = {"query": query["query"], "query_type": query.get("query_type", "T-KIS"),
               "retrieval_database": query.get("retrieval_database", "infoshotpp"),
               "image_models": query.get("image_models", ["pe"]), "scope": query.get("scope", {"mode": "all"}),
               "previous_hints": [], **VARIANTS[variant]}
    start = time.monotonic()
    response = await client.post("/api/agent/runs", json=payload)
    response.raise_for_status()
    snapshot = response.json()
    run_id = snapshot["run_id"]
    try:
        while not snapshot["finished"]:
            if time.monotonic() - start > timeout:
                raise TimeoutError("Benchmark run exceeded client timeout")
            await asyncio.sleep(poll)
            response = await client.get(f"/api/agent/runs/{run_id}")
            response.raise_for_status()
            snapshot = response.json()
        response = await client.get(f"/api/agent/runs/{run_id}/trace")
        response.raise_for_status()
        trace = response.json()
        return snapshot, trace, time.monotonic() - start
    finally:
        # Client interruption must not leave paid agents running.
        if not snapshot.get("finished"):
            try:
                await asyncio.shield(client.delete(f"/api/agent/runs/{run_id}"))
            except httpx.HTTPError:
                pass


def write_report(output, rows):
    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    by_task = {task: summarize([r for r in rows if r.get("query_type", "T-KIS") == task])
               for task in sorted({r.get("query_type", "T-KIS") for r in rows})}
    (output / "summary_by_task.json").write_text(json.dumps(by_task, ensure_ascii=False, indent=2))
    variants = {r["variant"] for r in rows}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        variants = set(json.loads(manifest_path.read_text())["config"]["variants"])
    present = {}
    for row in rows:
        present.setdefault(row["query_id"], set()).add(row["variant"])
    common = {q for q, completed in present.items() if completed >= variants}
    paired = summarize([r for r in rows if r["query_id"] in common])
    (output / "summary_paired.json").write_text(json.dumps({"query_ids": sorted(common), "summary": paired}, ensure_ascii=False, indent=2))
    lines = ["# CAD-VR benchmark", "", "Moment-level Recall; TRAKE requires a complete ordered sequence; QA requires the accepted answer.",
             "Unknown CLI cost/token usage stays null. ECE/Brier use final candidate relevance, not individual constraint labels.",
             f"Queries complete across all configured variants: {len(common)}. See summary_paired.json for matched-query comparisons. Partial rows below may have different N.", "",
             "| Variant | N | R@1 | R@5 | MRR | Rescue@1 | p50 s | p95 s | Agents/query | $/query |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return "unknown" if v is None else f"{v:.4f}"
    for name, s in summary.items():
        lines.append(f"| {name} | {s['queries']} | {fmt(s['r1'])} | {fmt(s['r5'])} | {fmt(s['mrr'])} | {fmt(s['agent_rescue_at_1'])} | {fmt(s['latency_p50_s'])} | {fmt(s['latency_p95_s'])} | {fmt(s['codex_calls'] + s['claude_calls'])} | {fmt(s['cost_usd_mean'])} |")
    lines.extend(["", "Failure and agent invocation rates (0–1); failed CLI runs remain in the denominator.", "",
                  "| Variant | Controller failed | Agent failed | Retrieval failed | Fallback | No CLI | Codex only | Claude only | Both |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name, s in summary.items():
        values = [s[k] for k in ("failed", "agent_failed", "retrieval_failed", "fallback", "no_cli_rate", "codex_only_rate", "claude_only_rate", "both_agents_rate")]
        lines.append(f"| {name} | " + " | ".join(fmt(v) for v in values) + " |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


async def main_async(args):
    queries = [q for q in load_queries(args.dataset) if q["split"] == args.split]
    if args.limit:
        queries = queries[:args.limit]
    if not queries:
        raise ValueError("No queries in selected split")
    variants = args.variants.split(",")
    if len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError(f"Choose distinct variants from {list(VARIANTS)}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    rows_path = output / "results.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()] if rows_path.exists() else []
    async with httpx.AsyncClient(base_url=args.url.rstrip("/"), timeout=httpx.Timeout(60, read=args.timeout)) as client:
        response = await client.get("/api/agent/status")
        response.raise_for_status()
        status = response.json()
        health_response = await client.get("/api/health")
        health_response.raise_for_status()
        health = health_response.json()
        # Freeze only non-secret runtime metadata.
        config = {"dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                  "split": args.split, "query_ids": [q["query_id"] for q in queries], "variants": variants,
                  "variant_parameters": {v: VARIANTS[v] for v in variants}, "tolerance_s": args.tolerance,
                  "seed": args.seed, "status": {k: status.get(k) for k in ("policy", "agents", "jev", "timeout_seconds", "controller_config")},
                  "mode": health.get("mode", "unknown")}
        # Code fingerprint catches dirty working tree changes, not just HEAD.
        root = Path(__file__).resolve().parents[2]
        code = sorted((root / "backend/app").rglob("*.py")) + sorted((root / "benchmarks/agent").glob("*.py"))
        config["code_sha256"] = hashlib.sha256(b"".join(str(p.relative_to(root)).encode() + p.read_bytes() for p in code)).hexdigest()
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
                raise ValueError("Run configuration changed; use a new output directory")
        else:
            git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "config": config, "git_head": git,
                                                "created_at": datetime.now(timezone.utc).isoformat()}, indent=2))
        done = {(r["query_id"], r["variant"]) for r in rows}
        rng = random.Random(args.seed)
        for query in queries:
            order = list(variants)
            rng.shuffle(order)
            for variant in order:
                if (query["query_id"], variant) in done:
                    continue
                print(f"{query['query_id']} {variant}", flush=True)
                started = time.monotonic()
                try:
                    snapshot, trace, wall = await run_one(client, query, variant, args.timeout, args.poll)
                    fitted_ids = snapshot.get("controller", {}).get("calibration_query_ids", [])
                    if args.split == "test" and query["query_id"] in fitted_ids:
                        raise ValueError("Calibration leakage: test query was used for fitting")
                    metric = measure(query, snapshot, trace["trace"], wall_s=wall, tolerance_s=args.tolerance)
                    row = {"query_id": query["query_id"], "split": args.split, "query_type": query.get("query_type", "T-KIS"),
                           "variant": variant, "snapshot": snapshot, "trace": trace, "metrics": metric}
                except (httpx.HTTPError, TimeoutError) as exc:
                    # Keep failed queries in the denominator; never quietly drop them.
                    snapshot = {"controller": {"status": "failed"}}
                    row = {"query_id": query["query_id"], "split": args.split, "query_type": query.get("query_type", "T-KIS"),
                           "variant": variant, "error": type(exc).__name__,
                           "metrics": measure(query, snapshot, [], wall_s=time.monotonic() - started, tolerance_s=args.tolerance)}
                    row["metrics"].update(cost_usd=None, tokens=None)
                with rows_path.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    f.flush()
                rows.append(row)
                write_report(output, rows)
                if any(a.get("error_kind") == "quota" for a in row.get("snapshot", {}).get("agents", {}).values()):
                    (output / "STOPPED.json").write_text(json.dumps({"reason": "agent_quota", "query_id": query["query_id"], "variant": variant}, indent=2))
                    raise SystemExit("Agent quota exhausted; result saved. Restore CLI quota before resuming.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--variants", default="A,B,C,D,E,F")
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--tolerance", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--poll", type=float, default=0.5)
    args = p.parse_args()
    if not all(math.isfinite(v) for v in (args.tolerance, args.timeout, args.poll)) or args.tolerance < 0 or args.timeout <= 0 or args.poll <= 0 or (args.limit is not None and args.limit <= 0):
        p.error("tolerance must be nonnegative; timeout, poll, limit must be positive")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
