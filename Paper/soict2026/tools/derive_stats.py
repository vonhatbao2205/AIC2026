"""Derive every number the SOICT 2026 manuscript reports that is not already in
the committed benchmark snapshot (benchmarks/agent/results/soict-final-tol1).

Reads the git-ignored raw runs of the frozen final benchmark, makes no model or
network calls, and writes data/paper_stats.json. Run from the repository root:

    python3 Paper/soict2026/tools/derive_stats.py

Sources (unchanged): benchmarks/agent/runs/soict-final-dev-test-pooled-tol1-v1/
{test,dev}/{main,cumulative}/results.jsonl + evaluation-dataset.jsonl.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from benchmarks.agent.metrics import calibration_metrics, first_rank, video_rank  # noqa: E402

RUNS = REPO / "benchmarks/agent/runs/soict-final-dev-test-pooled-tol1-v1"
SNAP = REPO / "benchmarks/agent/results/soict-final-tol1"
OUT = Path(__file__).resolve().parents[1] / "data/paper_stats.json"
VARIANTS = "ABCDEF"
BOOTSTRAP, SEED = 10_000, 2026


def load_rows(cohort: str, exp: str) -> list[dict]:
    with open(RUNS / cohort / exp / "results.jsonl") as f:
        return [json.loads(line) for line in f]


def load_queries(cohort: str, exp: str) -> dict[str, dict]:
    with open(RUNS / cohort / exp / "evaluation-dataset.jsonl") as f:
        return {q["query_id"]: q for q in map(json.loads, f)}


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def paired(by, qids, a, b, key, idx):
    d = [float(by[q][a]["metrics"][key] or 0) - float(by[q][b]["metrics"][key] or 0) for q in qids]
    boots = sorted(sum(d[i] for i in s) / len(s) for s in idx)
    return {"delta": sum(d) / len(d), "lo": boots[int(.025 * len(boots))], "hi": boots[int(.975 * len(boots)) - 1]}


def mcnemar(by, qids, a, b, key="r1"):
    n10 = sum(1 for q in qids if by[q][a]["metrics"][key] and not by[q][b]["metrics"][key])
    n01 = sum(1 for q in qids if not by[q][a]["metrics"][key] and by[q][b]["metrics"][key])
    n = n10 + n01
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n) if n else 1.0
    return {"a_only": n10, "b_only": n01, "p_exact": p}


def controller_stats(rows):
    """Routing decisions, guard overrides and decision-layer overhead per variant."""
    out = {}
    for v in "DEF":
        R = [r for r in rows if r["variant"] == v]
        first, stops, props, n_dec, overridden = collections.Counter(), collections.Counter(), 0, 0, 0
        jev_lat, jev_cost, jev_time, share, jev_tok = [], [], [], [], []
        second_stop_p = []
        for r in R:
            ctrl = r["snapshot"]["controller"]
            stops[ctrl.get("stop_reason")] += 1
            decisions = [t for t in r["trace"]["trace"] if t["kind"] == "decision"]
            actions = [t["action"] for t in r["trace"]["trace"] if t["kind"] == "action"]
            first[actions[0] if actions else "none"] += 1
            n_dec += len(decisions)
            props += sum(1 for d in decisions if d["answers"]["action"]["choice"] == "STOP")
            overridden += sum(1 for t in r["trace"]["trace"] if t["kind"] == "stop_overridden")
            if len(decisions) > 1:
                second_stop_p.append(decisions[1]["answers"]["action"]["probabilities"].get("STOP", 0.0))
            calls = [c for c in ctrl.get("decision_calls") or [] if c.get("backend") == "jev"]
            jev_lat += [c["latency_ms"] for c in calls if c.get("latency_ms")]
            jev_tok += [c["input_tokens"] for c in calls if c.get("input_tokens")]
            jev_cost.append(sum(c.get("cost_usd") or 0 for c in calls))
            t = sum(c.get("latency_ms") or 0 for c in calls) / 1000
            jev_time.append(t)
            share.append(t / r["metrics"]["latency_s"])
        out[v] = {"queries": len(R), "first_action": dict(first), "stop_reason": dict(stops),
                  "router_decisions": n_dec, "router_stop_proposals": props, "stop_overridden": overridden,
                  "second_decision_stop_prob_median": median(second_stop_p),
                  "second_decision_stop_prob_gt_half": sum(1 for p in second_stop_p if p > .5),
                  "second_decisions": len(second_stop_p),
                  "jev_call_latency_ms_p50": median(jev_lat),
                  "jev_call_latency_ms_p95": sorted(jev_lat)[int(.95 * len(jev_lat))] if jev_lat else None,
                  "jev_input_tokens_p50": median(jev_tok), "jev_calls_total": len(jev_lat),
                  "jev_cost_usd_per_query_mean": statistics.mean(jev_cost),
                  "jev_time_s_per_query_mean": statistics.mean(jev_time),
                  "jev_wall_share_median": median(share)}
    return out


def agent_stats(rows):
    out = {}
    for v in "BCEF":
        R = [r for r in rows if r["variant"] == v]
        entry = {"tools_per_query": collections.Counter()}
        for name in ("claude", "codex"):
            done = [r["snapshot"]["agents"][name] for r in R if r["snapshot"]["agents"].get(name, {}).get("status") == "done"]
            entry[name] = {"runs": len(done), "elapsed_s_p50": median(a.get("elapsed_s") for a in done),
                           "tool_calls_mean": statistics.mean(a.get("tool_calls") or 0 for a in done) if done else None}
        for r in R:
            for t in r["trace"]["tools"]:
                entry["tools_per_query"][t["tool"]] += 1 / len(R)
        entry["tools_per_query"] = {k: round(n, 3) for k, n in entry["tools_per_query"].most_common()}
        entry["tokens_mean"] = statistics.mean(r["metrics"].get("tokens") or 0 for r in R)
        out[v] = entry
    retrieval = [r for r in rows if r["variant"] == "A"]
    channels = collections.Counter(c for r in retrieval for c in {c for f in r["snapshot"]["baseline_ranking"] for c in f.get("channels", [])})
    out["retrieval_channels_queries"] = dict(channels)
    out["retrieval_profiles"] = dict(collections.Counter(r["snapshot"]["retrieval_database"] for r in retrieval))
    out["retrieval_latency_s_p50"] = median(r["metrics"]["latency_s"] for r in retrieval)
    return out


def router_replay(rows, queries):
    """Post-hoc counterfactual: accept the router's first STOP proposal.

    Uses the ranking already published before that decision. This is a trace
    analysis on logged runs, not a deployed or tuned policy.
    """
    out = {}
    for v in "EF":
        res = []
        for r in (r for r in rows if r["variant"] == v):
            q, m = queries[r["query_id"]], r["metrics"]
            final = {"r1": m["r1"], "r5": m["r5"], "video_r1": int(m["video_rank"] == 1),
                     "agents": m["codex_calls"] + m["claude_calls"], "latency": m["latency_s"]}
            invoked, ranking, alt = set(), r["snapshot"].get("baseline_ranking") or [], None
            for ev in r["trace"]["trace"]:
                if ev["kind"] == "agent_results":
                    invoked.update(ev["agents"])
                elif ev["kind"] == "ranking":
                    ranking = ev["ranking"]
                elif ev["kind"] == "decision" and ev["answers"]["action"]["choice"] == "STOP":
                    rank = first_rank(ranking, q, 1.0)
                    alt = {"r1": int(rank == 1), "r5": int(rank is not None and rank <= 5),
                           "video_r1": int(video_rank(ranking, q) == 1), "agents": len(invoked), "latency": ev["at_s"]}
                    break
            res.append({"proposed": alt is not None, "final": final, "accept": alt or final})
        summary = {}
        for key in ("final", "accept"):
            summary[key] = {k: statistics.mean(x[key][k] for x in res) for k in ("r1", "r5", "video_r1", "agents")}
            summary[key]["latency_p50"] = median(x[key]["latency"] for x in res)
        summary["proposals"] = sum(x["proposed"] for x in res)
        summary["r1_on_proposals_final"] = sum(x["final"]["r1"] for x in res if x["proposed"])
        summary["r1_on_proposals_accept"] = sum(x["accept"]["r1"] for x in res if x["proposed"])
        out[v] = summary
    return out


def latency_samples(rows):
    """Per-call/per-run wall times of each layer in the adaptive variant E."""
    E = [r for r in rows if r["variant"] == "E"]
    agent = lambda name: [r["snapshot"]["agents"][name]["elapsed_s"] for r in E
                          if r["snapshot"]["agents"].get(name, {}).get("status") == "done"]
    return {"retrieval_s": [r["metrics"]["latency_s"] for r in rows if r["variant"] == "A"],
            "jev_call_s": [c["latency_ms"] / 1000 for r in E for c in r["snapshot"]["controller"].get("decision_calls") or []
                           if c.get("backend") == "jev" and c.get("latency_ms")],
            "claude_s": agent("claude"), "codex_s": agent("codex"),
            "end_to_end_s": [r["metrics"]["latency_s"] for r in E]}


def post_agent_decisions(rows, queries):
    """Router P(STOP) at the first decision after an agent returned, paired with
    whether the ranking published at that moment already had a correct top-1."""
    out = {}
    for v in "EF":
        res = []
        for r in (r for r in rows if r["variant"] == v):
            q, invoked, ranking = queries[r["query_id"]], set(), r["snapshot"].get("baseline_ranking") or []
            for ev in r["trace"]["trace"]:
                if ev["kind"] == "agent_results":
                    invoked.update(ev["agents"])
                elif ev["kind"] == "ranking":
                    ranking = ev["ranking"]
                elif ev["kind"] == "decision" and invoked:
                    rank = first_rank(ranking, q, 1.0)
                    res.append({"p_stop": ev["answers"]["action"]["probabilities"].get("STOP", 0.0),
                                "choice": ev["answers"]["action"]["choice"], "top1_correct": int(rank == 1),
                                "query_type": r["query_type"]})
                    break
        out[v] = res
    return out


def reliability(rows, variant, bins=5):
    pairs = [tuple(p) for r in rows if r["variant"] == variant for p in r["metrics"].get("calibration") or []]
    edges = [i / bins for i in range(bins + 1)]
    table = []
    for lo, hi in zip(edges, edges[1:]):
        b = [(p, y) for p, y in pairs if lo <= p < hi or (hi == 1 and p == 1)]
        table.append({"lo": lo, "hi": hi, "n": len(b), "conf": statistics.mean(p for p, _ in b) if b else None,
                      "acc": statistics.mean(y for _, y in b) if b else None})
    return {"bins": table, **calibration_metrics(pairs)}


def final_ranking_diagnostic(rows, queries, idx):
    """Remove ONLY final verification ordering from the same logged candidates.

    EvidenceBoard.ranked(False) sorts by (-fusion_score, first_seen_s, key).
    All three fields are saved in every final snapshot. No model is rerun and
    no candidate, answer, earlier verification or routing decision is changed.
    This is not a live no-verifier policy, nor the missing always-two+verifier
    ablation. Controls A/B/C/D must reproduce their entire logged order.
    """
    cells, records = collections.defaultdict(dict), []
    for r in rows:
        qid, v = r["query_id"], r["variant"]
        ranking = r["snapshot"]["ranking"]
        assert len({f["submit_keyframe_id"] for f in ranking}) == len(ranking)
        unverified = sorted(ranking, key=lambda f: (
            -f["fusion_score"], f["first_seen_s"], f["submit_keyframe_id"]))
        rank = first_rank(ranking, queries[qid], 1.0)
        raw_rank = first_rank(unverified, queries[qid], 1.0)
        assert int(rank == 1) == r["metrics"]["r1"], (qid, v)
        if v in "ABCD":
            assert unverified == ranking, ("control order mismatch", qid, v)
        cells[v][qid] = {
            "verified": {"metrics": {"r1": int(rank == 1)}},
            "rrf": {"metrics": {"r1": int(raw_rank == 1)}},
        }
        records.append({"query_id": qid, "variant": v,
                        "logged_rank": rank, "same_candidates_rrf_rank": raw_rank,
                        "ranking_changed": unverified != ranking,
                        "top_frame_changed": bool(ranking and ranking[0] != unverified[0])})
    result = {"scope": "Final ordering only; fixed logged candidates, answers and trajectories.",
              "controls": "All 344 A/B/C/D rankings reproduced exactly.", "variants": {}}
    for v in VARIANTS:
        ids = sorted(cells[v])
        assert len(ids) == len(idx[0])
        result["variants"][v] = {
            "queries": len(ids),
            "logged_correct": sum(cells[v][q]["verified"]["metrics"]["r1"] for q in ids),
            "rrf_correct": sum(cells[v][q]["rrf"]["metrics"]["r1"] for q in ids),
            "logged_minus_rrf": paired(cells[v], ids, "verified", "rrf", "r1", idx),
            "discordance": mcnemar(cells[v], ids, "verified", "rrf"),
            "ranking_changed": sum(r["ranking_changed"] for r in records if r["variant"] == v),
            "top_frame_changed": sum(r["top_frame_changed"] for r in records if r["variant"] == v),
        }
    result["per_query"] = records
    return result


def main():
    rows = load_rows("test", "main")
    queries = load_queries("test", "main")
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["query_id"]][r["variant"]] = r
    qids = sorted(q for q in by if set(by[q]) == set(VARIANTS))
    random.seed(SEED)
    idx = [[random.randrange(len(qids)) for _ in qids] for _ in range(BOOTSTRAP)]
    pairs = [("E", "C"), ("E", "B"), ("E", "A"), ("E", "D"), ("E", "F"), ("F", "C"), ("F", "A"), ("C", "B"), ("C", "A"), ("B", "A")]
    stats = {
        "source": {"runs": str(RUNS.relative_to(REPO)), "snapshot": str(SNAP.relative_to(REPO)),
                   "bootstrap": BOOTSTRAP, "seed": SEED, "unit": "base query (paired)", "tolerance_s": 1},
        "test_queries": len(qids),
        "test_composition": dict(collections.Counter(queries[q]["query_type"] for q in qids)),
        "test_query_chars_mean": statistics.mean(len(queries[q]["query"]) for q in qids),
        "test_target_videos": len({t["video_id"] for q in qids for t in queries[q]["targets"]}),
        "paired": {f"{a}-{b}": {k: paired(by, qids, a, b, k, idx) for k in ("r1", "r5", "mrr", "moment_r1")} for a, b in pairs},
        "mcnemar_r1": {f"{a}-{b}": mcnemar(by, qids, a, b) for a, b in pairs},
        "controller": controller_stats(rows),
        "agents": agent_stats(rows),
        "router_replay": router_replay(rows, queries),
        "post_agent_decisions": post_agent_decisions(rows, queries),
        "latency_samples": latency_samples(rows),
        "reliability": {v: reliability(rows, v) for v in "DEF"},
        "final_ranking_diagnostic": final_ranking_diagnostic(rows, queries, idx),
    }
    # Cumulative-hint paired contrasts (fixed 24-query TEST cohort, per level).
    with open(SNAP / "test/cumulative/hint_results.csv") as f:
        hint_rows = list(csv.DictReader(f))
    cum = {}
    for stage in ("h1", "h2", "full"):
        cell = collections.defaultdict(dict)
        for r in hint_rows:
            if r["hint_stage"] == stage:
                cell[r["query_id"]][r["variant"]] = {"metrics": {
                    "video_r1": int(r["video_rank"] == "1"), "r1": int(r["r1"]), "moment_r1": int(r["moment_r1"])}}
        ids = sorted(q for q in cell if set(cell[q]) == {"A", "C", "F"})
        rnd = random.Random(SEED)
        bidx = [[rnd.randrange(len(ids)) for _ in ids] for _ in range(BOOTSTRAP)]
        cum[stage] = {"queries": len(ids), **{f"{a}-{b}": {k: paired(cell, ids, a, b, k, bidx) for k in ("video_r1", "r1")}
                                             for a, b in (("F", "C"), ("F", "A"), ("C", "A"))}}
    stats["cumulative_paired"] = cum
    stats["source"]["sha256"] = {}
    for filename in ("results.jsonl", "evaluation-dataset.jsonl"):
        path = RUNS / "test/main" / filename
        with path.open("rb") as f:
            stats["source"]["sha256"][filename] = hashlib.file_digest(f, "sha256").hexdigest()
    summary = json.loads((SNAP / "test/main/summary.json").read_text())
    stats["main"] = {v: {k: summary[v][k] for k in ("r1", "r5", "mrr", "moment_r1", "video_r1", "agent_rescue_at_1",
                                                    "rescue1_eligible_queries", "latency_p50_s", "latency_p95_s",
                                                    "codex_calls", "claude_calls", "jev_calls", "tool_calls", "tokens_mean",
                                                    "time_to_first_correct_p50_s", "time_to_first_correct_observed_queries")}
                     for v in VARIANTS}
    video = json.loads((SNAP / "test/main/video-report/summary.json").read_text())
    stats["video_report_keys"] = list(video)[:10]
    with open(SNAP / "test/main/paper_tables.csv") as f:
        stats["by_task"] = [row for row in csv.DictReader(f)]
    for cohort in ("dev", "pooled"):
        s = json.loads((SNAP / cohort / "main/summary.json").read_text())
        stats[f"{cohort}_r1"] = {v: s[v]["r1"] for v in VARIANTS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
