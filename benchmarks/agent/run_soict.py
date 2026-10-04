"""Run the SOICT main A–F and Cumulative-Hint Evaluation A/C/F experiments.

Each experiment has its own manifest and results. All runs are serial; a quota
stop propagates immediately. This command never fits calibration on test.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

from .run_benchmark import main_async

EXPERIMENTS = {
    "main": ("full", "A,B,C,D,E,F"),
    "cumulative": ("cumulative", "A,C,F"),
    "ablations": ("full", "F,rule,rule_router,llm_router,no_verifier,claude_verifier,base_tools,compare_only,no_roles"),
}


async def run_suite(args):
    if args.experiment == "ablations" and args.split != "dev":
        raise ValueError("The SOICT ablation/tuning phase uses --split dev")
    experiments = ["main", "cumulative"] if args.experiment == "all" else [args.experiment]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        mode, variants = EXPERIMENTS[experiment]
        options = argparse.Namespace(**vars(args))
        options.hint_mode, options.variants = mode, variants
        options.output = str(output / f"{experiment}-{args.split}")
        await main_async(options)
    plans = {name: json.loads((output / f"{name}-{args.split}" / "run_plan.json").read_text()) for name in experiments}
    (output / f"suite-{args.split}.json").write_text(json.dumps({
        "split": args.split, "dry_run": args.dry_run,
        "planned_runs": sum(p["planned_runs"] for p in plans.values()),
        "experiments": {name: {"directory": f"{name}-{args.split}", "base_queries": p["base_queries"],
                               "planned_runs": p["planned_runs"]} for name, p in plans.items()},
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", choices=["all", *EXPERIMENTS], default="all")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, help="Base queries per experiment, after eligibility filtering")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tolerance", type=float, default=5.0, help="Moment tolerance in seconds (default: 5)")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--poll", type=float, default=.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.tolerance, args.timeout, args.poll)) or args.tolerance < 0 or args.timeout <= 0 or args.poll <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("tolerance must be nonnegative; timeout, poll, limit must be positive")
    asyncio.run(run_suite(args))


if __name__ == "__main__":
    main()
