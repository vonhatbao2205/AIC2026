"""Fit temperature from development-run final relevance probabilities only."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from app.agent.decision.calibration import fit_temperature  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", default="F")
    args = p.parse_args()
    samples = []
    models, aggregations = set(), set()
    for line in Path(args.results).read_text().splitlines():
        row = json.loads(line)
        if row["variant"] != args.variant:
            continue
        if row.get("split") != "dev":
            raise ValueError("Cannot fit on test or unspecified split")
        if row.get("snapshot", {}).get("controller", {}).get("calibrated"):
            raise ValueError("Input must contain raw, uncalibrated development probabilities")
        calls = row.get("snapshot", {}).get("controller", {}).get("decision_calls", [])
        verifier = row.get("snapshot", {}).get("controller", {}).get("verifier", "jev")
        models.update(c["model"] for c in calls if c.get("backend") == verifier)
        aggregations.add("constraints" if row.get("snapshot", {}).get("policy") == "full" else "holistic")
        samples.extend({"query_id": row["query_id"], "split": "dev", "probability": p, "label": y}
                       for p, y in row["metrics"]["calibration"])
    artifact = fit_temperature(samples)
    if len(models) != 1 or len(aggregations) != 1:
        raise ValueError("Fit needs one observed verifier model and one aggregation scheme")
    artifact.update(variant=args.variant, model=models.pop(), aggregation=aggregations.pop())
    Path(args.output).write_text(json.dumps(artifact, indent=2))
    print(f"Fitted on {artifact['n']} candidate labels from {len(artifact['query_ids'])} development queries")


if __name__ == "__main__":
    main()
