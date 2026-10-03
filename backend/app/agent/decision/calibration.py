"""Temperature calibration, fitted only on an explicitly held-out dev split."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


def geometric_score(probabilities: list[float], weights: list[float]) -> float:
    if not probabilities or len(probabilities) != len(weights) or sum(weights) <= 0:
        return 0.0
    return math.exp(sum(w * math.log(max(1e-9, min(1.0, p))) for p, w in zip(probabilities, weights)) / sum(weights))


@dataclass(frozen=True)
class Calibration:
    temperature: float = 1.0
    fitted: bool = False
    query_ids: tuple[str, ...] = ()

    def apply(self, p: float) -> float:
        p = min(1 - 1e-9, max(1e-9, p))
        z = max(-30.0, min(30.0, math.log(p / (1 - p)) / self.temperature))
        return 1 / (1 + math.exp(-z))

    @classmethod
    def load(cls, path: str | None, *, model: str | None = None, aggregation: str | None = None):
        if not path:
            return cls()
        data = json.loads(Path(path).read_text())
        t = float(data["temperature"])
        if model and data.get("model") and data["model"] != model:
            raise ValueError("Calibration model does not match this verifier")
        if aggregation and data.get("aggregation") and data["aggregation"] != aggregation:
            raise ValueError("Calibration aggregation does not match this policy")
        if not math.isfinite(t) or not .05 <= t <= 20 or data.get("split") != "dev" or not data.get("query_ids"):
            raise ValueError("Calibration needs a valid temperature and development query IDs")
        return cls(t, True, tuple(data["query_ids"]))


def fit_temperature(rows: list[dict]) -> dict:
    if not rows or any(row.get("split") != "dev" or not row.get("query_id") for row in rows):
        raise ValueError("Fit requires labeled development rows with query_id and split=dev")
    for row in rows:
        p, y = row["probability"], row["label"]
        if not math.isfinite(p) or not 0 <= p <= 1 or y not in (0, 1):
            raise ValueError("Invalid calibration observation")
    def loss(t):
        cal = Calibration(t)
        return sum(-y * math.log(cal.apply(p)) - (1 - y) * math.log(1 - cal.apply(p))
                   for p, y in ((r["probability"], r["label"]) for r in rows)) / len(rows)
    candidates = [math.exp(math.log(.05) + i * math.log(400) / 300) for i in range(301)]
    t = min(candidates, key=loss)
    return {"version": 1, "split": "dev", "temperature": t, "n": len(rows), "nll": loss(t),
            "query_ids": sorted({r["query_id"] for r in rows})}
