"""OpenRouter Decisions API, deliberately separate from chat completions."""
from __future__ import annotations

import math
import time

import httpx

from ...adapters.http_pool import PooledHttpClient


class DecisionError(Exception):
    """Sanitized provider failure. Never includes request headers/body."""


def probability(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise DecisionError("Invalid probability in decision response")
    return float(value)


def validate_answers(answers, questions):
    if not isinstance(answers, dict):
        raise DecisionError("Missing typed answers")
    for key, question in questions.items():
        answer = answers.get(key)
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise DecisionError(f"Missing or invalid answer type: {key}")
        if question["type"] == "noul":
            probability(answer.get("noul"))
        elif question["type"] == "choice":
            options = question["criteria"]
            if answer.get("choice") not in options:
                raise DecisionError(f"Unknown choice: {key}")
            probs = answer.get("probabilities")
            if not isinstance(probs, dict) or set(probs) != set(options):
                raise DecisionError(f"Incomplete probability distribution: {key}")
            if abs(sum(probability(v) for v in probs.values()) - 1) > .02:
                raise DecisionError(f"Invalid probability distribution: {key}")


class JevClient:
    URL = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, settings):
        self.settings = settings
        self.http = PooledHttpClient(timeout=settings.jev_timeout_seconds)
        self.calls: list[dict] = []

    async def decide(self, state: dict, questions: dict) -> dict:
        s = self.settings
        if not s.openrouter_api_key or s.mock_mode:
            raise DecisionError("Jev unavailable: missing OpenRouter key or mock mode")
        started = time.monotonic()
        record = {"backend": "jev", "model": s.jev_model, "questions": list(questions), "cost_usd": None,
                  "input_tokens": None, "output_tokens": None, "ok": False}
        self.calls.append(record)
        try:
            response = await self.http.get().post(self.URL, headers={"Authorization": f"Bearer {s.openrouter_api_key}"},
                                                  json={"model": s.jev_model, "state": state, "questions": questions})
            if response.status_code != 200:
                raise DecisionError(f"Jev HTTP {response.status_code}")
            data = response.json()
            answers = data.get("answers")
            usage = data.get("usage") or {}
            for target, names in {"input_tokens": ("input_tokens", "inputTokens", "prompt_tokens"),
                                  "output_tokens": ("output_tokens", "outputTokens", "completion_tokens"),
                                  "cost_usd": ("cost",)}.items():
                for name in names:
                    v = usage.get(name)
                    if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0:
                        record[target] = v
                        break
            record["resolved_model"] = data.get("model")
            validate_answers(answers, questions)
            record["ok"] = True
            return answers
        except DecisionError:
            raise
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
            raise DecisionError(f"Jev request failed: {type(exc).__name__}") from None
        finally:
            record["latency_ms"] = round((time.monotonic() - started) * 1000, 2)

    async def close(self):
        if self.http._client is not None:
            await self.http._client.aclose()
