"""Client for the NVILA-8B QA worker hosted by the companion Colab notebook.

The worker is deliberately task-specific: it performs UIT-style answer hotspot
prediction, candidate-answer suggestion, and post-search visual verification.
All credentials remain server-side and mock mode never performs network I/O.
"""
from __future__ import annotations

from typing import Any

import httpx

from ..config import Settings
from .http_pool import failure_reason
from .qa_vision import QaVisionUnavailable


class NvilaUnavailable(QaVisionUnavailable):
    pass


class NvilaQaClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.mock = settings.mock_mode
        self._base = (settings.nvila_base_url or "").rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.s.nvila_token:
            headers["Authorization"] = f"Bearer {self.s.nvila_token}"
        return headers

    async def health(self) -> dict[str, Any]:
        if self.mock:
            return {
                "ok": True,
                "mode": "mock",
                "model": "mock-nvila-8b",
                "visual_verification_pass": True,
            }
        if not self.s.has_nvila:
            return {"ok": False, "mode": "disabled", "error": "NVILA_BASE_URL is not configured"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._base}/health", headers=self._headers())
                response.raise_for_status()
                return {"ok": True, **response.json()}
        except Exception as exc:  # noqa: BLE001 - health reports upstream details
            return {"ok": False, "mode": "unreachable", "error": failure_reason(exc)}

    async def analyze(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        *,
        max_answers: int = 5,
    ) -> dict[str, Any]:
        if self.mock:
            return _mock_analysis(question, candidates, max_answers)
        if not self.s.has_nvila:
            raise NvilaUnavailable(
                "NVILA QA is not configured. Run aic26_nvila8b_qa_colab_server.ipynb "
                "and set NVILA_BASE_URL/NVILA_TOKEN in backend/.env."
            )
        return await self._post("/qa/analyze", {
            "question": question,
            "candidates": candidates,
            "max_answers": max_answers,
            "two_pass": True,
        })

    async def verify_grounded(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        proposed_answers: list[dict[str, Any]],
        *,
        hotspot_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Ask NVILA to check web-search alternatives against visual evidence."""
        if self.mock:
            return _mock_verification(proposed_answers)
        if not self.s.has_nvila:
            raise NvilaUnavailable("NVILA visual verification is not configured")
        return await self._post("/qa/verify-grounded", {
            "question": question,
            "candidates": candidates,
            "proposed_answers": proposed_answers,
            "visual_hotspot_ids": hotspot_ids or [],
        })

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            timeout = httpx.Timeout(self.s.nvila_timeout_seconds, connect=20.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self._base}{path}",
                    json=payload,
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise NvilaUnavailable(f"NVILA worker returned {exc.response.status_code}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001 - normalized for the API layer
            raise NvilaUnavailable(f"NVILA worker unreachable: {exc}") from exc
        if not isinstance(data, dict):
            raise NvilaUnavailable("NVILA worker returned a non-object response")
        return data


def _mock_analysis(question: str, candidates: list[dict[str, Any]], max_answers: int) -> dict[str, Any]:
    first = candidates[0]
    answer = "Bản tin thời sự"
    return {
        "question": question,
        "model": "mock-nvila-8b",
        "mode": "mock",
        "answerable": True,
        "best_answer": answer,
        "best_candidate_id": first["candidate_id"],
        "candidate_answers": [
            {
                "answer": answer,
                "confidence": 0.82,
                "supporting_candidate_ids": [first["candidate_id"]],
                "reason": "Mock visual evidence for deterministic tests.",
            }
        ][:max_answers],
        "hotspots": [
            {
                "candidate_id": first["candidate_id"],
                "relevance": 0.91,
                "answer_support": "Mock answer-bearing frame.",
            }
        ],
        "uncertainty": "Mock mode; verify against the frame.",
        "latency_ms": 1.0,
        "cached": False,
        "warnings": [],
    }


def _mock_verification(proposed_answers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": "mock-nvila-8b",
        "mode": "mock",
        "verdicts": [
            {
                "answer": option.get("answer"),
                "status": "supported",
                "visual_confidence": 0.90,
                "supporting_candidate_ids": option.get("supporting_candidate_ids") or [],
                "reason": "Mock NVILA confirms consistency with the supplied visual clue.",
            }
            for option in proposed_answers
            if option.get("answer")
        ],
        "latency_ms": 1.0,
        "uncertainty": "",
        "warnings": [],
    }
