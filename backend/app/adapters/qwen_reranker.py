"""Qwen3-VL-Reranker client (Colab A100 FastAPI worker).

POST {QWEN_RERANKER_URL}/rerank
  {"query": ..., "documents": [{"id": ..., "image": ...}], "top_k": ...}
  -> {"scores": [{"id": ..., "score": 0.98, "rank": 0}], "failed": [...]}

The worker is a single-tower pointwise reranker: it scores each `(query, image)`
pair directly, so candidates may come from ANY retriever (PE-Core here) and no
image ever has to be re-encoded. Run `aic26_qwen3vl_reranker8b_colab_server.ipynb`
to bring one up.

Every failure raises `QwenRerankerUnavailable`; the caller is expected to keep the
retriever's own order rather than surfacing the error, because reranking is a
refinement and must never become a single point of failure for visual search.
"""
from __future__ import annotations

import hashlib
from typing import Any

import httpx

from ..config import Settings
from .http_pool import PooledHttpClient


class QwenRerankerUnavailable(RuntimeError):
    """The rerank worker is unreachable, unconfigured or returned garbage."""


class QwenRerankerClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.mock = settings.mock_mode
        self._base = (settings.qwen_reranker_url or "").rstrip("/")
        self._http = PooledHttpClient()

    @property
    def enabled(self) -> bool:
        """Whether a rerank request can be attempted at all.

        Mock mode answers with a deterministic reorder so the console and the
        test suite can exercise the toggle without a GPU worker.
        """
        if not self.s.qwen_reranker_enabled:
            return False
        return self.mock or self.s.has_qwen_reranker

    @property
    def candidate_k(self) -> int:
        """How deep to retrieve BEFORE reranking.

        Reranking only reorders what it is given, so the candidate pool has to be
        wider than the result the operator asked for; a frame PE ranked 143rd is
        invisible to the reranker if only 100 candidates are sent.
        """
        return self.s.qwen_reranker_candidates

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.s.qwen_reranker_token:
            headers["Authorization"] = f"Bearer {self.s.qwen_reranker_token}"
        return headers

    async def health(self) -> dict[str, Any]:
        if not self.s.qwen_reranker_enabled:
            return {"ok": False, "mode": "disabled", "reason": "QWEN_RERANKER_ENABLED=false"}
        if self.mock:
            return {"ok": True, "mode": "mock", "model": "mock-qwen3-vl-reranker"}
        if not self.s.has_qwen_reranker:
            return {"ok": False, "mode": "disabled", "error": "QWEN_RERANKER_URL is not configured"}
        try:
            response = await self._http.get().get(
                f"{self._base}/health", headers=self._headers(), timeout=10.0
            )
            response.raise_for_status()
            return {"ok": True, **response.json()}
        except Exception as exc:  # noqa: BLE001 - health reports upstream details
            return {
                "ok": False,
                "mode": "unreachable",
                "error": str(exc).strip() or type(exc).__name__,
            }

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        *,
        instruction: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score `(query, document)` pairs; returns `[{"id", "score"}]` best first.

        Documents the worker could not score (a 404 keyframe, say) are simply
        absent from the result — the caller re-appends them in their original
        order rather than dropping frames PE did find.
        """
        if not documents:
            return []
        if self.mock:
            return _mock_scores(query, documents, top_k)
        if not self.enabled:
            raise QwenRerankerUnavailable(
                "Qwen reranker is not configured. Run "
                "aic26_qwen3vl_reranker8b_colab_server.ipynb and set "
                "QWEN_RERANKER_URL/QWEN_RERANKER_TOKEN in backend/.env."
            )
        payload: dict[str, Any] = {"query": query, "documents": documents}
        if instruction:
            payload["instruction"] = instruction
        if top_k:
            payload["top_k"] = int(top_k)
        timeout = self.s.qwen_reranker_timeout_seconds
        try:
            response = await self._http.get().post(
                f"{self._base}/rerank",
                json=payload,
                headers=self._headers(),
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            # Reranking a wide pool is genuinely slow the first time: the worker
            # downloads every keyframe before it can score anything. Say which
            # knob to turn instead of reporting a bare timeout.
            raise QwenRerankerUnavailable(
                f"no response after {timeout:.0f}s while scoring {len(documents)} keyframes "
                f"({type(exc).__name__}). Increase QWEN_RERANKER_TIMEOUT_SECONDS or reduce "
                "QWEN_RERANKER_CANDIDATES."
            ) from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text.strip()[:200]
            raise QwenRerankerUnavailable(
                f"worker returned HTTP {exc.response.status_code}"
                + (f": {body}" if body else "")
                + (" — does QWEN_RERANKER_TOKEN match the notebook secret?"
                   if exc.response.status_code == 401 else "")
            ) from exc
        except Exception as exc:  # noqa: BLE001 - normalized for the fail-open caller
            # httpx connect/protocol errors stringify to "", which would leave the
            # operator staring at a reason that says nothing. The class name is
            # the minimum useful detail, so it is never dropped.
            detail = str(exc).strip() or type(exc).__name__
            raise QwenRerankerUnavailable(f"could not reach worker: {detail}") from exc
        if not isinstance(data, dict):
            raise QwenRerankerUnavailable("Qwen reranker returned a non-object response")
        scores = data.get("scores")
        if not isinstance(scores, list):
            raise QwenRerankerUnavailable("Qwen reranker response has no `scores` array")
        out: list[dict[str, Any]] = []
        for item in scores:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("id") or "")
            if not doc_id:
                continue
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                continue
            out.append({"id": doc_id, "score": score})
        return out


def _mock_scores(
    query: str, documents: list[dict[str, Any]], top_k: int | None
) -> list[dict[str, Any]]:
    """Deterministic pseudo-reranking (mock mode only).

    The order deliberately differs from the input order so a test can tell that
    the rerank stage ran, while staying stable for the same query and candidates.
    """
    scored = []
    for document in documents:
        doc_id = str(document.get("id") or "")
        if not doc_id:
            continue
        digest = hashlib.sha256(f"{query}|{doc_id}".encode("utf-8")).digest()
        score = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        scored.append({"id": doc_id, "score": round(score, 6)})
    scored.sort(key=lambda item: -item["score"])
    return scored[: top_k or len(scored)]
