"""Qwen3-VL-Embedding-8B text encoder client for InfoShot++.

POST ``{QWEN3_VL_ENCODER_URL}/encode-text`` with ``{"texts": [...]}`` and
receive native, FP32-normalized 4096-d vectors in ``{"vectors": [...]}``.
The companion Colab notebook owns the exact query instruction and model
contract. This client deliberately never falls back to pseudo-vectors in live
mode: searching a real collection with a fabricated vector would look healthy
while returning random frames.
"""
from __future__ import annotations

import math
from typing import Any

import httpx

from ..config import Settings
from .http_pool import PooledHttpClient
from .pe_encoder import _pseudo_vector

QWEN3_VL_DIM = 4096


class Qwen3VlEncoderUnavailable(RuntimeError):
    """The Qwen embedding worker is absent, unreachable, or returned bad vectors."""


class Qwen3VlEncoderClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.mock = settings.mock_mode
        self._base = (settings.qwen3_vl_encoder_url or "").rstrip("/")
        self._http = PooledHttpClient()

    @property
    def enabled(self) -> bool:
        return bool(self.s.is_infoshotpp and (self.mock or self.s.has_qwen3_vl_encoder))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.s.qwen3_vl_encoder_token:
            headers["Authorization"] = f"Bearer {self.s.qwen3_vl_encoder_token}"
        return headers

    async def health(self) -> dict[str, Any]:
        if not self.s.is_infoshotpp:
            return {
                "ok": False,
                "mode": "disabled",
                "reason": "Qwen3-VL embeddings are indexed only for InfoShot++",
            }
        if self.mock:
            return {
                "ok": True,
                "mode": "mock",
                "model": "mock-qwen3-vl-embedding-8b",
                "dim": QWEN3_VL_DIM,
            }
        if not self.s.has_qwen3_vl_encoder:
            return {
                "ok": False,
                "mode": "disabled",
                "reason": "QWEN3_VL_ENCODER_URL/TOKEN are not configured",
            }
        try:
            response = await self._http.get().get(
                f"{self._base}/health", headers=self._headers(), timeout=10.0
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("dim") is not None:
                if int(payload["dim"]) != QWEN3_VL_DIM:
                    return {
                        "ok": False,
                        "mode": "incompatible",
                        "error": (
                            f"worker reports dim={payload['dim']}; expected {QWEN3_VL_DIM}"
                        ),
                    }
            return {"ok": True, **payload}
        except Exception as exc:  # noqa: BLE001 - health returns upstream detail
            return {
                "ok": False,
                "mode": "unreachable",
                "error": str(exc).strip() or type(exc).__name__,
            }

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.s.is_infoshotpp:
            raise Qwen3VlEncoderUnavailable(
                "Qwen3-VL image embeddings are not available for the BTC profile"
            )
        if self.mock:
            return [_pseudo_vector(text, QWEN3_VL_DIM) for text in texts]
        if not self.s.has_qwen3_vl_encoder:
            raise Qwen3VlEncoderUnavailable(
                "Qwen3-VL encoder is not configured. Start the Colab encoder notebook and set "
                "QWEN3_VL_ENCODER_URL/QWEN3_VL_ENCODER_TOKEN."
            )

        timeout = self.s.qwen3_vl_encoder_timeout_seconds
        try:
            response = await self._http.get().post(
                f"{self._base}/encode-text",
                json={"texts": texts},
                headers=self._headers(),
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise Qwen3VlEncoderUnavailable(
                f"Qwen3-VL encoder did not respond after {timeout:.0f}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()[:200]
            raise Qwen3VlEncoderUnavailable(
                f"Qwen3-VL encoder returned HTTP {exc.response.status_code}"
                + (f": {detail}" if detail else "")
            ) from exc
        except Exception as exc:  # noqa: BLE001 - normalize transport/JSON errors
            detail = str(exc).strip() or type(exc).__name__
            raise Qwen3VlEncoderUnavailable(f"could not reach Qwen3-VL encoder: {detail}") from exc

        vectors = payload.get("vectors") if isinstance(payload, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise Qwen3VlEncoderUnavailable(
                "Qwen3-VL encoder response must contain one vector per input text"
            )
        out: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != QWEN3_VL_DIM:
                got = len(vector) if isinstance(vector, list) else "non-list"
                raise Qwen3VlEncoderUnavailable(
                    f"Qwen3-VL encoder returned dimension {got}; expected {QWEN3_VL_DIM}"
                )
            try:
                numeric = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise Qwen3VlEncoderUnavailable(
                    "Qwen3-VL encoder returned a non-numeric vector"
                ) from exc
            if not all(math.isfinite(value) for value in numeric):
                raise Qwen3VlEncoderUnavailable(
                    "Qwen3-VL encoder returned NaN or infinite vector values"
                )
            norm = math.sqrt(math.fsum(value * value for value in numeric))
            if abs(norm - 1.0) > 5e-3:
                raise Qwen3VlEncoderUnavailable(
                    f"Qwen3-VL encoder returned non-unit vector (L2 norm={norm:.6f})"
                )
            out.append(numeric)
        return out
