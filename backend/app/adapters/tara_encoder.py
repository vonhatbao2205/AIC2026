"""TARA text encoder client for the immutable 3584-d InfoShot++ clip index."""
from __future__ import annotations

import math
from typing import Any

from ..config import Settings
from .http_pool import PooledHttpClient
from .pe_encoder import _pseudo_vector

DIM = 3584
MODEL_ID = "bpiyush/TARA"
MODEL_REVISION = "3e7cb730d86ae10da7eb1c85f17e45ece5a5e353"
# The worker reports the L21-L30 artifact's fingerprint. The batch-2 artifacts have
# their own fingerprints but the same embedding contract (611ef1c9…), so one worker
# serves the whole collection.
SEMANTIC_FINGERPRINT = "ef197331649dfb93e7057304f294d8bd1cd98bf22b1902bca48daf6dc4874819"


class TaraEncoderUnavailable(RuntimeError):
    pass


class TaraEncoderClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self._base = (settings.tara_encoder_url or "").rstrip("/")
        self._http = PooledHttpClient()

    def _headers(self) -> dict[str, str]:
        # The Colab worker behind tara.baoencoder.site has no bearer token; one is
        # sent only when TARA_ENCODER_TOKEN is set for a server that still wants it.
        headers = {"Content-Type": "application/json"}
        if self.s.tara_encoder_token:
            headers["Authorization"] = f"Bearer {self.s.tara_encoder_token}"
        return headers

    @staticmethod
    def _validate_contract(payload: dict[str, Any]) -> None:
        if (payload.get("model") != MODEL_ID
                or payload.get("revision") != MODEL_REVISION
                or payload.get("semantic_fingerprint") != SEMANTIC_FINGERPRINT
                or payload.get("dim") != DIM
                or payload.get("l2_normalized") is not True):
            raise TaraEncoderUnavailable("TARA worker contract differs from indexed clips")

    async def health(self) -> dict[str, Any]:
        if not self.s.is_infoshotpp or not self.s.tara_enabled:
            return {"ok": False, "mode": "disabled"}
        if self.s.mock_mode:
            return {"ok": True, "mode": "mock", "dim": DIM}
        if not self.s.tara_encoder_url:
            return {"ok": False, "mode": "disabled", "reason": "TARA_ENCODER_URL missing"}
        try:
            response = await self._http.get().get(
                f"{self._base}/health", headers=self._headers(), timeout=10.0
            )
            response.raise_for_status()
            payload = response.json()
            self._validate_contract(payload)
            return {"ok": True, **payload}
        except Exception as exc:  # noqa: BLE001 - health should report the cause
            return {"ok": False, "mode": "unreachable", "error": str(exc) or type(exc).__name__}

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.s.is_infoshotpp or not self.s.tara_enabled:
            raise TaraEncoderUnavailable("TARA is disabled for this retrieval profile")
        if self.s.mock_mode:
            return [_pseudo_vector(text, DIM) for text in texts]
        if not self.s.has_tara_search:
            raise TaraEncoderUnavailable("TARA encoder or Milvus is not configured")
        try:
            response = await self._http.get().post(
                f"{self._base}/encode-text", json={"texts": texts},
                headers=self._headers(), timeout=self.s.tara_encoder_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            self._validate_contract(payload)
        except TaraEncoderUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TaraEncoderUnavailable(f"TARA worker unavailable: {str(exc) or type(exc).__name__}") from exc
        vectors = payload.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise TaraEncoderUnavailable("TARA worker returned wrong vector count")
        out = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != DIM:
                raise TaraEncoderUnavailable("TARA worker returned wrong vector dimension")
            try:
                values = [float(value) for value in vector]
            except (ValueError, TypeError) as exc:
                raise TaraEncoderUnavailable("TARA worker returned nonnumeric vector") from exc
            if not all(math.isfinite(value) for value in values):
                raise TaraEncoderUnavailable("TARA worker returned nonfinite vector")
            norm = math.sqrt(math.fsum(value * value for value in values))
            if abs(norm - 1.0) > 5e-3:
                raise TaraEncoderUnavailable(f"TARA vector norm is {norm:.6f}")
            out.append(values)
        return out
