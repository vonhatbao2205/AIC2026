"""PE-Core-G14 text encoder client (Kaggle FastAPI server).

POST {PE_ENCODER_URL}/encode-text  {"texts": [...]}  ->  {"dim":1280,"vectors":[[...]]}
Optional bearer auth via PE_ENCODER_TOKEN. In mock mode returns a deterministic
unit-norm pseudo-vector (never used for real Milvus search in mock mode).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import httpx

from ..config import Settings

PE_DIM = 1280
GLAP_DIM = 1024


class PeEncoderClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.mock = settings.mock_mode or not settings.has_pe_encoder
        self._base = (settings.pe_encoder_url or "").rstrip("/")

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.s.pe_encoder_token:
            h["Authorization"] = f"Bearer {self.s.pe_encoder_token}"
        return h

    async def health(self) -> dict[str, Any]:
        if self.mock:
            return {"ok": True, "mode": "mock"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base}/health", headers=self._headers())
                resp.raise_for_status()
                return {"ok": True, **resp.json()}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        if self.mock:
            return [_pseudo_vector(t) for t in texts]
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base}/encode-text",
                json={"texts": texts},
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
        return data["vectors"]


class GlapEncoderClient:
    """GLAP audio↔text encoder client (POST {glap_url}/encode-audio-text -> 1024-d).

    Same Kaggle server as PE by default. Produces text embeddings in the GLAP
    audio space so a text query can be cosine-searched against the audio vectors
    in Milvus `aic26_audio_glap_v1`."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.mock = settings.mock_mode or not settings.has_glap
        self._base = (settings.glap_url or "").rstrip("/")

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.s.pe_encoder_token:
            h["Authorization"] = f"Bearer {self.s.pe_encoder_token}"
        return h

    async def health(self) -> dict[str, Any]:
        if self.mock:
            return {"ok": True, "mode": "mock"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Cheap check: the shared server exposes /health.
                resp = await client.get(f"{self._base}/health", headers=self._headers())
                resp.raise_for_status()
                return {"ok": True}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        if self.mock:
            return [_pseudo_vector(t, GLAP_DIM) for t in texts]
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base}/encode-audio-text",
                json={"texts": texts},
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()["vectors"]


def _pseudo_vector(text: str, dim: int = PE_DIM) -> list[float]:
    """Deterministic unit-norm vector derived from text hash (mock only)."""
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    for i in range(dim):
        b = seed[i % len(seed)]
        vals.append(((b / 255.0) * 2.0) - 1.0)
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]
