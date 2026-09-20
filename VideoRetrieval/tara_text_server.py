#!/usr/bin/env python3
"""TARA text-only FastAPI worker for the AIC2026 Colab notebook.

This file is embedded verbatim in the notebook so the notebook is standalone.
Run under its pinned Python 3.10 environment, with TARA_SERVICE_TOKEN set.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Header, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MODEL_ID = "bpiyush/TARA"
MODEL_REVISION = "3e7cb730d86ae10da7eb1c85f17e45ece5a5e353"
SEMANTIC_FINGERPRINT = "ef197331649dfb93e7057304f294d8bd1cd98bf22b1902bca48daf6dc4874819"
DIM = 3584
MAX_TEXTS = 16
MAX_CHARS = 4096
MAX_TOTAL_CHARS = 16_384
SERVICE_TOKEN = os.environ.get("TARA_SERVICE_TOKEN", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
if len(SERVICE_TOKEN) < 24:
    raise RuntimeError("TARA_SERVICE_TOKEN must have at least 24 characters")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


snapshot = Path(snapshot_download(
    repo_id=MODEL_ID, revision=MODEL_REVISION, token=HF_TOKEN,
    cache_dir="/content/tara-model-cache", ignore_patterns=["*.png", "*.md", "assets/*"],
))
checks = {
    "model.safetensors.index.json": "59a64242446007de3eb6502c7d2ab52cc2721a0037fb920ae716b9218a4ad8af",
    "modeling_tara.py": "793b51bd4df2427e62c70737ccab4a3dbe2ca3173e5f762f6b2e8a5b1b78a9b0",
    "tarsier2/default_config.yaml": "d1708c94328c66c518db434ae88939cb071d9f03e9cabb47eac27ea7fae35443",
}
for name, expected in checks.items():
    if sha256_file(snapshot / name) != expected:
        raise RuntimeError(f"Checkpoint file hash mismatch: {name}")
index = json.loads((snapshot / "model.safetensors.index.json").read_text())
if int(index.get("metadata", {}).get("total_size", -1)) != 16_582_751_232:
    raise RuntimeError("Checkpoint tensor size differs from the video artifact")
for name in set(index["weight_map"].values()):
    if not (snapshot / name).is_file():
        raise RuntimeError(f"Missing weight shard: {name}")

sys.path.insert(0, str(snapshot))
try:
    import shared.utils  # noqa: F401
except Exception:
    # Same minimal fallback used by the video-embedding notebook. The official
    # TARA loader only needs YAML loading and its repository root here.
    import types
    import yaml

    shared = sys.modules.get("shared") or types.ModuleType("shared")
    utils = types.ModuleType("shared.utils")
    io = types.ModuleType("shared.utils.io")
    log = types.ModuleType("shared.utils.log")
    io.load_yml = lambda path: yaml.safe_load(Path(path).read_text())
    log.repo_path = str(snapshot)
    utils.io = io
    utils.log = log
    shared.utils = utils
    sys.modules.update({
        "shared": shared, "shared.utils": utils,
        "shared.utils.io": io, "shared.utils.log": log,
    })

from modeling_tara import TARA  # noqa: E402

MODEL = TARA.from_pretrained(
    str(snapshot), device_map={"": 0},
    attn_implementation="flash_attention_2", low_cpu_mem_usage=True,
)
MODEL.model.requires_grad_(False)
MODEL.model.eval()
PARAMETER = next(MODEL.model.parameters())
if PARAMETER.device.type != "cuda" or PARAMETER.dtype != torch.bfloat16:
    raise RuntimeError("TARA is not resident on CUDA in bfloat16")
if MODEL.processor.tokenizer.padding_side != "left":
    raise RuntimeError("TARA text pooling requires left padding")
if int(MODEL.model.config.text_config.hidden_size) != DIM:
    raise RuntimeError("TARA hidden size differs from video embeddings")
if MODEL.video_eol_prompt != "<video>\nSummary above video in one word:":
    raise RuntimeError("Unexpected TARA video prompt")

LOCK = threading.Lock()
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_CHARS)]


class EncodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[Text] = Field(min_length=1, max_length=MAX_TEXTS)

    @model_validator(mode="after")
    def total_chars(self):
        if sum(map(len, self.texts)) > MAX_TOTAL_CHARS:
            raise ValueError("Total text length exceeds limit")
        return self


def require_bearer(authorization: str | None) -> None:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, SERVICE_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


@torch.inference_mode()
def encode_text(texts: list[str]) -> np.ndarray:
    with LOCK:
        raw = MODEL.encode_text(texts)
        vectors = F.normalize(raw.float(), p=2, dim=-1).cpu().numpy().astype(np.float32)
    if vectors.shape != (len(texts), DIM) or not np.isfinite(vectors).all():
        raise RuntimeError("TARA returned wrong shape or nonfinite vectors")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5, rtol=0):
        raise RuntimeError("TARA vectors are not unit norm")
    return vectors


APP = FastAPI(title="AIC2026 TARA Text Encoder", version="1.0.0")


@APP.get("/health")
def health():
    return {
        "ok": True, "model": MODEL_ID, "revision": MODEL_REVISION,
        "semantic_fingerprint": SEMANTIC_FINGERPRINT,
        "dim": DIM, "l2_normalized": True,
        "device": str(PARAMETER.device), "dtype": str(PARAMETER.dtype),
        "gpu": torch.cuda.get_device_name(0),
        "max_texts": MAX_TEXTS, "max_chars": MAX_CHARS,
    }


@APP.post("/encode-text")
def api_encode_text(req: EncodeRequest, authorization: str | None = Header(default=None)):
    require_bearer(authorization)
    started = time.perf_counter()
    try:
        vectors = encode_text(req.texts)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="GPU out of memory") from exc
    return {
        **health(), "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "vectors": vectors.tolist(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(APP, host="0.0.0.0", port=int(os.environ.get("TARA_PORT", "8001")))
