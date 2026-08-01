"""Runtime configuration read from environment variables.

Secrets (Elastic / Milvus / NVIDIA / PE token) live only in the backend process
and are never serialized to API responses. See `.env.example` for the full list.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Existing env vars take precedence,
    so `VAR=… uvicorn …` still overrides the file."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_file_secret(path: str | None) -> str | None:
    """Read a secret from a local file (dev convenience). Returns None if absent."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    value = p.read_text(encoding="utf-8").strip()
    return value or None


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


@dataclass
class Settings:
    """Backend settings. `mock_mode` lets the UI be developed without live services."""

    elastic_endpoint: str | None = None
    elastic_api_key: str | None = None
    milvus_endpoint: str | None = None
    milvus_token: str | None = None
    pe_encoder_url: str | None = None
    pe_encoder_token: str | None = None
    # GLAP audio↔text encoder (/encode-audio-text). Usually the SAME Kaggle server
    # as PE; if unset, falls back to pe_encoder_url.
    glap_encoder_url: str | None = None
    media_base_url: str = "https://media.example.com"
    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    # Routing parser model. qwen3-next (3B-active MoE) is the fastest model that
    # still returns the full valid JSON schema (~28s vs ~45s for the 550B).
    nvidia_model: str = "qwen/qwen3-next-80b-a3b-instruct"
    # A fast NIM model for lightweight calls (query expansion). Qwen3-next is a
    # 3B-active MoE: ~2s, strong multilingual, and reliably outputs ENGLISH for
    # Vietnamese input (llama-3.1-8b echoed Vietnamese; the 550B is ~45s/call).
    nvidia_fast_model: str = "qwen/qwen3-next-80b-a3b-instruct"
    # NVILA-8B visual QA server (the companion Colab notebook exposes /qa/analyze).
    # The URL is a volatile Cloudflare quick-tunnel URL unless a named tunnel is used.
    nvila_base_url: str | None = None
    nvila_token: str | None = None
    nvila_timeout_seconds: float = 240.0
    nvila_max_candidates: int = 12
    # Gemini text model with the built-in Google Search grounding tool. This
    # resolves external facts after NVILA has extracted visual/ASR clues.
    gemini_api_key: str | None = None
    gemini_grounding_model: str = "gemini-3.6-flash"
    gemini_grounding_fallback_models: list[str] = field(
        default_factory=lambda: ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
    )
    gemini_grounding_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_grounding_enabled: bool = True
    gemini_grounding_timeout_seconds: float = 60.0
    gemini_grounding_auto_threshold: float = 0.55
    # Slim-schema parse: ~2x faster but A/B showed it occasionally misroutes
    # query_type (TKIS→TRAKE) on ambiguous queries — off by default (accuracy).
    slim_parse: bool = False
    dres_base_url: str | None = None
    dres_token: str | None = None

    # Index / collection names (override only if you re-uploaded with a new prefix).
    idx_keyframe_map: str = "aic26_keyframe_map_v1"
    idx_ocr: str = "aic26_ocr_keyframes_v1"
    idx_speech: str = "aic26_speech_segments_v1"
    idx_audio: str = "aic26_audio_windows_v1"
    milvus_image_collection: str = "aic26_image_peg14_v1"
    milvus_audio_collection: str = "aic26_audio_glap_v1"

    # When true, adapters return deterministic fixtures instead of calling services.
    mock_mode: bool = False

    # VI→EN translation fallback for the PE visual query when the LLM parser is off.
    translate_to_en: bool = True

    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @property
    def has_elastic(self) -> bool:
        return bool(self.elastic_endpoint and self.elastic_api_key)

    @property
    def has_milvus(self) -> bool:
        return bool(self.milvus_endpoint and self.milvus_token)

    @property
    def has_pe_encoder(self) -> bool:
        return bool(self.pe_encoder_url)

    @property
    def glap_url(self) -> str | None:
        """GLAP encoder base URL — its own, or the shared PE server."""
        return self.glap_encoder_url or self.pe_encoder_url

    @property
    def has_glap(self) -> bool:
        return bool(self.glap_url and self.has_milvus)

    @property
    def has_llm(self) -> bool:
        return bool(self.nvidia_api_key)

    @property
    def has_nvila(self) -> bool:
        # The companion worker always requires Bearer auth. Treat a URL without
        # its matching token as incomplete rather than attempting live calls.
        return bool(self.nvila_base_url and self.nvila_token)

    @property
    def has_google_grounding(self) -> bool:
        return bool(self.gemini_grounding_enabled and self.gemini_api_key)

    @property
    def has_dres(self) -> bool:
        return bool(self.dres_base_url and self.dres_token)


@lru_cache
def get_settings() -> Settings:
    backend_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    _load_dotenv(backend_root / ".env")

    def file_default(name: str) -> str | None:
        return _read_file_secret(str(repo_root / name))

    mock_env = (_env("AIC26_MOCK_MODE", "false") or "false").lower()
    mock_mode = mock_env in {"1", "true", "yes", "on"}

    translate_env = (_env("TRANSLATE_TO_EN", "true") or "true").lower()
    translate_to_en = translate_env in {"1", "true", "yes", "on"}

    slim_env = (_env("SLIM_PARSE", "false") or "false").lower()
    slim_parse = slim_env in {"1", "true", "yes", "on"}

    try:
        nvila_timeout_seconds = float(_env("NVILA_TIMEOUT_SECONDS", "240") or "240")
    except ValueError:
        nvila_timeout_seconds = 240.0
    try:
        nvila_max_candidates = int(_env("NVILA_MAX_CANDIDATES", "12") or "12")
    except ValueError:
        nvila_max_candidates = 12
    grounding_env = (_env("GEMINI_GROUNDING_ENABLED", "true") or "true").lower()
    gemini_grounding_enabled = grounding_env in {"1", "true", "yes", "on"}
    try:
        gemini_grounding_timeout_seconds = float(_env("GEMINI_GROUNDING_TIMEOUT_SECONDS", "60") or "60")
    except ValueError:
        gemini_grounding_timeout_seconds = 60.0
    try:
        gemini_grounding_auto_threshold = float(_env("GEMINI_GROUNDING_AUTO_THRESHOLD", "0.55") or "0.55")
    except ValueError:
        gemini_grounding_auto_threshold = 0.55
    grounding_fallbacks_raw = _env(
        "GEMINI_GROUNDING_FALLBACK_MODELS",
        "gemini-3.5-flash,gemini-3.1-flash-lite",
    ) or ""
    gemini_grounding_fallback_models = [
        model.strip() for model in grounding_fallbacks_raw.split(",") if model.strip()
    ]

    cors = _env("CORS_ORIGINS")
    cors_origins = [o.strip() for o in cors.split(",")] if cors else ["*"]

    return Settings(
        elastic_endpoint=_env("ELASTIC_ENDPOINT") or file_default("elastic_endpoint.txt"),
        elastic_api_key=_env("ELASTIC_API_KEY") or file_default("elastic_apikey.txt"),
        milvus_endpoint=_env("MILVUS_ENDPOINT") or file_default("milvus_endpoint.txt"),
        milvus_token=_env("MILVUS_TOKEN") or file_default("milvus_token.txt"),
        pe_encoder_url=_env("PE_ENCODER_URL"),
        pe_encoder_token=_env("PE_ENCODER_TOKEN"),
        glap_encoder_url=_env("GLAP_ENCODER_URL"),
        media_base_url=(_env("MEDIA_BASE_URL") or "https://media.example.com").rstrip("/"),
        nvidia_api_key=_env("NVIDIA_API_KEY") or file_default("nvidia_api_key.txt"),
        nvidia_base_url=_env("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1",
        nvidia_model=_env("NVIDIA_MODEL") or "qwen/qwen3-next-80b-a3b-instruct",
        nvidia_fast_model=_env("NVIDIA_FAST_MODEL") or "qwen/qwen3-next-80b-a3b-instruct",
        nvila_base_url=(_env("NVILA_BASE_URL") or "").rstrip("/") or None,
        nvila_token=_env("NVILA_TOKEN") or file_default("nvila_token.txt"),
        nvila_timeout_seconds=max(30.0, nvila_timeout_seconds),
        nvila_max_candidates=min(24, max(1, nvila_max_candidates)),
        gemini_api_key=_env("GEMINI_API_KEY") or file_default("gemini_api_key.txt"),
        gemini_grounding_model=_env("GEMINI_GROUNDING_MODEL") or "gemini-3.6-flash",
        gemini_grounding_fallback_models=gemini_grounding_fallback_models,
        gemini_grounding_base_url=(_env("GEMINI_GROUNDING_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/"),
        gemini_grounding_enabled=gemini_grounding_enabled,
        gemini_grounding_timeout_seconds=max(10.0, gemini_grounding_timeout_seconds),
        gemini_grounding_auto_threshold=min(1.0, max(0.0, gemini_grounding_auto_threshold)),
        dres_base_url=_env("DRES_BASE_URL"),
        dres_token=_env("DRES_TOKEN"),
        idx_keyframe_map=_env("IDX_KEYFRAME_MAP") or "aic26_keyframe_map_v1",
        idx_ocr=_env("IDX_OCR") or "aic26_ocr_keyframes_v1",
        idx_speech=_env("IDX_SPEECH") or "aic26_speech_segments_v1",
        idx_audio=_env("IDX_AUDIO") or "aic26_audio_windows_v1",
        milvus_image_collection=_env("MILVUS_IMAGE_COLLECTION") or "aic26_image_peg14_v1",
        milvus_audio_collection=_env("MILVUS_AUDIO_COLLECTION") or "aic26_audio_glap_v1",
        mock_mode=mock_mode,
        translate_to_en=translate_to_en,
        slim_parse=slim_parse,
        cors_origins=cors_origins,
    )
