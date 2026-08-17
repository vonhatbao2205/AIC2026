"""Runtime configuration read from environment variables.

Secrets (Elastic / Milvus / NVIDIA / PE token) live only in the backend process
and are never serialized to API responses. See `.env.example` for the full list.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

from . import paths


# What this process took from a .env file, so a re-read can drop it first.
# Without that, importing a new config would be a no-op: the previous values are
# already in os.environ, and os.environ always wins over the file. The value is
# kept alongside the key so a variable someone else has since changed is left
# alone — this loader only ever takes back what is still its own.
_DOTENV_APPLIED: dict[str, str] = {}


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse `.env` text into an ordered mapping. Comments and blanks are dropped."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Environment variables set outside the
    file take precedence, so `VAR=… uvicorn …` and Docker `environment:` still
    override it; values this loader itself applied earlier do not."""
    for key, applied in _DOTENV_APPLIED.items():
        if os.environ.get(key) == applied:
            os.environ.pop(key, None)
    _DOTENV_APPLIED.clear()
    if not path.exists():
        return
    for key, value in parse_dotenv(path.read_text(encoding="utf-8")).items():
        if key not in os.environ:
            os.environ[key] = value
            _DOTENV_APPLIED[key] = value


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
    # Two independent image-retrieval datasets.  The unnumbered fields above are
    # the active aliases consumed by adapters; ``for_retrieval_database`` fills
    # them per request while keeping old Settings(...) call sites compatible.
    milvus_endpoint_1: str | None = None
    milvus_token_1: str | None = None
    milvus_endpoint_2: str | None = None
    milvus_token_2: str | None = None
    retrieval_database: str = "btc"
    pe_encoder_url: str | None = None
    pe_encoder_token: str | None = None
    # GLAP audio↔text encoder (/encode-audio-text). Usually the SAME Kaggle server
    # as PE; if unset, falls back to pe_encoder_url.
    glap_encoder_url: str | None = None
    media_base_url: str = "https://media.example.com"
    # Active keyframe origin. Videos always stay on media_base_url (Cloudflare R2).
    keyframe_media_base_url: str = "https://media.example.com"
    keyframe_media_base_url_2: str = (
        "https://huggingface.co/buckets/Baonenha1/aic26-media/resolve"
    )
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
    # DeepSeek model with the server-side web_search tool, used to resolve
    # external facts after NVILA has extracted visual/ASR clues. The tool is only
    # available on the Responses API and only for deepseek-v4-flash.
    deepseek_api_key: str | None = None
    deepseek_grounding_model: str = "deepseek-v4-flash"
    deepseek_grounding_base_url: str = "https://api.deepseek.com"
    deepseek_grounding_enabled: bool = True
    deepseek_grounding_timeout_seconds: float = 90.0
    deepseek_grounding_auto_threshold: float = 0.55
    # Thinking is enabled by default and draws from the same budget as the answer.
    deepseek_grounding_max_output_tokens: int = 8000
    deepseek_grounding_reasoning_effort: str = "high"
    # Slim-schema parse: ~2x faster but A/B showed it occasionally misroutes
    # query_type (TKIS→TRAKE) on ambiguous queries — off by default (accuracy).
    slim_parse: bool = False
    # --- DRES (official BTC evaluation server, Client API v2) ---
    # Auth is a session: POST /api/v2/login returns a sessionId that every other
    # call carries as ?session=…. Credentials stay backend-side.
    dres_base_url: str | None = None
    dres_username: str | None = None
    dres_password: str | None = None
    # Pre-issued session id (skips login; cannot be renewed when it expires).
    dres_session: str | None = None
    # Pin one evaluation run; empty means "auto-pick by the current task type".
    dres_evaluation_id: str | None = None
    # KIS/TRAKE answers are temporal ranges. The operator submits one instant, so
    # the range is padded symmetrically: a zero-length range is rejected by strict
    # overlap checks, and ±0.5 s stays far inside a typical KIS target segment.
    dres_segment_pad_ms: int = 500
    dres_timeout_seconds: float = 20.0

    # Index / collection names (override only if you re-uploaded with a new prefix).
    idx_keyframe_map: str = "aic26_keyframe_map_v1"
    idx_keyframe_map_1: str = "aic26_keyframe_map_v1"
    idx_keyframe_map_2: str = "aic26_keyframe_map_infoshotpp_v1"
    idx_ocr: str = "aic26_ocr_keyframes_v1"
    idx_speech: str = "aic26_speech_segments_v1"
    idx_audio: str = "aic26_audio_windows_v1"
    # OD frame documents (`od-frame-v5`) behind the V-KIS canvas channel.
    idx_objects: str = "aic26_od_frames_v1"
    milvus_image_collection: str = "aic26_image_peg14_v1"
    milvus_image_collection_1: str = "aic26_image_peg14_v1"
    milvus_image_collection_2: str = "aic26_image_peg14_infoshotpp_v1"
    milvus_audio_collection: str = "aic26_audio_glap_v1"

    # When true, adapters return deterministic fixtures instead of calling services.
    mock_mode: bool = False

    # VI→EN translation fallback for the PE visual query when the LLM parser is off.
    translate_to_en: bool = True

    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    def __post_init__(self) -> None:
        # Direct Settings(media_base_url=...) construction is common in tests and
        # small tools. Preserve the historical one-origin behavior unless the
        # keyframe origin was explicitly set.
        if self.keyframe_media_base_url == "https://media.example.com":
            self.keyframe_media_base_url = self.media_base_url.rstrip("/")

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
        return bool(self.retrieval_database == "btc" and self.glap_url and self.has_milvus)

    @property
    def is_infoshotpp(self) -> bool:
        return self.retrieval_database == "infoshotpp"

    def for_retrieval_database(self, database: str) -> "Settings":
        """Return request-scoped adapter/media settings for one dataset."""
        if database not in {"btc", "infoshotpp"}:
            raise ValueError(f"Unknown retrieval database: {database}")
        if database == "btc":
            return replace(
                self,
                retrieval_database="btc",
                milvus_endpoint=self.milvus_endpoint_1 or self.milvus_endpoint,
                milvus_token=self.milvus_token_1 or self.milvus_token,
                idx_keyframe_map=self.idx_keyframe_map_1,
                milvus_image_collection=self.milvus_image_collection_1,
                keyframe_media_base_url=self.media_base_url,
            )
        return replace(
            self,
            retrieval_database="infoshotpp",
            milvus_endpoint=self.milvus_endpoint_2,
            milvus_token=self.milvus_token_2,
            idx_keyframe_map=self.idx_keyframe_map_2,
            milvus_image_collection=self.milvus_image_collection_2,
            keyframe_media_base_url=self.keyframe_media_base_url_2,
        )

    @property
    def has_llm(self) -> bool:
        return bool(self.nvidia_api_key)

    @property
    def has_nvila(self) -> bool:
        # The companion worker always requires Bearer auth. Treat a URL without
        # its matching token as incomplete rather than attempting live calls.
        return bool(self.nvila_base_url and self.nvila_token)

    @property
    def has_web_grounding(self) -> bool:
        return bool(self.deepseek_grounding_enabled and self.deepseek_api_key)

    @property
    def has_dres(self) -> bool:
        """Enough to reach DRES: a host plus either credentials or a live session.

        Mock mode never talks to the real evaluation server — fixtures are not
        answers, and the whole test suite runs with mock mode on.
        """
        if self.mock_mode:
            return False
        return bool(
            self.dres_base_url and (self.dres_session or (self.dres_username and self.dres_password))
        )


@lru_cache
def get_settings() -> Settings:
    # A checkout reads backend/.env; the packaged app points AIC26_CONFIG_DIR at
    # a mounted volume so an imported config survives replacing the image.
    _load_dotenv(paths.env_file())

    # Dev-convenience secret files. They were moved into API_KEY/ when the repo
    # layout was flattened; the repo root stays as a fallback for older checkouts
    # and for anyone who still keeps the loose files there.
    secret_dirs = paths.secret_dirs()

    def file_default(name: str) -> str | None:
        for directory in secret_dirs:
            value = _read_file_secret(str(directory / name))
            if value:
                return value
        return None

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
    grounding_env = (_env("DEEPSEEK_GROUNDING_ENABLED", "true") or "true").lower()
    deepseek_grounding_enabled = grounding_env in {"1", "true", "yes", "on"}
    try:
        deepseek_grounding_timeout_seconds = float(_env("DEEPSEEK_GROUNDING_TIMEOUT_SECONDS", "90") or "90")
    except ValueError:
        deepseek_grounding_timeout_seconds = 90.0
    try:
        deepseek_grounding_auto_threshold = float(_env("DEEPSEEK_GROUNDING_AUTO_THRESHOLD", "0.55") or "0.55")
    except ValueError:
        deepseek_grounding_auto_threshold = 0.55
    try:
        deepseek_grounding_max_output_tokens = int(
            _env("DEEPSEEK_GROUNDING_MAX_OUTPUT_TOKENS", "8000") or "8000"
        )
    except ValueError:
        deepseek_grounding_max_output_tokens = 8000
    reasoning_effort = (_env("DEEPSEEK_GROUNDING_REASONING_EFFORT", "high") or "high").lower()
    if reasoning_effort not in {"low", "high", "max"}:
        reasoning_effort = "high"

    try:
        dres_segment_pad_ms = int(_env("DRES_SEGMENT_PAD_MS", "500") or "500")
    except ValueError:
        dres_segment_pad_ms = 500
    try:
        dres_timeout_seconds = float(_env("DRES_TIMEOUT_SECONDS", "20") or "20")
    except ValueError:
        dres_timeout_seconds = 20.0

    cors = _env("CORS_ORIGINS")
    cors_origins = [o.strip() for o in cors.split(",")] if cors else ["*"]

    legacy_milvus_endpoint = _env("MILVUS_ENDPOINT") or file_default("milvus_endpoint.txt")
    legacy_milvus_token = _env("MILVUS_TOKEN") or file_default("milvus_token.txt")
    milvus_endpoint_1 = _env("MILVUS_ENDPOINT_1") or legacy_milvus_endpoint
    milvus_token_1 = _env("MILVUS_TOKEN_1") or legacy_milvus_token
    media_base_url = (_env("MEDIA_BASE_URL") or "https://media.example.com").rstrip("/")
    idx_keyframe_map_1 = _env("IDX_KEYFRAME_MAP_1") or _env("IDX_KEYFRAME_MAP") or "aic26_keyframe_map_v1"
    image_collection_1 = (
        _env("MILVUS_IMAGE_COLLECTION_1")
        or _env("MILVUS_IMAGE_COLLECTION")
        or "aic26_image_peg14_v1"
    )

    return Settings(
        elastic_endpoint=_env("ELASTIC_ENDPOINT") or file_default("elastic_endpoint.txt"),
        elastic_api_key=_env("ELASTIC_API_KEY") or file_default("elastic_apikey.txt"),
        milvus_endpoint=milvus_endpoint_1,
        milvus_token=milvus_token_1,
        milvus_endpoint_1=milvus_endpoint_1,
        milvus_token_1=milvus_token_1,
        milvus_endpoint_2=_env("MILVUS_ENDPOINT_2"),
        milvus_token_2=_env("MILVUS_TOKEN_2"),
        pe_encoder_url=_env("PE_ENCODER_URL"),
        pe_encoder_token=_env("PE_ENCODER_TOKEN"),
        glap_encoder_url=_env("GLAP_ENCODER_URL"),
        media_base_url=media_base_url,
        keyframe_media_base_url=media_base_url,
        keyframe_media_base_url_2=(
            _env("KEYFRAME_MEDIA_BASE_URL_2")
            or "https://huggingface.co/buckets/Baonenha1/aic26-media/resolve"
        ).rstrip("/"),
        nvidia_api_key=_env("NVIDIA_API_KEY") or file_default("nvidia_api_key.txt"),
        nvidia_base_url=_env("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1",
        nvidia_model=_env("NVIDIA_MODEL") or "qwen/qwen3-next-80b-a3b-instruct",
        nvidia_fast_model=_env("NVIDIA_FAST_MODEL") or "qwen/qwen3-next-80b-a3b-instruct",
        nvila_base_url=(_env("NVILA_BASE_URL") or "").rstrip("/") or None,
        nvila_token=_env("NVILA_TOKEN") or file_default("nvila_token.txt"),
        nvila_timeout_seconds=max(30.0, nvila_timeout_seconds),
        nvila_max_candidates=min(24, max(1, nvila_max_candidates)),
        deepseek_api_key=_env("DEEPSEEK_API_KEY") or file_default("deepseek_apikey.txt"),
        deepseek_grounding_model=_env("DEEPSEEK_GROUNDING_MODEL") or "deepseek-v4-flash",
        deepseek_grounding_base_url=(_env("DEEPSEEK_GROUNDING_BASE_URL") or "https://api.deepseek.com").rstrip("/"),
        deepseek_grounding_enabled=deepseek_grounding_enabled,
        deepseek_grounding_timeout_seconds=max(10.0, deepseek_grounding_timeout_seconds),
        deepseek_grounding_auto_threshold=min(1.0, max(0.0, deepseek_grounding_auto_threshold)),
        deepseek_grounding_max_output_tokens=min(64000, max(1000, deepseek_grounding_max_output_tokens)),
        deepseek_grounding_reasoning_effort=reasoning_effort,
        # The public BTC host; inside SELab (I87) override with http://10.0.1.21:20740.
        dres_base_url=(_env("DRES_BASE_URL") or "http://if-wan4.selab.edu.vn:20740").rstrip("/"),
        dres_username=_env("DRES_USERNAME"),
        dres_password=_env("DRES_PASSWORD"),
        dres_session=_env("DRES_SESSION"),
        dres_evaluation_id=_env("DRES_EVALUATION_ID"),
        dres_segment_pad_ms=max(0, dres_segment_pad_ms),
        dres_timeout_seconds=max(5.0, dres_timeout_seconds),
        idx_keyframe_map=idx_keyframe_map_1,
        idx_keyframe_map_1=idx_keyframe_map_1,
        idx_keyframe_map_2=(
            _env("IDX_KEYFRAME_MAP_2") or "aic26_keyframe_map_infoshotpp_v1"
        ),
        idx_ocr=_env("IDX_OCR") or "aic26_ocr_keyframes_v1",
        idx_speech=_env("IDX_SPEECH") or "aic26_speech_segments_v1",
        idx_audio=_env("IDX_AUDIO") or "aic26_audio_windows_v1",
        idx_objects=_env("IDX_OBJECTS") or "aic26_od_frames_v1",
        milvus_image_collection=image_collection_1,
        milvus_image_collection_1=image_collection_1,
        milvus_image_collection_2=(
            _env("MILVUS_IMAGE_COLLECTION_2") or "aic26_image_peg14_infoshotpp_v1"
        ),
        milvus_audio_collection=_env("MILVUS_AUDIO_COLLECTION") or "aic26_audio_glap_v1",
        mock_mode=mock_mode,
        translate_to_en=translate_to_en,
        slim_parse=slim_parse,
        cors_origins=cors_origins,
    )
