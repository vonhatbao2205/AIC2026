"""Declared shape of the operator-editable configuration.

The packaged app has no shell: an operator imports a `.env` file through the UI
instead of exporting variables. That screen needs to say which keys exist, which
are mandatory before search works at all, and which must never be echoed back —
so the key list is data here rather than being implied by `get_settings()`.

`config.Settings` stays the single source of truth for *behaviour*; this module
only describes those same variables for humans.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigKey:
    key: str
    label: str
    #: Secret values are never sent to the browser — only a masked hint.
    secret: bool = False
    #: Retrieval cannot run at all without these (mock mode excepted).
    required: bool = False


@dataclass(frozen=True)
class ConfigGroup:
    name: str
    summary: str
    keys: tuple[ConfigKey, ...]


GROUPS: tuple[ConfigGroup, ...] = (
    ConfigGroup(
        "Elastic Cloud",
        "OCR, speech, audio and timeline lookups.",
        (
            ConfigKey("ELASTIC_ENDPOINT", "Cluster endpoint URL", required=True),
            ConfigKey("ELASTIC_API_KEY", "API key", secret=True, required=True),
        ),
    ),
    ConfigGroup(
        "Milvus / Zilliz",
        "Hai database image vector độc lập: BTC đầy đủ và InfoShot++ PE-only.",
        (
            ConfigKey("MILVUS_ENDPOINT_1", "BTC cluster endpoint", required=True),
            ConfigKey("MILVUS_TOKEN_1", "BTC token", secret=True, required=True),
            ConfigKey("MILVUS_ENDPOINT_2", "InfoShot++ cluster endpoint"),
            ConfigKey("MILVUS_TOKEN_2", "InfoShot++ token", secret=True),
        ),
    ),
    ConfigGroup(
        "Encoders",
        "PE-Core-G14 text→image, and GLAP for audio↔text.",
        (
            ConfigKey("PE_ENCODER_URL", "PE encoder base URL", required=True),
            ConfigKey("PE_ENCODER_TOKEN", "PE encoder bearer token", secret=True),
            ConfigKey("GLAP_ENCODER_URL", "GLAP base URL (blank = reuse PE)"),
        ),
    ),
    ConfigGroup(
        "Media",
        "Where keyframes and videos are served from.",
        (
            ConfigKey("MEDIA_BASE_URL", "Cloudflare R2 base (BTC keyframes + all videos)", required=True),
            ConfigKey("KEYFRAME_MEDIA_BASE_URL_2", "Hugging Face base for InfoShot++ keyframes"),
        ),
    ),
    ConfigGroup(
        "DRES",
        "The evaluation server answers are submitted to.",
        (
            ConfigKey("DRES_BASE_URL", "Server URL"),
            ConfigKey("DRES_USERNAME", "Participant username"),
            ConfigKey("DRES_PASSWORD", "Participant password", secret=True),
            ConfigKey("DRES_SESSION", "Pre-issued session id (optional)", secret=True),
            ConfigKey("DRES_EVALUATION_ID", "Pin one evaluation run (optional)"),
            ConfigKey("DRES_SEGMENT_PAD_MS", "± ms padded around the picked instant"),
            ConfigKey("DRES_TIMEOUT_SECONDS", "Request timeout"),
        ),
    ),
    ConfigGroup(
        "Query parser (optional)",
        "NVIDIA NIM. Without a key the deterministic heuristics are used.",
        (
            ConfigKey("NVIDIA_API_KEY", "NIM API key", secret=True),
            ConfigKey("NVIDIA_BASE_URL", "NIM base URL"),
            ConfigKey("NVIDIA_MODEL", "Routing parser model"),
            ConfigKey("NVIDIA_FAST_MODEL", "Fast model for query expansion"),
            ConfigKey("SLIM_PARSE", "Slim schema: faster, can misroute (true/false)"),
        ),
    ),
    ConfigGroup(
        "QA copilot (optional)",
        "NVILA-8B visual QA worker and DeepSeek web grounding.",
        (
            ConfigKey("NVILA_BASE_URL", "Colab worker URL"),
            ConfigKey("NVILA_TOKEN", "Worker bearer token", secret=True),
            ConfigKey("NVILA_TIMEOUT_SECONDS", "Request timeout"),
            ConfigKey("NVILA_MAX_CANDIDATES", "Frames sent per analysis"),
            ConfigKey("DEEPSEEK_API_KEY", "DeepSeek API key", secret=True),
            ConfigKey("DEEPSEEK_GROUNDING_MODEL", "Grounding model"),
            ConfigKey("DEEPSEEK_GROUNDING_BASE_URL", "DeepSeek base URL"),
            ConfigKey("DEEPSEEK_GROUNDING_ENABLED", "Enable web grounding (true/false)"),
            ConfigKey("DEEPSEEK_GROUNDING_TIMEOUT_SECONDS", "Request timeout"),
            ConfigKey("DEEPSEEK_GROUNDING_MAX_OUTPUT_TOKENS", "Output budget"),
            ConfigKey("DEEPSEEK_GROUNDING_REASONING_EFFORT", "low | high | max"),
            ConfigKey("DEEPSEEK_GROUNDING_AUTO_THRESHOLD", "Auto-search below this confidence"),
        ),
    ),
    ConfigGroup(
        "Index names",
        "Override only if the corpus was re-uploaded under a new prefix.",
        (
            ConfigKey("IDX_KEYFRAME_MAP_1", "BTC keyframe map index"),
            ConfigKey("IDX_KEYFRAME_MAP_2", "InfoShot++ keyframe map index"),
            ConfigKey("IDX_OCR", "BTC OCR index (alias of IDX_OCR_1)"),
            ConfigKey("IDX_SPEECH", "BTC speech index (alias of IDX_SPEECH_1)"),
            ConfigKey("IDX_AUDIO", "BTC audio index (alias of IDX_AUDIO_1)"),
            ConfigKey("IDX_OCR_2", "InfoShot++ OCR index (v2)"),
            ConfigKey("IDX_SPEECH_2", "InfoShot++ speech index (v2)"),
            ConfigKey("IDX_AUDIO_2", "InfoShot++ audio index (v2)"),
            ConfigKey("OCR_MISSING_CATEGORIES_2", "Category chưa có OCR cho InfoShot++ (vd L26)"),
            ConfigKey("IDX_OBJECTS", "Object detection index"),
            ConfigKey("MILVUS_IMAGE_COLLECTION_1", "BTC image vector collection"),
            ConfigKey("MILVUS_IMAGE_COLLECTION_2", "InfoShot++ image vector collection"),
            ConfigKey("MILVUS_AUDIO_COLLECTION", "Audio vector collection"),
        ),
    ),
    ConfigGroup(
        "Runtime",
        "Local behaviour of this instance.",
        (
            ConfigKey("AIC26_MOCK_MODE", "Run on fixtures, no live services (true/false)"),
            ConfigKey("TRANSLATE_TO_EN", "VI→EN translation fallback (true/false)"),
            ConfigKey("WHISPER_MODEL", "Voice input model (tiny|base|small|medium)"),
            ConfigKey("CORS_ORIGINS", "Allowed browser origins"),
        ),
    ),
)

ALL_KEYS: tuple[ConfigKey, ...] = tuple(k for g in GROUPS for k in g.keys)
KEYS_BY_NAME: dict[str, ConfigKey] = {k.key: k for k in ALL_KEYS}
SECRET_KEYS: frozenset[str] = frozenset(k.key for k in ALL_KEYS if k.secret)
REQUIRED_KEYS: tuple[str, ...] = tuple(k.key for k in ALL_KEYS if k.required)


def mask(key: str, value: str) -> str:
    """What the browser is allowed to see for `key`.

    Secrets collapse to a fixed-width dot run plus the last four characters —
    enough to tell two keys apart when checking which one got imported, and not
    enough to use. The run is fixed so the length does not leak either.
    """
    if key not in SECRET_KEYS:
        return value
    if not value:
        return ""
    tail = value[-4:] if len(value) >= 12 else ""
    return f"••••••••{tail}"
