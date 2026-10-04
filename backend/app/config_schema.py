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
        "Two independent databases; InfoShot++ stores PE and Qwen in separate collections.",
        (
            ConfigKey("MILVUS_ENDPOINT_1", "BTC cluster endpoint", required=True),
            ConfigKey("MILVUS_TOKEN_1", "BTC token", secret=True, required=True),
            ConfigKey("MILVUS_ENDPOINT_2", "InfoShot++ cluster endpoint"),
            ConfigKey("MILVUS_TOKEN_2", "InfoShot++ token", secret=True),
        ),
    ),
    ConfigGroup(
        "Encoders",
        "PE-Core-G14, Qwen3-VL-Embedding-8B, TARA clip retrieval, and GLAP audio↔text.",
        (
            ConfigKey("PE_ENCODER_URL", "PE encoder base URL", required=True),
            ConfigKey("PE_ENCODER_TOKEN", "PE encoder bearer token", secret=True),
            ConfigKey("QWEN3_VL_ENCODER_URL", "Qwen3-VL embedding encoder base URL"),
            ConfigKey(
                "QWEN3_VL_ENCODER_TOKEN",
                "Qwen3-VL embedding bearer token",
                secret=True,
            ),
            ConfigKey("QWEN3_VL_ENCODER_TIMEOUT_SECONDS", "Qwen text encode timeout"),
            ConfigKey("TARA_ENABLED", "Enable TARA clip retrieval (true/false)"),
            ConfigKey("TARA_ENCODER_URL", "TARA text encoder base URL"),
            ConfigKey("TARA_ENCODER_TOKEN", "TARA encoder bearer token (optional; the Colab worker has none)", secret=True),
            ConfigKey("TARA_ENCODER_TIMEOUT_SECONDS", "TARA text encode timeout"),
            ConfigKey("GLAP_ENCODER_URL", "GLAP base URL (blank = reuse PE)"),
        ),
    ),
    ConfigGroup(
        "Media",
        "Where keyframes and videos are served from.",
        (
            ConfigKey("MEDIA_BASE_URL", "Cloudflare R2 base (BTC keyframes, default video origin)", required=True),
            ConfigKey("KEYFRAME_MEDIA_BASE_URL_2", "Cloudflare R2 base for InfoShot++ keyframes"),
            ConfigKey("KEYFRAME_MEDIA_FALLBACK_BASE_URL_2", "Hugging Face fallback for InfoShot++ keyframes"),
            ConfigKey("VIDEO_MEDIA_BASE_URL_1", "BTC video origin (blank = MEDIA_BASE_URL)"),
            ConfigKey("VIDEO_MEDIA_BASE_URL_2", "InfoShot++ video origin (HF bucket)"),
        ),
    ),
    ConfigGroup(
        "DRES",
        "The evaluation server answers are submitted to.",
        (
            ConfigKey("DRES_ENABLED", "Submit live to DRES (false = CSV pack only)"),
            ConfigKey("DRES_BASE_URL", "Server URL"),
            ConfigKey("DRES_USERNAME", "Participant username"),
            ConfigKey("DRES_PASSWORD", "Participant password", secret=True),
            ConfigKey("DRES_SESSION", "Pre-issued session id (optional)", secret=True),
            ConfigKey("DRES_EVALUATION_ID", "Pin one evaluation run (optional)"),
            ConfigKey("DRES_COLLECTION_NAME", "Media collection name (when DRES asks for it)"),
            ConfigKey("DRES_SEGMENT_PAD_MS", "± ms padded around the picked instant"),
            ConfigKey("DRES_TIMEOUT_SECONDS", "Request timeout"),
        ),
    ),
    ConfigGroup(
        "Query LLM (optional)",
        "Routing parser (LLM switch) and query expansion (Expand). DeepSeek by default; "
        "without a key the deterministic heuristics are used.",
        (
            ConfigKey("QUERY_LLM_API_KEY", "API key (defaults to DEEPSEEK_API_KEY)", secret=True),
            ConfigKey("QUERY_LLM_BASE_URL", "Base URL (https://api.deepseek.com)"),
            ConfigKey("QUERY_LLM_MODEL", "Model (deepseek-flash = V4.1 Flash)"),
            ConfigKey("QUERY_LLM_TIMEOUT_SECONDS", "Request timeout"),
        ),
    ),
    ConfigGroup(
        "Translation fallback (optional)",
        "NVIDIA NIM model used only when both Google translation endpoints fail.",
        (
            ConfigKey("NVIDIA_API_KEY", "NIM API key", secret=True),
            ConfigKey("NVIDIA_BASE_URL", "NIM base URL"),
            ConfigKey("NVIDIA_FAST_MODEL", "Translation model"),
        ),
    ),
    ConfigGroup(
        "QA copilot (optional)",
        "Visual passes on DeepSeek V4.1 Flash (default) or the NVILA-8B worker; DeepSeek web grounding.",
        (
            ConfigKey("QA_VISION_BACKEND", "deepseek | nvila"),
            ConfigKey("QA_VISION_MODEL", "Vision model (deepseek-flash)"),
            ConfigKey("QA_VISION_REASONING_EFFORT", "low | high | max"),
            ConfigKey("QA_VISION_IMAGE_DETAIL", "low | high | original | auto"),
            ConfigKey("QA_VISION_TIMEOUT_SECONDS", "Request timeout"),
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
        "Visual reranker (optional)",
        "Qwen3-VL-Reranker worker rescoring the merged PE+Qwen pool; ticked per search.",
        (
            ConfigKey("QWEN_RERANKER_URL", "Colab worker URL"),
            ConfigKey("QWEN_RERANKER_TOKEN", "Worker bearer token", secret=True),
            ConfigKey("QWEN_RERANKER_ENABLED", "Allow the rerank tick box (true/false)"),
            ConfigKey("QWEN_RERANKER_CANDIDATES", "Candidates retrieved before reranking"),
            ConfigKey("QWEN_RERANKER_TIMEOUT_SECONDS", "Request timeout"),
        ),
    ),
    ConfigGroup(
        "Agent search (optional)",
        "Codex CLI and Claude Code CLI search next to the main pipeline (AGENT button). "
        "The CLIs run on this machine under their own logins.",
        (
            ConfigKey("AGENT_ENABLED", "Allow agent search (true/false)"),
            ConfigKey("OPENROUTER_API_KEY", "OpenRouter key for Jev decisions", secret=True),
            ConfigKey("AGENT_LLM_JUDGE_MODEL", "Optional generative judge ablation (anthropic/claude-opus-5.5)"),
            ConfigKey("JEV_MODEL", "Decision model (typesafe/jev-1.13)"),
            ConfigKey("JEV_TIMEOUT_SECONDS", "Timeout per decision request (12)"),
            ConfigKey("AGENT_POLICY", "full | adaptive | rerank | retrieval | codex | parallel | parallel_verify | rule"),
            ConfigKey("AGENT_MAX_STEPS", "Controller action budget (6)"),
            ConfigKey("AGENT_MAX_TOOL_CALLS", "Tool budget per query (40)"),
            ConfigKey("AGENT_VERIFY_TOP_K", "Candidates verified per decision (5)"),
            ConfigKey("AGENT_STOP_THRESHOLD", "Minimum supported probability to stop (0.9)"),
            ConfigKey("AGENT_STOP_MARGIN", "Required top-candidate probability gap (0.1)"),
            ConfigKey("AGENT_CALIBRATION_PATH", "Dev calibration artifact or aggregation profiles bundle with fitted stopping parameters"),
            ConfigKey("AGENT_CODEX_BIN", "Codex CLI executable (codex)"),
            ConfigKey("AGENT_CLAUDE_BIN", "Claude Code CLI executable (claude)"),
            ConfigKey("AGENT_CODEX_MODEL", "Codex model (gpt-6.1-sol)"),
            ConfigKey("AGENT_CODEX_REASONING_EFFORT", "Codex effort: low | medium | high | xhigh (high)"),
            ConfigKey("AGENT_CLAUDE_MODEL", "Claude model (claude-opus-5-5)"),
            ConfigKey("AGENT_CLAUDE_EFFORT", "Claude effort: low | medium | high | xhigh | max (high)"),
            ConfigKey("AGENT_CODEX_FAST", "Codex fast mode, priority tier (true/false)"),
            ConfigKey("AGENT_CLAUDE_FAST", "Claude Code fast mode (true/false)"),
            ConfigKey("AGENT_TIMEOUT_SECONDS", "Hard stop per agent run"),
            ConfigKey("AGENT_MAX_CONCURRENT", "Parallel runs per CLI across tabs"),
            ConfigKey("AGENT_BACKEND_URL", "Backend URL the agents' tool bridge calls (blank = auto)"),
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
            ConfigKey("OCR_MISSING_CATEGORIES_2", "InfoShot++ categories without OCR (e.g. L26)"),
            ConfigKey("MILVUS_IMAGE_COLLECTION_1", "BTC image vector collection"),
            ConfigKey("MILVUS_IMAGE_COLLECTION_2", "InfoShot++ image vector collection"),
            ConfigKey(
                "MILVUS_QWEN3_VL_IMAGE_COLLECTION_2",
                "InfoShot++ Qwen3-VL image vector collection",
            ),
            ConfigKey("MILVUS_TARA_COLLECTION_2", "InfoShot++ TARA clip vector collection"),
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
