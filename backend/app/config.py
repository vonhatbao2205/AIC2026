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

#: Public read endpoint of the team's Hugging Face storage bucket. The browse UI
#: is `.../tree/<path>`; `/resolve/<path>` is what actually serves bytes (and
#: honours Range requests, which `<video>` seeking needs).
HF_MEDIA_BASE_URL = "https://huggingface.co/buckets/Baonenha1/aic26-media/resolve"


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
    # Qwen3-VL-Embedding-8B text encoder. Its 4096-d space is independent from
    # PE-Core's 1280-d space, so it has its own worker credentials and Milvus
    # collection and is combined with PE by rank only.
    qwen3_vl_encoder_url: str | None = None
    qwen3_vl_encoder_token: str | None = None
    qwen3_vl_encoder_timeout_seconds: float = 120.0
    # TARA video-clip text encoder, pinned to the InfoShot++ L21-L30 artifact.
    tara_encoder_url: str | None = None
    tara_encoder_token: str | None = None
    tara_encoder_timeout_seconds: float = 120.0
    tara_enabled: bool = False
    # GLAP audio↔text encoder (/encode-audio-text). Usually the SAME Kaggle server
    # as PE; if unset, falls back to pe_encoder_url.
    glap_encoder_url: str | None = None
    media_base_url: str = "https://media.example.com"
    # Active keyframe/video origins for the profile handling this request.
    keyframe_media_base_url: str = "https://media.example.com"
    keyframe_media_base_url_2: str = HF_MEDIA_BASE_URL
    # Where the console retries a keyframe when its primary origin fails. The
    # active alias is selected per retrieval profile below, just like the base.
    keyframe_media_fallback_base_url: str = ""
    keyframe_media_fallback_base_url_1: str = ""
    keyframe_media_fallback_base_url_2: str = ""
    # Videos are split the same way keyframes are, because the two origins do not
    # hold the same corpus: the Hugging Face bucket carries the 873 InfoShot++
    # videos (L21–L30) only, so pointing profile 1 at it would 404 every K01–K20
    # video. Empty means "same origin as media_base_url".
    video_media_base_url: str = ""
    video_media_base_url_1: str = ""
    video_media_base_url_2: str = HF_MEDIA_BASE_URL
    # Where the console retries a video the primary origin failed to serve.
    #
    # The fast path for L21-L30 is a named Cloudflare tunnel — measured at ~63 ms
    # to first byte against ~834 ms for the Hugging Face bucket, which answers
    # every range request with a 302 to its CDN and pays that round trip again on
    # every seek. But a tunnel terminates on a machine that can be switched off,
    # while the bucket is always-on hosting: speed and availability are not the
    # same property, so the bucket stays configured as the fallback rather than
    # being replaced.
    # Active alias for the profile handling this request, filled by
    # `for_retrieval_database` exactly like `video_media_base_url` above.
    video_media_fallback_base_url: str = ""
    video_media_fallback_base_url_1: str = ""
    video_media_fallback_base_url_2: str = ""
    # Query LLM: the routing parser (the console's "LLM" switch) and visual query
    # expansion ("Expand"). OpenAI-compatible chat completions; the default is
    # DeepSeek's `deepseek-flash` (DeepSeek-V4.1-Flash), called with thinking
    # off and JSON output on. The key defaults to DEEPSEEK_API_KEY.
    query_llm_api_key: str | None = None
    query_llm_base_url: str = "https://api.deepseek.com"
    query_llm_model: str = "deepseek-flash"
    query_llm_timeout_seconds: float = 30.0
    # NVIDIA NIM: only the LLM fallback of VI->EN translation when both Google
    # endpoints fail. Qwen3-next reliably answers in English for Vietnamese input.
    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_fast_model: str = "qwen/qwen3-next-80b-a3b-instruct"
    # QA copilot visual passes (pass 1: hotspots + answers; pass 3: web answers
    # checked against the frames). "deepseek": DeepSeek V4.1 Flash reads the
    # keyframes itself (image input, thinking on) with DEEPSEEK_API_KEY. "nvila":
    # the NVILA-8B Colab worker below.
    qa_vision_backend: str = "deepseek"
    qa_vision_base_url: str = "https://api.deepseek.com"
    qa_vision_model: str = "deepseek-flash"
    qa_vision_reasoning_effort: str = "high"
    # "high" keeps small on-screen text (plates, prices, poem lines) readable.
    qa_vision_image_detail: str = "high"
    qa_vision_timeout_seconds: float = 180.0
    qa_vision_max_output_tokens: int = 16000
    # NVILA-8B visual QA server (the companion Colab notebook exposes /qa/analyze).
    # The URL is a volatile Cloudflare quick-tunnel URL unless a named tunnel is used.
    nvila_base_url: str | None = None
    nvila_token: str | None = None
    nvila_timeout_seconds: float = 240.0
    nvila_max_candidates: int = 12
    # Qwen3-VL-Reranker worker (aic26_qwen3vl_reranker8b_colab_server.ipynb).
    # Off by default: it is an opt-in refinement the operator ticks per search,
    # and it costs a second GPU session that is not always running.
    qwen_reranker_url: str | None = None
    qwen_reranker_token: str | None = None
    qwen_reranker_enabled: bool = True
    # A cold rerank downloads every candidate keyframe on the worker before it
    # can score anything, so the first request of a session is far slower than
    # the rest. 30s timed out in practice on a 200-candidate pool.
    qwen_reranker_timeout_seconds: float = 120.0
    # Candidates retrieved from Milvus BEFORE reranking. Must exceed the result
    # depth the operator asked for, or the reranker can only shuffle what the
    # retrievers already ranked highly (see reranker.md §7). It is also the size
    # of the MERGED pool: each selected index retrieves this deep, the union is
    # then cut back to this many candidates in total, so ticking a second index
    # does not double the reranker's GPU cost — it splits the depth between them.
    qwen_reranker_candidates: int = 200
    # DeepSeek model with the server-side web_search tool, used to resolve
    # external facts after NVILA has extracted visual/ASR clues. The tool is only
    # available on the Responses API and only for deepseek-v4-flash.
    deepseek_api_key: str | None = None
    # The one model whose server-side web_search actually runs (see the adapter);
    # at "low" effort it answers in ~5-15 s with a real search.
    deepseek_grounding_model: str = "deepseek-v4-pro"
    deepseek_grounding_base_url: str = "https://api.deepseek.com"
    deepseek_grounding_enabled: bool = True
    deepseek_grounding_timeout_seconds: float = 90.0
    deepseek_grounding_auto_threshold: float = 0.55
    # Thinking is enabled by default and draws from the same budget as the answer.
    deepseek_grounding_max_output_tokens: int = 8000
    deepseek_grounding_reasoning_effort: str = "low"
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
    # Master switch for live submission. On for the final round, which is judged
    # live on DRES (HD-ChungKet-2026). DRES_ENABLED=false falls back to the
    # preliminary round's workflow: answers collected in the Submission tab and
    # exported as a CSV pack, so nothing reaches the evaluation server.
    dres_enabled: bool = True
    # DRES media collection, sent as `mediaItemCollectionName` with every video
    # answer. Needed when the server holds several collections and cannot
    # resolve a bare video name; empty = let DRES pick.
    dres_collection_name: str | None = None

    # Index / collection names (override only if you re-uploaded with a new prefix).
    # `_1` = BTC keyframes (v1 indices), `_2` = InfoShot++ keyframes (v2 indices).
    # The unsuffixed name is the active alias `for_retrieval_database` fills in.
    idx_keyframe_map: str = "aic26_keyframe_map_v1"
    idx_keyframe_map_1: str = "aic26_keyframe_map_v1"
    idx_keyframe_map_2: str = "aic26_keyframe_map_infoshotpp_v1"
    idx_ocr: str = "aic26_ocr_keyframes_v1"
    idx_ocr_1: str = "aic26_ocr_keyframes_v1"
    idx_ocr_2: str = "aic26_ocr_keyframes_v2"
    idx_speech: str = "aic26_speech_segments_v1"
    idx_speech_1: str = "aic26_speech_segments_v1"
    idx_speech_2: str = "aic26_speech_segments_v2"
    idx_audio: str = "aic26_audio_windows_v1"
    idx_audio_1: str = "aic26_audio_windows_v1"
    idx_audio_2: str = "aic26_audio_windows_v2"
    # Categories whose OCR is not in the active OCR index. The InfoShot++ OCR
    # artifact has no L26 yet, and an operator has to be able to see that a
    # negative OCR result there means "not indexed", not "not on screen".
    ocr_missing_categories: tuple[str, ...] = ()
    ocr_missing_categories_2: tuple[str, ...] = ("L26",)
    milvus_image_collection: str = "aic26_image_peg14_v1"
    milvus_image_collection_1: str = "aic26_image_peg14_v1"
    milvus_image_collection_2: str = "aic26_image_peg14_infoshotpp_v1"
    milvus_qwen3_vl_image_collection: str = "aic26_image_qwen3vl8b_infoshotpp_v3"
    milvus_qwen3_vl_image_collection_2: str = "aic26_image_qwen3vl8b_infoshotpp_v3"
    milvus_tara_collection: str = "aic26_tara_clips_infoshotpp_v1"
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
        if not self.video_media_base_url:
            self.video_media_base_url = self.media_base_url.rstrip("/")

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
    def has_qwen3_vl_encoder(self) -> bool:
        # The companion Colab worker always authenticates with a bearer token.
        return bool(self.qwen3_vl_encoder_url and self.qwen3_vl_encoder_token)

    @property
    def has_qwen3_vl_search(self) -> bool:
        return bool(
            self.is_infoshotpp
            and self.has_qwen3_vl_encoder
            and self.has_milvus
            and self.milvus_qwen3_vl_image_collection
        )

    @property
    def has_tara_search(self) -> bool:
        return bool(
            self.is_infoshotpp and self.tara_enabled and self.has_milvus
            and self.tara_encoder_url and self.tara_encoder_token
            and self.milvus_tara_collection
        )

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

    @property
    def unsupported_channels(self) -> frozenset[str]:
        """Channels this profile has no index for, so retrieval can say so.

        Since the v2 metadata upload, InfoShot++ answers OCR, speech and audio
        from its own `*_v2` indices, and the V-KIS sketch searches whichever PE
        image collection the profile has, so no channel is missing today. Kept
        as the one place a future profile declares what it lacks.
        """
        return frozenset()

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
                idx_ocr=self.idx_ocr_1,
                idx_speech=self.idx_speech_1,
                idx_audio=self.idx_audio_1,
                ocr_missing_categories=(),
                milvus_image_collection=self.milvus_image_collection_1,
                keyframe_media_base_url=self.media_base_url,
                keyframe_media_fallback_base_url=self.keyframe_media_fallback_base_url_1.rstrip("/"),
                video_media_base_url=(self.video_media_base_url_1 or self.media_base_url).rstrip("/"),
                video_media_fallback_base_url=self.video_media_fallback_base_url_1.rstrip("/"),
            )
        return replace(
            self,
            retrieval_database="infoshotpp",
            milvus_endpoint=self.milvus_endpoint_2,
            milvus_token=self.milvus_token_2,
            idx_keyframe_map=self.idx_keyframe_map_2,
            idx_ocr=self.idx_ocr_2,
            idx_speech=self.idx_speech_2,
            idx_audio=self.idx_audio_2,
            ocr_missing_categories=self.ocr_missing_categories_2,
            milvus_image_collection=self.milvus_image_collection_2,
            milvus_qwen3_vl_image_collection=self.milvus_qwen3_vl_image_collection_2,
            keyframe_media_base_url=self.keyframe_media_base_url_2,
            keyframe_media_fallback_base_url=self.keyframe_media_fallback_base_url_2.rstrip("/"),
            video_media_base_url=(self.video_media_base_url_2 or self.media_base_url).rstrip("/"),
            video_media_fallback_base_url=self.video_media_fallback_base_url_2.rstrip("/"),
        )

    @property
    def has_llm(self) -> bool:
        """A query LLM is configured (routing parser + expansion)."""
        return bool(self.query_llm_api_key and self.query_llm_model)

    @property
    def has_qa_vision(self) -> bool:
        """The selected visual QA backend is configured."""
        if self.qa_vision_backend == "nvila":
            return self.has_nvila
        return bool(self.deepseek_api_key)

    @property
    def has_nvila(self) -> bool:
        # The companion worker always requires Bearer auth. Treat a URL without
        # its matching token as incomplete rather than attempting live calls.
        return bool(self.nvila_base_url and self.nvila_token)

    @property
    def has_qwen_reranker(self) -> bool:
        # Like the NVILA worker, the rerank service always requires Bearer auth;
        # a URL without its token is incomplete, not "almost configured".
        return bool(self.qwen_reranker_url and self.qwen_reranker_token)

    @property
    def has_web_grounding(self) -> bool:
        return bool(self.deepseek_grounding_enabled and self.deepseek_api_key)

    @property
    def has_dres(self) -> bool:
        """Enough to reach DRES: a host plus either credentials or a live session.

        Mock mode never talks to the real evaluation server — fixtures are not
        answers, and the whole test suite runs with mock mode on.
        """
        if self.mock_mode or not self.dres_enabled:
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

    try:
        query_llm_timeout_seconds = float(_env("QUERY_LLM_TIMEOUT_SECONDS", "30") or "30")
    except ValueError:
        query_llm_timeout_seconds = 30.0
    deepseek_api_key = _env("DEEPSEEK_API_KEY") or file_default("deepseek_apikey.txt")
    try:
        qa_vision_timeout_seconds = min(600.0, max(20.0, float(_env("QA_VISION_TIMEOUT_SECONDS", "180") or "180")))
    except ValueError:
        qa_vision_timeout_seconds = 180.0
    try:
        qa_vision_max_output_tokens = min(64000, max(2000, int(_env("QA_VISION_MAX_OUTPUT_TOKENS", "16000") or "16000")))
    except ValueError:
        qa_vision_max_output_tokens = 16000

    try:
        nvila_timeout_seconds = float(_env("NVILA_TIMEOUT_SECONDS", "240") or "240")
    except ValueError:
        nvila_timeout_seconds = 240.0
    try:
        nvila_max_candidates = int(_env("NVILA_MAX_CANDIDATES", "12") or "12")
    except ValueError:
        nvila_max_candidates = 12
    try:
        qwen3_vl_encoder_timeout_seconds = float(
            _env("QWEN3_VL_ENCODER_TIMEOUT_SECONDS", "120") or "120"
        )
    except ValueError:
        qwen3_vl_encoder_timeout_seconds = 120.0
    try:
        tara_encoder_timeout_seconds = float(_env("TARA_ENCODER_TIMEOUT_SECONDS", "120") or "120")
    except ValueError:
        tara_encoder_timeout_seconds = 120.0
    reranker_env = (_env("QWEN_RERANKER_ENABLED", "true") or "true").lower()
    qwen_reranker_enabled = reranker_env in {"1", "true", "yes", "on"}
    try:
        qwen_reranker_timeout_seconds = float(_env("QWEN_RERANKER_TIMEOUT_SECONDS", "120") or "120")
    except ValueError:
        qwen_reranker_timeout_seconds = 120.0
    try:
        qwen_reranker_candidates = int(_env("QWEN_RERANKER_CANDIDATES", "200") or "200")
    except ValueError:
        qwen_reranker_candidates = 200
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
    reasoning_effort = (_env("DEEPSEEK_GROUNDING_REASONING_EFFORT", "low") or "low").lower()
    if reasoning_effort not in {"low", "high", "max"}:
        reasoning_effort = "low"

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
    # `IDX_OCR`/`IDX_SPEECH`/`IDX_AUDIO` keep naming the BTC (profile 1) indices so
    # an existing .env keeps working; the InfoShot++ v2 indices get their own keys.
    idx_ocr_1 = _env("IDX_OCR_1") or _env("IDX_OCR") or "aic26_ocr_keyframes_v1"
    idx_speech_1 = _env("IDX_SPEECH_1") or _env("IDX_SPEECH") or "aic26_speech_segments_v1"
    idx_audio_1 = _env("IDX_AUDIO_1") or _env("IDX_AUDIO") or "aic26_audio_windows_v1"
    missing_ocr_2 = tuple(
        part.strip().upper()
        for part in (_env("OCR_MISSING_CATEGORIES_2", "L26") or "").split(",")
        if part.strip()
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
        qwen3_vl_encoder_url=(_env("QWEN3_VL_ENCODER_URL") or "").rstrip("/") or None,
        qwen3_vl_encoder_token=(
            _env("QWEN3_VL_ENCODER_TOKEN") or file_default("qwen3_vl_encoder_token.txt")
        ),
        qwen3_vl_encoder_timeout_seconds=max(10.0, qwen3_vl_encoder_timeout_seconds),
        tara_encoder_url=(_env("TARA_ENCODER_URL") or "").rstrip("/") or None,
        tara_encoder_token=_env("TARA_ENCODER_TOKEN") or file_default("tara_encoder_token.txt"),
        tara_encoder_timeout_seconds=max(10.0, tara_encoder_timeout_seconds),
        tara_enabled=(_env("TARA_ENABLED", "false") or "false").lower() in {"1", "true", "yes", "on"},
        glap_encoder_url=_env("GLAP_ENCODER_URL"),
        media_base_url=media_base_url,
        keyframe_media_base_url=media_base_url,
        keyframe_media_base_url_2=(_env("KEYFRAME_MEDIA_BASE_URL_2") or HF_MEDIA_BASE_URL).rstrip("/"),
        keyframe_media_fallback_base_url_1=(
            _env("KEYFRAME_MEDIA_FALLBACK_BASE_URL_1") or ""
        ).rstrip("/"),
        keyframe_media_fallback_base_url_2=(
            _env("KEYFRAME_MEDIA_FALLBACK_BASE_URL_2") or ""
        ).rstrip("/"),
        # Set VIDEO_MEDIA_BASE_URL_1 to the HF base too, once K01–K20 videos have
        # been migrated there; until then profile 1 must keep its R2 origin.
        video_media_base_url_1=(_env("VIDEO_MEDIA_BASE_URL_1") or media_base_url).rstrip("/"),
        video_media_base_url_2=(_env("VIDEO_MEDIA_BASE_URL_2") or HF_MEDIA_BASE_URL).rstrip("/"),
        # Unset by default: a fallback is only meaningful once the primary is
        # something other than the bucket, and pointing both at one origin would
        # dress a single point of failure up as redundancy.
        video_media_fallback_base_url_1=(_env("VIDEO_MEDIA_FALLBACK_BASE_URL_1") or "").rstrip("/"),
        video_media_fallback_base_url_2=(_env("VIDEO_MEDIA_FALLBACK_BASE_URL_2") or "").rstrip("/"),
        query_llm_api_key=_env("QUERY_LLM_API_KEY") or deepseek_api_key,
        query_llm_base_url=(_env("QUERY_LLM_BASE_URL") or "https://api.deepseek.com").rstrip("/"),
        query_llm_model=_env("QUERY_LLM_MODEL") or "deepseek-flash",
        query_llm_timeout_seconds=min(120.0, max(5.0, query_llm_timeout_seconds)),
        nvidia_api_key=_env("NVIDIA_API_KEY") or file_default("nvidia_api_key.txt"),
        nvidia_base_url=_env("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1",
        nvidia_fast_model=_env("NVIDIA_FAST_MODEL") or "qwen/qwen3-next-80b-a3b-instruct",
        nvila_base_url=(_env("NVILA_BASE_URL") or "").rstrip("/") or None,
        nvila_token=_env("NVILA_TOKEN") or file_default("nvila_token.txt"),
        nvila_timeout_seconds=max(30.0, nvila_timeout_seconds),
        nvila_max_candidates=min(24, max(1, nvila_max_candidates)),
        qwen_reranker_url=(_env("QWEN_RERANKER_URL") or "").rstrip("/") or None,
        qwen_reranker_token=_env("QWEN_RERANKER_TOKEN") or file_default("qwen_reranker_token.txt"),
        qwen_reranker_enabled=qwen_reranker_enabled,
        qwen_reranker_timeout_seconds=max(5.0, qwen_reranker_timeout_seconds),
        # The worker itself caps a request at 400 documents.
        qwen_reranker_candidates=min(400, max(10, qwen_reranker_candidates)),
        deepseek_api_key=deepseek_api_key,
        deepseek_grounding_model=_env("DEEPSEEK_GROUNDING_MODEL") or "deepseek-v4-pro",
        qa_vision_backend=("nvila" if (_env("QA_VISION_BACKEND") or "").strip().lower() == "nvila" else "deepseek"),
        qa_vision_base_url=(_env("QA_VISION_BASE_URL") or "https://api.deepseek.com").rstrip("/"),
        qa_vision_model=_env("QA_VISION_MODEL") or "deepseek-flash",
        qa_vision_reasoning_effort=(_env("QA_VISION_REASONING_EFFORT") or "high").strip().lower(),
        qa_vision_image_detail=(_env("QA_VISION_IMAGE_DETAIL") or "high").strip().lower(),
        qa_vision_timeout_seconds=qa_vision_timeout_seconds,
        qa_vision_max_output_tokens=qa_vision_max_output_tokens,
        deepseek_grounding_base_url=(_env("DEEPSEEK_GROUNDING_BASE_URL") or "https://api.deepseek.com").rstrip("/"),
        deepseek_grounding_enabled=deepseek_grounding_enabled,
        deepseek_grounding_timeout_seconds=max(10.0, deepseek_grounding_timeout_seconds),
        deepseek_grounding_auto_threshold=min(1.0, max(0.0, deepseek_grounding_auto_threshold)),
        deepseek_grounding_max_output_tokens=min(64000, max(1000, deepseek_grounding_max_output_tokens)),
        deepseek_grounding_reasoning_effort=reasoning_effort,
        # The final-round host (HD-ChungKet-2026); the organisers may announce
        # another URL on the day, which DRES_BASE_URL overrides.
        dres_base_url=(_env("DRES_BASE_URL") or "https://eventretrieval.one").rstrip("/"),
        dres_username=_env("DRES_USERNAME"),
        dres_password=_env("DRES_PASSWORD"),
        dres_session=_env("DRES_SESSION"),
        dres_evaluation_id=_env("DRES_EVALUATION_ID"),
        dres_segment_pad_ms=max(0, dres_segment_pad_ms),
        dres_timeout_seconds=max(5.0, dres_timeout_seconds),
        dres_enabled=(_env("DRES_ENABLED", "true") or "true").lower() in {"1", "true", "yes", "on"},
        dres_collection_name=_env("DRES_COLLECTION_NAME"),
        idx_keyframe_map=idx_keyframe_map_1,
        idx_keyframe_map_1=idx_keyframe_map_1,
        idx_keyframe_map_2=(
            _env("IDX_KEYFRAME_MAP_2") or "aic26_keyframe_map_infoshotpp_v1"
        ),
        idx_ocr=idx_ocr_1,
        idx_ocr_1=idx_ocr_1,
        idx_ocr_2=_env("IDX_OCR_2") or "aic26_ocr_keyframes_v2",
        idx_speech=idx_speech_1,
        idx_speech_1=idx_speech_1,
        idx_speech_2=_env("IDX_SPEECH_2") or "aic26_speech_segments_v2",
        idx_audio=idx_audio_1,
        idx_audio_1=idx_audio_1,
        idx_audio_2=_env("IDX_AUDIO_2") or "aic26_audio_windows_v2",
        ocr_missing_categories_2=missing_ocr_2,
        milvus_image_collection=image_collection_1,
        milvus_image_collection_1=image_collection_1,
        milvus_image_collection_2=(
            _env("MILVUS_IMAGE_COLLECTION_2") or "aic26_image_peg14_infoshotpp_v1"
        ),
        milvus_qwen3_vl_image_collection=(
            _env("MILVUS_QWEN3_VL_IMAGE_COLLECTION_2")
            or "aic26_image_qwen3vl8b_infoshotpp_v3"
        ),
        milvus_qwen3_vl_image_collection_2=(
            _env("MILVUS_QWEN3_VL_IMAGE_COLLECTION_2")
            or "aic26_image_qwen3vl8b_infoshotpp_v3"
        ),
        milvus_tara_collection=(
            _env("MILVUS_TARA_COLLECTION_2") or "aic26_tara_clips_infoshotpp_v1"
        ),
        milvus_audio_collection=_env("MILVUS_AUDIO_COLLECTION") or "aic26_audio_glap_v1",
        mock_mode=mock_mode,
        translate_to_en=translate_to_en,
        cors_origins=cors_origins,
    )
