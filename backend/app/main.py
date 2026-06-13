"""AIC26 retrieval backend — FastAPI app.

Endpoints:
  GET  /api/health
  POST /api/query/parse
  POST /api/search
  POST /api/search/trake
  GET  /api/keyframes/{submit_keyframe_id:path}
  GET  /api/videos/{video_id}/timeline
  POST /api/videos/{video_id}/snap
  POST /api/submit
  GET  /api/submit/history

Secrets stay server-side; responses never include Elastic/Milvus/NVIDIA creds.
"""
from __future__ import annotations

import asyncio

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .identity import canonical_submit_keyframe_id, parse_submit_keyframe_id
from .media import MediaUrlBuilder
from .models import (
    ParseRequest,
    SearchRequest,
    SimpleSearchRequest,
    SnapRequest,
    SubmitRequest,
    TranslateRequest,
    TrakeSearchRequest,
)
from .translate import translate_vi_to_en
from .services.search_service import SearchService, ServiceUnavailable
from .services.submit_service import DuplicateSubmitError, SubmitService
from .services.timeline_service import TimelineService
from .services.trake_service import TrakeService
from .trake import snap_to_keyframe

settings = get_settings()
app = FastAPI(title="AIC26 Retrieval Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_service = SearchService(settings)
trake_service = TrakeService(settings, search_service)
timeline_service = TimelineService(settings)
submit_service = SubmitService(settings)
media = MediaUrlBuilder(settings.media_base_url)


@app.get("/api/health")
async def health():
    elastic = await search_service.elastic.health()
    milvus = await search_service.milvus.health()
    pe = await search_service.pe.health()
    services = {"elastic": elastic, "milvus": milvus, "pe_encoder": pe}
    ok = settings.mock_mode or all(s.get("ok") for s in services.values())
    return {
        "ok": ok,
        "mode": "mock" if settings.mock_mode else "live",
        "services": services,
        "capabilities": {
            "llm_query_parser": settings.has_llm and not settings.mock_mode,
            "dres_submit": settings.has_dres,
            "audio_vector_search": settings.has_glap and not settings.mock_mode,
            "vlm_rerank": False,
            "vlm_cot": False,
        },
        "warnings": [
            f"{name} unreachable: {s.get('error')}"
            for name, s in services.items()
            if not s.get("ok")
        ],
    }


@app.post("/api/query/parse")
async def parse_query(req: ParseRequest):
    parsed = await search_service.parser.parse(
        req.query,
        req.query_type_hint,
        req.previous_hints,
        req.manual_overrides.model_dump(),
        use_llm=req.use_llm,
    )
    return parsed


@app.post("/api/search/simple")
async def search_simple(req: SimpleSearchRequest):
    """Minimal vector-only search: flat keyframe list ordered by cosine."""
    try:
        return await search_service.simple_image_search(req.query, top_k=req.top_k)
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/search")
async def search(req: SearchRequest):
    payload = req.model_dump()
    if req.feedback is not None:
        payload["feedback"] = req.feedback.model_dump()
    payload["manual_overrides"] = req.manual_overrides.model_dump()
    return await search_service.search(payload)


@app.post("/api/search/trake")
async def search_trake(req: TrakeSearchRequest):
    payload = req.model_dump()
    payload["manual_overrides"] = req.manual_overrides.model_dump()
    return await trake_service.search_trake(payload)


@app.get("/api/keyframes/{submit_keyframe_id:path}")
async def get_keyframe(submit_keyframe_id: str):
    try:
        canonical = canonical_submit_keyframe_id(submit_keyframe_id)
        parsed = parse_submit_keyframe_id(canonical)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    record = await search_service.elastic.get_keyframe(canonical)
    return {
        "image_id": canonical,
        "submit_keyframe_id": canonical,
        "video_id": parsed.video_id,
        "keyframe_n": parsed.keyframe_n,
        "pts_time": (record or {}).get("pts_time"),
        "fps": (record or {}).get("fps"),
        "frame_idx": (record or {}).get("frame_idx"),
        "keyframe_url": media.keyframe_url(parsed.video_id, parsed.keyframe_n),
        "video_url": media.video_url(parsed.video_id),
        "found": record is not None,
    }


@app.get("/api/videos/{video_id}/timeline")
async def video_timeline(video_id: str):
    return await timeline_service.build(video_id)


@app.post("/api/videos/{video_id}/snap")
async def snap_frame(video_id: str, req: SnapRequest):
    """Snap a raw paused time to the nearest BTC keyframe for TRAKE frame-pick."""
    timeline = await timeline_service.build(video_id)
    fps = req.fps or timeline.get("fps") or 25.0
    result = snap_to_keyframe(req.raw_time, fps, timeline["keyframes"])
    if result is None:
        raise HTTPException(status_code=404, detail=f"No keyframes for {video_id}")
    return {
        "video_id": video_id,
        "fps": fps,
        "submit_keyframe_id": result.submit_keyframe_id,
        "keyframe_n": result.keyframe_n,
        "frame_idx": result.frame_idx,
        "pts_time": result.pts_time,
        "raw_time": result.raw_time,
        "raw_frame_idx": result.raw_frame_idx,
        "delta_frames": result.delta_frames,
        "delta_seconds": round(result.delta_seconds, 4),
        "far": result.far,
        "keyframe_url": media.keyframe_url(video_id, result.keyframe_n),
    }


@app.post("/api/submit")
async def submit(req: SubmitRequest):
    try:
        entry = await submit_service.submit(
            task_id=req.task_id,
            query_type=req.query_type,
            payload=req.payload.model_dump(),
            allow_duplicate=req.allow_duplicate,
        )
    except DuplicateSubmitError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "duplicate_submit",
                "submit_keyframe_id": exc.submit_keyframe_id,
                "task_id": exc.task_id,
            },
        ) from exc
    return entry


@app.get("/api/submit/history")
async def submit_history(task_id: str | None = None):
    return {"history": submit_service.history(task_id)}


@app.post("/api/translate")
async def translate_endpoint(req: TranslateRequest):
    en = await translate_vi_to_en(req.text)
    return {"text": req.text, "text_en": en}


@app.post("/api/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...), translate: bool = True):
    """Whisper speech-to-text for cross-browser voice input. Returns the Vietnamese
    transcript and (optionally) its English translation."""
    from .transcribe import available, transcribe_audio

    if not available():
        raise HTTPException(
            status_code=503,
            detail="Whisper unavailable — install faster-whisper and ensure the model can download.",
        )
    data = await audio.read()
    try:
        text = await asyncio.to_thread(transcribe_audio, data)  # CPU-bound → thread
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    text_en = await translate_vi_to_en(text) if (translate and text) else None
    return {"text": text, "text_en": text_en}
