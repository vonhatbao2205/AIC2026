"""AIC26 retrieval backend — FastAPI app.

Endpoints:
  GET  /api/health
  GET  /api/config
  GET  /api/config/template
  POST /api/config/import
  POST /api/config/reload
  POST /api/query/parse
  GET  /api/search/scope
  POST /api/search
  POST /api/search/trake
  POST /api/answers/generate
  POST /api/qa/analyze
  GET  /api/keyframes/{submit_keyframe_id:path}
  GET  /api/videos/{video_id}/timeline
  POST /api/videos/{video_id}/snap
  GET  /api/dres/status
  POST /api/dres/login
  GET  /api/dres/evaluations
  GET  /api/dres/current-task
  GET  /api/dres/task-hint
  POST /api/submit/preview
  POST /api/submit
  GET  /api/submit/history
  DEL  /api/submit/history

Secrets stay server-side; responses never include Elastic/Milvus/NVIDIA creds.
"""
from __future__ import annotations

import asyncio
import re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .adapters.deepseek_grounding import DeepSeekGroundingClient, WebGroundingUnavailable
from .adapters.dres_client import DresClient, DresError, DresNotConfigured
from .adapters.nvila_client import NvilaQaClient, NvilaUnavailable
from . import paths
from .canvas import palette_manifest
from .config import Settings, get_settings
from .identity import canonical_submit_keyframe_id, parse_submit_keyframe_id
from .media import MediaUrlBuilder
from .models import (
    AnswerGenerateRequest,
    CanvasSearchRequest,
    ParseRequest,
    QaAnalyzeRequest,
    RetrievalDatabase,
    SearchRequest,
    SimpleSearchRequest,
    SnapRequest,
    SubmitRequest,
    TranslateRequest,
    TrakeSearchRequest,
)
from .scope import catalogue as scope_catalogue, match_topics
from .translate import translate_vi_to_en
from .services import config_service
from .services.answer_service import AnswerService
from .services.canvas_service import CanvasService
from .services.search_service import SearchService, ServiceUnavailable
from .services.submit_service import DuplicateSubmitError, SubmitFormatError, SubmitService
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


def build_runtime() -> Settings:
    """(Re)create every service from the current configuration.

    The packaged app is configured through the UI, not the shell: after an
    operator imports a `.env` the process must pick the new credentials up
    without a restart. Endpoints resolve these names at call time, so rebinding
    the module globals swaps the whole stack atomically from the caller's view.
    """
    global settings, search_service, trake_service, canvas_service, timeline_service
    global search_services, trake_services, timeline_services, media_builders, profile_settings
    global answer_services
    global dres_client, submit_service, media, nvila_qa, web_grounding_client

    get_settings.cache_clear()
    settings = get_settings()
    profile_settings = {
        name: settings.for_retrieval_database(name) for name in ("btc", "infoshotpp")
    }
    search_services = {name: SearchService(cfg) for name, cfg in profile_settings.items()}
    trake_services = {
        name: TrakeService(profile_settings[name], search_services[name]) for name in profile_settings
    }
    timeline_services = {name: TimelineService(cfg) for name, cfg in profile_settings.items()}
    media_builders = {
        name: MediaUrlBuilder(cfg.keyframe_media_base_url, cfg.video_media_base_url)
        for name, cfg in profile_settings.items()
    }
    answer_services = {
        name: AnswerService(profile_settings[name], search_services[name], trake_services[name])
        for name in profile_settings
    }
    # Backward-compatible aliases used by tests and a few internal call sites.
    search_service = search_services["btc"]
    trake_service = trake_services["btc"]
    timeline_service = timeline_services["btc"]
    canvas_service = CanvasService(profile_settings["btc"], search_service)
    dres_client = DresClient(settings)
    # History is reloaded from disk with the new service; it is not credentials.
    submit_service = SubmitService(settings, dres_client)
    media = media_builders["btc"]
    nvila_qa = NvilaQaClient(settings)
    web_grounding_client = DeepSeekGroundingClient(settings)
    return settings


build_runtime()


#: Services whose absence actually degrades retrieval. Everything else is an
#: accessory the console already hides when its worker is gone — the Rerank tick
#: box disappears, the QA panel says NVILA is unavailable — so repeating it in
#: the health banner only teaches the operator to ignore a banner that is
#: supposed to mean "your search results are wrong". `ok` has always been
#: computed from these alone; the banner just did not agree with it.
CORE_RETRIEVAL_SERVICES = frozenset(
    {"elastic", "milvus", "pe_encoder", "qwen3_vl_encoder", "qwen3_vl_milvus"}
)


@app.get("/api/health")
async def health(retrieval_database: RetrievalDatabase = "btc"):
    selected = search_services[retrieval_database]
    selected_settings = profile_settings[retrieval_database]
    (
        elastic,
        milvus,
        pe,
        qwen3_vl_encoder,
        qwen3_vl_milvus,
        nvila,
        reranker,
        grounding,
        dres,
    ) = await asyncio.gather(
        selected.elastic.health(),
        selected.milvus.health(),
        selected.pe.health(),
        selected.qwen3_vl.health(),
        selected.milvus.health_qwen_image(),
        nvila_qa.health(),
        selected.reranker.health(),
        web_grounding_client.health(),
        dres_client.health(),
    )
    objects = (
        await canvas_service.objects.health()
        if retrieval_database == "btc"
        else {"ok": False, "mode": "disabled", "reason": "not indexed for InfoShot++"}
    )
    services = {
        "elastic": elastic,
        "milvus": milvus,
        "pe_encoder": pe,
        "qwen3_vl_encoder": qwen3_vl_encoder,
        "qwen3_vl_milvus": qwen3_vl_milvus,
        "nvila_qa": nvila,
        "qwen_reranker": reranker,
        "web_grounding": grounding,
        "object_index": objects,
        "dres": dres,
    }
    # NVILA is an optional QA accelerator: a stopped Colab session must not mark
    # the core retrieval stack unhealthy.
    pe_image_ready = bool(milvus.get("ok") and pe.get("ok"))
    qwen3_vl_image_ready = bool(
        selected_settings.is_infoshotpp
        and qwen3_vl_encoder.get("ok")
        and qwen3_vl_milvus.get("ok")
    )
    if selected_settings.is_infoshotpp:
        ok = selected_settings.mock_mode or bool(
            elastic.get("ok") and (pe_image_ready or qwen3_vl_image_ready)
        )
    else:
        ok = selected_settings.mock_mode or bool(elastic.get("ok") and pe_image_ready)
    has_live_nvila = settings.has_nvila and not settings.mock_mode and bool(nvila.get("ok"))
    has_qa_nvila = settings.mock_mode or has_live_nvila
    has_qa_visual_verification = settings.mock_mode or (
        has_live_nvila and bool(nvila.get("visual_verification_pass"))
    )
    return {
        "ok": ok,
        "mode": "mock" if settings.mock_mode else "live",
        "services": services,
        # The console needs the origins, not just the built URLs: when a primary
        # is down every asset under it fails, so the retry swaps the origin.
        "media": {
            "keyframe_base_url": selected_settings.keyframe_media_base_url,
            "keyframe_fallback_base_url": selected_settings.keyframe_media_fallback_base_url,
            "video_base_url": selected_settings.video_media_base_url,
            "video_fallback_base_url": selected_settings.video_media_fallback_base_url,
        },
        "capabilities": {
            "llm_query_parser": settings.has_llm and not settings.mock_mode,
            "dres_submit": settings.has_dres,
            "dres_connected": bool(dres.get("ok")),
            "audio_vector_search": selected_settings.has_glap and not settings.mock_mode,
            "image_pe_search": selected_settings.mock_mode or pe_image_ready,
            "qwen3_vl_embedding_search": qwen3_vl_image_ready,
            # Both profiles now have their own OCR/speech/audio indices (BTC: the
            # `*_v1` set, InfoShot++: the `*_v2` set built on the new keyframe map).
            "ocr_search": "ocr" not in selected_settings.unsupported_channels,
            "speech_search": "speech" not in selected_settings.unsupported_channels,
            "audio_search": "audio" not in selected_settings.unsupported_channels,
            # NVILA currently reranks/grounds only the QA candidate pack; it is
            # not a general reranker for T-KIS/V-KIS/TRAKE retrieval results.
            "vlm_rerank": False,
            # Qwen3-VL-Reranker over the PE candidate pool. The console only
            # offers the tick box when a worker can actually answer, so the
            # operator never turns on a refinement that silently does nothing.
            "visual_rerank": bool(reranker.get("ok")),
            # The worker returns short verification reasons, never hidden or
            # free-form chain-of-thought traces.
            "vlm_cot": False,
            "qa_nvila": has_qa_nvila,
            "qa_hotspot_prediction": has_qa_nvila,
            "qa_candidate_answers": has_qa_nvila,
            "qa_visual_verification": has_qa_visual_verification,
            "qa_web_grounding": settings.mock_mode or settings.has_web_grounding,
            # V-KIS canvas: OD spatial match needs the object index to answer.
            "canvas_object_search": retrieval_database == "btc" and (
                settings.mock_mode or bool(objects.get("ok"))
            ),
        },
        "retrieval_database": retrieval_database,
        "indices": {
            "keyframe_map": selected_settings.idx_keyframe_map,
            "ocr": selected_settings.idx_ocr,
            "speech": selected_settings.idx_speech,
            "audio": selected_settings.idx_audio,
            "image_pe": selected_settings.milvus_image_collection,
            "image_qwen": (
                selected_settings.milvus_qwen3_vl_image_collection
                if selected_settings.is_infoshotpp
                else None
            ),
        },
        # Categories the active OCR index does not cover, so a blank OCR result
        # there reads as "not indexed" rather than "no text on screen".
        "ocr_missing_categories": list(selected_settings.ocr_missing_categories),
        "warnings": [
            f"{name} unreachable: {s.get('error')}"
            for name, s in services.items()
            if not s.get("ok")
            and s.get("mode") != "disabled"
            and name in CORE_RETRIEVAL_SERVICES
        ],
    }


@app.get("/api/config")
async def config_status():
    """Which variables are set, for the settings screen. Secrets come back masked."""
    return config_service.status(settings)


@app.get("/api/config/template")
async def config_template():
    """The documented `.env` template, so an operator can fill one in from the app."""
    return PlainTextResponse(
        config_service.template_text(),
        headers={"Content-Disposition": 'attachment; filename=".env.example"'},
    )


@app.post("/api/config/import")
async def config_import(file: UploadFile = File(...), replace: bool = Form(False)):
    """Import an uploaded `.env` and rebuild the service stack around it.

    Merges by default: keys the file omits keep their current value. `replace`
    makes the uploaded file the whole configuration.
    """
    raw = await file.read()
    if len(raw) > 256_000:
        raise HTTPException(status_code=413, detail="Config file too large (max 256 KB).")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="Not a UTF-8 text file — expected a .env file."
        ) from exc
    try:
        summary = config_service.apply_env_text(text, replace=replace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    build_runtime()
    return {**summary, "status": config_service.status(settings)}


@app.post("/api/config/reload")
async def config_reload():
    """Re-read the config file from disk and rebuild the services."""
    build_runtime()
    return config_service.status(settings)


@app.post("/api/query/parse")
async def parse_query(req: ParseRequest):
    parsed = await search_service.parser.parse(
        req.query,
        req.query_type_hint,
        req.previous_hints,
        req.manual_overrides.model_dump(),
        use_llm=req.use_llm,
        translate=req.translate,
    )
    return parsed


@app.get("/api/search/scope")
async def search_scope(retrieval_database: RetrievalDatabase = "btc", query: str = ""):
    """The folder catalogue this profile can search, plus what the topic heuristic
    makes of `query` — the console draws its folder picker from this and previews
    the auto selection while the operator is still typing."""
    body = scope_catalogue(retrieval_database)
    body["suggestion"] = (
        search_services[retrieval_database]
        .resolve_scope({"mode": "auto"}, query=query)
        .to_dict()
        if query.strip()
        else None
    )
    body["topic_keywords"] = [
        {"topic_id": match.topic_id, "keywords": list(match.keywords)}
        for match in match_topics(query)
    ]
    return body


@app.post("/api/search/simple")
async def search_simple(req: SimpleSearchRequest):
    """Minimal vector-only search: flat keyframe list ordered by cosine."""
    try:
        return await search_services[req.retrieval_database].simple_image_search(
            req.query,
            top_k=req.top_k,
            scope_spec=req.scope.model_dump(),
            rerank=req.rerank,
            image_models=req.image_models,
            translate=req.translate,
        )
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/search")
async def search(req: SearchRequest):
    payload = req.model_dump()
    if req.feedback is not None:
        payload["feedback"] = req.feedback.model_dump()
    payload["manual_overrides"] = req.manual_overrides.model_dump()
    payload["scope"] = req.scope.model_dump()
    return await search_services[req.retrieval_database].search(payload)


@app.post("/api/search/trake")
async def search_trake(req: TrakeSearchRequest):
    payload = req.model_dump()
    payload["manual_overrides"] = req.manual_overrides.model_dump()
    payload["scope"] = req.scope.model_dump()
    return await trake_services[req.retrieval_database].search_trake(payload)


@app.post("/api/answers/generate")
async def generate_answers(req: AnswerGenerateRequest):
    """The ordered list of up to 100 answers for one query.

    Ranking policy in `app.answer_gen`; the parameters shipped as defaults were
    fitted on the L21-L30 ground-truth queries (`benchmarks/run_answer_gen.py`).
    """
    payload = req.model_dump()
    payload["manual_overrides"] = req.manual_overrides.model_dump()
    payload["scope"] = req.scope.model_dump()
    if req.feedback is not None:
        payload["feedback"] = req.feedback.model_dump()
    try:
        return await answer_services[req.retrieval_database].generate(payload)
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/canvas/palette")
async def canvas_palette():
    """Labels and colours the V-KIS canvas may use.

    Served from the backend so the picker can only offer what the OD run was
    actually prompted with — an icon for a class outside that vocabulary would
    always return zero frames.
    """
    return palette_manifest()


@app.post("/api/search/canvas")
async def search_canvas(req: CanvasSearchRequest):
    """V-KIS canvas search: canvas JSON → OD spatial match + PE text, RRF-fused."""
    if req.retrieval_database != "btc":
        raise HTTPException(
            status_code=400,
            detail=(
                "InfoShot++ đã có PE image + OCR/speech/audio (index v2), nhưng V-KIS "
                "canvas vẫn khoá: chưa có object detection cho keyframe InfoShot++."
            ),
        )
    return await canvas_service.search(req.model_dump())


@app.post("/api/qa/analyze")
async def analyze_qa(req: QaAnalyzeRequest):
    """Run UIT-style two-stage visual QA over already-retrieved candidates.

    The browser supplies canonical keyframe IDs plus optional OCR/ASR/audio cues.
    Image URLs are reconstructed here, preventing arbitrary fetch targets from
    being forwarded to the Colab worker.
    """
    selected_media = media_builders[req.retrieval_database]
    candidates: list[dict] = []
    for index, item in enumerate(req.candidates[: settings.nvila_max_candidates]):
        try:
            canonical = canonical_submit_keyframe_id(item.submit_keyframe_id)
            parsed_id = parse_submit_keyframe_id(canonical)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        cues = []
        for evidence in item.evidence[:8]:
            evidence_type = str(evidence.get("type") or "unknown")[:24]
            text = str(evidence.get("text") or "").strip()[:700]
            if not text and evidence_type == "audio":
                text = str(evidence.get("top1_label") or "").strip()[:120]
            if text:
                cues.append({
                    "type": evidence_type,
                    "text": text,
                    "start": evidence.get("start"),
                    "end": evidence.get("end"),
                })
        candidates.append({
            "candidate_id": f"C{index + 1:02d}",
            "submit_keyframe_id": canonical,
            "video_id": parsed_id.video_id,
            "keyframe_n": parsed_id.keyframe_n,
            "frame_idx": item.frame_idx,
            "pts_time": item.pts_time,
            "retrieval_score": item.retrieval_score,
            "image_url": selected_media.keyframe_url(parsed_id.video_id, parsed_id.keyframe_n),
            "evidence": cues,
        })

    try:
        result = await nvila_qa.analyze(
            req.question.strip(),
            candidates,
            max_answers=req.max_answers,
        )
    except NvilaUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    visual = _normalize_qa_analysis(req.question.strip(), result, candidates, source="nvila")
    grounding_meta = {
        "requested": req.web_grounding,
        "available": settings.mock_mode or settings.has_web_grounding,
        "attempted": False,
        "used": False,
        "model": None,
        "queries": [],
        "sources": [],
        "summary": "",
        "latency_ms": 0.0,
        "search_suggestions_html": "",
        "visual_verification": {
            "attempted": False,
            "used": False,
            "model": None,
            "latency_ms": 0.0,
            "verdicts": [],
            "rejected_answers": [],
            "uncertainty": "",
        },
    }
    if _should_ground_with_web(req.web_grounding, visual, candidates):
        grounding_meta["attempted"] = True
        try:
            grounded_raw = await web_grounding_client.ground(
                req.question.strip(), candidates, visual, max_answers=req.max_answers,
            )
            grounded = _normalize_qa_analysis(
                req.question.strip(),
                grounded_raw,
                candidates,
                source="web",
                sources=grounded_raw.get("sources") or [],
            )
            grounding_meta.update({
                "model": str(grounded_raw.get("model") or settings.deepseek_grounding_model),
                "queries": [str(item)[:500] for item in (grounded_raw.get("queries") or [])[:8]],
                "sources": grounded_raw.get("sources") or [],
                "summary": str(grounded_raw.get("summary") or "")[:1600],
                "latency_ms": _nonnegative_float(grounded_raw.get("latency_ms")),
                "search_suggestions_html": str(grounded_raw.get("search_suggestions_html") or "")[:30000],
            })
            if grounded["candidate_answers"]:
                grounding_meta["visual_verification"]["attempted"] = True
                verification_options = [
                    {
                        "answer": option["answer"],
                        "confidence": option["confidence"],
                        "supporting_candidate_ids": option["supporting_candidate_ids"],
                        "reason": option["reason"],
                        "web_sources": option["web_sources"],
                    }
                    for option in grounded["candidate_answers"]
                ]
                try:
                    verification_raw = await nvila_qa.verify_grounded(
                        req.question.strip(),
                        candidates,
                        verification_options,
                        hotspot_ids=[item["candidate_id"] for item in visual["hotspots"]],
                    )
                    grounded, verification_meta = _apply_visual_verification(
                        grounded, verification_raw, candidates,
                    )
                except NvilaUnavailable as exc:
                    grounded, verification_meta = _apply_visual_verification(
                        grounded, None, candidates,
                    )
                    visual["warnings"].append(f"NVILA pass-3 verification failed: {exc}"[:500])
                grounding_meta["visual_verification"].update(verification_meta)
            grounding_meta["used"] = bool(grounded["candidate_answers"])
            visual = _merge_qa_analyses(visual, grounded, req.max_answers)
        except WebGroundingUnavailable as exc:
            visual["warnings"].append(str(exc)[:500])
    elif req.web_grounding == "on" and not grounding_meta["available"]:
        visual["warnings"].append("Web grounding requested but DEEPSEEK_API_KEY is not configured")
    visual["web_grounding"] = grounding_meta
    visual["total_latency_ms"] = round(
        visual["latency_ms"]
        + grounding_meta["latency_ms"]
        + grounding_meta["visual_verification"]["latency_ms"],
        1,
    )
    return visual


@app.get("/api/keyframes/{submit_keyframe_id:path}")
async def get_keyframe(
    submit_keyframe_id: str, retrieval_database: RetrievalDatabase = "btc"
):
    try:
        canonical = canonical_submit_keyframe_id(submit_keyframe_id)
        parsed = parse_submit_keyframe_id(canonical)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    selected_search = search_services[retrieval_database]
    selected_media = media_builders[retrieval_database]
    record = await selected_search.elastic.get_keyframe(canonical)
    return {
        "image_id": canonical,
        "submit_keyframe_id": canonical,
        "video_id": parsed.video_id,
        "keyframe_n": parsed.keyframe_n,
        "pts_time": (record or {}).get("pts_time"),
        "fps": (record or {}).get("fps"),
        "frame_idx": (record or {}).get("frame_idx"),
        "retrieval_database": retrieval_database,
        "keyframe_url": selected_media.keyframe_url(parsed.video_id, parsed.keyframe_n),
        "video_url": selected_media.video_url(parsed.video_id),
        "found": record is not None,
    }


@app.get("/api/videos/{video_id}/timeline")
async def video_timeline(video_id: str, retrieval_database: RetrievalDatabase = "btc"):
    return await timeline_services[retrieval_database].build(video_id)


@app.post("/api/videos/{video_id}/snap")
async def snap_frame(video_id: str, req: SnapRequest):
    """Snap a raw paused time to the selected profile's nearest keyframe."""
    selected_timeline = timeline_services[req.retrieval_database]
    selected_media = media_builders[req.retrieval_database]
    timeline = await selected_timeline.build(video_id)
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
        "retrieval_database": req.retrieval_database,
        "keyframe_url": selected_media.keyframe_url(video_id, result.keyframe_n),
    }


@app.get("/api/dres/status")
async def dres_status():
    """Connection state of the DRES adapter (never returns the credentials)."""
    health = await dres_client.health()
    return {
        "configured": settings.has_dres,
        "base_url": dres_client.base_url,
        "logged_in": bool(health.get("logged_in")),
        "username": health.get("username") or settings.dres_username,
        "user": dres_client.user,
        "pinned_evaluation_id": settings.dres_evaluation_id,
        "segment_pad_ms": settings.dres_segment_pad_ms,
        "error": health.get("error"),
    }


@app.post("/api/dres/login")
async def dres_login():
    """Force a fresh login (use after the session expired or the host changed)."""
    try:
        user = await dres_client.login(force=True)
    except DresNotConfigured as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DresError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"ok": True, "user": user, "base_url": dres_client.base_url}


@app.get("/api/dres/evaluations")
async def dres_evaluations(force: bool = False):
    """Every visible run with its currently open task and remaining time."""
    try:
        evaluations = await submit_service.overview(force=force)
    except DresNotConfigured as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DresError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "evaluations": evaluations,
        "server_time": await dres_client.server_time(),
        "pinned_evaluation_id": settings.dres_evaluation_id,
    }


@app.get("/api/dres/current-task")
async def dres_current_task(evaluation_id: str | None = None, query_type: str = "T-KIS"):
    """The open task of one run (auto-picked from the query type when omitted)."""
    try:
        resolved = await submit_service.resolve_evaluation(query_type, evaluation_id)
        task, state = await asyncio.gather(
            dres_client.current_task(resolved), dres_client.evaluation_state(resolved)
        )
    except (SubmitFormatError, DresNotConfigured) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DresError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"evaluation_id": resolved, "task": task, "state": state}


@app.get("/api/dres/task-hint")
async def dres_task_hint(evaluation_id: str | None = None, query_type: str = "T-KIS", force: bool = False):
    """The statement of the open task (what the DRES viewer shows the team)."""
    try:
        return await submit_service.task_hint(query_type, evaluation_id, force=force)
    except (SubmitFormatError, DresNotConfigured) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DresError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/submit/preview")
async def submit_preview(req: SubmitRequest):
    """The exact DRES body this submit would send — shown in the submit guard."""
    try:
        return await submit_service.prepare(
            query_type=req.query_type,
            payload=req.payload.model_dump(),
            evaluation_id=req.evaluation_id,
            task_name=req.task_name,
            task_id=req.task_id,
            answer_mode=req.answer_mode,
            pad_ms=req.segment_pad_ms,
        )
    except SubmitFormatError as exc:
        raise HTTPException(status_code=400, detail={"error": "invalid_format", "message": str(exc)}) from exc


@app.post("/api/submit")
async def submit(req: SubmitRequest):
    try:
        entry = await submit_service.submit(
            query_type=req.query_type,
            payload=req.payload.model_dump(),
            evaluation_id=req.evaluation_id,
            task_name=req.task_name,
            task_id=req.task_id,
            answer_mode=req.answer_mode,
            pad_ms=req.segment_pad_ms,
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
    except SubmitFormatError as exc:
        # Wrong format is caught before anything reaches DRES.
        raise HTTPException(
            status_code=400, detail={"error": "invalid_format", "message": str(exc)}
        ) from exc
    return entry


@app.get("/api/submit/history")
async def submit_history(task_id: str | None = None):
    return {"history": submit_service.history(task_id)}


@app.delete("/api/submit/history")
async def clear_submit_history(task_id: str | None = None, ids: str | None = None):
    """Delete local history entries: `?ids=` (comma-separated), `?task_id=`, or all.

    Only the operator's local log: submissions already sent to DRES stand.
    Whatever is removed is backed up next to the history file first.
    """
    id_list = [i for i in (ids or "").split(",") if i.strip()] if ids is not None else None
    return submit_service.clear_history(task_id, id_list)


@app.post("/api/translate")
async def translate_endpoint(req: TranslateRequest):
    en = await translate_vi_to_en(req.text)
    return {"text": req.text, "text_en": en}


def _score01(value, default: float = 0.0) -> float:
    try:
        return round(min(1.0, max(0.0, float(value))), 4)
    except (TypeError, ValueError):
        return default


def _nonnegative_float(value, default: float = 0.0) -> float:
    try:
        return round(max(0.0, float(value)), 1)
    except (TypeError, ValueError):
        return default


_QA_PLACEHOLDERS = {
    "...",
    "short caveat",
    "short verification reason",
    "short visible cue",
    "visual candidate answer",
}


def _is_placeholder(value: object) -> bool:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip(" `\t\r\n.,:;!?\"'").casefold()
    return normalized in _QA_PLACEHOLDERS


def _source_host(value: object) -> str:
    """Bare hostname of a URL (or of a title that already is a domain)."""
    host = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", str(value or "").strip().casefold())
    host = host.split("/")[0].split("?")[0].split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _trusted_web_sources(sources: list[dict] | None) -> list[dict]:
    """Citations the web stage actually opened, de-duplicated."""
    trusted: list[dict] = []
    seen: set[str] = set()
    for item in (sources or [])[:12]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")
        if not url.startswith(("https://", "http://")) or url in seen:
            continue
        seen.add(url)
        trusted.append({"title": str(item.get("title") or "")[:300], "url": url[:2000]})
    return trusted


def _answer_web_sources(raw: dict, trusted: list[dict]) -> list[dict]:
    """Cite only the sources the model attributed to THIS answer.

    The web stage reports one source list for the whole response, so attaching
    all of it to every answer would show answer B's citation next to answer A —
    an operator verifying by clicking the link would be misled. The model
    declares the domains behind each answer; anything that cannot be matched
    back to a visited source is cited nowhere rather than everywhere.
    """
    if not trusted:
        return []
    declared = {
        _source_host(item)
        for item in (raw.get("source_domains") or raw.get("source_urls") or [])[:12]
    }
    declared.discard("")
    if not declared:
        return []
    return [
        source
        for source in trusted
        if _source_host(source["url"]) in declared or _source_host(source["title"]) in declared
    ]


def _normalize_qa_analysis(
    question: str,
    result: dict,
    candidates: list[dict],
    *,
    source: str = "nvila",
    sources: list[dict] | None = None,
) -> dict:
    """Keep only grounded IDs and attach canonical frame metadata to NVILA output."""
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    trusted_sources = _trusted_web_sources(sources)
    hotspots = []
    for raw in result.get("hotspots") or []:
        candidate = by_id.get(str(raw.get("candidate_id") or "").upper())
        if not candidate:
            continue
        relevance = _score01(raw.get("relevance"))
        support_text = str(raw.get("answer_support") or "")[:500]
        if relevance <= 0 or _is_placeholder(support_text):
            continue
        hotspots.append({
            "candidate_id": candidate["candidate_id"],
            "submit_keyframe_id": candidate["submit_keyframe_id"],
            "video_id": candidate["video_id"],
            "keyframe_n": candidate["keyframe_n"],
            "frame_idx": candidate["frame_idx"],
            "pts_time": candidate["pts_time"],
            "keyframe_url": candidate["image_url"],
            "relevance": relevance,
            "answer_support": support_text,
        })
    hotspots.sort(key=lambda item: -item["relevance"])

    answers = []
    for raw in result.get("candidate_answers") or []:
        answer = str(raw.get("answer") or "").strip()[:500]
        confidence = _score01(raw.get("confidence"))
        if not answer or _is_placeholder(answer) or confidence <= 0:
            continue
        supporting_ids = []
        supporting_frames = []
        for raw_id in raw.get("supporting_candidate_ids") or []:
            candidate = by_id.get(str(raw_id).upper())
            if candidate and candidate["candidate_id"] not in supporting_ids:
                supporting_ids.append(candidate["candidate_id"])
                supporting_frames.append({
                    "candidate_id": candidate["candidate_id"],
                    "submit_keyframe_id": candidate["submit_keyframe_id"],
                    "video_id": candidate["video_id"],
                    "frame_idx": candidate["frame_idx"],
                    "pts_time": candidate["pts_time"],
                    "keyframe_url": candidate["image_url"],
                })
        # Candidate answers without a known supporting frame are ungrounded and
        # must never reach the operator as selectable answers.
        if not supporting_frames:
            continue
        reason = str(raw.get("reason") or "")[:700]
        answers.append({
            "answer": answer,
            "confidence": confidence,
            "supporting_candidate_ids": supporting_ids,
            "supporting_frames": supporting_frames,
            "reason": "" if _is_placeholder(reason) else reason,
            "source": source,
            "web_sources": _answer_web_sources(raw, trusted_sources),
        })
    answers.sort(key=lambda item: -item["confidence"])
    if trusted_sources and len(answers) == 1 and not answers[0]["web_sources"]:
        # A single grounded answer is the unambiguous owner of the whole search,
        # so the response-level citations can be attributed to it safely.
        answers[0]["web_sources"] = list(trusted_sources)

    best_id = str(result.get("best_candidate_id") or "").upper()
    best_candidate = by_id.get(best_id)
    proposed_best = str(result.get("best_answer") or "").strip()[:500]
    grounded_best = next((item for item in answers if item["answer"] == proposed_best), None)
    if grounded_best is None and answers:
        grounded_best = answers[0]
    best_answer = grounded_best["answer"] if grounded_best else ""
    if grounded_best and (best_candidate is None or best_id not in grounded_best["supporting_candidate_ids"]):
        best_id = grounded_best["supporting_candidate_ids"][0]
        best_candidate = by_id[best_id]
    if best_candidate is None and hotspots:
        best_id = hotspots[0]["candidate_id"]
        best_candidate = by_id.get(best_id)

    return {
        "question": question,
        "model": str(result.get("model") or "NVILA-8B"),
        "mode": str(result.get("mode") or ("mock" if settings.mock_mode else "live")),
        "answerable": bool(result.get("answerable", bool(answers))) and bool(answers),
        "best_answer": best_answer,
        "best_candidate_id": best_id if best_candidate else None,
        "best_submit_keyframe_id": best_candidate["submit_keyframe_id"] if best_candidate else None,
        "candidate_answers": answers,
        "hotspots": hotspots,
        "uncertainty": "" if _is_placeholder(result.get("uncertainty")) else str(result.get("uncertainty") or "")[:700],
        "candidate_count": len(candidates),
        "latency_ms": _nonnegative_float(result.get("latency_ms")),
        "cached": bool(result.get("cached", False)),
        "warnings": [str(w)[:300] for w in (result.get("warnings") or [])[:8]],
    }


def _answer_key(answer: str) -> str:
    return "".join(char for char in answer.casefold() if char.isalnum())


# A pass-1 answer that pass 3 contradicts stays selectable but must sink below
# every option that is not under dispute.
_CONTRADICTED_VISUAL_CAP = 0.30


def _apply_visual_verification(
    grounded: dict,
    verification_raw: dict | None,
    candidates: list[dict],
) -> tuple[dict, dict]:
    """Apply NVILA pass-3 verdicts to web answers.

    Contradicted answers are removed. Insufficient/unverified answers remain
    available to the operator but cannot outrank strongly supported visual
    answers solely on web-search confidence.
    """
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    option_keys = {
        _answer_key(option["answer"]): option
        for option in grounded.get("candidate_answers") or []
    }
    verdict_by_key: dict[str, dict] = {}
    normalized_verdicts = []
    for raw in (verification_raw or {}).get("verdicts") or []:
        key = _answer_key(str(raw.get("answer") or ""))
        if not key or key not in option_keys or key in verdict_by_key:
            continue
        status = str(raw.get("status") or "insufficient").strip().casefold()
        if status not in {"supported", "contradicted", "insufficient"}:
            status = "insufficient"
        confidence = _score01(raw.get("visual_confidence"))
        supporting_ids = []
        for raw_id in raw.get("supporting_candidate_ids") or []:
            candidate_id = str(raw_id).upper()
            if candidate_id in by_id and candidate_id not in supporting_ids:
                supporting_ids.append(candidate_id)
        if status in {"supported", "contradicted"} and (confidence <= 0 or not supporting_ids):
            status = "insufficient"
        reason = str(raw.get("reason") or "").strip()[:700]
        verdict = {
            "answer": option_keys[key]["answer"],
            "status": status,
            "visual_confidence": confidence,
            "supporting_candidate_ids": supporting_ids,
            "reason": "" if _is_placeholder(reason) else reason,
        }
        verdict_by_key[key] = verdict
        normalized_verdicts.append(verdict)

    kept = []
    rejected = []
    rejected_verdicts: dict[str, dict] = {}
    verifier_returned = verification_raw is not None
    for option in grounded.get("candidate_answers") or []:
        key = _answer_key(option["answer"])
        verdict = verdict_by_key.get(key)
        if verdict is None:
            verdict = {
                "answer": option["answer"],
                "status": "insufficient" if verifier_returned else "unverified",
                "visual_confidence": 0.0,
                "supporting_candidate_ids": [],
                "reason": (
                    "NVILA did not return a verdict for this option."
                    if verifier_returned else "NVILA pass-3 verification was unavailable."
                ),
            }
            normalized_verdicts.append(verdict)
        option["visual_verification"] = verdict
        if verdict["status"] == "contradicted":
            rejected.append(option["answer"])
            rejected_verdicts[key] = verdict
            continue
        if verdict["status"] == "supported":
            option["confidence"] = round(
                0.70 * option["confidence"] + 0.30 * verdict["visual_confidence"], 4,
            )
            verified_ids = verdict["supporting_candidate_ids"]
            if verified_ids:
                option["supporting_candidate_ids"] = verified_ids
                option["supporting_frames"] = [
                    {
                        "candidate_id": candidate_id,
                        "submit_keyframe_id": by_id[candidate_id]["submit_keyframe_id"],
                        "video_id": by_id[candidate_id]["video_id"],
                        "frame_idx": by_id[candidate_id]["frame_idx"],
                        "pts_time": by_id[candidate_id]["pts_time"],
                        "keyframe_url": by_id[candidate_id]["image_url"],
                    }
                    for candidate_id in verified_ids
                ]
        else:
            option["confidence"] = min(option["confidence"], 0.49 if verifier_returned else 0.40)
        kept.append(option)

    kept.sort(key=lambda item: -item["confidence"])
    grounded["candidate_answers"] = kept
    grounded["_rejected_verdicts"] = rejected_verdicts
    grounded["answerable"] = bool(kept)
    grounded["best_answer"] = kept[0]["answer"] if kept else ""
    if kept:
        first_frame = kept[0]["supporting_frames"][0]
        grounded["best_candidate_id"] = first_frame["candidate_id"]
        grounded["best_submit_keyframe_id"] = first_frame["submit_keyframe_id"]
    else:
        grounded["best_candidate_id"] = None
        grounded["best_submit_keyframe_id"] = None

    return grounded, {
        "used": bool(verdict_by_key),
        "model": str((verification_raw or {}).get("model") or "") or None,
        "latency_ms": _nonnegative_float((verification_raw or {}).get("latency_ms")),
        "verdicts": normalized_verdicts[:8],
        "rejected_answers": rejected[:8],
        "uncertainty": str((verification_raw or {}).get("uncertainty") or "")[:700],
    }


def _merge_qa_analyses(visual: dict, grounded: dict, max_answers: int) -> dict:
    """Merge visual and web alternatives without allowing web to alter frame IDs."""
    merged: dict[str, dict] = {}
    rejected_verdicts = grounded.get("_rejected_verdicts") or {}
    for option in [*(visual.get("candidate_answers") or []), *(grounded.get("candidate_answers") or [])]:
        key = _answer_key(option["answer"])
        if not key:
            continue
        contradiction = rejected_verdicts.get(key)
        if contradiction is not None:
            if option.get("source") != "nvila":
                # A web claim NVILA contradicted carries no visual grounding of
                # its own, so it is dropped outright.
                continue
            # Pass 1 inspected every candidate; pass 3 only sees the hotspot ∪
            # web-cited subset. A verdict from the narrower view may therefore
            # never have looked at the frame this answer was grounded on, so it
            # demotes and labels the option instead of deleting it silently.
            option = {
                **option,
                "confidence": min(option["confidence"], _CONTRADICTED_VISUAL_CAP),
                "visual_verification": contradiction,
            }
        current = merged.get(key)
        if current is None:
            merged[key] = {
                **option,
                "supporting_candidate_ids": list(option.get("supporting_candidate_ids") or []),
                "supporting_frames": list(option.get("supporting_frames") or []),
                "web_sources": list(option.get("web_sources") or []),
            }
            continue
        directly_grounded = current.get("source") in {"nvila", "hybrid"}
        current["confidence"] = max(current["confidence"], option["confidence"])
        if current.get("source") != option.get("source"):
            current["source"] = "hybrid"
        verdict = option.get("visual_verification")
        # A pass-3 downgrade describes the WEB claim. When pass 1 independently
        # grounded the same answer on the frames, surfacing "insufficient visual
        # evidence" next to a high visual confidence contradicts itself at the
        # exact moment the operator picks an answer — so only a reinforcing or
        # contradicting verdict is shown for a directly-grounded option.
        if verdict and (verdict["status"] in {"supported", "contradicted"} or not directly_grounded):
            current["visual_verification"] = verdict
        for candidate_id in option.get("supporting_candidate_ids") or []:
            if candidate_id not in current["supporting_candidate_ids"]:
                current["supporting_candidate_ids"].append(candidate_id)
        known_frames = {frame["candidate_id"] for frame in current["supporting_frames"]}
        current["supporting_frames"].extend(
            frame for frame in (option.get("supporting_frames") or [])
            if frame["candidate_id"] not in known_frames
        )
        known_urls = {item["url"] for item in current["web_sources"]}
        current["web_sources"].extend(
            item for item in (option.get("web_sources") or []) if item["url"] not in known_urls
        )
        if option.get("reason") and option["reason"] not in current.get("reason", ""):
            current["reason"] = " · ".join(filter(None, [current.get("reason"), option["reason"]]))[:700]

    answers = sorted(merged.values(), key=lambda item: -item["confidence"])[:max_answers]
    best = answers[0] if answers else None
    best_frame = best["supporting_frames"][0] if best and best["supporting_frames"] else None
    visual["candidate_answers"] = answers
    visual["answerable"] = bool(answers)
    visual["best_answer"] = best["answer"] if best else ""
    visual["best_candidate_id"] = best_frame["candidate_id"] if best_frame else None
    visual["best_submit_keyframe_id"] = best_frame["submit_keyframe_id"] if best_frame else None
    if grounded.get("uncertainty"):
        visual["uncertainty"] = " · ".join(filter(None, [visual.get("uncertainty"), grounded["uncertainty"]]))[:700]
    return visual


# Word-bounded so short terms cannot fire on unrelated syllables: a plain
# substring test made "ai" match "hai"/"tai"/"vai" and spend a web search plus
# an NVILA pass-3 round trip on questions that resolve no named entity at all.
_ENTITY_QUESTION_RE = re.compile(
    r"\b(?:ai|tên|người nào|thương hiệu|nhãn hiệu|cửa hàng|công ty|tổ chức"
    r"|địa danh|quốc gia|thành phố|name|who|which brand|which company"
    r"|which city|which country)\b",
    re.IGNORECASE,
)


def _should_ground_with_web(mode: str, visual: dict, candidates: list[dict]) -> bool:
    if mode == "off" or not (settings.mock_mode or settings.has_web_grounding):
        return False
    if mode == "on":
        return True
    # Keep the existing deterministic visual fixture stable; explicit `on`
    # exercises the mock web-search branch in dedicated tests.
    if settings.mock_mode:
        return False
    answers = visual.get("candidate_answers") or []
    top_confidence = max((answer.get("confidence", 0.0) for answer in answers), default=0.0)
    if not visual.get("answerable") or top_confidence < settings.deepseek_grounding_auto_threshold:
        return True
    question = str(visual.get("question") or "")
    has_text_clue = any(candidate.get("evidence") for candidate in candidates)
    return has_text_clue and bool(_ENTITY_QUESTION_RE.search(question))


@app.post("/api/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...), translate: bool = True):
    """Whisper speech-to-text for cross-browser voice input.

    Always returns what was actually said (`text`). `text_en` is the VI→EN
    translation and is `null` when the console's translate tick box is off, so
    dictating a query never silently rewrites the operator's words.
    """
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


# --- Bundled UI -------------------------------------------------------------
# The packaged app serves the built console from the same origin as the API: one
# process, one port, no CORS and no second server for the operator to start. A
# source checkout has no build output and skips this, leaving `npm run dev` and
# its /api proxy in charge.


class _SpaStaticFiles(StaticFiles):
    """Static files with a single-page fallback for unknown paths.

    API 404s are left alone — answering them with index.html would turn a typo'd
    endpoint into a 200 and an unreadable client-side error.
    """

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or path.startswith("api/"):
                raise
            return await super().get_response("index.html", scope)


_ui_dir = paths.static_dir()
if _ui_dir is not None:
    app.mount("/", _SpaStaticFiles(directory=_ui_dir, html=True), name="ui")
