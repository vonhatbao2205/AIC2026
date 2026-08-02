"""AIC26 retrieval backend — FastAPI app.

Endpoints:
  GET  /api/health
  POST /api/query/parse
  POST /api/search
  POST /api/search/trake
  POST /api/qa/analyze
  GET  /api/keyframes/{submit_keyframe_id:path}
  GET  /api/videos/{video_id}/timeline
  POST /api/videos/{video_id}/snap
  POST /api/submit
  GET  /api/submit/history

Secrets stay server-side; responses never include Elastic/Milvus/NVIDIA creds.
"""
from __future__ import annotations

import asyncio
import re

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .adapters.deepseek_grounding import DeepSeekGroundingClient, WebGroundingUnavailable
from .adapters.nvila_client import NvilaQaClient, NvilaUnavailable
from .canvas import palette_manifest
from .config import get_settings
from .identity import canonical_submit_keyframe_id, parse_submit_keyframe_id
from .media import MediaUrlBuilder
from .models import (
    CanvasSearchRequest,
    ParseRequest,
    QaAnalyzeRequest,
    SearchRequest,
    SimpleSearchRequest,
    SnapRequest,
    SubmitRequest,
    TranslateRequest,
    TrakeSearchRequest,
)
from .translate import translate_vi_to_en
from .services.canvas_service import CanvasService
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
canvas_service = CanvasService(settings, search_service)
timeline_service = TimelineService(settings)
submit_service = SubmitService(settings)
media = MediaUrlBuilder(settings.media_base_url)
nvila_qa = NvilaQaClient(settings)
web_grounding_client = DeepSeekGroundingClient(settings)


@app.get("/api/health")
async def health():
    elastic, milvus, pe, nvila, grounding, objects = await asyncio.gather(
        search_service.elastic.health(),
        search_service.milvus.health(),
        search_service.pe.health(),
        nvila_qa.health(),
        web_grounding_client.health(),
        canvas_service.objects.health(),
    )
    services = {
        "elastic": elastic,
        "milvus": milvus,
        "pe_encoder": pe,
        "nvila_qa": nvila,
        "web_grounding": grounding,
        "object_index": objects,
    }
    # NVILA is an optional QA accelerator: a stopped Colab session must not mark
    # the core retrieval stack unhealthy.
    ok = settings.mock_mode or all(s.get("ok") for s in (elastic, milvus, pe))
    has_live_nvila = settings.has_nvila and not settings.mock_mode and bool(nvila.get("ok"))
    has_qa_nvila = settings.mock_mode or has_live_nvila
    has_qa_visual_verification = settings.mock_mode or (
        has_live_nvila and bool(nvila.get("visual_verification_pass"))
    )
    return {
        "ok": ok,
        "mode": "mock" if settings.mock_mode else "live",
        "services": services,
        "capabilities": {
            "llm_query_parser": settings.has_llm and not settings.mock_mode,
            "dres_submit": settings.has_dres,
            "audio_vector_search": settings.has_glap and not settings.mock_mode,
            # NVILA currently reranks/grounds only the QA candidate pack; it is
            # not a general reranker for T-KIS/V-KIS/TRAKE retrieval results.
            "vlm_rerank": False,
            # The worker returns short verification reasons, never hidden or
            # free-form chain-of-thought traces.
            "vlm_cot": False,
            "qa_nvila": has_qa_nvila,
            "qa_hotspot_prediction": has_qa_nvila,
            "qa_candidate_answers": has_qa_nvila,
            "qa_visual_verification": has_qa_visual_verification,
            "qa_web_grounding": settings.mock_mode or settings.has_web_grounding,
            # V-KIS canvas: OD spatial match needs the object index to answer.
            "canvas_object_search": settings.mock_mode or bool(objects.get("ok")),
        },
        "warnings": [
            f"{name} unreachable: {s.get('error')}"
            for name, s in services.items()
            if not s.get("ok") and s.get("mode") != "disabled"
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
    return await canvas_service.search(req.model_dump())


@app.post("/api/qa/analyze")
async def analyze_qa(req: QaAnalyzeRequest):
    """Run UIT-style two-stage visual QA over already-retrieved candidates.

    The browser supplies canonical keyframe IDs plus optional OCR/ASR/audio cues.
    Image URLs are reconstructed here, preventing arbitrary fetch targets from
    being forwarded to the Colab worker.
    """
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
            "image_url": media.keyframe_url(parsed_id.video_id, parsed_id.keyframe_n),
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
