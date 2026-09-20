"""V-KIS canvas search: canvas JSON → OD spatial match + PE text (+ raster), fused.

Channels, all derived from the same drawing:

  `object_layout` — OD candidates from Elastic, then one-to-one Hungarian
                    assignment per frame (app.canvas). Primary channel: the only
                    one that knows *where* things are.
  `image_pe`      — rule-generated English sentences through the existing PE →
                    Milvus path, which recovers frames whose objects the detector
                    missed or never had in its prompted vocabulary.
  `canvas_image`  — the rendered drawing through PE's image encoder. Optional and
                    low-weighted: it is the only way to express something with no
                    OD label and no easy words (a rice field, an empty sky), but a
                    sketch sits far outside the photo distribution PE was trained
                    on, so it nudges the ranking rather than driving it.

They are fused with the same RRF the main search uses, so canvas results grade
into the rest of the console (group-by-video, evidence, submit guard) instead of
being a parallel universe.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from ..adapters.object_elastic import ObjectElasticClient
from ..adapters.pe_encoder import PeImageEncoderMissing
from ..canvas import CanvasQuery, canvas_to_queries, grid7_of, match_canvas, parse_canvas
from ..config import Settings
from ..fusion import group_by_video, reciprocal_rank_fusion
from ..media import MediaUrlBuilder
from ..types import Channel, ChannelHit, Evidence
from .search_service import SearchService

# The layout channel outranks the text channel: it is the only one that can tell
# "person on the left of a red car" from "person and red car somewhere". PE text
# still earns its weight on frames the detector under-covered. The raster stays
# at the low end of the 0.15–0.25 band sketch-retrieval work suggests.
DEFAULT_WEIGHTS: dict[str, float] = {
    "object_layout": 1.0, "image_pe": 0.6, "canvas_image": 0.2,
}

# Candidate pool per canvas. Assignment is O(objects² × detections) per frame and
# objects stay in the single digits, so a few hundred frames costs milliseconds.
DEFAULT_CANDIDATE_POOL = 400


class CanvasService:
    def __init__(self, settings: Settings, search_service: SearchService | None = None):
        self.s = settings
        self.search_service = search_service or SearchService(settings)
        self.objects = ObjectElasticClient(settings)
        self.media = MediaUrlBuilder(settings.keyframe_media_base_url, settings.video_media_base_url)

    async def search(self, req: dict[str, Any]) -> dict[str, Any]:
        t_total = time.perf_counter()
        canvas = parse_canvas(req.get("canvas") or {})
        if not canvas.objects and not canvas.action_text:
            return {
                "canvas": _canvas_echo(canvas, []),
                "groups": [],
                "latency_ms": {"total_ms": 0.0, "channels": {}},
                "warnings": ["Canvas is empty; drag at least one object into the drawing area."],
                "mode": "mock" if self.s.mock_mode else "live",
            }

        queries_en = canvas_to_queries(canvas)
        top_k = int(req.get("top_k") or 100)
        pool = int(req.get("candidate_pool") or DEFAULT_CANDIDATE_POOL)
        # The drawing is the query, so an auto scope has no text to read: only an
        # explicit manual selection can narrow a canvas search.
        scope = self.search_service.resolve_scope(req.get("scope"), query=canvas.action_text)

        layout_task = asyncio.create_task(
            self._run_object_layout(canvas, pool=pool, top_k=top_k, categories=scope.categories)
        )
        pe_task = asyncio.create_task(
            self.search_service._run_image_pe({"queries_en": queries_en}, top_k, scope.categories)
            if queries_en
            else _no_hits()
        )
        raster_task = asyncio.create_task(
            self._run_canvas_image(canvas.image, top_k, scope.categories)
            if canvas.image
            else _no_hits()
        )

        latency: dict[str, Any] = {"channels": {}}
        warnings: list[str] = []
        channel_hits: dict[Channel, list[ChannelHit]] = {}
        for name, task in (
            ("object_layout", layout_task),
            ("image_pe", pe_task),
            ("canvas_image", raster_task),
        ):
            try:
                hits, ms = await task
            except Exception as exc:  # noqa: BLE001 - one dead channel must not kill the search
                hits, ms = [], 0.0
                warnings.append(f"{name} channel unavailable: {exc}")
            channel_hits[name] = hits
            latency["channels"][name] = round(ms, 1)

        weights = dict(DEFAULT_WEIGHTS)
        weights.update({k: float(v) for k, v in (req.get("weights") or {}).items() if k in weights})

        t_fuse = time.perf_counter()
        fused = reciprocal_rank_fusion(channel_hits, weights=weights)
        await self.search_service._enrich_pts(fused)
        groups = group_by_video(fused)
        latency["fusion_ms"] = round((time.perf_counter() - t_fuse) * 1000, 1)
        latency["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)

        return {
            "canvas": _canvas_echo(canvas, queries_en),
            "scope": scope.to_dict(),
            "groups": [
                self.search_service._serialize_group(group) for group in groups[: int(req.get("max_videos") or 50)]
            ],
            "latency_ms": latency,
            "warnings": warnings,
            "mode": "mock" if self.s.mock_mode else "live",
        }

    async def _run_object_layout(
        self, canvas: CanvasQuery, *, pool: int, top_k: int, categories: tuple[str, ...] = ()
    ) -> tuple[list[ChannelHit], float]:
        """OD candidates → per-frame Hungarian assignment → ranked hits."""
        t0 = time.perf_counter()
        if not canvas.objects:
            return [], 0.0
        colors = {obj.label: obj.color for obj in canvas.objects if obj.color}
        positions = {obj.label: grid7_of(*obj.center) for obj in canvas.objects}
        candidates = await self.objects.search_canvas_candidates(
            canvas.labels,
            colors=colors,
            positions=positions,
            exclude_labels=canvas.exclude_labels,
            categories=categories,
            size=pool,
        )

        scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for frame in candidates:
            result = match_canvas(canvas, frame.get("detections") or [])
            if result["score"] <= 0:
                continue
            scored.append((result["score"], frame, result))
        scored.sort(key=lambda item: (-item[0], item[1]["submit_keyframe_id"]))

        hits = [
            ChannelHit(
                channel="object_layout",
                submit_keyframe_id=frame["submit_keyframe_id"],
                video_id=frame["video_id"],
                keyframe_n=int(frame["keyframe_n"]),
                score=score,
                rank=rank,
                evidence=Evidence(
                    type="object_layout",
                    score=score,
                    text=_match_summary(result),
                    extra={
                        "matches": result["matches"],
                        "missing": result["missing"],
                        "coverage": result["coverage"],
                        "excluded_hits": result["excluded_hits"],
                        "frame_size": {"width": frame.get("width"), "height": frame.get("height")},
                    },
                ),
            )
            for rank, (score, frame, result) in enumerate(scored[:top_k])
        ]
        return hits, (time.perf_counter() - t0) * 1000


    async def _run_canvas_image(
        self, image: str, top_k: int, categories: tuple[str, ...] = ()
    ) -> tuple[list[ChannelHit], float]:
        """Rendered canvas → PE image embedding → Milvus kNN."""
        t0 = time.perf_counter()
        try:
            vectors = await self.search_service.pe.encode_image([image])
        except PeImageEncoderMissing:
            raise
        except Exception as exc:  # noqa: BLE001 - normalized by the caller's warning
            raise RuntimeError(f"PE image encode failed: {exc}") from exc
        if not vectors:
            return [], (time.perf_counter() - t0) * 1000
        raw = await asyncio.to_thread(
            self.search_service.milvus.search_image,
            vectors[0],
            top_k=top_k,
            categories=categories,
        )
        hits = [
            ChannelHit(
                channel="canvas_image",
                submit_keyframe_id=item["submit_keyframe_id"],
                video_id=item["video_id"],
                keyframe_n=int(item["keyframe_n"]),
                score=float(item["score"]),
                rank=rank,
                evidence=Evidence(
                    type="canvas_image",
                    score=float(item["score"]),
                    text="drawing match (PE image)",
                ),
            )
            for rank, item in enumerate(raw)
        ]
        return hits, (time.perf_counter() - t0) * 1000


async def _no_hits() -> tuple[list[ChannelHit], float]:
    return [], 0.0


def _match_summary(result: dict[str, Any]) -> str:
    """One line the operator can read at a glance in the evidence card."""
    parts = [
        f"{match['label']}"
        + (f" · {match['dominant_color']}" if match.get("dominant_color") else "")
        + f" @{match.get('position') or '?'}"
        for match in result["matches"]
    ]
    text = ", ".join(parts) if parts else "no object matched"
    if result["missing"]:
        text += f" · missing: {', '.join(result['missing'])}"
    return text


def _canvas_echo(canvas: CanvasQuery, queries_en: list[str]) -> dict[str, Any]:
    """What the backend understood, so the UI can show the generated PE text."""
    return {
        "mode": canvas.mode,
        # The raster is echoed as a flag, never as bytes: it can be megabytes and
        # the browser already has it.
        "has_image": bool(canvas.image),
        "objects": [
            {
                "id": obj.id,
                "label": obj.label,
                "bbox": list(obj.bbox),
                "color": obj.color,
                "required": obj.required,
            }
            for obj in canvas.objects
        ],
        "exclude_labels": canvas.exclude_labels,
        "action_text": canvas.action_text,
        "queries_en": queries_en,
    }
