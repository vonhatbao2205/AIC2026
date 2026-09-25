"""V-KIS sketch search: drawing → PE-Core image embedding → the profile's PE Milvus.

One channel, `canvas_image`: the flattened drawing is encoded by PE-Core-G14's
image tower — the same model and preprocessing that produced the keyframe
vectors — and searched by cosine in the SELECTED profile's collection (BTC
`aic26_image_peg14_v1` or the InfoShot++ one), so a sketch works wherever PE
image search does. No object detection is involved.

Before searching, the query loses the direction PE gives blank (uniform) images,
unless the request turns `suppress_blank` off: otherwise a mostly-flat sketch —
which is what sketches are — ranks title cards and fades first (see app.canvas).

The hits still go through the same RRF, pts enrichment and group-by-video as the
main search, so canvas results behave like any other result set in the console
(timeline, evidence, submit guard) instead of being a parallel universe.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from ..adapters.pe_encoder import PeImageEncoderMissing
from ..canvas import (
    BLANK_REFERENCE_IMAGES,
    BLANK_SUPPRESSION,
    blank_direction,
    parse_canvas_image,
    without_blank_direction,
)
from ..config import Settings
from ..fusion import group_by_video, reciprocal_rank_fusion
from ..types import ChannelHit, Evidence
from .search_service import SearchService


class CanvasService:
    def __init__(self, settings: Settings, search_service: SearchService | None = None):
        self.s = settings
        self.search_service = search_service or SearchService(settings)
        # Encoded once per service (a config reload builds a new one) and then
        # reused: the references never change for a given PE server.
        self._blank_direction: list[float] | None = None

    async def search(self, req: dict[str, Any]) -> dict[str, Any]:
        t_total = time.perf_counter()
        mode = "mock" if self.s.mock_mode else "live"
        image = parse_canvas_image((req.get("canvas") or {}).get("image"))
        if not image:
            return {
                "canvas": {"has_image": False},
                "groups": [],
                "latency_ms": {"total_ms": 0.0, "channels": {}},
                "warnings": ["The sketch is empty or is not an inline PNG/JPEG image."],
                "mode": mode,
            }

        top_k = int(req.get("top_k") or 100)
        suppress_blank = bool(req.get("suppress_blank", True))
        # The drawing is the query, so an auto scope has no text to read: only an
        # explicit manual selection can narrow a sketch search.
        scope = self.search_service.resolve_scope(req.get("scope"), query="")

        warnings: list[str] = []
        latency: dict[str, Any] = {"channels": {}}
        try:
            hits, ms = await self._run_canvas_image(
                image, top_k, scope.categories, suppress_blank=suppress_blank
            )
        except PeImageEncoderMissing as exc:
            hits, ms = [], 0.0
            warnings.append(str(exc))
        except Exception as exc:  # noqa: BLE001 - reported to the operator, never a 500
            hits, ms = [], 0.0
            warnings.append(f"Sketch search unavailable: {exc}")
        latency["channels"]["canvas_image"] = round(ms, 1)

        t_fuse = time.perf_counter()
        fused = reciprocal_rank_fusion({"canvas_image": hits})
        await self.search_service._enrich_pts(fused)
        await self.search_service._enrich_overlay(fused)
        groups = group_by_video(fused)
        latency["fusion_ms"] = round((time.perf_counter() - t_fuse) * 1000, 1)
        latency["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)

        return {
            # The raster is echoed as a flag, never as bytes: it can be megabytes
            # and the browser already has it.
            "canvas": {"has_image": True, "suppress_blank": suppress_blank},
            "retrieval_database": self.s.retrieval_database,
            "scope": scope.to_dict(),
            "groups": [
                self.search_service._serialize_group(group) for group in groups[: int(req.get("max_videos") or 50)]
            ],
            "latency_ms": latency,
            "warnings": warnings,
            "mode": mode,
        }

    async def _run_canvas_image(
        self,
        image: str,
        top_k: int,
        categories: tuple[str, ...] = (),
        *,
        suppress_blank: bool = True,
    ) -> tuple[list[ChannelHit], float]:
        """Rendered drawing → PE image embedding (minus the blank direction) → Milvus kNN."""
        t0 = time.perf_counter()
        # The references ride in the same request as the sketch (the server takes
        # up to 8 images), so the first search pays no extra round trip.
        need_references = suppress_blank and self._blank_direction is None
        batch = [image, *BLANK_REFERENCE_IMAGES] if need_references else [image]
        try:
            vectors = await self.search_service.pe.encode_image(batch)
        except PeImageEncoderMissing:
            raise
        except Exception as exc:  # noqa: BLE001 - normalized by the caller's warning
            raise RuntimeError(f"PE image encode failed: {exc}") from exc
        if not vectors:
            return [], (time.perf_counter() - t0) * 1000
        query = vectors[0]
        if need_references and len(vectors) == len(batch):
            self._blank_direction = blank_direction(vectors[1:])
        if suppress_blank and self._blank_direction is not None:
            query = without_blank_direction(query, self._blank_direction, BLANK_SUPPRESSION)
        raw = await asyncio.to_thread(
            self.search_service.milvus.search_image,
            query,
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
                    text="sketch match (PE image)",
                ),
            )
            for rank, item in enumerate(raw)
        ]
        return hits, (time.perf_counter() - t0) * 1000
