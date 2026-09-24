"""Turn one query into the ordered answer list the submission CSV is built from.

Thin on purpose: the ranking policy lives in `app.answer_gen`, this only decides
which retrieval path feeds it (grouped frames for T-KIS/V-KIS/QA, assembled
sequences for TRAKE) and shapes the result for the console.

A caller that already has a result on screen can pass `groups`/`sequences`
straight in — the operator's scope, feedback and channel overrides are already
baked into those, and re-running the search would silently throw that work away.
"""
from __future__ import annotations

import time
from typing import Any

from ..answer_gen import (
    MAX_ANSWERS,
    AnswerGenParams,
    anchor_budget,
    build_pools,
    complete_sequences,
    generate_from_pools,
    generate_trake_rows,
)
from ..avs import AVS_PARAMS, generate_avs
from ..config import Settings
from ..media import MediaUrlBuilder
from .search_service import SearchService
from .trake_service import TrakeService

#: Retrieval depth used when the caller does not pin one. Deeper than the
#: console's default because the tail of the list is insurance, not precision:
#: R@100 can only be earned by frames that made it into the pool at all.
DEFAULT_TOP_K = 400
DEFAULT_MAX_VIDEOS = 100


class AnswerService:
    def __init__(self, settings: Settings, search: SearchService, trake: TrakeService):
        self.s = settings
        self.search = search
        self.trake = trake
        self.media = MediaUrlBuilder(settings.keyframe_media_base_url, settings.video_media_base_url)

    async def generate(self, req: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        params = AnswerGenParams.from_dict(req.get("params"))
        limit = max(1, min(int(req.get("limit") or MAX_ANSWERS), MAX_ANSWERS))
        query_type = str(req.get("query_type_hint") or "T-KIS")
        answer_text = (req.get("answer_text") or "").strip()
        taken = [
            (str(t.get("video_id") or ""), [int(f) for f in (t.get("frames") or [])])
            for t in (req.get("taken") or [])
        ]
        taken = [(video, frames) for video, frames in taken if video and frames]

        groups = req.get("groups")
        sequences = req.get("sequences")
        reused = bool(groups or sequences)
        scope: dict[str, Any] | None = None
        warnings: list[str] = []
        latency: dict[str, Any] = {}

        if query_type == "TRAKE":
            event_count = req.get("event_count") or 0
            if not sequences:
                payload = self._search_payload(req, default_top_k=max(DEFAULT_TOP_K, 400))
                result = await self.trake.search_trake(payload)
                sequences = result.get("sequences") or []
                scope = result.get("scope")
                # Only a fallback: the statement the client parsed is the authority
                # on how many moments the row must carry.
                event_count = event_count or len(result.get("events") or [])
            sequences = sequences or []
            if not event_count:
                event_count = max((len(s.get("frames") or []) for s in sequences), default=0)
            usable = complete_sequences(sequences, event_count)
            rows = generate_trake_rows(
                sequences,
                params,
                limit=limit,
                event_count=event_count,
                taken=[(video, tuple(frames)) for video, frames in taken],
            )
            answers = [self._decorate_sequence(row, answer_text) for row in rows]
            diagnostics = {
                "n_sequences": len(sequences),
                "n_complete": len(usable),
                "event_count": event_count,
            }
            dropped = len(sequences) - len(usable)
            if dropped:
                warnings.append(
                    f"Dropped {dropped}/{len(sequences)} sequences with fewer than {event_count} events; "
                    "each TRAKE row must have the required frame count; missing frames invalidate the submission file."
                )
            if not usable:
                warnings.append(
                    f"No sequence contains all {event_count} events, so no answers were generated "
                    "for this question. Increase retrieval depth, broaden scope or select frames manually."
                )
        else:
            if not groups:
                payload = self._search_payload(req, default_top_k=DEFAULT_TOP_K)
                result = await self.search.search(payload)
                groups = result.get("groups") or []
                scope = result.get("scope")
                warnings = list(result.get("warnings") or [])
                latency = dict(result.get("latency_ms") or {})
            if query_type == "AVS":
                params = AVS_PARAMS
                rows, diagnostics = generate_avs(
                    groups or [], limit=limit,
                    taken=[(video, frames[0]) for video, frames in taken],
                )
                answers = [self._decorate(row, "") for row in rows]
            else:
                # One pool build, used for both the ranking and the report on it.
                pools = build_pools(groups or [], params.validated())
                answers = [
                    self._decorate(a.to_dict(), answer_text)
                    for a in generate_from_pools(
                        pools,
                        params,
                        limit=limit,
                        # One frame per answer outside TRAKE, so the first is the one.
                        taken=[(video, frames[0]) for video, frames in taken],
                    )
                ]
                diagnostics = {
                    "n_videos_in_pool": len(pools),
                    "n_anchors": sum(len(p.anchors) for p in pools),
                    "anchor_budget": anchor_budget(params.validated()),
                    "ambiguous_videos": sum(1 for p in pools if p.ambiguous),
                    "videos_used": len({a["video_id"] for a in answers}),
                    "taken": len(taken),
                }

        if len(answers) < limit:
            warnings.append(
                f"Generated only {len(answers)}/{limit} answers; insufficient candidates "
                "(try increasing retrieval depth or broadening scope)."
            )
        latency["answer_gen_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "query": req.get("query", ""),
            "query_type": query_type,
            "retrieval_database": self.s.retrieval_database,
            "reused_results": reused,
            "scope": scope,
            "params": params.validated().to_dict(),
            "answers": answers,
            "diagnostics": diagnostics,
            "warnings": warnings,
            "latency_ms": latency,
            "mode": "mock" if self.s.mock_mode else "live",
        }

    # ---- helpers -------------------------------------------------------
    def _search_payload(self, req: dict[str, Any], *, default_top_k: int) -> dict[str, Any]:
        return {
            "retrieval_database": self.s.retrieval_database,
            "image_models": req.get("image_models") or ["pe"],
            "query": req.get("query", ""),
            "query_type_hint": req.get("query_type_hint", "auto"),
            "scope": req.get("scope") or {"mode": "auto", "categories": []},
            "traffic": req.get("traffic") or "auto",
            "previous_hints": req.get("previous_hints") or [],
            "manual_overrides": req.get("manual_overrides")
            or {"force_channels": [], "disable_channels": []},
            "parsed": req.get("parsed"),
            "feedback": req.get("feedback"),
            "use_llm": bool(req.get("use_llm")),
            "expand": bool(req.get("expand")),
            "translate": bool(req.get("translate", True)),
            "top_k": int(req.get("top_k") or default_top_k),
            "max_videos": int(req.get("max_videos") or DEFAULT_MAX_VIDEOS),
        }

    def _decorate(self, row: dict[str, Any], answer_text: str) -> dict[str, Any]:
        keyframe_id = row.get("keyframe_id")
        return {
            **row,
            "frames": [row["frame_idx"]],
            "keyframe_ids": [keyframe_id],
            "pts_times": [row.get("pts_time")],
            "answer": answer_text,
            # Offsets are plain frame indices, not extracted keyframes, so only
            # an anchor can resolve to a still.
            "keyframe_url": self.media.keyframe_url_from_submit_id(keyframe_id)
            if keyframe_id
            else None,
            "video_url": self.media.video_url(row["video_id"]),
        }

    def _decorate_sequence(self, row: dict[str, Any], answer_text: str) -> dict[str, Any]:
        return {**row, "answer": answer_text, "video_url": self.media.video_url(row["video_id"])}
