"""TRAKE search: two-pass per-event retrieval, then optimal ordered assembly.

Pass 1: retrieve each event globally (wide pool).
Pass 2: for promising videos that cover only SOME events, re-search that exact
video (Milvus filtered by video_id) for each MISSING event — this surfaces the
video's own best frame for the gap event, turning 3/4 coverage into 4/4 when the
costume/scene really is in that video. A relative similarity floor prevents
low-confidence fill candidates from overstating coverage when the event is absent.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..config import Settings
from ..media import MediaUrlBuilder
from ..trake import assemble_trake_sequences
from ..types import FusedFrame
from .search_service import SearchService


class TrakeService:
    def __init__(self, settings: Settings, search_service: SearchService | None = None):
        self.s = settings
        self.search = search_service or SearchService(settings)
        self.media = MediaUrlBuilder(settings.keyframe_media_base_url, settings.video_media_base_url)

    async def search_trake(self, req: dict[str, Any]) -> dict[str, Any]:
        query = req.get("query", "")
        image_models = self.search.resolve_image_models(req.get("image_models"))
        parsed = req.get("parsed")
        if parsed is None:
            parsed = await self.search.parser.parse(
                query,
                "TRAKE",
                req.get("previous_hints"),
                req.get("manual_overrides"),
                use_llm=req.get("use_llm", False),
            )
        if req.get("expand"):
            await self.search.expand_image_queries(parsed)
        trake_cfg = parsed.get("trake", {})
        events = trake_cfg.get("events") or []
        if not events:
            # Fall back to a single event from the whole query.
            events = [{"event_index": 1, "image_pe_queries_en": [query], "description_vi": query}]

        # Run retrieval per event using a channels config derived from the event.
        # Use a wide candidate pool so the true video's frame for each event is
        # captured (the DP then finds the optimal ordered assignment per video).
        top_k = req.get("top_k") or 400
        # One scope for the whole sequence: every event has to be found in the
        # same video, so letting events resolve their own folders could only ever
        # produce sequences that cannot be assembled.
        scope = self.search.resolve_scope(req.get("scope"), query=query)
        # Run all events' retrieval concurrently.
        results = await asyncio.gather(
            *(
                self.search.retrieve(
                    self._event_to_parsed(parsed, ev),
                    top_k=top_k,
                    categories=scope.categories,
                    image_models=image_models,
                )
                for ev in events
            )
        )
        event_frames = [[f for g in groups for f in g.frames] for groups, _ in results]
        per_event_meta = [
            {
                "event_index": ev.get("event_index"),
                "description_vi": ev.get("description_vi"),
                "candidate_count": len(event_frames[idx]),
            }
            for idx, ev in enumerate(events)
        ]

        # ---- Pass 2: targeted in-video fill for missing events ----
        # No scope needed here: pass 2 searches inside videos pass 1 already
        # returned, which are in scope by construction.
        await self._fill_missing_events(events, event_frames, image_models=image_models)

        fallback = (trake_cfg.get("fallback_policy") or "").startswith("allow_partial")
        sequences = assemble_trake_sequences(event_frames, fallback_partial=fallback)

        return {
            "retrieval_database": self.s.retrieval_database,
            "image_models": list(image_models),
            "query": query,
            "parsed": parsed,
            "scope": scope.to_dict(),
            "events": per_event_meta,
            "sequences": [self._serialize_sequence(s) for s in sequences],
            "mode": "mock" if self.s.mock_mode else "live",
        }

    async def _fill_missing_events(
        self,
        events: list[dict[str, Any]],
        event_frames: list[list[FusedFrame]],
        *,
        max_promising: int = 15,
        accept_ratio: float = 0.45,
        image_models: tuple[str, ...] = ("pe",),
    ) -> None:
        """Pass 2: for promising partial videos, search that video for its missing
        events and add the in-video best frame that is BOTH above a relative
        similarity floor AND temporally placeable between the video's already-found
        neighbour frames. An order-infeasible fill would just be dropped by the
        assembly DP (strictly-increasing time), wasting the one shot at that gap —
        so prefer the highest-scoring hit that can actually slot in."""
        n = len(events)
        if n < 2:
            return

        # Per video: covered event positions -> pts of their pass-1 candidates.
        # (pts-less frames are invisible to assembly, so ignore them here too.)
        cover_pts: dict[str, dict[int, list[float]]] = {}
        for i, frames in enumerate(event_frames):
            for f in frames:
                if f.pts_time is None:
                    continue
                cover_pts.setdefault(f.video_id, {}).setdefault(i, []).append(f.pts_time)
        promising = sorted(
            (v for v, evs in cover_pts.items() if 1 <= len(evs) < n),
            key=lambda v: -len(cover_pts[v]),
        )[:max_promising]
        if not promising:
            return

        ev_queries = [
            (ev.get("image_pe_queries_en") or [ev.get("description_en_visual") or ev.get("description_vi") or ""])[0]
            for ev in events
        ]
        image_models = self.search.resolve_image_models(image_models)

        async def _encode(model: str):
            encoder = self.search.pe if model == "pe" else self.search.qwen3_vl
            return model, await encoder.encode_text(ev_queries)

        encoded = await asyncio.gather(
            *(_encode(model) for model in image_models), return_exceptions=True
        )
        vectors_by_model = {
            model: vectors
            for outcome in encoded
            if not isinstance(outcome, BaseException)
            for model, vectors in (outcome,)
        }
        if not vectors_by_model:
            return

        async def _search(model: str, vec, **kw):
            searcher = (
                self.search.milvus.search_image
                if model == "pe"
                else self.search.milvus.search_qwen_image
            )
            try:
                return await asyncio.to_thread(searcher, vec, **kw)
            except Exception:  # noqa: BLE001
                return []

        # Reference cosine remains model-local and is used only for that model's
        # relative acceptance floor. Cross-model combination below is rank-only.
        ref_jobs = [
            (model, event_index)
            for model in vectors_by_model
            for event_index in range(n)
        ]
        ref_hits = await asyncio.gather(
            *(
                _search(model, vectors_by_model[model][event_index], top_k=1)
                for model, event_index in ref_jobs
            )
        )
        ref = {
            (model, event_index): (hits[0]["score"] if hits else 0.0)
            for (model, event_index), hits in zip(ref_jobs, ref_hits)
        }

        # Targeted in-video search for every (promising video, missing event), parallel.
        jobs = [
            (vid, i, model)
            for vid in promising
            for i in range(n)
            if i not in cover_pts[vid]
            for model in vectors_by_model
            if ref.get((model, i), 0.0) > 0
        ]
        if not jobs:
            return
        job_hits = await asyncio.gather(
            *(
                _search(model, vectors_by_model[model][i], top_k=3, video_id=vid)
                for vid, i, model in jobs
            )
        )

        by_gap: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
        for (vid, i, model), hits in zip(jobs, job_hits):
            floor = accept_ratio * ref[(model, i)]
            accepted = [hit for hit in hits if hit["score"] >= floor]
            if accepted:
                by_gap.setdefault((vid, i), {})[model] = accepted

        ranked_by_gap: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for gap, per_model in by_gap.items():
            if len(per_model) == 1:
                model, rows = next(iter(per_model.items()))
                ref_score = ref[(model, gap[1])]
                ranked_by_gap[gap] = [
                    {**row, "fill_quality": float(row["score"]) / ref_score}
                    for row in rows
                ]
                continue
            fused: dict[str, dict[str, Any]] = {}
            for model, rows in per_model.items():
                for rank, row in enumerate(rows):
                    entry = fused.setdefault(
                        row["submit_keyframe_id"], {"row": row, "rrf": 0.0}
                    )
                    entry["rrf"] += 1.0 / (60 + rank + 1)
            ordered = sorted(
                fused.values(),
                key=lambda item: (-item["rrf"], item["row"]["submit_keyframe_id"]),
            )
            best_rrf = ordered[0]["rrf"] if ordered else 1.0
            ranked_by_gap[gap] = [
                {**entry["row"], "fill_quality": entry["rrf"] / best_rrf}
                for entry in ordered
            ]

        # Batch-fetch pts for every candidate up front so feasibility can be tested
        # before choosing (one _mget instead of a post-hoc per-pick lookup).
        cand_ids = {
            hit["submit_keyframe_id"]
            for hits in ranked_by_gap.values()
            for hit in hits
        }
        recs = await self.search.elastic.get_keyframes_by_ids(list(cand_ids)) if cand_ids else {}

        new: list[tuple[int, FusedFrame]] = []
        for (vid, i), hits in ranked_by_gap.items():
            covered = cover_pts[vid]
            # Loosest necessary window: after some earlier-event candidate, before
            # some later-event candidate. A pick outside it can never be ordered.
            lower = min((p for ev, ps in covered.items() if ev < i for p in ps), default=None)
            upper = max((p for ev, ps in covered.items() if ev > i for p in ps), default=None)

            chosen = None  # first feasible hit above floor (best, since score-sorted)
            fallback = None  # highest-scoring hit above floor with a known pts
            for h in hits:
                rec = recs.get(h["submit_keyframe_id"])
                pts = rec.get("pts_time") if rec else None
                if pts is None:
                    continue
                if fallback is None:
                    fallback = (h, rec, pts)
                if (lower is None or pts > lower) and (upper is None or pts < upper):
                    chosen = (h, rec, pts)
                    break
            pick = chosen or fallback
            if pick is None:
                continue
            h, rec, pts = pick
            new.append((
                i,
                FusedFrame(
                    submit_keyframe_id=h["submit_keyframe_id"],
                    video_id=h["video_id"],
                    keyframe_n=int(h["keyframe_n"]),
                    pts_time=pts,
                    score=round(0.02 * h["fill_quality"], 5),
                    frame_idx=rec.get("frame_idx"),
                    fps=rec.get("fps"),
                    via_fill=True,
                ),
            ))

        for i, fr in new:
            event_frames[i].append(fr)

    def _event_to_parsed(self, parsed: dict[str, Any], ev: dict[str, Any]) -> dict[str, Any]:
        """Build a per-event channels config reusing the base routing."""
        base_channels = parsed.get("channels", {})
        channels = {
            "image_pe": {
                "enabled": True,
                "weight": 1.0,
                "queries_en": ev.get("image_pe_queries_en") or [ev.get("description_en_visual") or ev.get("description_vi") or ""],
            },
            "ocr": {
                "enabled": bool(ev.get("ocr_queries_vi")),
                "weight": base_channels.get("ocr", {}).get("weight", 0.8) or 0.8,
                "queries_vi": ev.get("ocr_queries_vi") or [],
            },
            "speech": {
                "enabled": bool(ev.get("speech_queries_vi")),
                "weight": base_channels.get("speech", {}).get("weight", 0.8) or 0.8,
                "queries_vi": ev.get("speech_queries_vi") or [],
            },
            "audio": {
                "enabled": bool(ev.get("audio_queries_en")),
                "weight": base_channels.get("audio", {}).get("weight", 0.7) or 0.7,
                "queries_en": ev.get("audio_queries_en") or [],
                "sound_labels_en": ev.get("audio_queries_en") or [],
            },
        }
        return {
            "channels": channels,
            "filters": parsed.get("filters", {}),
            "rerank_policy": parsed.get("rerank_policy", {"rrf_k": 60}),
        }

    def _serialize_sequence(self, seq) -> dict[str, Any]:
        d = seq.to_dict()
        for f in d["frames"]:
            f["keyframe_url"] = self.media.keyframe_url(seq.video_id, f["keyframe_n"])
        d["video_url"] = self.media.video_url(seq.video_id)
        return d
