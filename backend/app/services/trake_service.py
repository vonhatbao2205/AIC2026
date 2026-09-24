"""TRAKE search: two-pass per-event retrieval around one video-centric map.

Pass 1: retrieve each event globally (wide pool), then fold the flat per-event
results into `video_id -> per-event candidates` immediately. That map — not the
list of chains — is the intermediate representation everything after it reads,
which is what lets one abstraction serve pass-2 targeting, the DP, the heatmap
and the ranking instead of each of them grouping by video again.

A preliminary pass over that map (temporal NMS + the exact DP + the
coverage-dominant `trake_video_score`) says how strong each video's evidence
already is, and pass 2 spends its budget accordingly: videos that are one event
short of a chain first, weaker/thinner ones only while queries remain.

Pass 2: for those videos, re-search that exact video (Milvus filtered by
video_id) for each MISSING event — this surfaces the video's own best frame for
the gap event, turning 3/4 coverage into 4/4 when the costume/scene really is in
that video. A relative similarity floor prevents low-confidence fill candidates
from overstating coverage when the event is absent.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..config import Settings
from ..media import MediaUrlBuilder
from ..trake import TrakeVideoResult, build_trake_videos, select_pass2_gaps
from ..types import Evidence, FusedFrame
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
                translate=bool(req.get("translate", True)),
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
        # One camera / time / stage for the whole sequence too: every event
        # happens in the one video the statement describes.
        traffic = self.search.resolve_traffic(
            req.get("traffic"), query=query, hints=req.get("previous_hints")
        )
        # Run all events' retrieval concurrently.
        results = await asyncio.gather(
            *(
                self.search.retrieve(
                    self._event_to_parsed(parsed, ev),
                    top_k=top_k,
                    categories=scope.categories,
                    image_models=image_models,
                    **({"frames": traffic.frames} if traffic.frames is not None else {}),
                )
                for ev in events
            )
        )
        event_frames = [[f for g in groups for f in g.frames] for groups, _ in results]
        tara_warning = None
        if self.s.has_tara_search and (parsed.get("channels") or {}).get("tara", {}).get("enabled", True):
            try:
                await self._add_tara_event_candidates(events, event_frames, scope.categories, parsed)
            except Exception as exc:  # noqa: BLE001 - preserve the frame retrieval path
                tara_warning = f"TARA TRAKE candidates unavailable: {exc}"
        per_event_meta = [
            {
                "event_index": ev.get("event_index"),
                "description_vi": ev.get("description_vi"),
                "candidate_count": len(event_frames[idx]),
            }
            for idx, ev in enumerate(events)
        ]

        # ---- Preliminary assembly (drives pass 2) ----
        # Partial by definition here — pass 2 has not run — so the preliminary
        # pass always allows partial chains regardless of the query's policy.
        # Its DP is what says which events are still missing FROM THE ANSWER,
        # which is a different (and larger) set than the events with no candidate.
        preliminary = build_trake_videos(event_frames, fallback_partial=True, max_results=None)
        gaps = select_pass2_gaps(preliminary, len(events))

        # ---- Pass 2: targeted in-video fill for the events the DP could not place ----
        # No scope needed here: pass 2 searches inside videos pass 1 already
        # returned, which are in scope by construction.
        await self._fill_missing_events(
            events, event_frames, gaps=gaps, image_models=image_models
        )

        fallback = (trake_cfg.get("fallback_policy") or "").startswith("allow_partial")
        videos = build_trake_videos(event_frames, fallback_partial=fallback)
        preliminary_rank = {v.video_id: i + 1 for i, v in enumerate(preliminary)}
        durations = await self.search.elastic.video_durations([v.video_id for v in videos])
        for video in videos:
            video.duration_s = durations.get(video.video_id)

        return {
            "retrieval_database": self.s.retrieval_database,
            "image_models": list(image_models),
            "query": query,
            "parsed": parsed,
            "scope": scope.to_dict(),
            "traffic": traffic.to_dict(),
            "events": per_event_meta,
            # The video-centric result the console renders. `sequences` is the
            # same assembly seen as flat chains, kept because the answer
            # generator and the benchmarks are written against chains.
            "videos": [self._serialize_video(v, preliminary_rank) for v in videos],
            "sequences": [self._serialize_sequence(v.sequence) for v in videos],
            "pass2": {
                "targets": list(gaps),
                "queries": sum(len(missing) for missing in gaps.values()),
                "candidates": len(preliminary),
            },
            "mode": "mock" if self.s.mock_mode else "live",
            "warnings": [tara_warning] if tara_warning else [],
        }

    async def _add_tara_event_candidates(
        self,
        events: list[dict[str, Any]],
        event_frames: list[list[FusedFrame]],
        categories: tuple[str, ...],
        parsed: dict[str, Any],
    ) -> None:
        """Add real keyframe-backed TARA event clips to the existing temporal DP."""
        if parsed.get("translation_failed"):
            return
        queries = [
            (ev.get("description_en_visual") or (ev.get("image_pe_queries_en") or [""])[0]).strip()
            for ev in events
        ]
        active = [(i, query) for i, query in enumerate(queries) if query]
        if not active:
            return
        vectors = await self.search.tara.encode_text([query for _, query in active])
        raw = await asyncio.gather(*(
            asyncio.to_thread(
                self.search.milvus.search_tara_clips, vector,
                scale="event", top_k=600, categories=categories,
            )
            for vector in vectors
        ))
        video_filter = set((parsed.get("filters") or {}).get("video_ids") or [])
        category_filter = set((parsed.get("filters") or {}).get("categories") or [])
        selected_by_event: list[tuple[int, list[dict[str, Any]]]] = []
        positions: list[tuple[str, float]] = []
        for (event_index, _), clips in zip(active, raw):
            selected: list[dict[str, Any]] = []
            times_by_video: dict[str, list[float]] = {}
            for clip in clips:
                video_id = clip["video_id"]
                if video_filter and video_id not in video_filter:
                    continue
                if category_filter and clip["category"] not in category_filter:
                    continue
                middle = clip["center_frame_idx"] / clip["fps"]
                kept = times_by_video.setdefault(video_id, [])
                if len(kept) >= 4 or any(abs(middle - earlier) < 3.0 for earlier in kept):
                    continue
                kept.append(middle)
                selected.append(clip)
                positions.append((video_id, middle))
                if len(selected) >= 160:
                    break
            selected_by_event.append((event_index, selected))
        keyframes = await self.search.elastic.nearest_keyframes_by_time(positions)
        cursor = 0
        for event_index, clips in selected_by_event:
            existing = {frame.submit_keyframe_id: frame for frame in event_frames[event_index]}
            for rank, clip in enumerate(clips, 1):
                keyframe = keyframes[cursor]
                cursor += 1
                if not keyframe or keyframe.get("video_id") != clip["video_id"]:
                    continue
                key = keyframe.get("submit_keyframe_id")
                if not key:
                    continue
                contribution = 1.0 / (60 + rank)
                evidence = Evidence(
                    type="tara", score=float(clip["score"]),
                    start=float(clip["start_time"]), end=float(clip["end_time"]),
                    extra={"clip_id": clip["clip_id"], "scale": "event"},
                )
                if key in existing:
                    frame = existing[key]
                    if "tara" not in frame.channels:
                        frame.channels.append("tara")
                        frame.score += contribution
                    frame.per_channel_score["tara"] = max(
                        frame.per_channel_score.get("tara", 0.0), float(clip["score"])
                    )
                    frame.evidence.append(evidence)
                    continue
                frame = FusedFrame(
                    submit_keyframe_id=key, video_id=clip["video_id"],
                    keyframe_n=int(keyframe["keyframe_n"]),
                    pts_time=float(keyframe["pts_time"]),
                    frame_idx=int(keyframe["frame_idx"]),
                    fps=float(keyframe["fps"]), score=contribution,
                    channels=["tara"], evidence=[evidence],
                    per_channel_score={"tara": float(clip["score"])},
                )
                event_frames[event_index].append(frame)
                existing[key] = frame

    async def _fill_missing_events(
        self,
        events: list[dict[str, Any]],
        event_frames: list[list[FusedFrame]],
        *,
        gaps: dict[str, list[int]] | None = None,
        max_promising: int = 15,
        accept_ratio: float = 0.45,
        # Hypotheses handed to the assembly DP per gap. Bounded by what the
        # targeted search returns anyway, and the per-event NMS cap downstream
        # keeps the candidate pool from growing.
        max_fills_per_gap: int = 3,
        image_models: tuple[str, ...] = ("pe",),
    ) -> None:
        """Pass 2: for each `(video, missing event)` in `gaps`, search that video
        for the event and add the in-video best frame that is BOTH above a
        relative similarity floor AND temporally placeable between the video's
        already-found neighbour frames. An order-infeasible fill would just be
        dropped by the assembly DP (strictly-increasing time), wasting the one
        shot at that gap — so prefer the highest-scoring hit that can actually
        slot in.

        `gaps` comes from `select_pass2_gaps`, i.e. from the preliminary DP:
        the events missing from the ANSWER, not merely the events with no
        candidate. Without it this falls back to the covered-event count, which
        is what the direct-service callers and older tests pass."""
        n = len(events)
        if n < 2:
            return

        # Per video: event positions with a pass-1 candidate -> their pts.
        # (pts-less frames are invisible to assembly, so ignore them here too.)
        cover_pts: dict[str, dict[int, list[float]]] = {}
        seen_ids: dict[tuple[str, int], set[str]] = {}
        for i, frames in enumerate(event_frames):
            for f in frames:
                if f.pts_time is None:
                    continue
                cover_pts.setdefault(f.video_id, {}).setdefault(i, []).append(f.pts_time)
                seen_ids.setdefault((f.video_id, i), set()).add(f.submit_keyframe_id)
        if gaps is None:
            gaps = {
                video_id: [i for i in range(n) if i not in cover_pts[video_id]]
                for video_id in sorted(
                    (v for v, evs in cover_pts.items() if 1 <= len(evs) < n),
                    key=lambda v: -len(cover_pts[v]),
                )
            }
        # A video pass 1 left completely empty has no neighbour to order a fill
        # against, so its queries would be spent on an unplaceable frame.
        planned = [
            (video_id, missing)
            for video_id, missing in gaps.items()
            if missing and cover_pts.get(video_id)
        ][:max_promising]
        if not planned:
            return
        promising = [video_id for video_id, _ in planned]

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

        # Targeted in-video search for every (video, event the DP could not place).
        jobs = [
            (vid, i, model)
            for vid, missing in planned
            for i in missing
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

        # Two different questions, deliberately answered by two different numbers.
        # ORDER comes from RRF across the models, because a 1280-d PE cosine and a
        # 4096-d Qwen cosine are only comparable by rank. QUALITY stays
        # model-local: `score / that model's best hit for the event`, which is
        # what the heat row and the acceptance floor mean. Normalising RRF by the
        # gap's own best made every top candidate 1.0 — a frame that had scraped
        # past the 45% floor was reported to the operator as a perfect match.
        ranked_by_gap: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for gap, per_model in by_gap.items():
            quality: dict[str, float] = {}
            for model, rows in per_model.items():
                ref_score = ref[(model, gap[1])]
                for row in rows:
                    ratio = float(row["score"]) / ref_score if ref_score > 0 else 0.0
                    key = row["submit_keyframe_id"]
                    quality[key] = max(quality.get(key, 0.0), ratio)
            if len(per_model) == 1:
                rows = next(iter(per_model.values()))
                ranked_by_gap[gap] = [
                    {**row, "fill_quality": quality[row["submit_keyframe_id"]]}
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
            ranked_by_gap[gap] = [
                {**entry["row"], "fill_quality": quality[entry["row"]["submit_keyframe_id"]]}
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
            # Loosest NECESSARY window: after some earlier-event candidate, before
            # some later-event candidate. It is not sufficient — with E1@90 and
            # E2@100 a candidate at 95 clears the bound and still cannot follow
            # E2 — which is exactly why it is used to ORDER the hypotheses and
            # never to pick among them.
            lower = min((p for ev, ps in covered.items() if ev < i for p in ps), default=None)
            upper = max((p for ev, ps in covered.items() if ev > i for p in ps), default=None)
            already = seen_ids.get((vid, i), set())

            candidates: list[tuple[int, int, dict[str, Any], dict[str, Any], float]] = []
            for rank, h in enumerate(hits):
                if h["submit_keyframe_id"] in already:
                    continue  # pass 1 already offered this frame for this event
                rec = recs.get(h["submit_keyframe_id"])
                pts = rec.get("pts_time") if rec else None
                if pts is None:
                    continue
                placeable = (lower is None or pts > lower) and (upper is None or pts < upper)
                candidates.append((0 if placeable else 1, rank, h, rec, pts))
            # Retrieval was already paid for down to top-3 per gap, and the
            # assembly DP is precisely the machinery for choosing among ordered
            # alternatives. Committing to one hit here threw the rest away before
            # the DP could see them: pick E2's best at 80 s and E3's best at 20 s
            # and the chain is dead, while E2@30 + E3@70 would have completed it.
            # So hand the DP the hypotheses and let it solve.
            candidates.sort(key=lambda item: (item[0], item[1]))
            for _, _, h, rec, pts in candidates[:max_fills_per_gap]:
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
                        # The undistorted ratio, for the heatmap: `score` puts the
                        # fill on a small scale that OVERLAPS real evidence, so
                        # the heat row would misreport it.
                        fill_quality=float(h["fill_quality"]),
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

    def _serialize_video(
        self, video: TrakeVideoResult, preliminary_rank: dict[str, int]
    ) -> dict[str, Any]:
        """One video card: coverage, quality, per-event peaks and representatives.

        No heatmap image is rendered here — the peaks travel as sparse points and
        the console draws them, so the same data serves the heat row, the
        representative thumbnails and the click-to-seek targets."""
        d = video.to_dict()
        d["video_url"] = self.media.video_url(video.video_id)
        for event in d["events"]:
            for peak in [*event["peaks"], event["representative"]]:
                if peak is not None:
                    peak["keyframe_url"] = self.media.keyframe_url(
                        video.video_id, peak["keyframe_n"]
                    )
        for frame in d["best_chain"]:
            frame["keyframe_url"] = self.media.keyframe_url(video.video_id, frame["keyframe_n"])
        # How far the video moved once pass 2 had run — the one number that says
        # whether the fill budget was spent on the right videos.
        d["preliminary_rank"] = preliminary_rank.get(video.video_id)
        return d
