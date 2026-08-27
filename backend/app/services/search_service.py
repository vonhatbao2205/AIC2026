"""Search orchestration: parse -> per-channel retrieval -> RRF -> group-by-video.

Produces the canonical result shape consumed by the frontend, including built
media URLs, channel attribution, evidence, and latency breakdown.
"""
from __future__ import annotations

import asyncio
import functools
import time
from typing import Any

from ..adapters.elastic_client import ElasticClient
from ..adapters.milvus_client import MilvusClient
from ..adapters.pe_encoder import GlapEncoderClient, PeEncoderClient
from ..adapters.qwen_reranker import QwenRerankerClient
from ..config import Settings
from ..fusion import group_by_video, reciprocal_rank_fusion
from ..media import MediaUrlBuilder
from ..query_parser import QueryParser
from ..scope import ResolvedScope, resolve_scope
from ..types import Channel, ChannelHit, Evidence, FusedFrame, VideoGroup


#: Upper bound of the console's retrieval-depth slider; every returned frame is
#: enriched so it can be submitted.
MAX_ENRICHED_FRAMES = 1000


class ServiceUnavailable(Exception):
    """A required upstream (PE encoder / Milvus) is unreachable in live mode."""


class SearchService:
    def __init__(self, settings: Settings):
        self.s = settings
        self.elastic = ElasticClient(settings)
        self.milvus = MilvusClient(settings)
        self.pe = PeEncoderClient(settings)
        self.glap = GlapEncoderClient(settings)
        self.reranker = QwenRerankerClient(settings)
        self.parser = QueryParser(settings)
        self.media = MediaUrlBuilder(settings.keyframe_media_base_url, settings.video_media_base_url)

    # ---- channel runners ----------------------------------------------
    async def _run_image_pe(
        self,
        cfg: dict[str, Any],
        top_k: int,
        categories: tuple[str, ...] = (),
        *,
        rerank: bool = False,
        rerank_info: dict[str, Any] | None = None,
    ) -> tuple[list[ChannelHit], float]:
        """PE image search with QUERY EXPANSION: each query variant is searched
        separately and the results are fused by max cosine per frame (a frame
        matching ANY phrasing strongly is kept). One variant == plain search.

        With `rerank`, Milvus is asked for a WIDER candidate pool than the caller
        wants and Qwen3-VL rescores it before the list is cut back to `top_k`.
        The reranker judges `(query, image)` pairs, so it has to run here — at
        frame level, on the PE candidate union — rather than after RRF, where a
        low visual score would overrule the OCR/speech/audio channels on frames
        whose evidence is not visual at all.
        """
        t0 = time.perf_counter()
        queries = [q for q in (cfg.get("queries_en") or []) if q and q.strip()]
        if not queries:
            return [], 0.0
        rerank_on = bool(rerank and self.reranker.enabled)
        candidate_k = max(top_k, self.reranker.candidate_k) if rerank_on else top_k
        vectors = await self.pe.encode_text(queries)
        search = functools.partial(
            self.milvus.search_image, top_k=candidate_k, categories=categories
        )
        if self.milvus.mock:
            raws = [search(vector) for vector in vectors]
        else:
            raws = await asyncio.gather(*(asyncio.to_thread(search, vector) for vector in vectors))
        # Union variants, keep the best (max) cosine per frame.
        best: dict[str, dict[str, Any]] = {}
        for raw in raws:
            for r in raw:
                kf = r["submit_keyframe_id"]
                if kf not in best or r["score"] > best[kf]["score"]:
                    best[kf] = r
        fused = sorted(best.values(), key=lambda r: -r["score"])[:candidate_k]
        if rerank_on:
            fused = await self._rerank_frames(queries[0], fused, rerank_info)
        fused = fused[:top_k]
        hits = [
            ChannelHit(
                channel="image_pe",
                submit_keyframe_id=r["submit_keyframe_id"],
                video_id=r["video_id"],
                keyframe_n=int(r["keyframe_n"]),
                score=float(r["score"]),
                rank=i,
                evidence=Evidence(
                    type="image_pe",
                    score=float(r["score"]),
                    extra=(
                        {"rerank_score": round(float(r["rerank_score"]), 4)}
                        if r.get("rerank_score") is not None
                        else {}
                    ),
                ),
            )
            for i, r in enumerate(fused)
        ]
        return hits, (time.perf_counter() - t0) * 1000

    async def _rerank_frames(
        self,
        query: str,
        rows: list[dict[str, Any]],
        info: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Reorder PE candidates by Qwen3-VL relevance. FAIL-OPEN.

        `retrieve()` turns any channel exception into an empty channel, so letting
        a rerank failure escape would delete the whole PE result the moment the
        Colab tunnel drops. A refinement is not allowed to cost more than it adds:
        every failure path here returns the PE order untouched.

        Only the FIRST query variant is sent. Reranking each expansion paraphrase
        would triple the GPU cost to answer a question the operator did not ask —
        the reranker judges candidates against the real query.
        """
        started = time.perf_counter()
        report = info if info is not None else {}
        if not rows or not query.strip():
            return rows
        documents = [
            {
                "id": r["submit_keyframe_id"],
                "image": self.media.keyframe_url(r["video_id"], int(r["keyframe_n"])),
            }
            for r in rows
        ]
        try:
            ranked = await self.reranker.rerank(query, documents)
        except Exception as exc:  # noqa: BLE001 - the whole point is to stay open
            report.update({
                "ok": False,
                "candidates": len(rows),
                "reranked": 0,
                "ms": round((time.perf_counter() - started) * 1000, 1),
                "error": str(exc)[:300],
            })
            return rows
        by_id = {r["submit_keyframe_id"]: r for r in rows}
        ordered: list[dict[str, Any]] = []
        for item in ranked:
            row = by_id.pop(item["id"], None)
            if row is None:
                continue
            row["rerank_score"] = item["score"]
            ordered.append(row)
        # Frames the worker could not score (unreachable keyframe image, or cut by
        # its own top_k) keep their PE order behind the reranked ones — dropping
        # them would lose recall PE had already paid for.
        ordered.extend(by_id.values())
        report.update({
            "ok": True,
            "candidates": len(rows),
            "reranked": len(ordered) - len(by_id),
            "ms": round((time.perf_counter() - started) * 1000, 1),
        })
        return ordered

    async def _run_similar(
        self, seed_ids: list[str], top_k: int, categories: tuple[str, ...] = ()
    ) -> tuple[list[ChannelHit], float]:
        """Image-to-image kNN seeded by the frames the operator marked.

        The stored PE-G14 embedding is fetched by primary key (no re-encoding),
        then each seed runs its own Milvus search and the union keeps the best
        cosine per frame — the same max-fusion the text channel uses for query
        variants. Emitting this as a normal ranked channel is what keeps positive
        feedback on the same scale as every other signal under RRF.
        """
        t0 = time.perf_counter()
        if not seed_ids:
            return [], 0.0
        vectors = await asyncio.to_thread(self.milvus.get_image_vectors, seed_ids)
        if not vectors:
            return [], (time.perf_counter() - t0) * 1000
        raws = await asyncio.gather(
            *(
                asyncio.to_thread(
                    self.milvus.search_image, vector, top_k=top_k, categories=categories
                )
                for vector in vectors.values()
            )
        )
        best: dict[str, dict[str, Any]] = {}
        for raw in raws:
            for r in raw:
                kf = r["submit_keyframe_id"]
                if kf not in best or r["score"] > best[kf]["score"]:
                    best[kf] = r
        fused = sorted(best.values(), key=lambda r: -r["score"])[:top_k]
        hits = [
            ChannelHit(
                channel="similar",
                submit_keyframe_id=r["submit_keyframe_id"],
                video_id=r["video_id"],
                keyframe_n=int(r["keyframe_n"]),
                score=float(r["score"]),
                rank=i,
                evidence=Evidence(
                    type="similar",
                    score=float(r["score"]),
                    extra={"seeds": len(vectors)},
                ),
            )
            for i, r in enumerate(fused)
        ]
        return hits, (time.perf_counter() - t0) * 1000

    async def _run_ocr(
        self, cfg: dict[str, Any], top_k: int, categories: tuple[str, ...] = ()
    ) -> tuple[list[ChannelHit], float]:
        t0 = time.perf_counter()
        tf = cfg.get("time_filters") or {}
        raw = await self.elastic.search_ocr(
            cfg.get("queries_vi") or [],
            cfg.get("queries_folded") or [],
            exact_phrases=cfg.get("exact_phrases") or [],
            numbers=cfg.get("numbers") or [],
            hour=tf.get("hour"),
            clock=tf.get("clock"),
            categories=categories,
            size=top_k,
        )
        hits = [
            ChannelHit(
                channel="ocr",
                submit_keyframe_id=r["submit_keyframe_id"],
                video_id=r["video_id"],
                keyframe_n=int(r["keyframe_n"]),
                score=float(r["score"]),
                rank=i,
                evidence=Evidence(
                    type="ocr",
                    score=float(r["score"]),
                    text=r.get("text_clean"),
                    extra={k: r.get(k) for k in ("clock", "hour") if r.get(k) is not None},
                ),
            )
            for i, r in enumerate(raw)
        ]
        return hits, (time.perf_counter() - t0) * 1000

    async def _run_speech(
        self, cfg: dict[str, Any], top_k: int, categories: tuple[str, ...] = ()
    ) -> tuple[list[ChannelHit], float]:
        t0 = time.perf_counter()
        raw = await self.elastic.search_speech(
            cfg.get("queries_vi") or [],
            exact_phrases=cfg.get("exact_phrases") or [],
            categories=categories,
            size=top_k,
        )
        hits = [
            ChannelHit(
                channel="speech",
                submit_keyframe_id=r["submit_keyframe_id"],
                video_id=r["video_id"],
                keyframe_n=int(r["keyframe_n"]),
                score=float(r["score"]),
                rank=i,
                evidence=Evidence(
                    type="speech",
                    score=float(r["score"]),
                    text=r.get("text"),
                    start=r.get("start"),
                    end=r.get("end"),
                    extra={k: r.get(k) for k in ("confidence_bucket", "segment_role") if r.get(k)},
                ),
            )
            for i, r in enumerate(raw)
            if r.get("submit_keyframe_id")
        ]
        return hits, (time.perf_counter() - t0) * 1000

    async def _run_audio(
        self, cfg: dict[str, Any], top_k: int, categories: tuple[str, ...] = ()
    ) -> tuple[list[ChannelHit], float]:
        """Audio channel = Elastic tags/caption (BM25 + demotions) FUSED with GLAP
        audio-vector search (semantic sound match). The two ranked lists are merged
        by rank (RRF) so their different score scales don't compete."""
        t0 = time.perf_counter()
        queries = cfg.get("queries_en") or []
        labels = cfg.get("sound_labels_en") or []

        elastic_task = self.elastic.search_audio(queries, labels, categories=categories, size=top_k)
        glap_evidence: dict[str, Evidence] = {}
        glap_raw: list[dict[str, Any]] = []
        if self.glap is not None and not self.glap.mock and (queries or labels):
            try:
                vectors = await self.glap.encode_text(queries or labels)
                vec = vectors[0]
                glap_raw = await asyncio.to_thread(
                    self.milvus.search_audio, vec, top_k=top_k, categories=categories
                )
            except Exception:  # noqa: BLE001 - GLAP optional; fall back to Elastic only
                glap_raw = []
        raw = await elastic_task

        # Rank-RRF merge over submit_keyframe_id (k small so both lists matter).
        k = 30
        scored: dict[str, dict[str, Any]] = {}
        for rank, r in enumerate(raw):
            kf = r.get("submit_keyframe_id")
            if not kf:
                continue
            e = scored.setdefault(kf, {"r": r, "score": 0.0})
            e["score"] += 1.0 / (k + rank + 1)
            e["evidence"] = Evidence(
                type="audio", score=float(r["score"]), text=r.get("caption"),
                start=r.get("start"), end=r.get("end"),
                extra={x: r.get(x) for x in ("top1_label", "caption_quality") if r.get(x)},
            )
        for rank, r in enumerate(glap_raw):
            kf = r.get("submit_keyframe_id")
            if not kf:
                continue
            e = scored.setdefault(kf, {"r": r, "score": 0.0})
            e["score"] += 1.0 / (k + rank + 1)
            if "evidence" not in e:  # only the vector signal hit this frame
                e["evidence"] = Evidence(
                    type="audio", score=float(r["score"]), start=r.get("start"), end=r.get("end"),
                    extra={"top1_label": r.get("top1_label"), "match": "glap-vector"},
                )

        ranked = sorted(scored.items(), key=lambda kv: -kv[1]["score"])
        hits = [
            ChannelHit(
                channel="audio",
                submit_keyframe_id=kf,
                video_id=e["r"]["video_id"],
                keyframe_n=int(e["r"]["keyframe_n"]),
                score=e["score"],
                rank=i,
                evidence=e.get("evidence"),
            )
            for i, (kf, e) in enumerate(ranked)
        ]
        return hits, (time.perf_counter() - t0) * 1000

    # ---- retrieval over a parsed routing object -----------------------
    async def retrieve(
        self,
        parsed: dict[str, Any],
        *,
        top_k: int = 100,
        feedback: dict[str, Any] | None = None,
        categories: tuple[str, ...] = (),
        rerank: bool = False,
    ) -> tuple[list[VideoGroup], dict[str, Any]]:
        """`categories` is the resolved search scope: the dataset folders this
        search may return frames from. It is pushed into every channel's own
        query (so top_k is filled from inside the scope) AND re-applied to the
        hits, which is what keeps mock mode and any future channel honest."""
        channels_cfg = parsed.get("channels", {})
        filters = parsed.get("filters", {})
        unsupported = self.s.unsupported_channels
        # Per-call, so two concurrent searches never write each other's report.
        rerank_info: dict[str, Any] = {}
        runners = {
            name: runner
            for name, runner in (
                (
                    "image_pe",
                    functools.partial(
                        self._run_image_pe, rerank=rerank, rerank_info=rerank_info
                    ),
                ),
                ("ocr", self._run_ocr),
                ("speech", self._run_speech),
                ("audio", self._run_audio),
            )
            if name not in unsupported
        }
        feedback = feedback or {}
        tasks: dict[Channel, asyncio.Task] = {}
        for name, runner in runners.items():
            cfg = channels_cfg.get(name, {})
            if cfg.get("enabled"):
                tasks[name] = asyncio.create_task(runner(cfg, top_k, categories))
        # Feedback-driven, not parser-driven: it exists only while the operator
        # keeps frames marked.
        positive_frames = [str(kf) for kf in (feedback.get("positive_frames") or []) if kf]
        if positive_frames:
            tasks["similar"] = asyncio.create_task(
                self._run_similar(positive_frames, top_k, categories)
            )

        latency: dict[str, Any] = {"channels": {}}
        channel_hits: dict[Channel, list[ChannelHit]] = {}
        weights: dict[str, float] = {}
        warnings: list[str] = []
        disabled = [
            name for name in sorted(unsupported) if channels_cfg.get(name, {}).get("enabled")
        ]
        if disabled:
            warnings.append(
                f"{self.s.retrieval_database} chưa có dữ liệu cho kênh: " + ", ".join(disabled)
            )
        if self.s.ocr_missing_categories and "ocr" in tasks:
            warnings.append(
                "OCR "
                + self.s.retrieval_database
                + " chưa index category: "
                + ", ".join(self.s.ocr_missing_categories)
            )
        for name, task in tasks.items():
            try:
                hits, ms = await task
            except Exception as exc:  # noqa: BLE001 - one dead channel must not kill the search
                hits, ms = [], 0.0
                warnings.append(f"{name} channel unavailable: {exc}")
            channel_hits[name] = _apply_filters(hits, filters, categories)
            latency["channels"][name] = round(ms, 1)
            weights[name] = float(channels_cfg.get(name, {}).get("weight") or 1.0)
        if rerank_info:
            latency["reranker"] = rerank_info
            if not rerank_info.get("ok"):
                warnings.append(
                    "Reranker không dùng được, giữ nguyên thứ tự PE: "
                    + str(rerank_info.get("error") or "unknown")
                )
        if warnings:
            latency["warnings"] = warnings

        t_fuse = time.perf_counter()
        rrf_k = (parsed.get("rerank_policy") or {}).get("rrf_k", 60)
        fused = reciprocal_rank_fusion(
            channel_hits,
            weights=weights,
            k=rrf_k,
            negative_frames=set(feedback.get("negative_frames") or []),
        )
        await self._enrich_pts(fused)
        groups = group_by_video(
            fused,
            prioritized_videos=set(feedback.get("positive_videos") or []),
            deprioritized_videos=set(feedback.get("negative_videos") or []),
        )
        latency["fusion_ms"] = round((time.perf_counter() - t_fuse) * 1000, 1)
        return groups, latency

    async def _enrich_pts(self, frames: list[FusedFrame]) -> None:
        """Fill pts_time / frame_idx / fps from the keyframe map (single batched
        _mget). frame_idx + fps are needed to build the video_id+frame_idx submit.

        The cap covers the whole result set the console can ask for. It used to
        stop at 300, which silently made every frame past that rank
        unsubmittable — they came back with no frame_idx at all — as soon as the
        retrieval-depth slider went above it."""
        need = [f for f in frames[:MAX_ENRICHED_FRAMES] if f.pts_time is None or f.frame_idx is None]
        if not need:
            return
        recs = await self.elastic.get_keyframes_by_ids([f.submit_keyframe_id for f in need])
        for frame in need:
            rec = recs.get(frame.submit_keyframe_id)
            if not rec:
                continue
            if frame.pts_time is None and rec.get("pts_time") is not None:
                frame.pts_time = rec.get("pts_time")
            if frame.frame_idx is None and rec.get("frame_idx") is not None:
                frame.frame_idx = rec.get("frame_idx")
            if frame.fps is None and rec.get("fps") is not None:
                frame.fps = rec.get("fps")

    # ---- simple flat vector search ------------------------------------
    async def simple_image_search(
        self,
        query: str,
        top_k: int = 60,
        scope_spec: dict[str, Any] | None = None,
        rerank: bool = False,
    ) -> dict[str, Any]:
        """Pure PE→Milvus image search returning a FLAT keyframe list ordered by
        cosine similarity (no parser, no fusion, no group-by-video).

        Simple Search is the second visual code path; hooking only the full search
        would leave this tab silently on plain PE while its tick box says otherwise.
        """
        t0 = time.perf_counter()
        query = (query or "").strip()
        scope = self.resolve_scope(scope_spec, query=query)
        if not query:
            return {
                "query": query,
                "results": [],
                "retrieval_database": self.s.retrieval_database,
                "scope": scope.to_dict(),
                "mode": "mock" if self.s.mock_mode else "live",
                "latency_ms": 0,
            }
        # Translate VI→EN so the English-centric PE encoder gets an English query.
        search_text = query
        translation_failed = False
        if self.s.translate_to_en and not self.s.mock_mode:
            from ..translate import translate_vi_to_en_status

            search_text, translated_ok = await translate_vi_to_en_status(
                query, settings=self.s
            )
            translation_failed = not translated_ok
        try:
            vectors = await self.pe.encode_text([search_text])
        except Exception as exc:  # noqa: BLE001
            raise ServiceUnavailable(
                f"PE text-encoder unreachable ({self.s.pe_encoder_url}). "
                f"Restart the Kaggle PE server and update PE_ENCODER_URL. Detail: {exc}"
            ) from exc
        rerank_on = bool(rerank and self.reranker.enabled)
        candidate_k = max(top_k, self.reranker.candidate_k) if rerank_on else top_k
        try:
            raw = (
                self.milvus.search_image(vectors[0], top_k=candidate_k, categories=scope.categories)
                if self.milvus.mock
                else await asyncio.to_thread(
                    self.milvus.search_image,
                    vectors[0],
                    top_k=candidate_k,
                    categories=scope.categories,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise ServiceUnavailable(f"Milvus image search failed: {exc}") from exc
        rerank_info: dict[str, Any] = {}
        if rerank_on:
            raw = await self._rerank_frames(search_text, list(raw), rerank_info)
        raw = raw[:top_k]
        results = [
            {
                "image_id": r["submit_keyframe_id"],
                "submit_keyframe_id": r["submit_keyframe_id"],
                "video_id": r["video_id"],
                "keyframe_n": int(r["keyframe_n"]),
                "score": round(float(r["score"]), 6),
                "rerank_score": (
                    round(float(r["rerank_score"]), 4)
                    if r.get("rerank_score") is not None
                    else None
                ),
                "keyframe_url": self.media.keyframe_url(r["video_id"], int(r["keyframe_n"])),
                "video_url": self.media.video_url(r["video_id"]),
            }
            for r in raw
        ]
        body: dict[str, Any] = {
            "query": query,
            "retrieval_database": self.s.retrieval_database,
            "scope": scope.to_dict(),
            "translated_query": search_text if search_text != query else None,
            "results": results,
            "mode": "mock" if self.s.mock_mode else "live",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        if rerank_info:
            body["reranker"] = rerank_info
        if translation_failed:
            body["warnings"] = [
                "Không dịch được query sang tiếng Anh — PE đang nhận tiếng Việt "
                "nên kết quả sẽ kém. Bật LLM hoặc kiểm tra mạng."
            ]
        return body

    def resolve_scope(self, spec: dict[str, Any] | None, *, query: str) -> ResolvedScope:
        """Resolve a request's scope spec against THIS profile's folder catalogue."""
        return resolve_scope(spec, query=query, retrieval_database=self.s.retrieval_database)

    async def expand_image_queries(self, parsed: dict[str, Any]) -> None:
        """Append Nemotron-generated visual paraphrases to the image_pe channel
        (and each TRAKE event), so retrieval fuses several phrasings of the query."""
        img = (parsed.get("channels") or {}).get("image_pe") or {}
        if img.get("enabled"):
            base = (img.get("queries_en") or [None])[0] or parsed.get("translated_en_visual")
            variants = await self.parser.expand_visual(base) if base else []
            if variants:
                merged = list(dict.fromkeys([*(img.get("queries_en") or []), *variants]))
                img["queries_en"] = merged
        trake = parsed.get("trake") or {}
        if trake.get("enabled"):
            for ev in trake.get("events", []):
                base = (ev.get("image_pe_queries_en") or [None])[0] or ev.get("description_en_visual")
                if not base:
                    continue
                variants = await self.parser.expand_visual(base)
                if variants:
                    ev["image_pe_queries_en"] = list(dict.fromkeys([*(ev.get("image_pe_queries_en") or []), *variants]))

    # ---- public search ------------------------------------------------
    async def search(self, req: dict[str, Any]) -> dict[str, Any]:
        t_total = time.perf_counter()
        query = req.get("query", "")
        parse_provided = req.get("parsed")
        t_parse = time.perf_counter()
        if parse_provided:
            parsed = parse_provided
            parse_ms = 0.0
        else:
            parsed = await self.parser.parse(
                query,
                req.get("query_type_hint", "auto"),
                req.get("previous_hints"),
                req.get("manual_overrides"),
                use_llm=req.get("use_llm", False),
            )
            parse_ms = (time.perf_counter() - t_parse) * 1000

        if req.get("expand"):
            await self.expand_image_queries(parsed)

        scope = self.resolve_scope(req.get("scope"), query=query)
        groups, latency = await self.retrieve(
            parsed,
            top_k=req.get("top_k", 100),
            feedback=req.get("feedback"),
            categories=scope.categories,
            rerank=bool(req.get("rerank")),
        )
        if parsed.get("translation_failed"):
            latency.setdefault("warnings", []).append(
                "Không dịch được query sang tiếng Anh — PE đang nhận tiếng Việt nên "
                "keyframe trả về sẽ không khớp. Bật LLM hoặc kiểm tra mạng."
            )
        latency["parse_ms"] = round(parse_ms, 1)
        latency["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)

        return {
            "query": query,
            "retrieval_database": self.s.retrieval_database,
            "parsed": parsed,
            "scope": scope.to_dict(),
            "groups": [self._serialize_group(g) for g in groups[: req.get("max_videos", 50)]],
            "latency_ms": latency,
            "warnings": latency.get("warnings", []),
            "mode": "mock" if self.s.mock_mode else "live",
        }

    # ---- serialization ------------------------------------------------
    def _serialize_frame(self, f: FusedFrame) -> dict[str, Any]:
        return {
            "image_id": f.submit_keyframe_id,
            "submit_keyframe_id": f.submit_keyframe_id,
            "video_id": f.video_id,
            "keyframe_n": f.keyframe_n,
            "frame_idx": f.frame_idx,
            "fps": f.fps,
            "pts_time": f.pts_time,
            "score": round(f.score, 6),
            "channels": list(f.channels),
            "per_channel_score": {k: round(v, 4) for k, v in f.per_channel_score.items()},
            "keyframe_url": self.media.keyframe_url(f.video_id, f.keyframe_n),
            "video_url": self.media.video_url(f.video_id),
            "evidence": [e.to_dict() for e in f.evidence],
        }

    def _serialize_group(self, g: VideoGroup) -> dict[str, Any]:
        return {
            "video_id": g.video_id,
            "video_score": round(g.video_score, 6),
            "max_score": round(g.max_score, 6),
            "mean_top_score": round(g.mean_top_score, 6),
            "frame_count": g.frame_count,
            "timestamp_dispersion": round(g.timestamp_dispersion, 3),
            "ambiguous": g.ambiguous,
            "channels": list(g.channels),
            "video_url": self.media.video_url(g.video_id),
            "frames": [self._serialize_frame(f) for f in g.frames],
        }


def _apply_filters(
    hits: list[ChannelHit], filters: dict[str, Any], scope_categories: tuple[str, ...] = ()
) -> list[ChannelHit]:
    """Post-filter by the parser's own filters and by the resolved search scope.

    Both narrow to categories, and they are intersected rather than merged: the
    parser's `filters.categories` is what the query itself asked for, the scope is
    what the operator (or the topic heuristic) allowed, and a frame has to satisfy
    both."""
    video_ids = set(filters.get("video_ids") or [])
    categories = set(filters.get("categories") or [])
    scope = set(scope_categories)
    if not video_ids and not categories and not scope:
        return hits
    out = []
    for h in hits:
        if video_ids and h.video_id not in video_ids:
            continue
        category = h.submit_keyframe_id.split("/")[0]
        if categories and category not in categories:
            continue
        if scope and category not in scope:
            continue
        out.append(h)
    return out
