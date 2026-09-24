"""Search orchestration: parse -> per-channel retrieval -> RRF -> group-by-video.

Produces the canonical result shape consumed by the frontend, including built
media URLs, channel attribution, evidence, and latency breakdown.
"""
from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Sequence
from typing import Any

from ..adapters.elastic_client import ElasticClient
from ..adapters.milvus_client import MilvusClient
from ..adapters.pe_encoder import GlapEncoderClient, PeEncoderClient
from ..adapters.qwen3_vl_encoder import Qwen3VlEncoderClient
from ..adapters.qwen_reranker import QwenRerankerClient
from ..adapters.tara_encoder import TaraEncoderClient
from ..config import Settings
from ..fusion import DEFAULT_RRF_K, group_by_video, reciprocal_rank_fusion
from ..identity import group_from_video_id
from ..media import MediaUrlBuilder
from ..query_parser import QueryParser
from ..scope import ResolvedScope, resolve_scope
from ..tara_fusion import SCALES, fuse_tara_scales, fuse_video_rankings
from ..types import Channel, ChannelHit, Evidence, FusedFrame, VideoGroup


#: Upper bound of the console's retrieval-depth slider; every returned frame is
#: enriched so it can be submitted.
MAX_ENRICHED_FRAMES = 1000
IMAGE_MODELS = ("pe", "qwen3_vl")
#: The standalone channel each retriever emits when no reranker merges them.
VISUAL_CHANNEL: dict[str, Channel] = {"pe": "image_pe", "qwen3_vl": "image_qwen"}


def _translation_warning(image_models: Sequence[str]) -> str | None:
    """What a failed VI→EN translation actually costs, per selected model.

    Only PE-Core needs English: it is an English-centric CLIP, so an untranslated
    Vietnamese query lands outside the space it was trained on and the keyframes
    look plausible while matching nothing. Qwen3-VL-Embedding is multilingual and
    is queried in Vietnamese by design (see the encoder handoff's reference
    query), so warning about it sends the operator chasing a problem that is not
    there — and a Qwen-only search has nothing to warn about at all.
    """
    if "pe" not in image_models:
        return None
    tail = (
        " Qwen3-VL is unaffected (multilingual)."
        if "qwen3_vl" in image_models
        else ""
    )
    return (
        "Could not translate the query to English. PE Core requires English, so its results "
        "may be poor. Enable the LLM or check the network connection." + tail
    )


class ServiceUnavailable(Exception):
    """A required upstream (PE encoder / Milvus) is unreachable in live mode."""


class SearchService:
    def __init__(self, settings: Settings):
        self.s = settings
        self.elastic = ElasticClient(settings)
        self.milvus = MilvusClient(settings)
        self.pe = PeEncoderClient(settings)
        self.qwen3_vl = Qwen3VlEncoderClient(settings)
        self.tara = TaraEncoderClient(settings)
        self.glap = GlapEncoderClient(settings)
        self.reranker = QwenRerankerClient(settings)
        self.parser = QueryParser(settings)
        self.media = MediaUrlBuilder(settings.keyframe_media_base_url, settings.video_media_base_url)

    def resolve_image_models(self, models: Any = None) -> tuple[str, ...]:
        """Normalize the API selection for internal/direct service callers too.

        The result is in CANONICAL `IMAGE_MODELS` order, not the caller's. Model
        order is not a preference the caller gets to express: it decides which
        retriever wins the round-robin's odd slot when the candidate pool is cut,
        so honouring it would make `["pe","qwen3_vl"]` and `["qwen3_vl","pe"]`
        two different searches over the same two indices.
        """
        selected = tuple(models if models is not None else ("pe",))
        if not selected:
            raise ValueError("At least one image embedding model must be selected")
        if len(set(selected)) != len(selected) or any(m not in IMAGE_MODELS for m in selected):
            raise ValueError(f"Unknown or duplicate image_models: {list(selected)!r}")
        if "qwen3_vl" in selected and not self.s.is_infoshotpp:
            raise ValueError("qwen3_vl image search is available only for infoshotpp")
        return tuple(model for model in IMAGE_MODELS if model in selected)

    # ---- channel runners ----------------------------------------------
    async def _search_visual_model(
        self,
        model: str,
        queries: Sequence[str],
        k: int,
        categories: tuple[str, ...] = (),
        *,
        video_id: str | None = None,
        vector_cache: dict | None = None,
    ) -> list[dict[str, Any]]:
        """One embedding space -> one ranked list of at most `k` rows.

        QUERY EXPANSION is max-fused here and ONLY here: each variant is searched
        separately and a frame keeps its best cosine (a frame matching ANY phrasing
        strongly is kept). That is legal because every variant was encoded by the
        same model into the same collection. The same trick across models would
        not be — PE's 1280-d cosine and Qwen's 4096-d cosine have different
        distributions, which is what rank fusion exists for.
        """
        from ..retrieval_work import count
        encoder = self.pe if model == "pe" else self.qwen3_vl
        searcher = (
            self.milvus.search_image if model == "pe" else self.milvus.search_qwen_image
        )
        if vector_cache is not None:
            if not self.s.mock_mode and (self.milvus.mock or encoder.mock):
                raise ServiceUnavailable(f"{model} is not configured for live progressive retrieval")
            missing = list(dict.fromkeys(q for q in queries if (model, q) not in vector_cache))
            if missing:
                count("query_vectors", len(missing))
                encoded = await encoder.encode_text(missing)
                if len(encoded) != len(missing):
                    raise ServiceUnavailable(f"{model} returned an incomplete vector batch")
                for q, vector in zip(missing, encoded):
                    vector_cache[(model, q)] = vector
            vectors = [vector_cache[(model, q)] for q in queries]
        else:
            count("query_vectors", len(queries))
            vectors = await encoder.encode_text(list(queries))
        count("index_calls", len(vectors))
        search = functools.partial(searcher, top_k=k, categories=categories,
                                   **({"video_id": video_id} if video_id is not None else {}))
        if self.milvus.mock:
            raws = [search(vector) for vector in vectors]
        else:
            raws = await asyncio.gather(
                *(asyncio.to_thread(search, vector) for vector in vectors)
            )
        best: dict[str, dict[str, Any]] = {}
        for raw in raws:
            for row in raw:
                kf_id = row["submit_keyframe_id"]
                if kf_id not in best or row["score"] > best[kf_id]["score"]:
                    best[kf_id] = row
        return sorted(best.values(), key=lambda row: -row["score"])[:k]

    @staticmethod
    def _model_hits(model: str, rows: list[dict[str, Any]]) -> list[ChannelHit]:
        """One retriever's rows as its own ranked channel."""
        channel: Channel = VISUAL_CHANNEL[model]
        return [
            ChannelHit(
                channel=channel,
                submit_keyframe_id=row["submit_keyframe_id"],
                video_id=row["video_id"],
                keyframe_n=int(row["keyframe_n"]),
                score=float(row["score"]),
                rank=rank,
                evidence=Evidence(
                    type=channel,
                    score=float(row["score"]),
                    extra={"model": model},
                ),
            )
            for rank, row in enumerate(rows)
        ]

    async def _run_image_model(
        self,
        model: str,
        cfg: dict[str, Any],
        top_k: int,
        categories: tuple[str, ...] = (),
    ) -> tuple[list[ChannelHit], float]:
        t0 = time.perf_counter()
        queries = [q for q in (cfg.get("queries_en") or []) if q and q.strip()]
        if not queries:
            return [], 0.0
        rows = await self._search_visual_model(model, queries, top_k, categories)
        return self._model_hits(model, rows), (time.perf_counter() - t0) * 1000

    async def _run_image_pe(
        self, cfg: dict[str, Any], top_k: int, categories: tuple[str, ...] = ()
    ) -> tuple[list[ChannelHit], float]:
        """PE-Core text->image as a standalone ranked channel.

        Reranking no longer lives here: it needs the Qwen candidates too, so it
        moved up into `_run_visual`. This entry point is what the canvas service
        (and the no-rerank path) uses to get PE's own list.
        """
        return await self._run_image_model("pe", cfg, top_k, categories)

    async def _run_image_qwen(
        self, cfg: dict[str, Any], top_k: int, categories: tuple[str, ...] = ()
    ) -> tuple[list[ChannelHit], float]:
        """Qwen3-VL text->image retrieval in its independent 4096-d space."""
        return await self._run_image_model("qwen3_vl", cfg, top_k, categories)

    async def _run_tara(
        self, query: str, top_k: int, categories: tuple[str, ...]
    ) -> tuple[list[dict[str, Any]], float]:
        """Search each temporal scale separately, then collapse clips per video."""
        started = time.perf_counter()
        vector = (await self.tara.encode_text([query]))[0]
        depth = min(1000, max(300, top_k * 3))
        rows = await asyncio.gather(*(
            asyncio.to_thread(
                self.milvus.search_tara_clips, vector,
                scale=scale, top_k=depth, categories=categories,
            )
            for scale in SCALES
        ))
        fused = fuse_tara_scales(dict(zip(SCALES, rows)))
        return fused, (time.perf_counter() - started) * 1000

    async def _run_visual(
        self,
        cfg: dict[str, Any],
        top_k: int,
        categories: tuple[str, ...] = (),
        *,
        image_models: tuple[str, ...] = ("pe",),
        rerank: bool = False,
        rerank_info: dict[str, Any] | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        report: dict[str, Any] | None = None,
        vector_cache: dict | None = None,
    ) -> tuple[dict[Channel, list[ChannelHit]], dict[str, float], list[str]]:
        """The whole visual stage: retrieve -> union -> rerank -> ONE ranking.

        THE CANDIDATE UNION HAPPENS BEFORE RERANKING. The previous topology
        reranked the PE pool alone while Qwen ran a separate branch that only met
        PE at the global RRF, so a frame that only Qwen retrieved was never shown
        to the reranker at all: the one model able to judge it against the query
        never saw it, and it arrived at fusion carrying nothing but a Qwen rank.
        Now both retrievers fill one pool, the reranker judges that pool, and the
        result is a single `image_visual` ranking that meets OCR / speech / audio
        in the global RRF.

        Without a reranker there is nothing to merge the two spaces into, so PE
        and Qwen stay two channels and meet in the global RRF exactly as before —
        that is already correct rank fusion, and their cosine values are still
        never added.

        FAIL-OPEN RESTORES THE BASELINE TOPOLOGY, not merely the baseline order:
        a reranker that could not answer hands back `image_pe` + `image_qwen`,
        the same two channels an unticked search produces. Returning the union
        order under a single `image_visual` channel would still have changed the
        final ranking — RRF adds one contribution per channel — after a refinement
        that contributed nothing.

        Returns `(hits per channel, milliseconds per channel, warnings)`. A dead
        model is a warning, not an exception: only losing EVERY selected model
        raises, so one stopped Colab worker degrades the search instead of
        emptying it.
        """
        # A channel the caller selected always reports a latency, even at 0.0:
        # a missing row in the console reads as "not selected", which is a
        # different thing from "selected and found nothing".
        idle: dict[str, float] = {VISUAL_CHANNEL[m]: 0.0 for m in image_models}
        queries = [q for q in (cfg.get("queries_en") or []) if q and q.strip()]
        if not queries or not image_models:
            return {}, idle, []
        rerank_on = bool(rerank and self.reranker.enabled)
        # Reranking only reorders what it is given, so retrieve deeper than the
        # operator asked for — a frame ranked 143rd is invisible to a reranker
        # sent 100 candidates.
        depth = max(top_k, self.reranker.candidate_k) if rerank_on else top_k

        async def run(model: str) -> tuple[str, list[dict[str, Any]], float]:
            started = time.perf_counter()
            extra = {"vector_cache": vector_cache} if vector_cache is not None else {}
            rows = await self._search_visual_model(model, queries, depth, categories, **extra)
            return model, rows, (time.perf_counter() - started) * 1000

        outcomes = await asyncio.gather(
            *(run(model) for model in image_models), return_exceptions=True
        )
        model_rows: dict[str, list[dict[str, Any]]] = {}
        model_ms: dict[str, float] = {}
        warnings: list[str] = []
        failures: list[BaseException] = []
        alive = 0
        for model, outcome in zip(image_models, outcomes):
            if isinstance(outcome, BaseException):
                warnings.append(f"{VISUAL_CHANNEL[model]} channel unavailable: {outcome}")
                failures.append(outcome)
                continue
            alive += 1
            _, rows, ms = outcome
            model_ms[model] = ms
            if rows:
                model_rows[model] = rows
        if not alive and failures:
            raise failures[0]
        stage_report = report if report is not None else {}
        stage_report.update({
            "models": {
                model: {"rows": len(model_rows.get(model, ())), "ms": round(ms, 1)}
                for model, ms in model_ms.items()
            },
            "merged": rerank_on,
            "rrf_k": rrf_k,
        })
        if not model_rows:
            return {}, idle, warnings

        def baseline() -> tuple[dict[Channel, list[ChannelHit]], dict[str, float]]:
            """Each retriever as its own channel — the shape a search WITHOUT the
            rerank tick box produces.

            `rows[:top_k]` is exactly what a `top_k` retrieval would have returned:
            the deeper candidate list is the same cosine ordering, so falling back
            here lands on the baseline result rather than near it.
            """
            hits: dict[Channel, list[ChannelHit]] = {}
            latency = dict(idle)
            for model, rows in model_rows.items():
                channel = VISUAL_CHANNEL[model]
                hits[channel] = self._model_hits(model, rows[:top_k])
                latency[channel] = model_ms[model]
            return hits, latency

        if not rerank_on:
            hits, latency = baseline()
            return hits, latency, warnings

        started = time.perf_counter()
        pool = _union_visual_candidates(model_rows, rrf_k=rrf_k, pool_size=depth)
        stage_report["pool"] = len(pool)
        stage_report["pool_by_model"] = {
            model: sum(1 for row in pool if model in row["retrievers"])
            for model in model_rows
        }
        rerank_report = rerank_info if rerank_info is not None else {}
        ranked = await self._rerank_frames(queries[0], pool, rerank_report)
        if rerank_report.get("ok") is False:
            # FAIL-OPEN IS A TOPOLOGY DECISION, not just an ordering one. RRF adds
            # one contribution PER CHANNEL, so collapsing PE and Qwen into a single
            # `image_visual` channel re-weights the whole fusion against
            # OCR/speech/audio — and it would do that even here, where the reranker
            # scored nothing at all. A refinement that died has to leave the
            # pipeline in the exact shape it would have had if the operator had
            # never ticked the box, which means handing back both channels.
            stage_report["merged"] = False
            hits, latency = baseline()
            return hits, latency, warnings

        ordered = ranked[:top_k]
        merged: list[ChannelHit] = []
        for rank, row in enumerate(ordered):
            # DISPLAY ONLY. RRF fuses on rank, so this value never decides an
            # ordering — which is what lets a rerank logit and the union's RRF
            # score (for candidates the worker could not reach) sit in one field.
            scored = row.get("rerank_score") is not None
            display = float(row["rerank_score"] if scored else row["union_score"])
            merged.append(
                ChannelHit(
                    channel="image_visual",
                    submit_keyframe_id=row["submit_keyframe_id"],
                    video_id=row["video_id"],
                    keyframe_n=int(row["keyframe_n"]),
                    score=display,
                    rank=rank,
                    evidence=Evidence(
                        type="image_visual",
                        score=display,
                        extra={
                            # Which retriever(s) actually found the frame is still
                            # the operator's read on WHY it is here, so the merge
                            # keeps it rather than flattening it into "visual".
                            "models": list(row["retrievers"]),
                            "per_model_score": {
                                model: round(score, 6)
                                for model, score in row["per_model_score"].items()
                            },
                            "union_score": round(float(row["union_score"]), 6),
                            **(
                                {"rerank_score": round(float(row["rerank_score"]), 4)}
                                if scored
                                else {}
                            ),
                        },
                    ),
                )
            )
        stage_report["ms"] = round((time.perf_counter() - started) * 1000, 1)
        # The two retrievals ran concurrently, so the stage cost is the slower of
        # them plus the union+rerank that had to wait for both.
        retrieval_ms = max(model_ms.values(), default=0.0)
        return (
            {"image_visual": merged},
            {"image_visual": retrieval_ms + stage_report["ms"]},
            warnings,
        )

    async def _rerank_frames(
        self,
        query: str,
        rows: list[dict[str, Any]],
        info: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Reorder visual candidates by Qwen3-VL relevance. FAIL-OPEN.

        `retrieve()` turns any channel exception into an empty channel, so letting
        a rerank failure escape would delete the whole visual result the moment the
        Colab tunnel drops. A refinement is not allowed to cost more than it adds:
        every failure path here returns the retrieval order untouched.

        `rows` is the UNION pool, so the reranker is judging PE and Qwen candidates
        against each other — the one place in the pipeline where candidates from
        two embedding spaces are compared by something that actually looks at the
        image rather than by their incomparable cosine values.

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
        # its own top_k) keep their retrieval order behind the reranked ones —
        # dropping them would lose recall the retrievers had already paid for.
        ordered.extend(by_id.values())
        report.update({
            "ok": True,
            "candidates": len(rows),
            "reranked": len(ordered) - len(by_id),
            "ms": round((time.perf_counter() - started) * 1000, 1),
        })
        return ordered

    async def _run_similar(
        self,
        seed_ids: list[str],
        top_k: int,
        categories: tuple[str, ...] = (),
        image_models: tuple[str, ...] = ("pe",),
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
        async def run_model(model: str) -> tuple[str, list[dict[str, Any]], int]:
            getter = (
                self.milvus.get_image_vectors
                if model == "pe"
                else self.milvus.get_qwen_image_vectors
            )
            searcher = (
                self.milvus.search_image
                if model == "pe"
                else self.milvus.search_qwen_image
            )
            vectors = await asyncio.to_thread(getter, seed_ids)
            if not vectors:
                return model, [], 0
            raws = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        searcher, vector, top_k=top_k, categories=categories
                    )
                    for vector in vectors.values()
                )
            )
            # Scores are comparable across seeds encoded by the same model.
            best: dict[str, dict[str, Any]] = {}
            for raw in raws:
                for row in raw:
                    kf_id = row["submit_keyframe_id"]
                    if kf_id not in best or row["score"] > best[kf_id]["score"]:
                        best[kf_id] = row
            ranked = sorted(best.values(), key=lambda row: -row["score"])[:top_k]
            return model, ranked, len(vectors)

        outcomes = await asyncio.gather(
            *(run_model(model) for model in image_models), return_exceptions=True
        )
        model_lists: dict[str, list[dict[str, Any]]] = {}
        seed_counts: dict[str, int] = {}
        failures: list[BaseException] = []
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                failures.append(outcome)
                continue
            model, rows, seed_count = outcome
            if rows:
                model_lists[model] = rows
                seed_counts[model] = seed_count
        if not model_lists:
            if failures:
                raise failures[0]
            return [], (time.perf_counter() - t0) * 1000

        # One selected model preserves its own cosine ordering. With two models,
        # fuse ranks only; adding 1280-d PE cosine to 4096-d Qwen cosine is invalid.
        if len(model_lists) == 1:
            only_model, fused = next(iter(model_lists.items()))
            model_scores = {
                row["submit_keyframe_id"]: {only_model: float(row["score"])} for row in fused
            }
        else:
            by_id: dict[str, dict[str, Any]] = {}
            for model, rows in model_lists.items():
                for rank, row in enumerate(rows):
                    entry = by_id.setdefault(
                        row["submit_keyframe_id"],
                        {"row": row, "score": 0.0, "model_scores": {}},
                    )
                    entry["score"] += 1.0 / (60 + rank + 1)
                    entry["model_scores"][model] = float(row["score"])
            ordered = sorted(by_id.values(), key=lambda item: (-item["score"], item["row"]["submit_keyframe_id"]))
            fused = [{**entry["row"], "score": entry["score"]} for entry in ordered[:top_k]]
            model_scores = {
                entry["row"]["submit_keyframe_id"]: entry["model_scores"]
                for entry in ordered[:top_k]
            }
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
                    extra={
                        "models": list(model_lists),
                        "seeds": sum(seed_counts.values()),
                        "per_model_score": model_scores[r["submit_keyframe_id"]],
                    },
                ),
            )
            for i, r in enumerate(fused)
        ]
        return hits, (time.perf_counter() - t0) * 1000

    async def _run_ocr(
        self, cfg: dict[str, Any], top_k: int, categories: tuple[str, ...] = ()
    ) -> tuple[list[ChannelHit], float]:
        from ..retrieval_work import count
        count("index_calls")
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
        from ..retrieval_work import count
        count("index_calls")
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
        from ..retrieval_work import count
        t0 = time.perf_counter()
        queries = cfg.get("queries_en") or []
        labels = cfg.get("sound_labels_en") or []

        elastic_task = self.elastic.search_audio(queries, labels, categories=categories, size=top_k)
        glap_evidence: dict[str, Evidence] = {}
        glap_raw: list[dict[str, Any]] = []
        if self.glap is not None and not self.glap.mock and (queries or labels):
            try:
                count("query_vectors", len(queries or labels))
                vectors = await self.glap.encode_text(queries or labels)
                vec = vectors[0]
                count("index_calls")
                glap_raw = await asyncio.to_thread(
                    self.milvus.search_audio, vec, top_k=top_k, categories=categories
                )
            except Exception:  # noqa: BLE001 - GLAP optional; fall back to Elastic only
                count("glap_failures")
                glap_raw = []
        count("index_calls")
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
        image_models: tuple[str, ...] = ("pe",),
        trace: Any = None,
        vector_cache: dict | None = None,
    ) -> tuple[list[VideoGroup], dict[str, Any]]:
        """`categories` is the resolved search scope: the dataset folders this
        search may return frames from. It is pushed into every channel's own
        query (so top_k is filled from inside the scope) AND re-applied to the
        hits, which is what keeps mock mode and any future channel honest."""
        channels_cfg = parsed.get("channels", {})
        filters = parsed.get("filters", {})
        unsupported = self.s.unsupported_channels
        image_models = self.resolve_image_models(image_models)
        rrf_k = (parsed.get("rerank_policy") or {}).get("rrf_k", DEFAULT_RRF_K)
        # Per-call, so two concurrent searches never write each other's report.
        rerank_info: dict[str, Any] = {}
        visual_info: dict[str, Any] = {}
        runners = {
            name: runner
            for name, runner in (
                ("ocr", self._run_ocr),
                ("speech", self._run_speech),
                ("audio", self._run_audio),
            )
            if name not in unsupported
        }
        feedback = feedback or {}
        tasks: dict[str, asyncio.Task] = {}
        # The visual stage is ONE task for both retrievers plus the reranker: the
        # reranker has to see PE's and Qwen's candidates in the same pool, which
        # it cannot do while they are two independent channel tasks.
        visual_models = tuple(
            model for model in image_models if VISUAL_CHANNEL[model] not in unsupported
        )
        visual_cfg = channels_cfg.get("image_pe", {})
        if visual_cfg.get("enabled") and visual_models:
            tasks["visual"] = asyncio.create_task(
                self._run_visual(
                    visual_cfg,
                    top_k,
                    categories,
                    image_models=visual_models,
                    rerank=rerank,
                    rerank_info=rerank_info,
                    rrf_k=rrf_k,
                    report=visual_info,
                    **({"vector_cache": vector_cache} if vector_cache is not None else {}),
                )
            )
        tara_cfg = channels_cfg.get("tara") or {}
        tara_query = (tara_cfg.get("queries_en") or [
            parsed.get("translated_en_visual") or parsed.get("original_query") or ""
        ])[0]
        if (self.s.has_tara_search and tara_cfg.get("enabled", True)
                and parsed.get("query_type") in {"T-KIS", "QA"}
                and not parsed.get("translation_failed") and tara_query.strip()):
            tasks["tara"] = asyncio.create_task(
                self._run_tara(tara_query, top_k, categories)
            )
        for name, runner in runners.items():
            cfg = channels_cfg.get(name, {})
            if cfg.get("enabled"):
                tasks[name] = asyncio.create_task(runner(cfg, top_k, categories))
        # Feedback-driven, not parser-driven: it exists only while the operator
        # keeps frames marked.
        positive_frames = [str(kf) for kf in (feedback.get("positive_frames") or []) if kf]
        if positive_frames:
            tasks["similar"] = asyncio.create_task(
                self._run_similar(
                    positive_frames, top_k, categories, image_models=image_models
                )
            )

        latency: dict[str, Any] = {"channels": {}}
        channel_hits: dict[Channel, list[ChannelHit]] = {}
        tara_videos: list[dict[str, Any]] = []
        weights: dict[str, float] = {}
        warnings: list[str] = []
        channel_status: dict[str, str] = {}
        if parsed.get("translation_failed") and self.s.has_tara_search and tara_cfg.get("enabled", True):
            warnings.append("TARA skipped: English translation failed")
        disabled = [
            name for name in sorted(unsupported) if channels_cfg.get(name, {}).get("enabled")
        ]
        if disabled:
            warnings.append(
                f"{self.s.retrieval_database} has no data for channels: " + ", ".join(disabled)
            )
        if self.s.ocr_missing_categories and "ocr" in tasks:
            warnings.append(
                "OCR "
                + self.s.retrieval_database
                + " has no index for categories: "
                + ", ".join(self.s.ocr_missing_categories)
            )
        for name, task in tasks.items():
            if name == "tara":
                try:
                    tara_videos, tara_ms = await task
                except Exception as exc:  # noqa: BLE001 - preserve existing search
                    tara_videos, tara_ms = [], 0.0
                    warnings.append(f"tara channel unavailable: {exc}")
                video_ids = set(filters.get("video_ids") or [])
                parser_categories = set(filters.get("categories") or [])
                tara_videos = [
                    item for item in tara_videos
                    if (not video_ids or item["video_id"] in video_ids)
                    and (not parser_categories or group_from_video_id(item["video_id"]) in parser_categories)
                ]
                latency["channels"]["tara"] = round(tara_ms, 1)
                weights["tara"] = float(tara_cfg.get("weight") or 1.0)
                continue
            if name == "visual":
                try:
                    visual_hits, visual_ms, visual_warnings = await task
                except Exception as exc:  # noqa: BLE001 - a dead stage is not an outage
                    # Every selected retriever is gone; name the channels the
                    # console expected so a blank visual row reads as a failure
                    # rather than as "nothing matched".
                    visual_warnings = [f"visual channel unavailable: {exc}"]
                    visual_hits = {}
                    visual_ms = {VISUAL_CHANNEL[m]: 0.0 for m in visual_models}
                warnings.extend(visual_warnings)
                # Weights are configured on `image_pe`: it is the master switch
                # for the whole visual stage, whichever channel(s) it emits.
                visual_weight = float(channels_cfg.get("image_pe", {}).get("weight") or 1.0)
                for channel in visual_ms:
                    hits = visual_hits.get(channel, [])
                    channel_hits[channel] = _apply_filters(hits, filters, categories)
                    latency["channels"][channel] = round(visual_ms[channel], 1)
                    weights[channel] = visual_weight
                    model = next((m for m in visual_models if VISUAL_CHANNEL[m] == channel), None)
                    channel_status[channel] = "ok" if (
                        model in visual_info.get("models", {}) or channel == "image_visual"
                    ) else "unavailable"
                continue
            try:
                hits, ms = await task
                channel_status[name] = "ok"
            except Exception as exc:  # noqa: BLE001 - one dead channel must not kill the search
                hits, ms = [], 0.0
                channel_status[name] = "unavailable"
                warnings.append(f"{name} channel unavailable: {exc}")
            channel_hits[name] = _apply_filters(hits, filters, categories)
            latency["channels"][name] = round(ms, 1)
            weights[name] = float(channels_cfg.get(name, {}).get("weight") or 1.0)
        if visual_info:
            latency["visual"] = visual_info
        if rerank_info:
            latency["reranker"] = rerank_info
            if not rerank_info.get("ok"):
                warnings.append(
                    "Reranker unavailable; preserving retrieval order: "
                    + str(rerank_info.get("error") or "unknown")
                )
        if warnings:
            latency["warnings"] = warnings

        t_fuse = time.perf_counter()
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
        if tara_videos:
            groups = fuse_video_rankings(
                groups, channel_hits, tara_videos, weights=weights, k=rrf_k,
                positive_videos=set(feedback.get("positive_videos") or []),
                negative_videos=set(feedback.get("negative_videos") or []),
                negative_frames=set(feedback.get("negative_frames") or []),
            )
        latency["fusion_ms"] = round((time.perf_counter() - t_fuse) * 1000, 1)
        if trace is not None:
            import copy
            trace.channel_hits = copy.deepcopy(channel_hits)
            trace.channel_status = channel_status
            trace.weights = dict(weights)
            trace.categories = categories
            trace.rrf_k = rrf_k
            trace.latency = copy.deepcopy(latency)
            trace.queries = {
                name: list((channels_cfg.get("image_pe" if name in {"image_pe", "image_qwen", "image_visual"} else name) or {}).get("queries_en") or
                           (channels_cfg.get(name) or {}).get("queries_vi") or [])
                for name in channel_hits
            }
            trace.query_config = copy.deepcopy(channels_cfg)
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
        image_models: Any = None,
        translate: bool = True,
    ) -> dict[str, Any]:
        """Flat visual search over the selected embedding model(s).

        Same topology as the full pipeline's visual stage. Without reranking, one
        model keeps its own cosine order and two models are fused by RRF, never by
        adding their incompatible cosine scores. With reranking, both retrievers
        fill ONE candidate pool first and the reranker judges that pool — so a
        frame only Qwen found is scored against the query like any other, instead
        of skipping the reranker because it was not in PE's list.
        """
        t0 = time.perf_counter()
        query = (query or "").strip()
        selected = self.resolve_image_models(image_models)
        scope = self.resolve_scope(scope_spec, query=query)
        if not query:
            return {
                "query": query,
                "results": [],
                "retrieval_database": self.s.retrieval_database,
                "image_models": list(selected),
                "scope": scope.to_dict(),
                "mode": "mock" if self.s.mock_mode else "live",
                "latency_ms": 0,
            }
        # Use the same translated visual query for a fair PE/Qwen rank ensemble.
        # `translate` is the operator's tick box: off means the query goes to the
        # encoders exactly as typed (an English query, a proper noun, a phrase
        # the translator keeps mangling).
        search_text = query
        translation_failed = False
        if translate and self.s.translate_to_en and not self.s.mock_mode:
            from ..translate import translate_vi_to_en_status

            search_text, translated_ok = await translate_vi_to_en_status(
                query, settings=self.s
            )
            translation_failed = not translated_ok

        rerank_on = bool(rerank and self.reranker.enabled)
        rerank_info: dict[str, Any] = {}
        model_latency: dict[str, float] = {}

        # Every selected retriever over-retrieves when reranking, not just PE:
        # the pool the reranker sees has to be deeper than the answer, in BOTH
        # spaces, or the union is only wide on one side.
        depth = max(top_k, self.reranker.candidate_k) if rerank_on else top_k

        async def run_model(model: str) -> tuple[str, list[dict[str, Any]]]:
            started = time.perf_counter()
            rows = await self._search_visual_model(
                model, [search_text], depth, scope.categories
            )
            model_latency[model] = round((time.perf_counter() - started) * 1000, 1)
            return model, rows

        outcomes = await asyncio.gather(
            *(run_model(model) for model in selected), return_exceptions=True
        )
        model_rows: dict[str, list[dict[str, Any]]] = {}
        warnings: list[str] = []
        for model, outcome in zip(selected, outcomes):
            if isinstance(outcome, BaseException):
                warnings.append(f"{model} image search unavailable: {outcome}")
                continue
            returned_model, rows = outcome
            # The reranker needs the full over-retrieved list; every other path
            # only ever wants the answer depth.
            model_rows[returned_model] = rows if rerank_on else rows[:top_k]
        if not model_rows:
            detail = "; ".join(warnings) or "no image model returned results"
            raise ServiceUnavailable(detail)

        per_model_score: dict[str, dict[str, float]] = {}
        row_by_id: dict[str, dict[str, Any]] = {}
        for model, rows in model_rows.items():
            for row in rows:
                kf_id = row["submit_keyframe_id"]
                per_model_score.setdefault(kf_id, {})[model] = float(row["score"])
                row_by_id.setdefault(kf_id, row)

        def baseline_rows() -> list[dict[str, Any]]:
            """The ranking a search WITHOUT the rerank tick box produces.

            One model keeps its own cosine order; two are fused by RRF over their
            full lists. Kept in one place because the rerank path has to be able
            to fall back onto EXACTLY this when the worker dies.
            """
            if len(model_rows) == 1:
                only_model, raw = next(iter(model_rows.items()))
                return [
                    {**row, "fused_score": float(row["score"]), "models": [only_model]}
                    for row in raw[:top_k]
                ]
            channel_hits: dict[Channel, list[ChannelHit]] = {}
            for model, rows in model_rows.items():
                channel = VISUAL_CHANNEL[model]
                channel_hits[channel] = [
                    ChannelHit(
                        channel=channel,
                        submit_keyframe_id=row["submit_keyframe_id"],
                        video_id=row["video_id"],
                        keyframe_n=int(row["keyframe_n"]),
                        score=float(row["score"]),
                        rank=rank,
                    )
                    for rank, row in enumerate(rows)
                ]
            fused_frames = reciprocal_rank_fusion(channel_hits, k=DEFAULT_RRF_K)[:top_k]
            return [
                {
                    **row_by_id[frame.submit_keyframe_id],
                    "fused_score": frame.score,
                    "models": [
                        "pe" if channel == "image_pe" else "qwen3_vl"
                        for channel in frame.channels
                    ],
                }
                for frame in fused_frames
            ]

        if not rerank_on:
            ranked_rows = baseline_rows()
        else:
            # Union BEFORE the reranker, exactly as in `_run_visual`: one pool,
            # one verdict, one ranking.
            pool = _union_visual_candidates(
                model_rows, rrf_k=DEFAULT_RRF_K, pool_size=depth
            )
            ordered = await self._rerank_frames(search_text, pool, rerank_info)
            if rerank_info.get("ok") is False:
                # Same rule as `_run_visual`: a dead reranker returns the search to
                # its baseline, and the pool is NOT that baseline — the round-robin
                # cut it to `depth` candidates in TOTAL, so fusing it would answer
                # differently from an unticked search over the same two indices.
                model_rows = {model: rows[:top_k] for model, rows in model_rows.items()}
                ranked_rows = baseline_rows()
            else:
                ranked_rows = [
                    {
                        **row,
                        "fused_score": float(
                            row["rerank_score"]
                            if row.get("rerank_score") is not None
                            else row["union_score"]
                        ),
                        "models": list(row["retrievers"]),
                    }
                    for row in ordered[:top_k]
                ]

        results = [
            {
                "image_id": r["submit_keyframe_id"],
                "submit_keyframe_id": r["submit_keyframe_id"],
                "video_id": r["video_id"],
                "keyframe_n": int(r["keyframe_n"]),
                "score": round(float(r["fused_score"]), 6),
                "models": r["models"],
                "per_model_score": {
                    model: round(score, 6)
                    for model, score in per_model_score[r["submit_keyframe_id"]].items()
                },
                "rerank_score": (
                    round(float(r["rerank_score"]), 4)
                    if r.get("rerank_score") is not None
                    else None
                ),
                "keyframe_url": self.media.keyframe_url(r["video_id"], int(r["keyframe_n"])),
                "video_url": self.media.video_url(r["video_id"]),
            }
            for r in ranked_rows
        ]
        body: dict[str, Any] = {
            "query": query,
            "retrieval_database": self.s.retrieval_database,
            "image_models": list(selected),
            "scope": scope.to_dict(),
            "translated_query": search_text if search_text != query else None,
            "results": results,
            "mode": "mock" if self.s.mock_mode else "live",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model_latency_ms": model_latency,
        }
        if rerank_info:
            body["reranker"] = rerank_info
        if translation_failed:
            notice = _translation_warning(selected)
            if notice:
                warnings.append(notice)
        if warnings:
            body["warnings"] = warnings
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
        image_models = self.resolve_image_models(req.get("image_models"))
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
                translate=bool(req.get("translate", True)),
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
            image_models=image_models,
        )
        if parsed.get("translation_failed"):
            notice = _translation_warning(image_models)
            if notice:
                latency.setdefault("warnings", []).append(notice)
        latency["parse_ms"] = round(parse_ms, 1)
        latency["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)

        return {
            "query": query,
            "retrieval_database": self.s.retrieval_database,
            "image_models": list(image_models),
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
            "best_clip": g.best_clip,
        }


def _union_visual_candidates(
    model_rows: dict[str, list[dict[str, Any]]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    pool_size: int,
) -> list[dict[str, Any]]:
    """Merge the retrievers' ranked lists into ONE candidate pool to rerank.

    Two rules, and they answer two different questions.

    WHO GETS IN — selection is round-robin over the models' ranks, so every
    retriever is guaranteed roughly `pool_size / n_models` of the pool. A frame
    only Qwen found, at Qwen rank 3, is in the pool even when PE's list is longer
    and scores higher on paper. Taking the top `pool_size` of the fused order
    instead would let one model's deep tail crowd out the other model's head,
    which is exactly the model-diversity loss this stage exists to prevent.

    IN WHAT ORDER — RRF over the same lists, rank-only. PE cosine and Qwen cosine
    are never added; agreement between two independent spaces shows up as two
    rank contributions instead. That order is what the reranker receives, and,
    more importantly, what survives untouched when the reranker is down.

    With one model selected both rules collapse to "its own cosine order", so a
    PE-only search reaches the reranker exactly as it did before.

    NOTE ON DEPTH: the pool is capped at `pool_size` in total, not per model, so
    the GPU cost of reranking does not double when a second retriever is ticked.
    Two models therefore each contribute about half the depth one model would.
    Raise `QWEN_RERANKER_CANDIDATES` if you want both retrievers at full depth.
    """
    # Canonical order, never `model_rows`' insertion order: this function is a
    # pure ranking primitive and must not answer differently because a caller
    # built its dict the other way round.
    models = [model for model in IMAGE_MODELS if model in model_rows]
    entries: dict[str, dict[str, Any]] = {}
    for model in models:
        rows = model_rows[model]
        for rank, row in enumerate(rows):
            kf_id = row["submit_keyframe_id"]
            entry = entries.get(kf_id)
            if entry is None:
                entry = entries[kf_id] = {
                    "row": dict(row),
                    "union_score": 0.0,
                    "retrievers": [],
                    "per_model_score": {},
                    "best_rank": rank,
                }
            entry["union_score"] += 1.0 / (rrf_k + rank + 1)
            entry["per_model_score"][model] = float(row["score"])
            if model not in entry["retrievers"]:
                entry["retrievers"].append(model)
            entry["best_rank"] = min(entry["best_rank"], rank)

    selected: list[str] = []
    seen: set[str] = set()
    deepest = max((len(rows) for rows in model_rows.values()), default=0)
    for rank in range(deepest):
        for model in models:
            rows = model_rows[model]
            if rank >= len(rows):
                continue
            kf_id = rows[rank]["submit_keyframe_id"]
            if kf_id in seen:
                continue
            seen.add(kf_id)
            selected.append(kf_id)
            if len(selected) >= pool_size:
                break
        if len(selected) >= pool_size:
            break

    pool = [entries[kf_id] for kf_id in selected]
    pool.sort(
        key=lambda e: (-e["union_score"], e["best_rank"], e["row"]["submit_keyframe_id"])
    )
    return [
        {
            **entry["row"],
            "union_score": entry["union_score"],
            "retrievers": [m for m in IMAGE_MODELS if m in entry["retrievers"]],
            "per_model_score": entry["per_model_score"],
        }
        for entry in pool
    ]


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
