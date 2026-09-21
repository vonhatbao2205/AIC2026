"""Online and offline replay share this engine; labels never enter retrieval."""
from __future__ import annotations

import asyncio
import copy
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..fusion import group_by_video, reciprocal_rank_fusion
from ..services.search_service import SearchService, ServiceUnavailable, VISUAL_CHANNEL
from ..types import Evidence, FusedFrame
from ..retrieval_work import work
from .core import bounded_candidates, gate_local, memory_score, merge_local, rank_evidence, representatives, rescue_pairs, stability
from .models import ProgressiveConfig, RetrievalTrace


@dataclass
class Observation:
    trace: RetrievalTrace
    groups: list[dict]
    parsed: dict
    video_ranks: dict[str, int] = field(default_factory=dict)
    global_cutoffs: dict[str, float | None] = field(init=False)
    localization_frames: dict[str, list[dict]] = field(default_factory=dict)

    def __post_init__(self):
        # Freeze before any rescue/backfill. Empty or failed globals have no
        # observed frontier, even if a later local request succeeds.
        self.global_cutoffs = {
            channel: min((h.score for h in hits if math.isfinite(h.score)), default=None)
            if self.trace.channel_status.get(channel) == "ok" else None
            for channel, hits in self.trace.channel_hits.items()
        }
        if not self.video_ranks:
            self.video_ranks = {g["video_id"]: i + 1 for i, g in enumerate(self.groups)}

    @property
    def valid(self) -> bool:
        return any(s == "ok" for s in self.trace.channel_status.values())


@dataclass
class EngineState:
    ledger: list[dict] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    cumulative: Observation | None = None
    rankings: list[list[str]] = field(default_factory=list)
    snapshot: dict = field(default_factory=lambda: {"groups": [], "turn": 0, "history": []})
    # Cache is session/config owned, and never persisted across model/profile changes.
    vectors: dict = field(default_factory=dict)
    global_cache: dict[str, Observation] = field(default_factory=dict)
    local_cache: dict[tuple, list] = field(default_factory=dict)


class ProgressiveEngine:
    def __init__(self, search: SearchService, config: ProgressiveConfig):
        self.search, self.config = search, config
        self.models = search.resolve_image_models(config.image_models)
        self.categories = search.resolve_scope(config.scope.model_dump(), query="").categories
        self.local_semaphore = asyncio.Semaphore(4)

    async def _global(self, text: str, state: EngineState, counts: dict) -> Observation:
        if text in state.global_cache:
            counts["global_cache_hits"] += 1
            return copy.deepcopy(state.global_cache[text])
        parsed = copy.deepcopy(await self.search.parser.parse(
            text, "T-KIS", [], use_llm=False, translate=self.config.translate,
        ))
        # No TARA/expansion/reranker in v1: preserve a single query per model so
        # rescued raw similarities have exactly the same meaning as global ones.
        parsed["channels"].setdefault("tara", {})["enabled"] = False
        if not self.config.hybrid:
            for name, channel in parsed["channels"].items():
                if name != "image_pe":
                    channel["enabled"] = False
        if not self.search.s.mock_mode and self.config.hybrid and self.search.elastic.mock:
            raise ServiceUnavailable("Live hybrid retrieval requires a configured Elastic index")
        trace = RetrievalTrace()
        measured = {}
        token = work.set(measured)
        try:
            groups, _ = await self.search.retrieve(
                parsed, top_k=self.config.top_k, categories=self.categories,
                image_models=self.models, trace=trace, vector_cache=state.vectors,
            )
        finally:
            work.reset(token)
        trace.work = measured
        counts["index_calls"] += measured.get("index_calls", 0)
        counts["global_index_calls"] += measured.get("index_calls", 0)
        counts["query_vectors"] += measured.get("query_vectors", 0)
        if measured.get("glap_failures"):
            trace.latency.setdefault("warnings", []).append("GLAP failed; audio used Elastic only")
        counts["frames_returned"] += sum(len(h) for h in trace.channel_hits.values())
        observation = Observation(trace, [self.search._serialize_group(g) for g in groups], parsed)
        # Failed/degraded observations are deliberately not memoized as healthy.
        if observation.valid and all(s == "ok" for s in trace.channel_status.values()) and not parsed.get("translation_failed") and not measured.get("glap_failures"):
            state.global_cache[text] = copy.deepcopy(observation)
        return observation

    @staticmethod
    def _tag(observation: Observation, hint_id: str, turn: int):
        for hits in observation.trace.channel_hits.values():
            for hit in hits:
                hit.evidence = copy.deepcopy(hit.evidence) or Evidence(type=hit.channel)
                hit.evidence.extra.update(hint_id=hint_id, origin="global", discovered_at_turn=turn,
                                          original_rank=hit.rank + 1)

    async def _regroup(self, observation: Observation, visible_videos: set[str] | None = None,
                       localization_videos: set[str] | None = None):
        trace = observation.trace
        frames = reciprocal_rank_fusion(trace.channel_hits, weights=trace.weights, k=trace.rrf_k)
        await self.search._enrich_pts(frames)
        observation.groups = [self.search._serialize_group(g) for g in group_by_video(frames)]
        # Preserve ranks in the full raw candidate pool even when evidence memory
        # is evicted. Removing unrelated videos must not promote rank 200 to 1.
        observation.video_ranks = {g["video_id"]: i + 1 for i, g in enumerate(observation.groups)}
        # Display-only frames never enter RRF or group_by_video: even a zero
        # score there could otherwise affect mean/cluster support and ranking.
        local_frames: dict[str, FusedFrame] = {}
        for channel, hits in trace.localization_hits.items():
            for hit in hits:
                frame = local_frames.setdefault(hit.submit_keyframe_id, FusedFrame(
                    hit.submit_keyframe_id, hit.video_id, hit.keyframe_n, hit.pts_time, 0.0,
                ))
                frame.channels.append(channel)
                frame.per_channel_score[channel] = hit.score
                if hit.evidence is not None:
                    frame.evidence.append(hit.evidence)
        await self.search._enrich_pts(list(local_frames.values()))
        observation.localization_frames = {}
        for frame in local_frames.values():
            observation.localization_frames.setdefault(frame.video_id, []).append(self.search._serialize_frame(frame))
        if visible_videos is not None:
            localization = observation.localization_frames
            self._retain(observation, visible_videos)
            if localization_videos is not None:
                observation.localization_frames = {v: f for v, f in localization.items() if v in localization_videos}

    @staticmethod
    def _retain(observation: Observation, videos: set[str]):
        observation.groups = [g for g in observation.groups if g["video_id"] in videos]
        observation.video_ranks = {v: r for v, r in observation.video_ranks.items() if v in videos}
        observation.localization_frames = {v: f for v, f in observation.localization_frames.items() if v in videos}

    async def _local(self, model: str, video: str, obs: Observation, state: EngineState, counts: dict):
        channel = VISUAL_CHANNEL[model]
        queries = obs.trace.queries.get(channel, [])
        if obs.trace.channel_status.get(channel) != "ok" or len(queries) != 1:
            raise ServiceUnavailable("In-video search requires one valid model/query observation")
        key = (model, queries[0], video)
        if key in state.local_cache:
            counts["local_cache_hits"] += 1
            return copy.deepcopy(state.local_cache[key])
        async with self.local_semaphore:
            measured = {}
            token = work.set(measured)
            try:
                rows = await asyncio.wait_for(self.search._search_visual_model(
                    model, queries, self.config.local_k, self.categories,
                    video_id=video, vector_cache=state.vectors,
                ), timeout=20)
            finally:
                work.reset(token)
                counts["index_calls"] += measured.get("index_calls", 0)
                counts["local_index_calls"] += measured.get("index_calls", 0)
                counts["query_vectors"] += measured.get("query_vectors", 0)
        counts["frames_returned"] += len(rows)
        hits = self.search._model_hits(model, rows)
        state.local_cache[key] = copy.deepcopy(hits)
        return hits

    async def _rescue(self, state: EngineState, current: Observation, counts: dict, turn: int) -> list[dict]:
        observations = state.observations
        pairs = rescue_pairs(
            state.rankings[-1] if state.rankings else [],
            [g["video_id"] for g in current.groups],
            {g["video_id"] for g in observations[-1].groups},
            [{g["video_id"] for g in o.groups} for o in observations[:-1]],
            self.config.survivor_limit, self.config.newcomer_limit, self.config.pair_budget,
        )
        counts["admitted_pairs"] = len(pairs)

        async def one(video, index, origin, model):
            started = time.perf_counter()
            try:
                hits = await self._local(model, video, observations[index], state, counts)
                for hit in hits:
                    hit.evidence = copy.deepcopy(hit.evidence) or Evidence(type=hit.channel)
                    hit.evidence.extra.update(hint_id=state.ledger[index]["hint_id"], origin=origin,
                                              discovered_at_turn=turn, local_rank=hit.rank + 1)
                return index, model, hits, {"video_id": video, "hint_id": state.ledger[index]["hint_id"],
                    "origin": origin, "model": model, "status": "ok", "rows": len(hits),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
            except Exception as exc:
                return index, model, [], {"video_id": video, "hint_id": state.ledger[index]["hint_id"],
                    "origin": origin, "model": model, "status": "unavailable", "error_type": type(exc).__name__}

        outcomes = await asyncio.gather(*(one(v, i, origin, m) for v, i, origin in pairs for m in self.models))
        changed: dict[int, set[str]] = {}
        display: dict[int, set[str]] = {}
        for i, model, hits, job in outcomes:
            if hits:
                channel = VISUAL_CHANNEL[model]
                obs = observations[i]
                eligible, localization = gate_local(hits, obs.global_cutoffs.get(channel))
                job.update(global_cutoff=obs.global_cutoffs.get(channel),
                           rank_eligible_rows=len(eligible), localization_only_rows=len(localization))
                if eligible:
                    obs.trace.channel_hits[channel] = merge_local(obs.trace.channel_hits.get(channel, []), eligible)
                obs.trace.localization_hits[channel] = merge_local(obs.trace.localization_hits.get(channel, []), localization)
                visible = changed.setdefault(i, {g["video_id"] for g in obs.groups})
                # A weak backfill must not reactivate an evicted video's old
                # global rank merely because the raw trace still contains it.
                visible.update(h.video_id for h in eligible)
                local_visible = display.setdefault(i, set(obs.localization_frames))
                local_visible.update(h.video_id for h in hits)
        for i in sorted(changed):
            await self._regroup(observations[i], changed[i], display[i])
        return [o[3] for o in outcomes]

    def _rank(self, state: EngineState, cumulative: Observation, method: str) -> list[dict]:
        previous = state.rankings[-1] if state.rankings else []
        current = [g["video_id"] for g in cumulative.groups]
        latest = state.observations[-1]
        stateless = method in {"cumulative", "latest", "dual_view"}
        candidates = set(previous) | set(current) | {g["video_id"] for g in latest.groups}
        if stateless:
            candidates = set(current)
            if method == "dual_view":
                candidates.update(latest.video_ranks)
        observations = [o for o in state.observations if o.valid] if not stateless else []
        ranks = [o.video_ranks for o in observations]
        if method == "hint_rrf":
            # Independent-hint RRF scores the full observed union, including
            # videos outside all previous displayed/retained rankings.
            candidates = {v for ranking in ranks for v in ranking}
        cumulative_ranks = {v: i + 1 for i, v in enumerate(current)}
        scores, memory = {}, {}
        for v in candidates:
            es = [rank_evidence(r.get(v)) for r in ranks]
            memory[v] = memory_score(es, self.config.epsilon, method == "phm_arithmetic")
            if method in {"cumulative", "latest"}:
                scores[v] = rank_evidence(cumulative_ranks.get(v))
            elif method == "dual_view":
                scores[v] = .5 * rank_evidence(latest.video_ranks.get(v)) + .5 * rank_evidence(cumulative_ranks.get(v))
            elif method == "hint_rrf":
                scores[v] = sum(es)
            else:
                w = self.config.memory_weight
                scores[v] = w * memory[v] + (1 - w) * rank_evidence(cumulative_ranks.get(v))
        current_order = sorted(scores, key=lambda v: (-scores[v], v))
        order = (current_order[:self.config.memory_limit] if method == "hint_rrf" else
                 bounded_candidates(scores, current_order, previous if not stateless else [], self.config.memory_limit))
        # Candidate moments come from the cumulative view first. Earlier hints
        # remain explicit evidence, not an implicit final-answer selector.
        preferred = {g["video_id"]: g for g in cumulative.groups}
        by_hint = [{g["video_id"]: g for g in o.groups}
                   if not stateless or i == len(state.observations) - 1 else {}
                   for i, o in enumerate(state.observations)]
        output = []
        for video in order:
            source = preferred.get(video) or next((x[video] for x in reversed(by_hint) if video in x), None)
            if source is None:
                continue
            group = copy.deepcopy(source)
            support, evidence_frames = [], []
            for i, (obs, lookup) in enumerate(zip(state.observations, by_hint)):
                unused = stateless and i < len(state.observations) - 1
                ranked_frames = lookup.get(video, {}).get("frames", [])
                local_frames = obs.localization_frames.get(video, []) if not unused else []
                frames = representatives([*ranked_frames, *local_frames])
                evidence_frames.extend(frames)
                support.append({"hint_id": state.ledger[i]["hint_id"],
                    "status": "not_used" if unused else "observed" if ranked_frames else "localization_only" if local_frames else "unobserved" if obs.valid else "unavailable",
                    "reason": "stateless_baseline" if unused else None if frames else "outside_retained_memory_or_top_k" if obs.valid else "channel_failure",
                    "rank_evidence": rank_evidence(obs.video_ranks.get(video)) if not unused and obs.valid else 0.0,
                    "channels": obs.trace.channel_status, "frames": frames})
            # Preserve the preferred order, append evidence moments for timeline navigation.
            frames = representatives(group["frames"])
            seen = {f["submit_keyframe_id"] for f in frames}
            for f in evidence_frames:
                if f["submit_keyframe_id"] not in seen:
                    seen.add(f["submit_keyframe_id"])
                    frames.append(f)
            times = sorted(f["pts_time"] for f in evidence_frames if f.get("pts_time") is not None)
            group.update(video_score=round(scores[video], 8), frames=frames, frame_count=len(frames))
            group["progressive"] = {"memory_score": memory.get(video, 0),
                "cumulative_rank": cumulative_ranks.get(video),
                "trajectory": [(r.index(video) + 1) if video in r else None for r in state.rankings] + [order.index(video) + 1],
                "hint_evidence": support,
                "dispersed": any(b - a > 30 for a, b in zip(times, times[1:])),
                "moment_source": ("latest" if method in {"latest", "hint_rrf"} else "cumulative")
                    if video in preferred else "delta" if method == "dual_view" else "historical_evidence"}
            output.append(group)
        return output

    async def run(self, ledger: list[dict], state: EngineState | None = None) -> EngineState:
        revision_started = time.perf_counter()
        revision_budget: dict[str, int] = {}
        state = copy.deepcopy(state) if state is not None else EngineState()
        # Editing/retracting a hint invalidates derived memory. Raw request caches
        # remain reusable; no cached merged/rescued rankings are used as globals.
        prefix = ledger[:len(state.ledger)] == state.ledger
        if not prefix:
            state = EngineState(vectors=state.vectors, global_cache=state.global_cache, local_cache=state.local_cache)
        for hint in ledger[len(state.ledger):]:
            started = time.perf_counter()
            counts = {k: 0 for k in ["index_calls", "global_index_calls", "local_index_calls", "frames_returned",
                                     "global_cache_hits", "local_cache_hits", "admitted_pairs", "query_vectors"]}
            method = self.config.method
            turn = len(state.ledger) + 1
            state.ledger.append(copy.deepcopy(hint))
            delta_text = hint["cumulative_text"] if method == "cumulative" else hint["delta_text"]
            delta = await self._global(delta_text, state, counts)
            self._tag(delta, hint["hint_id"], turn)
            await self._regroup(delta)
            state.observations.append(delta)
            cumulative = delta if method in {"latest", "hint_rrf", "cumulative"} else await self._global(hint["cumulative_text"], state, counts)
            state.cumulative = cumulative
            if not delta.valid and not cumulative.valid:
                raise ServiceUnavailable("All progressive retrieval channels failed; previous snapshot retained")
            before = self._rank(state, cumulative, method)
            jobs = []
            if method in {"phm", "phm_arithmetic"} and turn > 1:
                jobs = await self._rescue(state, cumulative, counts, turn)
            groups = self._rank(state, cumulative, method)
            ranking = [g["video_id"] for g in groups]
            if method != "hint_rrf":
                for observation in state.observations:
                    self._retain(observation, set(ranking))
            previous = state.rankings[-1] if state.rankings else []
            observed_stability = stability(previous, ranking, state.rankings)
            warnings = list(dict.fromkeys(delta.trace.latency.get("warnings", []) + cumulative.trace.latency.get("warnings", [])))
            if delta.parsed.get("translation_failed") or cumulative.parsed.get("translation_failed"):
                warnings.append("English translation failed; visual query coverage is degraded")
            if any(j["status"] != "ok" for j in jobs):
                warnings.append("Some in-video retrievals failed; available evidence retained")
            state.rankings.append(ranking)
            state.snapshot = {"groups": groups[:self.config.max_videos], "turn": turn,
                "query": hint["cumulative_text"], "parsed": cumulative.parsed,
                "retrieval_database": self.config.retrieval_database, "image_models": list(self.models),
                "scope": self.search.resolve_scope(self.config.scope.model_dump(), query="").to_dict(),
                "hint_ledger": copy.deepcopy(state.ledger), "stability": observed_stability,
                "history": [{"turn": i + 1, "ranking": list(r)} for i, r in enumerate(state.rankings)],
                "ranking_before_rescue": [g["video_id"] for g in before],
                "budget": counts, "rescue": jobs, "warnings": warnings,
                "degraded": bool(warnings) or not delta.valid or not cumulative.valid or any(
                    status != "ok" for obs in (delta, cumulative) for status in obs.trace.channel_status.values()
                ),
                "latency_ms": {"total_ms": round((time.perf_counter() - started) * 1000, 2)},
                "effective_config": {**self.config.model_dump(), "parser": "heuristic", "rerank": False,
                                     "expand": False, "tara": False,
                                     "rescue_scoring": "global_frontier_v1",
                                     "hint_rrf_memory": "full_observed_union_v1" if method == "hint_rrf" else None,
                                     "dual_view_weights": {"delta": .5, "cumulative": .5} if method == "dual_view" else None},
                "mode": "mock" if self.search.s.mock_mode else "live"}
            for key, value in counts.items():
                revision_budget[key] = revision_budget.get(key, 0) + value
        state.snapshot["revision_budget"] = revision_budget
        state.snapshot["revision_latency_ms"] = round((time.perf_counter() - revision_started) * 1000, 2)
        state.snapshot["no_op"] = not revision_budget
        return state

    @staticmethod
    def trace_export(state: EngineState) -> dict:
        """Full precision observations for auditing; query vectors are not needed."""
        return {"ledger": copy.deepcopy(state.ledger), "rankings": copy.deepcopy(state.rankings),
                "observations": [asdict(o) for o in state.observations],
                "cumulative_observation": asdict(state.cumulative) if state.cumulative else None,
                "snapshot": copy.deepcopy(state.snapshot)}
