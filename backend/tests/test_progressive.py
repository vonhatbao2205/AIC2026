import asyncio
import copy
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.progressive.core import derive_ledger, gate_local, memory_score, merge_local, rescue_pairs, bounded_candidates
from app.progressive.engine import ProgressiveEngine, Observation
from app.progressive.models import HintInput, HintUpdate, ProgressiveConfig, RetrievalTrace
from app.progressive.replay import replay_events
from app.services.progressive_service import ProgressiveService, ProgressiveError
from app.services.search_service import SearchService
from app.types import ChannelHit


def hints(*texts, mode="delta"):
    return [HintInput(hint_id=f"h{i}", raw_text=t, input_mode=mode) for i, t in enumerate(texts)]


def test_ledger_cumulative_duplicate_edit_and_reveal():
    delta = derive_ledger(hints("a red car", "on a bridge"))
    cumulative = derive_ledger(hints("a red car", "a red car on a bridge", mode="cumulative"))
    assert [x["delta_text"] for x in delta] == [x["delta_text"] for x in cumulative]
    assert len(derive_ledger(hints("a red car", "a red car"))) == 1
    edited = derive_ledger(hints("a red car", "a blue car", mode="cumulative"))
    assert [x["delta_text"] for x in edited] == ["a blue car"]
    future = hints("now", "FUTURE_SECRET")
    future[1].revealed_at = 30000
    assert "FUTURE_SECRET" not in json.dumps(derive_ledger(future, 29999))
    assert len(derive_ledger(future, 30000)) == 2
    future[0].enabled = False
    assert derive_ledger(future, 30000)[0]["cumulative_text"] == "FUTURE_SECRET"


def test_only_local_hits_crossing_frontier_enter_ranking_and_models_stay_separate():
    def hit(video, score, channel="image_pe"):
        return ChannelHit(channel, video + ":1", video, 1, score, 0)
    global_hits = [hit("00047", .9), hit("00048", .8)]
    local = [hit("00049", .2), hit("00047", .9), hit("00050", .8), hit("00051", .85)]
    eligible, localization = gate_local(local, .8)
    merged = merge_local(global_hits, eligible)
    assert [h.video_id for h in merged] == ["00047", "00051", "00048", "00050"]
    assert [h.rank for h in merged] == [0, 1, 2, 3]
    assert [h.video_id for h in localization] == ["00049"]
    assert localization[0].evidence.extra == {
        "local_score": .2, "global_cutoff": .8, "below_global_cutoff": True,
        "rank_eligible": False, "ranking_exclusion": "below_global_cutoff", "rank_evidence": 0.0,
    }
    assert local[0].evidence is None
    assert global_hits[1].rank == 0  # no mutation of the original trace
    with pytest.raises(ValueError):
        merge_local(global_hits, [hit("other", .99, "image_qwen")])
    assert memory_score([1, 0]) == pytest.approx(.1 ** .5)


@pytest.mark.parametrize("cutoff", [None, float("nan"), float("inf")])
def test_unknown_frontier_cannot_grant_rank_evidence(cutoff):
    hit = ChannelHit("image_pe", "00047:1", "00047", 1, .99, 0)
    eligible, localization = gate_local([hit], cutoff)
    assert not eligible
    assert localization[0].evidence.extra["ranking_exclusion"] == "unknown_global_cutoff"
    assert localization[0].evidence.extra["rank_evidence"] == 0


def test_frontier_is_original_finite_successful_global_observation():
    hits = [ChannelHit("image_pe", str(i), str(i), 1, score, i)
            for i, score in enumerate([.9, .8, float("nan")])]
    obs = Observation(RetrievalTrace(channel_hits={"image_pe": hits}, channel_status={"image_pe": "ok"}), [], {})
    assert obs.global_cutoffs == {"image_pe": .8}
    obs.trace.channel_hits["image_pe"] = hits[:1]
    assert copy.deepcopy(obs).global_cutoffs == {"image_pe": .8}
    empty = Observation(RetrievalTrace(channel_hits={"image_pe": []}, channel_status={"image_pe": "ok"}), [], {})
    failed = Observation(RetrievalTrace(channel_hits={"image_pe": hits}, channel_status={"image_pe": "unavailable"}), [], {})
    assert empty.global_cutoffs == failed.global_cutoffs == {"image_pe": None}
    assert gate_local([hits[2]], .8) == ([], [])


def test_memory_bound_and_pair_budget():
    scores = {f"v{i:03}": i / 1000 for i in range(500)}
    order = bounded_candidates(scores, list(scores)[:20], list(scores)[20:40], 250)
    assert len(order) == 250
    assert set(list(scores)[:40]) <= set(order)
    jobs = rescue_pairs(list(scores)[:30], list(scores)[30:60], set(), [set()] * 10)
    assert len(jobs) == 16
    assert sum(origin == "rescue" for _, _, origin in jobs) == 8
    assert len({v for v, _, origin in jobs if origin == "backfill"}) == 4


def test_eviction_drops_evidence_without_promoting_remaining_pool_rank():
    observation = Observation(RetrievalTrace(), [{"video_id": f"v{i:03}"} for i in range(300)], {})
    ProgressiveEngine._retain(observation, {"v199", "v249"})
    assert [g["video_id"] for g in observation.groups] == ["v199", "v249"]
    assert observation.video_ranks == {"v199": 200, "v249": 250}


@pytest.mark.asyncio
async def test_h1_baseline_equivalence_duplicate_no_cost_and_snapshot_immutable(settings):
    search = SearchService(settings)
    config = ProgressiveConfig(translate=False, top_k=20)
    engine = ProgressiveEngine(search, config)
    ledger = derive_ledger(hints("a red car"))
    state = await engine.run(ledger)
    baseline = await ProgressiveEngine(search, config.model_copy(update={"method": "cumulative"})).run(ledger)
    assert state.snapshot["groups"]
    assert state.rankings == baseline.rankings
    frozen = copy.deepcopy(state.snapshot)
    second = await engine.run(derive_ledger(hints("a red car", "on a bridge")), state)
    assert state.snapshot == frozen
    assert second.rankings[0] == state.rankings[0]
    assert second.snapshot["turn"] == 2
    assert second.snapshot["budget"]["admitted_pairs"] <= 16
    same = await engine.run(derive_ledger(hints("a red car", "a red car")), state)
    assert same.vectors == state.vectors
    assert same.rankings == state.rankings
    edited = await engine.run(derive_ledger(hints("a blue car")), second)
    assert edited.snapshot["turn"] == 1
    assert len(edited.observations) == 1
    assert "a red car" not in json.dumps(edited.snapshot)


@pytest.mark.asyncio
async def test_session_idempotency_failure_ttl_and_future_not_logged(settings, tmp_path, monkeypatch):
    search = SearchService(settings)
    service = ProgressiveService({"btc": search}, tmp_path)
    sid = service.create(ProgressiveConfig(translate=False, top_k=10))["session_id"]
    ledger = hints("a red car", "FUTURE_SECRET")
    ledger[1].revealed_at = 30000
    update = HintUpdate(expected_revision=0, client_request_id="one", hints=ledger, elapsed_ms=0)
    a = await service.update(sid, update)
    assert await service.update(sid, update) == a
    assert "FUTURE_SECRET" not in (tmp_path / f"{sid}.jsonl").read_text()
    events = [json.loads(line) for line in (tmp_path / f"{sid}.jsonl").read_text().splitlines()]
    assert replay_events(events) == [a]
    with pytest.raises(ProgressiveError, match="reused"):
        await service.update(sid, update.model_copy(update={"hints": hints("different")}))

    async def fail(*args, **kwargs):
        raise RuntimeError("worker down")
    monkeypatch.setattr(ProgressiveEngine, "run", fail)
    with pytest.raises(ProgressiveError) as exc:
        await service.update(sid, HintUpdate(expected_revision=1, client_request_id="two", hints=hints("a blue car")))
    assert exc.value.status == 503
    assert service.snapshot(sid)["groups"] == a["groups"]
    assert service.snapshot(sid)["committed_revision"] == 1
    service.sessions[sid].updated -= 3601
    with pytest.raises(ProgressiveError) as exc:
        service.snapshot(sid)
    assert exc.value.status == 410


@pytest.mark.asyncio
async def test_late_revision_cannot_overwrite_and_tabs_are_isolated(settings, tmp_path, monkeypatch):
    service = ProgressiveService({"btc": SearchService(settings)}, tmp_path)
    first = service.create(ProgressiveConfig(translate=False, top_k=10))["session_id"]
    other = service.create(ProgressiveConfig(translate=False, top_k=10))["session_id"]
    original = ProgressiveEngine.run
    started, release = asyncio.Event(), asyncio.Event()

    async def slow(self, ledger, base):
        if len(ledger) == 1:
            started.set()
            await release.wait()
        return await original(self, ledger, base)
    monkeypatch.setattr(ProgressiveEngine, "run", slow)
    pending = asyncio.create_task(service.update(first, HintUpdate(expected_revision=0, client_request_id="h1", hints=hints("a car"))))
    await started.wait()
    newer = await service.update(first, HintUpdate(expected_revision=1, client_request_id="h2", hints=hints("a car", "on a bridge")))
    release.set()
    with pytest.raises(ProgressiveError) as exc:
        await pending
    assert exc.value.status == 409
    assert service.snapshot(first)["groups"] == newer["groups"]
    assert service.snapshot(first)["pending_revision"] is None
    assert service.snapshot(other)["turn"] == 0
    service.close(first)
    with pytest.raises(ProgressiveError):
        service.snapshot(first)


def test_api_contract():
    with TestClient(app) as client:
        created = client.post("/api/progressive/sessions", json={"translate": False, "top_k": 10})
        assert created.status_code == 200
        sid = created.json()["session_id"]
        result = client.put(f"/api/progressive/sessions/{sid}/hints", json={
            "expected_revision": 0, "client_request_id": "test", "hints": [h.model_dump() for h in hints("a red car")],
        })
        assert result.status_code == 200, result.text
        assert result.json()["groups"][0]["progressive"]["trajectory"] == [1]
        assert client.delete(f"/api/progressive/sessions/{sid}").status_code == 200
        assert client.get(f"/api/progressive/sessions/{sid}").status_code == 410
        assert client.post("/api/progressive/sessions", json={"scope": {"mode": "auto"}}).status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("backfill_score", [.91, .2])
async def test_rescue_backfill_provenance_newcomer_and_original_snapshot(settings, monkeypatch, backfill_score):
    search = SearchService(settings)
    a, b, c, d = [f"L21_V00{i}" for i in range(1, 5)]
    calls = []
    async def rows(model, queries, k, categories, *, video_id=None, vector_cache=None):
        query = queries[0]
        calls.append((model, query, video_id))
        if video_id:
            scores = {(a, "later"): .1, (b, "later"): .05, (c, "early"): backfill_score, (d, "early"): .2}
            pairs = [(video_id, scores[(video_id, query)])]
        else:
            pairs = {"early": [(a, .9), (b, .8)], "later": [(c, .9), (d, .8)],
                     "early later": [(c, .95), (a, .85)]}[query]
        return [{"video_id": v, "keyframe_n": 1, "submit_keyframe_id": f"L21/{v}/001", "score": score} for v, score in pairs]
    monkeypatch.setattr(search, "_search_visual_model", rows)
    engine = ProgressiveEngine(search, ProgressiveConfig(translate=False))
    first = await engine.run(derive_ledger(hints("early")))
    frozen = copy.deepcopy(first)
    final = await engine.run(derive_ledger(hints("early", "later")), first)
    if backfill_score >= .8:
        assert final.rankings[-1][0] == c
    assert c not in first.rankings[0]
    assert first == frozen
    assert final.rankings[0] == [a, b]
    latest_pool = final.observations[1].trace.channel_hits["image_pe"]
    assert [h.video_id for h in latest_pool] == [c, d]
    weak = final.observations[1].trace.localization_hits["image_pe"]
    assert [h.video_id for h in weak] == [a, b]
    assert weak[0].evidence.extra["origin"] == "rescue"
    a_group = next(g for g in final.snapshot["groups"] if g["video_id"] == a)
    support = a_group["progressive"]["hint_evidence"][1]
    assert support["status"] == "localization_only"
    assert support["rank_evidence"] == 0
    assert support["frames"][0]["score"] == 0
    assert support["frames"][0]["submit_keyframe_id"] in {f["submit_keyframe_id"] for f in a_group["frames"]}
    pool_kind = "channel_hits" if backfill_score >= .8 else "localization_hits"
    old_pool = getattr(final.observations[0].trace, pool_kind)["image_pe"]
    new_evidence = next(h for h in old_pool if h.video_id == c)
    assert new_evidence.evidence.extra["discovered_at_turn"] == 2
    assert new_evidence.evidence.extra["origin"] == "backfill"
    assert len([call for call in calls if call[1] == "early" and call[2] is None]) == 1
    assert all(o.global_cutoffs["image_pe"] == .8 for o in final.observations)
    if backfill_score < .8:
        baseline = await ProgressiveEngine(search, engine.config.model_copy(update={"method": "phm_no_rescue"})).run(derive_ledger(hints("early", "later")))
        assert final.rankings == baseline.rankings
        assert [(g["video_id"], g["video_score"], g["progressive"]["memory_score"])
                for g in final.snapshot["groups"]] == [
                    (g["video_id"], g["video_score"], g["progressive"]["memory_score"])
                    for g in baseline.snapshot["groups"]]
        # Local caches hold raw hits only; replaying an edit must gate them again.
        replay = await engine.run(derive_ledger(hints("early", "later")),
                                 await engine.run(derive_ledger(hints("early")), final))
        assert replay.rankings == final.rankings
        assert replay.snapshot["budget"]["local_cache_hits"] > 0
        assert replay.observations[1].video_ranks == final.observations[1].video_ranks


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["cumulative", "latest", "dual_view"])
async def test_stateless_baselines_do_not_reuse_old_candidates_or_moments(settings, monkeypatch, method):
    search = SearchService(settings)
    a, b, c = [f"L21_V00{i}" for i in range(1, 4)]
    calls = []
    async def rows(model, queries, k, categories, *, video_id=None, vector_cache=None):
        assert video_id is None  # no rescue for these baselines
        calls.append(queries[0])
        pairs = {"early": [(a, 1), (b, 1)], "later": [(b, 2)], "early later": [(c, 2)]}[queries[0]]
        return [{"video_id": v, "keyframe_n": n, "submit_keyframe_id": f"L21/{v}/{n:03}", "score": .9}
                for v, n in pairs]
    monkeypatch.setattr(search, "_search_visual_model", rows)
    engine = ProgressiveEngine(search, ProgressiveConfig(translate=False, method=method))
    state = await engine.run(derive_ledger(hints("early", "later")))
    assert set(state.rankings[-1]) == ({b, c} if method == "dual_view" else {c} if method == "cumulative" else {b})
    assert all(f["keyframe_n"] == 2 for g in state.snapshot["groups"] for f in g["frames"])
    for group in state.snapshot["groups"]:
        assert group["progressive"]["memory_score"] == 0
        assert group["progressive"]["hint_evidence"][0]["status"] == "not_used"
        assert group["progressive"]["hint_evidence"][0]["frames"] == []
    if method == "dual_view":
        assert calls == ["early", "later", "early later"]
        assert all(g["video_score"] == .5 for g in state.snapshot["groups"])
    assert state.snapshot["rescue"] == []


@pytest.mark.asyncio
async def test_rescue_uses_separate_pe_and_qwen_frontiers(settings, monkeypatch):
    search = SearchService(settings.for_retrieval_database("infoshotpp"))
    a, b, c, d = [f"L21_V00{i}" for i in range(1, 5)]
    async def rows(model, queries, k, categories, *, video_id=None, vector_cache=None):
        if video_id:
            pairs = [(video_id, .5 if model == "pe" else .35)]
        else:
            videos = {"early": [a, b], "later": [c, d], "early later": [c, a]}[queries[0]]
            pairs = list(zip(videos, [.9, .8] if model == "pe" else [.4, .3]))
        return [{"video_id": v, "keyframe_n": 1, "submit_keyframe_id": f"L21/{v}/001", "score": score} for v, score in pairs]
    monkeypatch.setattr(search, "_search_visual_model", rows)
    engine = ProgressiveEngine(search, ProgressiveConfig(retrieval_database="infoshotpp", image_models=["pe", "qwen3_vl"], translate=False))
    state = await engine.run(derive_ledger(hints("early", "later")))
    trace = state.observations[1].trace
    assert a not in {h.video_id for h in trace.channel_hits["image_pe"]}
    assert a in {h.video_id for h in trace.localization_hits["image_pe"]}
    assert a in {h.video_id for h in trace.channel_hits["image_qwen"]}
    assert not trace.localization_hits["image_qwen"]
    assert state.observations[1].global_cutoffs == {"image_pe": .8, "image_qwen": .3}


@pytest.mark.asyncio
async def test_weak_backfill_cannot_reactivate_evicted_global_evidence(settings, monkeypatch):
    search = SearchService(settings)
    a, b, c = [f"L21_V00{i}" for i in range(1, 4)]
    async def rows(model, queries, k, categories, *, video_id=None, vector_cache=None):
        pairs = [(video_id, .01)] if video_id else {
            "early": [(a, .9), (b, .8)], "later": [(c, .9)], "early later": [(a, .9), (c, .8)],
        }[queries[0]]
        return [{"video_id": v, "keyframe_n": 1, "submit_keyframe_id": f"L21/{v}/001", "score": score} for v, score in pairs]
    monkeypatch.setattr(search, "_search_visual_model", rows)
    engine = ProgressiveEngine(search, ProgressiveConfig(translate=False))
    state = await engine.run(derive_ledger(hints("early")))
    engine._retain(state.observations[0], {b})
    state.rankings[-1] = [b]  # simulate eviction while keeping the original raw trace
    final = await engine.run(derive_ledger(hints("early", "later")), state)
    assert a in {h.video_id for h in final.observations[0].trace.channel_hits["image_pe"]}
    assert a not in final.observations[0].video_ranks
    assert a in final.observations[0].localization_frames
    support = next(g for g in final.snapshot["groups"] if g["video_id"] == a)["progressive"]["hint_evidence"][0]
    assert support["status"] == "localization_only"
    assert support["rank_evidence"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_global", [False, True])
async def test_display_only_rescue_preserves_empty_or_tied_global_ranking(settings, monkeypatch, empty_global):
    search = SearchService(settings)
    a, b, c, d = [f"L21_V00{i}" for i in range(1, 5)]
    async def rows(model, queries, k, categories, *, video_id=None, vector_cache=None):
        if video_id:
            # Even a high score cannot establish rank if no global frontier exists.
            pairs = [(video_id, .99 if empty_global and queries[0] == "later" else .01)]
        else:
            # Equal scores with non-ID global order: display-only work must not
            # re-sort these ranks as an incidental side effect of merging [].
            pairs = {"early": [(b, .9), (a, .9)], "later": [] if empty_global else [(d, .8), (c, .8)],
                     "early later": [(c, .9), (a, .8)]}[queries[0]]
        return [{"video_id": v, "keyframe_n": 1, "submit_keyframe_id": f"L21/{v}/001", "score": score} for v, score in pairs]
    monkeypatch.setattr(search, "_search_visual_model", rows)
    config = ProgressiveConfig(translate=False)
    ledger = derive_ledger(hints("early", "later"))
    rescued = await ProgressiveEngine(search, config).run(ledger)
    plain = await ProgressiveEngine(search, config.model_copy(update={"method": "phm_no_rescue"})).run(ledger)
    assert rescued.rankings == plain.rankings
    assert [o.video_ranks for o in rescued.observations] == [o.video_ranks for o in plain.observations]
    assert [g["video_score"] for g in rescued.snapshot["groups"]] == [g["video_score"] for g in plain.snapshot["groups"]]
    assert rescued.observations[1].trace.localization_hits["image_pe"]
    if empty_global:
        assert all(h.evidence.extra["ranking_exclusion"] == "unknown_global_cutoff"
                   for h in rescued.observations[1].trace.localization_hits["image_pe"])


@pytest.mark.asyncio
async def test_live_visual_failure_keeps_old_memory_and_partial_channel_is_degraded(settings, tmp_path, monkeypatch):
    search = SearchService(settings.for_retrieval_database("infoshotpp"))
    service = ProgressiveService({"infoshotpp": search}, tmp_path)
    config = ProgressiveConfig(retrieval_database="infoshotpp", image_models=["pe", "qwen3_vl"], translate=False, top_k=10)
    sid = service.create(config)["session_id"]
    old = await service.update(sid, HintUpdate(expected_revision=0, client_request_id="one", hints=hints("a car")))
    original = search._search_visual_model
    async def partial(model, *args, **kwargs):
        if model == "qwen3_vl":
            raise RuntimeError("qwen offline")
        return await original(model, *args, **kwargs)
    monkeypatch.setattr(search, "_search_visual_model", partial)
    next_result = await service.update(sid, HintUpdate(expected_revision=1, client_request_id="two", hints=hints("a car", "on bridge")))
    assert next_result["degraded"]
    assert next_result["groups"]
    assert service.sessions[sid].state.rankings[0] == [g["video_id"] for g in old["groups"]]
    async def all_failed(*args, **kwargs):
        raise RuntimeError("all offline")
    monkeypatch.setattr(search, "_search_visual_model", all_failed)
    with pytest.raises(ProgressiveError):
        await service.update(sid, HintUpdate(expected_revision=2, client_request_id="three", hints=hints("a car", "on bridge", "at night")))
    assert service.snapshot(sid)["groups"] == next_result["groups"]
