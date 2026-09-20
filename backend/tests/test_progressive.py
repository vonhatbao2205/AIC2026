import asyncio
import copy
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.progressive.core import derive_ledger, memory_score, merge_local, rescue_pairs, bounded_candidates
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


def test_local_rank_is_not_global_rank_and_models_stay_separate():
    def hit(video, score, channel="image_pe"):
        return ChannelHit(channel, video + ":1", video, 1, score, 0)
    global_hits = [hit("00047", .9), hit("00048", .8)]
    merged = merge_local(global_hits, [hit("00049", .2), hit("00047", .9)])
    assert [h.video_id for h in merged] == ["00047", "00048", "00049"]
    assert [h.rank for h in merged] == [0, 1, 2]
    assert global_hits[1].rank == 0  # no mutation of the original trace
    with pytest.raises(ValueError):
        merge_local(global_hits, [hit("other", .99, "image_qwen")])
    assert memory_score([1, 0]) == pytest.approx(.1 ** .5)


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
async def test_rescue_backfill_provenance_newcomer_and_original_snapshot(settings, monkeypatch):
    search = SearchService(settings)
    a, b, c, d = [f"L21_V00{i}" for i in range(1, 5)]
    calls = []
    async def rows(model, queries, k, categories, *, video_id=None, vector_cache=None):
        query = queries[0]
        calls.append((model, query, video_id))
        if video_id:
            scores = {(a, "later"): .1, (b, "later"): .05, (c, "early"): .91, (d, "early"): .2}
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
    assert final.rankings[-1][0] == c
    assert c not in first.rankings[0]
    assert first == frozen
    assert final.rankings[0] == [a, b]
    latest_pool = final.observations[1].trace.channel_hits["image_pe"]
    assert [h.video_id for h in latest_pool] == [c, d, a, b]
    assert latest_pool[2].evidence.extra["origin"] == "rescue"
    old_pool = final.observations[0].trace.channel_hits["image_pe"]
    new_evidence = next(h for h in old_pool if h.video_id == c)
    assert new_evidence.evidence.extra["discovered_at_turn"] == 2
    assert new_evidence.evidence.extra["origin"] == "backfill"
    assert len([call for call in calls if call[1] == "early" and call[2] is None]) == 1


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
