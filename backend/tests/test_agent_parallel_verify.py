"""C+V must acquire evidence like C and verify only after both agents finish."""
import asyncio
from types import SimpleNamespace

import pytest

from app.agent.controller import Controller
from app.agent.decision.jev_client import DecisionError
from app.agent.runs import AgentRun, AgentState
from app.models import AgentRunRequest
from benchmarks.agent.metrics import measure


def frame(n):
    return {"submit_keyframe_id": f"v/{n}", "video_id": "v", "frame_idx": n * 100,
            "fps": 25, "pts_time": n * 4, "evidence": [{"type": "ocr", "text": "red bicycle"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "provider", "quota"])
async def test_final_verifier_after_parallel_acquisition(settings, monkeypatch, failure):
    run = AgentRun(AgentRunRequest(query="a red bicycle beside a tree", policy="parallel_verify"))
    run.controller_active = True
    for name in ("codex", "claude"):
        run.agents[name] = AgentState(name, "mock", "high", run.started)
    started = set()
    both_started = asyncio.Event()
    async def agent(run, state, binary, url):
        state.begin()
        started.add(state.name)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), .5)  # Fails if launched serially.
        assert not any(c["scores"] for c in run.board.candidates.values())
        state.finish("done")
    controller = Controller(SimpleNamespace(_run_agent=agent), run,
        {"codex": "mock", "claude": "mock"}, "http://local", settings,
        {"groups": [{"frames": [frame(1), frame(2)]}]})
    calls = []
    async def decide(state, questions):
        assert all(a.status == "done" for a in run.agents.values())
        calls.append(questions)
        assert len(questions) == 2  # E's holistic query check, not F's decomposition.
        if failure:
            raise DecisionError("Jev HTTP 429" if failure == "quota" else "Jev HTTP 503")
        return {key: {"probabilities": {
            "supported": .01 if key == "c0_0" else .99,
            "refuted": .99 if key == "c0_0" else .01, "unknown": 0}}
            for key in questions}
    async def forbidden(*args, **kwargs):
        pytest.fail("C+V must never route adaptively or execute evidence macros")
    monkeypatch.setattr(controller.client, "decide", decide)
    monkeypatch.setattr("app.agent.controller.route", forbidden)
    monkeypatch.setattr(controller, "execute", forbidden)
    await controller.run_controller()
    assert run.finished and run.agent_invocations == {"codex": 1, "claude": 1}
    assert len(calls) == 1
    before = run.snapshot()["pre_verification_ranking"]
    assert [r["submit_keyframe_id"] for r in before] == ["v/1", "v/2"]
    assert all(r.get("probability") is None for r in before)
    assert len([t for t in run.trace if t["kind"] == "pre_verification"]) == 1
    if failure == "quota":
        assert run.controller["stop_reason"] == "model_quota"
    else:
        assert run.controller["stop_reason"] == "fixed_verification_complete"
        assert bool(run.controller.get("fallback")) == (failure == "provider")
    assert run.ranking[0]["submit_keyframe_id"] == ("v/1" if failure else "v/2")


def test_missing_cli_is_counted_as_failure_for_fixed_pair():
    q = {"query_type": "T-KIS", "targets": [{"video_id": "v", "frame_idx": 100}]}
    snapshot = {"policy": "parallel_verify", "agents": {"claude": {"status": "unavailable"}}}
    assert measure(q, snapshot, [], wall_s=1)["agent_failed"]
