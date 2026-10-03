"""Closed-loop decisions, provenance, budgets and metrics with no paid calls."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from app import main
from app.agent.controller import Controller
from app.agent.decision.calibration import Calibration, fit_temperature, geometric_score
from app.agent.decision.jev_client import DecisionError, JevClient
from app.agent.decision.stopping import can_stop
from app.agent.decision.verifier import verify
from app.agent.evidence.board import EvidenceBoard
from app.agent.evidence.compiler import compile_constraints
from app.agent.runs import AgentRun, AgentState, AgentService
from app.agent.tools import AgentTools
from app.models import AgentRunRequest


def frame(n=1, **extra):
    return {"submit_keyframe_id": f"K01/K01_V001/{n:03d}", "video_id": "K01_V001",
            "pts_time": n * 4.0, "frame_idx": n * 100, "fps": 25, "keyframe_n": n,
            "evidence": [{"type": "ocr", "text": "a red bicycle"}], **extra}


class Judge:
    def __init__(self, p=.97):
        self.p = p
        self.calls = []

    async def decide(self, state, questions):
        self.calls.append((state, questions))
        return {k: {"type": "choice", "choice": "supported", "probabilities": {
            "supported": self.p, "refuted": 0.0, "unknown": 1-self.p}} for k in questions}


@pytest.mark.asyncio
async def test_decisions_http_contract_and_usage(settings):
    settings.mock_mode = False
    settings.openrouter_api_key = "private-key"
    client = JevClient(settings)
    def respond(req):
        assert str(req.url) == JevClient.URL
        assert req.headers["Authorization"] == "Bearer private-key"
        assert json.loads(req.content)["model"] == "typesafe/jev-1.13"
        return httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": .8}},
                                         "usage": {"input_tokens": 40, "output_tokens": 3, "cost": .001}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client.http.get = lambda: http
        assert (await client.decide({}, {"q": {"type": "noul", "instructions": "q"}}))["q"]["noul"] == .8
    assert client.calls[0]["input_tokens"] == 40
    assert "private-key" not in json.dumps(client.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"answers": {"q": {"type": "noul", "noul": 1.2}}},
                                      {"answers": {"q": {"type": "choice", "choice": "a"}}}])
async def test_malformed_decisions_fail_closed(settings, payload):
    settings.mock_mode = False
    settings.openrouter_api_key = "secret"
    client = JevClient(settings)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))) as http:
        client.http.get = lambda: http
        with pytest.raises(DecisionError):
            await client.decide({}, {"q": {"type": "noul", "instructions": "q"}})


@pytest.mark.asyncio
async def test_provider_errors_never_echo_key_or_body(settings):
    settings.mock_mode = False
    settings.openrouter_api_key = "secret"
    client = JevClient(settings)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="secret"))) as http:
        client.http.get = lambda: http
        with pytest.raises(DecisionError, match="Jev HTTP 401") as error:
            await client.decide({}, {"q": {"type": "noul"}})
    assert "secret" not in str(error.value)


@pytest.mark.asyncio
async def test_no_visual_evidence_cannot_be_certified_even_by_overconfident_judge():
    board = EvidenceBoard("a red bicycle", "T-KIS")
    f = frame(evidence=[])
    board.seed({"groups": [{"frames": [f]}]})
    board.report({**f, "agent": "codex", "confidence": 1, "reason": "looks correct"}, .1)
    await verify(Judge(1), board, Calibration(), limit=5)
    assert not can_stop(board, .9, .1)
    assert board.ranked()[0]["constraint_scores"]["query"]["unknown"] == 1


@pytest.mark.asyncio
async def test_verified_candidate_stops_but_contradicted_constraint_does_not():
    board = EvidenceBoard("a red bicycle", "T-KIS")
    board.seed({"groups": [{"frames": [frame()]}]})
    await verify(Judge(), board, Calibration(), limit=5)
    assert can_stop(board, .9, .1)
    board.candidates[frame()["submit_keyframe_id"]]["scores"]["query"]["supported"] = .3
    assert not can_stop(board, .9, .1)


def test_provenance_deduplicates_and_boards_are_query_local():
    board = EvidenceBoard("q", "T-KIS")
    f = frame()
    board.seed({"groups": [{"frames": [f]}]})
    board.seed({"groups": [{"frames": [f]}]}, source="search:codex")
    assert len(board.candidates[f["submit_keyframe_id"]]["evidence"]) == 1
    assert not EvidenceBoard("q", "T-KIS").candidates
    assert board.disagreement() is None


def test_uninspected_agent_claim_is_not_visual_evidence():
    board = EvidenceBoard("q", "T-KIS")
    f = frame(evidence=[])
    board.report({**f, "agent": "codex", "reason": "red", "confidence": .99}, 1)
    assert not board.candidates[f["submit_keyframe_id"]]["evidence"]
    board.inspected["codex"] = {f["submit_keyframe_id"]}
    board.report({**f, "agent": "codex", "reason": "red", "confidence": .99}, 2)
    assert len(board.candidates[f["submit_keyframe_id"]]["evidence"]) == 1


def test_compiler_retains_whole_query_and_event_order():
    query = "Con lân đỏ. E1: nhảy qua (4). E2: cúi chào."
    cs = compile_constraints(query, "TRAKE")
    assert cs[0].text == query
    assert [c.event for c in cs if c.event] == [1, 2]
    assert "(4)" in cs[1].text


def test_calibration_fit_rejects_test_labels_and_reduces_overconfidence(tmp_path):
    with pytest.raises(ValueError):
        fit_temperature([{"split": "test", "query_id": "q", "probability": .99, "label": 0}])
    artifact = fit_temperature([{"split": "dev", "query_id": str(i), "probability": .99, "label": i % 2} for i in range(20)])
    assert artifact["temperature"] > 1
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(artifact))
    cal = Calibration.load(str(path))
    assert cal.fitted and cal.apply(.99) < .99
    assert geometric_score([.99, .01], [1, 1]) < .11


@pytest.mark.asyncio
async def test_controller_stops_without_launching_agents_on_evidence(settings, monkeypatch):
    settings.agent_max_steps = 3
    run = AgentRun(AgentRunRequest(query="a red bicycle", policy="full"))
    run.controller_active = True
    run.agents["codex"] = AgentState("codex", "m", "high", run.started)
    service = SimpleNamespace()
    controller = Controller(service, run, {"codex": "unused"}, "http://local", settings,
                            {"groups": [{"frames": [frame()]}]})
    async def decide(state, questions):
        return await Judge().decide(state, questions)
    monkeypatch.setattr(controller.client, "decide", decide)
    await controller.run_controller()
    assert run.finished
    assert run.agents["codex"].status == "skipped"
    assert run.controller["stop_reason"] == "evidence_sufficient"
    assert run.agent_invocations["codex"] == 0


@pytest.mark.asyncio
async def test_controller_fallback_escalates_and_respects_selected_agents(settings, monkeypatch):
    run = AgentRun(AgentRunRequest(query="q", agents=["claude"], policy="adaptive"))
    run.controller_active = True
    run.agents["claude"] = AgentState("claude", "m", "high", run.started)
    launched = []
    async def launch(run, state, binary, url):
        launched.append(state.name)
        state.begin()
        state.finish("done")
    service = SimpleNamespace(_run_agent=launch)
    controller = Controller(service, run, {"claude": "unused"}, "http://local", settings, {"groups": []})
    await controller.run_controller()
    assert launched == ["claude"]
    assert run.controller["fallback"]
    assert run.finished


@pytest.mark.asyncio
async def test_tool_budget_and_no_late_candidate_mutation():
    run = AgentRun(AgentRunRequest(query="q"))
    run.agents["codex"] = AgentState("codex", "m", "high", run.started)
    run.max_tool_calls = 1
    tools = AgentTools(lambda: main.search_services)
    result = await tools.call("video_text", {"video_id": "K01_V001"}, run, "codex")
    assert not result["isError"]
    result = await tools.call("report_candidate", {"keyframe_id": "K01/K01_V001/006", "confidence": .9}, run, "codex")
    assert result["isError"] and not run.candidates
    run.cancelled = True
    assert (await tools.call("video_text", {"video_id": "K01_V001"}, run, "codex"))["isError"]


@pytest.mark.asyncio
async def test_duplicate_tool_calls_share_one_query_request(monkeypatch):
    run = AgentRun(AgentRunRequest(query="q"))
    run.agents["codex"] = AgentState("codex", "m", "high", run.started)
    tools = AgentTools(lambda: main.search_services)
    calls = []
    async def text(args, run, agent):
        calls.append(agent)
        await asyncio.sleep(.01)
        return [{"type": "text", "text": "ok"}]
    monkeypatch.setattr(tools, "_video_text", text)
    await asyncio.gather(*(tools.call("video_text", {"video_id": "K01_V001"}, run, agent) for agent in ("codex", "claude")))
    assert len(calls) == 1 and run.cache_hits == 1


def test_seed_cannot_cross_queries_or_profiles(settings):
    service = AgentService(lambda: settings, lambda: main.search_services)
    key = service.remember_retrieval({"query": "q", "scope": {"mode": "all", "categories": []}}, {"groups": []})
    assert service.take_retrieval(AgentRunRequest(query="q", retrieval_id=key)) == {"groups": []}
    assert service.take_retrieval(AgentRunRequest(query="different", retrieval_id=key)) is None
    assert service.take_retrieval(AgentRunRequest(query="q", retrieval_database="infoshotpp", retrieval_id=key)) is None
    assert service.remember_retrieval({"query": "q", "previous_hints": ["old"]}, {}) is None


@pytest.mark.asyncio
async def test_new_tools_render_comparison_and_local_probe(monkeypatch):
    import io
    from PIL import Image
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(stream, "JPEG")
    tools = AgentTools(lambda: main.search_services)
    async def fetch(*args):
        return stream.getvalue()
    monkeypatch.setattr(tools, "_fetch", fetch)
    run = AgentRun(AgentRunRequest(query="red shirt"))
    run.agents["codex"] = AgentState("codex", "m", "high", run.started)
    comparison = await tools.call("compare_candidates", {"keyframe_ids": ["K01/K01_V001/006", "K01/K01_V001/007"]}, run, "codex")
    assert not comparison["isError"]
    assert any(c["type"] == "image" for c in comparison["content"])
    assert "columns A/B" in comparison["content"][0]["text"]
    probe = await tools.call("constraint_probe", {"keyframe_id": "K01/K01_V001/006", "constraint": "red shirt"}, run, "codex")
    assert not probe["isError"] and "Local evidence" in probe["content"][0]["text"]
    assert run.board.inspected["codex"]
    assert not any(t["tool"] == "search" for t in run.board.history)


@pytest.mark.asyncio
async def test_center_time_and_tool_ablation(monkeypatch):
    tools = AgentTools(lambda: main.search_services)
    run = AgentRun(AgentRunRequest(query="q", toolset="base"))
    run.agents["codex"] = AgentState("codex", "m", "high", run.started)
    result = await tools.call("compare_candidates", {}, run, "codex")
    assert result["isError"] and run.tool_count == 0
    result = await tools.call("video_frames", {"video_id": "K01_V001", "center_time": 10, "start": 2}, run, "codex")
    assert result["isError"] and "not both" in result["content"][0]["text"]


def test_openrouter_secret_is_not_in_cli_environment(monkeypatch):
    from app.agent.cli import agent_process_env
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    assert "OPENROUTER_API_KEY" not in agent_process_env()


def test_role_biases_share_capabilities_and_ignore_previous_hints():
    from app.agent.prompt import build_prompt
    req = AgentRunRequest(query="q", previous_hints=["cross-query secret"])
    a = build_prompt(req, timeout_seconds=30, agent="codex")
    b = build_prompt(req, timeout_seconds=30, agent="claude")
    assert "SCOUT" in a and "INVESTIGATOR" in b
    for prompt in (a, b):
        assert "every enabled tool" in prompt
        assert "cross-query secret" not in prompt


def test_compiler_exposes_single_sentence_details_without_inventing_text():
    text = "Một người đàn ông mặc áo vàng đang sửa xe đạp bên đường."
    constraints = compile_constraints(text, "T-KIS")
    details = [c for c in constraints if c.source_span is not None]
    assert {c.kind for c in details} >= {"clothing", "action", "location"}
    assert all(text[c.source_span[0]:c.source_span[1]].strip() == c.text for c in details)
    assert all(c.requires == ("query",) for c in details)
    assert constraints[0].text == text


@pytest.mark.asyncio
async def test_qa_missing_or_changed_answer_must_be_verified():
    board = EvidenceBoard("What is written on the sign?", "QA")
    f = frame()
    board.seed({"groups": [{"frames": [f]}]})
    await verify(Judge(1), board, Calibration(), limit=5)
    assert not can_stop(board, .9, .1)
    board.report({**f, "agent": "codex", "answer": "red bicycle", "confidence": .99}, 1)
    await verify(Judge(.99), board, Calibration(), limit=5)
    assert can_stop(board, .9, .1)
    board.report({**f, "agent": "codex", "answer": "changed claim", "confidence": .99}, 2)
    assert board.ranked()[0]["probability"] is None
    assert not can_stop(board, .9, .1)


@pytest.mark.asyncio
async def test_trake_events_of_one_video_are_not_competitors():
    board = EvidenceBoard("E1: jumps. E2: bows.", "TRAKE")
    board.seed({"groups": [{"frames": [frame(1, event=1), frame(6, event=2)]}]})
    await verify(Judge(.99), board, Calibration(), limit=1)
    assert len(board.shortlist(1)) == 2  # Complete leading chain overrides k=1.
    assert can_stop(board, .9, .1)
    board.candidates[frame(6)["submit_keyframe_id"]]["frame"]["time"] = 1
    assert not can_stop(board, .9, .1)


def test_shortlist_includes_both_agents_and_distinct_retrieval_moments():
    board = EvidenceBoard("q", "T-KIS")
    board.seed({"groups": [{"frames": [frame(i) for i in range(1, 10)]}]})
    for i, agent, conf in [(5, "codex", .99), (6, "codex", .98), (9, "claude", .7)]:
        board.report({**frame(i), "agent": agent, "confidence": conf}, 1)
    selected = board.shortlist(3)
    assert frame(5)["submit_keyframe_id"] in selected
    assert frame(9)["submit_keyframe_id"] in selected
    assert frame(1)["submit_keyframe_id"] in selected


def test_closed_board_rejects_late_evidence():
    board = EvidenceBoard("q", "T-KIS")
    board.closed = True
    board.seed({"groups": [{"frames": [frame()]}]})
    board.report({**frame(), "agent": "codex", "confidence": .9}, 1)
    board.remember_text("K01_V001", 0, 100, "speech", "hello", "segment")
    assert not board.candidates and not board.local_text


@pytest.mark.asyncio
async def test_controller_timeout_cancels_http_tool_work(settings, monkeypatch):
    settings.agent_timeout_seconds = .03
    run = AgentRun(AgentRunRequest(query="q", policy="codex"))
    run.controller_active = True
    run.agents["codex"] = AgentState("codex", "m", "high", run.started)
    tools = AgentTools(lambda: main.search_services)
    tool_cancelled = asyncio.Event()
    async def slow_text(args, run, agent):
        try:
            await asyncio.sleep(60)
        finally:
            tool_cancelled.set()
    monkeypatch.setattr(tools, "_video_text", slow_text)
    task = asyncio.create_task(tools.call("video_text", {"video_id": "K01_V001"}, run, "codex"))
    async def slow_agent(run, state, binary, url):
        await asyncio.sleep(60)
    controller = Controller(SimpleNamespace(_run_agent=slow_agent), run, {"codex": "fake"}, "http://local", settings, {"groups": []})
    await controller.run_controller()
    response = await task
    assert response["isError"] and tool_cancelled.is_set()
    assert run.board.closed and not run.tool_tasks and run.finished


def test_dense_ocr_cannot_drop_visual_observation_or_hide_it_from_judge():
    board = EvidenceBoard("red bicycle", "T-KIS")
    f = frame(evidence=[])
    key = f["submit_keyframe_id"]
    board.seed({"groups": [{"frames": [f]}]})
    for i in range(24):
        board.add_evidence(key, "ocr", "same ticker", str(i))
    board.inspected["codex"] = {key}
    board.report({**f, "agent": "codex", "reason": "Visible bicycle is red", "confidence": .9}, 1)
    board.add_evidence(key, "ocr", "another ticker", "new")
    assert len(board.candidates[key]["evidence"]) == 24
    context = board.state()["candidates"][0]["evidence"]
    assert any(e["kind"] == "agent_visual_observation" for e in context)
    assert sum(e["text"] == "same ticker" for e in context) == 1


@pytest.mark.asyncio
async def test_unknown_shortlist_does_not_demote_unassessed_correct_frame():
    board = EvidenceBoard("red bicycle", "T-KIS")
    frames = [frame(i, evidence=[]) for i in range(1, 10)]
    board.seed({"groups": [{"frames": frames}]})
    before = [f["submit_keyframe_id"] for f in board.ranked()]
    await verify(Judge(1), board, Calibration(), limit=5)
    assert [f["submit_keyframe_id"] for f in board.ranked()] == before


@pytest.mark.asyncio
async def test_supported_rescue_promotes_and_refuted_candidate_demotes():
    board = EvidenceBoard("red bicycle", "T-KIS")
    board.seed({"groups": [{"frames": [frame(1), frame(2)]}]})
    await verify(Judge(.97), board, Calibration(), limit=2)
    first = board.candidates[frame(1)["submit_keyframe_id"]]
    first["scores"]["query"] = {"supported": .01, "unknown": .01, "refuted": .98}
    first["probability"] = .01
    assert board.ranked()[0]["submit_keyframe_id"] == frame(2)["submit_keyframe_id"]
    assert board.ranked(verified=False)[0]["submit_keyframe_id"] == frame(1)["submit_keyframe_id"]


@pytest.mark.asyncio
async def test_calibrated_stopping_keeps_separate_raw_evidence_guard():
    board = EvidenceBoard("red bicycle", "T-KIS")
    board.seed({"groups": [{"frames": [frame()]}]})
    await verify(Judge(.8), board, Calibration(), limit=1)
    assert not can_stop(board, .9, .1)
    await verify(Judge(.8), board, Calibration(.5, True, ("dev",)), limit=1)
    assert can_stop(board, .9, .1)
    candidate = board.candidates[frame()["submit_keyframe_id"]]
    candidate["scores"]["query"] = {"supported": .4, "unknown": .6, "refuted": 0}
    candidate["probability"] = .99
    assert not can_stop(board, .9, .1)


@pytest.mark.asyncio
async def test_stop_override_executes_last_comparison_then_exhausts(settings, monkeypatch):
    settings.agent_max_steps = 3
    run = AgentRun(AgentRunRequest(query="q", policy="full", router="rule"))
    run.controller_active = True
    controller = Controller(SimpleNamespace(), run, {}, "http://local", settings,
                            {"groups": [{"frames": [frame(1), frame(9)]}]})
    controller.actions.update({"SEARCH_MORE", "INSPECT_TOP1"})
    async def unknown():
        return False
    actions = []
    async def execute(action):
        actions.append(action)
        controller.actions.add(action)
    monkeypatch.setattr(controller, "evaluate", unknown)
    monkeypatch.setattr(controller, "execute", execute)
    await controller.run_controller()
    assert actions == ["COMPARE_TOP2"]
    assert run.controller["stop_reason"] == "actions_exhausted"
