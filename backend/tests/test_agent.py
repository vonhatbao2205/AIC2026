"""Agent sidecar search: MCP bridge, tool API, run lifecycle, CLI locking-down.

No real Codex/Claude is ever started: the CLIs are replaced by small scripts
that print the same JSONL events the real ones do.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.agent import cli
from app.agent.cli import (
    agent_process_env,
    claude_argv,
    codex_argv,
    describe_call,
    read_claude_event,
    read_codex_event,
)
from app.agent.runs import AgentRun, AgentState
from app.agent.tools import AgentTools, _render_sheet
from app.models import AgentRunRequest


# ---- MCP bridge -------------------------------------------------------------
class _StubBackend(BaseHTTPRequestHandler):
    calls: list[tuple[str, str, dict, str | None]] = []

    def _reply(self, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        self.calls.append(("GET", self.path, {}, self.headers.get("X-AIC-Agent-Token")))
        self._reply({"tools": [{"name": "search", "inputSchema": {"type": "object"}}]})

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append(("POST", self.path, body, self.headers.get("X-AIC-Agent-Name")))
        self._reply({"content": [{"type": "text", "text": "ok"}], "isError": False})

    def log_message(self, *args):
        pass


def _rpc(proc: subprocess.Popen, message: dict) -> dict:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def test_mcp_bridge_forwards_tools_with_the_run_token():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubBackend)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _StubBackend.calls = []
    env = {
        **os.environ,
        "AIC_AGENT_API": f"http://127.0.0.1:{server.server_port}",
        "AIC_AGENT_TOKEN": "tok-1",
        "AIC_AGENT_NAME": "codex",
    }
    proc = subprocess.Popen(
        [sys.executable, str(cli.MCP_SCRIPT)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
    )
    try:
        init = _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
        assert init["result"]["protocolVersion"] == "2025-06-18"
        assert "tools" in init["result"]["capabilities"]
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        listed = _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert listed["result"]["tools"][0]["name"] == "search"
        called = _rpc(proc, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "xe đạp"}},
        })
        assert called["result"] == {"content": [{"type": "text", "text": "ok"}], "isError": False}
        unknown = _rpc(proc, {"jsonrpc": "2.0", "id": 4, "method": "resources/list"})
        assert unknown["error"]["code"] == -32601
    finally:
        proc.kill()
        server.shutdown()
    assert ("GET", "/api/agent/tools", {}, "tok-1") in _StubBackend.calls
    assert ("POST", "/api/agent/tools/search", {"arguments": {"query": "xe đạp"}}, "codex") in _StubBackend.calls


# ---- CLI command lines and environment -----------------------------------------
def test_codex_is_locked_to_the_bridge(settings, tmp_path):
    argv = codex_argv(settings, binary="codex", workdir=tmp_path, prompt="find it", bridge={"AIC_AGENT_TOKEN": "t"})
    joined = " ".join(argv)
    assert argv[:2] == ["codex", "exec"] and argv[-1] == "find it"
    assert "--disable shell_tool" in joined and "--disable unified_exec" in joined
    # Without it Codex rejects every MCP call under approval_policy=never.
    assert 'mcp_servers.aic.default_tools_approval_mode="approve"' in argv
    assert ["--sandbox", "read-only"] == argv[argv.index("--sandbox"):argv.index("--sandbox") + 2]
    assert ["--model", "gpt-6-sol"] == argv[argv.index("--model"):argv.index("--model") + 2]
    assert 'model_reasoning_effort="high"' in argv


def test_claude_has_no_builtin_tools(settings):
    argv = claude_argv(settings, binary="claude", prompt="find it", bridge={"AIC_AGENT_TOKEN": "t"})
    assert argv[:3] == ["claude", "-p", "find it"]
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--allowedTools") + 1] == "mcp__aic"
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5"
    assert argv[argv.index("--effort") + 1] == "high"
    config = json.loads(argv[argv.index("--mcp-config") + 1])
    assert config["mcpServers"]["aic"]["env"] == {"AIC_AGENT_TOKEN": "t"}


def test_fast_mode_is_on_by_default_and_can_be_turned_off(settings, tmp_path):
    codex = codex_argv(settings, binary="codex", workdir=tmp_path, prompt="p", bridge={})
    claude = claude_argv(settings, binary="claude", prompt="p", bridge={})
    assert 'service_tier="fast"' in codex
    assert json.loads(claude[claude.index("--settings") + 1]) == {"fastMode": True}

    settings.agent_codex_fast = False
    settings.agent_claude_fast = False
    assert 'service_tier="fast"' not in codex_argv(settings, binary="codex", workdir=tmp_path, prompt="p", bridge={})
    assert "--settings" not in claude_argv(settings, binary="claude", prompt="p", bridge={})


def test_agent_environment_drops_backend_credentials(monkeypatch):
    monkeypatch.setenv("ELASTIC_API_KEY", "secret-elastic")
    monkeypatch.setenv("DRES_PASSWORD", "secret-dres")
    monkeypatch.setenv("HOME", "/home/someone")
    env = agent_process_env()
    assert "ELASTIC_API_KEY" not in env and "DRES_PASSWORD" not in env
    assert "AIC26_MOCK_MODE" not in env
    assert env["HOME"] == "/home/someone"


# ---- stream readers -------------------------------------------------------------
def _state() -> AgentState:
    return AgentState("codex", "m", "high", time.monotonic())


def test_codex_stream_reports_tools_messages_and_failures():
    state = _state()
    read_codex_event({"type": "item.started", "item": {
        "type": "mcp_tool_call", "tool": "search", "arguments": '{"query": "áo vàng", "mode": "visual"}',
    }}, state)
    read_codex_event({"type": "item.completed", "item": {"type": "agent_message", "text": "Best: L23_V001 at 12s"}}, state)
    read_codex_event({"type": "turn.failed", "error": {"message": "You've hit your usage limit"}}, state)
    assert state.tool_calls == 1
    assert state.steps[0]["text"] == "search [visual] “áo vàng”"
    assert state.summary == "Best: L23_V001 at 12s"
    assert cli.classify_error(state.fail_reason) == "quota"


def test_claude_stream_reads_tool_use_and_result():
    state = _state()
    read_claude_event({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "mcp__aic__video_frames", "input": {"video_id": "L21_V001", "start": 10, "end": 40}},
    ]}}, state)
    read_claude_event({"type": "result", "subtype": "success", "result": "Found it.", "total_cost_usd": 0.12}, state)
    assert state.steps[0]["text"] == "video_frames L21_V001 10–40s"
    assert state.summary == "Found it." and state.cost_usd == 0.12
    assert state.fail_reason is None


def test_outdated_cli_is_labelled_update():
    message = "API Error: 400 Claude Code 2.1.223 does not support this model; version 2.1.280 or newer is required."
    assert cli.classify_error(message) == "update"


def test_describe_report_candidate():
    assert describe_call("mcp__aic__report_candidate", {"video_id": "N001-V001", "time": 5, "confidence": 0.5}) == \
        "report_candidate N001-V001 @ 5s (0.5)"


# ---- tools ----------------------------------------------------------------------
def _run(**overrides) -> AgentRun:
    run = AgentRun(AgentRunRequest(query="thủ tướng", **overrides))
    run.agents["codex"] = AgentState("codex", "m", "high", run.started)
    return run


def _tools() -> AgentTools:
    return AgentTools(lambda: main.search_services)


def test_report_candidate_resolves_and_updates_in_place():
    tools, run = _tools(), _run()
    args = {"keyframe_id": "K01/K01_V001/006", "confidence": 0.4, "reason": "anchor visible"}
    first = asyncio.run(tools.call("report_candidate", args, run, "codex"))
    assert first["isError"] is False
    again = asyncio.run(tools.call("report_candidate", {**args, "confidence": 0.9}, run, "codex"))
    assert "Updated" in again["content"][0]["text"]
    assert len(run.candidates) == 1
    candidate = run.candidates[0]
    assert candidate["confidence"] == 0.9 and candidate["video_id"] == "K01_V001"
    assert candidate["pts_time"] == 20.0 and candidate["frame_idx"] == 500
    assert candidate["keyframe_url"].endswith("/Keyframes/Keyframes_K01/K01_V001/006.jpg")


def test_report_candidate_by_time_snaps_to_nearest_keyframe():
    tools, run = _tools(), _run()
    result = asyncio.run(tools.call(
        "report_candidate", {"video_id": "K01_V001", "time": 9.0, "confidence": 0.5, "reason": "r"}, run, "claude",
    ))
    assert result["isError"] is False
    assert run.candidates[0]["submit_keyframe_id"] == "K01/K01_V001/003"
    assert run.candidates[0]["time"] == 9.0


def test_report_candidate_rejects_bad_input():
    result = asyncio.run(_tools().call("report_candidate", {"confidence": 0.5, "reason": "r"}, _run(), "codex"))
    assert result["isError"] is True


def test_search_tool_lists_videos_and_keyframes():
    result = asyncio.run(_tools().call("search", {"query": "thủ tướng nhật bản", "mode": "ocr"}, _run(), "codex"))
    text = result["content"][0]["text"]
    assert result["isError"] is False
    assert "search mode=ocr" in text and "K01_V001" in text and "K01/K01_V001/" in text


def test_video_frames_returns_a_labelled_contact_sheet(monkeypatch):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 36), (200, 10, 10)).save(buffer, "JPEG")
    tools = _tools()

    async def fake_fetch(url, fallback):
        return buffer.getvalue()

    monkeypatch.setattr(tools, "_fetch", fake_fetch)
    result = asyncio.run(tools.call("video_frames", {"video_id": "K01_V001", "count": 8}, _run(), "codex"))
    assert result["isError"] is False
    text, image = result["content"]
    assert text["text"].startswith("K01_V001: duration 116.0s; 8 keyframes")
    assert image["type"] == "image" and image["mimeType"] == "image/jpeg" and image["data"]


def test_video_frames_unknown_video_is_a_tool_error():
    result = asyncio.run(_tools().call("video_frames", {"video_id": "K99_V999"}, _run(), "codex"))
    assert result["isError"] is True and "Unknown video" in result["content"][0]["text"]


def test_list_videos_reads_like_a_programme_guide():
    result = asyncio.run(_tools().call("list_videos", {"folder": "k01"}, _run(), "codex"))
    text = result["content"][0]["text"]
    assert result["isError"] is False
    assert text.splitlines()[0].startswith("K01 — 60-second news")
    assert "K01_V001  1:56  \"lũ gây sạt lở cô lập hoàn toàn bốn bản\"" in text
    assert "K01_V002  1:56  (no speech)" in text


def test_list_videos_pages_a_whole_series():
    tools, run = _tools(), _run()
    first = asyncio.run(tools.call("list_videos", {"folder": "K", "limit": 1}, run, "codex"))["content"][0]["text"]
    assert "K01_V001" in first and "offset=1" in first
    second = asyncio.run(tools.call("list_videos", {"folder": "K", "offset": 1}, run, "codex"))["content"][0]["text"]
    assert "K01_V002" in second and "K01_V001" not in second


def test_browsing_rejects_a_folder_the_profile_lacks():
    result = asyncio.run(_tools().call("list_videos", {"folder": "N001"}, _run(), "codex"))
    assert result["isError"] is True and "Unknown folder" in result["content"][0]["text"]


def test_camera_and_race_videos_are_described_from_the_catalogue():
    from app.agent.tools import _video_note

    assert _video_note("S01-V006").startswith("race stage 6 — Chương trình trực tiếp chặng 06")
    assert _video_note("N001-V001").startswith("camera An Dương Vương – Lê Hồng Phong; 2026-06-10 11:00:16")
    assert _video_note("L21_V001") is None


def test_folder_frames_shows_one_tile_per_video_or_per_folder(monkeypatch):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 36), (10, 200, 10)).save(buffer, "JPEG")
    tools = _tools()

    async def fake_fetch(url, fallback):
        return buffer.getvalue()

    monkeypatch.setattr(tools, "_fetch", fake_fetch)
    folder = asyncio.run(tools.call("folder_frames", {"folder": "K01", "position": 0}, _run(), "codex"))
    text = folder["content"][0]["text"]
    assert "one keyframe from each video" in text
    assert "#1 K01_V001 (1:56) K01/K01_V001/001" in text and "#2 K01_V002" in text
    assert folder["content"][1]["type"] == "image"
    series = asyncio.run(tools.call("folder_frames", {"folder": "L"}, _run(), "codex"))
    assert "the first video of each folder" in series["content"][0]["text"]
    assert "#1 L26_V001" in series["content"][0]["text"]


def test_video_outline_lists_what_is_said_and_shown():
    result = asyncio.run(_tools().call("video_outline", {"video_id": "K01_V001", "step": 30}, _run(), "codex"))
    text = result["content"][0]["text"]
    assert result["isError"] is False
    assert "one line per 30s" in text
    assert "[0:00] kính chào quý vị" in text and "HTV7 HD Bản tin thời sự" in text
    assert "[0:30] lũ gây sạt lở cô lập hoàn toàn bốn bản | —" in text


def test_contact_sheet_draws_missing_tiles():
    jpeg = _render_sheet([None, None, None], ["#1 0.0s", "#2 4.0s", "#3 8.0s"], (128, 72), 2)
    assert jpeg[:2] == b"\xff\xd8"


# ---- API + lifecycle with fake CLIs -----------------------------------------------
FAKE_CLI = """#!{python}
import json, os, sys, time
mode = os.environ.get("FAKE_AGENT_MODE", "ok")
if mode == "slow":
    time.sleep(60)
if mode == "quota":
    print("ERROR: You've hit your usage limit. Try again later.", file=sys.stderr)
    sys.exit(1)
print(json.dumps({{"type": "item.started", "item": {{"type": "mcp_tool_call", "tool": "search", "arguments": {{"query": "x"}}}}}}))
print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "No match found."}}}}))
print(json.dumps({{"type": "assistant", "message": {{"content": [{{"type": "tool_use", "name": "mcp__aic__search", "input": {{"query": "x"}}}}]}}}}))
print(json.dumps({{"type": "result", "subtype": "success", "result": "No match found."}}))
"""


@pytest.fixture
def fake_clis(tmp_path, monkeypatch):
    script = tmp_path / "fake-agent"
    script.write_text(FAKE_CLI.format(python=sys.executable))
    script.chmod(0o755)
    monkeypatch.setattr(main.settings, "agent_codex_bin", str(script))
    monkeypatch.setattr(main.settings, "agent_claude_bin", str(script))
    monkeypatch.setattr(main.settings, "agent_enabled", True)
    return script


def _wait(client: TestClient, run_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/agent/runs/{run_id}").json()
        if body["finished"]:
            return body
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} did not finish: {body}")


def test_agent_run_completes_and_reads_both_streams(fake_clis):
    with TestClient(main.app) as client:
        started = client.post("/api/agent/runs", json={"query": "người đi xe đạp", "query_type": "T-KIS"})
        assert started.status_code == 200
        run = _wait(client, started.json()["run_id"])
    assert run["agents"]["codex"]["status"] == "done"
    assert run["agents"]["claude"]["status"] == "done"
    assert run["agents"]["codex"]["summary"] == "No match found."
    assert run["agents"]["claude"]["tool_calls"] == 1


def test_tool_api_requires_the_live_run_token(fake_clis, monkeypatch):
    monkeypatch.setenv("FAKE_AGENT_MODE", "slow")
    with TestClient(main.app) as client:
        assert client.get("/api/agent/tools").status_code == 403
        assert client.post("/api/agent/tools/search", json={"arguments": {}}).status_code == 403
        run_id = client.post("/api/agent/runs", json={"query": "q"}).json()["run_id"]
        token = main.agent_service.get(run_id).token
        headers = {"X-AIC-Agent-Token": token, "X-AIC-Agent-Name": "claude"}
        tools = client.get("/api/agent/tools", headers=headers).json()["tools"]
        assert {t["name"] for t in tools} == {
            "search", "list_videos", "folder_frames", "video_outline",
            "video_frames", "view_frames", "video_text", "report_candidate",
        }
        reported = client.post("/api/agent/tools/report_candidate", headers=headers, json={
            "arguments": {"keyframe_id": "K01/K01_V001/006", "confidence": 0.8, "reason": "anchor"},
        })
        assert reported.json()["isError"] is False
        snapshot = client.get(f"/api/agent/runs/{run_id}").json()
        assert snapshot["candidates"][0]["agent"] == "claude"
        cancelled = client.delete(f"/api/agent/runs/{run_id}").json()
        assert {a["status"] for a in cancelled["agents"].values()} == {"cancelled"}
        # A cancelled run's token no longer opens the tool API.
        assert client.get("/api/agent/tools", headers=headers).status_code == 403


def test_new_run_replaces_the_previous_one(fake_clis, monkeypatch):
    monkeypatch.setenv("FAKE_AGENT_MODE", "slow")
    with TestClient(main.app) as client:
        first = client.post("/api/agent/runs", json={"query": "a"}).json()["run_id"]
        second = client.post("/api/agent/runs", json={"query": "b", "replaces": first}).json()["run_id"]
        assert client.get(f"/api/agent/runs/{first}").json()["agents"]["codex"]["status"] == "cancelled"
        client.delete(f"/api/agent/runs/{second}")


def test_agent_timeout_kills_the_process(fake_clis, monkeypatch):
    monkeypatch.setenv("FAKE_AGENT_MODE", "slow")
    monkeypatch.setattr(main.settings, "agent_timeout_seconds", 1.0)
    with TestClient(main.app) as client:
        run_id = client.post("/api/agent/runs", json={"query": "q", "agents": ["codex"]}).json()["run_id"]
        run = _wait(client, run_id)
    assert run["agents"]["codex"]["status"] == "timeout"
    assert run["agents"]["codex"]["error_kind"] == "timeout"


def test_quota_failure_is_labelled(fake_clis, monkeypatch):
    monkeypatch.setenv("FAKE_AGENT_MODE", "quota")
    with TestClient(main.app) as client:
        run_id = client.post("/api/agent/runs", json={"query": "q", "agents": ["codex"]}).json()["run_id"]
        run = _wait(client, run_id)
    assert run["agents"]["codex"]["status"] == "failed"
    assert run["agents"]["codex"]["error_kind"] == "quota"


def test_missing_cli_is_unavailable_not_an_error(monkeypatch):
    monkeypatch.setattr(main.settings, "agent_codex_bin", "definitely-not-a-real-cli")
    monkeypatch.setattr(main.settings, "agent_claude_bin", "definitely-not-a-real-cli")
    with TestClient(main.app) as client:
        body = client.post("/api/agent/runs", json={"query": "q"}).json()
        assert body["finished"] is True
        assert body["agents"]["codex"]["status"] == "unavailable"
        health = client.get("/api/health").json()
        assert health["capabilities"]["agent_codex"] is False


def test_disabled_agents_refuse_to_start(monkeypatch):
    monkeypatch.setattr(main.settings, "agent_enabled", False)
    with TestClient(main.app) as client:
        assert client.post("/api/agent/runs", json={"query": "q"}).status_code == 409


def test_unknown_run_is_404():
    with TestClient(main.app) as client:
        assert client.get("/api/agent/runs/nope").status_code == 404
        assert client.delete("/api/agent/runs/nope").status_code == 404
