"""Agent runs: Codex and Claude searching next to the main pipeline.

A run is started when the operator presses Search with the AGENT button on. It
starts a bounded decision controller and returns at once. The controller reuses
server-issued retrieval evidence, selectively invokes CLIs, and publishes a
combined ranking. The console polls it without delaying the main search.

Isolation rules, in order of importance:
- the main search never awaits an agent, and an agent failure is a status,
  never an exception that reaches another endpoint;
- a new search in the same tab cancels that tab's previous run (`replaces`),
  and every agent is killed at `AGENT_TIMEOUT_SECONDS`: a candidate that
  arrives after the task closes is worth nothing;
- at most `AGENT_MAX_CONCURRENT` processes per CLI at once, across every tab,
  at lower CPU priority; the rest queue;
- the CLIs get no backend credentials — only a per-run token for the tool API.
"""
from __future__ import annotations

import asyncio
import atexit
import copy
import json
import os
import secrets
import shutil
import signal
import tempfile
import time
import uuid
from collections import OrderedDict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .cli import (
    agent_process_env,
    bridge_env,
    classify_error,
    claude_argv,
    codex_argv,
    read_claude_event,
    read_codex_event,
)
from ..scope import resolve_scope
from .prompt import build_prompt
from .tools import AgentTools
from .evidence.board import EvidenceBoard
from .controller import Controller, POLICIES

if TYPE_CHECKING:
    from ..config import Settings
    from ..models import AgentRunRequest
    from ..services.search_service import SearchService

TERMINAL = frozenset({"done", "failed", "timeout", "cancelled", "unavailable", "skipped"})
#: A Codex `item.completed` line for a contact sheet carries the whole base64
#: image, far past asyncio's 64 KiB default line limit.
_STREAM_LIMIT = 64 * 1024 * 1024
_MAX_RUNS = 40
_FINISHED_TTL_S = 3600.0
_KILL_GRACE_S = 3.0


class AgentDisabled(Exception):
    pass


class AgentState:
    def __init__(self, name: str, model: str, effort: str, run_started: float, *, fast: bool = False):
        self.name = name
        self.model = model
        self.effort = effort
        self.fast = fast
        self.status = "pending"
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.summary = ""
        self.error: str | None = None
        self.error_kind: str | None = None
        self.fail_reason: str | None = None
        self.tool_calls = 0
        self.cost_usd: float | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.steps: deque[dict[str, Any]] = deque(maxlen=60)
        self._run_started = run_started

    def step(self, kind: str, text: str) -> None:
        text = " ".join(str(text or "").split())
        if not text:
            return
        self.steps.append({
            "at": round(time.monotonic() - self._run_started, 1),
            "kind": kind,
            "text": text if len(text) <= 400 else text[:399] + "…",
        })

    def begin(self) -> None:
        self.status = "running"
        self.started_at = time.monotonic()

    def finish(self, status: str, *, error: str | None = None, kind: str | None = None) -> None:
        if self.status in TERMINAL:
            return
        self.status = status
        self.finished_at = time.monotonic()
        if error:
            self.error = error[:800]
            self.error_kind = kind or classify_error(error)

    def to_dict(self) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = None
        if self.started_at is not None:
            elapsed = round((self.finished_at or now) - self.started_at, 1)
        return {
            "name": self.name,
            "status": self.status,
            "model": self.model,
            "effort": self.effort,
            "fast": self.fast,
            "elapsed_s": elapsed,
            "tool_calls": self.tool_calls,
            "summary": self.summary,
            "error": self.error,
            "error_kind": self.error_kind,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "steps": list(self.steps)[-25:],
        }


class AgentRun:
    def __init__(self, request: "AgentRunRequest"):
        self.id = uuid.uuid4().hex[:12]
        self.token = secrets.token_urlsafe(24)
        self.request = request
        # The folders the operator's filter narrows this query to, resolved ONCE
        # with the same rules as the main search (a manual pick, or the topic
        # heuristic on the query in `auto`). The agents look there first; the
        # prompt, the `search` default and the "outside the filter" flag on a
        # candidate all read this.
        self.scope = resolve_scope(
            request.scope.model_dump(), query=request.query, retrieval_database=request.retrieval_database,
        )
        self.created_at = time.time()
        self.started = time.monotonic()
        self.agents: dict[str, AgentState] = {}
        self.candidates: list[dict[str, Any]] = []
        self.tasks: dict[str, asyncio.Task] = {}
        self.tool_tasks: set[asyncio.Task] = set()
        self.cancelled = False
        self.updated = time.monotonic()
        self.finished_at: float | None = None
        self.policy = request.policy or "full"
        self.board = EvidenceBoard(request.query, request.query_type)
        self.baseline: list[dict] = []
        self.pre_verification_ranking: list[dict] | None = None
        self.ranking: list[dict] = []
        self.trace: list[dict] = []
        self.controller: dict = {"status": "pending", "steps": 0}
        self.controller_active = False
        self.tool_count = 0
        self.max_tool_calls = 40
        self.cache_hits = 0
        self.tool_cache: dict = {}
        self.tool_locks: dict[str, asyncio.Lock] = {}
        self.agent_invocations = {"codex": 0, "claude": 0}

    def publish_ranking(self):
        self.ranking = self.board.ranked(verified=self.policy not in {"retrieval", "codex", "parallel", "rule"})
        self.trace.append({"kind": "ranking", "at_s": round(time.monotonic() - self.started, 3),
                           "ranking": copy.deepcopy(self.ranking[:100])})

    @property
    def finished(self) -> bool:
        return not self.controller_active and all(state.status in TERMINAL for state in self.agents.values())

    def add_candidate(self, candidate: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Store what an agent reported; a repeat of the same frame updates it."""
        self.updated = time.monotonic()
        key = (candidate["agent"], candidate["submit_keyframe_id"], candidate.get("event"))
        for existing in self.candidates:
            if (existing["agent"], existing["submit_keyframe_id"], existing.get("event")) == key:
                existing.update({
                    field: candidate[field]
                    for field in ("confidence", "reason", "answer", "time")
                    if candidate.get(field) is not None
                })
                existing["updated_at_s"] = round(time.monotonic() - self.started, 1)
                return existing, True
        stored = {
            **candidate,
            "id": f"{candidate['agent']}-{sum(1 for c in self.candidates if c['agent'] == candidate['agent']) + 1}",
            "found_at_s": round(time.monotonic() - self.started, 1),
        }
        self.candidates.append(stored)
        return stored, False

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "query": self.request.query,
            "query_type": self.request.query_type,
            "retrieval_database": self.request.retrieval_database,
            "created_at": self.created_at,
            "elapsed_s": round((self.finished_at or time.monotonic()) - self.started, 3),
            "finished": self.finished,
            "agents": {name: state.to_dict() for name, state in self.agents.items()},
            "candidates": list(self.candidates),
            "policy": self.policy,
            "controller": self.controller,
            "ranking": self.ranking,
            "baseline_ranking": self.baseline,
            "pre_verification_ranking": self.pre_verification_ranking,
            "metrics": {"tool_calls": self.tool_count, "cache_hits": self.cache_hits,
                        "agent_disagreement": self.board.disagreement(),
                        "agent_calls": self.agent_invocations},
            "scope": {
                "mode": self.scope.mode,
                "active": self.scope.active,
                "categories": list(self.scope.categories),
                "reason": self.scope.reason_en,
            },
        }


def _signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class AgentService:
    def __init__(
        self,
        settings: Callable[[], "Settings"],
        services: Callable[[], dict[str, "SearchService"]],
        trake_services: Callable | None = None,
    ):
        # Both resolved per call: a config import swaps the whole service stack,
        # and a run that outlives the swap must use the new one.
        self._settings = settings
        self._trake_services = trake_services
        self.tools = AgentTools(services)
        self._runs: OrderedDict[str, AgentRun] = OrderedDict()
        self._by_token: dict[str, str] = {}
        self._retrievals: OrderedDict[str, tuple] = OrderedDict()
        self._procs: set[asyncio.subprocess.Process] = set()
        self._gates: dict[tuple[int, str], asyncio.Semaphore] = {}
        atexit.register(self._kill_all)

    # ---- introspection -------------------------------------------------
    def _binary(self, name: str) -> str | None:
        s = self._settings()
        configured = s.agent_codex_bin if name == "codex" else s.agent_claude_bin
        return shutil.which(configured)

    @staticmethod
    def _calibration_hash(path):
        if not path:
            return None
        import hashlib
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return "unreadable"

    def status(self) -> dict[str, Any]:
        s = self._settings()
        return {
            "enabled": s.agent_enabled,
            "policy": s.agent_policy,
            "controller_config": {"max_steps": s.agent_max_steps, "max_tool_calls": s.agent_max_tool_calls,
                                  "verify_top_k": s.agent_verify_top_k, "stop_threshold": s.agent_stop_threshold,
                                  "stop_margin": s.agent_stop_margin, "jev_timeout_seconds": s.jev_timeout_seconds,
                                  "llm_judge_model": s.agent_llm_judge_model,
                                  "calibration_sha256": self._calibration_hash(s.agent_calibration_path)},
            "jev": {"model": s.jev_model, "configured": bool(s.openrouter_api_key) and not s.mock_mode},
            "timeout_seconds": s.agent_timeout_seconds,
            "max_concurrent": s.agent_max_concurrent,
            "agents": {
                "codex": {
                    "available": s.agent_enabled and self._binary("codex") is not None,
                    "model": s.agent_codex_model,
                    "effort": s.agent_codex_reasoning_effort,
                    "fast": s.agent_codex_fast,
                },
                "claude": {
                    "available": s.agent_enabled and self._binary("claude") is not None,
                    "model": s.agent_claude_model,
                    "effort": s.agent_claude_effort,
                    "fast": s.agent_claude_fast,
                },
            },
            "running": sum(
                1 for run in self._runs.values()
                for state in run.agents.values() if state.status in {"queued", "running"}
            ),
        }

    def get(self, run_id: str) -> AgentRun | None:
        return self._runs.get(run_id)

    def authorize(self, token: str | None) -> AgentRun | None:
        """The live run a tool call belongs to, or None (unknown, stale or cancelled)."""
        if not token:
            return None
        run = self._runs.get(self._by_token.get(token, ""))
        if run is None or run.cancelled or run.finished:
            return None
        return run

    # ---- lifecycle ------------------------------------------------------
    async def start(self, request: "AgentRunRequest", *, backend_url: str) -> dict[str, Any]:
        if request.replaces:
            await self.cancel(request.replaces)
        s = self._settings()
        if not s.agent_enabled:
            raise AgentDisabled("Agent search is disabled (AGENT_ENABLED=false).")
        run = AgentRun(request)
        binaries: dict[str, str] = {}
        for name in dict.fromkeys(request.agents):
            model = s.agent_codex_model if name == "codex" else s.agent_claude_model
            effort = s.agent_codex_reasoning_effort if name == "codex" else s.agent_claude_effort
            fast = s.agent_codex_fast if name == "codex" else s.agent_claude_fast
            state = AgentState(name, model, effort, run.started, fast=fast)
            binary = self._binary(name)
            if binary is None:
                configured = s.agent_codex_bin if name == "codex" else s.agent_claude_bin
                state.finish("unavailable", error=f"{configured} not found on PATH", kind="cli")
            else:
                binaries[name] = binary
            run.agents[name] = state
        self._runs[run.id] = run
        self._by_token[run.token] = run.id
        run.policy = request.policy or s.agent_policy
        if run.policy not in POLICIES:
            run.policy = "full"
        run.max_tool_calls = s.agent_max_tool_calls
        run.controller_active = True
        seed = self.take_retrieval(request)
        controller = Controller(self, run, binaries, backend_url, s, seed)
        run.tasks["controller"] = asyncio.create_task(controller.run_controller(), name=f"controller-{run.id}")
        self._collect_garbage()
        return run.snapshot()

    def remember_retrieval(self, payload: dict, result: dict) -> str | None:
        # Previous hints and feedback are excluded from SOICT evidence.
        if payload.get("previous_hints") or payload.get("feedback"):
            return None
        key = secrets.token_urlsafe(24)
        self._retrievals[key] = (time.monotonic(), payload, result)
        while len(self._retrievals) > 40:
            self._retrievals.popitem(last=False)
        return key

    def take_retrieval(self, request):
        if not request.retrieval_id:
            return None
        entry = self._retrievals.get(request.retrieval_id)
        if not entry:
            return None
        created, payload, result = entry
        if time.monotonic() - created > 120:
            self._retrievals.pop(request.retrieval_id, None)
            return None
        if (payload.get("query") != request.query or payload.get("retrieval_database", "btc") != request.retrieval_database
                or payload.get("image_models", ["pe"]) != request.image_models
                or payload.get("scope", {"mode": "all", "categories": []}) != request.scope.model_dump()):
            return None
        return result

    async def cancel(self, run_id: str) -> dict[str, Any] | None:
        run = self._runs.get(run_id)
        if run is None:
            return None
        run.cancelled = True
        run.controller_active = False
        run.board.closed = True
        pending = [task for task in {*run.tasks.values(), *run.tool_tasks} if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for state in run.agents.values():
            state.finish("cancelled")
        run.controller.update(status="cancelled", stop_reason="cancelled")
        run.finished_at = run.finished_at or time.monotonic()
        run.updated = run.finished_at
        return run.snapshot()

    def _gate(self, name: str, limit: int) -> asyncio.Semaphore:
        # Per event loop: a semaphore is bound to the loop it first waits on.
        key = (id(asyncio.get_running_loop()), name)
        gate = self._gates.get(key)
        if gate is None:
            gate = self._gates[key] = asyncio.Semaphore(limit)
        return gate

    async def _run_agent(self, run: AgentRun, state: AgentState, binary: str, backend_url: str) -> None:
        s = self._settings()
        state.status = "queued"
        proc: asyncio.subprocess.Process | None = None
        workdir: Path | None = None
        stderr_tail: deque[str] = deque(maxlen=30)
        try:
            async with self._gate(state.name, s.agent_max_concurrent):
                if run.cancelled:
                    state.finish("cancelled")
                    return
                workdir = Path(tempfile.mkdtemp(prefix=f"aic26-agent-{state.name}-"))
                prompt = build_prompt(run.request, timeout_seconds=max(1, s.agent_timeout_seconds - (time.monotonic() - run.started)),
                                      scope=run.scope, agent=state.name if run.request.specialization else None,
                                      evidence=run.board.state(s.agent_verify_top_k))
                bridge = bridge_env(backend_url, run.token, state.name)
                if state.name == "codex":
                    argv = codex_argv(s, binary=binary, workdir=workdir, prompt=prompt, bridge=bridge)
                    reader = read_codex_event
                else:
                    argv = claude_argv(s, binary=binary, prompt=prompt, bridge=bridge)
                    reader = read_claude_event
                nice = shutil.which("nice")
                if nice:
                    # Below the backend's own priority: the CLI's local work must
                    # never slow the main search down.
                    argv = [nice, "-n", "10", *argv]
                state.begin()
                state.step("info", f"started {state.model} ({state.effort}{', fast' if state.fast else ''})")
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workdir),
                    env=agent_process_env(),
                    start_new_session=True,
                    limit=_STREAM_LIMIT,
                )
                self._procs.add(proc)

                async def pump_stdout() -> None:
                    assert proc is not None and proc.stdout is not None
                    async for raw in proc.stdout:
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except ValueError:
                            stderr_tail.append(line[:500])
                            continue
                        if isinstance(event, dict):
                            reader(event, state)
                            run.updated = time.monotonic()

                async def pump_stderr() -> None:
                    assert proc is not None and proc.stderr is not None
                    async for raw in proc.stderr:
                        line = raw.decode("utf-8", errors="replace").rstrip()
                        if line:
                            stderr_tail.append(line[:500])

                try:
                    await asyncio.wait_for(
                        asyncio.gather(pump_stdout(), pump_stderr(), proc.wait()),
                        timeout=s.agent_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    await self._terminate(proc)
                    state.finish(
                        "timeout",
                        error=f"Stopped after {int(s.agent_timeout_seconds)} s (AGENT_TIMEOUT_SECONDS).",
                        kind="timeout",
                    )
                    return
                if proc.returncode == 0 and not state.fail_reason:
                    state.finish("done")
                else:
                    detail = state.fail_reason or " | ".join(list(stderr_tail)[-4:]) or f"exit code {proc.returncode}"
                    state.finish("failed", error=detail, kind=classify_error(detail + " " + " ".join(stderr_tail)))
        except asyncio.CancelledError:
            if proc is not None:
                await self._terminate(proc)
            state.finish("cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - a broken agent is a status, never an outage
            state.finish("failed", error=f"{type(exc).__name__}: {exc}", kind="cli")
        finally:
            if proc is not None:
                if proc.returncode is None:
                    _signal_group(proc, signal.SIGKILL)
                self._procs.discard(proc)
            if workdir is not None:
                shutil.rmtree(workdir, ignore_errors=True)
            run.updated = time.monotonic()

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        """SIGTERM the CLI's whole process group (it spawns the MCP bridge), then SIGKILL."""
        if proc.returncode is not None:
            return
        _signal_group(proc, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=_KILL_GRACE_S)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _signal_group(proc, signal.SIGKILL)

    def _kill_all(self) -> None:
        """At interpreter exit: no agent outlives the backend that owns its run."""
        for proc in list(self._procs):
            if proc.returncode is None:
                _signal_group(proc, signal.SIGKILL)

    def _collect_garbage(self) -> None:
        now = time.monotonic()
        for run_id, run in list(self._runs.items()):
            too_many = len(self._runs) > _MAX_RUNS
            stale = run.finished and now - run.updated > _FINISHED_TTL_S
            if (too_many and run.finished) or stale:
                self._runs.pop(run_id, None)
                self._by_token.pop(run.token, None)
