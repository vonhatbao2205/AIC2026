"""Bounded fast/slow cascade, shared by the console and benchmark harness."""
from __future__ import annotations

import asyncio
import copy
import time

from .decision.calibration import Calibration
from .decision.jev_client import DecisionError, JevClient
from .decision.llm_client import LLMDecisionClient
from .decision.router import fallback_action, route
from .decision.stopping import can_stop
from .decision.verifier import verify
from .cli import classify_error

POLICIES = {"retrieval", "codex", "parallel", "parallel_verify", "rerank", "adaptive", "full", "rule"}


class ModelQuotaError(Exception):
    """An interrupted attempt must be retried from a fresh query state."""


class Controller:
    def __init__(self, service, run, binaries, backend_url, settings, seed=None):
        self.service, self.run, self.binaries = service, run, binaries
        self.backend_url, self.s, self.seed = backend_url, settings, seed
        self.client = LLMDecisionClient(settings) if run.request.verifier == "llm" else JevClient(settings)
        self.router_client = (self.client if run.request.router == run.request.verifier else
                              LLMDecisionClient(settings) if run.request.router == "llm" else JevClient(settings))
        self.calibration = Calibration()
        self.invoked: set[str] = set()
        self.actions: set[str] = set()
        self.jev_failed = False
        self._verified_version = -1
        self._stop_supported = False
        self.stop_threshold = settings.agent_stop_threshold
        self.stop_margin = settings.agent_stop_margin

    def trace(self, kind: str, **fields):
        self.run.trace.append({"at_s": round(time.monotonic() - self.run.started, 3), "kind": kind, **copy.deepcopy(fields)})

    async def launch(self, names):
        jobs = []
        for name in names:
            if name not in self.binaries or name in self.invoked:
                continue
            self.invoked.add(name)
            self.run.agent_invocations[name] += 1
            task = asyncio.create_task(self.service._run_agent(self.run, self.run.agents[name], self.binaries[name], self.backend_url))
            self.run.tasks[name] = task
            jobs.append(task)
        if jobs:
            try:
                for completed in asyncio.as_completed(jobs):
                    await completed
                    if any(self.run.agents[name].error_kind == "quota" for name in names if name in self.run.agents):
                        raise ModelQuotaError()
            except BaseException:
                # A parallel peer must not keep consuming quota after one CLI
                # exhausted its allowance. Also preserve deadline cancellation.
                for task in jobs:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)
                raise
        if self.run.tool_tasks:
            await asyncio.gather(*list(self.run.tool_tasks), return_exceptions=True)
        self.trace("agent_results", agents=list(names), candidates=list(self.run.candidates))

    async def execute(self, action: str):
        run = self.run
        self.actions.add(action)
        self.trace("action", action=action)
        if action.startswith("CALL_"):
            await self.launch({"CALL_CODEX": ["codex"], "CALL_CLAUDE": ["claude"], "CALL_BOTH": ["codex", "claude"]}[action])
            return
        rows = run.board.ranked()
        if action == "SEARCH_MORE":
            missing = next((c for c in run.board.constraints if c.id != "query" and
                            (not rows or rows[0].get("constraint_scores", {}).get(c.id, {}).get("supported", 0) < self.s.agent_stop_threshold)),
                           run.board.constraints[0])
            mode = run.controller.get("modality", "visual")
            await self.service.tools.call("search", {"query": missing.text,
                                          "mode": mode if mode in {"ocr", "speech", "visual"} else "hybrid"}, run, "controller")
        else:
            # Jev is text-only: backend macros collect local text. A visual
            # agent calls compare_candidates itself when image inspection is needed.
            for frame in rows[:2 if action == "COMPARE_TOP2" else 1]:
                t = frame.get("pts_time")
                if t is not None:
                    await self.service.tools.call("video_text", {"video_id": frame["video_id"], "start": max(0, t - 15), "end": t + 15}, run, "controller")

    async def evaluate(self):
        if self.jev_failed or self.run.request.verifier == "none":
            return False
        if self._verified_version == self.run.board.version:
            return self._stop_supported
        try:
            observations = await verify(self.client, self.run.board, self.calibration,
                                        limit=self.s.agent_verify_top_k, constraints=self.run.policy == "full")
            self.trace("verification", observations=observations)
            self.run.publish_ranking()
            self._verified_version = self.run.board.version
            self._stop_supported = can_stop(self.run.board, self.stop_threshold, self.stop_margin,
                                            require_constraints=self.run.policy == "full")
            self.trace("stop_check", can_stop=self._stop_supported, threshold=self.stop_threshold,
                       margin=self.stop_margin, agents_invoked=sorted(self.invoked))
            return self._stop_supported
        except DecisionError as exc:
            if classify_error(str(exc)) == "quota":
                self.run.controller["quota_error"] = "decision_provider"
                raise ModelQuotaError() from None
            self.jev_failed = True
            self.run.controller["fallback"] = str(exc)
            self.trace("provider_error", error=str(exc))
            return False

    async def work(self):
        run, s = self.run, self.s
        try:
            self.calibration = Calibration.load(s.agent_calibration_path,
                model=s.agent_llm_judge_model if run.request.verifier == "llm" else s.jev_model,
                aggregation="constraints" if run.policy == "full" else "holistic")
        except (ValueError, OSError, KeyError, TypeError):
            run.controller["calibration_warning"] = "Invalid calibration artifact; using raw probabilities"
        run.controller["calibrated"] = self.calibration.fitted
        run.controller["calibration_query_ids"] = list(self.calibration.query_ids)
        if self.calibration.stop_threshold is not None:
            self.stop_threshold = self.calibration.stop_threshold
        if self.calibration.stop_margin is not None:
            self.stop_margin = self.calibration.stop_margin
        task_stopping = self.calibration.stopping_by_task.get(run.request.query_type, {})
        self.stop_threshold = task_stopping.get("threshold", self.stop_threshold)
        self.stop_margin = task_stopping.get("margin", self.stop_margin)
        run.controller.update(stop_threshold=self.stop_threshold, stop_margin=self.stop_margin,
                              calibration_temperature=self.calibration.temperature)
        run.controller["status"] = "retrieving"
        if self.seed is None:
            # No previous_hints or previous query state in the experimental path.
            payload = {
                "query": run.request.query, "query_type_hint": run.request.query_type,
                "image_models": run.request.image_models, "scope": run.request.scope.model_dump(),
                "previous_hints": [], "use_llm": False, "expand": False, "translate": True,
                "rerank": False, "top_k": 150, "max_videos": 30,
            }
            try:
                trake = getattr(self.service, "_trake_services", None)
                if run.request.query_type == "TRAKE" and trake is not None:
                    self.seed = await trake()[run.request.retrieval_database].search_trake(payload)
                else:
                    self.seed = await self.service.tools._service(run).search(payload)
            except Exception as exc:
                run.controller["retrieval_error"] = type(exc).__name__
                self.trace("retrieval_error", error=type(exc).__name__)
                self.seed = {"groups": []}
        if "sequences" in self.seed:
            self.seed = {**self.seed, "groups": [{"video_id": seq["video_id"], "frames": [
                {**f, "video_id": seq["video_id"], "video_url": seq.get("video_url"), "event": f["event_index"]}
                for f in seq["frames"]]} for seq in self.seed["sequences"]]}
        run.board.seed(self.seed, at=time.monotonic() - run.started)
        run.baseline = run.board.ranked(verified=False)
        warnings = self.seed.get("warnings", [])
        run.controller["retrieval_degraded"] = any("unavailable" in str(w).lower() for w in warnings)
        if not run.baseline and run.controller["retrieval_degraded"]:
            run.controller.setdefault("retrieval_error", "Retrieval channels unavailable")
        self.trace("retrieval", ranking=run.baseline, warnings=self.seed.get("warnings", []))
        run.publish_ranking()
        if run.policy == "retrieval":
            run.controller["stop_reason"] = "retrieval_only"
            return
        if run.policy in {"codex", "parallel"}:
            await self.launch(["codex"] if run.policy == "codex" else list(self.binaries))
            run.controller["stop_reason"] = "fixed_policy_complete"
            return
        if run.policy == "parallel_verify":
            # C+V: identical acquisition to C, followed by E's holistic verifier.
            # No pre-verification, adaptive routing, or early stopping. Preserve
            # the same-evidence RRF ordering for the within-run ranking contrast.
            await self.launch(list(self.binaries))
            run.pre_verification_ranking = copy.deepcopy(run.board.ranked(verified=False))
            self.trace("pre_verification", ranking=run.pre_verification_ranking)
            run.controller["status"] = "verifying"
            await self.evaluate()
            run.controller["stop_reason"] = "fixed_verification_complete"
            return
        if run.policy == "rerank":
            await self.evaluate()
            run.controller["stop_reason"] = "reranking_complete"
            return
        run.controller["status"] = "deciding"
        initial = True
        for step in range(s.agent_max_steps):
            run.controller["steps"] = step + 1
            if run.cancelled:
                return
            if run.policy != "rule" and await self.evaluate():
                run.controller["stop_reason"] = "evidence_sufficient"
                self.trace("stop", reason="evidence_sufficient")
                return
            available = ["STOP"]
            for name in self.binaries:
                if name not in self.invoked:
                    available.append("CALL_" + name.upper())
            if "CALL_CODEX" in available and "CALL_CLAUDE" in available:
                available.append("CALL_BOTH")
            if run.policy == "full" and not self.jev_failed and run.tool_count < run.max_tool_calls:
                if "SEARCH_MORE" not in self.actions:
                    available.append("SEARCH_MORE")
                if run.board.candidates and "INSPECT_TOP1" not in self.actions:
                    available.append("INSPECT_TOP1")
                if len(run.board.candidates) >= 2 and "COMPARE_TOP2" not in self.actions and run.request.toolset != "base":
                    available.append("COMPARE_TOP2")
            if available == ["STOP"]:
                run.controller["stop_reason"] = "actions_exhausted"
                return
            action = fallback_action(available)
            if run.policy != "rule" and run.request.router != "rule" and not self.jev_failed:
                try:
                    answers = await route(self.router_client, run.board, available, initial=initial,
                                          remaining_steps=s.agent_max_steps - step, limit=s.agent_verify_top_k)
                    self.trace("decision", answers=answers, available_actions=available)
                    action = answers["action"]["choice"]
                    for field in ("modality", "complexity"):
                        if field in answers:
                            run.controller[field] = answers[field]["choice"]
                except DecisionError as exc:
                    if classify_error(str(exc)) == "quota":
                        run.controller["quota_error"] = "decision_provider"
                        raise ModelQuotaError() from None
                    self.jev_failed = True
                    run.controller["fallback"] = str(exc)
                    self.trace("provider_error", error=str(exc))
            initial = False
            # STOP chosen without sufficient evidence cannot certify success.
            # Escalate conservatively if an untried action remains.
            if action == "STOP":
                if run.request.verifier == "none" and not self.jev_failed:
                    run.controller["stop_reason"] = "router_stop_unverified"
                    return
                action = fallback_action(available)
                self.trace("stop_overridden", reason="insufficient_evidence", action=action)
            if action == "STOP":
                run.controller["stop_reason"] = "actions_exhausted"
                return
            run.controller["action"] = action
            run.controller["status"] = "searching"
            await self.execute(action)
            run.publish_ranking()
        if run.policy != "rule" and await self.evaluate():
            run.controller["stop_reason"] = "evidence_sufficient"
        else:
            run.controller["stop_reason"] = "step_budget"

    async def run_controller(self):
        run = self.run
        try:
            await asyncio.wait_for(self.work(), timeout=self.s.agent_timeout_seconds)
            run.controller["status"] = "done"
        except ModelQuotaError:
            run.controller.update(status="interrupted", stop_reason="model_quota")
            self.trace("quota_interruption")
        except asyncio.TimeoutError:
            run.controller.update(status="timeout", stop_reason="time_budget")
            self.trace("timeout")
            for state in run.agents.values():
                if state.status == "cancelled" and not run.cancelled:
                    state.status = "timeout"
                    state.error = "Query wall-clock budget exhausted"
                    state.error_kind = "timeout"
        except asyncio.CancelledError:
            run.controller.update(status="cancelled", stop_reason="cancelled")
            raise
        except Exception as exc:
            # Upstream exceptions may contain credentials. Keep only the type.
            run.controller.update(status="failed", error=type(exc).__name__, stop_reason="error")
            self.trace("error", error=type(exc).__name__)
        finally:
            run.board.closed = True
            outstanding_tools = [task for task in run.tool_tasks if not task.done()]
            for task in outstanding_tools:
                task.cancel()
            if outstanding_tools:
                await asyncio.gather(*outstanding_tools, return_exceptions=True)
            for name, state in run.agents.items():
                if state.status not in {"done", "failed", "timeout", "cancelled", "unavailable"}:
                    state.summary = "Not needed by controller" if not run.cancelled else "Run cancelled"
                    state.finish("skipped" if not run.cancelled else "cancelled")
            decision_calls = list(self.client.calls)
            if self.router_client is not self.client:
                decision_calls.extend(self.router_client.calls)
            run.controller["decision_calls"] = decision_calls
            run.controller["jev_calls"] = [c for c in decision_calls if c.get("backend") == "jev"]
            run.controller["verifier"] = run.request.verifier
            run.controller["router"] = run.request.router
            run.publish_ranking()
            run.controller_active = False
            run.finished_at = time.monotonic()
            run.updated = run.finished_at
            await self.client.close()
            if self.router_client is not self.client:
                await self.router_client.close()
