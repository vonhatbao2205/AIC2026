"""Client for the official DRES evaluation server (Client API v2).

Protocol (clientapi.json 2.0.5 — see `instruction.md`):

    POST /api/v2/login                                  -> {id, username, role, sessionId}
    GET  /api/v2/client/evaluation/list?session=…       -> [ApiClientEvaluationInfo]
    GET  /api/v2/client/evaluation/currentTask/{id}?…   -> {name, taskGroup, taskType, duration}
    GET  /api/v2/evaluation/{id}/state?session=…        -> {taskStatus, timeLeft, timeElapsed}
    POST /api/v2/submit/{id}?session=…                  -> {status, submission, description}

Every authenticated call carries the session token as the `session` query
parameter (DRES also accepts its cookie, but the query form survives the
frontend → backend → DRES hop). Sessions expire, so a 401/403 triggers one
re-login and one retry; that is what keeps a long competition run alive without
the operator touching anything.

Credentials never leave the backend: the frontend only ever sees task/verdict
data returned by the endpoints in `app.main`.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from ..config import Settings
from .http_pool import PooledHttpClient


class DresError(RuntimeError):
    """A DRES call failed. `status` is the HTTP code when the server answered."""

    def __init__(self, message: str, *, status: int | None = None, detail: Any = None):
        super().__init__(message)
        self.status = status
        self.detail = detail


class DresNotConfigured(DresError):
    def __init__(self) -> None:
        super().__init__("DRES is not configured (set DRES_BASE_URL + DRES_USERNAME/DRES_PASSWORD)")


class DresClient:
    """Session-managing DRES v2 client shared by the whole backend process."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._pool = PooledHttpClient(
            timeout=httpx.Timeout(settings.dres_timeout_seconds, connect=10.0)
        )
        # A session id supplied by env skips login entirely (and cannot be renewed).
        self._session_id: str | None = settings.dres_session
        self._user: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    # ---- plumbing ---------------------------------------------------------
    @property
    def configured(self) -> bool:
        return self.s.has_dres

    @property
    def base_url(self) -> str:
        return (self.s.dres_base_url or "").rstrip("/")

    @property
    def can_login(self) -> bool:
        return bool(self.s.dres_username and self.s.dres_password)

    @property
    def user(self) -> dict[str, Any] | None:
        return self._user

    @property
    def logged_in(self) -> bool:
        return bool(self._session_id)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _get_lock(self) -> asyncio.Lock:
        """One lock per running loop (tests spin up their own loop per test)."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    @staticmethod
    def _describe(resp: httpx.Response) -> str:
        try:
            body = resp.json()
        except ValueError:
            return resp.text[:200] or f"HTTP {resp.status_code}"
        if isinstance(body, dict) and body.get("description"):
            return str(body["description"])
        return str(body)[:200]

    # ---- session ----------------------------------------------------------
    async def login(self, *, force: bool = False) -> dict[str, Any]:
        """Log in and cache the session id. Idempotent unless `force`."""
        if not self.configured:
            raise DresNotConfigured()
        async with self._get_lock():
            if self._session_id and not force:
                return self._user or {"sessionId": self._session_id}
            if not self.can_login:
                # Session-only configuration: nothing to renew.
                if self._session_id:
                    return {"sessionId": self._session_id}
                raise DresNotConfigured()
            client = self._pool.get()
            try:
                resp = await client.post(
                    f"{self.base_url}/api/v2/login",
                    json={"username": self.s.dres_username, "password": self.s.dres_password},
                    headers={"Content-Type": "application/json"},
                )
            except httpx.HTTPError as exc:
                self._last_error = f"login failed: {exc}"
                raise DresError(self._last_error) from exc
            if resp.status_code != 200:
                self._last_error = f"login rejected ({resp.status_code}): {self._describe(resp)}"
                raise DresError(self._last_error, status=resp.status_code)
            data = resp.json()
            session_id = data.get("sessionId")
            if not session_id:
                self._last_error = "login response has no sessionId"
                raise DresError(self._last_error, status=200, detail=data)
            self._session_id = session_id
            self._user = {
                "id": data.get("id"),
                "username": data.get("username"),
                "role": data.get("role"),
            }
            self._last_error = None
            return self._user

    async def _session(self) -> str:
        if self._session_id:
            return self._session_id
        await self.login()
        return self._session_id or ""

    async def _call(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        accept: tuple[int, ...] = (200,),
        none_on: tuple[int, ...] = (),
    ) -> Any:
        """Authenticated call with a single re-login retry on an expired session."""
        if not self.configured:
            raise DresNotConfigured()
        client = self._pool.get()
        url = f"{self.base_url}{path}"

        async def once(session: str) -> httpx.Response:
            try:
                return await client.request(method, url, params={"session": session}, json=json_body)
            except httpx.HTTPError as exc:
                self._last_error = f"{method} {path} failed: {exc}"
                raise DresError(self._last_error) from exc

        resp = await once(await self._session())
        if resp.status_code in (401, 403) and self.can_login:
            await self.login(force=True)
            resp = await once(self._session_id or "")

        if resp.status_code in none_on:
            return None
        if resp.status_code not in accept:
            self._last_error = f"{method} {path} -> {resp.status_code}: {self._describe(resp)}"
            detail = None
            try:
                detail = resp.json()
            except ValueError:
                pass
            raise DresError(self._last_error, status=resp.status_code, detail=detail)
        self._last_error = None
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError as exc:
            raise DresError(f"{method} {path}: response is not JSON", status=resp.status_code) from exc

    # ---- evaluation state -------------------------------------------------
    async def evaluations(self) -> list[dict[str, Any]]:
        """All evaluation runs visible to this participant account."""
        data = await self._call("GET", "/api/v2/client/evaluation/list")
        return list(data or [])

    async def current_task(self, evaluation_id: str) -> dict[str, Any] | None:
        """The task template that is currently open, or None when none is."""
        return await self._call(
            "GET", f"/api/v2/client/evaluation/currentTask/{evaluation_id}", none_on=(404,)
        )

    async def evaluation_state(self, evaluation_id: str) -> dict[str, Any] | None:
        """Task status + remaining time (`timeLeft` is null for open-ended tasks)."""
        return await self._call(
            "GET", f"/api/v2/evaluation/{evaluation_id}/state", none_on=(403, 404)
        )

    async def task_hint(self, evaluation_id: str, task_template_id: str) -> dict[str, Any] | None:
        """The task statement itself (ApiHintContent).

        Not part of `clientapi.json` — it lives in the full API (`/openapi.json`)
        but is readable by a PARTICIPANT session, and it is what the DRES viewer
        shows the teams: the TEXT query for T-KIS/QA, and the image/video hint
        for V-KIS. Returns None before the task is released (403/404).
        """
        return await self._call(
            "GET",
            f"/api/v2/evaluation/{evaluation_id}/template/task/{task_template_id}/hint",
            none_on=(403, 404),
        )

    async def server_time(self) -> int | None:
        """DRES clock (ms). Unauthenticated, so it also works as a liveness probe."""
        client = self._pool.get()
        try:
            resp = await client.get(f"{self.base_url}/api/v2/status/time")
            resp.raise_for_status()
            return int(resp.json().get("timeStamp"))
        except (httpx.HTTPError, ValueError, TypeError):
            return None

    # ---- submission -------------------------------------------------------
    async def submit(self, evaluation_id: str, answer_sets: list[dict[str, Any]]) -> dict[str, Any]:
        """POST one submission. 200 = judged, 202 = accepted, verdict pending."""
        # A submission is irreversible and a wrong one is scored against the team.
        # The suite runs in mock mode, but one leaked setting once turned that off
        # mid-run and the submit tests posted 34 wrong answers to the official
        # server. Refuse outright under pytest: no configuration mistake should be
        # able to reach this line from a test.
        if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("AIC26_ALLOW_TEST_SUBMIT"):
            raise DresError(
                "Refusing to submit from a test run. Set AIC26_ALLOW_TEST_SUBMIT=1 "
                "only against a local DRES simulator, never the official server."
            )
        body = {"answerSets": answer_sets}
        client = self._pool.get()
        url = f"{self.base_url}/api/v2/submit/{evaluation_id}"

        async def once(session: str) -> httpx.Response:
            try:
                return await client.post(
                    url,
                    params={"session": session},
                    json=body,
                    headers={"Content-Type": "application/json"},
                )
            except httpx.HTTPError as exc:
                self._last_error = f"submit failed: {exc}"
                raise DresError(self._last_error) from exc

        resp = await once(await self._session())
        if resp.status_code in (401, 403) and self.can_login:
            await self.login(force=True)
            resp = await once(self._session_id or "")

        try:
            payload = resp.json()
        except ValueError:
            payload = {"description": resp.text[:300]}
        if resp.status_code not in (200, 202):
            self._last_error = f"submit rejected ({resp.status_code}): {self._describe(resp)}"
            raise DresError(
                self._last_error, status=resp.status_code, detail=payload
            )
        self._last_error = None
        return {
            "http_status": resp.status_code,
            "accepted": bool(payload.get("status", True)),
            # ApiVerdictStatus: CORRECT | WRONG | INDETERMINATE | UNDECIDABLE
            "verdict": payload.get("submission"),
            "description": payload.get("description"),
            "raw": payload,
        }

    # ---- health -----------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "mode": "disabled", "error": None, "logged_in": False}
        try:
            await self.login()
        except DresError as exc:
            return {"ok": False, "mode": "live", "error": str(exc), "logged_in": False}
        return {
            "ok": True,
            "mode": "live",
            "error": None,
            "logged_in": True,
            "username": (self._user or {}).get("username") or self.s.dres_username,
            "base_url": self.base_url,
        }
