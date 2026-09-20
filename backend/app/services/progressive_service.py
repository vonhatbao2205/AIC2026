"""Single-process session ownership, idempotent updates and replay journals."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
import uuid
import subprocess
from functools import lru_cache
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from ..progressive.core import derive_ledger
from ..progressive.engine import EngineState, ProgressiveEngine
from ..progressive.models import HintUpdate, ProgressiveConfig


@lru_cache(maxsize=1)
def code_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, timeout=2).decode().strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"git_commit": commit, "backend_source_sha256": digest.hexdigest()}


class ProgressiveError(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message
        super().__init__(message)


@dataclass
class Session:
    id: str
    config: ProgressiveConfig
    revision: int = 0
    committed_revision: int = 0
    pending_revision: int | None = None
    updated: float = field(default_factory=time.monotonic)
    state: EngineState = field(default_factory=EngineState)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    requests: OrderedDict = field(default_factory=OrderedDict)
    last_error: str | None = None


class ProgressiveService:
    def __init__(self, searches: dict, journal_dir: Path, *, ttl_seconds: float = 3600, max_sessions: int = 128):
        self.searches, self.journal_dir = searches, journal_dir
        self.ttl_seconds, self.max_sessions = ttl_seconds, max_sessions
        self.sessions: dict[str, Session] = {}

    def _expire(self):
        now = time.monotonic()
        for key, session in list(self.sessions.items()):
            if now - session.updated > self.ttl_seconds:
                del self.sessions[key]

    def _get(self, session_id: str) -> Session:
        self._expire()
        session = self.sessions.get(session_id)
        if session is None:
            raise ProgressiveError(410, "Progressive session expired or closed; start a new session")
        session.updated = time.monotonic()
        return session

    def _journal(self, session: Session, event: str, **payload):
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        with (self.journal_dir / f"{session.id}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, "session_id": session.id,
                "timestamp_ms": round(time.time() * 1000), **payload}, ensure_ascii=False, allow_nan=False) + "\n")

    def create(self, config: ProgressiveConfig) -> dict:
        self._expire()
        if len(self.sessions) >= self.max_sessions:
            raise ProgressiveError(429, "Too many active progressive sessions; close an unused session")
        search = self.searches[config.retrieval_database]
        search.resolve_image_models(config.image_models)
        session = Session(uuid.uuid4().hex, config.model_copy(deep=True))
        registry = {key: getattr(search.s, key) for key in (
            "milvus_image_collection", "milvus_qwen3_vl_image_collection", "milvus_audio_collection",
            "idx_keyframe_map", "idx_ocr", "idx_speech", "idx_audio",
        )}
        self._journal(session, "created", config=config.model_dump(), code=code_identity(),
                      index_registry=registry, mode="mock" if search.s.mock_mode else "live")
        self.sessions[session.id] = session
        return self.snapshot(session.id)

    def snapshot(self, session_id: str) -> dict:
        session = self._get(session_id)
        return {**copy.deepcopy(session.state.snapshot), "session_id": session.id,
            "revision": session.revision, "committed_revision": session.committed_revision,
            "pending_revision": session.pending_revision, "last_error": session.last_error,
            "config": session.config.model_dump()}

    def close(self, session_id: str) -> dict:
        session = self._get(session_id)
        self._journal(session, "closed", revision=session.revision)
        del self.sessions[session_id]
        return {"session_id": session_id, "closed": True}

    async def update(self, session_id: str, update: HintUpdate) -> dict:
        session = self._get(session_id)
        digest = hashlib.sha256(update.model_dump_json().encode()).hexdigest()
        async with session.lock:
            known = session.requests.get(update.client_request_id)
            if known is not None:
                previous_digest, task = known
                if previous_digest != digest:
                    raise ProgressiveError(409, "client_request_id was reused for different input")
            else:
                if update.expected_revision != session.revision:
                    raise ProgressiveError(409, f"Revision conflict; current revision is {session.revision}")
                if sum(not item[1].done() for item in session.requests.values()) >= 4:
                    raise ProgressiveError(429, "Four revisions are still computing for this session")
                active = derive_ledger(update.hints, update.elapsed_ms)
                revision = session.revision + 1
                released_input = [h.model_dump() for h in update.hints if h.revealed_at is None or (
                    update.elapsed_ms is not None and h.revealed_at <= update.elapsed_ms
                )]
                self._journal(session, "update", revision=revision, client_request_id=update.client_request_id,
                              active_ledger=active, input_ledger=released_input, elapsed_ms=update.elapsed_ms)
                session.revision = revision
                session.pending_revision = revision
                session.last_error = None
                base = copy.deepcopy(session.state)
                task = asyncio.create_task(self._compute(session, revision, active, base))
                # Consume exceptions even when the browser disconnects. Duplicate
                # requests still await the same completed task and get its result.
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                session.requests[update.client_request_id] = (digest, task)
                for key in list(session.requests):
                    if len(session.requests) <= 64:
                        break
                    if session.requests[key][1].done():
                        del session.requests[key]
        return copy.deepcopy(await asyncio.shield(task))

    async def _compute(self, session: Session, revision: int, ledger: list[dict], base: EngineState):
        try:
            engine = ProgressiveEngine(self.searches[session.config.retrieval_database], session.config)
            state = await asyncio.wait_for(engine.run(ledger, base), timeout=180)
            async with session.lock:
                if self.sessions.get(session.id) is not session or session.revision != revision:
                    self._journal(session, "superseded", revision=revision)
                    raise ProgressiveError(409, "Result superseded by a newer revision or closed session")
                self._journal(session, "committed", revision=revision, **engine.trace_export(state))
                session.state = state
                session.committed_revision = revision
                session.pending_revision = None
                return self.snapshot(session.id)
        except Exception as exc:
            async with session.lock:
                if session.revision == revision:
                    session.pending_revision = None
                    session.last_error = type(exc).__name__
                self._journal(session, "failed", revision=revision, error_type=type(exc).__name__)
            if isinstance(exc, ProgressiveError):
                raise
            raise ProgressiveError(503, "Progressive retrieval failed; previous snapshot retained") from exc
