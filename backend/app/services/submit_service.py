"""Submission service: dedup guard, local history, optional DRES adapter.

A submit is blocked when the same `submit_keyframe_id` was already submitted for
the same task (the core "reduce wrong submits" guard). When DRES env vars are
present, submissions are forwarded to `POST {DRES_BASE_URL}/api/submissions`;
otherwise history is stored locally only.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from ..config import Settings

# Map the UI query-type labels to DRES task_type values.
DRES_TASK_TYPE = {"T-KIS": "TKIS", "V-KIS": "VKIS", "QA": "QA", "TRAKE": "TRAKE"}


class DuplicateSubmitError(Exception):
    def __init__(self, key: str, task_id: str):
        super().__init__(f"{key} already submitted for task {task_id}")
        self.submit_keyframe_id = key  # kept name for API back-compat (the duplicate key)
        self.task_id = task_id


class SubmitService:
    def __init__(self, settings: Settings, history_path: Path | None = None):
        self.s = settings
        self._lock = threading.Lock()
        self.history_path = history_path or (
            Path(__file__).resolve().parents[2] / "data" / "submit_history.json"
        )
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if self.history_path.exists():
            try:
                return json.loads(self.history_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _persist(self) -> None:
        self.history_path.write_text(
            json.dumps(self._history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def history(self, task_id: str | None = None) -> list[dict[str, Any]]:
        if task_id is None:
            return list(self._history)
        return [h for h in self._history if h.get("task_id") == task_id]

    def _dedup_keys(self, payload: dict[str, Any]) -> set[str]:
        """Keys identifying what was submitted, for duplicate detection.

        KIS/QA: one "video_id:frame_idx" key. TRAKE: one key for the whole
        ordered sequence "video_id|f1,f2,f3".
        """
        video_id = payload.get("video_id")
        events = payload.get("events") or []
        if events:
            seq = ",".join(str(e.get("frame_idx")) for e in events)
            return {f"{video_id}|{seq}"}
        if video_id is not None and payload.get("frame_idx") is not None:
            return {f"{video_id}:{payload['frame_idx']}"}
        return set()

    def is_duplicate(self, task_id: str, payload: dict[str, Any]) -> str | None:
        """Return the first key that duplicates a prior submit for this task."""
        new_keys = self._dedup_keys(payload)
        for h in self._history:
            if h.get("task_id") != task_id:
                continue
            overlap = new_keys & self._dedup_keys(h.get("payload", {}))
            if overlap:
                return sorted(overlap)[0]
        return None

    async def submit(
        self,
        *,
        task_id: str,
        query_type: str,
        payload: dict[str, Any],
        allow_duplicate: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            dup = self.is_duplicate(task_id, payload)
            if dup and not allow_duplicate:
                raise DuplicateSubmitError(dup, task_id)

        dres_result = None
        status = "local"
        if self.s.has_dres:
            try:
                dres_result = await self._submit_dres(task_id, query_type, payload)
                status = "dres_ok"
            except Exception as exc:  # noqa: BLE001 - record but keep local entry
                dres_result = {"error": str(exc)}
                status = "dres_error"

        entry = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "task_id": task_id,
            "query_type": query_type,
            "payload": payload,
            "dedup_keys": sorted(self._dedup_keys(payload)),
            "status": status,
            "dres": dres_result,
            "was_duplicate": bool(dup),
        }
        with self._lock:
            self._history.append(entry)
            self._persist()
        return entry

    def _dres_body(self, task_id: str, query_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Build the DRES /api/submissions body: video_id + frame_idx (+ events/answer)."""
        task_type = DRES_TASK_TYPE.get(query_type, query_type)
        body: dict[str, Any] = {"task_id": task_id, "task_type": task_type, "video_id": payload.get("video_id")}
        if task_type == "TRAKE":
            body["events"] = [
                {"event_index": e.get("event_index"), "frame_idx": e.get("frame_idx")}
                for e in (payload.get("events") or [])
            ]
        else:
            body["frame_idx"] = payload.get("frame_idx")
            if payload.get("timestamp") is not None:
                body["timestamp"] = payload.get("timestamp")
            if task_type == "QA":
                body["answer"] = payload.get("answer")
        return body

    async def _submit_dres(self, task_id, query_type, payload) -> dict[str, Any]:
        body = self._dres_body(task_id, query_type, payload)
        headers = {
            "Authorization": f"Bearer {self.s.dres_token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{self.s.dres_base_url.rstrip('/')}/api/submissions",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
