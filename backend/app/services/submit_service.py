"""Submission service: DRES v2 answer building, dedup guard, local history.

Two jobs, in this order:

1. **Correct format.** DRES v2 takes `{"answerSets":[{"taskName", "answers":[…]}]}`.
   An answer carries `mediaItemName` + `start`/`end` **in milliseconds** (KIS,
   one answer per event for TRAKE) and/or `text`. QA sends all four in ONE
   answer — the segment plus the answer — which is the form the BTC payload
   example documents. `build_answer_sets` picks the shape from the query type.
2. **Reduce wrong submits.** A submit is blocked when the same answer was
   already sent for the same open task (per evaluation + task name), and the
   local history keeps every attempt with the DRES verdict for the operator.

The task name is resolved from the live `currentTask` endpoint at submit time
unless the operator overrides it, so it always matches the open task exactly.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .. import paths
from ..adapters.dres_client import DresClient, DresError, DresNotConfigured
from ..config import Settings

# How a UI query type maps onto the DRES answer shape.
#   temporal      -> mediaItemName + start/end (ms)
#   text          -> text only
#   temporal_text -> both in ONE answer: the QA form BTC documents, where the
#                    answer must identify the segment AND state the answer.
ANSWER_MODE_BY_QUERY_TYPE = {
    "T-KIS": "temporal",
    "V-KIS": "temporal",
    "TRAKE": "temporal",
    "QA": "temporal_text",
}

# Tokens that identify which evaluation run belongs to which query type. Matched
# against the run name + its task groups/types ("tkis", "T-KIS Group", …).
EVALUATION_KEYWORDS = {
    "T-KIS": (r"t-?kis", r"textual"),
    "V-KIS": (r"v-?kis", r"visual"),
    "QA": (r"\bqa\b", r"question"),
    "TRAKE": (r"trake", r"multi-?event", r"sequence"),
}

_WS = re.compile(r"\s+")

# Base64 hint media above this is not worth pushing through the console.
_MAX_HINT_CONTENT = 12_000_000


class DuplicateSubmitError(Exception):
    def __init__(self, key: str, task_id: str):
        super().__init__(f"{key} already submitted for task {task_id}")
        self.submit_keyframe_id = key  # kept name for API back-compat (the duplicate key)
        self.task_id = task_id


class SubmitFormatError(ValueError):
    """The payload cannot be turned into a valid DRES answer set."""


def _clean_media_item_name(video_id: str | None) -> str:
    """DRES wants the bare video name — no path, no `.mp4`."""
    name = (video_id or "").strip().split("/")[-1]
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    return name


def _ms_from(
    *,
    timestamp: float | None,
    frame_idx: int | None,
    fps: float | None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    pad_ms: int = 0,
) -> tuple[int, int]:
    """Resolve one picked instant into the DRES `[start, end]` window in ms.

    Explicit `start_ms`/`end_ms` win (the operator can widen the window in the
    submit guard). Otherwise the instant comes from `pts_time`, or from
    `frame_idx / fps` when only a frame number is known.
    """
    if start_ms is not None or end_ms is not None:
        start = int(start_ms if start_ms is not None else end_ms)
        end = int(end_ms if end_ms is not None else start_ms)
        if end < start:
            start, end = end, start
        return max(0, start), max(0, end)

    center_ms: float | None = None
    if timestamp is not None:
        center_ms = float(timestamp) * 1000.0
    elif frame_idx is not None and fps:
        center_ms = (float(frame_idx) / float(fps)) * 1000.0
    if center_ms is None:
        raise SubmitFormatError(
            "Thiếu thời điểm để nộp: cần timestamp (pts_time) hoặc frame_idx + fps."
        )
    start = int(round(center_ms)) - int(pad_ms)
    end = int(round(center_ms)) + int(pad_ms)
    return max(0, start), max(0, end)


def resolve_answer_mode(query_type: str, requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    return ANSWER_MODE_BY_QUERY_TYPE.get(query_type, "temporal")


def build_answers(
    *,
    query_type: str,
    payload: dict[str, Any],
    answer_mode: str = "auto",
    pad_ms: int = 0,
) -> list[dict[str, Any]]:
    """Build the DRES `answers` array for one task. Raises SubmitFormatError."""
    mode = resolve_answer_mode(query_type, answer_mode)

    text = (payload.get("answer") or "").strip()
    if mode in {"text", "temporal_text"} and not text:
        raise SubmitFormatError("Task QA cần `text` (đáp án) — ô answer đang trống.")
    if mode == "text":
        return [{"text": text}]

    media_item = _clean_media_item_name(payload.get("video_id"))
    if not media_item:
        raise SubmitFormatError("Thiếu `mediaItemName` (video_id) cho task KIS/QA/TRAKE.")

    if mode == "item":
        return [{"mediaItemName": media_item}]

    if mode == "temporal_text":
        # One answer carrying the segment AND the answer text, exactly as the BTC
        # payload example shows for QA.
        start, end = _ms_from(
            timestamp=payload.get("timestamp"),
            frame_idx=payload.get("frame_idx"),
            fps=payload.get("fps"),
            start_ms=payload.get("start_ms"),
            end_ms=payload.get("end_ms"),
            pad_ms=pad_ms,
        )
        return [{"mediaItemName": media_item, "start": start, "end": end, "text": text}]

    if mode != "temporal":
        raise SubmitFormatError(f"answer_mode không hợp lệ: {mode!r}")

    events = payload.get("events") or []
    if events:
        # TRAKE: one temporal answer per event, in event order.
        ordered = sorted(events, key=lambda e: e.get("event_index") or 0)
        answers = []
        for event in ordered:
            start, end = _ms_from(
                timestamp=event.get("pts_time"),
                frame_idx=event.get("frame_idx"),
                fps=event.get("fps") or payload.get("fps"),
                start_ms=event.get("start_ms"),
                end_ms=event.get("end_ms"),
                pad_ms=pad_ms,
            )
            answers.append(
                {"mediaItemName": _clean_media_item_name(event.get("video_id") or media_item),
                 "start": start, "end": end}
            )
        return answers

    start, end = _ms_from(
        timestamp=payload.get("timestamp"),
        frame_idx=payload.get("frame_idx"),
        fps=payload.get("fps"),
        start_ms=payload.get("start_ms"),
        end_ms=payload.get("end_ms"),
        pad_ms=pad_ms,
    )
    return [{"mediaItemName": media_item, "start": start, "end": end}]


def build_answer_sets(
    *,
    task_name: str | None,
    query_type: str,
    payload: dict[str, Any],
    answer_mode: str = "auto",
    pad_ms: int = 0,
) -> list[dict[str, Any]]:
    """The full `answerSets` body sent to POST /api/v2/submit/{evaluationId}."""
    answers = build_answers(
        query_type=query_type, payload=payload, answer_mode=answer_mode, pad_ms=pad_ms
    )
    answer_set: dict[str, Any] = {"answers": answers}
    # taskName is optional for DRES (it infers the open task) but the BTC rules
    # require an exact match, so it is always sent when known.
    if task_name:
        answer_set["taskName"] = task_name
    return [answer_set]


def is_qa_task_type(task_type: str | None) -> bool:
    """True when DRES expects a TEXT answer for this task ("Question Answering")."""
    if not task_type:
        return False
    text = task_type.lower()
    return "question" in text or "answer" in text or text.strip() in {"qa", "q&a"}


def answer_shape_mismatch(task_type: str | None, mode: str) -> str | None:
    """Warn when the built answer shape contradicts the open task's type.

    DRES reads `text` first and ignores the media fields when it is present (and
    vice versa), so a mismatch is a silently-wrong submit, not an error.
    """
    if not task_type:
        return None
    if is_qa_task_type(task_type) and mode not in {"text", "temporal_text"}:
        return (
            f"Task DRES là “{task_type}” (cần `text`) nhưng payload đang gửi "
            f"“{mode}” — đáp án bạn gõ SẼ BỊ BỎ. Đổi 'answer as' về auto."
        )
    if not is_qa_task_type(task_type) and mode in {"text", "temporal_text"}:
        return (
            f"Task DRES là “{task_type}” (cần `mediaItemName` + start/end) nhưng "
            f"payload đang kèm `text` — DRES có thể chấm sai. Đổi 'answer as' về auto."
        )
    return None


def _matches_query_type(evaluation: dict[str, Any], query_type: str) -> bool:
    patterns = EVALUATION_KEYWORDS.get(query_type)
    if not patterns:
        return False
    haystack = " ".join(
        [str(evaluation.get("name") or "")]
        + [
            f"{t.get('taskGroup') or ''} {t.get('taskType') or ''} {t.get('name') or ''}"
            for t in (evaluation.get("taskTemplates") or [])
        ]
    ).lower()
    return any(re.search(p, haystack) for p in patterns)


class SubmitService:
    def __init__(
        self,
        settings: Settings,
        dres: DresClient | None = None,
        history_path: Path | None = None,
    ):
        self.s = settings
        self.dres = dres if dres is not None else DresClient(settings)
        self._lock = threading.Lock()
        self.history_path = history_path or (paths.data_dir() / "submit_history.json")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, Any]] = self._load()
        self._overview_cache: tuple[float, list[dict[str, Any]]] | None = None
        # Task statements, keyed by (evaluation_id, task_template_id).
        self._hint_cache: dict[tuple[str, str], dict[str, Any]] = {}

    # ---- history ----------------------------------------------------------
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

    def clear_history(
        self, task_id: str | None = None, ids: list[str] | None = None
    ) -> dict[str, Any]:
        """Drop local history: specific entries, one task's, or all of it.

        This also erases what the dedup guard remembers, so the removed entries
        are written to a timestamped backup next to the history file first —
        DRES keeps the authoritative record either way.
        """
        with self._lock:
            if ids is not None:
                wanted = set(ids)
                removed = [h for h in self._history if h.get("id") in wanted]
            else:
                removed = self.history(task_id)
            if not removed:
                return {"deleted": 0, "remaining": len(self._history), "backup": None}

            backup = self.history_path.with_suffix(f".{int(time.time())}.bak.json")
            backup.write_text(
                json.dumps(removed, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            dropped = {id(h) for h in removed}
            self._history = [h for h in self._history if id(h) not in dropped]
            self._persist()
            return {
                "deleted": len(removed),
                "remaining": len(self._history),
                "backup": backup.name,
                "task_id": task_id,
            }

    # ---- dedup ------------------------------------------------------------
    def _dedup_keys(
        self, payload: dict[str, Any], query_type: str = "", answer_mode: str = "auto"
    ) -> set[str]:
        """Keys identifying what was submitted, for duplicate detection.

        A QA answer is judged on the segment AND the text, so only re-sending the
        exact same pair is a wasted attempt: a new answer for the same frame, or
        the same answer pinned to another video, is a genuinely different guess.
        KIS: one "video_id:frame_idx" key. TRAKE: one key for the whole ordered
        sequence "video_id|f1,f2,f3".
        """
        mode = resolve_answer_mode(query_type, answer_mode)
        text = _WS.sub(" ", (payload.get("answer") or "").strip()).casefold()
        if mode == "text" or (not payload.get("video_id") and text):
            return {f"text:{text}"} if text else set()

        video_id = payload.get("video_id")
        events = payload.get("events") or []
        if events:
            seq = ",".join(str(e.get("frame_idx")) for e in events)
            return {f"{video_id}|{seq}"}
        if video_id is not None and payload.get("frame_idx") is not None:
            frame_key = f"{video_id}:{payload['frame_idx']}"
            return {f"{frame_key}|text:{text}"} if mode == "temporal_text" else {frame_key}
        return set()

    def is_duplicate(
        self, task_id: str, payload: dict[str, Any], query_type: str = "", answer_mode: str = "auto"
    ) -> str | None:
        """Return the first key that duplicates a prior submit for this task."""
        new_keys = self._dedup_keys(payload, query_type, answer_mode)
        for h in self._history:
            if h.get("task_id") != task_id:
                continue
            overlap = new_keys & self._dedup_keys(
                h.get("payload", {}), h.get("query_type", ""), h.get("answer_mode") or "auto"
            )
            if overlap:
                return sorted(overlap)[0]
        return None

    # ---- DRES evaluation / task resolution --------------------------------
    async def overview(self, *, force: bool = False) -> list[dict[str, Any]]:
        """Evaluations + their open task + run state, in one shot for the UI.

        Cached for 2 s: the console polls this on a timer and several open tabs
        must not multiply the load on the BTC server.
        """
        now = time.time()
        if not force and self._overview_cache and now - self._overview_cache[0] < 2.0:
            return self._overview_cache[1]

        evaluations = await self.dres.evaluations()

        async def enrich(evaluation: dict[str, Any]) -> dict[str, Any]:
            eval_id = evaluation.get("id")
            task, state = await asyncio.gather(
                self.dres.current_task(eval_id),
                self.dres.evaluation_state(eval_id),
                return_exceptions=True,
            )
            return {
                "id": eval_id,
                "name": evaluation.get("name"),
                "status": evaluation.get("status"),
                "type": evaluation.get("type"),
                "teams": evaluation.get("teams") or [],
                "task_templates": evaluation.get("taskTemplates") or [],
                "current_task": task if isinstance(task, dict) else None,
                "state": state if isinstance(state, dict) else None,
            }

        enriched = list(await asyncio.gather(*(enrich(e) for e in evaluations)))
        self._overview_cache = (now, enriched)
        return enriched

    async def resolve_evaluation(self, query_type: str, evaluation_id: str | None = None) -> str:
        """Pick the evaluation run to submit to: explicit → pinned → by type."""
        if evaluation_id:
            return evaluation_id
        if self.s.dres_evaluation_id:
            return self.s.dres_evaluation_id
        overview = await self.overview()
        active = [e for e in overview if (e.get("status") or "").upper() == "ACTIVE"] or overview
        matching = [e for e in active if _matches_query_type(e, query_type)]
        # An evaluation with an open task beats one that is merely active.
        running = [e for e in matching if e.get("current_task")]
        for pool in (running, matching, active):
            if len(pool) == 1:
                return str(pool[0]["id"])
        if not active:
            raise SubmitFormatError("DRES không có evaluation nào cho tài khoản này.")
        names = ", ".join(str(e.get("name")) for e in active)
        raise SubmitFormatError(
            f"Không tự chọn được evaluation cho {query_type} — hãy chọn thủ công (có: {names})."
        )

    async def task_hint(
        self, query_type: str = "T-KIS", evaluation_id: str | None = None, *, force: bool = False
    ) -> dict[str, Any]:
        """The statement of the open task: text for T-KIS/QA, media for V-KIS.

        Cached per task template — a hint never changes while a task is open, and
        a V-KIS image/video hint is far too big to re-fetch on every UI poll.
        """
        resolved = await self.resolve_evaluation(query_type, evaluation_id)
        state = await self.dres.evaluation_state(resolved) or {}
        template_id = state.get("taskTemplateId")
        task = await self.dres.current_task(resolved)
        base = {
            "evaluation_id": resolved,
            "task_template_id": template_id,
            "task_name": (task or {}).get("name"),
            "task_type": (task or {}).get("taskType"),
            "task_status": state.get("taskStatus"),
            "text": "",
            "elements": [],
            "warnings": [],
        }
        if not template_id:
            base["warnings"].append("Chưa có task nào được mở.")
            return base

        key = (resolved, template_id)
        if not force and key in self._hint_cache:
            return {**base, **self._hint_cache[key]}

        raw = await self.dres.task_hint(resolved, template_id)
        if raw is None:
            base["warnings"].append("DRES chưa cho xem đề bài của task này.")
            return base

        elements: list[dict[str, Any]] = []
        texts: list[str] = []
        warnings: list[str] = []
        for element in sorted(raw.get("sequence") or [], key=lambda e: e.get("offset") or 0):
            content_type = (element.get("contentType") or "EMPTY").upper()
            content = element.get("content") or ""
            if content_type == "EMPTY" or not content:
                continue
            if content_type == "TEXT":
                texts.append(content)
            elif len(content) > _MAX_HINT_CONTENT:
                # A base64 video hint can be tens of MB; keep the console usable.
                warnings.append(f"Hint {content_type.lower()} quá lớn ({len(content) // 1_000_000} MB) — xem trên DRES viewer.")
                continue
            elements.append(
                {"content_type": content_type, "content": content, "offset": element.get("offset") or 0}
            )
        hint = {
            "text": "\n\n".join(texts),
            "elements": elements,
            "loop": bool(raw.get("loop")),
            "warnings": warnings,
        }
        self._hint_cache[key] = hint
        return {**base, **hint}

    async def resolve_task_name(self, evaluation_id: str, task_name: str | None = None) -> str:
        if task_name:
            return task_name
        task = await self.dres.current_task(evaluation_id)
        if not task or not task.get("name"):
            raise SubmitFormatError(
                "Evaluation này chưa mở task nào (currentTask trống) — không thể nộp."
            )
        return str(task["name"])

    # ---- submit -----------------------------------------------------------
    async def prepare(
        self,
        *,
        query_type: str,
        payload: dict[str, Any],
        evaluation_id: str | None = None,
        task_name: str | None = None,
        task_id: str | None = None,
        answer_mode: str = "auto",
        pad_ms: int | None = None,
    ) -> dict[str, Any]:
        """Resolve target task + build the exact DRES body, without sending it.

        Powers the submit guard's preview: the operator sees the JSON that will
        hit DRES (and any duplicate warning) before committing.
        """
        pad = self.s.dres_segment_pad_ms if pad_ms is None else max(0, int(pad_ms))
        resolved_eval: str | None = None
        resolved_task: str | None = task_name
        task_type: str | None = None
        warnings: list[str] = []

        if self.s.has_dres:
            try:
                resolved_eval = await self.resolve_evaluation(query_type, evaluation_id)
                open_task = await self.dres.current_task(resolved_eval) or {}
                task_type = open_task.get("taskType")
                resolved_task = task_name or open_task.get("name")
                if not resolved_task:
                    raise SubmitFormatError(
                        "Evaluation này chưa mở task nào (currentTask trống) — không thể nộp."
                    )
            except (DresError, DresNotConfigured) as exc:
                warnings.append(f"DRES: {exc}")
        else:
            warnings.append("DRES chưa cấu hình — submit chỉ lưu local history.")

        mode = resolve_answer_mode(query_type, answer_mode)
        # The DRES task type is the authority on the answer shape, not the tab the
        # operator happens to be on. A mismatch here silently drops the answer.
        mismatch = answer_shape_mismatch(task_type, mode)
        if mismatch:
            warnings.append(mismatch)

        answer_sets = build_answer_sets(
            task_name=resolved_task,
            query_type=query_type,
            payload=payload,
            answer_mode=answer_mode,
            pad_ms=pad,
        )
        scope = self._task_scope(resolved_eval, resolved_task, task_id)
        return {
            "evaluation_id": resolved_eval,
            "task_name": resolved_task,
            "task_type": task_type,
            "task_id": scope,
            "answer_mode": mode,
            "answer_mode_mismatch": bool(mismatch),
            "segment_pad_ms": pad,
            "body": {"answerSets": answer_sets},
            "duplicate_key": self.is_duplicate(scope, payload, query_type, answer_mode),
            "url": (
                f"{self.dres.base_url}/api/v2/submit/{resolved_eval}" if resolved_eval else None
            ),
            "warnings": warnings,
        }

    @staticmethod
    def _task_scope(evaluation_id: str | None, task_name: str | None, task_id: str | None) -> str:
        """Dedup/history scope: the open DRES task, or the manual label."""
        if task_name:
            return f"{evaluation_id}/{task_name}" if evaluation_id else task_name
        return task_id or "unknown-task"

    async def submit(
        self,
        *,
        query_type: str,
        payload: dict[str, Any],
        evaluation_id: str | None = None,
        task_name: str | None = None,
        task_id: str | None = None,
        answer_mode: str = "auto",
        pad_ms: int | None = None,
        allow_duplicate: bool = False,
    ) -> dict[str, Any]:
        prepared = await self.prepare(
            query_type=query_type,
            payload=payload,
            evaluation_id=evaluation_id,
            task_name=task_name,
            task_id=task_id,
            answer_mode=answer_mode,
            pad_ms=pad_ms,
        )
        scope = prepared["task_id"]

        with self._lock:
            dup = self.is_duplicate(scope, payload, query_type, answer_mode)
            if dup and not allow_duplicate:
                raise DuplicateSubmitError(dup, scope)

        dres_result: dict[str, Any] | None = None
        status = "local"
        verdict = None
        if self.s.has_dres and prepared["evaluation_id"]:
            try:
                dres_result = await self.dres.submit(
                    prepared["evaluation_id"], prepared["body"]["answerSets"]
                )
                verdict = dres_result.get("verdict")
                status = "dres_ok"
            except DresError as exc:  # recorded, but the local entry is still kept
                dres_result = {"error": str(exc), "status": exc.status, "detail": exc.detail}
                status = "dres_error"
            except Exception as exc:  # noqa: BLE001 - never lose the history entry
                dres_result = {"error": str(exc)}
                status = "dres_error"

        entry = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "task_id": scope,
            "evaluation_id": prepared["evaluation_id"],
            "task_name": prepared["task_name"],
            "task_type": prepared["task_type"],
            "query_type": query_type,
            "payload": payload,
            "answer_sets": prepared["body"]["answerSets"],
            "answer_mode": prepared["answer_mode"],
            "dedup_keys": sorted(self._dedup_keys(payload, query_type, answer_mode)),
            "status": status,
            "verdict": verdict,
            "dres": dres_result,
            "was_duplicate": bool(dup),
            "warnings": prepared["warnings"],
        }
        with self._lock:
            self._history.append(entry)
            self._persist()
        return entry
