"""The tools a Codex / Claude agent run can call, served at /api/agent/tools.

Every tool is a thin view over services the console already uses — the same
`SearchService` (PE / Qwen / OCR / speech / audio + RRF), the same keyframe map,
the same media URLs — so an agent sees the corpus exactly as the operator does,
and it never holds an Elastic/Milvus credential of its own.

Results are shaped as MCP `tools/call` results (`content` + `isError`), which the
stdio bridge (`mcp_server.py`) passes through unchanged. Images go back inline as
JPEG: one contact sheet per call, never a folder of files on disk.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
from collections import OrderedDict, deque
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..adapters.http_pool import PooledHttpClient
from ..identity import canonical_submit_keyframe_id, group_from_video_id, parse_submit_keyframe_id, video_id_prefix
from ..scope import CATEGORY_LABELS_EN, in_scope, profile_categories
from ..traffic import load_catalog

if TYPE_CHECKING:
    from ..services.search_service import SearchService
    from .runs import AgentRun

#: Channels each search mode keeps. The parser's own routing decides the rest in
#: `hybrid`; the narrow modes force one evidence type so an agent can ask
#: "where is this TEXT on screen" without the image channels drowning it.
_MODE_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "hybrid": {"force_channels": [], "disable_channels": []},
    "visual": {"force_channels": ["image_pe"], "disable_channels": ["ocr", "speech", "audio", "tara"]},
    "ocr": {"force_channels": ["ocr"], "disable_channels": ["image_pe", "speech", "audio", "tara"]},
    "speech": {"force_channels": ["speech"], "disable_channels": ["image_pe", "ocr", "audio", "tara"]},
}
_FRAMES_PER_VIDEO = 4
_OVERVIEW_MAX = 48
_VIEW_MAX = 12
_IMAGE_CACHE_SIZE = 768
_FETCH_CONCURRENCY = 8
_LIST_PAGE_MAX = 60
#: Where `list_videos` samples a line of speech, as fractions of the video.
_GIST_POINTS = (0.15, 0.5, 0.85)
_OUTLINE_MAX_LINES = 60
_GUIDE_PATH = Path(__file__).resolve().parents[1] / "batch2_video_guide.json"

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": (
            "Search the video corpus with the competition's retrieval engine. Returns the best-matching "
            "videos, each with its top keyframes (keyframe id, time in seconds, the channels that matched, "
            "and OCR / speech evidence text). Modes: 'hybrid' = every channel fused (default); 'visual' = "
            "image embeddings only (describe what is SEEN, short concrete phrases, English works best); "
            "'ocr' = text printed on screen (use the exact Vietnamese words); 'speech' = what is said "
            "(Vietnamese words likely spoken). Run several searches with different phrasings and modes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "mode": {"type": "string", "enum": list(_MODE_OVERRIDES), "default": "hybrid"},
                "folders": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Only these dataset folders, e.g. [\"L23\", \"S01\"]; [\"ALL\"] searches every folder. "
                        "Omit to stay inside the operator's folder filter (see SCOPE)."
                    ),
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8, "description": "Videos to return."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_videos",
        "description": (
            "Browse the programme guide of one folder (e.g. 'L26', 'N033', 'S01') or a whole series "
            "('L', 'K', 'M', 'N', 'S'): every video with its length and what it is about — traffic "
            "cameras: junction, date and clock range; S01: the race stage; other folders: three sample "
            "lines of what is said (at 15%, 50%, 85% of the video). Page with offset."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string", "description": "Folder code or series letter."},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": _LIST_PAGE_MAX, "default": 40},
            },
            "required": ["folder"],
            "additionalProperties": False,
        },
    },
    {
        "name": "folder_frames",
        "description": (
            "Look across many videos at once: one representative keyframe per video as a labelled "
            "grid. A folder ('L26') shows each of its videos; a series letter ('N') shows the first "
            "video of each folder — one tile per traffic camera. `position` picks where in each video "
            "the frame is taken (0 = start, 0.5 = middle). Page with offset."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string"},
                "position": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "count": {"type": "integer", "minimum": 4, "maximum": _OVERVIEW_MAX, "default": 48},
            },
            "required": ["folder"],
            "additionalProperties": False,
        },
    },
    {
        "name": "video_outline",
        "description": (
            "Table of contents of one video: for each step of time, the first thing said and the "
            "on-screen text. Read it to find which part of a video holds a story or scene before "
            "opening frames. Long videos get a coarser step automatically (at most 60 lines)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "start": {"type": "number", "minimum": 0, "description": "Seconds; default 0."},
                "end": {"type": "number", "minimum": 0, "description": "Seconds; default end of video."},
                "step": {"type": "number", "minimum": 10, "default": 60, "description": "Seconds per line."},
            },
            "required": ["video_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "video_frames",
        "description": (
            "Contact sheet of one video: keyframes sampled evenly between start and end seconds, drawn as "
            "one labelled image grid (tile label = #index and time). Skim a whole video first, then call "
            "again with a narrow window around the interesting tiles to zoom in."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string", "description": "e.g. L21_V001, N033-V002, S01-V004"},
                "start": {"type": "number", "minimum": 0, "description": "Seconds; default 0."},
                "end": {"type": "number", "minimum": 0, "description": "Seconds; default end of video."},
                "count": {"type": "integer", "minimum": 4, "maximum": _OVERVIEW_MAX, "default": 24},
            },
            "required": ["video_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "view_frames",
        "description": (
            "Large view of specific keyframes (up to 12) to check details: colours, text, counts, people. "
            "Pass keyframe ids exactly as other tools print them, e.g. L21/L21_V001/045."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keyframe_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": _VIEW_MAX},
            },
            "required": ["keyframe_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "video_text",
        "description": "Speech transcript and on-screen OCR text of one video between start and end seconds.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "start": {"type": "number", "minimum": 0},
                "end": {"type": "number", "minimum": 0},
            },
            "required": ["video_id", "start", "end"],
            "additionalProperties": False,
        },
    },
    {
        "name": "report_candidate",
        "description": (
            "Show a candidate answer to the operator NOW (they see it while you keep working). Report as soon "
            "as a frame plausibly matches, then report the same keyframe again with a higher confidence once "
            "verified — a repeat updates it. Identify the frame by keyframe_id, or by video_id + time."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keyframe_id": {"type": "string"},
                "video_id": {"type": "string"},
                "time": {"type": "number", "minimum": 0, "description": "Seconds into the video."},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string", "description": "One or two sentences: which constraints the frame satisfies."},
                "answer": {"type": "string", "description": "QA tasks only: the answer text."},
                "event": {"type": "integer", "minimum": 1, "description": "TRAKE tasks only: which event (1-based) this frame is."},
            },
            "required": ["confidence", "reason"],
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    """A problem the agent caused and can fix (bad id, empty window): shown to it as text."""


def _text(value: str) -> dict[str, str]:
    return {"type": "text", "text": value}


def _fmt_time(seconds: float | None) -> str:
    return "?" if seconds is None else f"{float(seconds):.1f}s"


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt_clock(seconds: float | None) -> str:
    """12:05 / 1:02:05 — how a video's length and positions read in a guide."""
    if seconds is None:
        return "?"
    total = int(round(float(seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class _NearDuplicates:
    """Drops on-screen text that mostly repeats a recent line.

    A caption held over several keyframes, or a news ticker that scrolls a few
    letters per keyframe, repeats nearly every word: without this a 50 s window
    is 120 copies of one ticker line.
    """

    def __init__(self, threshold: float = 0.6, memory: int = 4):
        self.threshold = threshold
        self.recent: deque[set[str]] = deque(maxlen=memory)

    def is_new(self, text: str) -> bool:
        words = set(text.casefold().split())
        if not words or any(len(words & seen) / len(words | seen) >= self.threshold for seen in self.recent):
            return False
        self.recent.append(words)
        return True


def _screen_text(record: dict[str, Any], limit: int) -> str:
    parts = [record.get("text_clean"), record.get("text_banner"), record.get("text_hud"), record.get("text_ticker")]
    return _clip(" / ".join(p for p in parts if p), limit)


@lru_cache(maxsize=1)
def _batch2_guide() -> dict[str, str]:
    """What each S01 recording shows, from the batch-2 content sheet
    (`build_batch2_video_guide.py`). A missing file only drops the summaries."""
    try:
        return json.loads(_GUIDE_PATH.read_text(encoding="utf-8")).get("videos") or {}
    except (OSError, ValueError, AttributeError):
        return {}


def _video_note(video_id: str) -> str | None:
    """What the catalogues already know about a camera or race video, or None."""
    if video_id.startswith("S01-"):
        number = video_id.rsplit("V", 1)[-1]
        if not number.isdigit():
            return None
        summary = _batch2_guide().get(video_id)
        return f"race stage {int(number)}" + (f" — {summary}" if summary else "")
    if video_id.startswith("N"):
        try:
            catalog = load_catalog()
        except (OSError, ValueError, KeyError):
            return None
        video = catalog.videos.get(video_id)
        if video is None:
            return None
        camera = catalog.cameras.get(video.camera)
        return f"camera {camera.label if camera else video.camera}; {', '.join(video.dates)} {video.clock[0]}–{video.clock[1]}"
    return None


class AgentTools:
    def __init__(self, services: Callable[[], dict[str, "SearchService"]]):
        # Resolved per call: a config import rebuilds every service in place.
        self._services = services
        self._http = PooledHttpClient(timeout=20.0, follow_redirects=True)
        self._images: OrderedDict[str, bytes | None] = OrderedDict()
        self._fetch_gates: dict[int, asyncio.Semaphore] = {}
        # (profile, video-id prefix) -> the videos under it. The corpus does not
        # change while the backend runs, and a series listing is one aggregation
        # over millions of keyframe documents, so it is computed once.
        self._video_lists: dict[tuple[str, str], list[dict[str, Any]]] = {}

    @staticmethod
    def specs() -> list[dict[str, Any]]:
        return TOOL_SPECS

    async def call(self, name: str, arguments: dict[str, Any], run: "AgentRun", agent: str) -> dict[str, Any]:
        handler = {
            "search": self._search,
            "list_videos": self._list_videos,
            "folder_frames": self._folder_frames,
            "video_outline": self._video_outline,
            "video_frames": self._video_frames,
            "view_frames": self._view_frames,
            "video_text": self._video_text,
            "report_candidate": self._report_candidate,
        }.get(name)
        if handler is None:
            return {"content": [_text(f"Unknown tool {name!r}.")], "isError": True}
        try:
            content = await handler(arguments or {}, run, agent)
        except ToolError as exc:
            return {"content": [_text(str(exc))], "isError": True}
        except Exception as exc:  # noqa: BLE001 - an upstream outage is reported, not raised
            return {"content": [_text(f"{name} failed: {type(exc).__name__}: {_clip(exc, 300)}")], "isError": True}
        return {"content": content, "isError": False}

    def _service(self, run: "AgentRun") -> "SearchService":
        return self._services()[run.request.retrieval_database]

    # ---- search ---------------------------------------------------------
    async def _search(self, args: dict[str, Any], run: "AgentRun", agent: str) -> list[dict]:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ToolError("search needs a non-empty query.")
        mode = str(args.get("mode") or "hybrid")
        if mode not in _MODE_OVERRIDES:
            raise ToolError(f"mode must be one of {', '.join(_MODE_OVERRIDES)}.")
        limit = min(20, max(1, int(args.get("limit") or 8)))
        folders = [str(f).strip().upper() for f in (args.get("folders") or []) if str(f).strip()]
        in_filter = not folders and run.scope.active
        if any(folder in {"ALL", "*"} for folder in folders):
            scope: dict[str, Any] = {"mode": "all"}
        elif folders:
            scope = {"mode": "manual", "categories": folders}
        elif in_filter:
            # The folders the operator's query resolved to, not the request's
            # `auto`: re-running the topic heuristic on the agent's own search
            # text (often English) would quietly drop the filter.
            scope = {"mode": "manual", "categories": list(run.scope.categories)}
        else:
            scope = {"mode": "all"}
        service = self._service(run)
        result = await service.search({
            "query": query,
            "query_type_hint": "T-KIS",
            "image_models": list(run.request.image_models),
            "scope": scope,
            "traffic": "auto",
            "previous_hints": [],
            "manual_overrides": _MODE_OVERRIDES[mode],
            "use_llm": False,
            "expand": False,
            # Only the image channels read the English translation; an OCR or
            # speech search would spend a translator call it then ignores.
            "translate": mode in {"hybrid", "visual"},
            "rerank": False,
            "top_k": 150,
            "max_videos": limit,
        })
        groups = result.get("groups") or []
        lines = [f"search mode={mode} query={query!r}: {len(groups)} videos"]
        applied = (result.get("scope") or {}).get("categories") or []
        if applied:
            lines.append(
                f"scope: {', '.join(applied[:20])}{' …' if len(applied) > 20 else ''}"
                + (" (the operator's filter; folders=[\"ALL\"] searches everything)" if in_filter else "")
            )
        for rank, group in enumerate(groups, start=1):
            lines.append(
                f"{rank}. {group['video_id']}  score={group['video_score']:.4f}  channels={','.join(group.get('channels') or [])}"
            )
            for frame in group["frames"][:_FRAMES_PER_VIDEO]:
                evidence = []
                for item in frame.get("evidence") or []:
                    if item.get("text") and item.get("type") in {"ocr", "speech", "audio"}:
                        evidence.append(f"{item['type']}: \"{_clip(item['text'], 110)}\"")
                overlay = frame.get("overlay") or {}
                if overlay:
                    evidence.append("overlay: " + _clip(", ".join(f"{k}={v}" for k, v in overlay.items()), 110))
                lines.append(
                    f"   - {frame['submit_keyframe_id']}  t={_fmt_time(frame.get('pts_time'))}"
                    f"  [{','.join(frame.get('channels') or [])}]"
                    + (("  " + " | ".join(evidence[:2])) if evidence else "")
                )
        for warning in (result.get("warnings") or [])[:4]:
            lines.append(f"warning: {_clip(warning, 200)}")
        if not groups:
            lines.append("No results. Try other words, another mode, or drop the folder filter.")
        return [_text("\n".join(lines))]

    # ---- browsing the corpus -------------------------------------------
    def _resolve_folder(self, run: "AgentRun", raw: Any) -> tuple[str, str, bool]:
        """(folder, video-id prefix, is a whole series) for a folder code or series letter."""
        folder = str(raw or "").strip().upper()
        categories = profile_categories(run.request.retrieval_database)
        if folder in categories:
            return folder, video_id_prefix(folder), False
        series = sorted({category[0] for category in categories})
        if folder in series:
            return folder, folder, True
        raise ToolError(
            f"Unknown folder {folder!r} in {run.request.retrieval_database}. Use a folder code such as "
            f"{', '.join(categories[:3])} … or a series letter: {', '.join(series)}."
        )

    async def _videos(self, service: "SearchService", prefix: str) -> list[dict[str, Any]]:
        key = (service.s.retrieval_database, prefix)
        videos = self._video_lists.get(key)
        if videos is None:
            videos = await service.elastic.list_videos(prefix)
            if videos:  # an empty answer may be an outage; ask again next time
                self._video_lists[key] = videos
        return videos

    @staticmethod
    def _page(args: dict[str, Any], total: int, *, default: int, maximum: int, key: str) -> tuple[int, int]:
        offset = max(0, int(args.get("offset") or 0))
        size = min(maximum, max(1, int(args.get(key) or default)))
        if total and offset >= total:
            raise ToolError(f"offset {offset} is past the end ({total} items).")
        return offset, size

    async def _list_videos(self, args: dict[str, Any], run: "AgentRun", agent: str) -> list[dict]:
        service = self._service(run)
        folder, prefix, _ = self._resolve_folder(run, args.get("folder"))
        videos = await self._videos(service, prefix)
        if not videos:
            raise ToolError(f"No videos found in {folder}.")
        offset, limit = self._page(args, len(videos), default=40, maximum=_LIST_PAGE_MAX, key="limit")
        page = videos[offset:offset + limit]
        notes = {video["video_id"]: _video_note(video["video_id"]) for video in page}
        # Everything the catalogue cannot describe gets three sample lines of
        # speech instead: enough to tell a cooking episode's dish or a
        # documentary's place without opening the video.
        unnoted = [video for video in page if notes[video["video_id"]] is None]
        windows = [
            (video["video_id"], fraction * video["duration"], video["duration"] + 1.0)
            for video in unnoted
            for fraction in _GIST_POINTS
        ]
        gists: dict[str, str] = {}
        if windows:
            try:
                samples = await service.elastic.first_texts_in_windows("speech", windows)
            except Exception as exc:  # noqa: BLE001 - the guide still lists the videos
                samples = []
                gists = {video["video_id"]: f"(speech unavailable: {_clip(exc, 60)})" for video in unnoted}
            for index, video in enumerate(unnoted):
                rows = samples[index * len(_GIST_POINTS):(index + 1) * len(_GIST_POINTS)]
                lines = [f"\"{_clip(found[0].get('text'), 85)}\"" for found in rows if found]
                gists.setdefault(video["video_id"], " / ".join(dict.fromkeys(lines)) or "(no speech)")
        label = CATEGORY_LABELS_EN.get(folder)
        out = [
            f"{folder}{f' — {label}' if label else ''}: {len(videos)} videos; "
            f"showing {offset + 1}-{offset + len(page)} (id, length, content)"
        ]
        for video in page:
            video_id = video["video_id"]
            out.append(f"{video_id}  {_fmt_clock(video['duration'])}  {notes[video_id] or gists.get(video_id, '')}")
        remaining = len(videos) - offset - len(page)
        if remaining > 0:
            out.append(f"… {remaining} more: call again with offset={offset + len(page)}")
        return [_text("\n".join(out))]

    async def _folder_frames(self, args: dict[str, Any], run: "AgentRun", agent: str) -> list[dict]:
        service = self._service(run)
        folder, prefix, series = self._resolve_folder(run, args.get("folder"))
        videos = await self._videos(service, prefix)
        if series:
            # One tile per folder: for N that is one per traffic camera.
            first: dict[str, dict[str, Any]] = {}
            for video in videos:
                first.setdefault(group_from_video_id(video["video_id"]), video)
            videos = list(first.values())
        if not videos:
            raise ToolError(f"No videos found in {folder}.")
        offset, count = self._page(args, len(videos), default=_OVERVIEW_MAX, maximum=_OVERVIEW_MAX, key="count")
        position = min(1.0, max(0.0, float(args.get("position") if args.get("position") is not None else 0.5)))
        page = videos[offset:offset + count]
        records = await service.elastic.nearest_keyframes_by_time(
            [(video["video_id"], position * video["duration"]) for video in page]
        )
        frames: list[dict[str, Any]] = []
        labels: list[str] = []
        listing: list[str] = []
        for video, record in zip(page, records):
            if record is None:
                continue
            number = offset + len(frames) + 1
            note = _video_note(video["video_id"])
            frames.append(record)
            labels.append(f"#{number} {video['video_id']}")
            listing.append(
                f"#{number} {video['video_id']} ({_fmt_clock(video['duration'])}) {record['submit_keyframe_id']} "
                f"t={_fmt_time(record.get('pts_time'))}" + (f"  {note}" if note else "")
            )
        if not frames:
            raise ToolError(f"No keyframes found for the videos of {folder}.")
        what = "the first video of each folder" if series else "each video"
        header = (
            f"{folder}: one keyframe from {what}, taken at {position:.0%} of its length; "
            f"tiles {offset + 1}-{offset + len(page)} of {len(videos)}"
        )
        if offset + len(page) < len(videos):
            header += f" (next page: offset={offset + len(page)})"
        return await self._sheet(service, frames, header, tile=(256, 144), columns=6, labels=labels, listing=listing)

    async def _video_outline(self, args: dict[str, Any], run: "AgentRun", agent: str) -> list[dict]:
        video_id = str(args.get("video_id") or "").strip()
        if not video_id:
            raise ToolError("video_outline needs a video_id.")
        service = self._service(run)
        duration = (await service.elastic.video_durations([video_id])).get(video_id)
        if duration is None:
            raise ToolError(f"Unknown video {video_id!r} in {run.request.retrieval_database}; copy ids from other tools.")
        start = min(max(0.0, float(args.get("start") or 0.0)), duration)
        end_raw = args.get("end")
        end = duration if end_raw is None else min(max(start, float(end_raw)), duration)
        step = max(10.0, float(args.get("step") or 60.0), (end - start) / _OUTLINE_MAX_LINES)
        windows: list[tuple[str, float, float]] = []
        t = start
        while t < end or not windows:
            windows.append((video_id, t, t + step))
            t += step
        speech, ocr = await asyncio.gather(
            service.elastic.first_texts_in_windows("speech", windows),
            # Three per window: the first text of a window is often the ticker
            # or a logo; the longest of three is usually the headline.
            service.elastic.first_texts_in_windows("ocr", windows, per_window=3),
            return_exceptions=True,
        )
        note = _video_note(video_id)
        out = [
            f"{video_id}: length {_fmt_clock(duration)}{f'; {note}' if note else ''}. Outline "
            f"{_fmt_clock(start)}–{_fmt_clock(end)}, one line per {step:.0f}s (said | on screen):"
        ]
        if isinstance(speech, BaseException):
            out.append(f"speech unavailable: {_clip(speech, 120)}")
            speech = [[] for _ in windows]
        if isinstance(ocr, BaseException):
            out.append(f"ocr unavailable: {_clip(ocr, 120)}")
            ocr = [[] for _ in windows]
        fresh = _NearDuplicates()
        for (_, window_start, _), said_rows, screen_rows in zip(windows, speech, ocr):
            said = _clip(said_rows[0].get("text"), 120) if said_rows else ""
            screen = max((_screen_text(row, 100) for row in screen_rows), key=len, default="")
            if screen and not fresh.is_new(screen):
                screen = ""
            if said or screen:
                out.append(f"[{_fmt_clock(window_start)}] {said or '—'} | {screen or '—'}")
        if len(out) == 1:
            out.append("(no speech or on-screen text indexed for this video; use video_frames to look at it)")
        return [_text("\n".join(out))]

    # ---- contact sheets -----------------------------------------------
    async def _video_frames(self, args: dict[str, Any], run: "AgentRun", agent: str) -> list[dict]:
        video_id = str(args.get("video_id") or "").strip()
        if not video_id:
            raise ToolError("video_frames needs a video_id.")
        service = self._service(run)
        duration = (await service.elastic.video_durations([video_id])).get(video_id)
        if duration is None:
            raise ToolError(f"Unknown video {video_id!r} in {run.request.retrieval_database}; copy ids from search results.")
        start = min(max(0.0, float(args.get("start") or 0.0)), duration)
        end_raw = args.get("end")
        end = duration if end_raw is None else min(max(start, float(end_raw)), duration)
        count = min(_OVERVIEW_MAX, max(4, int(args.get("count") or 24)))
        if end - start < 1e-6:
            times = [start]
        else:
            times = [start + (end - start) * i / (count - 1) for i in range(count)]
        records = await service.elastic.nearest_keyframes_by_time([(video_id, t) for t in times])
        seen: set[str] = set()
        frames: list[dict[str, Any]] = []
        for record in records:
            if not record or record["submit_keyframe_id"] in seen:
                continue
            seen.add(record["submit_keyframe_id"])
            frames.append(record)
        frames.sort(key=lambda r: float(r.get("pts_time") or 0.0))
        if not frames:
            raise ToolError(f"No keyframes for {video_id} between {start:.1f}s and {end:.1f}s.")
        header = (
            f"{video_id}: duration {duration:.1f}s; {len(frames)} keyframes between {start:.1f}s and {end:.1f}s "
            "(tile label = #index time):"
        )
        return await self._sheet(service, frames, header, tile=(256, 144), columns=6)

    async def _view_frames(self, args: dict[str, Any], run: "AgentRun", agent: str) -> list[dict]:
        raw_ids = [str(i) for i in (args.get("keyframe_ids") or [])][:_VIEW_MAX]
        ids: list[str] = []
        for raw in raw_ids:
            try:
                ids.append(canonical_submit_keyframe_id(raw))
            except ValueError as exc:
                raise ToolError(f"Bad keyframe id {raw!r}: {exc}") from exc
        if not ids:
            raise ToolError("view_frames needs keyframe_ids.")
        service = self._service(run)
        records = await service.elastic.get_keyframes_by_ids(ids)
        frames = []
        for kf_id in ids:
            parsed = parse_submit_keyframe_id(kf_id)
            record = records.get(kf_id) or {}
            frames.append({
                "submit_keyframe_id": kf_id,
                "video_id": parsed.video_id,
                "keyframe_n": parsed.keyframe_n,
                "pts_time": record.get("pts_time"),
            })
        columns = 2 if len(frames) <= 4 else 3
        return await self._sheet(service, frames, f"{len(frames)} keyframes:", tile=(512, 288), columns=columns)

    async def _sheet(
        self,
        service: "SearchService",
        frames: list[dict[str, Any]],
        header: str,
        *,
        tile: tuple[int, int],
        columns: int,
        labels: list[str] | None = None,
        listing: list[str] | None = None,
    ) -> list[dict]:
        urls = [service.media.keyframe_url(f["video_id"], int(f["keyframe_n"])) for f in frames]
        fallback = (service.s.keyframe_media_fallback_base_url or "").rstrip("/")
        primary = service.media.keyframe_base_url
        images = await asyncio.gather(*(
            self._fetch(url, fallback + url[len(primary):] if fallback and url.startswith(primary) else None)
            for url in urls
        ))
        labels = labels or [f"#{i} {_fmt_time(f.get('pts_time'))}" for i, f in enumerate(frames, start=1)]
        listing = [
            line + ("" if image else " (image unavailable)")
            for line, image in zip(
                listing or [
                    f"#{i} {f['submit_keyframe_id']} t={_fmt_time(f.get('pts_time'))}"
                    for i, f in enumerate(frames, start=1)
                ],
                images,
            )
        ]
        jpeg = await asyncio.to_thread(_render_sheet, list(images), labels, tile, columns)
        return [
            _text(header + "\n" + "\n".join(listing)),
            {"type": "image", "data": base64.b64encode(jpeg).decode("ascii"), "mimeType": "image/jpeg"},
        ]

    async def _fetch(self, url: str, fallback: str | None) -> bytes | None:
        """One keyframe's bytes, shared by both agents through a small LRU.

        Codex and Claude often inspect the same candidate video; without the
        cache the second agent would download every keyframe again.
        """
        if url in self._images:
            self._images.move_to_end(url)
            return self._images[url]
        data: bytes | None = None
        loop_id = id(asyncio.get_running_loop())
        gate = self._fetch_gates.get(loop_id)
        if gate is None:
            gate = self._fetch_gates[loop_id] = asyncio.Semaphore(_FETCH_CONCURRENCY)
        async with gate:
            for candidate in filter(None, (url, fallback)):
                try:
                    response = await self._http.get().get(candidate)
                    if response.status_code == 200 and response.content:
                        data = response.content
                        break
                except Exception:  # noqa: BLE001 - a missing tile is drawn as such
                    continue
        # Failures are not cached: an origin that blipped should get a second try.
        if data is not None:
            self._images[url] = data
            while len(self._images) > _IMAGE_CACHE_SIZE:
                self._images.popitem(last=False)
        return data

    # ---- text ----------------------------------------------------------
    async def _video_text(self, args: dict[str, Any], run: "AgentRun", agent: str) -> list[dict]:
        video_id = str(args.get("video_id") or "").strip()
        start = max(0.0, float(args.get("start") or 0.0))
        end = max(start, float(args.get("end") if args.get("end") is not None else start + 60.0))
        if not video_id:
            raise ToolError("video_text needs a video_id.")
        service = self._service(run)
        speech, ocr = await asyncio.gather(
            service.elastic.get_video_speech_window(video_id, start, end, size=80),
            service.elastic.get_video_ocr_window(video_id, start, end, size=120),
            return_exceptions=True,
        )
        lines = [f"{video_id} text between {start:.1f}s and {end:.1f}s"]
        if isinstance(speech, BaseException):
            lines.append(f"speech unavailable: {_clip(speech, 160)}")
        else:
            lines.append(f"SPEECH ({len(speech)} segments):")
            lines.extend(
                f"  [{_fmt_time(seg.get('start'))}-{_fmt_time(seg.get('end'))}] {_clip(seg.get('text'), 300)}"
                for seg in speech
            )
        if isinstance(ocr, BaseException):
            lines.append(f"ocr unavailable: {_clip(ocr, 160)}")
        else:
            kept: list[str] = []
            fresh = _NearDuplicates()
            for rec in ocr:
                text = _screen_text(rec, 220)
                if fresh.is_new(text):
                    kept.append(f"  {rec.get('submit_keyframe_id')} t={_fmt_time(rec.get('pts_time'))}: {text}")
            lines.append(f"ON-SCREEN TEXT ({len(kept)} distinct of {len(ocr)} keyframes with text):")
            lines.extend(kept)
        body = "\n".join(lines)
        return [_text(body[:12000] + ("\n…(truncated; narrow the window)" if len(body) > 12000 else ""))]

    # ---- candidates ----------------------------------------------------
    async def _report_candidate(self, args: dict[str, Any], run: "AgentRun", agent: str) -> list[dict]:
        service = self._service(run)
        requested_time = args.get("time")
        record: dict[str, Any] | None = None
        if args.get("keyframe_id"):
            try:
                kf_id = canonical_submit_keyframe_id(str(args["keyframe_id"]))
            except ValueError as exc:
                raise ToolError(f"Bad keyframe_id: {exc}") from exc
            parsed = parse_submit_keyframe_id(kf_id)
            record = await service.elastic.get_keyframe(kf_id) or {
                "submit_keyframe_id": kf_id, "video_id": parsed.video_id, "keyframe_n": parsed.keyframe_n,
            }
        elif args.get("video_id") and requested_time is not None:
            video_id = str(args["video_id"]).strip()
            record = (await service.elastic.nearest_keyframes_by_time([(video_id, float(requested_time))]))[0]
            if record is None:
                raise ToolError(f"No keyframe near {requested_time}s in {video_id}.")
        else:
            raise ToolError("report_candidate needs keyframe_id, or video_id + time.")
        video_id = str(record.get("video_id") or parse_submit_keyframe_id(record["submit_keyframe_id"]).video_id)
        keyframe_n = int(record.get("keyframe_n") or parse_submit_keyframe_id(record["submit_keyframe_id"]).keyframe_n)
        event = args.get("event")
        folder = group_from_video_id(video_id)
        outside = run.scope.active and not in_scope(video_id, run.scope.categories)
        candidate = {
            "agent": agent,
            "video_id": video_id,
            "submit_keyframe_id": record["submit_keyframe_id"],
            "keyframe_n": keyframe_n,
            "frame_idx": record.get("frame_idx"),
            "fps": record.get("fps"),
            "pts_time": record.get("pts_time"),
            "time": float(requested_time) if requested_time is not None else record.get("pts_time"),
            "keyframe_url": service.media.keyframe_url(video_id, keyframe_n),
            "video_url": service.media.video_url(video_id),
            "confidence": round(min(1.0, max(0.0, float(args.get("confidence") or 0.0))), 3),
            "reason": _clip(args.get("reason"), 600),
            "answer": _clip(args.get("answer"), 200) or None,
            "event": int(event) if event is not None else None,
            "outside_scope": outside,
        }
        stored, updated = run.add_candidate(candidate)
        verb = "Updated" if updated else "Recorded"
        note = ""
        if outside:
            note = (
                f" It is OUTSIDE the operator's folder filter ({folder} is not in "
                f"{', '.join(run.scope.categories[:8])}{' …' if len(run.scope.categories) > 8 else ''}); "
                "say so, and why, in its reason and in your final answer."
            )
        return [_text(
            f"{verb} candidate {stored['submit_keyframe_id']} at {_fmt_time(stored['time'])} "
            f"(confidence {stored['confidence']:.2f}). The operator can see it now.{note} Keep verifying; "
            "report it again with a higher confidence once every constraint checks out, then finish."
        )]


def _render_sheet(
    images: list[bytes | None],
    labels: list[str],
    tile: tuple[int, int],
    columns: int,
) -> bytes:
    """Draw keyframes as one labelled grid and return it as JPEG bytes."""
    from PIL import Image, ImageDraw, ImageFont

    tile_w, tile_h = tile
    label_h = 22
    columns = max(1, min(columns, len(images)))
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # Pillow < 10.1 has one fixed-size default font
        font = ImageFont.load_default()
    for index, (data, label) in enumerate(zip(images, labels)):
        x = (index % columns) * tile_w
        y = (index // columns) * (tile_h + label_h)
        frame = None
        if data:
            try:
                frame = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:  # noqa: BLE001 - a corrupt keyframe is drawn as missing
                frame = None
        if frame is not None:
            frame.thumbnail((tile_w, tile_h))
            sheet.paste(frame, (x + (tile_w - frame.width) // 2, y + label_h + (tile_h - frame.height) // 2))
        else:
            draw.rectangle([x + 2, y + label_h + 2, x + tile_w - 3, y + label_h + tile_h - 3], outline=(90, 90, 90))
            draw.text((x + 10, y + label_h + tile_h // 2 - 8), "unavailable", fill=(150, 150, 150), font=font)
        draw.rectangle([x, y, x + tile_w - 1, y + label_h - 1], fill=(0, 0, 0))
        draw.text((x + 6, y + 2), label, fill=(255, 220, 0), font=font)
    out = io.BytesIO()
    sheet.save(out, "JPEG", quality=82)
    return out.getvalue()
