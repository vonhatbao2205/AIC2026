"""Traffic-camera and race-stage cues in a query -> the frames that can answer it.

Every N001–N100 camera burns a banner into its frames: the junction, the date and
the clock. Batch-2 OCR reads it (`banner_camera`, `banner_date`, `clock`), and
`build_traffic_camera_catalog.py` turns those readings into `traffic_cameras.json`:
the junction, date and per-minute keyframe windows of each of the 298 camera
videos. This module reads a query for the same things, plus an S01 race stage, and
resolves them into a `FrameFilter`:

* streets -> cameras. Naming both streets of a junction lands on that junction;
  naming one keeps every camera on that street. Streets are matched folded and on
  word boundaries, with the short forms banners and people use (NKKN, CMT8, ...).
* a date -> the videos recorded that day.
* a time of day -> the keyframes filmed then, per video, from the banner clock.
* "chặng 6" -> S01-V006: each S01 video is one stage (OCR handoff §7.5). Its
  per-frame `race_stage` misreads 6/8/9 as 0, so the video is the reliable key.

The filter narrows only the traffic cameras and the S01 race, never another
folder: "Hai Bà Trưng" is a camera's street and a news subject alike. A cue that
would leave its family empty is dropped with a warning rather than emptying the
search: a date no camera recorded is far more likely a misparse ("đường 3/2" is a
street) than a request for nothing.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .scope import FrameFilter
from .text_normalization import fold_vietnamese

CATALOG_PATH = Path(__file__).with_name("traffic_cameras.json")
#: Every traffic-camera video id starts with N, and no other folder's does.
CAMERA_FAMILY = "N"
RACE_FAMILY = "S01-"
RACE_STAGES = 12

#: Other ways a query writes a street the catalogue names (keys are folded words).
STREET_ALIASES: dict[str, tuple[str, ...]] = {
    "cach mang thang 8": ("cach mang thang tam", "cmt8", "cmt 8"),
    "nam ky khoi nghia": ("nkkn",),
    "nguyen thi minh khai": ("ntmk",),
    "dien bien phu": ("dbp",),
    "nguyen binh khiem": ("nbk",),
    "nga sau": ("nga 6",),
    "le van sy": ("le van si",),
    "cong truong dan chu": ("vong xoay dan chu",),
    "cong truong me linh": ("quang truong me linh", "me linh"),
    "duong so 41": ("duong 41",),
    "phu dong thien vuong": ("phu dong",),
}

# ---- query text -------------------------------------------------------
# Folding merges words that matter here: "chặng sáu" (stage 6) and "chặng sau"
# (the next stage), "7 giờ tối" (7 pm) and "7 giờ tôi" (7, I...), "tháng" (month)
# and "thắng" (won). A query typed with diacritics is therefore matched against
# the accented patterns; only one typed without any is matched folded, because
# then there is nothing to be accent-sensitive about (same rule as
# `scope.match_topics`).


class _Pattern:
    """One regex in an accented form and the folded form of the same source."""

    def __init__(self, source: str):
        self.accented = re.compile(source)
        self.folded = re.compile(fold_vietnamese(source))

    def pick(self, text: "_Query") -> re.Pattern[str]:
        return self.folded if text.unaccented else self.accented


@dataclass(frozen=True)
class _Query:
    text: str  # NFC, lower-case; folded when the query carries no diacritics
    unaccented: bool

    @classmethod
    def of(cls, raw: str) -> "_Query":
        lowered = unicodedata.normalize("NFC", raw or "").lower()
        unaccented = fold_vietnamese(lowered) == lowered
        return cls(fold_vietnamese(lowered) if unaccented else lowered, unaccented)


# ---- time of day ------------------------------------------------------
_APPROX = _Pattern(r"(?:khoảng|tầm|gần|chừng|xấp xỉ|around|about|approximately|~)\s*$")
_CLOCK = _Pattern(r"(?<![\d:])([01]?\d|2[0-3])\s*:\s*([0-5]\d)(?:\s*:\s*([0-5]\d))?(?![\d:])")
_HOUR_MINUTE = _Pattern(
    r"(?<![\d:])([01]?\d|2[0-3])\s*(?:h|g|giờ|gio)\s*([0-5]\d|\d(?=\s*(?:p|phút|phut)(?!\w)))"
    r"(?:\s*(?:p|phút|phut)(?!\w))?"
)
_HOUR = _Pattern(r"(?<![\d:])([01]?\d|2[0-3])\s*(?:h|g|giờ|gio)(?!\w)")
_AM_PM = _Pattern(r"(?<![\d:])(1[0-2]|0?[1-9])(?:\s*:\s*([0-5]\d))?\s*(am|pm)(?!\w)")
_PART_OF_DAY = _Pattern(r"\s*(sáng|trưa|chiều|tối|đêm)(?!\w)")
#: Spoken periods, as minutes of day. "sáng" alone is not one: it also means "bright".
_PERIODS: tuple[tuple[_Pattern, str, tuple[int, int]], ...] = tuple(
    (_Pattern(rf"(?<!\w){phrase}(?!\w)"), phrase, window)
    for phrase, window in (
        ("sáng sớm", (5 * 60, 7 * 60 + 59)),
        ("buổi sáng", (5 * 60, 11 * 60 + 59)),
        ("buổi trưa", (11 * 60, 13 * 60 + 59)),
        ("buổi chiều", (13 * 60, 17 * 60 + 59)),
        ("chiều tối", (17 * 60, 19 * 60 + 59)),
        ("buổi tối", (18 * 60, 23 * 60 + 59)),
        ("ban đêm", (19 * 60, 23 * 60 + 59)),
        ("morning", (5 * 60, 11 * 60 + 59)),
        ("afternoon", (12 * 60, 17 * 60 + 59)),
        ("evening", (17 * 60, 21 * 60 + 59)),
    )
)
#: Minutes either side of a stated time. The window is pushed into the index, so
#: one that is too tight makes the answer unreachable, while one that is a little
#: wide only lets a few neighbouring keyframes compete; people also round ("7 giờ"
#: for a banner at 18:58). An "about" widens it further.
EXACT_TOLERANCE = 2
APPROX_TOLERANCE = 5
HOUR_TOLERANCE = 5
APPROX_HOUR_TOLERANCE = 15

# ---- dates ------------------------------------------------------------
_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
            ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sept", "sep"), ("october", "oct"), ("november", "nov"), ("december", "dec"),
        ),
        start=1,
    )
    for name in names
    # Folded, "may" is Vietnamese "máy"/"may" far more often than the month.
    if name != "may"
}
_MONTH = "(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")(?![a-z])\.?"
_DATE_PATTERNS: tuple[tuple[_Pattern, tuple[str | None, ...]], ...] = (
    (_Pattern(r"(?<!\d)(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)"), ("year", "month", "day")),
    (_Pattern(r"(?<!\d)(\d{1,2})[/.-](\d{1,2})[/.-](20\d{2})(?!\d)"), ("day", "month", "year")),
    (_Pattern(r"(?<!\d)(\d{1,2})\s+tháng\s+(\d{1,2})(?:\s*(?:năm|,)?\s*(20\d{2}))?(?!\d)"), ("day", "month", "year")),
    (_Pattern(r"(?<![\d/.:])(\d{1,2})/(\d{1,2})(?![\d/])"), ("day", "month")),
    (_Pattern(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?\s*[.\-]?\s*" + _MONTH + r"(?:\s*,?\s*(20\d{2}))?"), ("day", "month", "year")),
    (_Pattern(_MONTH + r"\s+(\d{1,2})(?:st|nd|rd|th)?(?!\d)(?:\s*,?\s*(20\d{2}))?"), ("month", "day", "year")),
)

# ---- race stage -------------------------------------------------------
_STAGE_WORDS = {
    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "sáu": 6, "bảy": 7,
    "tám": 8, "chín": 9, "mười": 10, "mười một": 11, "mười hai": 12,
    "đầu": 1, "đầu tiên": 1, "mở màn": 1, "cuối": RACE_STAGES, "cuối cùng": RACE_STAGES,
}
_STAGE_WORDS_FOLDED = {fold_vietnamese(word): stage for word, stage in _STAGE_WORDS.items()}
_STAGE = _Pattern(
    r"(?<!\w)(?:chặng|stage)\s+(?:thứ\s+|số\s+)?(\d{1,2}|"
    + "|".join(sorted(_STAGE_WORDS, key=len, reverse=True))
    + r")(?!\w)"
)


def _words(text: str) -> str:
    """Folded, punctuation-free, space-padded text for word-boundary phrase tests."""
    return " " + re.sub(r"[^a-z0-9]+", " ", fold_vietnamese(text)).strip() + " "


def _clock(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


@dataclass(frozen=True)
class Camera:
    id: str
    label: str
    banner: str
    streets: tuple[str, ...]
    videos: tuple[str, ...]


@dataclass(frozen=True)
class CameraVideo:
    video_id: str
    camera: str
    keyframes: int
    dates: tuple[str, ...]
    clock: tuple[str, str]
    #: `(minute_of_day, first_n, last_n)`, tiling the video in keyframe order.
    minutes: tuple[tuple[int, int, int], ...]


class TrafficCatalog:
    """The cameras, their videos, and the street phrases that name them."""

    def __init__(self, cameras: list[Camera], videos: dict[str, CameraVideo]):
        self.cameras = {camera.id: camera for camera in cameras}
        self.videos = videos
        self.dates = frozenset(date for video in videos.values() for date in video.dates)
        # street key -> (display name, phrases that name it)
        self.streets: dict[str, tuple[str, tuple[str, ...]]] = {}
        for camera in cameras:
            for street in camera.streets:
                key = _words(street).strip()
                phrases = (key, *STREET_ALIASES.get(key, ()))
                self.streets.setdefault(key, (street, tuple(f" {phrase} " for phrase in phrases)))

    @classmethod
    def load(cls, path: Path = CATALOG_PATH) -> "TrafficCatalog":
        document = json.loads(path.read_text(encoding="utf-8"))
        cameras = [
            Camera(
                id=item["id"],
                label=item["label"],
                banner=item["banner"],
                streets=tuple(item["streets"]),
                videos=tuple(item["videos"]),
            )
            for item in document["cameras"]
        ]
        videos = {
            video_id: CameraVideo(
                video_id=video_id,
                camera=item["camera"],
                keyframes=int(item["keyframes"]),
                dates=tuple(item["dates"]),
                clock=tuple(item["clock"]),
                minutes=tuple(tuple(run) for run in item["minutes"]),
            )
            for video_id, item in document["videos"].items()
        }
        return cls(cameras, videos)

    def streets_in(self, text: str) -> list[str]:
        """Street keys the text names, in catalogue order."""
        haystack = _words(text)
        return [key for key, (_, phrases) in self.streets.items() if any(p in haystack for p in phrases)]

    def cameras_on(self, streets: list[str]) -> list[Camera]:
        """The cameras naming the most of these streets (a junction beats a single street)."""
        wanted = set(streets)
        scored = [
            (len(wanted & {_words(street).strip() for street in camera.streets}), camera)
            for camera in self.cameras.values()
        ]
        best = max((score for score, _ in scored), default=0)
        return [camera for score, camera in scored if best and score == best]


@lru_cache(maxsize=1)
def load_catalog() -> TrafficCatalog:
    return TrafficCatalog.load()


# ---- parsing ----------------------------------------------------------


@dataclass(frozen=True)
class TimeCue:
    start: int  # minute of day, inclusive
    end: int
    text: str


def _with_part_of_day(hour: int, part: str | None) -> int:
    part = fold_vietnamese(part) if part else None
    if part in {"chieu", "toi"} and hour < 12:
        return hour + 12
    if part == "trua" and hour < 11:
        return hour + 12
    if part == "dem" and 6 <= hour < 12:
        return hour + 12
    if part == "dem" and hour == 12:
        return 0
    return hour


def parse_time(raw: str) -> TimeCue | None:
    """The time of day a query names, as a minute window; explicit times beat periods."""
    query = _Query.of(raw)
    text = query.text
    spans: list[tuple[int, int, int, str]] = []  # (position, first minute, last minute, text)
    taken: list[tuple[int, int]] = []

    def overlaps(match: re.Match[str]) -> bool:
        return any(a < match.end() and match.start() < b for a, b in taken)

    for pattern in (_AM_PM, _CLOCK, _HOUR_MINUTE, _HOUR):
        for match in pattern.pick(query).finditer(text):
            if overlaps(match):
                continue
            end = match.end()
            hour = int(match.group(1))
            if pattern is _AM_PM:
                precise = match.group(2) is not None
                minute = int(match.group(2) or 0)
                hour = hour % 12 + (12 if match.group(3) == "pm" else 0)
            else:
                precise = pattern is not _HOUR
                minute = int(match.group(2)) if precise else 0
                part = _PART_OF_DAY.pick(query).match(text, end)
                if part:
                    hour = _with_part_of_day(hour, part.group(1))
                    end = part.end()
            taken.append((match.start(), end))
            approx = bool(_APPROX.pick(query).search(text[max(0, match.start() - 24): match.start()]))
            if precise:
                slack = APPROX_TOLERANCE if approx else EXACT_TOLERANCE
                first = last = hour * 60 + minute
            else:
                slack = APPROX_HOUR_TOLERANCE if approx else HOUR_TOLERANCE
                first, last = hour * 60, hour * 60 + 59
            spans.append((match.start(), max(0, first - slack), min(1439, last + slack), text[match.start():end].strip()))
    if spans:
        spans.sort()
        # "từ 19:02 đến 19:05" names both ends of one window.
        return TimeCue(min(s[1] for s in spans), max(s[2] for s in spans), " – ".join(s[3] for s in spans))
    for pattern, phrase, (start, end) in _PERIODS:
        if pattern.pick(query).search(text):
            return TimeCue(start, end, phrase)
    return None


def parse_dates(raw: str) -> list[tuple[int | None, int, int]]:
    """`(year or None, month, day)` for every date the text writes, deduplicated."""
    query = _Query.of(raw)
    found: list[tuple[int | None, int, int]] = []
    taken: list[tuple[int, int]] = []
    for pattern, order in _DATE_PATTERNS:
        for match in pattern.pick(query).finditer(query.text):
            if any(a < match.end() and match.start() < b for a, b in taken):
                continue
            values = dict(zip(order, match.groups()))
            month_raw = values.get("month") or ""
            month = int(month_raw) if month_raw.isdigit() else _MONTHS.get(month_raw, 0)
            day = int(values.get("day") or 0)
            year = int(values["year"]) if values.get("year") else None
            if 1 <= month <= 12 and 1 <= day <= 31:
                taken.append((match.start(), match.end()))
                if (year, month, day) not in found:
                    found.append((year, month, day))
    return found


def parse_stage(raw: str) -> int | None:
    query = _Query.of(raw)
    match = _STAGE.pick(query).search(query.text)
    if not match:
        return None
    value = match.group(1)
    if value.isdigit():
        return int(value)
    return (_STAGE_WORDS_FOLDED if query.unaccented else _STAGE_WORDS)[value]


# ---- resolution -------------------------------------------------------


def _merge(intervals: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return tuple((lo, hi) for lo, hi in merged)


def _intersect(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    out = [(max(lo1, lo2), min(hi1, hi2)) for lo1, hi1 in a for lo2, hi2 in b if max(lo1, lo2) <= min(hi1, hi2)]
    return _merge(out)


@dataclass
class TrafficResolution:
    """What a query's traffic / race cues restrict, and why."""

    mode: str
    frames: FrameFilter | None = None
    streets: list[str] = field(default_factory=list)
    cameras: list[Camera] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    time: TimeCue | None = None
    race_stage: int | None = None
    videos: int = 0
    keyframes: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.frames is not None

    @property
    def reason_en(self) -> str:
        if self.mode == "off":
            return "Traffic/race filter turned off for this search."
        if not self.active:
            return "No camera, date, time or race-stage cue in the query."
        parts: list[str] = []
        if self.videos:
            what = []
            if self.cameras:
                labels = ", ".join(camera.label for camera in self.cameras[:4])
                more = f" +{len(self.cameras) - 4}" if len(self.cameras) > 4 else ""
                what.append(f"{len(self.cameras)} camera(s) ({labels}{more})")
            if self.dates:
                what.append(", ".join(self.dates))
            if self.time:
                what.append(f"{_clock(self.time.start)}–{_clock(self.time.end)}")
            parts.append(f"Traffic cameras: {'; '.join(what)} → {self.videos} video(s), {self.keyframes} keyframes.")
        if self.race_stage:
            parts.append(f"Cycling race: stage {self.race_stage} (S01-V{self.race_stage:03d}).")
        return " ".join(parts) + " Other folders are not affected."

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "active": self.active,
            "streets": self.streets,
            "cameras": [
                {"id": camera.id, "label": camera.label, "banner": camera.banner, "videos": list(camera.videos)}
                for camera in self.cameras
            ],
            "dates": self.dates,
            "time": (
                {"from": _clock(self.time.start), "to": _clock(self.time.end), "text": self.time.text}
                if self.time
                else None
            ),
            "race_stage": self.race_stage,
            "videos": self.videos,
            "keyframes": self.keyframes,
            "warnings": self.warnings,
            "reason_en": self.reason_en,
        }


def resolve_traffic(text: str, *, mode: str = "auto", catalog: TrafficCatalog | None = None) -> TrafficResolution:
    """Resolve a query's camera / date / time / stage cues into a frame filter.

    `mode` is "auto" (apply what the query names) or "off".
    """
    if mode == "off":
        return TrafficResolution(mode="off")
    result = TrafficResolution(mode="auto")
    if not (text or "").strip():
        return result
    catalog = catalog or load_catalog()

    # ---- traffic cameras: each cue narrows what the previous ones left ----
    candidates = {video_id: ((1, video.keyframes),) for video_id, video in catalog.videos.items()}
    narrowed_by: list[str] = []
    streets = catalog.streets_in(text)
    if streets:
        cameras = catalog.cameras_on(streets)
        allowed = {video_id for camera in cameras for video_id in camera.videos}
        candidates = {video_id: windows for video_id, windows in candidates.items() if video_id in allowed}
        result.streets = [catalog.streets[key][0] for key in streets]
        result.cameras = cameras
        narrowed_by.append("camera")

    wanted_dates: list[str] = []
    parsed_dates = parse_dates(text)
    for year, month, day in parsed_dates:
        suffix = f"-{month:02d}-{day:02d}"
        wanted_dates += sorted(
            date for date in catalog.dates
            if date.endswith(suffix) and (year is None or date.startswith(f"{year}-")) and date not in wanted_dates
        )
    if parsed_dates and not wanted_dates:
        written = ", ".join(f"{day:02d}/{month:02d}" + (f"/{year}" if year else "") for year, month, day in parsed_dates)
        result.warnings.append(f"No traffic camera recorded on {written}; date ignored.")
    elif wanted_dates:
        # A video whose banner date OCR never read cannot be ruled out.
        dated = {
            video_id: windows
            for video_id, windows in candidates.items()
            if not catalog.videos[video_id].dates or set(catalog.videos[video_id].dates) & set(wanted_dates)
        }
        if dated:
            candidates = dated
            result.dates = wanted_dates
            narrowed_by.append("date")
        else:
            result.warnings.append(f"None of the matched cameras recorded on {', '.join(wanted_dates)}; date ignored.")

    time_cue = parse_time(text)
    if time_cue:
        timed: dict[str, tuple[tuple[int, int], ...]] = {}
        for video_id, windows in candidates.items():
            runs = _merge([(lo, hi) for minute, lo, hi in catalog.videos[video_id].minutes if time_cue.start <= minute <= time_cue.end])
            overlap = _intersect(windows, runs)
            if overlap:
                timed[video_id] = overlap
        if timed:
            candidates = timed
            result.time = time_cue
            narrowed_by.append("time")
        else:
            result.warnings.append(
                f"No {'matched ' if narrowed_by else ''}traffic camera was recording at "
                f"{_clock(time_cue.start)}–{_clock(time_cue.end)}; time ignored."
            )

    families: list[str] = []
    windows: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    if narrowed_by:
        families.append(CAMERA_FAMILY)
        for video_id in sorted(candidates):
            intervals = candidates[video_id]
            whole = intervals == ((1, catalog.videos[video_id].keyframes),)
            windows.append((video_id, () if whole else intervals))
        result.videos = len(candidates)
        result.keyframes = sum(hi - lo + 1 for intervals in candidates.values() for lo, hi in intervals)

    # ---- cycling race stage ----
    stage = parse_stage(text)
    if stage is not None:
        if 1 <= stage <= RACE_STAGES:
            families.append(RACE_FAMILY)
            windows.append((f"S01-V{stage:03d}", ()))
            result.race_stage = stage
        else:
            result.warnings.append(f"The S01 race has stages 1–{RACE_STAGES}; stage {stage} ignored.")

    if families:
        result.frames = FrameFilter(families=tuple(families), windows=tuple(sorted(windows)))
    return result
