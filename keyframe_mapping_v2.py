#!/usr/bin/env python3
"""InfoShot++ retrieval metadata staging — the v2 sibling of ``keyframe_mapping.py``.

``keyframe_mapping.py`` (v1) maps speech/audio onto the *BTC* keyframe set and is
left completely untouched; this script produces an independent set of staging
artifacts under a separate output directory for retrieval **profile 2**
(InfoShot++, categories L21–L30).

Source of truth for every identity here is the final InfoShot++ map:

    <map-root>/Lxx/Lxx_Vyyy.csv     header: n,pts_time,fps,frame_idx

with two *different* numbers per row that must never be conflated:

    frame_idx   source/decode frame index — names the JPEG ``f{frame_idx:08d}.jpg``
                and the canonical frame identity ``{video_id}@f{frame_idx:08d}``
    n           1-based ordinal of the keyframe inside the video — the number the
                application/submit identity is built from

Identities emitted (kept byte-identical in shape to what the backend, Milvus
profile-2 collection and the Hugging Face media bucket already use):

    keyframe_id          "{video_id}/{n:03d}"
    frame_id             "{video_id}@f{frame_idx:08d}"
    submit_keyframe_id   "{category}/{video_id}/{n:03d}"
    keyframe_name        "{n:03d}"

Outputs (default ``elastic_staging_v2/``):

    keyframe_map.jsonl              one record per InfoShot++ keyframe
    speech_segments_mapped.jsonl    ASR segments joined by nearest pts_time
    audio_windows_mapped.jsonl      audio-event windows joined by nearest pts_time
    ocr_keyframes_mapped.jsonl      OCR joined EXACTLY on (video_id, frame_idx)
    unmapped_records.jsonl          every source record that was not emitted
    mapping_summary_v2.json         full audit

Nothing under ``elastic_staging/`` is read or written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from array import array
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# Pure formatting/quality helpers shared with v1 so the two staging schemas cannot
# drift. None of these read the BTC keyframe map — the *index* below is what
# decides identity, and it is built only from the InfoShot++ CSVs.
from keyframe_mapping import (
    AUDIO_TAG_STOPLIST,
    caption_quality,
    confidence_bucket,
    json_dump_line,
    make_keyframe_id,
    make_submit_keyframe_id,
    make_time_span_id,
    segment_role,
    submit_category,
    write_summary,
)


DEFAULT_MAP_ROOT = Path("/home/bao/Projects/ExtractKeyframe/keyframe_L/infoshootpp/map-keyframes")
DEFAULT_OUT_DIR = Path("elastic_staging_v2")
DEFAULT_OCR_PATH = Path("OCR/ocr_clean_no_L26.jsonl")

EXPECTED_VIDEOS = 873
EXPECTED_KEYFRAMES = 1_339_055
EXPECTED_CATEGORIES = tuple(f"L{number:02d}" for number in range(21, 31))
MAP_HEADER = ["n", "pts_time", "fps", "frame_idx"]

#: OCR for these categories does not exist yet in the InfoShot++ OCR artifact.
#: Their absence is a known state of the corpus, not a mapping failure, so the
#: audit reports coverage against "corpus minus these" instead of the whole corpus.
DEFAULT_OCR_EXPECTED_MISSING_CATEGORIES = ("L26",)


def make_frame_id(video_id: str, frame_idx: int) -> str:
    """Canonical InfoShot++ frame identity — the JPEG's own name, not the ordinal."""
    return f"{video_id}@f{frame_idx:08d}"


@dataclass(frozen=True)
class Keyframe:
    n: int
    pts_time: float
    fps: float
    frame_idx: int


class VideoKeyframes:
    """One video's InfoShot++ map, stored as parallel arrays.

    The whole corpus is 1.34M rows; holding it as objects costs ~10x what the
    arrays below do, and every lookup this module needs is a binary search:

    * ``position_for_frame_idx`` — exact join for OCR. ``frame_idx`` is validated
      strictly increasing, so it is already a sorted key.
    * ``nearest_position``      — time join for speech/audio. ``pts_time`` is
      *not* assumed monotonic: 30 of the final L25 maps carry sub-frame
      regressions, so a separate sorted view is built for those videos.

    ``n`` is never stored: the loader proves it runs 1..len with no gaps, which
    makes ``n == position + 1`` an invariant rather than an assumption.
    """

    __slots__ = ("video_id", "category", "frame_idx", "pts_time", "fps", "_pts_sorted", "_pts_order")

    def __init__(
        self,
        video_id: str,
        category: str,
        frame_idx: "array[int]",
        pts_time: "array[float]",
        fps: "array[float]",
    ):
        self.video_id = video_id
        self.category = category
        self.frame_idx = frame_idx
        self.pts_time = pts_time
        self.fps = fps
        monotonic = all(pts_time[i] <= pts_time[i + 1] for i in range(len(pts_time) - 1))
        if monotonic:
            self._pts_sorted = pts_time
            self._pts_order: "array[int] | None" = None
        else:
            # Ties keep the lower ordinal first, matching v1's "prefer the earlier
            # keyframe" tie-break once the two candidates are equidistant.
            order = sorted(range(len(pts_time)), key=lambda i: (pts_time[i], i))
            self._pts_order = array("i", order)
            self._pts_sorted = array("d", (pts_time[i] for i in order))

    def __len__(self) -> int:
        return len(self.frame_idx)

    @property
    def is_pts_monotonic(self) -> bool:
        return self._pts_order is None

    def keyframe(self, position: int) -> Keyframe:
        return Keyframe(
            n=position + 1,
            pts_time=self.pts_time[position],
            fps=self.fps[position],
            frame_idx=self.frame_idx[position],
        )

    def position_for_frame_idx(self, frame_idx: int) -> int | None:
        position = bisect_left(self.frame_idx, frame_idx)
        if position < len(self.frame_idx) and self.frame_idx[position] == frame_idx:
            return position
        return None

    def nearest_position(self, time_s: float) -> int:
        def source_index(sorted_position: int) -> int:
            if self._pts_order is None:
                return sorted_position
            return self._pts_order[sorted_position]

        position = bisect_left(self._pts_sorted, time_s)
        if position <= 0:
            return source_index(0)
        if position >= len(self._pts_sorted):
            return source_index(len(self._pts_sorted) - 1)
        previous_delta = time_s - self._pts_sorted[position - 1]
        next_delta = self._pts_sorted[position] - time_s
        return source_index(position - 1 if previous_delta <= next_delta else position)


class InfoShotPPIndex:
    """Every InfoShot++ map CSV, validated on load."""

    def __init__(self, videos: dict[str, VideoKeyframes]):
        self.videos = videos

    @classmethod
    def from_csv_root(cls, root: Path) -> "InfoShotPPIndex":
        paths = sorted(root.glob("L*/L*_V*.csv"))
        if not paths:
            raise FileNotFoundError(f"Không tìm thấy map-keyframe CSV nào dưới {root}")
        videos: dict[str, VideoKeyframes] = {}
        for path in paths:
            video_id = path.stem
            category = path.parent.name
            if submit_category(video_id) != category:
                raise ValueError(f"{path}: video_id {video_id!r} không thuộc category {category!r}")
            if video_id in videos:
                raise ValueError(f"Trùng map cho video {video_id}")
            videos[video_id] = cls._read_video(path, video_id, category)
        return cls(videos)

    @staticmethod
    def _read_video(path: Path, video_id: str, category: str) -> VideoKeyframes:
        frame_idx = array("i")
        pts_time = array("d")
        fps_values = array("d")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != MAP_HEADER:
                raise ValueError(f"{path}: header {reader.fieldnames!r}, cần {MAP_HEADER!r}")
            previous_frame_idx = -1
            for line_number, row in enumerate(reader, start=2):
                try:
                    n = int(row["n"])
                    pts = float(row["pts_time"])
                    fps = float(row["fps"])
                    index = int(row["frame_idx"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: dòng map không hợp lệ") from exc
                if n != len(frame_idx) + 1:
                    raise ValueError(f"{path}:{line_number}: n={n}, cần {len(frame_idx) + 1}")
                if index < 0:
                    raise ValueError(f"{path}:{line_number}: frame_idx={index} âm")
                if index <= previous_frame_idx:
                    raise ValueError(
                        f"{path}:{line_number}: frame_idx={index} không tăng nghiêm ngặt "
                        f"(trước đó {previous_frame_idx})"
                    )
                if not math.isfinite(pts) or pts < 0 or not math.isfinite(fps) or fps <= 0:
                    raise ValueError(f"{path}:{line_number}: pts_time/fps không hợp lệ")
                frame_idx.append(index)
                pts_time.append(pts)
                fps_values.append(fps)
                previous_frame_idx = index
        if not frame_idx:
            raise ValueError(f"{path}: không có keyframe nào")
        return VideoKeyframes(video_id, category, frame_idx, pts_time, fps_values)

    # ---- audit ---------------------------------------------------------
    def audit(self) -> dict[str, Any]:
        """Corpus-level facts plus the duplicate/validity checks the task requires.

        The duplicate scan is exhaustive but never holds 1.34M ids at once:
        ``video_id`` is unique across the corpus (it is the dict key and every id
        emitted starts with it), so a collision can only happen *inside* one
        video. Each video is therefore scanned with its own id sets — at most
        ~5.2k entries — which covers ``submit_keyframe_id``, ``frame_id`` and
        ``(video_id, frame_idx)`` completely.
        """
        category_videos: Counter[str] = Counter()
        category_keyframes: Counter[str] = Counter()
        duplicate_submit_ids = 0
        duplicate_frame_ids = 0
        duplicate_video_frame_idx = 0
        invalid_frame_idx = 0
        non_monotonic_pts_videos: list[str] = []
        max_n = 0
        max_frame_idx = 0
        keyframes = 0

        for video_id, video in self.videos.items():
            category_videos[video.category] += 1
            category_keyframes[video.category] += len(video)
            keyframes += len(video)
            max_n = max(max_n, len(video))
            max_frame_idx = max(max_frame_idx, video.frame_idx[-1])
            if not video.is_pts_monotonic:
                non_monotonic_pts_videos.append(video_id)
            submit_ids: set[str] = set()
            frame_ids: set[str] = set()
            frame_indices: set[int] = set()
            for position in range(len(video)):
                keyframe = video.keyframe(position)
                if keyframe.frame_idx < 0:
                    invalid_frame_idx += 1
                submit_id = make_submit_keyframe_id(video_id, keyframe.n)
                frame_id = make_frame_id(video_id, keyframe.frame_idx)
                duplicate_submit_ids += submit_id in submit_ids
                duplicate_frame_ids += frame_id in frame_ids
                duplicate_video_frame_idx += keyframe.frame_idx in frame_indices
                submit_ids.add(submit_id)
                frame_ids.add(frame_id)
                frame_indices.add(keyframe.frame_idx)

        categories = sorted(category_videos)
        return {
            "videos": len(self.videos),
            "keyframes": keyframes,
            "categories": categories,
            "category_video_counts": dict(sorted(category_videos.items())),
            "category_keyframe_counts": dict(sorted(category_keyframes.items())),
            "max_keyframe_n": max_n,
            "max_frame_idx": max_frame_idx,
            "duplicate_submit_keyframe_id": duplicate_submit_ids,
            "duplicate_frame_id": duplicate_frame_ids,
            "duplicate_video_frame_idx": duplicate_video_frame_idx,
            "invalid_frame_idx": invalid_frame_idx,
            "videos_with_non_monotonic_pts_time": len(non_monotonic_pts_videos),
            "non_monotonic_pts_time_videos": sorted(non_monotonic_pts_videos)[:50],
        }

    def check_expectations(self, audit: dict[str, Any]) -> dict[str, Any]:
        checks = {
            "videos": {"expected": EXPECTED_VIDEOS, "actual": audit["videos"]},
            "keyframes": {"expected": EXPECTED_KEYFRAMES, "actual": audit["keyframes"]},
            "categories": {
                "expected": list(EXPECTED_CATEGORIES),
                "actual": audit["categories"],
            },
            "duplicate_submit_keyframe_id": {"expected": 0, "actual": audit["duplicate_submit_keyframe_id"]},
            "duplicate_frame_id": {"expected": 0, "actual": audit["duplicate_frame_id"]},
            "duplicate_video_frame_idx": {"expected": 0, "actual": audit["duplicate_video_frame_idx"]},
            "invalid_frame_idx": {"expected": 0, "actual": audit["invalid_frame_idx"]},
        }
        for check in checks.values():
            check["ok"] = check["expected"] == check["actual"]
        return checks

    def iter_rows(self) -> Iterator[tuple[VideoKeyframes, int]]:
        for video_id in sorted(self.videos):
            video = self.videos[video_id]
            for position in range(len(video)):
                yield video, position


# ---------------------------------------------------------------------------
# record builders
# ---------------------------------------------------------------------------


def keyframe_map_record(video: VideoKeyframes, position: int) -> dict[str, Any]:
    keyframe = video.keyframe(position)
    return {
        "keyframe_id": make_keyframe_id(video.video_id, keyframe.n),
        "frame_id": make_frame_id(video.video_id, keyframe.frame_idx),
        "submit_keyframe_id": make_submit_keyframe_id(video.video_id, keyframe.n),
        "video_id": video.video_id,
        "category_hint": video.category,
        "submit_category": video.category,
        "keyframe_n": keyframe.n,
        "keyframe_name": f"{keyframe.n:03d}",
        "pts_time": keyframe.pts_time,
        "fps": keyframe.fps,
        "frame_idx": keyframe.frame_idx,
    }


def keyframe_fields(video: VideoKeyframes, label: str, keyframe: Keyframe) -> dict[str, Any]:
    """v1's per-anchor field block, plus the InfoShot++ ``frame_id``."""
    return {
        f"{label}_keyframe_id": make_keyframe_id(video.video_id, keyframe.n),
        f"{label}_frame_id": make_frame_id(video.video_id, keyframe.frame_idx),
        f"{label}_submit_keyframe_id": make_submit_keyframe_id(video.video_id, keyframe.n),
        f"{label}_keyframe_n": keyframe.n,
        f"{label}_keyframe_pts_time": keyframe.pts_time,
        f"{label}_keyframe_frame_idx": keyframe.frame_idx,
    }


def map_time_span(video: VideoKeyframes, start: float, end: float) -> dict[str, Any]:
    """Anchor a [start, end] window onto start/center/end keyframes.

    The flat ``keyframe_*`` fields mirror the center anchor, which is what the
    backend reads (``submit_keyframe_id`` first, ``center_submit_keyframe_id``
    as a fallback) — identical to the v1 contract.
    """
    center = (start + end) / 2.0
    mapped: dict[str, Any] = {
        "duration": round(end - start, 6),
        "center_time": round(center, 6),
    }
    for label, time_s in (("start", start), ("center", center), ("end", end)):
        mapped.update(keyframe_fields(video, label, video.keyframe(video.nearest_position(time_s))))
    mapped["keyframe_id"] = mapped["center_keyframe_id"]
    mapped["frame_id"] = mapped["center_frame_id"]
    mapped["submit_keyframe_id"] = mapped["center_submit_keyframe_id"]
    mapped["keyframe_n"] = mapped["center_keyframe_n"]
    mapped["keyframe_name"] = f"{mapped['center_keyframe_n']:03d}"
    mapped["keyframe_pts_time"] = mapped["center_keyframe_pts_time"]
    mapped["keyframe_frame_idx"] = mapped["center_keyframe_frame_idx"]
    mapped["submit_category"] = video.category
    return mapped


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------


class UnmappedReport:
    """Every source record that did not become a v2 document, with its reason.

    Records are never silently dropped: speech/audio for the K01–K20 half of the
    BTC corpus have no InfoShot++ keyframe at all, and that has to be visible as
    a counted, listed outcome rather than as a smaller output file.
    """

    def __init__(self, path: Path):
        self._handle = path.open("w", encoding="utf-8")
        self.path = path
        self.by_reason: Counter[str] = Counter()
        self.by_artifact: Counter[str] = Counter()
        self.videos_by_reason: dict[str, set[str]] = {}

    def add(self, artifact: str, reason: str, video_id: str, **detail: Any) -> None:
        self.by_reason[reason] += 1
        self.by_artifact[artifact] += 1
        self.videos_by_reason.setdefault(reason, set()).add(video_id)
        json_dump_line(
            self._handle,
            {"artifact": artifact, "reason": reason, "video_id": video_id, **detail},
        )

    @property
    def total(self) -> int:
        return sum(self.by_reason.values())

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_reason": dict(sorted(self.by_reason.items())),
            "by_artifact": dict(sorted(self.by_artifact.items())),
            "videos_by_reason": {
                reason: {
                    "count": len(videos),
                    "categories": dict(sorted(Counter(submit_category(v) for v in videos).items())),
                    "sample": sorted(videos)[:20],
                }
                for reason, videos in sorted(self.videos_by_reason.items())
            },
            "report": str(self.path),
        }

    def close(self) -> None:
        self._handle.close()


def write_keyframe_map(index: InfoShotPPIndex, out_path: Path) -> dict[str, Any]:
    records = 0
    with out_path.open("w", encoding="utf-8") as out:
        for video, position in index.iter_rows():
            json_dump_line(out, keyframe_map_record(video, position))
            records += 1
    return {"records": records, "output": str(out_path)}


def _time_span_source(
    index: InfoShotPPIndex,
    path: Path,
    artifact: str,
    unmapped: UnmappedReport,
) -> Iterator[tuple[VideoKeyframes, int, dict[str, Any], float, float]]:
    """Yield (video, source_index, row, start, end) for rows this corpus can map."""
    file_video_id = path.name.split(".")[0]
    rows = json.loads(path.read_text(encoding="utf-8"))
    for source_index, row in enumerate(rows):
        video_id = str(row.get("video_id") or "")
        if video_id != file_video_id:
            unmapped.add(
                artifact,
                "video_id_file_mismatch",
                video_id or file_video_id,
                source_file=str(path),
                source_index=source_index,
                file_video_id=file_video_id,
            )
            continue
        video = index.videos.get(video_id)
        if video is None:
            unmapped.add(
                artifact,
                "video_not_in_infoshotpp_corpus",
                video_id,
                source_file=str(path),
                source_index=source_index,
            )
            continue
        try:
            start = float(row["start"])
            end = float(row["end"])
        except (KeyError, TypeError, ValueError):
            unmapped.add(
                artifact,
                "invalid_time_span",
                video_id,
                source_file=str(path),
                source_index=source_index,
                start=row.get("start"),
                end=row.get("end"),
            )
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            unmapped.add(
                artifact,
                "invalid_time_span",
                video_id,
                source_file=str(path),
                source_index=source_index,
                start=start,
                end=end,
            )
            continue
        yield video, source_index, row, start, end


def write_speech_segments(
    index: InfoShotPPIndex, speech_dir: Path, out_path: Path, unmapped: UnmappedReport
) -> dict[str, Any]:
    stats = {
        "files": 0,
        "files_with_records": 0,
        "records": 0,
        "videos": 0,
        "low_confidence_records": 0,
        "mid_confidence_records": 0,
        "missing_confidence_records": 0,
    }
    videos: set[str] = set()
    paths = sorted(speech_dir.glob("*.speech.json"))
    stats["files"] = len(paths)
    with out_path.open("w", encoding="utf-8") as out:
        for path in paths:
            emitted_here = 0
            for video, source_index, row, start, end in _time_span_source(
                index, path, "speech_segments", unmapped
            ):
                score = row.get("avg_word_score")
                bucket = confidence_bucket(score)
                words = row.get("words") or []
                record = {
                    "segment_id": make_time_span_id(video.video_id, start, end, f"s{source_index:06d}"),
                    "video_id": video.video_id,
                    "start": start,
                    "end": end,
                    "text": row.get("text", ""),
                    "avg_word_score": score,
                    "confidence_bucket": bucket,
                    "segment_role": segment_role(start, end),
                    "word_count": len(words),
                    "avg_logprob": row.get("avg_logprob"),
                    "no_speech_prob": row.get("no_speech_prob"),
                    "source_segment_idx": source_index,
                    "source_speech_file": str(path),
                }
                record.update(map_time_span(video, start, end))
                json_dump_line(out, record)
                emitted_here += 1
                stats["records"] += 1
                videos.add(video.video_id)
                if bucket == "low":
                    stats["low_confidence_records"] += 1
                elif bucket == "mid":
                    stats["mid_confidence_records"] += 1
                elif bucket == "missing":
                    stats["missing_confidence_records"] += 1
            if emitted_here:
                stats["files_with_records"] += 1
    stats["videos"] = len(videos)
    stats["output"] = str(out_path)
    return stats


def write_audio_windows(
    index: InfoShotPPIndex, audio_dir: Path, out_path: Path, unmapped: UnmappedReport
) -> dict[str, Any]:
    stats = {
        "files": 0,
        "files_with_records": 0,
        "records": 0,
        "videos": 0,
        "caption_records": 0,
        "stoplist_tag_records": 0,
    }
    videos: set[str] = set()
    paths = sorted(audio_dir.glob("*.audio.json"))
    stats["files"] = len(paths)
    with out_path.open("w", encoding="utf-8") as out:
        for path in paths:
            emitted_here = 0
            for video, source_index, row, start, end in _time_span_source(
                index, path, "audio_windows", unmapped
            ):
                tags = row.get("tags") or []
                tag_labels = [tag.get("label") for tag in tags]
                tag_scores = [tag.get("score") for tag in tags]
                top1 = tags[0] if tags else {}
                caption = row.get("caption")
                stoplist_hit = any(label in AUDIO_TAG_STOPLIST for label in tag_labels)
                record = {
                    "window_id": make_time_span_id(video.video_id, start, end, f"a{source_index:06d}"),
                    "video_id": video.video_id,
                    "start": start,
                    "end": end,
                    "glap_idx": row.get("glap_idx"),
                    "tags": tags,
                    "tag_labels": tag_labels,
                    "tag_scores": tag_scores,
                    "top1_label": top1.get("label"),
                    "top1_score": top1.get("score"),
                    "caption": caption,
                    "has_caption": bool(caption),
                    "caption_quality": caption_quality(caption),
                    "audio_stoplist_hit": stoplist_hit,
                    "top1_is_stoplisted": top1.get("label") in AUDIO_TAG_STOPLIST,
                    "source_audio_file": str(path),
                }
                record.update(map_time_span(video, start, end))
                json_dump_line(out, record)
                emitted_here += 1
                stats["records"] += 1
                videos.add(video.video_id)
                if caption:
                    stats["caption_records"] += 1
                if stoplist_hit:
                    stats["stoplist_tag_records"] += 1
            if emitted_here:
                stats["files_with_records"] += 1
    stats["videos"] = len(videos)
    stats["output"] = str(out_path)
    return stats


def parse_ocr_id(raw_id: str) -> tuple[str, str, int]:
    """Split an InfoShot++ OCR id ``Lxx/Lxx_Vyyy/f00012345``.

    The frame token is ``f`` + the zero-padded **frame_idx**, which is why the v1
    uploader's ``int(frame_name)`` cannot be reused: ``int("f00012345")`` raises,
    and the 12345 it hides is a decode index, never a keyframe ordinal.
    """
    parts = raw_id.split("/")
    if len(parts) != 3:
        raise ValueError(f"OCR id sai định dạng: {raw_id!r}")
    category, video_id, frame_token = parts
    if not frame_token.startswith("f") or not frame_token[1:].isdigit():
        raise ValueError(f"OCR frame token phải là f<frame_idx>, gặp {frame_token!r} trong {raw_id!r}")
    return category, video_id, int(frame_token[1:])


def write_ocr_keyframes(
    index: InfoShotPPIndex,
    ocr_path: Path,
    out_path: Path,
    unmapped: UnmappedReport,
    *,
    expected_missing_categories: tuple[str, ...],
) -> dict[str, Any]:
    """Project the InfoShot++ OCR artifact onto submit identity.

    The join is EXACT on ``(video_id, frame_idx)`` — nearest-timestamp matching is
    deliberately not available here, because an OCR record describes one specific
    JPEG and attaching it to a neighbouring keyframe would invent evidence.
    """
    seen_frame_idx: dict[str, set[int]] = {}
    categories: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    records_in = 0
    records_out = 0
    duplicate_records = 0
    map_n_mismatches = 0
    frame_id_mismatches = 0
    empty_text_clean = 0
    with_clock = 0
    per_video_out: Counter[str] = Counter()

    with ocr_path.open("r", encoding="utf-8") as source, out_path.open("w", encoding="utf-8") as out:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            records_in += 1
            record = json.loads(line)
            raw_id = str(record.get("id") or "")
            try:
                id_category, video_id, frame_idx = parse_ocr_id(raw_id)
            except ValueError as exc:
                unmapped.add(
                    "ocr_keyframes",
                    "invalid_ocr_id",
                    str(record.get("video_id") or ""),
                    source_file=str(ocr_path),
                    source_line=line_number,
                    ocr_id=raw_id,
                    detail=str(exc),
                )
                continue
            record_video_id = str(record.get("video_id") or video_id)
            if record_video_id != video_id:
                unmapped.add(
                    "ocr_keyframes",
                    "ocr_id_video_mismatch",
                    video_id,
                    source_file=str(ocr_path),
                    source_line=line_number,
                    ocr_id=raw_id,
                    record_video_id=record_video_id,
                )
                continue
            video = index.videos.get(video_id)
            if video is None:
                unmapped.add(
                    "ocr_keyframes",
                    "video_not_in_infoshotpp_corpus",
                    video_id,
                    source_file=str(ocr_path),
                    source_line=line_number,
                    ocr_id=raw_id,
                )
                continue
            if video.category != id_category:
                unmapped.add(
                    "ocr_keyframes",
                    "ocr_id_category_mismatch",
                    video_id,
                    source_file=str(ocr_path),
                    source_line=line_number,
                    ocr_id=raw_id,
                    map_category=video.category,
                )
                continue
            position = video.position_for_frame_idx(frame_idx)
            if position is None:
                unmapped.add(
                    "ocr_keyframes",
                    "frame_idx_not_in_map",
                    video_id,
                    source_file=str(ocr_path),
                    source_line=line_number,
                    ocr_id=raw_id,
                    frame_idx=frame_idx,
                )
                continue
            frames = seen_frame_idx.setdefault(video_id, set())
            if frame_idx in frames:
                duplicate_records += 1
                unmapped.add(
                    "ocr_keyframes",
                    "duplicate_video_frame_idx",
                    video_id,
                    source_file=str(ocr_path),
                    source_line=line_number,
                    ocr_id=raw_id,
                    frame_idx=frame_idx,
                )
                continue
            frames.add(frame_idx)

            keyframe = video.keyframe(position)
            # The artifact carries its own ``map_n``/``frame_id``; they are treated
            # as a cross-check against the CSV, never as the source of the join.
            source_map_n = record.get("map_n")
            if source_map_n is not None and int(source_map_n) != keyframe.n:
                map_n_mismatches += 1
                unmapped.add(
                    "ocr_keyframes",
                    "map_n_disagrees_with_map_csv",
                    video_id,
                    source_file=str(ocr_path),
                    source_line=line_number,
                    ocr_id=raw_id,
                    frame_idx=frame_idx,
                    source_map_n=source_map_n,
                    map_csv_n=keyframe.n,
                )
                continue
            frame_id = make_frame_id(video_id, frame_idx)
            if record.get("frame_id") and record["frame_id"] != frame_id:
                frame_id_mismatches += 1

            keyframe_name = f"{keyframe.n:03d}"
            text_clean = record.get("text_clean") or ""
            document = {
                "ocr_id": raw_id,
                "submit_keyframe_id": make_submit_keyframe_id(video_id, keyframe.n),
                "frame_id": frame_id,
                "keyframe_id": make_keyframe_id(video_id, keyframe.n),
                "video_id": video_id,
                "category": record.get("category") or video.category,
                "submit_category": video.category,
                "image_path": record.get("image_path"),
                "status": record.get("status"),
                "error": record.get("error"),
                "keyframe_n": keyframe.n,
                "keyframe_name": keyframe_name,
                "frame_idx": frame_idx,
                "pts_time": keyframe.pts_time,
                "fps": keyframe.fps,
                "text_clean": text_clean,
                "text_clean_fold": record.get("text_clean_fold") or "",
                "text_nfc": record.get("text_nfc") or "",
                "clock": record.get("clock"),
                "hour": record.get("hour"),
                "boxes": record.get("boxes") or [],
                "ts": record.get("ts"),
            }
            json_dump_line(out, document)
            records_out += 1
            per_video_out[video_id] += 1
            categories[video.category] += 1
            status_counts[str(record.get("status") or "")] += 1
            if not text_clean:
                empty_text_clean += 1
            if record.get("clock"):
                with_clock += 1

    coverage = _ocr_coverage(index, per_video_out, expected_missing_categories)
    return {
        "source": str(ocr_path),
        "output": str(out_path),
        "records_in": records_in,
        "records_out": records_out,
        "unmapped": records_in - records_out,
        "duplicate_video_frame_idx": duplicate_records,
        "duplicate_submit_keyframe_id": duplicate_records,
        "map_n_disagreements": map_n_mismatches,
        "frame_id_disagreements": frame_id_mismatches,
        "category_counts": dict(sorted(categories.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "empty_text_clean": empty_text_clean,
        "records_with_clock": with_clock,
        "coverage": coverage,
    }


def _ocr_coverage(
    index: InfoShotPPIndex,
    per_video_out: Counter[str],
    expected_missing_categories: tuple[str, ...],
) -> dict[str, Any]:
    """Compare OCR output against the corpus MINUS the categories known to be absent.

    Measuring against the full 1,339,055-keyframe corpus would flag the missing
    L26 OCR as a data loss; it is a known gap in the OCR artifact, so it is
    subtracted from the denominator and reported separately instead.
    """
    expected_missing = set(expected_missing_categories)
    per_category_records: Counter[str] = Counter()
    denominator = 0
    covered = 0
    incomplete_videos: list[dict[str, Any]] = []
    for video_id, video in index.videos.items():
        emitted = per_video_out.get(video_id, 0)
        per_category_records[video.category] += emitted
        if video.category in expected_missing:
            continue
        denominator += len(video)
        covered += emitted
        if emitted != len(video):
            incomplete_videos.append(
                {"video_id": video_id, "keyframes": len(video), "ocr_records": emitted}
            )
    observed_missing = sorted(
        category for category, count in per_category_records.items() if count == 0
    )
    return {
        "expected_missing_categories": sorted(expected_missing),
        "observed_missing_categories": observed_missing,
        "missing_categories_are_intentional": observed_missing == sorted(expected_missing),
        "corpus_keyframes_excluding_expected_missing": denominator,
        "ocr_records_covering_corpus": covered,
        "corpus_keyframes_without_ocr": denominator - covered,
        "videos_with_incomplete_ocr": len(incomplete_videos),
        "videos_with_incomplete_ocr_sample": incomplete_videos[:20],
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

ARTIFACTS = ("keyframe_map", "speech_segments", "audio_windows", "ocr_keyframes")


def build_staging(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.resolve() == Path("elastic_staging").resolve():
        raise SystemExit("--out-dir không được trỏ vào elastic_staging (staging v1 là bất biến).")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or ARTIFACTS)

    started = time.time()
    print(f"[map] đọc InfoShot++ map từ {args.map_dir}")
    index = InfoShotPPIndex.from_csv_root(args.map_dir)
    corpus = index.audit()
    expectations = index.check_expectations(corpus)
    print(
        f"[map] videos={corpus['videos']:,} keyframes={corpus['keyframes']:,} "
        f"categories={','.join(corpus['categories'])}"
    )
    failed = [name for name, check in expectations.items() if not check["ok"]]
    if failed and not args.allow_unexpected_corpus:
        raise SystemExit(
            "Corpus InfoShot++ không khớp kỳ vọng: "
            + json.dumps({name: expectations[name] for name in failed}, ensure_ascii=False)
        )

    unmapped = UnmappedReport(args.out_dir / "unmapped_records.jsonl")
    outputs: dict[str, str] = {}
    summary: dict[str, Any] = {}
    try:
        if "keyframe_map" in selected:
            print("[keyframe_map] ghi keyframe_map.jsonl …")
            summary["keyframe_map"] = write_keyframe_map(index, args.out_dir / "keyframe_map.jsonl")
            outputs["keyframe_map"] = summary["keyframe_map"]["output"]
            print(f"[keyframe_map] records={summary['keyframe_map']['records']:,}")

        if "speech_segments" in selected:
            print(f"[speech] đọc {args.speech_dir} …")
            summary["speech"] = write_speech_segments(
                index, args.speech_dir, args.out_dir / "speech_segments_mapped.jsonl", unmapped
            )
            outputs["speech_segments"] = summary["speech"]["output"]
            print(f"[speech] records={summary['speech']['records']:,} videos={summary['speech']['videos']:,}")

        if "audio_windows" in selected:
            print(f"[audio] đọc {args.audio_dir} …")
            summary["audio"] = write_audio_windows(
                index, args.audio_dir, args.out_dir / "audio_windows_mapped.jsonl", unmapped
            )
            outputs["audio_windows"] = summary["audio"]["output"]
            print(f"[audio] records={summary['audio']['records']:,} videos={summary['audio']['videos']:,}")

        if "ocr_keyframes" in selected:
            print(f"[ocr] đọc {args.ocr_path} …")
            summary["ocr"] = write_ocr_keyframes(
                index,
                args.ocr_path,
                args.out_dir / "ocr_keyframes_mapped.jsonl",
                unmapped,
                expected_missing_categories=tuple(args.ocr_expected_missing_categories),
            )
            outputs["ocr_keyframes"] = summary["ocr"]["output"]
            print(
                f"[ocr] in={summary['ocr']['records_in']:,} out={summary['ocr']['records_out']:,} "
                f"unmapped={summary['ocr']['unmapped']:,}"
            )
    finally:
        unmapped.close()

    unmapped_summary = unmapped.summary()
    # "Not in the InfoShot++ corpus" is the expected fate of every K01–K20 speech
    # and audio record; anything else means a record that SHOULD have mapped did not.
    in_corpus_unmapped = sum(
        count
        for reason, count in unmapped.by_reason.items()
        if reason != "video_not_in_infoshotpp_corpus"
    )
    unmapped_summary["out_of_corpus"] = unmapped.by_reason.get("video_not_in_infoshotpp_corpus", 0)
    unmapped_summary["in_corpus_failures"] = in_corpus_unmapped

    result = {
        "generated_at_unix": time.time(),
        "elapsed_s": round(time.time() - started, 2),
        "retrieval_profile": "infoshotpp",
        "schema_version": "v2",
        "sources": {
            "map_dir": str(args.map_dir),
            "speech_dir": str(args.speech_dir),
            "audio_dir": str(args.audio_dir),
            "ocr_path": str(args.ocr_path),
        },
        "corpus": corpus,
        "expectations": expectations,
        "artifacts_generated": sorted(selected),
        "unmapped": unmapped_summary,
        "outputs": outputs,
        **summary,
    }
    write_summary(args.out_dir / "mapping_summary_v2.json", result)

    if in_corpus_unmapped and not args.allow_in_corpus_unmapped:
        raise SystemExit(
            f"{in_corpus_unmapped:,} record thuộc corpus InfoShot++ không map được — "
            f"xem {unmapped.path}. Dùng --allow-in-corpus-unmapped để bỏ qua."
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_ROOT, help="Root chứa Lxx/Lxx_Vyyy.csv")
    parser.add_argument("--speech-dir", type=Path, default=Path("speech_out"))
    parser.add_argument("--audio-dir", type=Path, default=Path("results_audio_event/audio_out"))
    parser.add_argument("--ocr-path", type=Path, default=DEFAULT_OCR_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--only", nargs="*", choices=ARTIFACTS, help="Chỉ sinh các artifact được chọn.")
    parser.add_argument(
        "--ocr-expected-missing-categories",
        nargs="*",
        default=list(DEFAULT_OCR_EXPECTED_MISSING_CATEGORIES),
        help="Category chưa có OCR (mặc định L26) — vắng mặt là chủ ý, không phải lỗi thiếu dữ liệu.",
    )
    parser.add_argument(
        "--allow-unexpected-corpus",
        action="store_true",
        help="Cho phép chạy khi corpus khác 873 video / 1,339,055 keyframe / L21-L30.",
    )
    parser.add_argument(
        "--allow-in-corpus-unmapped",
        action="store_true",
        help="Không dừng khi có record thuộc corpus mà không map được.",
    )
    return parser.parse_args()


def main() -> None:
    summary = build_staging(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
