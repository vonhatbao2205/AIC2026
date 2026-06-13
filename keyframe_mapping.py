#!/usr/bin/env python3
"""Prepare audio/speech metadata with BTC keyframe mappings.

Raw extraction outputs stay untouched. This script creates JSONL staging files
that can later be bulk-indexed into Elasticsearch or joined with Milvus hits.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AUDIO_TAG_STOPLIST = {"Speech synthesizer", "Mantra"}
GENERIC_CAPTION_PATTERNS = (
    "contains non-speech sounds",
    "music and instruments",
    "non-speech sounds, music",
)
VIETNAMESE_DIACRITIC_RE = re.compile(
    r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệ"
    r"ìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Keyframe:
    n: int
    pts_time: float
    fps: float
    frame_idx: int


class KeyframeIndex:
    def __init__(self, by_video: dict[str, list[Keyframe]]):
        self.by_video = by_video
        self._times = {video: [kf.pts_time for kf in rows] for video, rows in by_video.items()}

    @classmethod
    def from_csv_root(cls, root: Path) -> "KeyframeIndex":
        by_video: dict[str, list[Keyframe]] = {}
        for path in sorted(root.rglob("*.csv")):
            video_id = path.stem
            rows: list[Keyframe] = []
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                expected = ["n", "pts_time", "fps", "frame_idx"]
                if reader.fieldnames != expected:
                    raise ValueError(f"{path} has header {reader.fieldnames}, expected {expected}")
                for row in reader:
                    rows.append(
                        Keyframe(
                            n=int(row["n"]),
                            pts_time=float(row["pts_time"]),
                            fps=float(row["fps"]),
                            frame_idx=int(row["frame_idx"]),
                        )
                    )
            if not rows:
                raise ValueError(f"{path} has no keyframes")
            if any(rows[i].pts_time > rows[i + 1].pts_time for i in range(len(rows) - 1)):
                raise ValueError(f"{path} has non-monotonic pts_time")
            if video_id in by_video:
                raise ValueError(f"Duplicate map for video {video_id}")
            by_video[video_id] = rows
        return cls(by_video)

    def nearest(self, video_id: str, time_s: float) -> Keyframe:
        rows = self.by_video.get(video_id)
        if rows is None:
            raise KeyError(f"Missing keyframe map for video {video_id}")

        times = self._times[video_id]
        pos = bisect.bisect_left(times, time_s)
        if pos <= 0:
            return rows[0]
        if pos >= len(rows):
            return rows[-1]

        prev_kf = rows[pos - 1]
        next_kf = rows[pos]
        prev_delta = abs(time_s - prev_kf.pts_time)
        next_delta = abs(next_kf.pts_time - time_s)
        return prev_kf if prev_delta <= next_delta else next_kf

    def iter_rows(self):
        for video_id in sorted(self.by_video):
            for keyframe in self.by_video[video_id]:
                yield video_id, keyframe


def make_keyframe_id(video_id: str, n: int) -> str:
    return f"{video_id}/{n:03d}"


def submit_category(video_id: str) -> str:
    return video_id.split("_", 1)[0]


def make_submit_keyframe_id(video_id: str, n: int) -> str:
    return f"{submit_category(video_id)}/{video_id}/{n:03d}"


def make_time_span_id(video_id: str, start: float, end: float, prefix: str) -> str:
    start_ms = int(round(start * 1000))
    end_ms = int(round(end * 1000))
    return f"{video_id}_{prefix}_{start_ms:09d}_{end_ms:09d}"


def keyframe_fields(video_id: str, label: str, keyframe: Keyframe) -> dict[str, Any]:
    return {
        f"{label}_keyframe_id": make_keyframe_id(video_id, keyframe.n),
        f"{label}_submit_keyframe_id": make_submit_keyframe_id(video_id, keyframe.n),
        f"{label}_keyframe_n": keyframe.n,
        f"{label}_keyframe_pts_time": keyframe.pts_time,
        f"{label}_keyframe_frame_idx": keyframe.frame_idx,
    }


def map_time_span(index: KeyframeIndex, video_id: str, start: float, end: float) -> dict[str, Any]:
    center = (start + end) / 2.0
    mapped: dict[str, Any] = {
        "duration": round(end - start, 6),
        "center_time": round(center, 6),
    }
    for label, time_s in (("start", start), ("center", center), ("end", end)):
        mapped.update(keyframe_fields(video_id, label, index.nearest(video_id, time_s)))
    mapped["keyframe_id"] = mapped["center_keyframe_id"]
    mapped["submit_keyframe_id"] = mapped["center_submit_keyframe_id"]
    mapped["keyframe_n"] = mapped["center_keyframe_n"]
    mapped["keyframe_pts_time"] = mapped["center_keyframe_pts_time"]
    mapped["keyframe_frame_idx"] = mapped["center_keyframe_frame_idx"]
    return mapped


def confidence_bucket(score: float | None) -> str:
    if score is None:
        return "missing"
    if score < 0.40:
        return "low"
    if score < 0.50:
        return "mid"
    return "high"


def caption_quality(caption: str | None) -> str:
    if not caption:
        return "none"
    lowered = caption.lower()
    if VIETNAMESE_DIACRITIC_RE.search(caption):
        return "vietnamese_asr"
    if any(pattern in lowered for pattern in GENERIC_CAPTION_PATTERNS):
        return "generic"
    return "useful"


def segment_role(start: float, end: float) -> str:
    if start < 15.0:
        return "intro"
    if start < 90.0 and (end - start) >= 15.0:
        return "preview"
    return "body"


def json_dump_line(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def write_keyframe_map(index: KeyframeIndex, out_path: Path) -> int:
    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for video_id, keyframe in index.iter_rows():
            record = {
                "keyframe_id": make_keyframe_id(video_id, keyframe.n),
                "submit_keyframe_id": make_submit_keyframe_id(video_id, keyframe.n),
                "video_id": video_id,
                "category_hint": submit_category(video_id),
                "submit_category": submit_category(video_id),
                "keyframe_n": keyframe.n,
                "keyframe_name": f"{keyframe.n:03d}",
                "pts_time": keyframe.pts_time,
                "fps": keyframe.fps,
                "frame_idx": keyframe.frame_idx,
            }
            json_dump_line(out, record)
            count += 1
    return count


def write_audio_windows(index: KeyframeIndex, audio_dir: Path, out_path: Path) -> dict[str, int]:
    stats = {
        "files": 0,
        "records": 0,
        "caption_records": 0,
        "stoplist_tag_records": 0,
    }
    with out_path.open("w", encoding="utf-8") as out:
        for path in sorted(audio_dir.glob("*.audio.json")):
            stats["files"] += 1
            rows = json.loads(path.read_text(encoding="utf-8"))
            for i, row in enumerate(rows):
                video_id = row["video_id"]
                start = float(row["start"])
                end = float(row["end"])
                tags = row.get("tags") or []
                tag_labels = [tag.get("label") for tag in tags]
                tag_scores = [tag.get("score") for tag in tags]
                top1 = tags[0] if tags else {}
                caption = row.get("caption")
                stoplist_hit = any(label in AUDIO_TAG_STOPLIST for label in tag_labels)

                record = {
                    "window_id": make_time_span_id(video_id, start, end, f"a{i:06d}"),
                    "video_id": video_id,
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
                record.update(map_time_span(index, video_id, start, end))
                json_dump_line(out, record)
                stats["records"] += 1
                if caption:
                    stats["caption_records"] += 1
                if stoplist_hit:
                    stats["stoplist_tag_records"] += 1
    return stats


def write_speech_segments(index: KeyframeIndex, speech_dir: Path, out_path: Path) -> dict[str, int]:
    stats = {
        "files": 0,
        "records": 0,
        "low_confidence_records": 0,
        "mid_confidence_records": 0,
        "missing_confidence_records": 0,
    }
    with out_path.open("w", encoding="utf-8") as out:
        for path in sorted(speech_dir.glob("*.speech.json")):
            stats["files"] += 1
            rows = json.loads(path.read_text(encoding="utf-8"))
            for i, row in enumerate(rows):
                video_id = row["video_id"]
                start = float(row["start"])
                end = float(row["end"])
                score = row.get("avg_word_score")
                bucket = confidence_bucket(score)
                words = row.get("words") or []

                record = {
                    "segment_id": make_time_span_id(video_id, start, end, f"s{i:06d}"),
                    "video_id": video_id,
                    "start": start,
                    "end": end,
                    "text": row.get("text", ""),
                    "avg_word_score": score,
                    "confidence_bucket": bucket,
                    "segment_role": segment_role(start, end),
                    "word_count": len(words),
                    "avg_logprob": row.get("avg_logprob"),
                    "no_speech_prob": row.get("no_speech_prob"),
                    "source_segment_idx": i,
                    "source_speech_file": str(path),
                }
                record.update(map_time_span(index, video_id, start, end))
                json_dump_line(out, record)
                stats["records"] += 1
                if bucket == "low":
                    stats["low_confidence_records"] += 1
                elif bucket == "mid":
                    stats["mid_confidence_records"] += 1
                elif bucket == "missing":
                    stats["missing_confidence_records"] += 1
    return stats


def write_summary(out_path: Path, summary: dict[str, Any]) -> None:
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_staging(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index = KeyframeIndex.from_csv_root(args.map_dir)
    keyframe_count = write_keyframe_map(index, out_dir / "keyframe_map.jsonl")
    audio_stats = write_audio_windows(index, args.audio_dir, out_dir / "audio_windows_mapped.jsonl")
    speech_stats = write_speech_segments(index, args.speech_dir, out_dir / "speech_segments_mapped.jsonl")

    summary = {
        "map_videos": len(index.by_video),
        "keyframes": keyframe_count,
        "audio": audio_stats,
        "speech": speech_stats,
        "outputs": {
            "keyframe_map": str(out_dir / "keyframe_map.jsonl"),
            "audio_windows": str(out_dir / "audio_windows_mapped.jsonl"),
            "speech_segments": str(out_dir / "speech_segments_mapped.jsonl"),
        },
    }
    write_summary(out_dir / "mapping_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, default=Path("map-keyframes-s1-s2"))
    parser.add_argument("--audio-dir", type=Path, default=Path("results_audio_event/audio_out"))
    parser.add_argument("--speech-dir", type=Path, default=Path("speech_out"))
    parser.add_argument("--out-dir", type=Path, default=Path("elastic_staging"))
    return parser.parse_args()


def main() -> None:
    summary = build_staging(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
