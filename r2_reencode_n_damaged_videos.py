#!/usr/bin/env python3
"""Re-encode the N traffic-camera videos whose own recordings are damaged.

`r2_publish_n_web_videos.py` fixed the malformed SEI of every N video, but 103
recordings also lost reference frames while the cameras were writing them
(`mmco: unref short failure`). Chrome conceals most of that damage, yet on some
keyframes it stops the whole pipeline (N027-V003 at 4:34), so those videos need
a clean bitstream, not a patched one.

Per video: decode the BTC original leniently (FFmpeg conceals the damage, as
the keyframe extraction's PyAV decode did), encode H.264 with NVENC keeping
every frame's original timestamp (`-copyts`, `-fps_mode passthrough`, the
source time base), then require:

* a strict decode of the output to be clean and to cover every packet;
* the output's frame times to equal the lenient decode of the source (≤ 2 ms);
* every keyframe of that video in the pinned batch-2 registry to have an output
  frame at its `pts_time` (≤ 2 ms), so seeking to a keyframe lands on it.

Copies go to a new key, `Videos_Web_v2/Videos_Nxxx/<id>.mp4`: the SEI-patched
copies under `Videos_Web/` are served `immutable` for a year, so overwriting
them would stay hidden behind the CDN. After each upload the video id is added
to the backend's `video_overrides.json`, which the media URL builder re-reads
on change — each video switches over as soon as it is published.

Run: ``uv run --with boto3 python r2_reencode_n_damaged_videos.py --videos-file <json> --work-dir <tmp>``
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from batch2_keyframes import DEFAULT_REGISTRY_ROOT, PROFILES, load_registry
from elastic_upload import read_r2_credentials
from r2_publish_n_web_videos import BUCKET, count_frames

TARGET_ROOT = "Videos_Web_v2"
PIPELINE = "aic2026-n-nvenc-reencode-v1"
DEFAULT_OVERRIDES = Path("backend/app/video_overrides.json")
TIME_TOLERANCE = 0.002  # seconds
_overrides_lock = threading.Lock()


def source_key(video_id: str) -> str:
    return f"Videos/Videos_{video_id.split('-')[0]}/{video_id}.mp4"


def target_key(video_id: str) -> str:
    return f"{TARGET_ROOT}/Videos_{video_id.split('-')[0]}/{video_id}.mp4"


def probe(path: Path, *args: str) -> str:
    return subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0", *args, "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout


def frame_times(path: Path) -> list[float]:
    """Times of the frames a lenient decode actually produces, in order."""
    out = probe(path, "-show_entries", "frame=best_effort_timestamp_time")
    return [float(v) for v in (line.strip().rstrip(",") for line in out.splitlines()) if v not in ("", "N/A")]


def packet_times(path: Path) -> list[float]:
    out = probe(path, "-show_entries", "packet=pts_time")
    return sorted(float(v) for v in (line.strip().rstrip(",") for line in out.splitlines()) if v not in ("", "N/A"))


def reencode(src: Path, out: Path) -> None:
    time_base = probe(src, "-show_entries", "stream=time_base").strip()
    timescale = time_base.split("/")[1] if "/" in time_base else "90000"
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", "-copyts", "-i", str(src),
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-profile:v", "high",
            "-rc", "vbr", "-cq", "25", "-b:v", "0", "-maxrate", "6M", "-bufsize", "12M",
            "-g", "25", "-bf", "0", "-pix_fmt", "yuv420p",
            "-fps_mode", "passthrough", "-enc_time_base", "demux", "-video_track_timescale", timescale,
            "-movflags", "+faststart", str(out),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"{src.name}: ffmpeg failed: {result.stderr.strip()[-300:]}")


def max_gap(expected: list[float], actual: list[float]) -> float:
    """Largest distance from an expected time to the nearest actual one."""
    worst = 0.0
    for value in expected:
        at = bisect.bisect_left(actual, value)
        nearest = min(
            (abs(actual[i] - value) for i in (at - 1, at) if 0 <= i < len(actual)), default=float("inf")
        )
        worst = max(worst, nearest)
    return worst


def write_overrides(path: Path, video_ids: set[str]) -> None:
    data = {
        "about": (
            "Videos republished under their own R2 prefix (see r2_reencode_n_damaged_videos.py); "
            "MediaUrlBuilder.video_url serves these ids from that root."
        ),
        "roots": {TARGET_ROOT: sorted(video_ids)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def add_override(path: Path, video_id: str) -> None:
    with _overrides_lock:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        write_overrides(path, set((data.get("roots") or {}).get(TARGET_ROOT, [])) | {video_id})


def published_copies(s3: Any) -> set[str]:
    """Video ids whose `Videos_Web_v2/` copy was made from the current original.

    This is the source of truth when several machines re-encode in parallel: a
    copy another machine uploaded counts as soon as it is on R2.
    """
    keys = [
        item["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{TARGET_ROOT}/")
        for item in page.get("Contents", [])
        if item["Key"].endswith(".mp4")
    ]

    def valid(key: str) -> str | None:
        video_id = key.rsplit("/", 1)[1][:-4]
        meta = s3.head_object(Bucket=BUCKET, Key=key).get("Metadata", {})
        source = s3.head_object(Bucket=BUCKET, Key=source_key(video_id))
        return video_id if meta.get("pipeline") == PIPELINE and meta.get("source-etag") == source["ETag"] else None

    with ThreadPoolExecutor(16) as pool:
        return {video for video in pool.map(valid, keys) if video}


def process(s3: Any, video_id: str, keyframes: dict[str, list[float]], work_dir: Path, overrides: Path) -> dict[str, Any]:
    source = s3.head_object(Bucket=BUCKET, Key=source_key(video_id))
    try:
        existing = s3.head_object(Bucket=BUCKET, Key=target_key(video_id))
    except s3.exceptions.ClientError:
        existing = None
    if existing and existing["Metadata"].get("source-etag") == source["ETag"] and \
            existing["Metadata"].get("pipeline") == PIPELINE:
        add_override(overrides, video_id)
        return {"video_id": video_id, "status": "skipped"}

    started = time.monotonic()
    src, out = work_dir / f"{video_id}.src.mp4", work_dir / f"{video_id}.mp4"
    try:
        s3.download_file(BUCKET, source_key(video_id), str(src))
        if src.stat().st_size != source["ContentLength"]:
            raise ValueError(f"{video_id}: downloaded size differs from the source")
        expected = frame_times(src)
        reencode(src, out)
        actual = packet_times(out)
        if len(actual) != len(expected):
            raise ValueError(f"{video_id}: output has {len(actual)} frames, lenient source decode {len(expected)}")
        timeline_gap = max_gap(expected, actual)
        keyframe_gap = max_gap(keyframes.get(video_id, []), actual)
        if timeline_gap > TIME_TOLERANCE or keyframe_gap > TIME_TOLERANCE:
            raise ValueError(f"{video_id}: timestamps moved (timeline {timeline_gap:.4f}s, keyframes {keyframe_gap:.4f}s)")
        strict, result = count_frames(out, strict=True)
        if result.returncode != 0 or result.stderr.strip() or strict != len(actual):
            raise ValueError(f"{video_id}: output strict decode {strict}/{len(actual)}: {result.stderr.strip()[:200]}")
        s3.upload_file(
            str(out), BUCKET, target_key(video_id),
            ExtraArgs={
                "ContentType": "video/mp4",
                "CacheControl": "public, max-age=31536000, immutable",
                "Metadata": {
                    "pipeline": PIPELINE,
                    "source-key": source_key(video_id),
                    "source-etag": source["ETag"],
                    "source-size": str(source["ContentLength"]),
                    "video-frames": str(len(actual)),
                    "keyframes-checked": str(len(keyframes.get(video_id, []))),
                },
            },
        )
        uploaded = s3.head_object(Bucket=BUCKET, Key=target_key(video_id))
        if uploaded["ContentLength"] != out.stat().st_size:
            raise ValueError(f"{video_id}: uploaded size differs from the encoded file")
        add_override(overrides, video_id)
        return {
            "video_id": video_id, "status": "published", "key": target_key(video_id),
            "frames": len(actual), "keyframes_checked": len(keyframes.get(video_id, [])),
            "max_keyframe_gap_ms": round(keyframe_gap * 1000, 3),
            "mb": round(out.stat().st_size / 1e6, 1), "seconds": round(time.monotonic() - started, 1),
        }
    finally:
        src.unlink(missing_ok=True)
        out.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-file", type=Path, help='JSON with {"damaged": [video ids]}')
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    parser.add_argument("--reverse", action="store_true", help="Work from the end of the list (second machine).")
    parser.add_argument(
        "--sync-overrides", action="store_true",
        help="Only rewrite the overrides file from the copies on R2 (no encoding).",
    )
    parser.add_argument("--credentials-file", type=Path, default=Path("CloudflareR2/cloudflareR2_api.txt"))
    parser.add_argument("--overrides-file", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()) / "aic26_n_reencode")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--report", type=Path, default=Path("r2_reencode_n_damaged_videos_report.json"))
    args = parser.parse_args()

    import boto3  # noqa: PLC0415 — uv run --with boto3

    creds = read_r2_credentials(args.credentials_file)
    s3 = boto3.client("s3", endpoint_url=creds["endpoint"], aws_access_key_id=creds["access_key"],
                      aws_secret_access_key=creds["secret_key"], region_name="auto")
    if args.sync_overrides:
        published = published_copies(s3)
        with _overrides_lock:
            write_overrides(args.overrides_file, published)
        print(f"{args.overrides_file}: {len(published)} videos served from {TARGET_ROOT}/", flush=True)
        return 0
    if args.videos_file is None:
        parser.error("--videos-file is required unless --sync-overrides")
    videos = json.loads(args.videos_file.read_text(encoding="utf-8"))["damaged"]
    if args.only:
        videos = [video for video in videos if video in set(args.only)]
    if args.reverse:
        videos = videos[::-1]
    keyframes: dict[str, list[float]] = {}
    for row in load_registry(PROFILES["N"], args.registry_root):
        keyframes.setdefault(row.video_id, []).append(row.pts_time)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(videos)} damaged N videos -> s3://{BUCKET}/{TARGET_ROOT}/", flush=True)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(process, s3, video, keyframes, args.work_dir, args.overrides_file): video for video in videos}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 — one bad video must not stop the rest
                result = {"video_id": futures[future], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            print(f"[{len(results)}/{len(videos)}] {json.dumps(result)}", flush=True)

    summary = Counter(result["status"] for result in results)
    args.report.write_text(json.dumps({"finished_at": datetime.now(UTC).isoformat(), "pipeline": PIPELINE,
                                       "summary": summary, "results": sorted(results, key=lambda r: r["video_id"])},
                                      indent=2) + "\n", encoding="utf-8")
    print(f"DONE {dict(summary)} (report: {args.report})", flush=True)
    return 1 if summary.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
