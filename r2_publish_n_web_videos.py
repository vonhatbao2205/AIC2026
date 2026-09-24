#!/usr/bin/env python3
"""Publish browser-playable copies of the batch-2 traffic-camera (N) videos on R2.

The N recordings were remuxed from the cameras' .mov files and still carry the
cameras' SEI messages, whose ``payload_size`` overruns their NAL unit. FFmpeg's
lenient default decoder skips them, but Chrome rejects the stream
(``PIPELINE_ERROR_DECODE`` on the third packet) and the console shows a black
player. Stripping SEI with a bitstream filter fails for the same reason: the
filter must parse the SEI to drop it.

This script instead overwrites every SEI NAL unit with a filler-data NAL unit
(type 12, ignored by every decoder) of exactly the same length. Nothing the
container describes moves — sample sizes, offsets, timestamps and every decoded
frame stay identical, so ``frame_idx``/``pts_time`` in the keyframe maps remain
valid — and the stream decodes under strict error detection.

Originals are never touched. Each copy goes to a new key that mirrors the layout:

    aic26-media/Videos/Videos_N001/N001-V001.mp4      (BTC original, unchanged)
    aic26-media/Videos_Web/Videos_N001/N001-V001.mp4  (this script's output)

A new key rather than an in-place overwrite, because the video host serves
``Cache-Control: public, max-age=31536000, immutable``: an overwritten object
would stay hidden behind the cached broken bytes, at the CDN and in browsers.

Per video: download, check size, record the packet table, patch, require the
packet table to be byte-identical and strict decoding to lose no more frames
than the lenient decoder does with no SEI complaint left (an unpatched recording
decodes about a third of them; what both lose is the cameras' own frame damage,
which Chrome conceals), upload with
provenance metadata, then HEAD the copy. A copy whose ``source-etag`` matches
the current original is skipped, so rerunning the command resumes.

Run: ``uv run --with boto3 python r2_publish_n_web_videos.py --work-dir <tmp>``
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from elastic_upload import read_r2_credentials

BUCKET = "aic26-media"
SOURCE_PREFIX = "Videos/"
TARGET_PREFIX = "Videos_Web/"
SOURCE_KEY = re.compile(r"Videos/Videos_(N\d{3})/(N\d{3})-V\d{3}\.mp4")
PIPELINE = "aic2026-n-sei-filler-v1"
NAL_LENGTH_SIZE = 4  # avcC lengthSizeMinusOne = 3 in every N recording
SEI, FILLER = 6, 12


def target_key(source_key: str) -> str:
    return TARGET_PREFIX + source_key[len(SOURCE_PREFIX):]


def packet_table(path: Path) -> list[list[str]]:
    """Every video packet's pts, dts, duration, size, flags and file position."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "packet=pts,dts,duration,size,pos,flags", "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    return list(csv.reader(out.splitlines()))


def replace_sei_with_filler(path: Path, packets: list[list[str]]) -> Counter[str]:
    """Rewrite SEI NAL units as same-length filler, in place; returns NAL type counts."""
    stats: Counter[str] = Counter()
    # ffprobe prints the requested fields in its own order: pts, dts, duration, size, pos, flags.
    with path.open("r+b") as handle:
        for _, _, _, size_text, pos_text, _ in packets:
            size, pos = int(size_text), int(pos_text)
            handle.seek(pos)
            data = bytearray(handle.read(size))
            offset, changed = 0, False
            while offset < size:
                length = int.from_bytes(data[offset:offset + NAL_LENGTH_SIZE], "big")
                offset += NAL_LENGTH_SIZE
                if length <= 1 or offset + length > size:
                    raise ValueError(f"{path.name}: bad NAL length {length} at file offset {pos + offset}")
                nal_type = data[offset] & 0x1F
                stats[f"nal_{nal_type}"] += 1
                if nal_type == SEI:
                    # forbidden_zero_bit 0, nal_ref_idc 0, type 12; ff_bytes; rbsp_trailing_bits.
                    data[offset] = FILLER
                    data[offset + 1:offset + length - 1] = b"\xff" * (length - 2)
                    data[offset + length - 1] = 0x80
                    changed = True
                offset += length
            if changed:
                handle.seek(pos)
                handle.write(data)
                stats["sei_replaced"] += 1
    return stats


def count_frames(path: Path, *, strict: bool) -> tuple[int, subprocess.CompletedProcess[str]]:
    """Decode the whole stream and count frames; ffprobe does it without a muxer.

    (An ffmpeg `-f null` run also reports the cameras' repeated DTS values, a
    muxer complaint about the unchanged timeline, not a decoding error.)
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", *(["-err_detect", "explode"] if strict else []),
         "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return int(result.stdout.strip() or 0), result


def strict_decode(path: Path, expected_frames: int) -> int:
    """Strict decoding must lose nothing the recording had not already lost.

    Some cameras dropped reference frames while recording (`mmco: unref short
    failure`); even FFmpeg's lenient decoder loses those frames from the original
    (N007-V003: 6,319 of 6,610), and Chrome conceals them and keeps playing. So
    the patched copy is compared with its own lenient decode, not with the packet
    count. A recording whose SEI is still broken decodes ~30 % under strict rules.
    """
    strict, result = count_frames(path, strict=True)
    lenient, _ = count_frames(path, strict=False)
    # Strict decoding may drop a few more of the damaged frames than the lenient
    # decoder conceals (N027-V001: 19 of 14,723); 1 % stays far from the ~70 %
    # a recording loses while its SEI is still broken.
    tolerance = max(2, expected_frames // 100)
    if result.returncode != 0 or "SEI" in result.stderr or strict < lenient - tolerance:
        raise ValueError(
            f"{path.name}: strict decode gave {strict} frames, lenient {lenient}, packets "
            f"{expected_frames}: {result.stderr.strip()[:300]}"
        )
    return strict


def process(s3: Any, source_key: str, work_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    video_id = source_key.rsplit("/", 1)[1][:-4]
    source = s3.head_object(Bucket=BUCKET, Key=source_key)
    destination = target_key(source_key)
    try:
        existing = s3.head_object(Bucket=BUCKET, Key=destination)
    except s3.exceptions.ClientError:
        existing = None
    if existing and existing.get("Metadata", {}).get("source-etag") == source["ETag"] and \
            existing["Metadata"].get("pipeline") == PIPELINE:
        return {"video_id": video_id, "status": "skipped"}
    if dry_run:
        return {"video_id": video_id, "status": "planned", "size": source["ContentLength"]}

    started = time.monotonic()
    local = work_dir / f"{video_id}.mp4"
    try:
        s3.download_file(BUCKET, source_key, str(local))
        if local.stat().st_size != source["ContentLength"]:
            raise ValueError(f"{video_id}: downloaded {local.stat().st_size} bytes, expected {source['ContentLength']}")
        before = packet_table(local)
        stats = replace_sei_with_filler(local, before)
        if packet_table(local) != before:
            raise ValueError(f"{video_id}: packet table changed after patching")
        strict_frames = strict_decode(local, len(before))
        s3.upload_file(
            str(local), BUCKET, destination,
            ExtraArgs={
                "ContentType": source.get("ContentType", "video/mp4"),
                "CacheControl": source.get("CacheControl", "public, max-age=31536000, immutable"),
                "Metadata": {
                    "pipeline": PIPELINE,
                    "source-key": source_key,
                    "source-etag": source["ETag"],
                    "source-size": str(source["ContentLength"]),
                    "video-frames": str(len(before)),
                    "sei-units-replaced": str(stats["sei_replaced"]),
                    "strict-decoded-frames": str(strict_frames),
                },
            },
        )
        uploaded = s3.head_object(Bucket=BUCKET, Key=destination)
        if uploaded["ContentLength"] != source["ContentLength"]:
            raise ValueError(f"{video_id}: uploaded size {uploaded['ContentLength']} differs from source")
        return {
            "video_id": video_id,
            "status": "published",
            "key": destination,
            "frames": len(before),
            "strict_decoded_frames": strict_frames,
            "sei_replaced": stats["sei_replaced"],
            "seconds": round(time.monotonic() - started, 1),
        }
    finally:
        local.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--credentials-file", type=Path, default=Path("CloudflareR2/cloudflareR2_api.txt"))
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()) / "aic26_n_web_videos")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", nargs="*", help="Process only these video ids (e.g. N070-V002).")
    parser.add_argument("--report", type=Path, default=Path("r2_publish_n_web_videos_report.json"))
    parser.add_argument("--dry-run", action="store_true", help="List what would be published; no download or write.")
    args = parser.parse_args()

    import boto3  # noqa: PLC0415 — only this script needs it (uv run --with boto3)

    creds = read_r2_credentials(args.credentials_file)
    s3 = boto3.client(
        "s3", endpoint_url=creds["endpoint"], aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"], region_name="auto",
    )
    keys = [
        item["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="Videos/Videos_N")
        for item in page.get("Contents", [])
        if SOURCE_KEY.fullmatch(item["Key"])
    ]
    if args.only:
        wanted = set(args.only)
        keys = [key for key in keys if key.rsplit("/", 1)[1][:-4] in wanted]
    print(f"{len(keys)} N videos -> s3://{BUCKET}/{TARGET_PREFIX}", flush=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(process, s3, key, args.work_dir, dry_run=args.dry_run): key for key in keys}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 — one bad video must not stop the rest
                video_id = futures[future].rsplit("/", 1)[1][:-4]
                result = {"video_id": video_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            print(f"[{len(results)}/{len(keys)}] {json.dumps(result)}", flush=True)

    summary = Counter(result["status"] for result in results)
    args.report.write_text(
        json.dumps({"finished_at": datetime.now(UTC).isoformat(), "pipeline": PIPELINE, "summary": summary,
                    "results": sorted(results, key=lambda r: r["video_id"])}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"DONE {dict(summary)} (report: {args.report})", flush=True)
    return 1 if summary.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
