#!/usr/bin/env python3
"""Upload locally extracted InfoShot++ keyframes to the AIC26 HF media bucket.

This is the local-keyframe counterpart of ``AIC2026_HF_Media_Migration_Colab``.
It projects the canonical local JPEG name ``f{frame_idx:08d}.jpg`` to the media
URL contract ``{keyframe_n:03d}.jpg`` by joining the final map CSV.  It never
assumes that frame_idx and keyframe_n are interchangeable.

Remote progress and commit objects intentionally use the same schema/binding as
the Colab notebook.  A local run can therefore upload keyframes, after which a
single Colab coordinator can upload videos and publish the final manifests.

The source corpus is read-only.  No local keyframe and no remote object is ever
deleted by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote


PIPELINE_VERSION = "aic26-hf-media-projection-v1"
DEST_LAYOUT_VERSION = "r2-compatible-keyframe-n-alias-v1"
EXPECTED_VIDEOS = 873
EXPECTED_KEYFRAMES = 1_339_055
ALL_CATEGORIES = tuple(f"L{number:02d}" for number in range(21, 31))
DEFAULT_SOURCE_ROOT = Path("/home/bao/Projects/ExtractKeyframe/infoshootpp")
DEFAULT_SOURCE_STATE_DIR = Path(
    "/home/bao/Projects/ExtractKeyframe/.hf_bucket_upload_state_DATA-AIC-Keyframe"
)
DEFAULT_DEST_BUCKET = "Baonenha1/aic26-media"
STATE_SCHEMA = 1


class UploadError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameMapRow:
    n: int
    pts_time: float
    fps: float
    frame_idx: int

    @property
    def source_frame_name(self) -> str:
        return f"f{self.frame_idx:08d}.jpg"

    @property
    def media_frame_name(self) -> str:
        # Minimum width only: n=5152 remains 5152.jpg.
        return f"{self.n:03d}.jpg"


@dataclass(frozen=True)
class Corpus:
    source_root: Path
    source_state_dir: Path
    source_bucket: str
    source_prefix: str
    source_dataset_manifest_sha256: str
    shards: dict[str, dict[str, Any]]
    map_paths: dict[str, Path]
    category_videos: dict[str, tuple[str, ...]]
    category_frames: dict[str, int]


@dataclass(frozen=True)
class VideoInventory:
    category: str
    video_id: str
    mapping: tuple[FrameMapRow, ...]
    local_files: tuple[Path, ...]
    file_sizes: tuple[int, ...]
    total_bytes: int
    inventory_sha256: str


def log(message: str) -> None:
    print(f"[HF local media] {message}", flush=True)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UploadError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise UploadError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.part")
    with partial.open("wb") as output:
        output.write(stable_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, path)


def natural_video_key(video_id: str) -> tuple[str, int, int]:
    match = re.fullmatch(r"([A-Z]+)(\d+)_V(\d+)", video_id.upper())
    return (
        (match.group(1), int(match.group(2)), int(match.group(3)))
        if match
        else (video_id, 0, 0)
    )


def parse_categories(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = re.split(r"[,;\s]+", value.strip()) if value.strip() else []
    else:
        tokens = [str(item).strip() for item in value]
    selected = {token.upper() for token in tokens if token}
    unknown = sorted(selected - set(ALL_CATEGORIES))
    if unknown:
        raise UploadError(f"Unknown categories: {unknown}")
    return tuple(category for category in ALL_CATEGORIES if category in selected)


def remote_join(*parts: str) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def keyframe_media_key(category: str, video_id: str, n: int) -> str:
    return f"Keyframes/Keyframes_{category}/{video_id}/{n:03d}.jpg"


def dest_key(dest_prefix: str, relative: str) -> str:
    return remote_join(dest_prefix, relative)


def keyframe_progress_path(dest_prefix: str, category: str) -> str:
    return dest_key(dest_prefix, f"manifest/_migration/progress/keyframes/{category}.json")


def keyframe_video_commit_path(dest_prefix: str, category: str, video_id: str) -> str:
    return dest_key(
        dest_prefix,
        f"manifest/_migration/commits/keyframes/{category}/{video_id}.json",
    )


def keyframe_category_commit_path(dest_prefix: str, category: str) -> str:
    return dest_key(dest_prefix, f"manifest/_migration/commits/keyframes/{category}.json")


def load_map_rows(path: Path) -> tuple[FrameMapRow, ...]:
    rows: list[FrameMapRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["n", "pts_time", "fps", "frame_idx"]:
            raise UploadError(f"Unexpected map header in {path}: {reader.fieldnames}")
        previous_frame_idx = -1
        for expected_n, raw in enumerate(reader, start=1):
            try:
                row = FrameMapRow(
                    n=int(raw["n"]),
                    pts_time=float(raw["pts_time"]),
                    fps=float(raw["fps"]),
                    frame_idx=int(raw["frame_idx"]),
                )
            except (TypeError, ValueError) as exc:
                raise UploadError(f"Invalid map value in {path} at row {expected_n + 1}") from exc
            if row.n != expected_n:
                raise UploadError(f"Non-contiguous keyframe n in {path}: {row.n} != {expected_n}")
            if row.frame_idx <= previous_frame_idx:
                raise UploadError(f"frame_idx is not strictly increasing in {path}")
            if not math.isfinite(row.pts_time) or not math.isfinite(row.fps) or row.fps <= 0:
                raise UploadError(f"Invalid pts_time/fps in {path}")
            # Tiny L25 pts_time regressions are valid. frame_idx is canonical.
            rows.append(row)
            previous_frame_idx = row.frame_idx
    if not rows:
        raise UploadError(f"Empty map CSV: {path}")
    return tuple(rows)


def audit_corpus(source_root: Path, source_state_dir: Path) -> Corpus:
    source_root = source_root.expanduser().resolve()
    source_state_dir = source_state_dir.expanduser().resolve()
    required = [
        source_root / "keyframes",
        source_root / "map-keyframes",
        source_root / "run_manifest.json",
        source_root / "run_summary.csv",
        source_state_dir / "dataset_manifest.json",
        source_state_dir / "keyframe_category_shards.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise UploadError("Missing source paths:\n" + "\n".join(missing))

    dataset_manifest_path = source_state_dir / "dataset_manifest.json"
    dataset_manifest_bytes = dataset_manifest_path.read_bytes()
    dataset = json.loads(dataset_manifest_bytes)
    if int(dataset.get("videos", -1)) != EXPECTED_VIDEOS:
        raise UploadError(f"dataset_manifest.json does not declare {EXPECTED_VIDEOS} videos")
    if int(dataset.get("keyframes", -1)) != EXPECTED_KEYFRAMES:
        raise UploadError(
            f"dataset_manifest.json does not declare {EXPECTED_KEYFRAMES:,} keyframes"
        )
    source_bucket = str(dataset.get("bucket_id") or "").strip().strip("/")
    source_prefix = str(dataset.get("remote_prefix") or "").strip().strip("/")
    if not source_bucket or not source_prefix:
        raise UploadError("Source bucket/prefix missing from dataset_manifest.json")

    run_manifest = load_json(source_root / "run_manifest.json")
    if int(run_manifest.get("video_count", -1)) != EXPECTED_VIDEOS:
        raise UploadError("run_manifest video_count mismatch")
    if int(run_manifest.get("total_keyframes", -1)) != EXPECTED_KEYFRAMES:
        raise UploadError("run_manifest total_keyframes mismatch")
    corpus_audit = dataset.get("corpus_audit") or {}
    if sha256_file(source_root / "run_manifest.json") != corpus_audit.get("run_manifest_sha256"):
        raise UploadError("Local run_manifest SHA-256 differs from the source bucket manifest")
    if sha256_file(source_root / "run_summary.csv") != corpus_audit.get("run_summary_sha256"):
        raise UploadError("Local run_summary SHA-256 differs from the source bucket manifest")

    shards: dict[str, dict[str, Any]] = {}
    with (source_state_dir / "keyframe_category_shards.jsonl").open(
        "r", encoding="utf-8"
    ) as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            category = str(item.get("category") or "")
            if category in shards:
                raise UploadError(f"Duplicate category shard: {category}")
            shards[category] = item
    if tuple(sorted(shards)) != ALL_CATEGORIES:
        raise UploadError(f"Unexpected shard categories: {tuple(sorted(shards))}")

    with (source_root / "run_summary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        summary_rows = list(csv.DictReader(stream))
    summary_by_video = {row["video_id"]: row for row in summary_rows}
    if len(summary_rows) != EXPECTED_VIDEOS or len(summary_by_video) != EXPECTED_VIDEOS:
        raise UploadError("run_summary.csv must contain 873 unique video IDs")

    map_paths: dict[str, Path] = {}
    category_videos: dict[str, tuple[str, ...]] = {}
    category_frames: dict[str, int] = {}
    for category in ALL_CATEGORIES:
        paths = sorted(
            (source_root / "map-keyframes" / category).glob("*.csv"),
            key=lambda path: natural_video_key(path.stem),
        )
        videos: list[str] = []
        frames = 0
        for path in paths:
            video_id = path.stem
            if video_id in map_paths:
                raise UploadError(f"Duplicate map video_id: {video_id}")
            if video_id.split("_", 1)[0] != category:
                raise UploadError(f"Map category mismatch: {path}")
            rows = load_map_rows(path)
            summary = summary_by_video.get(video_id)
            if summary is None or int(summary["final_frames"]) != len(rows):
                raise UploadError(f"Map/run_summary mismatch: {video_id}")
            map_paths[video_id] = path
            videos.append(video_id)
            frames += len(rows)
        shard = shards[category]
        if int(shard.get("video_count", -1)) != len(videos):
            raise UploadError(f"Shard video count mismatch: {category}")
        if int(shard.get("frame_count", -1)) != frames:
            raise UploadError(f"Shard frame count mismatch: {category}")
        category_videos[category] = tuple(videos)
        category_frames[category] = frames

    if len(map_paths) != EXPECTED_VIDEOS or sum(category_frames.values()) != EXPECTED_KEYFRAMES:
        raise UploadError("Final map inventory does not contain exactly 873 videos/1,339,055 frames")
    if set(map_paths) != set(summary_by_video):
        raise UploadError("Map video IDs differ from run_summary video IDs")
    return Corpus(
        source_root=source_root,
        source_state_dir=source_state_dir,
        source_bucket=source_bucket,
        source_prefix=source_prefix,
        source_dataset_manifest_sha256=sha256_bytes(dataset_manifest_bytes),
        shards=shards,
        map_paths=map_paths,
        category_videos=category_videos,
        category_frames=category_frames,
    )


def inventory_digest(mapping: Sequence[FrameMapRow], file_sizes: Sequence[int]) -> str:
    if len(mapping) != len(file_sizes):
        raise UploadError("Cannot hash inventory with different mapping/size lengths")
    digest = hashlib.sha256()
    for row, size in zip(mapping, file_sizes):
        digest.update(
            f"{row.n}\t{row.frame_idx}\t{row.source_frame_name}\t"
            f"{row.media_frame_name}\t{int(size)}\n".encode()
        )
    return digest.hexdigest()


def build_video_inventory(
    corpus: Corpus,
    category: str,
    video_id: str,
    *,
    verify_jpeg_magic: bool,
) -> VideoInventory:
    mapping = load_map_rows(corpus.map_paths[video_id])
    source_dir = corpus.source_root / "keyframes" / category / video_id
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise UploadError(f"Missing or unsafe source directory: {source_dir}")
    expected_names = {row.source_frame_name for row in mapping}
    actual_names: set[str] = set()
    with os.scandir(source_dir) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise UploadError(f"Unexpected non-regular source entry: {entry.path}")
            if not entry.name.lower().endswith(".jpg"):
                raise UploadError(f"Unexpected non-JPEG source file: {entry.path}")
            actual_names.add(entry.name)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise UploadError(
            f"Map/source inventory mismatch for {video_id}: missing={missing[:5]}, extra={extra[:5]}"
        )

    local_files: list[Path] = []
    file_sizes: list[int] = []
    for row in mapping:
        path = source_dir / row.source_frame_name
        stat = path.stat(follow_symlinks=False)
        if stat.st_size <= 0:
            raise UploadError(f"Empty JPEG: {path}")
        if verify_jpeg_magic:
            with path.open("rb") as stream:
                if stream.read(2) != b"\xff\xd8":
                    raise UploadError(f"Invalid JPEG magic: {path}")
        local_files.append(path)
        file_sizes.append(int(stat.st_size))
    return VideoInventory(
        category=category,
        video_id=video_id,
        mapping=mapping,
        local_files=tuple(local_files),
        file_sizes=tuple(file_sizes),
        total_bytes=sum(file_sizes),
        inventory_sha256=inventory_digest(mapping, file_sizes),
    )


def read_token(token_file: Path | None) -> str:
    value = os.environ.get("HF_TOKEN", "").strip()
    if not value and token_file is not None and token_file.is_file():
        value = token_file.read_text(encoding="utf-8").strip()
    if not value:
        raise UploadError("HF token missing. Export HF_TOKEN or pass --token-file")
    if not value.startswith("hf_") or "\n" in value or "\r" in value:
        raise UploadError("HF token must be one valid hf_... token")
    return value


def is_non_retryable(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in message
        for marker in (
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "invalid token",
            "permission denied",
            "bucket not found",
        )
    )


def retry_call(
    label: str,
    operation: Callable[[], Any],
    *,
    max_retries: int,
    retry_base_seconds: float,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            if is_non_retryable(exc) or attempt >= max_retries:
                raise UploadError(f"Failed {label}: {type(exc).__name__}: {exc}") from exc
            delay = min(180.0, retry_base_seconds * (2 ** min(attempt, 6))) + random.random()
            log(
                f"RETRY {attempt + 1}/{max_retries} {label}: "
                f"{type(exc).__name__}: {exc}; wait {delay:.1f}s"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


class HfBucketRemote:
    def __init__(
        self,
        *,
        bucket: str,
        prefix: str,
        token: str,
        upload_batch_files: int,
        max_retries: int,
        retry_base_seconds: float,
    ) -> None:
        try:
            from huggingface_hub import (
                BucketFile,
                HfApi,
                HfFileSystem,
                batch_bucket_files,
            )
        except ImportError as exc:
            raise UploadError(
                "huggingface_hub with Storage Bucket support is missing. Install "
                "huggingface_hub[hf_xet]>=1.23,<2"
            ) from exc
        self.bucket = bucket
        self.prefix = prefix
        self.token = token
        self.upload_batch_files = upload_batch_files
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.api = HfApi(token=token)
        self.fs = HfFileSystem(token=token)
        self.batch_bucket_files = batch_bucket_files
        self.BucketFile = BucketFile

    def key(self, relative: str) -> str:
        return dest_key(self.prefix, relative)

    def get_file(self, path: str) -> Any | None:
        items = retry_call(
            f"stat {path}",
            lambda: list(self.api.get_bucket_paths_info(self.bucket, [path], token=self.token)),
            max_retries=self.max_retries,
            retry_base_seconds=self.retry_base_seconds,
        )
        return next((item for item in items if item.path == path), None)

    def read_json(self, path: str) -> dict[str, Any] | None:
        if self.get_file(path) is None:
            return None

        def operation() -> dict[str, Any]:
            with self.fs.open(f"hf://buckets/{self.bucket}/{path}", "rb") as stream:
                value = json.load(stream)
            if not isinstance(value, dict):
                raise TypeError(f"Remote JSON is not an object: {path}")
            return value

        return retry_call(
            f"read {path}",
            operation,
            max_retries=self.max_retries,
            retry_base_seconds=self.retry_base_seconds,
        )

    def upload_pairs(self, pairs: Sequence[tuple[str | Path | bytes, str]], *, label: str) -> None:
        for offset in range(0, len(pairs), self.upload_batch_files):
            chunk = list(pairs[offset : offset + self.upload_batch_files])
            retry_call(
                f"{label} files {offset + 1}-{offset + len(chunk)}",
                lambda chunk=chunk: self.batch_bucket_files(
                    self.bucket,
                    add=chunk,
                    token=self.token,
                ),
                max_retries=self.max_retries,
                retry_base_seconds=self.retry_base_seconds,
            )

    def upload_json(self, path: str, value: Any, *, label: str) -> None:
        payload = stable_json_bytes(value)
        self.upload_pairs([(payload, path)], label=label)
        info = self.get_file(path)
        if info is None or int(info.size) != len(payload):
            raise UploadError(f"Remote JSON size verification failed: {path}")

    def list_files(self, prefix: str) -> dict[str, int]:
        def operation() -> dict[str, int]:
            result: dict[str, int] = {}
            for item in self.api.list_bucket_tree(
                self.bucket,
                prefix=prefix,
                recursive=True,
                token=self.token,
            ):
                if isinstance(item, self.BucketFile):
                    result[item.path] = int(item.size)
            return result

        return retry_call(
            f"list {prefix}",
            operation,
            max_retries=self.max_retries,
            retry_base_seconds=self.retry_base_seconds,
        )

    def verify_exact(self, expected: dict[str, int], *, prefix: str, label: str) -> None:
        actual = self.list_files(prefix)
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        mismatch = sorted(
            path for path in set(expected) & set(actual) if expected[path] != actual[path]
        )
        if missing or extra or mismatch:
            raise UploadError(
                f"Remote inventory mismatch {label}: missing={missing[:5]}, "
                f"extra={extra[:5]}, size={mismatch[:5]}"
            )


def commit_binding(
    corpus: Corpus,
    dest_bucket: str,
    dest_prefix: str,
    category: str,
) -> dict[str, Any]:
    shard = corpus.shards[category]
    return {
        "pipeline_version": PIPELINE_VERSION,
        "layout_version": DEST_LAYOUT_VERSION,
        "source_bucket": corpus.source_bucket,
        "source_prefix": corpus.source_prefix,
        "source_dataset_manifest_sha256": corpus.source_dataset_manifest_sha256,
        "dest_bucket": dest_bucket,
        "dest_prefix": dest_prefix,
        "category": category,
        "source_tar_sha256": shard["sha256"],
        "source_tar_bytes": int(shard["tar_bytes"]),
    }


def valid_video_commit(
    commit: dict[str, Any],
    inventory: VideoInventory,
    binding: dict[str, Any],
) -> bool:
    sizes = commit.get("file_sizes")
    return (
        commit.get("status") == "verified"
        and commit.get("binding") == binding
        and commit.get("video_id") == inventory.video_id
        and commit.get("category") == inventory.category
        and int(commit.get("frame_count", -1)) == len(inventory.mapping)
        and isinstance(sizes, list)
        and tuple(int(size) for size in sizes) == inventory.file_sizes
        and int(commit.get("total_bytes", -1)) == inventory.total_bytes
        and commit.get("inventory_sha256") == inventory.inventory_sha256
    )


def initial_remote_progress(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "keyframe_category_progress",
        "binding": binding,
        "status": "running",
        "videos": {},
        "stream_next_offset": 0,
        "updated_at": utc_now(),
    }


def process_video(
    remote: HfBucketRemote,
    corpus: Corpus,
    category: str,
    video_id: str,
    binding: dict[str, Any],
    *,
    verify_jpeg_magic: bool,
    verify_completed_on_resume: bool,
) -> dict[str, Any]:
    inventory = build_video_inventory(
        corpus,
        category,
        video_id,
        verify_jpeg_magic=verify_jpeg_magic,
    )
    commit_path = keyframe_video_commit_path(remote.prefix, category, video_id)
    existing = remote.read_json(commit_path)
    if existing is not None:
        if not valid_video_commit(existing, inventory, binding):
            raise UploadError(
                f"Existing remote commit is incompatible for {video_id}; use another destination prefix"
            )
        if verify_completed_on_resume:
            expected = {
                remote.key(keyframe_media_key(category, video_id, row.n)): size
                for row, size in zip(inventory.mapping, inventory.file_sizes)
            }
            remote.verify_exact(
                expected,
                prefix=remote.key(f"Keyframes/Keyframes_{category}/{video_id}") + "/",
                label=f"resume {video_id}",
            )
        log(f"SKIP {video_id}: compatible verified commit exists")
        return existing

    expected: dict[str, int] = {}
    pairs: list[tuple[Path, str]] = []
    for row, local_file, size in zip(
        inventory.mapping,
        inventory.local_files,
        inventory.file_sizes,
    ):
        remote_path = remote.key(keyframe_media_key(category, video_id, row.n))
        pairs.append((local_file, remote_path))
        expected[remote_path] = size
    remote.upload_pairs(pairs, label=f"upload {video_id}")
    remote.verify_exact(
        expected,
        prefix=remote.key(f"Keyframes/Keyframes_{category}/{video_id}") + "/",
        label=f"verify {video_id}",
    )
    commit = {
        "schema": 1,
        "kind": "keyframe_video_commit",
        "status": "verified",
        "binding": binding,
        "video_id": video_id,
        "category": category,
        "frame_count": len(inventory.mapping),
        "total_bytes": inventory.total_bytes,
        "file_sizes": list(inventory.file_sizes),
        "inventory_sha256": inventory.inventory_sha256,
        "media_prefix": keyframe_media_key(category, video_id, 1).rsplit("/", 1)[0],
        "completed_at": utc_now(),
    }
    remote.upload_json(commit_path, commit, label=f"commit {video_id}")
    log(
        f"DONE {video_id}: {len(inventory.mapping):,} JPEG, "
        f"{inventory.total_bytes / 1024**3:.2f} GiB"
    )
    return commit


def load_or_create_local_state(
    path: Path,
    *,
    corpus: Corpus,
    dest_bucket: str,
    dest_prefix: str,
) -> dict[str, Any]:
    binding = {
        "source_root": str(corpus.source_root),
        "source_dataset_manifest_sha256": corpus.source_dataset_manifest_sha256,
        "dest_bucket": dest_bucket,
        "dest_prefix": dest_prefix,
        "pipeline_version": PIPELINE_VERSION,
        "layout_version": DEST_LAYOUT_VERSION,
    }
    if path.exists():
        state = load_json(path)
        if state.get("schema") != STATE_SCHEMA or state.get("binding") != binding:
            raise UploadError(
                f"Local state binding mismatch: {path}. Use a different --state-file."
            )
        if not isinstance(state.get("videos"), dict):
            raise UploadError(f"Malformed local state: {path}")
        return state
    state = {
        "schema": STATE_SCHEMA,
        "binding": binding,
        "status": "running",
        "videos": {},
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    atomic_json(path, state)
    return state


def mark_category_complete(
    remote: HfBucketRemote,
    corpus: Corpus,
    category: str,
    binding: dict[str, Any],
    progress: dict[str, Any],
    *,
    verify_jpeg_magic: bool,
) -> None:
    expected_videos = corpus.category_videos[category]
    missing = [video for video in expected_videos if video not in progress["videos"]]
    if missing:
        raise UploadError(f"Cannot commit {category}; missing videos: {missing[:10]}")
    frame_count = sum(int(progress["videos"][video]["frame_count"]) for video in expected_videos)
    total_bytes = sum(int(progress["videos"][video]["total_bytes"]) for video in expected_videos)
    if frame_count != corpus.category_frames[category]:
        raise UploadError(f"Category frame total mismatch: {category}")
    category_commit = {
        "schema": 1,
        "kind": "keyframe_category_commit",
        "status": "verified",
        "binding": binding,
        "video_count": len(expected_videos),
        "frame_count": frame_count,
        "total_bytes": total_bytes,
        "source_verification": (
            "local_extracted_exact_map_inventory_"
            + ("jpeg_magic_" if verify_jpeg_magic else "")
            + "plus_remote_exact_path_size_audit"
        ),
        "completed_at": utc_now(),
    }
    remote.upload_json(
        keyframe_category_commit_path(remote.prefix, category),
        category_commit,
        label=f"commit category {category}",
    )
    progress["status"] = "verified"
    progress["stream_next_offset"] = None
    progress["summary"] = {
        "video_count": len(expected_videos),
        "frame_count": frame_count,
        "total_bytes": total_bytes,
    }
    progress["updated_at"] = utc_now()
    remote.upload_json(
        keyframe_progress_path(remote.prefix, category),
        progress,
        label=f"save final {category} progress",
    )
    log(
        f"CATEGORY DONE {category}: {len(expected_videos)} videos, "
        f"{frame_count:,} JPEG, {total_bytes / 1024**3:.2f} GiB"
    )


def dry_run_selected(
    corpus: Corpus,
    categories: Sequence[str],
    *,
    workers: int,
    verify_jpeg_magic: bool,
) -> None:
    total_videos = sum(len(corpus.category_videos[category]) for category in categories)
    total_frames = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="local-audit") as pool:
        futures = {
            pool.submit(
                build_video_inventory,
                corpus,
                category,
                video_id,
                verify_jpeg_magic=verify_jpeg_magic,
            ): video_id
            for category in categories
            for video_id in corpus.category_videos[category]
        }
        done = 0
        for future in as_completed(futures):
            inventory = future.result()
            done += 1
            total_frames += len(inventory.mapping)
            total_bytes += inventory.total_bytes
            if done % 25 == 0 or done == total_videos:
                log(f"DRY RUN {done}/{total_videos} videos, {total_frames:,} JPEG")
    expected_frames = sum(corpus.category_frames[category] for category in categories)
    if total_frames != expected_frames:
        raise UploadError(f"Dry-run count mismatch: {total_frames:,} != {expected_frames:,}")
    log(
        f"DRY RUN PASS: {total_videos} videos, {total_frames:,} JPEG, "
        f"{total_bytes / 1024**3:.2f} GiB; Hugging Face was not contacted"
    )


def preflight_destination(
    bucket: str,
    token: str,
    *,
    create_if_missing: bool,
    require_public: bool,
) -> None:
    try:
        from huggingface_hub import bucket_info, create_bucket
        from huggingface_hub.errors import BucketNotFoundError
    except ImportError as exc:
        raise UploadError("Install huggingface_hub[hf_xet]>=1.23,<2") from exc
    try:
        info = bucket_info(bucket, token=token)
    except BucketNotFoundError:
        if not create_if_missing:
            raise
        log(f"Creating destination bucket {bucket} (public)")
        create_bucket(bucket, private=False, exist_ok=True, token=token)
        info = bucket_info(bucket, token=token)
    if require_public and info.private:
        raise UploadError(
            f"Destination bucket {bucket} is private; make it public before media cutover"
        )
    log(
        f"DESTINATION {bucket}: private={info.private}, "
        f"files={info.total_files}, bytes={info.size}"
    )


def upload_selected(args: argparse.Namespace, corpus: Corpus, categories: Sequence[str]) -> None:
    token = read_token(args.token_file)
    preflight_destination(
        args.dest_bucket,
        token,
        create_if_missing=args.create_dest_bucket,
        require_public=args.require_public_dest,
    )
    remote = HfBucketRemote(
        bucket=args.dest_bucket,
        prefix=args.dest_prefix,
        token=token,
        upload_batch_files=args.upload_batch_files,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
    )
    local_state = load_or_create_local_state(
        args.state_file,
        corpus=corpus,
        dest_bucket=args.dest_bucket,
        dest_prefix=args.dest_prefix,
    )
    state_lock = threading.Lock()
    total_selected = sum(len(corpus.category_videos[category]) for category in categories)
    completed_this_run = 0

    for category in categories:
        binding = commit_binding(corpus, args.dest_bucket, args.dest_prefix, category)
        category_commit_path = keyframe_category_commit_path(args.dest_prefix, category)
        existing_category = remote.read_json(category_commit_path)
        if existing_category is not None:
            if (
                existing_category.get("status") != "verified"
                or existing_category.get("binding") != binding
                or int(existing_category.get("video_count", -1))
                != len(corpus.category_videos[category])
                or int(existing_category.get("frame_count", -1))
                != corpus.category_frames[category]
            ):
                raise UploadError(
                    f"Existing category commit is incompatible: {category_commit_path}"
                )
            if not args.verify_completed_on_resume:
                log(f"CATEGORY SKIP {category}: verified category commit exists")
                continue

        progress_path = keyframe_progress_path(args.dest_prefix, category)
        progress = remote.read_json(progress_path)
        if progress is None:
            progress = initial_remote_progress(binding)
        elif progress.get("binding") != binding or not isinstance(progress.get("videos"), dict):
            raise UploadError(
                f"Remote progress binding mismatch: {progress_path}; use another destination prefix"
            )
        log(
            f"CATEGORY {category}: {len(corpus.category_videos[category])} videos, "
            f"{corpus.category_frames[category]:,} JPEG, "
            f"remote progress={len(progress['videos'])}"
        )

        pool = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix=f"hf-{category}")
        futures: dict[Future[dict[str, Any]], str] = {}
        try:
            for video_id in corpus.category_videos[category]:
                future = pool.submit(
                    process_video,
                    remote,
                    corpus,
                    category,
                    video_id,
                    binding,
                    verify_jpeg_magic=args.verify_jpeg_magic,
                    verify_completed_on_resume=args.verify_completed_on_resume,
                )
                futures[future] = video_id
            for future in as_completed(futures):
                video_id = futures[future]
                commit = future.result()
                summary = {
                    "frame_count": int(commit["frame_count"]),
                    "total_bytes": int(commit["total_bytes"]),
                    "inventory_sha256": commit["inventory_sha256"],
                    "commit_path": keyframe_video_commit_path(
                        args.dest_prefix, category, video_id
                    ),
                }
                progress["videos"][video_id] = summary
                progress["updated_at"] = utc_now()
                remote.upload_json(
                    progress_path,
                    progress,
                    label=f"save {category} progress",
                )
                with state_lock:
                    local_state["videos"][video_id] = {
                        **summary,
                        "category": category,
                        "completed_at": utc_now(),
                    }
                    local_state["updated_at"] = utc_now()
                    atomic_json(args.state_file, local_state)
                completed_this_run += 1
                log(
                    f"PROGRESS {completed_this_run}/{total_selected} videos this run | "
                    f"{category} {len(progress['videos'])}/{len(corpus.category_videos[category])}"
                )
        except BaseException:
            for pending in futures:
                pending.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        mark_category_complete(
            remote,
            corpus,
            category,
            binding,
            progress,
            verify_jpeg_magic=args.verify_jpeg_magic,
        )

    local_state["status"] = "selected_categories_complete"
    local_state["selected_categories"] = list(categories)
    local_state["updated_at"] = utc_now()
    atomic_json(args.state_file, local_state)
    base = f"https://huggingface.co/buckets/{args.dest_bucket}/resolve"
    if args.dest_prefix:
        base += "/" + quote(args.dest_prefix, safe="/")
    log(f"UPLOAD COMPLETE for {list(categories)}")
    log(f"MEDIA_BASE_URL={base.rstrip('/')}")
    log(
        "Keyframe commits are Colab-compatible. Upload videos and run the notebook "
        "finalizer before publishing manifest/_SUCCESS.json."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-state-dir", type=Path, default=DEFAULT_SOURCE_STATE_DIR)
    parser.add_argument("--dest-bucket", default=DEFAULT_DEST_BUCKET)
    parser.add_argument("--dest-prefix", default="")
    parser.add_argument(
        "--categories",
        default=",".join(ALL_CATEGORIES),
        help="Comma/space-separated subset of L21..L30; distinct categories may run separately.",
    )
    parser.add_argument("--token-file", type=Path, default=Path("HF_BUCKET/HF_TOKEN.txt"))
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path(".hf_media_local_upload_state.json"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--upload-batch-files", type=int, default=1000)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-base-seconds", type=float, default=3.0)
    parser.add_argument(
        "--verify-jpeg-magic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verify-completed-on-resume", action="store_true")
    parser.add_argument(
        "--create-dest-bucket",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-public-dest",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.dest_bucket = str(args.dest_bucket).strip().strip("/")
    args.dest_prefix = str(args.dest_prefix).strip().strip("/")
    args.categories = parse_categories(args.categories)
    if not args.categories:
        parser.error("--categories cannot be empty")
    if not args.dest_bucket:
        parser.error("--dest-bucket cannot be empty")
    if ".." in Path(args.dest_prefix).parts:
        parser.error("--dest-prefix must not contain '..'")
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    if not 1 <= args.upload_batch_files <= 1000:
        parser.error("--upload-batch-files must be between 1 and 1000")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    return args


def run(args: argparse.Namespace) -> int:
    log("Auditing source metadata and final map CSV files...")
    corpus = audit_corpus(args.source_root, args.source_state_dir)
    if args.dest_bucket == corpus.source_bucket:
        if not args.dest_prefix:
            raise UploadError("Destination prefix is required when reusing the source bucket")
        if (
            args.dest_prefix == corpus.source_prefix
            or args.dest_prefix.startswith(corpus.source_prefix + "/")
        ):
            raise UploadError("Destination must not overlap the immutable source prefix")
    selected_frames = sum(corpus.category_frames[category] for category in args.categories)
    selected_videos = sum(len(corpus.category_videos[category]) for category in args.categories)
    log(
        f"METADATA PASS: selected={list(args.categories)}, "
        f"videos={selected_videos}, JPEG={selected_frames:,}"
    )
    if args.dry_run:
        dry_run_selected(
            corpus,
            args.categories,
            workers=args.workers,
            verify_jpeg_magic=args.verify_jpeg_magic,
        )
    else:
        upload_selected(args, corpus, args.categories)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print(
            "Interrupted. Verified per-video commits are safe; rerun the same command to resume.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
