#!/usr/bin/env python3
"""Audit and upload the InfoShot++ Qwen3-VL-Embedding-8B dataset to Milvus.

The embedding artifact uses ``frame_id = <video_id>@f<frame_idx:08d>`` as its
canonical identity.  The AIC application, however, currently routes media with
``submit_keyframe_id = <category>/<video_id>/<keyframe_n>``.  This uploader
therefore performs a strict join against the final InfoShot++ map CSV files.  It
never treats ``frame_idx`` as the keyframe ordinal.

The script is intentionally non-destructive: it can create a new collection or
resume one that it created previously, but it never drops a collection.  A
shard is checkpointed only after every upsert in that shard has succeeded;
replaying a partial shard is safe because the application id is the primary key.

Run it from the repository root with ``backend/.venv``'s interpreter: it is the
only environment here carrying numpy, pyarrow and pymilvus together (the root
``.venv`` has no pyarrow and its Python symlink is dangling, and there is no
system ``python3``).  Credentials come from ``backend/.env``
(``MILVUS_ENDPOINT_2``/``MILVUS_TOKEN_2``)::

    # 1. Validate the artifact and the map join without contacting Milvus (~1 min).
    ./backend/.venv/bin/python3 milvus_upload_qwen3_vl_embedding_8b.py \
        --dry-run --skip-parquet-sha256

    # 2. Create the collection and upload all 1,339,055 vectors. Re-run the same
    #    command to resume: completed shards are skipped.
    ./backend/.venv/bin/python3 milvus_upload_qwen3_vl_embedding_8b.py \
        --skip-parquet-sha256

Add ``--skip-parquet-sha256`` to step 2 as well if the ~20.6 GiB byte-hash audit
was already passed in step 1; every row is still validated either way.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np


DEFAULT_ARTIFACT_ROOT = Path(
    "/home/bao/Projects/EncoderModel/qwen3-vl-embedding-8b-4096-v3"
)
DEFAULT_MAP_ROOT = Path(
    "/home/bao/Projects/ExtractKeyframe/keyframe_L/infoshootpp/map-keyframes"
)
DEFAULT_COLLECTION = "aic26_image_qwen3vl8b_infoshotpp_v3"
EXPECTED_ROWS = 1_339_055
EXPECTED_DIM = 4096
EXPECTED_CATEGORIES = tuple(f"L{number}" for number in range(21, 31))
EXPECTED_SHARDS = 658
EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-Embedding-8B"
EXPECTED_MODEL_REVISION = "2c4565515e0f265c6511776e7193b22c0968ddc7"
EXPECTED_SEMANTIC_FINGERPRINT = (
    "8047495b9ffdd325610b7ea8ecefb15389564632a7e2e20c23ae6995e1454ac3"
)
VECTOR_FIELD = "embedding"
STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Shard:
    category: str
    shard_id: int
    row_start: int
    row_stop: int
    rows: int
    first_frame_id: str
    last_frame_id: str
    parquet_path: Path
    parquet_bytes: int
    parquet_sha256: str
    commit_path: Path
    commit_sha256: str

    @property
    def key(self) -> str:
        return f"{self.category}/part-{self.shard_id:05d}"


@dataclass(frozen=True)
class MapRow:
    keyframe_n: int
    frame_idx: int
    pts_time: float
    fps: float


@dataclass
class Sample:
    submit_keyframe_id: str
    frame_id: str
    video_id: str
    category: str
    keyframe_n: int
    frame_idx: int
    pts_time: float
    fps: float
    vector: np.ndarray


@dataclass(frozen=True)
class Audit:
    semantic_fingerprint: str
    manifest_sha256: str
    success_sha256: str
    map_inventory_sha256: str
    expected_rows: int
    shards: tuple[Shard, ...]
    map_files: dict[str, tuple[Path, ...]]
    map_rows: dict[str, int]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _import_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing pyarrow. Install it in the active environment with: "
            "uv pip install pyarrow"
        ) from exc
    return pa, pq


def _import_pymilvus() -> tuple[Any, Any]:
    try:
        from pymilvus import DataType, MilvusClient
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing pymilvus. Install it in the active environment with: "
            "uv pip install pymilvus"
        ) from exc
    return MilvusClient, DataType


def _validate_manifest(root: Path) -> tuple[dict[str, Any], str, str]:
    success_path = root / "_SUCCESS.json"
    manifest_path = root / "embedding_dataset_manifest.json"
    success = read_json(success_path)
    manifest = read_json(manifest_path)
    if success != manifest:
        raise ValueError("_SUCCESS.json and embedding_dataset_manifest.json differ")
    if success.get("complete") is not True:
        raise ValueError("Embedding dataset is not marked complete")
    if int(success.get("expected_total_rows") or -1) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows in embedding manifest")
    if int(success.get("total_committed_rows") or -1) != EXPECTED_ROWS:
        raise ValueError("Committed embedding row count is incomplete")
    config = success.get("semantic_config") or {}
    model = config.get("model") or {}
    storage = config.get("storage") or {}
    inference = config.get("inference") or {}
    if model.get("id") != EXPECTED_MODEL_ID:
        raise ValueError(f"Unexpected model: {model.get('id')!r}")
    if model.get("revision") != EXPECTED_MODEL_REVISION:
        raise ValueError(f"Unexpected model revision: {model.get('revision')!r}")
    if model.get("mrl_dimension") is not None:
        raise ValueError("Expected native 4096-d vectors, not an MRL projection")
    if int(model.get("embedding_dim") or -1) != EXPECTED_DIM:
        raise ValueError(f"Expected vector dimension {EXPECTED_DIM}")
    if storage.get("dtype") != "float32":
        raise ValueError(f"Expected float32 embeddings, got {storage.get('dtype')!r}")
    if inference.get("l2_normalize") is not True:
        raise ValueError("Embedding manifest does not declare L2 normalization")
    fingerprint = str(success.get("semantic_fingerprint") or "")
    if fingerprint != EXPECTED_SEMANTIC_FINGERPRINT:
        raise ValueError(
            f"Unexpected semantic_fingerprint: {fingerprint!r}; "
            "refusing to mix a different Qwen embedding contract"
        )
    retrieval = success.get("retrieval_contract") or {}
    if retrieval.get("model_id") != EXPECTED_MODEL_ID:
        raise ValueError("Retrieval contract model does not match the image vectors")
    if int(retrieval.get("dimension") or -1) != EXPECTED_DIM:
        raise ValueError("Retrieval contract does not declare native 4096-d queries")
    if retrieval.get("metric") != "COSINE" or retrieval.get("l2_normalize_fp32") is not True:
        raise ValueError("Retrieval contract must use FP32 L2 normalization and COSINE")
    expected_counts = (config.get("source") or {}).get("expected_counts") or {}
    if tuple(sorted(expected_counts)) != EXPECTED_CATEGORIES:
        raise ValueError(f"Expected categories {EXPECTED_CATEGORIES}, got {tuple(sorted(expected_counts))}")
    return success, sha256_file(manifest_path), sha256_file(success_path)


def _parquet_path_for_commit(root: Path, commit: dict[str, Any], commit_path: Path) -> Path:
    category = str(commit.get("category") or "")
    remote_name = Path(str(commit.get("parquet_path") or "")).name
    if not remote_name:
        remote_name = commit_path.with_suffix(".parquet").name
    return root / "embeddings" / category / remote_name


def audit_shards(
    root: Path,
    manifest: dict[str, Any],
    fingerprint: str,
    *,
    verify_sha256: bool,
) -> tuple[Shard, ...]:
    _, pq = _import_pyarrow()
    commits = sorted((root / "commits").glob("L*/part-*.json"))
    expected_shards = sum(
        int(record.get("expected_shards") or 0)
        for record in (manifest.get("per_category") or {}).values()
    )
    if expected_shards != EXPECTED_SHARDS:
        raise ValueError(f"Expected manifest to declare {EXPECTED_SHARDS} shards, got {expected_shards}")
    if len(commits) != expected_shards:
        raise ValueError(f"Expected {expected_shards} commit files, found {len(commits)}")

    shards: list[Shard] = []
    next_row = {category: 0 for category in EXPECTED_CATEGORIES}
    next_shard = {category: 0 for category in EXPECTED_CATEGORIES}
    category_rows = {category: 0 for category in EXPECTED_CATEGORIES}

    for commit_path in commits:
        commit = read_json(commit_path)
        category = str(commit.get("category") or "")
        if category not in next_row:
            raise ValueError(f"Unexpected category in {commit_path}: {category!r}")
        shard_id = int(commit.get("shard_id") if commit.get("shard_id") is not None else -1)
        if shard_id != next_shard[category]:
            raise ValueError(
                f"Non-contiguous shard id for {category}: {shard_id}, expected {next_shard[category]}"
            )
        row_start = int(commit.get("row_start") if commit.get("row_start") is not None else -1)
        row_stop = int(commit.get("row_stop") if commit.get("row_stop") is not None else -1)
        rows = int(commit.get("rows") or -1)
        if row_start != next_row[category] or row_stop - row_start != rows or rows <= 0:
            raise ValueError(f"Invalid row range in {commit_path}")
        if commit.get("semantic_fingerprint") != fingerprint:
            raise ValueError(f"Fingerprint mismatch in {commit_path}")
        if int(commit.get("embedding_dim") or -1) != EXPECTED_DIM:
            raise ValueError(f"Embedding dimension mismatch in {commit_path}")
        if commit.get("embedding_dtype") != "float32" or commit.get("l2_normalized") is not True:
            raise ValueError(f"Embedding storage contract mismatch in {commit_path}")

        parquet_path = _parquet_path_for_commit(root, commit, commit_path)
        expected_bytes = int(commit.get("parquet_bytes") or -1)
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)
        if parquet_path.stat().st_size != expected_bytes:
            raise ValueError(f"Parquet byte-size mismatch: {parquet_path}")
        expected_sha = str(commit.get("parquet_sha256") or "")
        if verify_sha256 and sha256_file(parquet_path) != expected_sha:
            raise ValueError(f"Parquet SHA-256 mismatch: {parquet_path}")

        parquet_file = pq.ParquetFile(parquet_path)
        if parquet_file.metadata.num_rows != rows:
            raise ValueError(f"Parquet row-count mismatch: {parquet_path}")
        schema = parquet_file.schema_arrow
        required = {
            "frame_id",
            "video_id",
            "category",
            "frame_idx",
            "pts_time",
            "fps",
            "image_relpath",
            VECTOR_FIELD,
        }
        if not required.issubset(schema.names):
            raise ValueError(f"Missing Parquet columns in {parquet_path}: {sorted(required - set(schema.names))}")
        vector_type = schema.field(VECTOR_FIELD).type
        if not getattr(vector_type, "list_size", None) == EXPECTED_DIM:
            raise ValueError(f"Bad embedding type in {parquet_path}: {vector_type}")
        metadata = schema.metadata or {}
        if metadata.get(b"semantic_fingerprint", b"").decode() != fingerprint:
            raise ValueError(f"Parquet semantic fingerprint mismatch: {parquet_path}")

        shard = Shard(
            category=category,
            shard_id=shard_id,
            row_start=row_start,
            row_stop=row_stop,
            rows=rows,
            first_frame_id=str(commit.get("first_frame_id") or ""),
            last_frame_id=str(commit.get("last_frame_id") or ""),
            parquet_path=parquet_path,
            parquet_bytes=expected_bytes,
            parquet_sha256=expected_sha,
            commit_path=commit_path,
            commit_sha256=sha256_file(commit_path),
        )
        shards.append(shard)
        next_shard[category] += 1
        next_row[category] = row_stop
        category_rows[category] += rows

    expected_counts = manifest["semantic_config"]["source"]["expected_counts"]
    for category in EXPECTED_CATEGORIES:
        expected = int(expected_counts[category])
        if category_rows[category] != expected:
            raise ValueError(
                f"Embedding rows for {category}: {category_rows[category]:,}, expected {expected:,}"
            )
    if sum(category_rows.values()) != EXPECTED_ROWS:
        raise ValueError("Embedding shard total does not match the manifest")
    return tuple(shards)


def audit_map_inventory(map_root: Path) -> tuple[dict[str, tuple[Path, ...]], dict[str, int], str]:
    inventory_digest = hashlib.sha256()
    files_by_category: dict[str, tuple[Path, ...]] = {}
    rows_by_category: dict[str, int] = {}
    for category in EXPECTED_CATEGORIES:
        files = tuple(sorted((map_root / category).glob("*.csv")))
        if not files:
            raise ValueError(f"No map CSV files found for {category} under {map_root}")
        files_by_category[category] = files
        row_count = 0
        for path in files:
            relative = path.relative_to(map_root).as_posix()
            digest = sha256_file(path)
            inventory_digest.update(f"{relative}\0{path.stat().st_size}\0{digest}\n".encode())
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != ["n", "pts_time", "fps", "frame_idx"]:
                    raise ValueError(f"Unexpected map schema in {path}: {reader.fieldnames}")
                previous_n = 0
                previous_frame_idx = -1
                for line_number, row in enumerate(reader, start=2):
                    try:
                        keyframe_n = int(row["n"])
                        frame_idx = int(row["frame_idx"])
                        pts_time = float(row["pts_time"])
                        fps = float(row["fps"])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid map row at {path}:{line_number}") from exc
                    if keyframe_n != previous_n + 1:
                        raise ValueError(f"Non-contiguous n at {path}:{line_number}")
                    if frame_idx <= previous_frame_idx:
                        raise ValueError(f"Non-increasing frame_idx at {path}:{line_number}")
                    if not math.isfinite(pts_time) or not math.isfinite(fps) or fps <= 0:
                        raise ValueError(f"Invalid pts_time/fps at {path}:{line_number}")
                    previous_n = keyframe_n
                    previous_frame_idx = frame_idx
                    row_count += 1
        rows_by_category[category] = row_count
    if sum(rows_by_category.values()) != EXPECTED_ROWS:
        raise ValueError(
            f"Map files contain {sum(rows_by_category.values()):,} rows, expected {EXPECTED_ROWS:,}"
        )
    return files_by_category, rows_by_category, inventory_digest.hexdigest()


def audit_inputs(root: Path, map_root: Path, *, verify_sha256: bool = True) -> Audit:
    manifest, manifest_sha, success_sha = _validate_manifest(root)
    fingerprint = str(manifest["semantic_fingerprint"])
    print(f"Auditing {EXPECTED_SHARDS} embedding shard commits and Parquet metadata...", flush=True)
    shards = audit_shards(root, manifest, fingerprint, verify_sha256=verify_sha256)
    print("Auditing 873 final InfoShot++ map CSV files...", flush=True)
    map_files, map_rows, map_digest = audit_map_inventory(map_root)
    expected_counts = manifest["semantic_config"]["source"]["expected_counts"]
    for category in EXPECTED_CATEGORIES:
        if map_rows[category] != int(expected_counts[category]):
            raise ValueError(
                f"Map/embedding count mismatch for {category}: "
                f"{map_rows[category]:,} != {int(expected_counts[category]):,}"
            )
    return Audit(
        semantic_fingerprint=fingerprint,
        manifest_sha256=manifest_sha,
        success_sha256=success_sha,
        map_inventory_sha256=map_digest,
        expected_rows=EXPECTED_ROWS,
        shards=shards,
        map_files=map_files,
        map_rows=map_rows,
    )


def load_category_map(category: str, paths: Sequence[Path]) -> dict[str, dict[int, MapRow]]:
    result: dict[str, dict[int, MapRow]] = {}
    for path in paths:
        video_id = path.stem
        if video_id.split("_", 1)[0] != category:
            raise ValueError(f"Map path/category mismatch: {path}")
        video: dict[int, MapRow] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                item = MapRow(
                    keyframe_n=int(row["n"]),
                    frame_idx=int(row["frame_idx"]),
                    pts_time=float(row["pts_time"]),
                    fps=float(row["fps"]),
                )
                if item.frame_idx in video:
                    raise ValueError(f"Duplicate frame_idx {item.frame_idx} in {path}")
                video[item.frame_idx] = item
        result[video_id] = video
    return result


def _float_matches(left: float, right: float, *, tolerance: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and math.isclose(
        left, right, rel_tol=tolerance, abs_tol=tolerance
    )


def _batch_columns(batch: Any) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for name in ("frame_id", "video_id", "category", "image_relpath"):
        columns[name] = batch.column(batch.schema.get_field_index(name)).to_pylist()
    for name in ("frame_idx", "pts_time", "fps"):
        columns[name] = batch.column(batch.schema.get_field_index(name)).to_numpy(
            zero_copy_only=False
        )
    vector_array = batch.column(batch.schema.get_field_index(VECTOR_FIELD))
    columns[VECTOR_FIELD] = vector_array.values.to_numpy(zero_copy_only=False).reshape(
        len(batch), EXPECTED_DIM
    )
    return columns


def validate_and_build_entities(
    batch: Any,
    category: str,
    remaining_map: dict[str, dict[int, MapRow]],
    *,
    build_entities: bool,
) -> tuple[list[dict[str, Any]], list[Sample], str, str]:
    columns = _batch_columns(batch)
    vectors = np.asarray(columns[VECTOR_FIELD], dtype=np.float32)
    if vectors.shape != (len(batch), EXPECTED_DIM):
        raise ValueError(f"Unexpected vector batch shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding batch contains NaN or infinity")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=2e-5, atol=2e-5):
        raise ValueError(f"Embedding norms outside tolerance: [{norms.min()}, {norms.max()}]")

    entities: list[dict[str, Any]] = []
    samples: list[Sample] = []
    first_frame_id = ""
    last_frame_id = ""
    for index in range(len(batch)):
        frame_id = str(columns["frame_id"][index])
        video_id = str(columns["video_id"][index])
        row_category = str(columns["category"][index])
        frame_idx = int(columns["frame_idx"][index])
        pts_time = float(columns["pts_time"][index])
        fps = float(columns["fps"][index])
        image_relpath = str(columns["image_relpath"][index])
        expected_frame_id = f"{video_id}@f{frame_idx:08d}"
        expected_image = f"infoshootpp/keyframes/{category}/{video_id}/f{frame_idx:08d}.jpg"
        if row_category != category or video_id.split("_", 1)[0] != category:
            raise ValueError(f"Category mismatch for {frame_id}")
        if frame_id != expected_frame_id:
            raise ValueError(f"Malformed frame_id {frame_id!r}; expected {expected_frame_id!r}")
        if image_relpath != expected_image:
            raise ValueError(f"Unexpected image_relpath for {frame_id}: {image_relpath!r}")
        video_map = remaining_map.get(video_id)
        map_row = video_map.pop(frame_idx, None) if video_map is not None else None
        if map_row is None:
            raise ValueError(f"No unique InfoShot++ map row for {frame_id}")
        if not video_map:
            remaining_map.pop(video_id, None)
        if not _float_matches(pts_time, map_row.pts_time, tolerance=2e-5):
            raise ValueError(
                f"pts_time mismatch for {frame_id}: parquet={pts_time}, map={map_row.pts_time}"
            )
        if not _float_matches(fps, map_row.fps, tolerance=2e-5):
            raise ValueError(f"fps mismatch for {frame_id}: parquet={fps}, map={map_row.fps}")

        keyframe_name = f"{map_row.keyframe_n:03d}"
        keyframe_id = f"{video_id}/{keyframe_name}"
        submit_id = f"{category}/{keyframe_id}"
        if build_entities:
            entities.append(
                {
                    "id": submit_id,
                    "image_id": submit_id,
                    "keyframe_id": keyframe_id,
                    "submit_keyframe_id": submit_id,
                    "frame_id": frame_id,
                    "category": category,
                    "video_id": video_id,
                    "keyframe_n": map_row.keyframe_n,
                    "frame_idx": frame_idx,
                    "pts_time": map_row.pts_time,
                    "fps": map_row.fps,
                    "image_path": image_relpath,
                    VECTOR_FIELD: vectors[index],
                }
            )
        if index == 0 or index == len(batch) - 1:
            samples.append(
                Sample(
                    submit_keyframe_id=submit_id,
                    frame_id=frame_id,
                    video_id=video_id,
                    category=category,
                    keyframe_n=map_row.keyframe_n,
                    frame_idx=frame_idx,
                    pts_time=map_row.pts_time,
                    fps=map_row.fps,
                    vector=vectors[index].copy(),
                )
            )
        first_frame_id = first_frame_id or frame_id
        last_frame_id = frame_id
    return entities, samples, first_frame_id, last_frame_id


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _read_optional_secret(path: Path | None) -> str:
    if path is None:
        return ""
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Credential file is empty: {path}")
    return value


def resolve_credentials(args: argparse.Namespace) -> tuple[str, str]:
    env_file = parse_env_file(args.env_file)
    endpoint = (
        _read_optional_secret(args.endpoint_file)
        or os.environ.get("MILVUS_ENDPOINT", "").strip()
        or os.environ.get("MILVUS_ENDPOINT_2", "").strip()
        or env_file.get("MILVUS_ENDPOINT", "").strip()
        or env_file.get("MILVUS_ENDPOINT_2", "").strip()
    )
    token = (
        _read_optional_secret(args.token_file)
        or os.environ.get("MILVUS_TOKEN", "").strip()
        or os.environ.get("MILVUS_TOKEN_2", "").strip()
        or env_file.get("MILVUS_TOKEN", "").strip()
        or env_file.get("MILVUS_TOKEN_2", "").strip()
    )
    if not endpoint or not token:
        raise ValueError(
            "Milvus credentials not found. Set MILVUS_ENDPOINT/MILVUS_TOKEN or "
            "MILVUS_ENDPOINT_2/MILVUS_TOKEN_2, use --env-file, or pass credential files."
        )
    return endpoint, token


def endpoint_identity(endpoint: str) -> str:
    parsed = urlsplit(endpoint if "://" in endpoint else f"https://{endpoint}")
    port = f":{parsed.port}" if parsed.port is not None else ""
    public_identity = (
        f"{parsed.scheme.lower()}://{parsed.hostname or ''}{port}{parsed.path.rstrip('/')}"
    )
    return sha256_text(public_identity)


def connect(endpoint: str, token: str, *, timeout: float) -> Any:
    MilvusClient, _ = _import_pymilvus()
    return MilvusClient(
        uri=endpoint,
        token=token,
        timeout=timeout,
        alias=f"aic26_qwen3vl8b_infoshotpp_upload_{time.time_ns()}",
    )


def expected_schema(DataType: Any) -> dict[str, tuple[Any, dict[str, Any]]]:
    return {
        "id": (DataType.VARCHAR, {"is_primary": True, "max_length": 64}),
        "image_id": (DataType.VARCHAR, {"max_length": 64}),
        "keyframe_id": (DataType.VARCHAR, {"max_length": 64}),
        "submit_keyframe_id": (DataType.VARCHAR, {"max_length": 64}),
        "frame_id": (DataType.VARCHAR, {"max_length": 64}),
        "category": (DataType.VARCHAR, {"max_length": 16}),
        "video_id": (DataType.VARCHAR, {"max_length": 32}),
        "keyframe_n": (DataType.INT64, {}),
        "frame_idx": (DataType.INT64, {}),
        "pts_time": (DataType.DOUBLE, {}),
        "fps": (DataType.FLOAT, {}),
        "image_path": (DataType.VARCHAR, {"max_length": 256}),
        VECTOR_FIELD: (DataType.FLOAT_VECTOR, {"dim": EXPECTED_DIM}),
    }


def create_collection(client: Any, collection: str) -> None:
    MilvusClient, DataType = _import_pymilvus()
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    for name, (datatype, params) in expected_schema(DataType).items():
        schema.add_field(field_name=name, datatype=datatype, **params)
    indexes = client.prepare_index_params()
    indexes.add_index(field_name=VECTOR_FIELD, index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=collection, schema=schema, index_params=indexes)


def _datatype_number(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(getattr(value, "value"))


def verify_collection_schema(client: Any, collection: str) -> None:
    _, DataType = _import_pymilvus()
    description = client.describe_collection(collection_name=collection)
    if description.get("enable_dynamic_field") is True:
        raise ValueError(f"Collection {collection} unexpectedly enables dynamic fields")
    actual = {str(field["name"]): field for field in description.get("fields") or []}
    expected = expected_schema(DataType)
    if set(actual) != set(expected):
        raise ValueError(
            f"Collection schema fields differ: actual={sorted(actual)}, expected={sorted(expected)}"
        )
    for name, (datatype, params) in expected.items():
        field = actual[name]
        actual_type = field.get("type", field.get("data_type"))
        if _datatype_number(actual_type) != _datatype_number(datatype):
            raise ValueError(f"Datatype mismatch for {name}: {actual_type} != {datatype}")
        actual_params = field.get("params") or {}
        for key, expected_value in params.items():
            if key == "is_primary":
                actual_value = bool(field.get("is_primary"))
            else:
                actual_value = actual_params.get(key, field.get(key))
                if actual_value is not None:
                    actual_value = int(actual_value)
            if actual_value != expected_value:
                raise ValueError(
                    f"Schema parameter mismatch for {name}.{key}: {actual_value!r} != {expected_value!r}"
                )
    indexes = client.list_indexes(collection_name=collection)
    if VECTOR_FIELD not in indexes:
        raise ValueError(f"Collection {collection} has no {VECTOR_FIELD} index")
    index = client.describe_index(collection_name=collection, index_name=VECTOR_FIELD)
    metric = index.get("metric_type") or (index.get("params") or {}).get("metric_type")
    if metric != "COSINE":
        raise ValueError(f"Expected COSINE index, got {metric!r}")


def collection_row_count(client: Any, collection: str) -> int:
    stats = client.get_collection_stats(collection_name=collection)
    for key in ("row_count", "num_entities"):
        if key in stats:
            return int(stats[key])
    raise ValueError(f"Milvus did not return a row count: {stats}")


def state_binding(audit: Audit, collection: str, endpoint_hash: str) -> dict[str, Any]:
    return {
        "artifact_manifest_sha256": audit.manifest_sha256,
        "artifact_success_sha256": audit.success_sha256,
        "semantic_fingerprint": audit.semantic_fingerprint,
        "map_inventory_sha256": audit.map_inventory_sha256,
        "expected_rows": audit.expected_rows,
        "embedding_dim": EXPECTED_DIM,
        "metric_type": "COSINE",
        "collection": collection,
        "endpoint_identity_sha256": endpoint_hash,
    }


def load_or_create_state(path: Path, binding: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if path.exists():
        state = read_json(path)
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported upload state schema in {path}")
        if state.get("binding") != binding:
            raise ValueError(
                f"Upload state binding differs in {path}; use a different --state-file for this target"
            )
        if not isinstance(state.get("completed_shards"), dict):
            raise ValueError(f"Malformed completed_shards in {path}")
        return state, False
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "binding": binding,
        "completed_shards": {},
        "uploaded_rows": 0,
        "complete": False,
    }
    atomic_write_json(path, state)
    return state, True


def upsert_with_retry(
    client_box: list[Any],
    endpoint: str,
    token: str,
    collection: str,
    entities: list[dict[str, Any]],
    *,
    max_retries: int,
    retry_sleep: float,
    timeout: float,
) -> None:
    for attempt in range(max_retries + 1):
        try:
            result = client_box[0].upsert(collection_name=collection, data=entities, timeout=timeout)
            count = result.get("upsert_count") if isinstance(result, dict) else None
            if count is not None and int(count) != len(entities):
                raise RuntimeError(f"Milvus acknowledged {count} of {len(entities)} rows")
            return
        except Exception as exc:
            if attempt >= max_retries:
                raise
            delay = retry_sleep * (2**attempt)
            print(
                f"  upsert failed ({type(exc).__name__}: {exc}); "
                f"retry {attempt + 1}/{max_retries} in {delay:g}s",
                flush=True,
            )
            time.sleep(delay)
            client_box[0] = connect(endpoint, token, timeout=timeout)


def upsert_shard_parallel(
    client_boxes: list[list[Any]],
    endpoint: str,
    token: str,
    collection: str,
    batches: Sequence[list[dict[str, Any]]],
    *,
    max_retries: int,
    retry_sleep: float,
    timeout: float,
) -> None:
    """Upload one shard over independent clients, then return as one checkpoint unit."""

    assignments: list[list[list[dict[str, Any]]]] = [list() for _ in client_boxes]
    for index, batch in enumerate(batches):
        assignments[index % len(client_boxes)].append(batch)

    def upload_assignment(worker: int) -> None:
        for entities in assignments[worker]:
            upsert_with_retry(
                client_boxes[worker],
                endpoint,
                token,
                collection,
                entities,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
                timeout=timeout,
            )

    with ThreadPoolExecutor(max_workers=len(client_boxes), thread_name_prefix="milvus-upsert") as pool:
        futures = [
            pool.submit(upload_assignment, worker)
            for worker in range(len(client_boxes))
            if assignments[worker]
        ]
        for future in futures:
            future.result()


def _sample_to_expected(sample: Sample) -> dict[str, Any]:
    return {
        "id": sample.submit_keyframe_id,
        "image_id": sample.submit_keyframe_id,
        "keyframe_id": f"{sample.video_id}/{sample.keyframe_n:03d}",
        "submit_keyframe_id": sample.submit_keyframe_id,
        "frame_id": sample.frame_id,
        "category": sample.category,
        "video_id": sample.video_id,
        "keyframe_n": sample.keyframe_n,
        "frame_idx": sample.frame_idx,
        "pts_time": sample.pts_time,
        "fps": sample.fps,
    }


def verify_remote_samples(client: Any, collection: str, samples: Sequence[Sample]) -> list[dict[str, Any]]:
    selected: list[Sample] = []
    seen_categories: set[str] = set()
    for sample in samples:
        if sample.category not in seen_categories:
            selected.append(sample)
            seen_categories.add(sample.category)
    output_fields = [
        "id",
        "image_id",
        "keyframe_id",
        "submit_keyframe_id",
        "frame_id",
        "category",
        "video_id",
        "keyframe_n",
        "frame_idx",
        "pts_time",
        "fps",
        VECTOR_FIELD,
    ]
    rows = client.get(
        collection_name=collection,
        ids=[item.submit_keyframe_id for item in selected],
        output_fields=output_fields,
    )
    by_id = {str(row.get("id") or row.get("submit_keyframe_id")): row for row in rows or []}
    results: list[dict[str, Any]] = []
    for sample in selected:
        row = by_id.get(sample.submit_keyframe_id)
        if row is None:
            raise ValueError(f"Remote sample is missing: {sample.submit_keyframe_id}")
        expected = _sample_to_expected(sample)
        for key, value in expected.items():
            actual = row.get(key)
            if isinstance(value, float):
                if not _float_matches(float(actual), value, tolerance=2e-5):
                    raise ValueError(f"Remote sample {sample.submit_keyframe_id} differs at {key}")
            elif actual != value:
                raise ValueError(
                    f"Remote sample {sample.submit_keyframe_id} differs at {key}: {actual!r} != {value!r}"
                )
        remote_vector = np.asarray(row.get(VECTOR_FIELD), dtype=np.float32)
        if remote_vector.shape != (EXPECTED_DIM,) or not np.isfinite(remote_vector).all():
            raise ValueError(f"Bad remote vector for {sample.submit_keyframe_id}")
        cosine = float(np.dot(remote_vector, sample.vector) / (
            np.linalg.norm(remote_vector) * np.linalg.norm(sample.vector)
        ))
        if cosine < 0.99999:
            raise ValueError(f"Remote vector differs for {sample.submit_keyframe_id}: cosine={cosine}")
        results.append({"submit_keyframe_id": sample.submit_keyframe_id, "vector_cosine": cosine})

    # A few self-searches validate that the index is loaded and serves the same identity contract.
    for sample in selected[:3]:
        search = client.search(
            collection_name=collection,
            data=[sample.vector],
            anns_field=VECTOR_FIELD,
            limit=1,
            output_fields=["submit_keyframe_id", "video_id", "keyframe_n"],
            search_params={"metric_type": "COSINE"},
        )
        if not search or not search[0]:
            raise ValueError(f"Self-search returned no result for {sample.submit_keyframe_id}")
        hit = search[0][0]
        entity = hit.get("entity") or {}
        hit_id = entity.get("submit_keyframe_id") or hit.get("id")
        score = float(hit.get("distance", 0.0))
        if hit_id != sample.submit_keyframe_id or score < 0.9999:
            raise ValueError(
                f"Self-search mismatch for {sample.submit_keyframe_id}: id={hit_id!r}, score={score}"
            )
    return results


def wait_for_exact_count(
    client: Any,
    collection: str,
    expected: int,
    *,
    timeout_seconds: float,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    last = -1
    while time.monotonic() < deadline:
        last = collection_row_count(client, collection)
        if last == expected:
            return last
        if last > expected:
            raise ValueError(
                f"Collection has {last:,} rows, exceeding expected {expected:,}; refusing to continue"
            )
        print(f"Waiting for Milvus row count: {last:,}/{expected:,}", flush=True)
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for row count {expected:,}; last count was {last:,}")


def run(args: argparse.Namespace) -> int:
    artifact_root = args.artifact_root.resolve()
    map_root = args.map_root.resolve()
    started = time.monotonic()
    audit = audit_inputs(
        artifact_root,
        map_root,
        verify_sha256=not args.skip_parquet_sha256,
    )
    print(
        f"Input audit passed: {len(audit.shards)} shards, {audit.expected_rows:,} rows, "
        f"fingerprint={audit.semantic_fingerprint[:12]}...",
        flush=True,
    )

    endpoint = token = ""
    endpoint_hash = "dry-run"
    client_boxes: list[list[Any]] = []
    state: dict[str, Any] | None = None
    state_was_new = False
    if not args.dry_run:
        endpoint, token = resolve_credentials(args)
        endpoint_hash = endpoint_identity(endpoint)
        binding = state_binding(audit, args.collection, endpoint_hash)
        state, state_was_new = load_or_create_state(args.state_file, binding)
        client_boxes = [[connect(endpoint, token, timeout=args.timeout)]]
        client = client_boxes[0][0]
        exists = client.has_collection(collection_name=args.collection)
        if exists:
            verify_collection_schema(client, args.collection)
            remote_count = collection_row_count(client, args.collection)
            if state_was_new and remote_count:
                raise ValueError(
                    f"Collection {args.collection} already contains {remote_count:,} rows but no prior "
                    "matching local state was found. Refusing to adopt it automatically."
                )
            print(f"Resuming existing collection {args.collection} ({remote_count:,} rows)", flush=True)
        else:
            print(f"Creating new collection {args.collection} (FLOAT_VECTOR/{EXPECTED_DIM}, COSINE)", flush=True)
            create_collection(client, args.collection)
            verify_collection_schema(client, args.collection)
        for _ in range(1, args.workers):
            client_boxes.append([connect(endpoint, token, timeout=args.timeout)])

    total_validated = 0
    total_uploaded_this_run = 0
    all_samples: list[Sample] = []
    completed = (state or {}).get("completed_shards") or {}

    for category in EXPECTED_CATEGORIES:
        print(f"[{category}] loading and joining final map...", flush=True)
        remaining_map = load_category_map(category, audit.map_files[category])
        category_shards = [shard for shard in audit.shards if shard.category == category]
        category_validated = 0
        for shard in category_shards:
            should_upload = not args.dry_run and shard.key not in completed
            shard_validated = 0
            shard_first = ""
            shard_last = ""
            upload_batches: list[list[dict[str, Any]]] = []
            _, pq = _import_pyarrow()
            parquet_file = pq.ParquetFile(shard.parquet_path)
            for batch in parquet_file.iter_batches(batch_size=args.batch_size):
                entities, samples, first_id, last_id = validate_and_build_entities(
                    batch,
                    category,
                    remaining_map,
                    build_entities=should_upload,
                )
                shard_first = shard_first or first_id
                shard_last = last_id
                shard_validated += len(batch)
                category_validated += len(batch)
                total_validated += len(batch)
                all_samples.extend(samples)
                if should_upload:
                    upload_batches.append(entities)
            if shard_validated != shard.rows:
                raise ValueError(f"Validated {shard_validated} rows in {shard.key}, expected {shard.rows}")
            if shard_first != shard.first_frame_id or shard_last != shard.last_frame_id:
                raise ValueError(
                    f"First/last frame identity mismatch in {shard.key}: "
                    f"{shard_first!r}/{shard_last!r}"
                )
            if should_upload:
                upsert_shard_parallel(
                    client_boxes,
                    endpoint,
                    token,
                    args.collection,
                    upload_batches,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    timeout=args.timeout,
                )
                total_uploaded_this_run += shard.rows
                assert state is not None
                state["completed_shards"][shard.key] = {
                    "rows": shard.rows,
                    "commit_sha256": shard.commit_sha256,
                    "parquet_sha256": shard.parquet_sha256,
                    "completed_at": utc_now(),
                }
                state["uploaded_rows"] = sum(
                    int(item["rows"]) for item in state["completed_shards"].values()
                )
                state["updated_at"] = utc_now()
                atomic_write_json(args.state_file, state)
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"  progress: uploaded this run {total_uploaded_this_run:,}; "
                    f"validated {total_validated:,}/{EXPECTED_ROWS:,}; "
                    f"overall rate {total_validated / elapsed:,.1f} rows/s",
                    flush=True,
                )
            action = "uploaded" if should_upload else ("validated/resumed" if not args.dry_run else "validated")
            print(f"  {shard.key}: {action} {shard.rows:,} rows", flush=True)
        if remaining_map:
            missing = sum(len(rows) for rows in remaining_map.values())
            example_video = next(iter(remaining_map))
            example_frame = next(iter(remaining_map[example_video]))
            raise ValueError(
                f"Embedding dataset did not consume {missing:,} map rows in {category}; "
                f"example {example_video}@f{example_frame:08d}"
            )
        if category_validated != audit.map_rows[category]:
            raise ValueError(f"Category validation count mismatch for {category}")
        if not args.dry_run:
            client_boxes[0][0].flush(collection_name=args.collection, timeout=args.timeout)
        print(f"[{category}] complete: {category_validated:,} rows", flush=True)

    if total_validated != EXPECTED_ROWS:
        raise ValueError(f"Validated {total_validated:,} rows, expected {EXPECTED_ROWS:,}")
    if args.dry_run:
        elapsed = time.monotonic() - started
        print(
            f"DRY RUN PASSED: joined and validated all {total_validated:,} rows in {elapsed:.1f}s; "
            "Milvus was not contacted.",
            flush=True,
        )
        return 0

    client = client_boxes[0][0]
    client.flush(collection_name=args.collection, timeout=args.timeout)
    client.load_collection(collection_name=args.collection, timeout=args.timeout)
    final_count = wait_for_exact_count(
        client,
        args.collection,
        EXPECTED_ROWS,
        timeout_seconds=args.count_wait_timeout,
    )
    verify_collection_schema(client, args.collection)
    sample_results = verify_remote_samples(client, args.collection, all_samples)
    assert state is not None
    state["complete"] = True
    state["remote_row_count"] = final_count
    state["verified_at"] = utc_now()
    state["updated_at"] = utc_now()
    atomic_write_json(args.state_file, state)
    verification = {
        "schema_version": 1,
        "verified_at": utc_now(),
        "collection": args.collection,
        "endpoint_identity_sha256": endpoint_hash,
        "semantic_fingerprint": audit.semantic_fingerprint,
        "artifact_manifest_sha256": audit.manifest_sha256,
        "map_inventory_sha256": audit.map_inventory_sha256,
        "row_count": final_count,
        "embedding_dim": EXPECTED_DIM,
        "metric_type": "COSINE",
        "remote_samples": sample_results,
        "self_search_samples": [item.submit_keyframe_id for item in all_samples[:3]],
        "status": "PASS",
    }
    atomic_write_json(args.verification_file, verification)
    elapsed = time.monotonic() - started
    print(
        f"UPLOAD VERIFIED: {args.collection} has exactly {final_count:,} rows; "
        f"schema/index/sample retrieval PASS ({elapsed:.1f}s).",
        flush=True,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--env-file", type=Path, default=Path("backend/.env"))
    parser.add_argument("--endpoint-file", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path(".milvus_upload_state_qwen3_vl_embedding_8b.json"),
    )
    parser.add_argument(
        "--verification-file",
        type=Path,
        default=Path("milvus_upload_qwen3_vl_embedding_8b_verification.json"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Independent Milvus upload connections (checkpointing remains shard-atomic).",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--count-wait-timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-parquet-sha256",
        action="store_true",
        help="Skip the ~20.6 GiB Parquet byte hash audit (metadata and every row are still validated).",
    )
    args = parser.parse_args(argv)
    # 1,024 native 4096-d FP32 vectors already carry ~16 MiB of vector payload;
    # larger requests can cross managed-Milvus/gRPC limits after scalar fields
    # and serialization overhead are included.
    if args.batch_size <= 0 or args.batch_size > 1024:
        parser.error("--batch-size must be between 1 and 1024")
    if args.workers <= 0 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if not args.collection or len(args.collection) > 255:
        parser.error("--collection is empty or too long")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print("Interrupted. Completed shards are checkpointed; rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
