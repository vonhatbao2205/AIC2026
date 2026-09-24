#!/usr/bin/env python3
"""Audit and upsert the batch-2 PE-Core-G14-448 vectors (M, N, S01) into the
InfoShot++ image collection that retrieval profile 2 already searches.

``aic26_image_peg14_infoshotpp_v1`` (``MILVUS_IMAGE_COLLECTION_2`` on
``MILVUS_ENDPOINT_2``) holds the 1,339,055 L21–L30 vectors that
``milvus_upload_pe_core.py`` uploaded and verified. The two batch-2 artifacts are
in the same embedding space — the embedding contract (model, checkpoint,
preprocessing, precision, storage; everything but the keyframe source) is
recomputed here and must equal L's ``d940db4a…`` — so their 873,328 vectors join
that collection with its exact schema and identity rules:

* primary key ``id = submit_keyframe_id = <category>/<video_id>/<n:03d>``, with
  ``n`` taken from the pinned keyframe registry joined on ``frame_id`` — never
  derived from ``frame_idx``;
* every vector row must equal its registry row (video, category, frame_idx,
  pts_time, fps, R2 key); every registry frame at quality ≥ 0.05 is consumed
  exactly once and no frame below that floor may appear.

Like its parent this script never creates or drops a collection. It refuses a
target that is not the collection recorded in
``milvus_upload_pe_core_verification.json`` (same endpoint identity, 1,339,055 L
rows), checkpoints shard-atomically (replaying a shard is an upsert of the same
primary keys), and finishes by proving with exact ``count(*)`` queries that
batch 2 is complete and L21–L30 did not move.

Run ``--dry-run`` first (local only, every row joined), then ``--check-remote``
(read-only), then the same command without either flag to upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from batch2_keyframes import (
    DEFAULT_REGISTRY_ROOT,
    MIN_QUALITY,
    PROFILES,
    R2_BUCKET,
    KeyframeRow,
    load_registries,
    profile_for_video,
    sha256_file,
)
from milvus_upload_pe_core import (
    DEFAULT_COLLECTION,
    EXPECTED_CATEGORIES as BASE_CATEGORIES,
    EXPECTED_DIM,
    EXPECTED_ROWS as BASE_ROWS,
    VECTOR_FIELD,
    Sample,
    _batch_columns,
    _float_matches,
    _import_pyarrow,
    _read_optional_secret,
    _sample_to_expected,
    atomic_write_json,
    connect,
    endpoint_identity,
    load_or_create_state,
    parse_env_file,
    read_json,
    upsert_shard_parallel,
    utc_now,
    verify_collection_schema,
)

DEFAULT_ENCODER_ROOT = Path("/home/bao/Projects/EncoderModel")
#: SHA-256 of the semantic config minus its `source` block; identical for L, M/N and S01.
EMBEDDING_CONTRACT = "d940db4aa194deeea4daeb05c80a5a90e3e3b498e28d06afed74894aa4dd1d09"
#: The L21–L30 artifact already in the collection (its contract is also d940db4a…).
BASE_FINGERPRINT = "2cce817755d48b2a59c50a2a345d5c5a42f59fc46b4b6bd1ce3668eca1b03141"


@dataclass(frozen=True)
class Artifact:
    name: str
    directory: str
    profiles: tuple[str, ...]
    semantic_fingerprint: str
    rows: int
    #: Manifest block keyed by work unit: a category for M/N, a video for S01.
    unit_key: str


ARTIFACTS = (
    Artifact(
        name="mn",
        directory="pe-core-g14-448-batch2-mn-v1",
        profiles=("M", "N"),
        semantic_fingerprint="c0386a1e2f26cb02941dadcf928b335dc5d073b7e18ff7410033a85ac0ed37e1",
        rows=252_352,
        unit_key="per_category",
    ),
    Artifact(
        name="s01",
        directory="pe-core-g14-448-batch2-s01-v1",
        profiles=("S",),
        semantic_fingerprint="7c4d7a3ec7b5f657b08cc5bf26391c9ba4fd1c6fb02a840fbdf31b72605d3fea",
        rows=620_976,
        unit_key="per_video",
    ),
)
BATCH2_ROWS = sum(artifact.rows for artifact in ARTIFACTS)  # 873,328
FINAL_ROWS = BASE_ROWS + BATCH2_ROWS  # 2,212,383


@dataclass(frozen=True)
class Shard:
    artifact: str
    unit: str
    category: str
    profile: str
    shard_id: int
    rows: int
    first_frame_id: str
    last_frame_id: str
    parquet_path: Path
    parquet_sha256: str
    commit_sha256: str

    @property
    def key(self) -> str:
        return f"{self.artifact}/{self.unit}/part-{self.shard_id:05d}"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# local audit
# ---------------------------------------------------------------------------


def validate_manifest(artifact: Artifact, root: Path) -> tuple[dict[str, Any], str]:
    success = read_json(root / "_SUCCESS.json")
    if success != read_json(root / "embedding_dataset_manifest.json"):
        raise ValueError(f"{root}: _SUCCESS.json and embedding_dataset_manifest.json differ")
    if success.get("complete") is not True:
        raise ValueError(f"{root}: embedding dataset is not complete")
    if not int(success.get("expected_total_rows") or -1) == int(success.get("total_committed_rows") or -2) == artifact.rows:
        raise ValueError(f"{root}: expected {artifact.rows:,} committed rows")
    config = success.get("semantic_config") or {}
    if success.get("semantic_fingerprint") != artifact.semantic_fingerprint:
        raise ValueError(f"{root}: semantic_fingerprint differs from the audited artifact")
    if canonical_sha256(config) != artifact.semantic_fingerprint:
        raise ValueError(f"{root}: semantic_config does not hash to its semantic_fingerprint")
    contract = canonical_sha256({key: value for key, value in config.items() if key != "source"})
    if contract != EMBEDDING_CONTRACT or success.get("embedding_contract_sha256") != EMBEDDING_CONTRACT:
        raise ValueError(f"{root}: embedding contract {contract[:12]}… is not L's {EMBEDDING_CONTRACT[:12]}…")
    if int((config.get("model") or {}).get("embedding_dim") or -1) != EXPECTED_DIM:
        raise ValueError(f"{root}: expected {EXPECTED_DIM}-d vectors")

    source = config.get("source") or {}
    if source.get("r2_bucket") != R2_BUCKET or (source.get("selection") or {}).get("min_quality") != MIN_QUALITY:
        raise ValueError(f"{root}: unexpected keyframe source or quality floor")
    profiles = source.get("profiles") or {}
    if tuple(sorted(profiles)) != tuple(sorted(artifact.profiles)):
        raise ValueError(f"{root}: source profiles {sorted(profiles)}, expected {list(artifact.profiles)}")
    for name in artifact.profiles:
        pinned, declared = PROFILES[name], profiles[name]
        wanted = {
            "frame_registry_sha256": pinned.registry_sha256,
            "pipeline_signature": pinned.pipeline_signature,
            "registry_rows": pinned.rows,
            "videos": pinned.videos,
        }
        for key, value in wanted.items():
            if declared.get(key) != value:
                raise ValueError(f"{root}: profile {name} {key}={declared.get(key)!r}, expected {value!r}")

    expected_counts = source.get("expected_counts") or {}
    units = success.get(artifact.unit_key) or {}
    if set(expected_counts) != set(units) or sum(expected_counts.values()) != artifact.rows:
        raise ValueError(f"{root}: work units in expected_counts and {artifact.unit_key} differ")
    for unit, record in units.items():
        if record.get("complete") is not True or int(record.get("committed_rows") or -1) != expected_counts[unit]:
            raise ValueError(f"{root}: work unit {unit} is incomplete")
    return success, sha256_file(root / "embedding_dataset_manifest.json")


def audit_shards(artifact: Artifact, root: Path, manifest: dict[str, Any], *, verify_sha256: bool) -> list[Shard]:
    _, pq = _import_pyarrow()
    units = manifest[artifact.unit_key]
    expected_counts = manifest["semantic_config"]["source"]["expected_counts"]
    commits = sorted((root / "commits").glob("*/part-*.json"))
    if len(commits) != sum(int(record["expected_shards"]) for record in units.values()):
        raise ValueError(f"{root}: {len(commits)} commits, manifest expects a different shard count")

    shards: list[Shard] = []
    next_row: dict[str, int] = {}
    next_shard: dict[str, int] = {}
    for commit_path in commits:
        commit = read_json(commit_path)
        unit = str(commit.get("work_unit") or commit.get("category") or "")
        if unit != commit_path.parent.name or unit not in units:
            raise ValueError(f"{commit_path}: unknown work unit {unit!r}")
        profile, category = profile_for_video(str(commit.get("first_frame_id") or "").split("@", 1)[0])
        if category != commit.get("category") or profile.name not in artifact.profiles:
            raise ValueError(f"{commit_path}: category/profile mismatch")
        if artifact.unit_key == "per_category" and unit != category:
            raise ValueError(f"{commit_path}: work unit {unit} is not category {category}")
        shard_id = int(commit.get("shard_id", -1))
        row_start, row_stop, rows = (int(commit.get(key, -1)) for key in ("row_start", "row_stop", "rows"))
        if (
            shard_id != next_shard.get(unit, 0)
            or row_start != next_row.get(unit, 0)
            or row_stop - row_start != rows
            or rows <= 0
        ):
            raise ValueError(f"{commit_path}: non-contiguous shard id or row range")
        if commit.get("semantic_fingerprint") != artifact.semantic_fingerprint:
            raise ValueError(f"{commit_path}: fingerprint mismatch")
        if commit.get("source_pipeline_signature") != profile.pipeline_signature:
            raise ValueError(f"{commit_path}: keyframe pipeline signature mismatch")
        if int(commit.get("embedding_dim") or -1) != EXPECTED_DIM or commit.get("l2_normalized") is not True:
            raise ValueError(f"{commit_path}: embedding storage contract mismatch")

        parquet_path = root / "embeddings" / unit / Path(str(commit.get("parquet_path") or "")).name
        if not parquet_path.is_file() or parquet_path.stat().st_size != int(commit.get("parquet_bytes") or -1):
            raise ValueError(f"{commit_path}: Parquet missing or size differs: {parquet_path}")
        if verify_sha256 and sha256_file(parquet_path) != commit.get("parquet_sha256"):
            raise ValueError(f"Parquet SHA-256 mismatch: {parquet_path}")
        parquet = pq.ParquetFile(parquet_path)
        schema = parquet.schema_arrow
        metadata = schema.metadata or {}
        required = {"frame_id", "video_id", "category", "frame_idx", "pts_time", "fps", "image_relpath", VECTOR_FIELD}
        if parquet.metadata.num_rows != rows or not required.issubset(schema.names):
            raise ValueError(f"{parquet_path}: row count or columns differ from the commit")
        if getattr(schema.field(VECTOR_FIELD).type, "list_size", None) != EXPECTED_DIM:
            raise ValueError(f"{parquet_path}: embedding is not fixed_size_list[{EXPECTED_DIM}]")
        if (
            metadata.get(b"semantic_fingerprint", b"").decode() != artifact.semantic_fingerprint
            or metadata.get(b"source_pipeline_signature", b"").decode() != profile.pipeline_signature
        ):
            raise ValueError(f"{parquet_path}: Arrow metadata fingerprint/signature mismatch")

        shards.append(
            Shard(
                artifact=artifact.name,
                unit=unit,
                category=category,
                profile=profile.name,
                shard_id=shard_id,
                rows=rows,
                first_frame_id=str(commit.get("first_frame_id")),
                last_frame_id=str(commit.get("last_frame_id")),
                parquet_path=parquet_path,
                parquet_sha256=str(commit.get("parquet_sha256")),
                commit_sha256=sha256_file(commit_path),
            )
        )
        next_row[unit] = row_stop
        next_shard[unit] = shard_id + 1
    for unit, expected in expected_counts.items():
        if next_row.get(unit) != int(expected):
            raise ValueError(f"{root}: work unit {unit} has {next_row.get(unit, 0):,} rows, expected {expected:,}")
    return shards


class DuplicateCounter:
    """Rows whose vector is byte-identical to an earlier row of the same video.

    Reported, not filtered: the handoff (§8.1) measures 9,450 in S01 — identical
    JPEGs, mostly a static "TRỰC TIẾP" card — and none in M/N.
    """

    def __init__(self) -> None:
        self._seen: dict[str, set[bytes]] = {}
        self.by_profile: Counter[str] = Counter()

    def reset(self) -> None:
        self._seen.clear()

    def add(self, profile: str, video_id: str, vector: np.ndarray) -> None:
        key = hashlib.blake2b(vector.tobytes(), digest_size=16).digest()
        seen = self._seen.setdefault(video_id, set())
        if key in seen:
            self.by_profile[profile] += 1
        else:
            seen.add(key)


def validate_and_build_entities(
    batch: Any,
    remaining: dict[str, KeyframeRow],
    *,
    build_entities: bool,
    duplicates: DuplicateCounter | None = None,
) -> tuple[list[dict[str, Any]], list[Sample], str, str]:
    """Join one record batch on ``frame_id`` against the embedded registry rows (consumed)."""
    columns = _batch_columns(batch)
    vectors = np.asarray(columns[VECTOR_FIELD], dtype=np.float32)
    if vectors.shape != (len(batch), EXPECTED_DIM) or not np.isfinite(vectors).all():
        raise ValueError(f"Bad embedding batch: shape {vectors.shape} or non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=2e-5, atol=2e-5):
        raise ValueError(f"Embedding norms outside tolerance: [{norms.min()}, {norms.max()}]")

    entities: list[dict[str, Any]] = []
    samples: list[Sample] = []
    for index in range(len(batch)):
        frame_id = str(columns["frame_id"][index])
        row = remaining.pop(frame_id, None)
        if row is None:
            raise ValueError(f"{frame_id}: no embedded registry row (unknown, duplicated, or below quality floor)")
        observed = {
            "video_id": str(columns["video_id"][index]),
            "category": str(columns["category"][index]),
            "frame_idx": int(columns["frame_idx"][index]),
            "image_relpath": str(columns["image_relpath"][index]),
        }
        wanted = {"video_id": row.video_id, "category": row.category, "frame_idx": row.frame_idx, "image_relpath": row.r2_key}
        if observed != wanted:
            raise ValueError(f"{frame_id}: vector row {observed} differs from registry {wanted}")
        pts_time = float(columns["pts_time"][index])
        fps = float(columns["fps"][index])
        if not _float_matches(pts_time, row.pts_time, tolerance=2e-5) or not _float_matches(fps, row.fps, tolerance=2e-5):
            raise ValueError(f"{frame_id}: pts_time/fps {pts_time}/{fps} differ from registry {row.pts_time}/{row.fps}")
        if duplicates is not None:
            duplicates.add(row.profile, row.video_id, vectors[index])
        if build_entities:
            entities.append(
                {
                    "id": row.submit_keyframe_id,
                    "image_id": row.submit_keyframe_id,
                    "keyframe_id": row.keyframe_id,
                    "submit_keyframe_id": row.submit_keyframe_id,
                    "frame_id": row.frame_id,
                    "category": row.category,
                    "video_id": row.video_id,
                    "keyframe_n": row.n,
                    "frame_idx": row.frame_idx,
                    "pts_time": row.pts_time,
                    "fps": row.fps,
                    "image_path": row.r2_key,
                    VECTOR_FIELD: vectors[index],
                }
            )
        if index == 0:
            samples.append(
                Sample(
                    submit_keyframe_id=row.submit_keyframe_id,
                    frame_id=row.frame_id,
                    video_id=row.video_id,
                    category=row.category,
                    keyframe_n=row.n,
                    frame_idx=row.frame_idx,
                    pts_time=row.pts_time,
                    fps=row.fps,
                    vector=vectors[index].copy(),
                )
            )
    first = str(columns["frame_id"][0]) if len(batch) else ""
    last = str(columns["frame_id"][len(batch) - 1]) if len(batch) else ""
    return entities, samples, first, last


# ---------------------------------------------------------------------------
# remote (Milvus)
# ---------------------------------------------------------------------------


def in_expr(field: str, values: Sequence[str]) -> str:
    return f"{field} in [{', '.join(json.dumps(value) for value in sorted(values))}]"


def count_where(client: Any, collection: str, expr: str, *, timeout: float) -> int:
    rows = client.query(
        collection_name=collection,
        filter=expr,
        output_fields=["count(*)"],
        timeout=timeout,
        consistency_level="Strong",
    )
    return int(rows[0]["count(*)"])


def remote_counts(client: Any, collection: str, batch2_categories: Sequence[str], *, timeout: float) -> dict[str, int]:
    total = count_where(client, collection, "", timeout=timeout)
    base = count_where(client, collection, in_expr("category", BASE_CATEGORIES), timeout=timeout)
    batch2 = count_where(client, collection, in_expr("category", batch2_categories), timeout=timeout)
    return {"total": total, "base_l21_l30": base, "batch2": batch2, "other": total - base - batch2}


def resolve_credentials(args: argparse.Namespace) -> tuple[str, str]:
    """Profile-2 credentials: explicit files, then the environment, then backend/.env."""
    env_file = parse_env_file(args.env_file)
    endpoint = (
        _read_optional_secret(args.endpoint_file)
        or os.environ.get("MILVUS_ENDPOINT_2", "").strip()
        or env_file.get("MILVUS_ENDPOINT_2", "").strip()
    )
    token = (
        _read_optional_secret(args.token_file)
        or os.environ.get("MILVUS_TOKEN_2", "").strip()
        or env_file.get("MILVUS_TOKEN_2", "").strip()
    )
    if not endpoint or not token:
        raise ValueError("Set MILVUS_ENDPOINT_2/MILVUS_TOKEN_2 (environment or --env-file) or pass credential files.")
    return endpoint, token


def check_base_verification(path: Path, collection: str, endpoint_hash: str) -> dict[str, Any]:
    """The collection must be the one milvus_upload_pe_core.py verified with the L vectors."""
    base = read_json(path)
    wanted = {
        "status": "PASS",
        "collection": collection,
        "endpoint_identity_sha256": endpoint_hash,
        "row_count": BASE_ROWS,
        "semantic_fingerprint": BASE_FINGERPRINT,
        "embedding_dim": EXPECTED_DIM,
        "metric_type": "COSINE",
    }
    for key, value in wanted.items():
        if base.get(key) != value:
            raise ValueError(
                f"{path}: {key}={base.get(key)!r}, expected {value!r}. This is not the verified "
                "L21–L30 collection on this endpoint; refusing to extend it."
            )
    return base


def preflight_remote(
    client: Any, collection: str, base: dict[str, Any], batch2_categories: Sequence[str], *, timeout: float
) -> dict[str, int]:
    if not client.has_collection(collection_name=collection):
        raise ValueError(f"Collection {collection} does not exist; upload L21–L30 with milvus_upload_pe_core.py first.")
    verify_collection_schema(client, collection)
    counts = remote_counts(client, collection, batch2_categories, timeout=timeout)
    print(f"[{collection}] remote before upload: {json.dumps(counts)}", flush=True)
    if counts["base_l21_l30"] != BASE_ROWS or counts["other"]:
        raise ValueError(
            f"{collection} holds {counts['base_l21_l30']:,} L21–L30 rows and {counts['other']:,} unknown rows; "
            f"expected {BASE_ROWS:,} and 0. Refusing."
        )
    base_ids = [item["submit_keyframe_id"] for item in base.get("remote_samples") or []]
    found = {row.get("id") for row in client.get(collection_name=collection, ids=base_ids, output_fields=["id"])}
    if not base_ids or found != set(base_ids):
        raise ValueError(f"L21–L30 sample ids recorded at verification are missing: {sorted(set(base_ids) - found)}")
    return counts


def verify_remote_samples(client: Any, collection: str, samples: Sequence[Sample]) -> list[dict[str, Any]]:
    """Field-exact read-back of one row per category, plus a self-search per profile."""
    per_category: dict[str, Sample] = {}
    for sample in samples:
        per_category.setdefault(sample.category, sample)
    selected = list(per_category.values())
    output_fields = [*_sample_to_expected(selected[0]), VECTOR_FIELD]
    rows = client.get(collection_name=collection, ids=[item.submit_keyframe_id for item in selected], output_fields=output_fields)
    by_id = {str(row.get("id")): row for row in rows or []}
    results: list[dict[str, Any]] = []
    for sample in selected:
        row = by_id.get(sample.submit_keyframe_id)
        if row is None:
            raise ValueError(f"Remote sample is missing: {sample.submit_keyframe_id}")
        for key, value in _sample_to_expected(sample).items():
            actual = row.get(key)
            same = _float_matches(float(actual), value, tolerance=2e-5) if isinstance(value, float) else actual == value
            if not same:
                raise ValueError(f"Remote sample {sample.submit_keyframe_id} differs at {key}: {actual!r} != {value!r}")
        remote = np.asarray(row.get(VECTOR_FIELD), dtype=np.float32)
        cosine = float(np.dot(remote, sample.vector) / (np.linalg.norm(remote) * np.linalg.norm(sample.vector)))
        if cosine < 0.99999:
            raise ValueError(f"Remote vector differs for {sample.submit_keyframe_id}: cosine={cosine}")
        results.append({"submit_keyframe_id": sample.submit_keyframe_id, "vector_cosine": cosine})

    # Byte-identical S01 duplicates (and near-identical intro cards) tie at 1.0, so
    # the sample must be among the top hits, not necessarily the first.
    first_per_profile: dict[str, Sample] = {}
    for sample in selected:
        first_per_profile.setdefault(profile_for_video(sample.video_id)[0].name, sample)
    for sample in first_per_profile.values():
        hits = client.search(
            collection_name=collection,
            data=[sample.vector],
            anns_field=VECTOR_FIELD,
            limit=10,
            output_fields=["submit_keyframe_id"],
            search_params={"metric_type": "COSINE"},
        )[0]
        scores = {(hit.get("entity") or {}).get("submit_keyframe_id") or hit.get("id"): float(hit.get("distance", 0.0)) for hit in hits}
        if scores.get(sample.submit_keyframe_id, 0.0) < 0.9999:
            raise ValueError(f"Self-search did not return {sample.submit_keyframe_id}: {scores}")
    return results


def wait_for_counts(
    client: Any, collection: str, expected: dict[str, int], batch2_categories: Sequence[str], *, timeout: float, wait: float
) -> dict[str, int]:
    deadline = time.monotonic() + wait
    while True:
        counts = remote_counts(client, collection, batch2_categories, timeout=timeout)
        if counts == expected:
            return counts
        if counts["total"] > expected["total"] or counts["other"] or time.monotonic() > deadline:
            raise ValueError(f"Remote counts {counts} != expected {expected}")
        print(f"Waiting for Milvus counts: {counts}", flush=True)
        time.sleep(5)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    registries = load_registries(root=args.registry_root)
    batch2_categories = sorted({row.category for rows in registries.values() for row in rows})
    manifests: dict[str, tuple[dict[str, Any], str]] = {}
    shards: list[Shard] = []
    for artifact in ARTIFACTS:
        root = args.encoder_root / artifact.directory
        manifests[artifact.name] = validate_manifest(artifact, root)
        artifact_shards = audit_shards(
            artifact, root, manifests[artifact.name][0], verify_sha256=not args.skip_parquet_sha256
        )
        shards.extend(artifact_shards)
        print(
            f"[artifact {artifact.name}] {len(artifact_shards)} shards, {artifact.rows:,} rows, "
            f"contract {EMBEDDING_CONTRACT[:12]}… = L — manifest/commit/Parquet audit PASS",
            flush=True,
        )

    endpoint = token = ""
    endpoint_hash = "dry-run"
    client_boxes: list[list[Any]] = []
    state: dict[str, Any] | None = None
    before: dict[str, int] = {}
    if not args.dry_run:
        endpoint, token = resolve_credentials(args)
        endpoint_hash = endpoint_identity(endpoint)
        base = check_base_verification(args.base_verification_file, args.collection, endpoint_hash)
        client_boxes = [[connect(endpoint, token, timeout=args.timeout)]]
        before = preflight_remote(client_boxes[0][0], args.collection, base, batch2_categories, timeout=args.timeout)
        if args.check_remote:
            print("REMOTE CHECK PASSED (read-only); nothing was written.", flush=True)
            return 0
        binding = {
            "collection": args.collection,
            "endpoint_identity_sha256": endpoint_hash,
            "embedding_contract_sha256": EMBEDDING_CONTRACT,
            "embedding_dim": EXPECTED_DIM,
            "metric_type": "COSINE",
            "base_rows": BASE_ROWS,
            "expected_batch2_rows": BATCH2_ROWS,
            "artifacts": {
                name: {"manifest_sha256": sha, "semantic_fingerprint": manifest["semantic_fingerprint"]}
                for name, (manifest, sha) in manifests.items()
            },
            "registry_sha256": {name: profile.registry_sha256 for name, profile in PROFILES.items()},
        }
        state, created = load_or_create_state(args.state_file, binding)
        if created and before["batch2"]:
            print(
                f"Note: {before['batch2']:,} batch-2 rows exist without local state; "
                "re-upserting all of them (same primary keys, idempotent).",
                flush=True,
            )
        for _ in range(1, args.workers):
            client_boxes.append([connect(endpoint, token, timeout=args.timeout)])

    _, pq = _import_pyarrow()
    completed = (state or {}).get("completed_shards") or {}
    duplicates = DuplicateCounter()
    all_samples: list[Sample] = []
    validated = uploaded = 0
    for artifact in ARTIFACTS:
        remaining = {
            row.frame_id: row for name in artifact.profiles for row in registries[name] if row.embedded
        }
        if len(remaining) != artifact.rows:
            raise ValueError(f"Registry has {len(remaining):,} embeddable frames for {artifact.name}, expected {artifact.rows:,}")
        current_unit = ""
        for shard in (item for item in shards if item.artifact == artifact.name):
            if shard.unit != current_unit:
                duplicates.reset()
                current_unit = shard.unit
            should_upload = not args.dry_run and shard.key not in completed
            batches: list[list[dict[str, Any]]] = []
            shard_rows = 0
            shard_first = shard_last = ""
            for batch in pq.ParquetFile(shard.parquet_path).iter_batches(batch_size=args.batch_size):
                entities, samples, first, last = validate_and_build_entities(
                    batch, remaining, build_entities=should_upload, duplicates=duplicates
                )
                shard_first = shard_first or first
                shard_last = last
                shard_rows += len(batch)
                all_samples.extend(samples)
                if should_upload:
                    batches.append(entities)
            if shard_rows != shard.rows or (shard_first, shard_last) != (shard.first_frame_id, shard.last_frame_id):
                raise ValueError(f"{shard.key}: rows or first/last frame_id differ from its commit")
            validated += shard_rows
            if should_upload:
                upsert_shard_parallel(
                    client_boxes,
                    endpoint,
                    token,
                    args.collection,
                    batches,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    timeout=args.timeout,
                )
                uploaded += shard.rows
                assert state is not None
                state["completed_shards"][shard.key] = {
                    "rows": shard.rows,
                    "commit_sha256": shard.commit_sha256,
                    "parquet_sha256": shard.parquet_sha256,
                    "completed_at": utc_now(),
                }
                state["uploaded_rows"] = sum(int(item["rows"]) for item in state["completed_shards"].values())
                state["updated_at"] = utc_now()
                atomic_write_json(args.state_file, state)
                rate = validated / max(time.monotonic() - started, 0.001)
                print(
                    f"  {shard.key}: uploaded {shard.rows:,} (this run {uploaded:,}; "
                    f"validated {validated:,}/{BATCH2_ROWS:,}; {rate:,.0f} rows/s)",
                    flush=True,
                )
        if remaining:
            example = next(iter(remaining))
            raise ValueError(f"{artifact.name}: {len(remaining):,} embeddable registry frames have no vector, e.g. {example}")
        if not args.dry_run:
            client_boxes[0][0].flush(collection_name=args.collection, timeout=args.timeout)
        print(f"[artifact {artifact.name}] every row joined to the registry ({artifact.rows:,})", flush=True)

    duplicate_report = {name: duplicates.by_profile.get(name, 0) for name in PROFILES}
    print(f"Byte-identical vectors within a video (kept, handoff §8.1): {duplicate_report}", flush=True)
    if validated != BATCH2_ROWS:
        raise ValueError(f"Validated {validated:,} rows, expected {BATCH2_ROWS:,}")
    if args.dry_run:
        print(
            f"DRY RUN PASSED: joined and validated all {validated:,} batch-2 vectors in "
            f"{time.monotonic() - started:.1f}s; Milvus was not contacted.",
            flush=True,
        )
        return 0

    client = client_boxes[0][0]
    client.flush(collection_name=args.collection, timeout=args.timeout)
    client.load_collection(collection_name=args.collection, timeout=args.timeout)
    expected = {"total": FINAL_ROWS, "base_l21_l30": BASE_ROWS, "batch2": BATCH2_ROWS, "other": 0}
    final = wait_for_counts(
        client, args.collection, expected, batch2_categories, timeout=args.timeout, wait=args.count_wait_timeout
    )
    by_profile = {
        name: count_where(
            client, args.collection, in_expr("category", sorted({row.category for row in registries[name]})), timeout=args.timeout
        )
        for name in PROFILES
    }
    wanted_by_profile = {name: sum(row.embedded for row in registries[name]) for name in PROFILES}
    if by_profile != wanted_by_profile:
        raise ValueError(f"Per-profile counts {by_profile} != expected {wanted_by_profile}")
    verify_collection_schema(client, args.collection)
    sample_results = verify_remote_samples(client, args.collection, all_samples)
    assert state is not None
    state.update(complete=True, remote_counts=final, verified_at=utc_now(), updated_at=utc_now())
    atomic_write_json(args.state_file, state)
    atomic_write_json(
        args.verification_file,
        {
            "schema_version": 1,
            "status": "PASS",
            "verified_at": utc_now(),
            "collection": args.collection,
            "endpoint_identity_sha256": endpoint_hash,
            "embedding_contract_sha256": EMBEDDING_CONTRACT,
            "artifacts": state["binding"]["artifacts"],
            "registry_sha256": state["binding"]["registry_sha256"],
            "before": before,
            "after": final,
            "batch2_by_profile": by_profile,
            "byte_identical_vectors_kept": duplicate_report,
            "remote_samples": sample_results,
        },
    )
    print(
        f"UPLOAD VERIFIED: {args.collection} holds {final['total']:,} rows "
        f"(L21–L30 {final['base_l21_l30']:,} + batch 2 {final['batch2']:,}); "
        f"schema/count/sample/self-search PASS ({time.monotonic() - started:.1f}s).",
        flush=True,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder-root", type=Path, default=DEFAULT_ENCODER_ROOT)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--env-file", type=Path, default=Path("backend/.env"))
    parser.add_argument("--endpoint-file", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument(
        "--base-verification-file",
        type=Path,
        default=Path("milvus_upload_pe_core_verification.json"),
        help="Verification written by milvus_upload_pe_core.py for the L21–L30 upload.",
    )
    parser.add_argument("--state-file", type=Path, default=Path(".milvus_upload_state_pe_core_batch2.json"))
    parser.add_argument(
        "--verification-file", type=Path, default=Path("milvus_upload_pe_core_batch2_verification.json")
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4, help="Independent upload connections (1–8).")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--count-wait-timeout", type=float, default=900.0)
    parser.add_argument(
        "--skip-parquet-sha256",
        action="store_true",
        help="Skip hashing the 4.3 GiB of Parquet (sizes, metadata and every row are still validated).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Full local audit and join; Milvus is not contacted.")
    mode.add_argument("--check-remote", action="store_true", help="Local manifest audit + read-only remote checks.")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 4096:
        parser.error("--batch-size must be between 1 and 4096")
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
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
