#!/usr/bin/env python3
"""Audit and upsert the batch-2 TARA clip vectors (M, N, S01) into the InfoShot++
TARA collection that already holds the L21–L30 clips.

``aic26_tara_clips_infoshotpp_v1`` holds the 168,536 L21–L30 clips that
``milvus_upload_tara.py`` uploaded and verified. Both batch-2 artifacts share L's
embedding contract ``611ef1c9…`` (model, prompt, frames, pixels, pooling, storage;
everything but the video source and the clip plan), recomputed here from each
``run_config.json``, so their clips join that collection with its exact schema.

M/N has no ``_SUCCESS.json`` and S01 is a partial snapshot (handoff §8.1, §8.4),
so the audit is commit-based: every committed shard and nothing else, with its
size, SHA-256, Arrow fingerprint and rows checked. The 1,591 byte-identical
black-tail clips of M10_V029 (handoff §8.2) are not indexed.

Like its parent this script never creates or drops a collection. It refuses a
collection whose L21–L30 rows are not the verified 168,536, checkpoints per
shard (replaying one is an upsert of the same clip_ids) and ends by proving with
exact ``count(*)`` queries that batch 2 is complete and L21–L30 did not move.
When more S01 shards are committed, raise the S01 counts below and rerun: shards
already in the state file are skipped.

Run from the repository root with backend/.venv/bin/python: ``--dry-run`` first
(local only), then ``--check-remote`` (read-only), then without either flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from milvus_upload_tara import COLLECTION, credentials, ensure_collection, write_state
from tara_artifact import CATEGORIES as L_CATEGORIES
from tara_artifact import EXPECTED_DIM, EXPECTED_ROWS as L_ROWS, REQUIRED_COLUMNS, sha256_file

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
L_STATE = ROOT / "tara_embeddings" / "milvus_upload_state.json"
L_FINGERPRINT = "ef197331649dfb93e7057304f294d8bd1cd98bf22b1902bca48daf6dc4874819"
#: SHA-256 of semantic_config minus `source` and `clip_plan`; identical for L, M/N and S01.
EMBEDDING_CONTRACT = "611ef1c9b8dac812723d0bd81bacff65346025a0ed6159e3a8cc7f571618e0b3"
SCALES = ("event", "sequence", "scene")


@dataclass(frozen=True)
class Artifact:
    name: str
    directory: str
    semantic_fingerprint: str
    #: Committed rows in the 2026-09-25 snapshot (handoff §1).
    rows: int


ARTIFACTS = (
    Artifact("mn", "tara-tarsier2-7b-3584-batch2-clip-v1",
             "ef1af4ae8bbcd96fba72d78842ea11a93d837cf981e485a619a3e461723b70ee", 283_454),
    Artifact("s01", "tara-tarsier2-7b-3584-batch2-s01-clip-v1",
             "e99c2efaf4395c4e850d6659413906f87c31dc62f5925804b5d6b42fe9643c0a", 45_756),
)
BATCH2_CATEGORIES = frozenset(
    [f"M{i:02d}" for i in range(1, 11)] + [f"N{i:03d}" for i in range(1, 101)] + ["S01"]
)
# Handoff §8.2: from 1112 s on, M10_V029 is a black screen and every clip has the same vector.
BLACK_TAIL_VIDEO = "M10_V029"
BLACK_TAIL_START = 1112.0
BLACK_TAIL_ROWS = 1_591
#: Indexed rows per set, after the black tail is dropped.
EXPECTED_BY_SET = {"M": 128_945 - BLACK_TAIL_ROWS, "N": 154_509, "S01": 45_756}
BATCH2_ROWS = sum(EXPECTED_BY_SET.values())  # 327,619
FINAL_ROWS = L_ROWS + BATCH2_ROWS  # 496,155
SET_FILTERS = {"M": 'category like "M%"', "N": 'category like "N%"', "S01": 'category == "S01"'}
L_FILTER = f"category in {json.dumps(list(L_CATEGORIES))}"
BLACK_TAIL_FILTER = f'video_id == "{BLACK_TAIL_VIDEO}" and start_time >= {BLACK_TAIL_START}'
OUTPUT_FIELDS = ["clip_id", "video_id", "category", "scale", "scale_index",
                 "start_time", "end_time", "fps", "center_frame_idx"]
_GROUP = re.compile(r"[_-]")


def set_of(category: str) -> str:
    return "S01" if category == "S01" else category[0]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Shard:
    artifact: Artifact
    unit: str
    shard_id: int
    rows: int
    first_clip_id: str
    last_clip_id: str
    parquet: Path
    parquet_sha256: str

    @property
    def key(self) -> str:
        return f"{self.artifact.name}/{self.unit}/part-{self.shard_id:05d}"


def audit_commits(artifact: Artifact, root: Path, *, verify_sha256: bool) -> list[Shard]:
    """Fingerprint, contract and every commit of one artifact; rows are checked later."""
    run = read_json(root / "run_config.json")
    config = run["semantic_config"]
    fingerprint = artifact.semantic_fingerprint
    if canonical_sha256(config) != fingerprint or run.get("semantic_fingerprint") != fingerprint:
        raise ValueError(f"{root}: semantic_config does not hash to {fingerprint[:12]}…")
    contract = canonical_sha256({k: v for k, v in config.items() if k not in ("source", "clip_plan")})
    if contract != EMBEDDING_CONTRACT:
        raise ValueError(f"{root}: embedding contract {contract[:12]}… is not L's {EMBEDDING_CONTRACT[:12]}…")

    prefix = f"derived/{artifact.directory}/"
    shards: list[Shard] = []
    for unit_dir in sorted(p for p in (root / "commits").iterdir() if p.is_dir()):
        unit = unit_dir.name
        next_row = 0
        for shard_id, commit_path in enumerate(sorted(unit_dir.glob("part-*.json"))):
            commit = read_json(commit_path)
            rows = int(commit.get("rows", -1))
            if (commit.get("shard_id") != shard_id
                    or commit.get("work_unit", commit.get("category")) != unit
                    or commit.get("semantic_fingerprint") != fingerprint
                    or commit.get("row_start") != next_row
                    or commit.get("row_stop") != next_row + rows or rows <= 0
                    or commit.get("embedding_dim") != EXPECTED_DIM
                    or commit.get("embedding_dtype") != "float32"
                    or commit.get("l2_normalized") is not True):
                raise ValueError(f"{commit_path}: invalid commit contract or non-contiguous rows")
            relpath = str(commit.get("parquet_path", "")).removeprefix(prefix)
            if relpath != f"embeddings/{unit}/part-{shard_id:05d}.parquet":
                raise ValueError(f"{commit_path}: unexpected parquet path {commit.get('parquet_path')!r}")
            parquet = root / relpath
            if not parquet.is_file() or parquet.stat().st_size != commit.get("parquet_bytes"):
                raise ValueError(f"{commit_path}: parquet missing or wrong size")
            if verify_sha256 and sha256_file(parquet) != commit.get("parquet_sha256"):
                raise ValueError(f"{parquet}: SHA-256 differs from its commit")
            shards.append(Shard(artifact, unit, shard_id, rows, commit["first_clip_id"],
                                commit["last_clip_id"], parquet, commit["parquet_sha256"]))
            next_row += rows
        success = root / "success" / f"{unit}.json"
        if success.is_file() and read_json(success).get("rows") != next_row:
            raise ValueError(f"{success}: rows differ from the {next_row:,} committed")
    total = sum(shard.rows for shard in shards)
    if total != artifact.rows:
        raise ValueError(f"{root}: {total:,} committed rows, snapshot has {artifact.rows:,}")
    loose = len(list((root / "embeddings").glob("*/*.parquet"))) - len(shards)
    if loose:
        print(f"[{artifact.name}] ignoring {loose} parquet file(s) without a commit", flush=True)
    return shards


@dataclass
class Checked:
    entities: list[dict[str, Any]]
    kept: int
    #: The shard's first indexed row, as uploaded (read back after the upload).
    sample: dict[str, Any] | None
    black_tail: list[bytes]


def check_shard(shard: Shard, seen_clips: set[str], *, build: bool) -> Checked:
    """Validate every row of one shard; drop the black tail; optionally build entities."""
    table = pq.read_table(shard.parquet)
    metadata = table.schema.metadata or {}
    if set(table.column_names) != REQUIRED_COLUMNS or table.num_rows != shard.rows:
        raise ValueError(f"{shard.key}: columns or row count differ from the commit")
    if metadata.get(b"semantic_fingerprint", b"").decode() != shard.artifact.semantic_fingerprint:
        raise ValueError(f"{shard.key}: Arrow fingerprint mismatch")
    embedding = table["embedding"].combine_chunks()
    frame_column = table["frame_indices"].combine_chunks()
    if embedding.type.list_size != EXPECTED_DIM or frame_column.type.list_size != 8:
        raise ValueError(f"{shard.key}: vector or frame_indices size mismatch")
    vectors = embedding.values.to_numpy().reshape(-1, EXPECTED_DIM)
    frames = frame_column.values.to_numpy().reshape(-1, 8)
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError(f"{shard.key}: invalid FP32 vectors")
    norms = np.linalg.norm(vectors, axis=1)
    if np.max(np.abs(norms - 1)) > 2e-5:
        raise ValueError(f"{shard.key}: vectors are not unit norm")

    column = {name: table[name].to_pylist() for name in OUTPUT_FIELDS if name != "center_frame_idx"}
    clips = column["clip_id"]
    if (clips[0], clips[-1]) != (shard.first_clip_id, shard.last_clip_id):
        raise ValueError(f"{shard.key}: first/last clip_id differ from the commit")
    if len(set(clips)) != len(clips) or not seen_clips.isdisjoint(clips):
        raise ValueError(f"{shard.key}: duplicate clip_id")
    seen_clips.update(clips)

    result = Checked([], 0, None, [])
    for i, clip_id in enumerate(clips):
        video_id, category, scale = column["video_id"][i], column["category"][i], column["scale"][i]
        start, end = float(column["start_time"][i]), float(column["end_time"][i])
        unit = category if shard.artifact.name == "mn" else video_id
        if (category not in BATCH2_CATEGORIES or unit != shard.unit
                or _GROUP.split(video_id, maxsplit=1)[0] != category
                or scale not in SCALES or column["scale_index"][i] != SCALES.index(scale)
                or not 0 <= start < end or not clip_id.startswith(f"{video_id}@{scale}@t")
                or abs(int(clip_id.rsplit("@t", 1)[1]) / 1000 - start) > 1e-3
                or np.any(np.diff(frames[i]) < 0)):
            raise ValueError(f"{shard.key}: row {clip_id!r} breaks the clip identity contract")
        if video_id == BLACK_TAIL_VIDEO and start >= BLACK_TAIL_START:
            result.black_tail.append(hashlib.blake2b(vectors[i].tobytes(), digest_size=16).digest())
            continue
        result.kept += 1
        if build or result.sample is None:
            entity = {name: column[name][i] for name in column}
            entity["center_frame_idx"] = int(frames[i, 4])
            entity["embedding"] = vectors[i]
            if result.sample is None:
                result.sample = entity
            if build:
                result.entities.append(entity)
    return result


_local = threading.local()


def upsert(uri: str, token: str, collection: str, entities: list[dict[str, Any]],
           batch_size: int, timeout: float) -> int:
    from pymilvus import MilvusClient

    if not hasattr(_local, "client"):
        _local.client = MilvusClient(uri=uri, token=token)
    for offset in range(0, len(entities), batch_size):
        batch = entities[offset:offset + batch_size]
        for attempt in range(5):
            try:
                result = _local.client.upsert(collection_name=collection, data=batch, timeout=timeout)
                reported = result.get("upsert_count") if isinstance(result, dict) else None
                if reported is not None and int(reported) != len(batch):
                    raise RuntimeError(f"Milvus upserted {reported} of {len(batch)} rows")
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
                _local.client = MilvusClient(uri=uri, token=token)
    return len(entities)


def count(client: Any, collection: str, expr: str) -> int:
    rows = client.query(collection_name=collection, filter=expr, output_fields=["count(*)"],
                        consistency_level="Strong")
    return int(rows[0]["count(*)"])


def remote_counts(client: Any, collection: str) -> dict[str, int]:
    counts = {"total": count(client, collection, ""), "L": count(client, collection, L_FILTER)}
    counts.update({name: count(client, collection, expr) for name, expr in SET_FILTERS.items()})
    counts["batch2"] = sum(counts[name] for name in SET_FILTERS)
    counts["other"] = counts["total"] - counts["L"] - counts["batch2"]
    counts["black_tail"] = count(client, collection, BLACK_TAIL_FILTER)
    return counts


def check_l_upload(collection: str, endpoint_sha256: str) -> None:
    """The collection must be the one milvus_upload_tara.py verified with the L clips."""
    state = read_json(L_STATE)
    binding = state.get("binding") or {}
    wanted = {"collection": collection, "endpoint_sha256": endpoint_sha256,
              "semantic_fingerprint": L_FINGERPRINT, "expected_rows": L_ROWS}
    if (state.get("complete") is not True or state.get("verified_rows") != L_ROWS
            or any(binding.get(key) != value for key, value in wanted.items())):
        raise ValueError(f"{L_STATE}: not the verified L21–L30 upload of {collection} on this endpoint")


def verify_samples(client: Any, collection: str, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Field-exact read-back of one row per work unit, plus a self-search per set and scale."""
    by_id: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(samples), 32):
        chunk = samples[offset:offset + 32]
        rows = client.get(collection_name=collection, ids=[s["clip_id"] for s in chunk],
                          output_fields=[*OUTPUT_FIELDS, "embedding"])
        by_id.update({row["clip_id"]: row for row in rows})
    for sample in samples:
        row = by_id.get(sample["clip_id"])
        if row is None:
            raise ValueError(f"Remote sample is missing: {sample['clip_id']}")
        for field in OUTPUT_FIELDS:
            want, got = sample[field], row[field]
            same = abs(float(got) - float(want)) <= 1e-4 if isinstance(want, float) else got == want
            if not same:
                raise ValueError(f"{sample['clip_id']}: remote {field}={got!r}, expected {want!r}")
        remote = np.asarray(row["embedding"], dtype=np.float32)
        if float(np.dot(remote, sample["embedding"])) < 0.99999:
            raise ValueError(f"{sample['clip_id']}: remote vector differs")

    # N's fixed cameras and S01's looping standby card tie near 1.0, so the sample
    # has to be among the top hits rather than first.
    searched = []
    firsts: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        firsts.setdefault((set_of(sample["category"]), sample["scale"]), sample)
    for (name, scale), sample in sorted(firsts.items()):
        hits = client.search(collection_name=collection, data=[sample["embedding"]], limit=10,
                             filter=f'scale == "{scale}"', output_fields=["clip_id"],
                             search_params={"metric_type": "COSINE"})[0]
        scores = {(hit.get("entity") or {}).get("clip_id") or hit.get("id"): float(hit["distance"])
                  for hit in hits}
        if scores.get(sample["clip_id"], 0.0) < 0.9999:
            raise ValueError(f"Self-search for {sample['clip_id']} missed it: {scores}")
        searched.append({"set": name, "scale": scale, "clip_id": sample["clip_id"],
                         "score": scores[sample["clip_id"]]})
    return searched


def wait_for_counts(client: Any, collection: str, expected: dict[str, int], timeout: float) -> dict[str, int]:
    deadline = time.monotonic() + timeout
    while True:
        counts = remote_counts(client, collection)
        if counts == expected:
            return counts
        if counts["total"] > expected["total"] or counts["other"] or time.monotonic() > deadline:
            raise ValueError(f"Remote counts {counts} != expected {expected}")
        print(f"Waiting for Milvus counts: {counts}", flush=True)
        time.sleep(5)


def wait_for_index(client: Any, collection: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        index = client.describe_index(collection_name=collection, index_name="embedding")
        if int(index.get("pending_index_rows", 0)) == 0 or time.monotonic() > deadline:
            return {key: index.get(key) for key in
                    ("index_type", "metric_type", "state", "total_rows", "indexed_rows", "pending_index_rows")}
        print(f"Waiting for the index: {index.get('indexed_rows')}/{index.get('total_rows')} indexed", flush=True)
        time.sleep(15)


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    shards: list[Shard] = []
    for artifact in ARTIFACTS:
        found = audit_commits(artifact, args.artifact_root / artifact.directory,
                              verify_sha256=not args.skip_parquet_sha256)
        shards.extend(found)
        checked = "size" if args.skip_parquet_sha256 else "size/SHA-256"
        print(f"[{artifact.name}] {len(found)} commits, {artifact.rows:,} rows, contract "
              f"{EMBEDDING_CONTRACT[:12]}… = L — commit/{checked} audit PASS", flush=True)

    uri = token = ""
    client = None
    state: dict[str, Any] = {"completed_shards": {}}
    before: dict[str, int] = {}
    endpoint_sha256 = "dry-run"
    if not args.dry_run:
        from pymilvus import MilvusClient

        uri, token = credentials(args.env_file)
        endpoint_sha256 = hashlib.sha256(uri.encode()).hexdigest()
        check_l_upload(args.collection, endpoint_sha256)
        client = MilvusClient(uri=uri, token=token)
        ensure_collection(client, args.collection, may_create=False)
        before = remote_counts(client, args.collection)
        print(f"[{args.collection}] before: {json.dumps(before)}", flush=True)
        if before["L"] != L_ROWS or before["other"] or before["black_tail"]:
            raise ValueError(f"{args.collection} must hold exactly {L_ROWS:,} L21–L30 rows, "
                             "no unknown rows and no M10_V029 black tail; refusing")
        if args.check_remote:
            print("REMOTE CHECK PASSED (read-only); nothing was written.", flush=True)
            return 0
        binding = {
            "collection": args.collection,
            "endpoint_sha256": endpoint_sha256,
            "embedding_contract_sha256": EMBEDDING_CONTRACT,
            "semantic_fingerprints": {a.name: a.semantic_fingerprint for a in ARTIFACTS},
            "black_tail": {"video_id": BLACK_TAIL_VIDEO, "start_time_gte": BLACK_TAIL_START},
        }
        if args.state_file.is_file():
            state = read_json(args.state_file)
            if state.get("binding") != binding:
                raise ValueError(f"{args.state_file} belongs to a different artifact or collection")
        else:
            state = {"binding": binding, "completed_shards": {}, "complete": False}
            if before["batch2"]:
                print(f"Note: {before['batch2']:,} batch-2 rows exist without local state; "
                      "re-upserting them (same clip_ids, idempotent).", flush=True)
        write_state(args.state_file, state)

    completed = state["completed_shards"]
    seen_clips: set[str] = set()
    samples: list[dict[str, Any]] = []
    black_tail: list[bytes] = []
    kept_by_set = {name: 0 for name in EXPECTED_BY_SET}
    validated = uploaded = 0
    inflight: dict[Future, tuple[Shard, int]] = {}

    def finish(done: set[Future]) -> None:
        nonlocal uploaded
        for future in done:
            shard, kept = inflight.pop(future)
            if future.result() != kept:
                raise RuntimeError(f"{shard.key}: upserted row count differs")
            completed[shard.key] = {"rows": kept, "parquet_sha256": shard.parquet_sha256}
            write_state(args.state_file, state)
            uploaded += kept
            rate = uploaded / max(time.monotonic() - started, 1e-3)
            print(f"  {shard.key}: {kept:,} rows upserted (this run {uploaded:,}; {rate:,.0f} rows/s)",
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="tara-b2") as pool:
        for shard in shards:
            done = completed.get(shard.key)
            if done and done["parquet_sha256"] != shard.parquet_sha256:
                raise ValueError(f"{shard.key}: state records a different parquet SHA-256")
            build = not args.dry_run and not done
            checked = check_shard(shard, seen_clips, build=build)
            validated += shard.rows
            black_tail.extend(checked.black_tail)
            if checked.sample is not None and shard.shard_id == 0:
                samples.append(checked.sample)
            if checked.kept:
                kept_by_set[set_of(checked.sample["category"])] += checked.kept
            if done and done["rows"] != checked.kept:
                raise ValueError(f"{shard.key}: state records {done['rows']} rows, shard keeps {checked.kept}")
            if build:
                while len(inflight) >= 2 * args.workers:
                    finish(wait(inflight, return_when=FIRST_COMPLETED)[0])
                future = pool.submit(upsert, uri, token, args.collection, checked.entities,
                                     args.batch_size, args.timeout)
                inflight[future] = (shard, checked.kept)
        while inflight:
            finish(wait(inflight, return_when=FIRST_COMPLETED)[0])

    if len(black_tail) != BLACK_TAIL_ROWS or len(set(black_tail)) != 1:
        raise ValueError(f"M10_V029 black tail: {len(black_tail)} rows, {len(set(black_tail))} distinct "
                         f"vectors; expected {BLACK_TAIL_ROWS} identical")
    if kept_by_set != EXPECTED_BY_SET:
        raise ValueError(f"Indexed rows per set {kept_by_set} != expected {EXPECTED_BY_SET}")
    print(f"Every row checked: {validated:,} clips, {len(seen_clips):,} unique clip_ids; "
          f"indexing {kept_by_set} and dropping {len(black_tail):,} identical M10_V029 black-tail clips",
          flush=True)
    if args.dry_run:
        print(f"DRY RUN PASSED in {time.monotonic() - started:.0f}s; Milvus was not contacted.", flush=True)
        return 0

    assert client is not None
    client.flush(collection_name=args.collection, timeout=args.timeout)
    expected = {"total": FINAL_ROWS, "L": L_ROWS, **EXPECTED_BY_SET, "batch2": BATCH2_ROWS,
                "other": 0, "black_tail": 0}
    after = wait_for_counts(client, args.collection, expected, args.count_wait_timeout)
    ensure_collection(client, args.collection, may_create=False)
    searched = verify_samples(client, args.collection, samples)
    index = wait_for_index(client, args.collection, args.index_wait_timeout)
    state.update(complete=True, remote_counts=after, verified_at=datetime.now(timezone.utc).isoformat())
    write_state(args.state_file, state)
    args.verification_file.write_text(json.dumps({
        "schema_version": 1,
        "status": "PASS",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "collection": args.collection,
        "endpoint_sha256": endpoint_sha256,
        "embedding_contract_sha256": EMBEDDING_CONTRACT,
        "artifacts": {a.name: {"directory": a.directory, "semantic_fingerprint": a.semantic_fingerprint,
                               "committed_rows": a.rows} for a in ARTIFACTS},
        "dropped_black_tail": {"video_id": BLACK_TAIL_VIDEO, "start_time_gte": BLACK_TAIL_START,
                               "rows": BLACK_TAIL_ROWS},
        "before": before,
        "after": after,
        "remote_samples_checked": len(samples),
        "self_search": searched,
        "index": index,
    }, indent=2) + "\n")
    print(f"UPLOAD VERIFIED: {args.collection} holds {after['total']:,} clips (L21–L30 {after['L']:,} + "
          f"M {after['M']:,} + N {after['N']:,} + S01 {after['S01']:,}); index {index['state']}, "
          f"{index['pending_index_rows']} pending ({time.monotonic() - started:.0f}s).", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", type=Path, default=ROOT,
                        help="Directory holding the two batch-2 artifact folders.")
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--env-file", type=Path, default=REPO / "backend" / ".env")
    parser.add_argument("--state-file", type=Path, default=REPO / ".milvus_upload_state_tara_batch2.json")
    parser.add_argument("--verification-file", type=Path,
                        default=ROOT / "milvus_upload_tara_batch2_verification.json")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--count-wait-timeout", type=float, default=900.0)
    parser.add_argument("--index-wait-timeout", type=float, default=1800.0)
    parser.add_argument("--skip-parquet-sha256", action="store_true",
                        help="Skip hashing the 4.4 GiB of Parquet (sizes, metadata and rows are still checked).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Local audit of every row; Milvus is not contacted.")
    mode.add_argument("--check-remote", action="store_true", help="Local commit audit + read-only remote checks.")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 1024:
        parser.error("--batch-size must be 1..1024")
    if not 1 <= args.workers <= 12:
        parser.error("--workers must be 1..12")
    return args


def main(argv: list[str] | None = None) -> int:
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
