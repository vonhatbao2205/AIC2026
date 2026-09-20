#!/usr/bin/env python3
"""Audit and idempotently index TARA clip vectors into InfoShot++ Milvus Cloud.

Run from the repository root with backend/.venv/bin/python. The collection is
versioned and never dropped. A local state file records completed shards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from tara_artifact import CATEGORIES, EXPECTED_DIM, EXPECTED_ROWS, PREFIX, ROOT, audit

COLLECTION = "aic26_tara_clips_infoshotpp_v1"


def read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip().strip('"\'')
    return result


def credentials(path: Path) -> tuple[str, str]:
    saved = read_env(path)
    uri = os.environ.get("MILVUS_ENDPOINT_2") or saved.get("MILVUS_ENDPOINT_2", "")
    token = os.environ.get("MILVUS_TOKEN_2") or saved.get("MILVUS_TOKEN_2", "")
    if not uri or not token:
        raise RuntimeError("MILVUS_ENDPOINT_2 and MILVUS_TOKEN_2 are required")
    return uri, token


def schema_spec(data_type):
    return {
        "clip_id": (data_type.VARCHAR, {"max_length": 96, "is_primary": True}),
        "video_id": (data_type.VARCHAR, {"max_length": 32}),
        "category": (data_type.VARCHAR, {"max_length": 8}),
        "scale": (data_type.VARCHAR, {"max_length": 16}),
        "scale_index": (data_type.INT64, {}),
        "start_time": (data_type.DOUBLE, {}),
        "end_time": (data_type.DOUBLE, {}),
        "fps": (data_type.FLOAT, {}),
        "center_frame_idx": (data_type.INT64, {}),
        "embedding": (data_type.FLOAT_VECTOR, {"dim": EXPECTED_DIM}),
    }


def ensure_collection(client, name: str, *, may_create: bool) -> bool:
    from pymilvus import DataType, MilvusClient

    created = False
    if not client.has_collection(name):
        if not may_create:
            raise RuntimeError(f"Collection {name} is absent; refusing to resume without state")
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        for field, (dtype, params) in schema_spec(DataType).items():
            schema.add_field(field_name=field, datatype=dtype, **params)
        indexes = client.prepare_index_params()
        indexes.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(collection_name=name, schema=schema, index_params=indexes)
        created = True
    description = client.describe_collection(collection_name=name)
    actual = {field["name"]: field for field in description["fields"]}
    expected = schema_spec(DataType)
    if set(actual) != set(expected) or description.get("enable_dynamic_field") is True:
        raise RuntimeError("Collection schema does not match TARA v1 contract")
    for field, (dtype, params) in expected.items():
        current = actual[field]
        if int(current.get("type", current.get("data_type"))) != int(dtype):
            raise RuntimeError(f"Collection datatype mismatch: {field}")
        for key, wanted in params.items():
            got = current.get("is_primary") if key == "is_primary" else (current.get("params") or {}).get(key)
            if got is not None and key != "is_primary":
                got = int(got)
            if got != wanted:
                raise RuntimeError(f"Collection field parameter mismatch: {field}.{key}")
    if "embedding" not in client.list_indexes(collection_name=name):
        raise RuntimeError("Collection lacks an embedding index")
    index = client.describe_index(collection_name=name, index_name="embedding")
    metric = index.get("metric_type") or (index.get("params") or {}).get("metric_type")
    if metric != "COSINE":
        raise RuntimeError(f"Collection metric is {metric!r}, expected COSINE")
    return created


def row_count(client, name: str) -> int:
    stats = client.get_collection_stats(collection_name=name)
    return int(stats.get("row_count", stats.get("num_entities", -1)))


def write_state(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(state, indent=2))
    os.replace(temp, path)


def records(path: Path):
    table = pq.read_table(path)
    vectors = table["embedding"].combine_chunks().values.to_numpy().reshape(-1, EXPECTED_DIM)
    frames = table["frame_indices"].combine_chunks().values.to_numpy().reshape(-1, 8)
    columns = {
        field: table[field].to_pylist()
        for field in ("clip_id", "video_id", "category", "scale", "scale_index", "start_time", "end_time", "fps")
    }
    for i in range(table.num_rows):
        yield {
            **{field: value[i] for field, value in columns.items()},
            "center_frame_idx": int(frames[i, 4]),
            "embedding": vectors[i].tolist(),
        }


def upload_shard(commit_path: Path, root: Path, collection: str, uri: str, token: str,
                 batch_size: int, local: threading.local) -> tuple[str, str, int]:
    from pymilvus import MilvusClient

    if not hasattr(local, "client"):
        local.client = MilvusClient(uri=uri, token=token)
    commit = json.loads(commit_path.read_text())
    category = commit["category"]
    key = f"{category}/part-{commit['shard_id']:05d}"
    path = root / commit["parquet_path"].removeprefix(PREFIX + "/")
    batch = []
    written = 0
    for entity in records(path):
        batch.append(entity)
        if len(batch) >= batch_size:
            for attempt in range(4):
                try:
                    local.client.upsert(collection_name=collection, data=batch, timeout=180)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
                    local.client = MilvusClient(uri=uri, token=token)
            written += len(batch)
            batch.clear()
    if batch:
        for attempt in range(4):
            try:
                local.client.upsert(collection_name=collection, data=batch, timeout=180)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
                local.client = MilvusClient(uri=uri, token=token)
        written += len(batch)
    if written != commit["rows"]:
        raise RuntimeError(f"{key}: upload row count mismatch")
    return key, commit["parquet_sha256"], written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--env-file", type=Path, default=Path("backend/.env"))
    parser.add_argument("--state-file", type=Path, default=ROOT / "milvus_upload_state.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 1024:
        raise ValueError("batch-size must be 1..1024")
    if not 1 <= args.workers <= 12:
        raise ValueError("workers must be 1..12")
    report = audit(args.root)
    print("Local TARA artifact PASS:", json.dumps(report, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    from pymilvus import MilvusClient

    uri, token = credentials(args.env_file)
    endpoint_hash = hashlib.sha256(uri.encode()).hexdigest()
    binding = {
        "collection": args.collection,
        "endpoint_sha256": endpoint_hash,
        "manifest_sha256": report["manifest_sha256"],
        "semantic_fingerprint": report["semantic_fingerprint"],
        "expected_rows": EXPECTED_ROWS,
    }
    state_exists = args.state_file.is_file()
    if state_exists:
        state = json.loads(args.state_file.read_text())
        if state.get("binding") != binding:
            raise RuntimeError("Upload state belongs to a different artifact or collection")
    else:
        state = {"binding": binding, "completed_shards": {}, "complete": False}
    client = MilvusClient(uri=uri, token=token)
    existed = client.has_collection(args.collection)
    if existed and not state_exists and row_count(client, args.collection) != 0:
        raise RuntimeError("Existing nonempty collection has no matching upload state")
    ensure_collection(client, args.collection, may_create=not existed)
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    write_state(args.state_file, state)

    local = threading.local()
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="tara-upload") as pool:
        for category in CATEGORIES:
            pending = []
            for commit_path in sorted((args.root / "commits" / category).glob("part-*.json")):
                commit = json.loads(commit_path.read_text())
                key = f"{category}/part-{commit['shard_id']:05d}"
                if key in state["completed_shards"]:
                    if state["completed_shards"][key] != commit["parquet_sha256"]:
                        raise RuntimeError(f"State hash mismatch for {key}")
                    continue
                pending.append(commit_path)
            futures = [
                pool.submit(upload_shard, path, args.root, args.collection,
                            uri, token, args.batch_size, local)
                for path in pending
            ]
            for future in as_completed(futures):
                key, sha, written = future.result()
                state["completed_shards"][key] = sha
                write_state(args.state_file, state)
                print(f"{key}: {written} rows upserted", flush=True)
            client.flush(collection_name=args.collection, timeout=180)
            print(f"{category}: flushed", flush=True)

    client.flush(collection_name=args.collection, timeout=180)
    for attempt in range(24):
        count = row_count(client, args.collection)
        if count == EXPECTED_ROWS:
            break
        if count > EXPECTED_ROWS:
            raise RuntimeError(f"Milvus contains too many rows: {count}")
        time.sleep(5)
    else:
        raise RuntimeError(f"Milvus row count remains {count}, expected {EXPECTED_ROWS}")
    state["complete"] = True
    state["verified_rows"] = count
    write_state(args.state_file, state)
    print(f"Milvus TARA collection PASS: {args.collection}, {count} rows", flush=True)


if __name__ == "__main__":
    main()
