#!/usr/bin/env python3
"""Append the batch-2 keyframes (M, N, S01) to the InfoShot++ keyframe map in Elastic.

Retrieval profile 2 reads a single keyframe map, ``aic26_keyframe_map_infoshotpp_v1``
(``IDX_KEYFRAME_MAP_2``), which already holds the 1,339,055 L21–L30 keyframes
written by ``elastic_upload.py --only keyframe_map_infoshotpp``. This script adds
the 874,800 batch-2 keyframes to that same index, field-for-field in the shape
the L documents have, so the backend reads both with no configuration change.

Source of truth is the three pinned ``frame_registry.parquet`` files (see
``batch2_keyframes.py``), not the ``map-keyframes.zip`` CSVs: the registry carries
``frame_id``, category and R2 key, which a map CSV does not. Every registry row
is indexed, including the 1,472 frames below quality 0.05 that were never
embedded — the map mirrors what is on R2 (timeline, neighbours, pts lookup); the
vector index alone decides what is searchable.

Safety properties:

* the index must already exist with the keyframe-map mapping — this script never
  creates, deletes or re-maps an index;
* batch-2 ids (``M01/…``, ``N001/…``, ``S01/…``) cannot collide with an L id, and
  the L document count is measured before and after and must not move;
* ``_id = submit_keyframe_id``, so replaying a bulk is an overwrite of the same
  document, never a duplicate;
* resume is per video: a video is checkpointed only after the bulk carrying its
  last document succeeded, and the state is bound to the registry SHA-256s, the
  index and the endpoint.

Run ``--dry-run`` first (local only), then ``--check-remote`` (read-only), then
the same command without either flag to upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from batch2_keyframes import DEFAULT_REGISTRY_ROOT, PROFILES, TOTAL_KEYFRAMES, KeyframeRow, load_registries
from elastic_upload import (
    DEFAULT_INFOSHOTPP_MAP_INDEX,
    EXPECTED_INFOSHOTPP_CATEGORIES,
    EXPECTED_INFOSHOTPP_KEYFRAMES,
    ElasticClient,
    build_bulk_payload,
    keyframe_mapping,
    read_secret,
)

ID_FIELD = "submit_keyframe_id"
STATE_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def endpoint_identity(endpoint: str) -> str:
    """Hash of scheme://host/path — binds the state to a cluster without storing its URL."""
    parsed = urlsplit(endpoint if "://" in endpoint else f"https://{endpoint}")
    public = f"{parsed.scheme.lower()}://{parsed.hostname or ''}{parsed.path.rstrip('/')}"
    return hashlib.sha256(public.encode("utf-8")).hexdigest()


def build_document(row: KeyframeRow) -> dict[str, Any]:
    """The document `elastic_upload.iter_infoshotpp_keyframes` builds for an L keyframe."""
    return {
        "keyframe_id": row.keyframe_id,
        "frame_id": row.frame_id,
        "submit_keyframe_id": row.submit_keyframe_id,
        "video_id": row.video_id,
        "category_hint": row.category,
        "submit_category": row.category,
        "keyframe_n": row.n,
        "keyframe_name": row.keyframe_name,
        "pts_time": row.pts_time,
        "fps": row.fps,
        "frame_idx": row.frame_idx,
    }


def plan_bulks(rows: Iterable[KeyframeRow], batch_size: int) -> Iterator[tuple[list[dict[str, Any]], list[str]]]:
    """Yield ``(documents, videos finished by this bulk)``; rows must be grouped by video.

    A video is reported only with the bulk that carries its last document, so a
    checkpoint written after that bulk succeeds never claims a video whose tail
    has not been sent yet.
    """
    batch: list[dict[str, Any]] = []
    finished: list[str] = []
    seen: set[str] = set()
    current: str | None = None
    for row in rows:
        if row.video_id != current:
            if row.video_id in seen:
                raise ValueError(f"Rows of {row.video_id} are not contiguous")
            if current is not None:
                finished.append(current)
            seen.add(row.video_id)
            current = row.video_id
        if len(batch) >= batch_size:
            yield batch, finished
            batch, finished = [], []
        batch.append(build_document(row))
    if current is not None:
        finished.append(current)
    if batch:
        yield batch, finished


# ---------------------------------------------------------------------------
# remote checks (read-only)
# ---------------------------------------------------------------------------


def terms_query(categories: Iterable[str]) -> dict[str, Any]:
    return {"terms": {"submit_category": sorted(categories)}}


def count_where(client: ElasticClient, index: str, query: dict[str, Any]) -> int:
    result = client.json_request("POST", f"{index}/_count", payload={"query": query}, timeout=120)
    return int(result["count"])


def category_counts(client: ElasticClient, index: str, categories: Sequence[str]) -> dict[str, int]:
    body = {
        "size": 0,
        "query": terms_query(categories),
        "aggs": {"by_category": {"terms": {"field": "submit_category", "size": len(categories) + 10}}},
    }
    result = client.json_request("POST", f"{index}/_search", payload=body, timeout=120)
    return {bucket["key"]: int(bucket["doc_count"]) for bucket in result["aggregations"]["by_category"]["buckets"]}


def verify_index_mapping(client: ElasticClient, index: str) -> None:
    if not client.index_exists(index):
        raise SystemExit(
            f"Index {index} does not exist. It is the L21–L30 map this script extends; create it with "
            "`python elastic_upload.py --only keyframe_map_infoshotpp` first."
        )
    result = client.json_request("GET", f"{index}/_mapping", timeout=60)
    actual = (next(iter(result.values())).get("mappings") or {}).get("properties") or {}
    for name, spec in keyframe_mapping()["mappings"]["properties"].items():
        if (actual.get(name) or {}).get("type") != spec["type"]:
            raise SystemExit(f"{index}: field {name} is {actual.get(name)!r}, expected type {spec['type']!r}")


def remote_counts(client: ElasticClient, index: str, categories: Sequence[str]) -> dict[str, int]:
    total = client.count(index)
    base = count_where(client, index, terms_query(EXPECTED_INFOSHOTPP_CATEGORIES))
    batch2 = count_where(client, index, terms_query(categories))
    return {"total": total, "base_l21_l30": base, "batch2": batch2, "other": total - base - batch2}


def preflight_remote(client: ElasticClient, index: str, categories: Sequence[str]) -> dict[str, int]:
    verify_index_mapping(client, index)
    client.refresh(index)
    counts = remote_counts(client, index, categories)
    print(f"[{index}] remote before upload: {json.dumps(counts)}", flush=True)
    if counts["base_l21_l30"] != EXPECTED_INFOSHOTPP_KEYFRAMES:
        raise SystemExit(
            f"{index} holds {counts['base_l21_l30']:,} L21–L30 keyframes, expected "
            f"{EXPECTED_INFOSHOTPP_KEYFRAMES:,}. Refusing to extend an index that is not the verified L map."
        )
    if counts["other"]:
        raise SystemExit(f"{index} holds {counts['other']:,} documents outside L21–L30 and batch 2; refusing.")
    return counts


def verify_upload(
    client: ElasticClient,
    index: str,
    rows_by_profile: dict[str, list[KeyframeRow]],
    before: dict[str, int],
) -> dict[str, Any]:
    client.refresh(index)
    rows = [row for profile_rows in rows_by_profile.values() for row in profile_rows]
    categories = sorted({row.category for row in rows})
    expected_by_category: dict[str, int] = {}
    for row in rows:
        expected_by_category[row.category] = expected_by_category.get(row.category, 0) + 1
    counts = remote_counts(client, index, categories)
    by_category = category_counts(client, index, categories)
    problems = [
        f"{category}: remote {by_category.get(category, 0):,}, registry {expected:,}"
        for category, expected in expected_by_category.items()
        if by_category.get(category, 0) != expected
    ]
    if counts["batch2"] != TOTAL_KEYFRAMES:
        problems.append(f"batch-2 total {counts['batch2']:,}, expected {TOTAL_KEYFRAMES:,}")
    if counts["base_l21_l30"] != before["base_l21_l30"]:
        problems.append(f"L21–L30 count moved {before['base_l21_l30']:,} -> {counts['base_l21_l30']:,}")
    if counts["other"]:
        problems.append(f"{counts['other']:,} documents outside L21–L30 and batch 2")

    samples: list[KeyframeRow] = []
    for profile_rows in rows_by_profile.values():
        samples.extend(profile_rows[position] for position in (0, len(profile_rows) // 2, -1))
    result = client.json_request(
        "POST", f"{index}/_mget", payload={"ids": [row.submit_keyframe_id for row in samples]}, timeout=60
    )
    for row, doc in zip(samples, result["docs"]):
        if not doc.get("found"):
            problems.append(f"sample {row.submit_keyframe_id} not found")
        elif doc["_source"] != build_document(row):
            problems.append(f"sample {row.submit_keyframe_id} differs: {doc['_source']}")
    if problems:
        raise SystemExit("Verification FAILED:\n  " + "\n  ".join(problems))
    return {
        "counts": counts,
        "batch2_by_profile": {
            name: sum(by_category[category] for category in {row.category for row in profile_rows})
            for name, profile_rows in rows_by_profile.items()
        },
        "samples": [row.submit_keyframe_id for row in samples],
    }


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


def state_binding(index: str, endpoint_hash: str) -> dict[str, Any]:
    return {
        "index": index,
        "endpoint_identity_sha256": endpoint_hash,
        "registry_sha256": {name: profile.registry_sha256 for name, profile in PROFILES.items()},
        "documents": TOTAL_KEYFRAMES,
    }


def load_or_create_state(path: Path, binding: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("schema_version") != STATE_SCHEMA_VERSION or state.get("binding") != binding:
            raise SystemExit(
                f"{path} was written for another index/endpoint/registry snapshot. "
                "Use a different --state-file (re-indexing is idempotent)."
            )
        return state
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "binding": binding,
        "completed_videos": [],
        "complete": False,
    }
    atomic_write_json(path, state)
    return state


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    rows_by_profile = load_registries(root=args.registry_root)
    rows = [row for profile_rows in rows_by_profile.values() for row in profile_rows]
    categories = sorted({row.category for row in rows})
    if set(categories) & EXPECTED_INFOSHOTPP_CATEGORIES:
        raise SystemExit("A batch-2 category collides with L21–L30; refusing.")

    if args.dry_run:
        mapped = set(keyframe_mapping()["mappings"]["properties"])
        documents = bulks = 0
        for docs, _ in plan_bulks(rows, args.batch_size):
            for doc in docs:
                if set(doc) != mapped:
                    raise SystemExit(f"Document fields differ from the index mapping: {sorted(set(doc) ^ mapped)}")
            documents += len(docs)
            bulks += 1
        print(json.dumps(build_document(rows_by_profile["N"][0]), ensure_ascii=False))
        print(
            f"DRY RUN PASSED: {documents:,} documents in {bulks} bulks for {len(categories)} categories "
            f"-> {args.index}; Elastic was not contacted ({time.monotonic() - started:.1f}s).",
            flush=True,
        )
        return 0

    endpoint = read_secret(args.endpoint_file)
    client = ElasticClient(endpoint, read_secret(args.api_key_file))
    info = client.json_request("GET", "/", expected=(200,), timeout=60)
    print(f"Connected to Elasticsearch {(info.get('version') or {}).get('number')}", flush=True)
    before = preflight_remote(client, args.index, categories)
    if args.check_remote:
        print("REMOTE CHECK PASSED (read-only); nothing was written.", flush=True)
        return 0

    state = load_or_create_state(args.state_file, state_binding(args.index, endpoint_identity(endpoint)))
    completed = set(state["completed_videos"])
    if before["batch2"] and not completed:
        print(
            f"Note: {before['batch2']:,} batch-2 documents exist without local state; "
            "re-indexing all of them (same _id, idempotent).",
            flush=True,
        )
    todo = [row for row in rows if row.video_id not in completed]
    print(f"Uploading {len(todo):,} documents ({len(completed)} videos already checkpointed)", flush=True)
    sent = 0
    for number, (docs, finished) in enumerate(plan_bulks(todo, args.batch_size), start=1):
        client.bulk_ndjson(build_bulk_payload(args.index, docs, id_field=ID_FIELD))
        sent += len(docs)
        completed.update(finished)
        state["completed_videos"] = sorted(completed)
        state["updated_at"] = utc_now()
        atomic_write_json(args.state_file, state)
        if number % 25 == 0:
            rate = sent / max(time.monotonic() - started, 0.001)
            print(f"  sent {sent:,}/{len(todo):,} ({rate:,.0f} docs/s)", flush=True)

    verification = verify_upload(client, args.index, rows_by_profile, before)
    state["complete"] = True
    state["verified_at"] = utc_now()
    atomic_write_json(args.state_file, state)
    atomic_write_json(
        args.verification_file,
        {
            "schema_version": 1,
            "status": "PASS",
            "verified_at": utc_now(),
            "index": args.index,
            "endpoint_identity_sha256": endpoint_identity(endpoint),
            "registry_sha256": state["binding"]["registry_sha256"],
            "before": before,
            **verification,
        },
    )
    print(
        f"UPLOAD VERIFIED: {args.index} now holds {verification['counts']['total']:,} keyframes "
        f"(L21–L30 {verification['counts']['base_l21_l30']:,} + batch 2 {verification['counts']['batch2']:,}) "
        f"in {time.monotonic() - started:.1f}s.",
        flush=True,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    parser.add_argument("--index", default=DEFAULT_INFOSHOTPP_MAP_INDEX)
    parser.add_argument("--endpoint-file", type=Path, default=Path("API_KEY/elastic_endpoint.txt"))
    parser.add_argument("--api-key-file", type=Path, default=Path("API_KEY/elastic_apikey.txt"))
    parser.add_argument("--state-file", type=Path, default=Path(".elastic_upload_batch2_keyframe_map_state.json"))
    parser.add_argument(
        "--verification-file", type=Path, default=Path("elastic_upload_batch2_keyframe_map_verification.json")
    )
    parser.add_argument("--batch-size", type=int, default=2000)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and build every document; no network.")
    mode.add_argument("--check-remote", action="store_true", help="Read-only checks against the index; no writes.")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 10_000:
        parser.error("--batch-size must be between 1 and 10000")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print("Interrupted. Finished videos are checkpointed; rerun the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
