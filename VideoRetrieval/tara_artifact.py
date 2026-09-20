#!/usr/bin/env python3
"""Download and audit the immutable TARA clip artifact from the HF bucket.

Run with backend/.venv/bin/python. The local artifact is ignored by git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

BUCKET = "Baonenha1/aic26-media"
PREFIX = "derived/tara-tarsier2-7b-3584-clip-v1"
ROOT = Path(__file__).resolve().parent / "tara_embeddings"
EXPECTED_ROWS = 168_536
EXPECTED_VIDEOS = 873
EXPECTED_DIM = 3584
CATEGORIES = tuple(f"L{i}" for i in range(21, 31))
REQUIRED_COLUMNS = {
    "clip_id", "video_id", "category", "scale", "scale_index",
    "start_time", "end_time", "fps", "duration_sec", "video_relpath",
    "frame_indices", "embedding",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def download(root: Path, token: str, *, metadata_only: bool = False) -> None:
    from huggingface_hub import download_bucket_files, list_bucket_tree

    remote = [
        item for item in list_bucket_tree(BUCKET, prefix=PREFIX, recursive=True, token=token)
        if getattr(item, "type", "file") == "file"
    ]
    if not remote:
        raise RuntimeError("TARA bucket prefix is empty")
    for offset in range(0, len(remote), 8):
        pending = []
        for item in remote[offset:offset + 8]:
            if metadata_only and item.path.endswith(".parquet"):
                continue
            target = root / item.path.removeprefix(PREFIX + "/")
            if target.is_file() and target.stat().st_size == item.size:
                continue
            pending.append((item, str(target)))
        if pending:
            download_bucket_files(BUCKET, files=pending, token=token)
        print(f"downloaded/verified remote files {min(offset + 8, len(remote))}/{len(remote)}", flush=True)


def audit(root: Path) -> dict:
    success_path = root / "_SUCCESS.json"
    manifest_path = root / "embedding_dataset_manifest.json"
    if not success_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Missing _SUCCESS.json or embedding_dataset_manifest.json")
    success = json.loads(success_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if success != manifest or not success.get("complete"):
        raise RuntimeError("Completion manifests disagree or are incomplete")
    if (success.get("total_committed_rows"), success.get("expected_total_rows"),
            success.get("expected_total_videos")) != (EXPECTED_ROWS, EXPECTED_ROWS, EXPECTED_VIDEOS):
        raise RuntimeError("Completion manifest has unexpected corpus size")
    run = json.loads((root / "run_config.json").read_text())
    fingerprint = success["semantic_fingerprint"]
    if run["semantic_fingerprint"] != fingerprint or run["semantic_config"] != success["semantic_config"]:
        raise RuntimeError("Run config and completion manifest disagree")

    seen_clips: set[str] = set()
    seen_videos: set[str] = set()
    totals: Counter[str] = Counter()
    shard_count = 0
    for category in CATEGORIES:
        category_info = success["per_category"][category]
        commits = sorted((root / "commits" / category).glob("part-*.json"))
        if len(commits) != category_info["expected_shards"]:
            raise RuntimeError(f"{category}: missing commit(s)")
        expected_start = 0
        for shard_id, commit_path in enumerate(commits):
            commit = json.loads(commit_path.read_text())
            shard_key = f"{category}/part-{shard_id:05d}"
            if (commit.get("shard_id") != shard_id or commit.get("category") != category
                    or commit.get("semantic_fingerprint") != fingerprint
                    or commit.get("row_start") != expected_start
                    or commit.get("row_stop") != expected_start + commit.get("rows", -1)
                    or commit.get("embedding_dim") != EXPECTED_DIM
                    or commit.get("embedding_dtype") != "float32"
                    or commit.get("l2_normalized") is not True):
                raise RuntimeError(f"{shard_key}: invalid commit contract")
            relpath = commit["parquet_path"].removeprefix(PREFIX + "/")
            if not relpath.startswith(f"embeddings/{category}/"):
                raise RuntimeError(f"{shard_key}: invalid parquet path")
            path = root / relpath
            if not path.is_file() or path.stat().st_size != commit["parquet_bytes"]:
                raise RuntimeError(f"{shard_key}: missing or wrong-size parquet")
            if sha256_file(path) != commit["parquet_sha256"]:
                raise RuntimeError(f"{shard_key}: SHA-256 mismatch")
            table = pq.read_table(path)
            if set(table.column_names) != REQUIRED_COLUMNS or table.num_rows != commit["rows"]:
                raise RuntimeError(f"{shard_key}: schema or row count mismatch")
            if table.schema.metadata.get(b"semantic_fingerprint", b"").decode() != fingerprint:
                raise RuntimeError(f"{shard_key}: parquet fingerprint mismatch")
            embedding = table["embedding"].combine_chunks()
            frames = table["frame_indices"].combine_chunks()
            if embedding.type.list_size != EXPECTED_DIM or frames.type.list_size != 8:
                raise RuntimeError(f"{shard_key}: vector or frame dimension mismatch")
            vectors = embedding.values.to_numpy().reshape(-1, EXPECTED_DIM)
            if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                raise RuntimeError(f"{shard_key}: invalid FP32 vectors")
            norms = np.linalg.norm(vectors, axis=1)
            if np.max(np.abs(norms - 1)) > 2e-5:
                raise RuntimeError(f"{shard_key}: vectors are not unit norm")
            clips = table["clip_id"].to_pylist()
            if clips[0] != commit["first_clip_id"] or clips[-1] != commit["last_clip_id"]:
                raise RuntimeError(f"{shard_key}: clip identity mismatch")
            if len(set(clips)) != len(clips) or seen_clips.intersection(clips):
                raise RuntimeError(f"{shard_key}: duplicate clip_id")
            seen_clips.update(clips)
            videos = table["video_id"].to_pylist()
            seen_videos.update(videos)
            if any(v.split("_")[0] != category for v in videos):
                raise RuntimeError(f"{shard_key}: category/video mismatch")
            if any(s not in {"event", "sequence", "scene"} for s in table["scale"].to_pylist()):
                raise RuntimeError(f"{shard_key}: unknown scale")
            start = table["start_time"].to_numpy()
            end = table["end_time"].to_numpy()
            if not (np.isfinite(start).all() and np.isfinite(end).all() and (end > start).all()):
                raise RuntimeError(f"{shard_key}: invalid time interval")
            totals[category] += len(clips)
            expected_start += len(clips)
            shard_count += 1
        if expected_start != category_info["expected_rows"] or not category_info["complete"]:
            raise RuntimeError(f"{category}: category count mismatch")
    if sum(totals.values()) != EXPECTED_ROWS or len(seen_videos) != EXPECTED_VIDEOS:
        raise RuntimeError("Corpus row or video count mismatch")
    return {
        "complete": True, "rows": sum(totals.values()), "videos": len(seen_videos),
        "shards": shard_count, "per_category": dict(totals),
        "semantic_fingerprint": fingerprint,
        "manifest_sha256": sha256_file(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--token-file", type=Path, default=Path("HF_BUCKET/HF_TOKEN.txt"))
    args = parser.parse_args()
    if args.download:
        token = args.token_file.read_text().strip()
        if not token:
            raise RuntimeError("HF token file is empty")
        download(args.root, token, metadata_only=args.metadata_only)
    if not args.metadata_only:
        report = audit(args.root)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
