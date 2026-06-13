#!/usr/bin/env python3
"""Upload AIC image and audio embeddings to Milvus/Zilliz.

The script creates two independent collections:

- image keyframe vectors from peG14.pkl
- audio GLAP vectors from results_audio_event/audio_out/*.glap.npy

Metadata remains authoritative in Elasticsearch. Milvus stores the vector plus
stable join IDs so backend search can merge vector hits with Elastic records.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_IMAGE_COLLECTION = "aic26_image_peg14_v1"
DEFAULT_AUDIO_COLLECTION = "aic26_audio_glap_v1"
VECTOR_FIELD = "embedding"

_KEYFRAME_IMAGE_RE = re.compile(
    r"^Keyframes_[^/]+/keyframes/(?P<video_id>[^/]+)/(?P<n>\d+)\.jpg$"
)


def read_secret(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"{path} is empty")
    return value


def vector_dtype_name(vector_dtype: str) -> str:
    normalized = vector_dtype.lower()
    if normalized == "float32":
        return "FLOAT_VECTOR"
    if normalized == "float16":
        return "FLOAT16_VECTOR"
    raise ValueError(f"Unsupported vector dtype: {vector_dtype}")


def numpy_vector_dtype(vector_dtype: str) -> np.dtype:
    vector_dtype_name(vector_dtype)
    return np.dtype(np.float16 if vector_dtype.lower() == "float16" else np.float32)


def parse_keyframe_image_path(image_path: str) -> dict[str, Any]:
    match = _KEYFRAME_IMAGE_RE.match(image_path)
    if not match:
        raise ValueError(f"Unexpected keyframe image path: {image_path}")

    video_id = match.group("video_id")
    keyframe_n = int(match.group("n"))
    category = video_id.split("_", 1)[0]
    keyframe_id = f"{video_id}/{keyframe_n:03d}"
    submit_keyframe_id = f"{category}/{keyframe_id}"

    return {
        "category": category,
        "video_id": video_id,
        "keyframe_n": keyframe_n,
        "keyframe_id": keyframe_id,
        "submit_keyframe_id": submit_keyframe_id,
        "image_path": image_path,
    }


def build_image_entity(image_path: str, vector: np.ndarray, vector_dtype: str) -> dict[str, Any]:
    meta = parse_keyframe_image_path(image_path)
    return {
        "id": meta["submit_keyframe_id"],
        "keyframe_id": meta["keyframe_id"],
        "submit_keyframe_id": meta["submit_keyframe_id"],
        "category": meta["category"],
        "video_id": meta["video_id"],
        "keyframe_n": meta["keyframe_n"],
        "image_path": meta["image_path"],
        VECTOR_FIELD: np.asarray(vector, dtype=numpy_vector_dtype(vector_dtype)),
    }


def build_audio_entity(record: dict[str, Any], vector: np.ndarray, vector_dtype: str) -> dict[str, Any]:
    window_id = record.get("window_id")
    if not window_id:
        raise ValueError("Audio metadata record is missing window_id")

    return {
        "id": window_id,
        "video_id": record.get("video_id") or "",
        "start": float(record.get("start") or 0.0),
        "end": float(record.get("end") or 0.0),
        "glap_idx": int(record.get("glap_idx") or 0),
        "keyframe_id": record.get("keyframe_id") or "",
        "submit_keyframe_id": record.get("submit_keyframe_id") or "",
        "keyframe_n": int(record.get("keyframe_n") or 0),
        "top1_label": record.get("top1_label") or "",
        VECTOR_FIELD: np.asarray(vector, dtype=numpy_vector_dtype(vector_dtype)),
    }


def load_pe_embeddings(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, dict):
        raise ValueError("peG14.pkl must contain an 'embeddings' dict")
    return data


def iter_image_entities(path: Path, vector_dtype: str, skip: int = 0) -> Iterator[dict[str, Any]]:
    data = load_pe_embeddings(path)
    embeddings = data["embeddings"]
    embedding_dim = int(data.get("embedding_dim") or 0)
    if embedding_dim and embedding_dim != 1280:
        raise ValueError(f"Expected PE-G14 dim 1280, got {embedding_dim}")

    for index, (image_path, vector) in enumerate(embeddings.items()):
        if index < skip:
            continue
        if getattr(vector, "shape", None) != (1280,):
            raise ValueError(f"Bad image vector shape for {image_path}: {getattr(vector, 'shape', None)}")
        yield build_image_entity(image_path, vector, vector_dtype)


def iter_audio_entities(
    staging_jsonl: Path,
    glap_dir: Path,
    vector_dtype: str,
    skip: int = 0,
) -> Iterator[dict[str, Any]]:
    current_video_id: str | None = None
    current_vectors: np.ndarray | None = None

    with staging_jsonl.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < skip:
                continue

            record = json.loads(line)
            video_id = record.get("video_id")
            if not video_id:
                raise ValueError(f"Audio metadata line {index + 1} is missing video_id")

            if video_id != current_video_id:
                glap_path = glap_dir / f"{video_id}.glap.npy"
                if not glap_path.exists():
                    raise FileNotFoundError(glap_path)
                current_vectors = np.load(glap_path, mmap_mode="r")
                if current_vectors.ndim != 2 or current_vectors.shape[1] != 1024:
                    raise ValueError(f"Bad GLAP shape for {video_id}: {current_vectors.shape}")
                current_video_id = video_id

            assert current_vectors is not None
            glap_idx = int(record.get("glap_idx") or 0)
            if glap_idx < 0 or glap_idx >= current_vectors.shape[0]:
                raise ValueError(f"Bad glap_idx {glap_idx} for {video_id}: {current_vectors.shape}")
            yield build_audio_entity(record, current_vectors[glap_idx], vector_dtype)


def _import_pymilvus() -> tuple[Any, Any]:
    try:
        from pymilvus import DataType, MilvusClient
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing pymilvus. Install it with: uv pip install pymilvus numpy"
        ) from exc
    return MilvusClient, DataType


def connect_client(endpoint_file: Path, token_file: Path) -> Any:
    MilvusClient, _ = _import_pymilvus()
    return MilvusClient(
        uri=read_secret(endpoint_file),
        token=read_secret(token_file),
        timeout=30,
        alias=f"aic26_upload_{time.time_ns()}",
    )


def _vector_data_type(DataType: Any, vector_dtype: str) -> Any:
    name = vector_dtype_name(vector_dtype)
    if name == "FLOAT_VECTOR":
        return DataType.FLOAT_VECTOR
    if not hasattr(DataType, "FLOAT16_VECTOR"):
        raise ValueError("Installed pymilvus does not expose FLOAT16_VECTOR")
    return DataType.FLOAT16_VECTOR


def ensure_image_collection(client: Any, collection_name: str, vector_dtype: str, drop: bool = False) -> None:
    MilvusClient, DataType = _import_pymilvus()
    if client.has_collection(collection_name=collection_name):
        if not drop:
            return
        client.drop_collection(collection_name=collection_name)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(field_name="keyframe_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="submit_keyframe_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=16)
    schema.add_field(field_name="video_id", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="keyframe_n", datatype=DataType.INT64)
    schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(
        field_name=VECTOR_FIELD,
        datatype=_vector_data_type(DataType, vector_dtype),
        dim=1280,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD,
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


def ensure_audio_collection(client: Any, collection_name: str, vector_dtype: str, drop: bool = False) -> None:
    MilvusClient, DataType = _import_pymilvus()
    if client.has_collection(collection_name=collection_name):
        if not drop:
            return
        client.drop_collection(collection_name=collection_name)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=96)
    schema.add_field(field_name="video_id", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="start", datatype=DataType.FLOAT)
    schema.add_field(field_name="end", datatype=DataType.FLOAT)
    schema.add_field(field_name="glap_idx", datatype=DataType.INT64)
    schema.add_field(field_name="keyframe_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="submit_keyframe_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="keyframe_n", datatype=DataType.INT64)
    schema.add_field(field_name="top1_label", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(
        field_name=VECTOR_FIELD,
        datatype=_vector_data_type(DataType, vector_dtype),
        dim=1024,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD,
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


def batched(items: Iterator[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_checkpoint(path: Path | None, key: str, value: int) -> None:
    if not path:
        return
    state: dict[str, Any] = {}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    state[key] = value
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def read_checkpoint(path: Path | None, key: str) -> int:
    if not path or not path.exists():
        return 0
    state = json.loads(path.read_text(encoding="utf-8"))
    return int(state.get(key) or 0)


def upload_entities(
    client: Any,
    collection_name: str,
    entities: Iterator[dict[str, Any]],
    *,
    batch_size: int,
    op: str,
    label: str,
    checkpoint_path: Path | None = None,
    checkpoint_key: str | None = None,
    already_uploaded: int = 0,
    max_retries: int = 3,
    retry_sleep: float = 5.0,
    client_factory: Callable[[], Any] | None = None,
) -> int:
    uploaded = already_uploaded
    started = time.time()
    for batch in batched(entities, batch_size):
        for attempt in range(max_retries + 1):
            if attempt > 0 and client_factory is not None:
                try:
                    client = client_factory()
                except Exception as exc:
                    if attempt >= max_retries:
                        raise
                    delay = retry_sleep * (attempt + 1)
                    print(
                        f"{label}: reconnect failed after {uploaded:,} rows "
                        f"({type(exc).__name__}: {exc}); retry {attempt + 1}/{max_retries} "
                        f"in {delay:g}s",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
            try:
                if op == "insert":
                    client.insert(collection_name=collection_name, data=batch)
                elif op == "upsert":
                    client.upsert(collection_name=collection_name, data=batch)
                else:
                    raise ValueError(f"Unsupported op: {op}")
                break
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                delay = retry_sleep * (attempt + 1)
                print(
                    f"{label}: batch failed after {uploaded:,} rows "
                    f"({type(exc).__name__}: {exc}); retry {attempt + 1}/{max_retries} "
                    f"in {delay:g}s",
                    flush=True,
                )
                time.sleep(delay)
        uploaded += len(batch)
        if checkpoint_key:
            write_checkpoint(checkpoint_path, checkpoint_key, uploaded)
        elapsed = max(time.time() - started, 0.001)
        rate = uploaded / elapsed
        print(f"{label}: uploaded {uploaded:,} rows ({rate:,.1f}/s)", flush=True)
    return uploaded


def get_row_count(client: Any, collection_name: str) -> int | None:
    try:
        stats = client.get_collection_stats(collection_name=collection_name)
    except Exception:
        return None
    for key in ("row_count", "num_entities"):
        if key in stats:
            return int(stats[key])
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-file", type=Path, default=Path("milvus_endpoint.txt"))
    parser.add_argument("--token-file", type=Path, default=Path("milvus_token.txt"))
    parser.add_argument("--pe-path", type=Path, default=Path("peG14.pkl"))
    parser.add_argument("--audio-staging", type=Path, default=Path("elastic_staging/audio_windows_mapped.jsonl"))
    parser.add_argument("--glap-dir", type=Path, default=Path("results_audio_event/audio_out"))
    parser.add_argument("--image-collection", default=DEFAULT_IMAGE_COLLECTION)
    parser.add_argument("--audio-collection", default=DEFAULT_AUDIO_COLLECTION)
    parser.add_argument("--image-vector-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--audio-vector-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--only", nargs="*", choices=("image", "audio"), default=("image", "audio"))
    parser.add_argument("--op", choices=("upsert", "insert"), default="upsert")
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("milvus_upload_checkpoint.json"))
    parser.add_argument("--ignore-checkpoint", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    client = connect_client(args.endpoint_file, args.token_file)
    client_factory = lambda: connect_client(args.endpoint_file, args.token_file)

    if "image" in args.only:
        ensure_image_collection(
            client,
            args.image_collection,
            args.image_vector_dtype,
            drop=args.drop_existing,
        )
        checkpoint_key = f"{args.image_collection}:image"
        skip = 0 if args.ignore_checkpoint else read_checkpoint(args.checkpoint_path, checkpoint_key)
        print(
            f"Uploading image embeddings to {args.image_collection} "
            f"as {vector_dtype_name(args.image_vector_dtype)}; skip={skip:,}",
            flush=True,
        )
        upload_entities(
            client,
            args.image_collection,
            iter_image_entities(args.pe_path, args.image_vector_dtype, skip=skip),
            batch_size=args.batch_size,
            op=args.op,
            label="image",
            checkpoint_path=args.checkpoint_path,
            checkpoint_key=checkpoint_key,
            already_uploaded=skip,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            client_factory=client_factory,
        )
        print(f"image remote_count={get_row_count(client, args.image_collection)}", flush=True)

    if "audio" in args.only:
        ensure_audio_collection(
            client,
            args.audio_collection,
            args.audio_vector_dtype,
            drop=args.drop_existing,
        )
        checkpoint_key = f"{args.audio_collection}:audio"
        skip = 0 if args.ignore_checkpoint else read_checkpoint(args.checkpoint_path, checkpoint_key)
        print(
            f"Uploading GLAP embeddings to {args.audio_collection} "
            f"as {vector_dtype_name(args.audio_vector_dtype)}; skip={skip:,}",
            flush=True,
        )
        upload_entities(
            client,
            args.audio_collection,
            iter_audio_entities(args.audio_staging, args.glap_dir, args.audio_vector_dtype, skip=skip),
            batch_size=args.batch_size,
            op=args.op,
            label="audio",
            checkpoint_path=args.checkpoint_path,
            checkpoint_key=checkpoint_key,
            already_uploaded=skip,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            client_factory=client_factory,
        )
        print(f"audio remote_count={get_row_count(client, args.audio_collection)}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
