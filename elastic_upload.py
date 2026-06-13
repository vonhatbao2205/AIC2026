#!/usr/bin/env python3
"""Create Elasticsearch indices and bulk-upload retrieval metadata."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any


DEFAULT_INDEX_PREFIX = "aic26"


def read_secret(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"{path} is empty")
    return value


def auth_headers(api_key: str, *, ndjson: bool = False) -> dict[str, str]:
    token = api_key.strip()
    if not token.lower().startswith("apikey "):
        token = f"ApiKey {token}"
    content_type = "application/x-ndjson" if ndjson else "application/json"
    return {
        "Authorization": token,
        "Content-Type": content_type,
        "Accept": "application/json",
    }


def endpoint_join(endpoint: str, path: str) -> str:
    return endpoint.rstrip("/") + "/" + path.lstrip("/")


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def build_bulk_payload(index_name: str, records: list[dict[str, Any]], *, id_field: str) -> bytes:
    lines: list[str] = []
    for record in records:
        doc_id = record[id_field]
        lines.append(json.dumps({"index": {"_index": index_name, "_id": doc_id}}, ensure_ascii=False))
        lines.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode("utf-8")


def submit_category(video_id: str) -> str:
    return video_id.split("_", 1)[0]


def normalize_ocr_record(record: dict[str, Any]) -> dict[str, Any]:
    raw_id = record["id"]
    parts = raw_id.split("/")
    if len(parts) != 3:
        raise ValueError(f"Unexpected OCR id format: {raw_id}")
    _, video_id, frame_name = parts
    keyframe_n = int(frame_name)
    category = record.get("category") or submit_category(video_id)
    submit_cat = submit_category(video_id)

    return {
        "ocr_id": raw_id,
        "submit_keyframe_id": f"{submit_cat}/{video_id}/{keyframe_n:03d}",
        "video_id": video_id,
        "category": category,
        "submit_category": submit_cat,
        "image_path": record.get("image_path"),
        "status": record.get("status"),
        "error": record.get("error"),
        "keyframe_n": keyframe_n,
        "keyframe_name": f"{keyframe_n:03d}",
        "text_clean": record.get("text_clean") or "",
        "text_clean_fold": record.get("text_clean_fold") or "",
        "text_nfc": record.get("text_nfc") or "",
        "clock": record.get("clock"),
        "hour": record.get("hour"),
        "boxes": record.get("boxes") or [],
        "ts": record.get("ts"),
    }


def identity_record(record: dict[str, Any]) -> dict[str, Any]:
    return record


@dataclass(frozen=True)
class UploadJob:
    logical_name: str
    index_name: str
    source_path: Path
    id_field: str
    mapping: dict[str, Any]
    transform: Callable[[dict[str, Any]], dict[str, Any]] = identity_record


class ElasticClient:
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        ndjson: bool = False,
        expected: tuple[int, ...] = (200,),
        timeout: int = 120,
        retries: int = 5,
    ) -> tuple[int, bytes]:
        retryable_http = {429, 502, 503, 504}
        for attempt in range(retries + 1):
            request = urllib.request.Request(
                endpoint_join(self.endpoint, path),
                data=body,
                method=method,
                headers=auth_headers(self.api_key, ndjson=ndjson),
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = response.read()
                    status = response.status
            except urllib.error.HTTPError as exc:
                data = exc.read()
                status = exc.code
                if status in retryable_http and attempt < retries:
                    time.sleep(min(2**attempt, 30))
                    continue
            except (TimeoutError, urllib.error.URLError) as exc:
                if attempt < retries:
                    print(f"Retrying {method} {path} after transient network error: {exc}")
                    time.sleep(min(2**attempt, 30))
                    continue
                raise
            if status not in expected:
                snippet = data[:1000].decode("utf-8", errors="replace")
                raise RuntimeError(f"{method} {path} returned HTTP {status}: {snippet}")
            return status, data
        raise RuntimeError(f"{method} {path} exhausted retries")

    def json_request(
        self,
        method: str,
        path: str,
        *,
        payload: Any | None = None,
        expected: tuple[int, ...] = (200,),
        timeout: int = 120,
    ) -> Any:
        body = None if payload is None else json_bytes(payload)
        _, data = self.request(method, path, body=body, expected=expected, timeout=timeout)
        if not data:
            return None
        return json.loads(data.decode("utf-8"))

    def index_exists(self, index_name: str) -> bool:
        status, _ = self.request("HEAD", index_name, expected=(200, 404), timeout=30)
        return status == 200

    def create_index_if_missing(self, index_name: str, mapping: dict[str, Any]) -> str:
        if self.index_exists(index_name):
            return "exists"
        self.json_request("PUT", index_name, payload=mapping, expected=(200,), timeout=120)
        return "created"

    def bulk_ndjson(self, payload: bytes) -> dict[str, Any]:
        _, data = self.request(
            "POST",
            "_bulk?refresh=false",
            body=payload,
            ndjson=True,
            expected=(200,),
            timeout=240,
        )
        result = json.loads(data.decode("utf-8"))
        if result.get("errors"):
            first_error = next(
                (
                    item
                    for item in result.get("items", [])
                    if item.get("index", {}).get("error")
                ),
                None,
            )
            raise RuntimeError(f"Bulk request had item errors: {first_error}")
        return result

    def refresh(self, index_name: str) -> None:
        self.json_request("POST", f"{index_name}/_refresh", expected=(200,), timeout=120)

    def count(self, index_name: str) -> int:
        result = self.json_request("GET", f"{index_name}/_count", expected=(200,), timeout=120)
        return int(result["count"])


def keyword(ignore_above: int = 512) -> dict[str, Any]:
    return {"type": "keyword", "ignore_above": ignore_above}


def text_with_keyword() -> dict[str, Any]:
    return {"type": "text", "fields": {"keyword": keyword(1024)}}


def base_index_settings() -> dict[str, Any]:
    return {}


def keyframe_mapping() -> dict[str, Any]:
    return {
        **base_index_settings(),
        "mappings": {
            "dynamic": False,
            "properties": {
                "keyframe_id": keyword(),
                "submit_keyframe_id": keyword(),
                "video_id": keyword(),
                "category_hint": keyword(),
                "submit_category": keyword(),
                "keyframe_n": {"type": "integer"},
                "keyframe_name": keyword(),
                "pts_time": {"type": "float"},
                "fps": {"type": "float"},
                "frame_idx": {"type": "integer"},
            },
        },
    }


KEYFRAME_FIELD_PROPS = {
    "keyframe_id": keyword(),
    "submit_keyframe_id": keyword(),
    "keyframe_n": {"type": "integer"},
    "keyframe_pts_time": {"type": "float"},
    "keyframe_frame_idx": {"type": "integer"},
    "start_keyframe_id": keyword(),
    "start_submit_keyframe_id": keyword(),
    "start_keyframe_n": {"type": "integer"},
    "start_keyframe_pts_time": {"type": "float"},
    "start_keyframe_frame_idx": {"type": "integer"},
    "center_keyframe_id": keyword(),
    "center_submit_keyframe_id": keyword(),
    "center_keyframe_n": {"type": "integer"},
    "center_keyframe_pts_time": {"type": "float"},
    "center_keyframe_frame_idx": {"type": "integer"},
    "end_keyframe_id": keyword(),
    "end_submit_keyframe_id": keyword(),
    "end_keyframe_n": {"type": "integer"},
    "end_keyframe_pts_time": {"type": "float"},
    "end_keyframe_frame_idx": {"type": "integer"},
}


def audio_mapping() -> dict[str, Any]:
    return {
        **base_index_settings(),
        "mappings": {
            "dynamic": False,
            "properties": {
                "window_id": keyword(),
                "video_id": keyword(),
                "start": {"type": "float"},
                "end": {"type": "float"},
                "duration": {"type": "float"},
                "center_time": {"type": "float"},
                "glap_idx": {"type": "integer"},
                "tags": {
                    "type": "nested",
                    "properties": {
                        "label": keyword(),
                        "score": {"type": "float"},
                    },
                },
                "tag_labels": keyword(),
                "tag_scores": {"type": "float"},
                "top1_label": keyword(),
                "top1_score": {"type": "float"},
                "caption": text_with_keyword(),
                "has_caption": {"type": "boolean"},
                "caption_quality": keyword(),
                "audio_stoplist_hit": {"type": "boolean"},
                "top1_is_stoplisted": {"type": "boolean"},
                "source_audio_file": keyword(1024),
                **KEYFRAME_FIELD_PROPS,
            },
        },
    }


def speech_mapping() -> dict[str, Any]:
    return {
        **base_index_settings(),
        "mappings": {
            "dynamic": False,
            "properties": {
                "segment_id": keyword(),
                "video_id": keyword(),
                "start": {"type": "float"},
                "end": {"type": "float"},
                "duration": {"type": "float"},
                "center_time": {"type": "float"},
                "text": text_with_keyword(),
                "avg_word_score": {"type": "float"},
                "confidence_bucket": keyword(),
                "segment_role": keyword(),
                "word_count": {"type": "integer"},
                "avg_logprob": {"type": "float"},
                "no_speech_prob": {"type": "float"},
                "source_segment_idx": {"type": "integer"},
                "source_speech_file": keyword(1024),
                **KEYFRAME_FIELD_PROPS,
            },
        },
    }


def ocr_mapping() -> dict[str, Any]:
    return {
        **base_index_settings(),
        "mappings": {
            "dynamic": False,
            "properties": {
                "ocr_id": keyword(1024),
                "submit_keyframe_id": keyword(),
                "video_id": keyword(),
                "category": keyword(),
                "submit_category": keyword(),
                "image_path": keyword(2048),
                "status": keyword(),
                "error": keyword(1024),
                "keyframe_n": {"type": "integer"},
                "keyframe_name": keyword(),
                "text_clean": text_with_keyword(),
                "text_clean_fold": text_with_keyword(),
                "text_nfc": text_with_keyword(),
                "clock": keyword(),
                "hour": {"type": "integer"},
                "boxes": {
                    "type": "nested",
                    "properties": {
                        "text": text_with_keyword(),
                        "box": {"type": "integer"},
                    },
                },
                "ts": {"type": "double"},
            },
        },
    }


def iter_jsonl(path: Path, transform: Callable[[dict[str, Any]], dict[str, Any]]) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield transform(json.loads(line))


def drop_first(iterator: Iterator[Any], count: int) -> Iterator[Any]:
    return islice(iterator, count, None)


def chunked(iterator: Iterator[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def index_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}_v1"


def build_jobs(prefix: str, staging_dir: Path, ocr_path: Path | None) -> list[UploadJob]:
    jobs = [
        UploadJob(
            logical_name="keyframe_map",
            index_name=index_name(prefix, "keyframe_map"),
            source_path=staging_dir / "keyframe_map.jsonl",
            id_field="submit_keyframe_id",
            mapping=keyframe_mapping(),
        ),
        UploadJob(
            logical_name="speech_segments",
            index_name=index_name(prefix, "speech_segments"),
            source_path=staging_dir / "speech_segments_mapped.jsonl",
            id_field="segment_id",
            mapping=speech_mapping(),
        ),
        UploadJob(
            logical_name="audio_windows",
            index_name=index_name(prefix, "audio_windows"),
            source_path=staging_dir / "audio_windows_mapped.jsonl",
            id_field="window_id",
            mapping=audio_mapping(),
        ),
    ]
    if ocr_path is not None:
        jobs.append(
            UploadJob(
                logical_name="ocr_keyframes",
                index_name=index_name(prefix, "ocr_keyframes"),
                source_path=ocr_path,
                id_field="submit_keyframe_id",
                mapping=ocr_mapping(),
                transform=normalize_ocr_record,
            )
        )
    return jobs


def upload_job(
    client: ElasticClient,
    job: UploadJob,
    batch_size: int,
    *,
    resume_existing: bool = False,
) -> dict[str, Any]:
    status = client.create_index_if_missing(job.index_name, job.mapping)
    skipped = 0
    if resume_existing and status == "exists":
        client.refresh(job.index_name)
        skipped = client.count(job.index_name)
    uploaded = 0
    took_ms = 0
    started = time.time()
    print(f"[{job.index_name}] index {status}; source={job.source_path}; skip={skipped:,}")
    records = iter_jsonl(job.source_path, job.transform)
    if skipped:
        records = drop_first(records, skipped)
    for batch in chunked(records, batch_size):
        payload = build_bulk_payload(job.index_name, batch, id_field=job.id_field)
        result = client.bulk_ndjson(payload)
        uploaded += len(batch)
        took_ms += int(result.get("took", 0))
        source_position = skipped + uploaded
        if source_position % (batch_size * 10) == 0:
            print(f"[{job.index_name}] uploaded source position {source_position:,}")
    client.refresh(job.index_name)
    remote_count = client.count(job.index_name)
    elapsed = round(time.time() - started, 2)
    print(
        f"[{job.index_name}] uploaded_new={uploaded:,}; skipped={skipped:,}; "
        f"remote_count={remote_count:,}; elapsed={elapsed}s"
    )
    return {
        "index": job.index_name,
        "logical_name": job.logical_name,
        "index_status": status,
        "source": str(job.source_path),
        "uploaded": uploaded,
        "skipped_existing": skipped,
        "remote_count": remote_count,
        "bulk_took_ms": took_ms,
        "elapsed_s": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-file", type=Path, default=Path("elastic_endpoint.txt"))
    parser.add_argument("--api-key-file", type=Path, default=Path("elastic_apikey.txt"))
    parser.add_argument("--staging-dir", type=Path, default=Path("elastic_staging"))
    parser.add_argument("--ocr-path", type=Path, default=Path("ocr_clean.jsonl"))
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--index-prefix", default=DEFAULT_INDEX_PREFIX)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["keyframe_map", "speech_segments", "audio_windows", "ocr_keyframes"],
        help="Upload only selected logical indices.",
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--summary-path", type=Path, default=Path("elastic_staging/elastic_upload_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint = read_secret(args.endpoint_file)
    api_key = read_secret(args.api_key_file)
    ocr_path = None if args.no_ocr else args.ocr_path
    jobs = build_jobs(args.index_prefix, args.staging_dir, ocr_path)
    if args.only:
        allowed = set(args.only)
        jobs = [job for job in jobs if job.logical_name in allowed]

    client = ElasticClient(endpoint, api_key)
    cluster_info = client.json_request("GET", "/", expected=(200,), timeout=60)
    print(
        "Connected to Elasticsearch",
        json.dumps(
            {
                "cluster_name": cluster_info.get("cluster_name"),
                "version": (cluster_info.get("version") or {}).get("number"),
                "tagline": cluster_info.get("tagline"),
            },
            ensure_ascii=False,
        ),
    )

    results = [
        upload_job(client, job, args.batch_size, resume_existing=args.resume_existing)
        for job in jobs
    ]
    summary = {
        "endpoint_file": str(args.endpoint_file),
        "index_prefix": args.index_prefix,
        "batch_size": args.batch_size,
        "resume_existing": args.resume_existing,
        "results": results,
    }
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
