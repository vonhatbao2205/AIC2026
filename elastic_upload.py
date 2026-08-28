#!/usr/bin/env python3
"""Create Elasticsearch indices and bulk-upload retrieval metadata.

Sources: local staging JSONL (BTC keyframe map / speech / audio / OCR), final
InfoShot++ map CSVs, and, for the object-detection index, the committed
`od-frame-v5` shards on Cloudflare R2 — read and checksum-verified straight from
the bucket, no local staging step.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import hmac
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Protocol


DEFAULT_INDEX_PREFIX = "aic26"
DEFAULT_R2_BUCKET = "aic26-media"
DEFAULT_MAX_BULK_BYTES = 8 * 1024 * 1024
DEFAULT_INFOSHOTPP_MAP_ROOT = Path(
    "/home/bao/Projects/ExtractKeyframe/keyframe_L/infoshootpp/map-keyframes"
)
DEFAULT_INFOSHOTPP_MAP_INDEX = "aic26_keyframe_map_infoshotpp_v1"
EXPECTED_INFOSHOTPP_VIDEOS = 873
EXPECTED_INFOSHOTPP_KEYFRAMES = 1_339_055
EXPECTED_INFOSHOTPP_CATEGORIES = {f"L{number:02d}" for number in range(21, 31)}


def read_secret(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"{path} is empty")
    return value


def read_r2_credentials(path: Path | None) -> dict[str, str]:
    """Resolve R2 S3 credentials from the environment, else from a `Label: value` file.

    Environment wins so a CI/container run never needs the loose secret file
    (`CloudflareR2/cloudflareR2_api.txt`, which is git-ignored).
    """
    creds = {
        "access_key": os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID") or "",
        "secret_key": os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY") or "",
        "account_id": os.environ.get("R2_ACCOUNT_ID") or "",
        "endpoint": os.environ.get("R2_ENDPOINT") or "",
    }
    labels = {
        "access key id": "access_key",
        "secret access key": "secret_key",
        "account id": "account_id",
        "endpoint": "endpoint",
    }
    if path is not None and path.exists() and not (creds["access_key"] and creds["secret_key"]):
        for raw in path.read_text(encoding="utf-8").splitlines():
            label, sep, value = raw.partition(":")
            field = labels.get(label.strip().lower())
            if sep and field and not creds[field]:
                creds[field] = value.strip()
    if not creds["endpoint"] and creds["account_id"]:
        creds["endpoint"] = f"https://{creds['account_id']}.r2.cloudflarestorage.com"
    missing = [name for name in ("access_key", "secret_key", "endpoint") if not creds[name]]
    if missing:
        raise RuntimeError(
            f"Thiếu R2 credentials: {', '.join(missing)}. Đặt R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/"
            f"R2_ACCOUNT_ID hoặc trỏ --r2-credentials-file tới file 'Label: value'."
        )
    return creds


class ObjectStore(Protocol):
    """The two read operations the OD import needs, so tests can pass a fake."""

    def list_keys(self, prefix: str) -> Iterator[str]: ...

    def get_object(self, key: str) -> bytes: ...


S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_EMPTY_PAYLOAD_SHA256 = hashlib.sha256(b"").hexdigest()


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


class R2Client:
    """Read-only S3/SigV4 client for Cloudflare R2, stdlib only.

    The OD import needs exactly two S3 operations (ListObjectsV2 + GetObject), so
    they are signed here rather than adding a boto3 dependency — this uploader is
    documented as runnable with a bare Python install.
    """

    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str, *, region: str = "auto"):
        self.endpoint = endpoint.rstrip("/")
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region

    def _canonical_uri(self, key: str) -> str:
        base = "/" + urllib.parse.quote(self.bucket, safe="")
        return f"{base}/{urllib.parse.quote(key, safe='/')}" if key else base

    def _signed_headers(self, canonical_uri: str, query: dict[str, str], now: datetime) -> dict[str, str]:
        host = urllib.parse.urlsplit(self.endpoint).netloc
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        canonical_query = urllib.parse.urlencode(sorted(query.items()), quote_via=urllib.parse.quote)
        canonical_headers = (
            f"host:{host}\nx-amz-content-sha256:{_EMPTY_PAYLOAD_SHA256}\nx-amz-date:{amz_date}\n"
        )
        signed_headers = "host;x-amz-content-sha256;x-amz-date"
        canonical_request = "\n".join([
            "GET", canonical_uri, canonical_query, canonical_headers, signed_headers, _EMPTY_PAYLOAD_SHA256,
        ])
        scope = f"{date_stamp}/{self.region}/s3/aws4_request"
        string_to_sign = "\n".join([
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ])
        signing_key = _sign(
            _sign(_sign(_sign(f"AWS4{self.secret_key}".encode("utf-8"), date_stamp), self.region), "s3"),
            "aws4_request",
        )
        signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        return {
            "Authorization": (
                f"AWS4-HMAC-SHA256 Credential={self.access_key}/{scope}, "
                f"SignedHeaders={signed_headers}, Signature={signature}"
            ),
            "x-amz-content-sha256": _EMPTY_PAYLOAD_SHA256,
            "x-amz-date": amz_date,
        }

    def _get(self, canonical_uri: str, query: dict[str, str], *, timeout: int = 180, retries: int = 5) -> bytes:
        url = self.endpoint + canonical_uri
        if query:
            url += "?" + urllib.parse.urlencode(sorted(query.items()), quote_via=urllib.parse.quote)
        for attempt in range(retries + 1):
            # Re-signed per attempt: an SigV4 signature expires, so a retry that
            # reused the first one would fail authentication instead of the
            # transient error it is meant to recover from.
            request = urllib.request.Request(
                url, method="GET", headers=self._signed_headers(canonical_uri, query, datetime.now(timezone.utc))
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                    time.sleep(min(2**attempt, 30))
                    continue
                snippet = exc.read()[:500].decode("utf-8", errors="replace")
                raise RuntimeError(f"R2 GET {canonical_uri} returned HTTP {exc.code}: {snippet}") from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                if attempt < retries:
                    print(f"Retrying R2 GET {canonical_uri} after transient network error: {exc}")
                    time.sleep(min(2**attempt, 30))
                    continue
                raise
        raise RuntimeError(f"R2 GET {canonical_uri} exhausted retries")

    def get_object(self, key: str) -> bytes:
        return self._get(self._canonical_uri(key), {})

    def list_keys(self, prefix: str) -> Iterator[str]:
        token: str | None = None
        while True:
            query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if token:
                query["continuation-token"] = token
            root = ET.fromstring(self._get(self._canonical_uri(""), query))
            for contents in root.findall(f"{S3_NS}Contents"):
                key = contents.findtext(f"{S3_NS}Key")
                if key:
                    yield key
            if root.findtext(f"{S3_NS}IsTruncated") != "true":
                return
            token = root.findtext(f"{S3_NS}NextContinuationToken")
            if not token:
                return


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


def normalize_od_record(record: dict[str, Any]) -> dict[str, Any]:
    """Attach the repo-wide submit identity to an OD frame document.

    OD documents are keyed by `document_id = "<video_id>:<frame_name>"`, but every
    other index here joins on `submit_keyframe_id`. Deriving it at index time keeps
    the OD index usable from the retrieval app without rewriting the immutable R2
    shards.
    """
    document_id = record.get("document_id")
    video_id = record.get("video_id")
    frame_name = str(record.get("frame_name") or "")
    if not document_id or not video_id or not frame_name.isdigit():
        raise ValueError(f"Unexpected OD document identity: {document_id!r} ({video_id!r}/{frame_name!r})")
    keyframe_n = int(frame_name)
    submit_cat = submit_category(video_id)
    return {
        **record,
        "submit_keyframe_id": f"{submit_cat}/{video_id}/{keyframe_n:03d}",
        "submit_category": submit_cat,
        "keyframe_n": keyframe_n,
    }


def identity_record(record: dict[str, Any]) -> dict[str, Any]:
    return record


@dataclass(frozen=True)
class UploadJob:
    logical_name: str
    index_name: str
    id_field: str
    mapping: dict[str, Any]
    source_label: str
    # Called with the number of leading records to skip (resume), so a source can
    # skip cheaply — the OD source drops whole committed shards without fetching.
    open_records: Callable[[int], Iterator[dict[str, Any]]]
    # Applied when the index already exists: adding properties to a `dynamic:
    # strict` mapping is the only way documents enriched here can be accepted by
    # an index that the OD notebook created with the bare v5 schema.
    mapping_patch: dict[str, Any] | None = None


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
                "frame_id": keyword(),
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


# Identity fields this uploader adds on top of the frozen `od-frame-v5` schema.
OD_SUBMIT_IDENTITY_PROPS = {
    "submit_keyframe_id": keyword(),
    "submit_category": keyword(),
    "keyframe_n": {"type": "integer"},
}


def od_frames_mapping() -> dict[str, Any]:
    """Strict `od-frame-v5` mapping — kept field-for-field in sync with
    `od_elastic.index_mapping()` in OD/OD_Kaggle_Notebook.ipynb, plus the submit
    identity fields above. `dynamic: strict` is deliberate: an unexpected field
    must fail the import instead of silently creating an unqueryable mapping."""
    detections_props = {
        "instance_id": {"type": "keyword"},
        "label": {"type": "keyword"},
        "canonical_label": {"type": "keyword"},
        "aliases": {"type": "keyword"},
        "label_group": {"type": "keyword"},
        "conf": {"type": "half_float"},
        "countable": {"type": "boolean"},
        "counted": {"type": "boolean"},
        "count_threshold_used": {"type": "half_float"},
        "count_source": {"type": "keyword"},
        "sources": {"type": "keyword"},
        "source_scores": {"type": "object", "enabled": False},
        "dominant_color": {"type": "keyword"},
        "color_names": {"type": "keyword"},
        "colors": {"type": "object", "enabled": False},
        "color_method": {"type": "keyword"},
        "color_confidence": {"type": "half_float"},
        "color_purity": {"type": "half_float"},
        "color_entropy": {"type": "half_float"},
        "color_delta_e_p50": {"type": "half_float"},
        "color_coverage": {"type": "half_float"},
        "color_reliable": {"type": "boolean"},
        "bbox_norm": {"properties": {axis: {"type": "float"} for axis in ("x1", "y1", "x2", "y2")}},
        "bbox_px": {"properties": {axis: {"type": "integer"} for axis in ("x1", "y1", "x2", "y2")}},
        "center_norm": {"properties": {"x": {"type": "float"}, "y": {"type": "float"}}},
        "size_norm": {
            "properties": {
                "width": {"type": "float"},
                "height": {"type": "float"},
                "area": {"type": "float"},
            }
        },
        "grid7": {"properties": {"row": {"type": "byte"}, "col": {"type": "byte"}, "id": {"type": "keyword"}}},
        "horizontal_zone": {"type": "keyword"},
        "vertical_zone": {"type": "keyword"},
        "position": {"type": "keyword"},
        "position_tags": {"type": "keyword"},
    }
    object_counts_props = {
        "label": {"type": "keyword"},
        "count": {"type": "integer"},
        "countable": {"type": "boolean"},
        "max_conf": {"type": "half_float"},
        "mean_conf": {"type": "half_float"},
        "aliases": {"type": "keyword"},
        "is_estimate": {"type": "boolean"},
        "is_lower_bound": {"type": "boolean"},
        "lower_bound_reasons": {"type": "keyword"},
        "count_quality": {"type": "keyword"},
        "thresholds_used": {"type": "half_float"},
        "sources": {"type": "keyword"},
    }
    return {
        # No `settings` block on purpose: this repo targets Elastic serverless,
        # which rejects number_of_shards/number_of_replicas/refresh_interval and
        # sizes indices itself. `od_elastic.index_mapping()` still sets them for a
        # self-managed cluster; the field mappings below are what must stay in sync.
        **base_index_settings(),
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "schema_version": {"type": "keyword"},
                "pipeline_version": {"type": "keyword"},
                "config_hash": {"type": "keyword"},
                "manifest_hash": {"type": "keyword"},
                "runtime_hash": {"type": "keyword"},
                "resolved_detector": {"type": "keyword"},
                "weight_sha256": {"type": "keyword"},
                "text_encoder_sha256": {"type": "keyword"},
                "prompt_embeddings_sha256": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "keyframe_id": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "frame_name": {"type": "keyword"},
                "frame_index": {"type": "integer"},
                "batch": {"type": "keyword"},
                "genre": {"type": "keyword"},
                "r2_key": {"type": "keyword", "index": False, "doc_values": False},
                "image_url": {"type": "keyword", "index": False, "doc_values": False},
                "input_etag": {"type": "keyword"},
                "status": {"type": "keyword"},
                "fallback_status": {"type": "keyword"},
                "fallback_reasons": {"type": "keyword"},
                "fallback_error": {"type": "text", "index": False},
                "error": {"type": "text", "index": False},
                "generated_at": {"type": "date"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "detection_count": {"type": "integer"},
                "counted_detection_count": {"type": "integer"},
                "labels": {"type": "keyword"},
                "canonical_labels": {"type": "keyword"},
                "spatial_tokens": {"type": "keyword"},
                "count_tokens": {"type": "keyword"},
                "object_text": {"type": "text", "analyzer": "standard"},
                "crowd_present": {"type": "boolean"},
                "counts_are_estimates": {"type": "boolean"},
                "count_is_lower_bound": {"type": "boolean"},
                "count_uncertain_labels": {"type": "keyword"},
                "count_notes": {"type": "keyword", "index": False},
                "pipeline": {"type": "object", "enabled": False},
                "object_counts": {"type": "nested", "dynamic": "strict", "properties": object_counts_props},
                "detections": {"type": "nested", "dynamic": "strict", "properties": detections_props},
                **OD_SUBMIT_IDENTITY_PROPS,
            },
        },
    }


OD_MARKER_RE = re.compile(r"part-(\d{6})\.complete\.json$")


def od_committed_shard_ids(r2: ObjectStore, output_prefix: str) -> list[int]:
    """Shard ids that reached the commit point (a completion marker exists).

    A raw `shards/part-*.jsonl.gz` without its marker is an interrupted shard and
    is deliberately ignored — the OD pipeline treats the marker, not the data
    file, as the commit.
    """
    ids = []
    for key in r2.list_keys(f"{output_prefix.rstrip('/')}/complete/"):
        match = OD_MARKER_RE.search(key)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def iter_od_documents(
    r2: ObjectStore,
    output_prefix: str,
    shard_ids: list[int],
    *,
    skip: int = 0,
    expected_hashes: dict[str, str] | None = None,
    verify_checksum: bool = True,
) -> Iterator[dict[str, Any]]:
    """Stream OD frame documents from committed R2 shards, newest identity checks first.

    Each shard is verified against its completion marker (namespace hashes, gzip
    SHA-256, document count) before any of its documents are indexed, so a
    truncated upload or a mixed-namespace prefix fails loudly instead of writing
    a half-valid index.
    """
    prefix = output_prefix.rstrip("/")
    remaining_skip = max(0, skip)
    for position, shard_id in enumerate(shard_ids, 1):
        marker_key = f"{prefix}/complete/part-{shard_id:06d}.complete.json"
        marker = json.loads(r2.get_object(marker_key).decode("utf-8"))
        document_count = int(marker.get("document_count") or 0)
        mismatched = [
            field
            for field, value in (expected_hashes or {}).items()
            if value and marker.get(field) != value
        ]
        if mismatched:
            raise RuntimeError(f"{marker_key}: namespace mismatch on {', '.join(mismatched)}")
        if int(marker.get("error_count") or 0):
            raise RuntimeError(f"{marker_key}: error_count != 0; shard chưa phải commit hợp lệ")
        if remaining_skip >= document_count:
            remaining_skip -= document_count
            continue
        payload = r2.get_object(f"{prefix}/shards/part-{shard_id:06d}.jsonl.gz")
        if verify_checksum and marker.get("sha256"):
            digest = hashlib.sha256(payload).hexdigest()
            if digest != marker["sha256"]:
                raise RuntimeError(
                    f"Shard part-{shard_id:06d} checksum mismatch: {digest} != {marker['sha256']}"
                )
        emitted = 0
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                emitted += 1
                if remaining_skip:
                    remaining_skip -= 1
                    continue
                yield json.loads(raw)
        if emitted != document_count:
            raise RuntimeError(
                f"Shard part-{shard_id:06d}: {emitted} documents != marker document_count {document_count}"
            )
        print(f"[od_frames] shard {position}/{len(shard_ids)} part-{shard_id:06d} ({document_count:,} docs)")


def od_source(
    r2: ObjectStore,
    output_prefix: str,
    *,
    expected_hashes: dict[str, str] | None = None,
    max_shards: int | None = None,
    verify_checksum: bool = True,
) -> Callable[[int], Iterator[dict[str, Any]]]:
    def open_records(skip: int) -> Iterator[dict[str, Any]]:
        shard_ids = od_committed_shard_ids(r2, output_prefix)
        if not shard_ids:
            raise RuntimeError(f"Không có completion marker nào dưới {output_prefix.rstrip('/')}/complete/")
        if max_shards is not None:
            shard_ids = shard_ids[:max_shards]
        print(
            f"[od_frames] committed shards={len(shard_ids)} "
            f"(part-{shard_ids[0]:06d}..part-{shard_ids[-1]:06d})"
        )
        for document in iter_od_documents(
            r2,
            output_prefix,
            shard_ids,
            skip=skip,
            expected_hashes=expected_hashes,
            verify_checksum=verify_checksum,
        ):
            yield normalize_od_record(document)

    return open_records


def jsonl_source(
    path: Path, transform: Callable[[dict[str, Any]], dict[str, Any]] = identity_record
) -> Callable[[int], Iterator[dict[str, Any]]]:
    def open_records(skip: int) -> Iterator[dict[str, Any]]:
        return drop_first(iter_jsonl(path, transform), skip)

    return open_records


def _infoshotpp_map_files(root: Path) -> list[Path]:
    files = sorted(root.glob("L*/L*_V*.csv"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy map-keyframe CSV dưới {root}")
    return files


def iter_infoshotpp_keyframes(root: Path, *, skip: int = 0) -> Iterator[dict[str, Any]]:
    """Read final InfoShot++ maps without conflating ordinal ``n`` and frame_idx.

    The final JPEG is ``f{frame_idx:08d}.jpg`` while the application identity is
    ``<category>/<video_id>/<n:03d>``.  ``pts_time`` is deliberately not required
    to be monotonic: the final L25 maps contain a handful of sub-frame regressions,
    while frame_idx remains the canonical strictly-increasing decode identity.
    """
    remaining = max(0, skip)
    for path in _infoshotpp_map_files(root):
        video_id = path.stem
        category = path.parent.name
        if video_id.split("_", 1)[0] != category:
            raise ValueError(f"{path}: category/video mismatch")
        expected_n = 1
        previous_frame_idx = -1
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["n", "pts_time", "fps", "frame_idx"]:
                raise ValueError(f"{path}: unexpected header {reader.fieldnames!r}")
            for line_number, row in enumerate(reader, 2):
                try:
                    n = int(row["n"])
                    pts_time = float(row["pts_time"])
                    fps = float(row["fps"])
                    frame_idx = int(row["frame_idx"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid map row") from exc
                if n != expected_n:
                    raise ValueError(f"{path}:{line_number}: n={n}, expected {expected_n}")
                if frame_idx <= previous_frame_idx:
                    raise ValueError(
                        f"{path}:{line_number}: frame_idx={frame_idx} is not strictly increasing"
                    )
                if not math.isfinite(pts_time) or not math.isfinite(fps) or fps <= 0:
                    raise ValueError(f"{path}:{line_number}: invalid pts_time/fps")
                expected_n += 1
                previous_frame_idx = frame_idx
                if remaining:
                    remaining -= 1
                    continue
                keyframe_name = f"{n:03d}"
                yield {
                    "keyframe_id": f"{video_id}/{keyframe_name}",
                    "frame_id": f"{video_id}@f{frame_idx:08d}",
                    "submit_keyframe_id": f"{category}/{video_id}/{keyframe_name}",
                    "video_id": video_id,
                    "category_hint": category,
                    "submit_category": category,
                    "keyframe_n": n,
                    "keyframe_name": keyframe_name,
                    "pts_time": pts_time,
                    "fps": fps,
                    "frame_idx": frame_idx,
                }


def infoshotpp_source(root: Path) -> Callable[[int], Iterator[dict[str, Any]]]:
    def open_records(skip: int) -> Iterator[dict[str, Any]]:
        return iter_infoshotpp_keyframes(root, skip=skip)

    return open_records


def audit_infoshotpp_maps(root: Path) -> dict[str, Any]:
    """Fully validate the fixed final corpus before the first Elastic write."""
    videos: set[str] = set()
    categories: set[str] = set()
    rows = 0
    for record in iter_infoshotpp_keyframes(root):
        rows += 1
        videos.add(record["video_id"])
        categories.add(record["submit_category"])
    if rows != EXPECTED_INFOSHOTPP_KEYFRAMES:
        raise RuntimeError(
            f"InfoShot++ map rows={rows:,}, expected {EXPECTED_INFOSHOTPP_KEYFRAMES:,}"
        )
    if len(videos) != EXPECTED_INFOSHOTPP_VIDEOS:
        raise RuntimeError(
            f"InfoShot++ videos={len(videos):,}, expected {EXPECTED_INFOSHOTPP_VIDEOS:,}"
        )
    if categories != EXPECTED_INFOSHOTPP_CATEGORIES:
        raise RuntimeError(
            f"InfoShot++ categories={sorted(categories)}, expected "
            f"{sorted(EXPECTED_INFOSHOTPP_CATEGORIES)}"
        )
    result = {"rows": rows, "videos": len(videos), "categories": sorted(categories)}
    print(f"[keyframe_map_infoshotpp] preflight PASS: {result}")
    return result


def iter_jsonl(path: Path, transform: Callable[[dict[str, Any]], dict[str, Any]]) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield transform(json.loads(line))


def drop_first(iterator: Iterator[Any], count: int) -> Iterator[Any]:
    return islice(iterator, count, None)


def chunked(
    iterator: Iterator[dict[str, Any]], batch_size: int, max_bytes: int = 0
) -> Iterator[list[dict[str, Any]]]:
    """Batch by record count and, when `max_bytes` is set, by serialized size.

    OCR/speech/audio records are small and uniform, but one OD frame document
    carries every detection with geometry and colour — a flat 2,000-record batch
    can exceed Elasticsearch's HTTP payload limit, so size closes the batch first.
    """
    batch: list[dict[str, Any]] = []
    batch_bytes = 0
    for item in iterator:
        item_bytes = len(json_bytes(item)) + 128 if max_bytes else 0  # +128 ≈ the action line
        if batch and max_bytes and batch_bytes + item_bytes > max_bytes:
            yield batch
            batch, batch_bytes = [], 0
        batch.append(item)
        batch_bytes += item_bytes
        if len(batch) >= batch_size:
            yield batch
            batch, batch_bytes = [], 0
    if batch:
        yield batch


def index_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}_v1"


def build_jobs(
    prefix: str,
    staging_dir: Path,
    ocr_path: Path | None,
    infoshotpp_map_root: Path | None = None,
    infoshotpp_map_index: str = DEFAULT_INFOSHOTPP_MAP_INDEX,
    od_job: UploadJob | None = None,
) -> list[UploadJob]:
    jobs = [
        UploadJob(
            logical_name="keyframe_map",
            index_name=index_name(prefix, "keyframe_map"),
            id_field="submit_keyframe_id",
            mapping=keyframe_mapping(),
            source_label=str(staging_dir / "keyframe_map.jsonl"),
            open_records=jsonl_source(staging_dir / "keyframe_map.jsonl"),
        ),
        UploadJob(
            logical_name="speech_segments",
            index_name=index_name(prefix, "speech_segments"),
            id_field="segment_id",
            mapping=speech_mapping(),
            source_label=str(staging_dir / "speech_segments_mapped.jsonl"),
            open_records=jsonl_source(staging_dir / "speech_segments_mapped.jsonl"),
        ),
        UploadJob(
            logical_name="audio_windows",
            index_name=index_name(prefix, "audio_windows"),
            id_field="window_id",
            mapping=audio_mapping(),
            source_label=str(staging_dir / "audio_windows_mapped.jsonl"),
            open_records=jsonl_source(staging_dir / "audio_windows_mapped.jsonl"),
        ),
    ]
    if infoshotpp_map_root is not None:
        jobs.append(
            UploadJob(
                logical_name="keyframe_map_infoshotpp",
                index_name=infoshotpp_map_index,
                id_field="submit_keyframe_id",
                mapping=keyframe_mapping(),
                source_label=str(infoshotpp_map_root),
                open_records=infoshotpp_source(infoshotpp_map_root),
            )
        )
    if ocr_path is not None:
        jobs.append(
            UploadJob(
                logical_name="ocr_keyframes",
                index_name=index_name(prefix, "ocr_keyframes"),
                id_field="submit_keyframe_id",
                mapping=ocr_mapping(),
                source_label=str(ocr_path),
                open_records=jsonl_source(ocr_path, normalize_ocr_record),
            )
        )
    if od_job is not None:
        jobs.append(od_job)
    return jobs


def build_od_job(args: argparse.Namespace) -> UploadJob | None:
    """OD frames job reading committed shards straight from R2 (no local staging)."""
    output_prefix = args.od_prefix
    expected_hashes: dict[str, str] = {}
    if args.od_run_summary is not None and args.od_run_summary.exists():
        summary = json.loads(args.od_run_summary.read_text(encoding="utf-8"))
        output_prefix = output_prefix or summary.get("output_prefix")
        expected_hashes = {
            field: summary[field]
            for field in ("config_hash", "manifest_hash", "runtime_hash")
            if summary.get(field)
        }
    if not output_prefix:
        return None
    creds = read_r2_credentials(args.r2_credentials_file)
    r2 = R2Client(creds["endpoint"], args.r2_bucket, creds["access_key"], creds["secret_key"])
    return UploadJob(
        logical_name="od_frames",
        index_name=args.od_index or index_name(args.index_prefix, "od_frames"),
        id_field="document_id",
        mapping=od_frames_mapping(),
        source_label=f"r2://{args.r2_bucket}/{output_prefix.rstrip('/')}",
        open_records=od_source(
            r2,
            output_prefix,
            expected_hashes=expected_hashes,
            max_shards=args.od_max_shards,
            verify_checksum=not args.od_skip_checksum,
        ),
        mapping_patch={"properties": OD_SUBMIT_IDENTITY_PROPS},
    )


def upload_job(
    client: ElasticClient,
    job: UploadJob,
    batch_size: int,
    *,
    resume_existing: bool = False,
    max_bulk_bytes: int = DEFAULT_MAX_BULK_BYTES,
) -> dict[str, Any]:
    status = client.create_index_if_missing(job.index_name, job.mapping)
    if status == "exists" and job.mapping_patch:
        client.json_request("PUT", f"{job.index_name}/_mapping", payload=job.mapping_patch, timeout=60)
    skipped = 0
    if resume_existing and status == "exists":
        client.refresh(job.index_name)
        skipped = client.count(job.index_name)
    uploaded = 0
    took_ms = 0
    started = time.time()
    print(f"[{job.index_name}] index {status}; source={job.source_label}; skip={skipped:,}")
    records = job.open_records(skipped)
    for batch in chunked(records, batch_size, max_bulk_bytes):
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
        "source": job.source_label,
        "uploaded": uploaded,
        "skipped_existing": skipped,
        "remote_count": remote_count,
        "bulk_took_ms": took_ms,
        "elapsed_s": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-file", type=Path, default=Path("API_KEY/elastic_endpoint.txt"))
    parser.add_argument("--api-key-file", type=Path, default=Path("API_KEY/elastic_apikey.txt"))
    parser.add_argument("--staging-dir", type=Path, default=Path("elastic_staging"))
    parser.add_argument("--ocr-path", type=Path, default=Path("OCR/ocr_clean.jsonl"))
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument(
        "--infoshootpp-map-root",
        type=Path,
        default=DEFAULT_INFOSHOTPP_MAP_ROOT,
        help="Root chứa Lxx/Lxx_Vxxx.csv của InfoShot++ final map.",
    )
    parser.add_argument(
        "--infoshootpp-map-index",
        default=DEFAULT_INFOSHOTPP_MAP_INDEX,
        help="Index riêng cho map InfoShot++; không ghi đè map BTC.",
    )
    parser.add_argument("--no-infoshootpp-map", action="store_true")
    parser.add_argument("--index-prefix", default=DEFAULT_INDEX_PREFIX)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[
            "keyframe_map",
            "keyframe_map_infoshotpp",
            "speech_segments",
            "audio_windows",
            "ocr_keyframes",
            "od_frames",
        ],
        help="Upload only selected logical indices.",
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--max-bulk-bytes", type=int, default=DEFAULT_MAX_BULK_BYTES)
    parser.add_argument("--summary-path", type=Path, default=Path("elastic_staging/elastic_upload_summary.json"))

    od = parser.add_argument_group("object detection (đọc shard đã commit trên Cloudflare R2)")
    od.add_argument(
        "--od-prefix",
        help="R2 output prefix, vd Derived/ObjectDetection/aic26-od-v5/<config>/input-<manifest>/runtime-<runtime>",
    )
    od.add_argument(
        "--od-run-summary",
        type=Path,
        help="JSON summary của run OD (chứa output_prefix + *_hash); dùng thay/bổ sung cho --od-prefix",
    )
    od.add_argument("--od-index", help=f"Mặc định {index_name(DEFAULT_INDEX_PREFIX, 'od_frames')}")
    od.add_argument("--od-max-shards", type=int, help="Chỉ import N shard đầu (chạy thử)")
    od.add_argument("--od-skip-checksum", action="store_true", help="Bỏ verify SHA-256 của shard")
    od.add_argument("--r2-bucket", default=DEFAULT_R2_BUCKET)
    od.add_argument(
        "--r2-credentials-file",
        type=Path,
        default=Path("CloudflareR2/cloudflareR2_api.txt"),
        help="File 'Label: value' (Access Key ID / Secret Access Key / Account ID); env R2_* được ưu tiên",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint = read_secret(args.endpoint_file)
    api_key = read_secret(args.api_key_file)
    ocr_path = None if args.no_ocr else args.ocr_path
    infoshotpp_root = None if args.no_infoshootpp_map else args.infoshootpp_map_root
    jobs = build_jobs(
        args.index_prefix,
        args.staging_dir,
        ocr_path,
        infoshotpp_root,
        args.infoshootpp_map_index,
        build_od_job(args),
    )
    if args.only:
        allowed = set(args.only)
        jobs = [job for job in jobs if job.logical_name in allowed]
    if any(job.logical_name == "keyframe_map_infoshotpp" for job in jobs):
        # Intentional second read: this pass proves the whole corpus is valid;
        # upload_job then streams it. No partially-audited index is created.
        audit_infoshotpp_maps(args.infoshootpp_map_root)

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
        upload_job(
            client,
            job,
            args.batch_size,
            resume_existing=args.resume_existing,
            max_bulk_bytes=args.max_bulk_bytes,
        )
        for job in jobs
    ]
    summary = {
        "endpoint_file": str(args.endpoint_file),
        "index_prefix": args.index_prefix,
        "batch_size": args.batch_size,
        "max_bulk_bytes": args.max_bulk_bytes,
        "resume_existing": args.resume_existing,
        "results": results,
    }
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
