#!/usr/bin/env python3
"""Create and fill the **v2** Elasticsearch indices for retrieval profile 2 (InfoShot++).

The v1 uploader (`elastic_upload.py`) and every `*_v1` index it created are left
untouched; this script only ever writes indices whose name ends in `_v2`:

    aic26_keyframe_map_v2      <- elastic_staging_v2/keyframe_map.jsonl
    aic26_speech_segments_v2   <- elastic_staging_v2/speech_segments_mapped.jsonl
    aic26_audio_windows_v2     <- elastic_staging_v2/audio_windows_mapped.jsonl
    aic26_ocr_keyframes_v2     <- elastic_staging_v2/ocr_keyframes_mapped.jsonl

Document shape stays field-for-field compatible with the v1 indices the backend
queries today (the mappings below are literally the v1 mappings plus the
InfoShot++ `frame_id` identity), so only the *identity* changes: every
`submit_keyframe_id` here is `<category>/<video_id>/<n:03d>` where `n` comes from
the final InfoShot++ map CSV.

Two safety properties this uploader has and v1 does not:

* **No count-based resume.** v1 resumed by reading the remote document count and
  skipping that many source rows, which is only sound when the source is
  guaranteed to be the same file in the same order. Here a resume is refused
  unless the staging file's SHA-256 matches the one recorded when the interrupted
  run started (`--resume`); the remote count is never used to decide what to skip.
* **`_v2`-only writes.** Any target index not ending in `_v2` is rejected before a
  single request is sent.

Run `--dry-run` first: it audits the staging files, verifies the corpus
expectations, and checks that the documents this script actually builds fit the
mappings it would create — without contacting Elasticsearch for anything but a
version handshake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The v1 uploader is the reference implementation for the HTTP/bulk plumbing and,
# more importantly, for the index mappings: importing them is what guarantees the
# v2 documents stay compatible with what the backend already queries.
from elastic_upload import (
    ElasticClient,
    audio_mapping,
    build_bulk_payload,
    chunked,
    keyframe_mapping,
    ocr_mapping,
    read_secret,
    speech_mapping,
)

DEFAULT_INDEX_PREFIX = "aic26"
DEFAULT_STAGING_DIR = Path("elastic_staging_v2")
DEFAULT_MAX_BULK_BYTES = 8 * 1024 * 1024
STATE_FILENAME = ".elastic_upload_v2_state.json"
PREFLIGHT_SAMPLE = 200

EXPECTED_VIDEOS = 873
EXPECTED_KEYFRAMES = 1_339_055
EXPECTED_CATEGORIES = [f"L{number:02d}" for number in range(21, 31)]

#: Never writable from this script, whatever the flags say.
PROTECTED_INDICES = frozenset(
    {
        "aic26_keyframe_map_v1",
        "aic26_keyframe_map_infoshotpp_v1",
        "aic26_speech_segments_v1",
        "aic26_audio_windows_v1",
        "aic26_ocr_keyframes_v1",
        "aic26_od_frames_v1",
    }
)


# ---------------------------------------------------------------------------
# mappings — v1 shape + the InfoShot++ frame identity
# ---------------------------------------------------------------------------


def keyword(ignore_above: int = 512) -> dict[str, Any]:
    return {"type": "keyword", "ignore_above": ignore_above}


#: `frame_id` anchors added to the time-span (speech/audio) documents alongside
#: v1's `*_keyframe_*` blocks, so a hit can be traced to the exact InfoShot++ JPEG.
FRAME_ID_PROPS = {
    "frame_id": keyword(),
    "start_frame_id": keyword(),
    "center_frame_id": keyword(),
    "end_frame_id": keyword(),
    "keyframe_name": keyword(),
    "submit_category": keyword(),
}


def _with_properties(mapping: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(mapping))  # deep copy; mappings are plain JSON
    merged["mappings"]["properties"].update(extra)
    return merged


def keyframe_mapping_v2() -> dict[str, Any]:
    """Identical to v1's keyframe map mapping — it already carries `frame_id`."""
    return keyframe_mapping()


def speech_mapping_v2() -> dict[str, Any]:
    return _with_properties(speech_mapping(), FRAME_ID_PROPS)


def audio_mapping_v2() -> dict[str, Any]:
    return _with_properties(audio_mapping(), FRAME_ID_PROPS)


def ocr_mapping_v2() -> dict[str, Any]:
    return _with_properties(
        ocr_mapping(),
        {
            "frame_id": keyword(),
            "keyframe_id": keyword(),
            "frame_idx": {"type": "integer"},
            "pts_time": {"type": "float"},
            "fps": {"type": "float"},
        },
    )


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UploadJobV2:
    logical_name: str
    index_name: str
    id_field: str
    mapping: dict[str, Any]
    source: Path
    #: Identity fields whose presence is proven for every sampled document.
    required_fields: tuple[str, ...] = ()

    def records(self, skip: int = 0) -> Iterator[dict[str, Any]]:
        """Stream the staging file, dropping the first `skip` records unparsed.

        Skipping before `json.loads` matters: a resume can be a million records
        deep, and parsing them only to throw them away would dominate the run.
        """
        with self.source.open(encoding="utf-8") as handle:
            position = 0
            for line in handle:
                if not line.strip():
                    continue
                position += 1
                if position <= skip:
                    continue
                yield json.loads(line)


def index_name_v2(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}_v2"


def build_jobs(prefix: str, staging_dir: Path, overrides: dict[str, str]) -> list[UploadJobV2]:
    specs = (
        (
            "keyframe_map",
            "keyframe_map.jsonl",
            "submit_keyframe_id",
            keyframe_mapping_v2(),
            ("submit_keyframe_id", "frame_id", "video_id", "keyframe_n", "keyframe_name", "submit_category"),
        ),
        (
            "speech_segments",
            "speech_segments_mapped.jsonl",
            "segment_id",
            speech_mapping_v2(),
            ("segment_id", "submit_keyframe_id", "video_id", "keyframe_n", "frame_id"),
        ),
        (
            "audio_windows",
            "audio_windows_mapped.jsonl",
            "window_id",
            audio_mapping_v2(),
            ("window_id", "submit_keyframe_id", "video_id", "keyframe_n", "frame_id"),
        ),
        (
            "ocr_keyframes",
            "ocr_keyframes_mapped.jsonl",
            "submit_keyframe_id",
            ocr_mapping_v2(),
            ("submit_keyframe_id", "frame_id", "video_id", "keyframe_n", "keyframe_name", "submit_category"),
        ),
    )
    return [
        UploadJobV2(
            logical_name=name,
            index_name=overrides.get(name) or index_name_v2(prefix, name),
            id_field=id_field,
            mapping=mapping,
            source=staging_dir / filename,
            required_fields=required,
        )
        for name, filename, id_field, mapping, required in specs
    ]


def assert_writable_index(index: str, *, allow_non_v2: bool) -> None:
    if index in PROTECTED_INDICES:
        raise SystemExit(f"Từ chối ghi vào index v1 đang phục vụ backend: {index}")
    if not index.endswith("_v2") and not allow_non_v2:
        raise SystemExit(
            f"Index {index!r} không kết thúc bằng '_v2'. "
            f"Dùng --allow-non-v2-index nếu thực sự cố ý."
        )


# ---------------------------------------------------------------------------
# source fingerprint + resume state
# ---------------------------------------------------------------------------


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Full SHA-256 plus size/mtime.

    A cheap prefix hash would leave the tail of a regenerated staging file
    unverified, and the tail is exactly where a resume continues — so the whole
    file is hashed. At ~1 GB/s this costs a couple of seconds per artifact and is
    the entire basis on which skipping source rows is allowed to be safe.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def count_records(path: Path) -> int:
    """Line count without parsing; a file not ending in a newline still counts its tail."""
    total = 0
    last = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            total += block.count(b"\n")
            last = block[-1:]
    if last and last != b"\n":
        total += 1
    return total


@dataclass
class UploadState:
    path: Path
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "UploadState":
        if not path.exists():
            return cls(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls(path=path)
        return cls(path=path, jobs=raw.get("jobs") or {})

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"version": 2, "jobs": self.jobs}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def resume_position(self, job: UploadJobV2, fingerprint: dict[str, Any]) -> int:
        """Rows already confirmed uploaded, or 0 when resuming would be unsafe."""
        entry = self.jobs.get(job.logical_name)
        if not entry:
            return 0
        if entry.get("index") != job.index_name:
            print(f"[{job.index_name}] state ghi cho index khác ({entry.get('index')}) → bỏ qua resume")
            return 0
        stored = entry.get("fingerprint") or {}
        if stored.get("sha256") != fingerprint["sha256"] or stored.get("size") != fingerprint["size"]:
            raise SystemExit(
                f"[{job.index_name}] source {job.source} đã thay đổi kể từ lần chạy trước "
                f"(sha256 khác). Resume theo vị trí dòng sẽ sai. Chạy lại không có --resume "
                f"(re-index toàn bộ; _id tất định nên an toàn) hoặc xoá {self.path}."
            )
        return int(entry.get("uploaded") or 0)

    def record(self, job: UploadJobV2, fingerprint: dict[str, Any], uploaded: int, *, completed: bool) -> None:
        self.jobs[job.logical_name] = {
            "index": job.index_name,
            "source": str(job.source),
            "fingerprint": fingerprint,
            "uploaded": uploaded,
            "completed": completed,
            "updated_at_unix": time.time(),
        }
        self.save()


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def mapped_property_names(mapping: dict[str, Any]) -> set[str]:
    return set(mapping["mappings"]["properties"])


def check_document_fits_mapping(
    mapping: dict[str, Any], records: list[dict[str, Any]], id_field: str, required: tuple[str, ...]
) -> dict[str, Any]:
    """Confirm the documents this script sends fit the index it would create.

    `dynamic: false` means Elasticsearch stores an unmapped field but never
    indexes it — a silent, query-time-only failure. Surfacing it here is the whole
    point of the check.
    """
    properties = mapping["mappings"]["properties"]
    unmapped_fields: set[str] = set()
    unmapped_nested: set[str] = set()
    missing_required: dict[str, int] = {}
    missing_id = 0
    for record in records:
        if not record.get(id_field):
            missing_id += 1
        for name in required:
            if record.get(name) in (None, ""):
                missing_required[name] = missing_required.get(name, 0) + 1
        for name, value in record.items():
            spec = properties.get(name)
            if spec is None:
                unmapped_fields.add(name)
                continue
            if spec.get("type") == "nested" and isinstance(value, list):
                sub = spec.get("properties") or {}
                for item in value:
                    if isinstance(item, dict):
                        unmapped_nested.update(f"{name}.{key}" for key in item if key not in sub)
    return {
        "sampled": len(records),
        "id_field": id_field,
        "documents_missing_id": missing_id,
        "documents_missing_required_field": missing_required,
        "fields_not_in_mapping": sorted(unmapped_fields),
        "nested_fields_not_in_mapping": sorted(unmapped_nested),
        "mapped_properties_unused": sorted(
            mapped_property_names(mapping) - {name for record in records for name in record}
        ),
    }


def load_staging_summary(staging_dir: Path) -> dict[str, Any] | None:
    path = staging_dir / "mapping_summary_v2.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def preflight(jobs: list[UploadJobV2], staging_dir: Path, *, allow_unmapped_fields: bool) -> dict[str, Any]:
    report: dict[str, Any] = {"staging_dir": str(staging_dir), "jobs": {}, "problems": []}

    summary = load_staging_summary(staging_dir)
    if summary is None:
        report["problems"].append(
            f"Không tìm thấy {staging_dir / 'mapping_summary_v2.json'} — chạy keyframe_mapping_v2.py trước."
        )
    else:
        corpus = summary.get("corpus") or {}
        report["corpus"] = {
            "videos": corpus.get("videos"),
            "keyframes": corpus.get("keyframes"),
            "categories": corpus.get("categories"),
            "duplicate_submit_keyframe_id": corpus.get("duplicate_submit_keyframe_id"),
            "duplicate_frame_id": corpus.get("duplicate_frame_id"),
            "duplicate_video_frame_idx": corpus.get("duplicate_video_frame_idx"),
            "invalid_frame_idx": corpus.get("invalid_frame_idx"),
        }
        checks = {
            "videos": (corpus.get("videos"), EXPECTED_VIDEOS),
            "keyframes": (corpus.get("keyframes"), EXPECTED_KEYFRAMES),
            "categories": (corpus.get("categories"), EXPECTED_CATEGORIES),
        }
        for name, (actual, expected) in checks.items():
            if actual != expected:
                report["problems"].append(f"corpus.{name}={actual!r}, kỳ vọng {expected!r}")
        for name in (
            "duplicate_submit_keyframe_id",
            "duplicate_frame_id",
            "duplicate_video_frame_idx",
            "invalid_frame_idx",
        ):
            if corpus.get(name):
                report["problems"].append(f"corpus.{name}={corpus.get(name)}, kỳ vọng 0")
        unmapped = summary.get("unmapped") or {}
        report["staging_unmapped"] = {
            "in_corpus_failures": unmapped.get("in_corpus_failures"),
            "out_of_corpus": unmapped.get("out_of_corpus"),
            "by_reason": unmapped.get("by_reason"),
        }
        if unmapped.get("in_corpus_failures"):
            report["problems"].append(
                f"{unmapped['in_corpus_failures']} record thuộc corpus không map được "
                f"(xem {unmapped.get('report')})"
            )
        ocr = summary.get("ocr") or {}
        if ocr:
            coverage = ocr.get("coverage") or {}
            report["ocr"] = {
                "records_out": ocr.get("records_out"),
                "unmapped": ocr.get("unmapped"),
                "expected_missing_categories": coverage.get("expected_missing_categories"),
                "observed_missing_categories": coverage.get("observed_missing_categories"),
                # L26 has no OCR yet. That is a known state of the artifact, so
                # coverage is measured against the corpus minus L26 and the count
                # is deliberately NOT compared to the full 1,339,055 keyframes.
                "missing_categories_are_intentional": coverage.get("missing_categories_are_intentional"),
                "corpus_keyframes_without_ocr": coverage.get("corpus_keyframes_without_ocr"),
            }
            if ocr.get("unmapped"):
                report["problems"].append(f"OCR: {ocr['unmapped']} record không map được keyframe")
            if not coverage.get("missing_categories_are_intentional", True):
                report["problems"].append(
                    f"OCR thiếu category ngoài dự kiến: {coverage.get('observed_missing_categories')}"
                )

    for job in jobs:
        entry: dict[str, Any] = {"index": job.index_name, "source": str(job.source)}
        if not job.source.exists():
            entry["exists"] = False
            report["problems"].append(f"Thiếu staging file {job.source}")
            report["jobs"][job.logical_name] = entry
            continue
        entry["exists"] = True
        entry["records"] = count_records(job.source)
        entry["size_bytes"] = job.source.stat().st_size
        sample: list[dict[str, Any]] = []
        with job.source.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    sample.append(json.loads(line))
                if len(sample) >= PREFLIGHT_SAMPLE:
                    break
        fit = check_document_fits_mapping(job.mapping, sample, job.id_field, job.required_fields)
        entry["mapping_check"] = fit
        entry["sample_document_id"] = sample[0][job.id_field] if sample else None
        if fit["documents_missing_id"]:
            report["problems"].append(
                f"{job.index_name}: {fit['documents_missing_id']} document thiếu {job.id_field}"
            )
        if fit["documents_missing_required_field"]:
            report["problems"].append(
                f"{job.index_name}: thiếu identity field {fit['documents_missing_required_field']}"
            )
        stray = fit["fields_not_in_mapping"] + fit["nested_fields_not_in_mapping"]
        if stray and not allow_unmapped_fields:
            report["problems"].append(
                f"{job.index_name}: field không có trong mapping (sẽ lưu nhưng KHÔNG index): {stray}"
            )
        report["jobs"][job.logical_name] = entry

    if summary and "keyframe_map" in report["jobs"]:
        staged = (summary.get("keyframe_map") or {}).get("records")
        actual = report["jobs"]["keyframe_map"].get("records")
        if staged is not None and actual is not None and staged != actual:
            report["problems"].append(
                f"keyframe_map.jsonl có {actual} dòng nhưng summary ghi {staged}"
            )
    return report


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


def upload_job(
    client: ElasticClient,
    job: UploadJobV2,
    *,
    batch_size: int,
    max_bulk_bytes: int,
    state: UploadState,
    fingerprint: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    status = client.create_index_if_missing(job.index_name, job.mapping)
    skip = state.resume_position(job, fingerprint) if resume else 0
    started = time.time()
    print(
        f"[{job.index_name}] index {status}; source={job.source}; "
        f"skip={skip:,} ({'resume theo sha256 khớp' if skip else 'từ đầu'})"
    )
    uploaded = 0
    took_ms = 0
    batches = 0
    for batch in chunked(job.records(skip), batch_size, max_bulk_bytes):
        result = client.bulk_ndjson(build_bulk_payload(job.index_name, batch, id_field=job.id_field))
        uploaded += len(batch)
        took_ms += int(result.get("took", 0))
        batches += 1
        if batches % 10 == 0:
            # Checkpointing lags the real position by at most 10 batches. Replaying
            # those on a resume is harmless: `_id` is derived from the document's
            # own identity, so a bulk `index` of the same row is an overwrite.
            state.record(job, fingerprint, skip + uploaded, completed=False)
            print(f"[{job.index_name}] source position {skip + uploaded:,}")
    state.record(job, fingerprint, skip + uploaded, completed=True)
    client.refresh(job.index_name)
    remote_count = client.count(job.index_name)
    elapsed = round(time.time() - started, 2)
    print(
        f"[{job.index_name}] uploaded_new={uploaded:,}; skipped={skip:,}; "
        f"remote_count={remote_count:,}; elapsed={elapsed}s"
    )
    return {
        "index": job.index_name,
        "logical_name": job.logical_name,
        "index_status": status,
        "source": str(job.source),
        "source_sha256": fingerprint["sha256"],
        "uploaded": uploaded,
        "skipped_resume": skip,
        "remote_count": remote_count,
        "expected_count": skip + uploaded,
        "count_matches": remote_count == skip + uploaded,
        "bulk_took_ms": took_ms,
        "elapsed_s": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--endpoint-file", type=Path, default=Path("API_KEY/elastic_endpoint.txt"))
    parser.add_argument("--api-key-file", type=Path, default=Path("API_KEY/elastic_apikey.txt"))
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--index-prefix", default=DEFAULT_INDEX_PREFIX)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["keyframe_map", "speech_segments", "audio_windows", "ocr_keyframes"],
        help="Chỉ upload các index v2 được chọn.",
    )
    parser.add_argument("--index-keyframe-map", help="Ghi đè tên index (mặc định aic26_keyframe_map_v2)")
    parser.add_argument("--index-speech-segments", help="Mặc định aic26_speech_segments_v2")
    parser.add_argument("--index-audio-windows", help="Mặc định aic26_audio_windows_v2")
    parser.add_argument("--index-ocr-keyframes", help="Mặc định aic26_ocr_keyframes_v2")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--max-bulk-bytes", type=int, default=DEFAULT_MAX_BULK_BYTES)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ preflight: audit staging + kiểm tra document khớp mapping. Không tạo index, không ghi.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục lần chạy trước — CHỈ khi sha256 của staging file không đổi.",
    )
    parser.add_argument(
        "--allow-unmapped-fields",
        action="store_true",
        help="Cho phép document chứa field không có trong mapping (sẽ không được index).",
    )
    parser.add_argument("--allow-non-v2-index", action="store_true")
    parser.add_argument(
        "--skip-preflight-problems",
        action="store_true",
        help="Vẫn upload dù preflight báo lỗi (không khuyến khích).",
    )
    parser.add_argument(
        "--summary-path", type=Path, default=None, help="Mặc định <staging-dir>/elastic_upload_v2_summary.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "keyframe_map": args.index_keyframe_map,
        "speech_segments": args.index_speech_segments,
        "audio_windows": args.index_audio_windows,
        "ocr_keyframes": args.index_ocr_keyframes,
    }
    jobs = build_jobs(args.index_prefix, args.staging_dir, {k: v for k, v in overrides.items() if v})
    if args.only:
        allowed = set(args.only)
        jobs = [job for job in jobs if job.logical_name in allowed]
    if not jobs:
        raise SystemExit("Không còn job nào sau khi lọc --only.")
    for job in jobs:
        assert_writable_index(job.index_name, allow_non_v2=args.allow_non_v2_index)

    print("=== PREFLIGHT ===")
    report = preflight(jobs, args.staging_dir, allow_unmapped_fields=args.allow_unmapped_fields)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["problems"] and not args.skip_preflight_problems:
        raise SystemExit(
            f"Preflight phát hiện {len(report['problems'])} vấn đề — không upload. "
            f"Sửa staging hoặc dùng --skip-preflight-problems nếu đã hiểu rõ."
        )

    summary_path = args.summary_path or (args.staging_dir / "elastic_upload_v2_summary.json")
    if args.dry_run:
        print("\n=== DRY RUN: không tạo index, không upload ===")
        for job in jobs:
            print(f"  {job.logical_name:<16} -> {job.index_name:<28} id={job.id_field}  src={job.source}")
        summary_path.write_text(
            json.dumps({"dry_run": True, "preflight": report}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nPreflight report: {summary_path}")
        return

    client = ElasticClient(read_secret(args.endpoint_file), read_secret(args.api_key_file))
    info = client.json_request("GET", "/", expected=(200,), timeout=60)
    print(
        "\nConnected to Elasticsearch",
        json.dumps(
            {
                "cluster_name": info.get("cluster_name"),
                "version": (info.get("version") or {}).get("number"),
            },
            ensure_ascii=False,
        ),
    )

    state = UploadState.load(args.staging_dir / STATE_FILENAME)
    results = []
    for job in jobs:
        fingerprint = file_fingerprint(job.source)
        results.append(
            upload_job(
                client,
                job,
                batch_size=args.batch_size,
                max_bulk_bytes=args.max_bulk_bytes,
                state=state,
                fingerprint=fingerprint,
                resume=args.resume,
            )
        )

    summary = {
        "schema_version": "v2",
        "retrieval_profile": "infoshotpp",
        "endpoint_file": str(args.endpoint_file),
        "index_prefix": args.index_prefix,
        "staging_dir": str(args.staging_dir),
        "batch_size": args.batch_size,
        "max_bulk_bytes": args.max_bulk_bytes,
        "resume": args.resume,
        "preflight": report,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    mismatched = [r["index"] for r in results if not r["count_matches"]]
    if mismatched:
        raise SystemExit(f"remote_count không khớp số document đã gửi: {mismatched}")


if __name__ == "__main__":
    main()
