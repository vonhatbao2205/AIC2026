#!/usr/bin/env python3
"""Append the batch-2 OCR and speech metadata to the InfoShot++ indices in Elastic.

Retrieval profile 2 answers OCR from ``aic26_ocr_keyframes_v2`` (``IDX_OCR_2``) and
speech from ``aic26_speech_segments_v2`` (``IDX_SPEECH_2``). Both hold L21–L30,
written by ``elastic_upload_v2.py`` from ``keyframe_mapping_v2.py`` staging. This
script adds batch 2 (M01–M10 news, N001–N100 traffic cameras, S01 cycling) to the
same two indices, in the shape of the L documents, so the backend reads both with
no configuration change:

    OCR     873,328 documents  <- final/{M,N,S}/ocr_clean.parquet  (AIC2026_Batch2_OCR_HANDOFF.md)
    speech   21,782 documents  <- speech_out_batch2.zip            (AIC2026_SPEECH_BATCH2_HANDOFF.md)

Every source file is pinned by SHA-256, and identity comes from the pinned keyframe
registries (``batch2_keyframes.py``), the rows the keyframe map and the PE vectors
already hold:

* an OCR row joins its registry row exactly on ``frame_id`` and must agree with it
  on ordinal, decode index, time and R2 key. The 1,472 black frames OCR skipped are
  not indexed: the L index holds ``status=ok`` frames only;
* a speech segment is anchored on the keyframes nearest its start, centre and end,
  as ``keyframe_mapping_v2.map_time_span`` does for L, but the category comes from
  the registry. The L helpers take it from ``video_id.split("_")``, which would file
  ``S01-V001`` under a category of its own. N has no audio track, so no speech.

Where batch 2 differs from L, on purpose:

* OCR post-process v3 keeps overlay text (M ticker, N camera banner, S01 race HUD,
  logos) out of ``text_clean``. The backend still reaches it through ``text_nfc``
  and ``boxes.text``, as it does for L, and it is also indexed in its own fields
  (``OCR_EXTRA_PROPERTIES``), which this script adds to the mapping. Adding fields
  is the only mapping change it makes: no existing field is altered and the L
  documents are not touched.
* The parquet does not carry the OCR ``ts``, and nothing reads it, so batch-2 OCR
  documents have none.
* Twelve news videos are narrated in English (manifest ``mean_avg_word_score < 0.3``).
  Their ``avg_word_score`` grades a Vietnamese aligner on English audio, not the
  transcript (speech handoff §6.1), so they get ``lang="en"`` and the
  ``confidence_bucket`` "missing" instead of "low"; the raw score is kept.

Safety, as in ``elastic_upload_batch2_keyframe_map.py``: the indices must already
exist, the L document count is measured before and after and must not move, batch-2
ids cannot collide with an L id, ``_id`` is deterministic so a replayed bulk
overwrites, and resume is per video with state bound to the source SHA-256s, the
index and the endpoint.

Run ``--dry-run`` first (local only), then ``--check-remote`` (read-only), then the
same command without either flag to upload.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import time
import zipfile
from array import array
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from batch2_keyframes import DEFAULT_REGISTRY_ROOT, MIN_QUALITY, PROFILES, KeyframeRow, load_registries, sha256_file
from elastic_upload import (
    EXPECTED_INFOSHOTPP_CATEGORIES,
    ElasticClient,
    build_bulk_payload,
    json_bytes,
    keyword,
    read_secret,
    text_with_keyword,
)
from elastic_upload_batch2_keyframe_map import (
    atomic_write_json,
    category_counts,
    count_where,
    endpoint_identity,
    terms_query,
    utc_now,
)
from elastic_upload_v2 import assert_writable_index, index_name_v2, ocr_mapping_v2, speech_mapping_v2
from keyframe_mapping import confidence_bucket, make_time_span_id, segment_role
from keyframe_mapping_v2 import Keyframe, VideoKeyframes

STATE_SCHEMA_VERSION = 1
DEFAULT_OCR_ROOT = Path("elastic_staging_batch2/source/ocr")
DEFAULT_SPEECH_ZIP = Path("elastic_staging_batch2/source/speech/speech_out_batch2.zip")
DEFAULT_OCR_INDEX = index_name_v2("aic26", "ocr_keyframes")
DEFAULT_SPEECH_INDEX = index_name_v2("aic26", "speech_segments")
DEFAULT_MAX_BULK_BYTES = 8 * 1024 * 1024

#: What each index held before batch 2: the verified `elastic_upload_v2.py` run.
BASE_DOCUMENTS = {"ocr": 779_995, "speech": 18_611}

# ---------------------------------------------------------------------------
# OCR source: hf://buckets/Baonenha1/DATA-AIC-Keyframe/ocr/batch2/hunyuanocr-1.5-spotting-v3.7-r2/final
# ---------------------------------------------------------------------------

OCR_PIPELINE_SIGNATURE = "ae38408251ce3f9b"
OCR_POSTPROCESS_VERSION = "aic-ocr-clean-batch2-region-v3"
BOX_REGIONS = frozenset({"scene", "ticker", "logo_clock", "banner", "hud", "unknown"})
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class OcrSource:
    profile: str
    sha256: str
    ok: int
    skipped: int

    @property
    def relpath(self) -> str:
        return f"final/{self.profile}/ocr_clean.parquet"

    @property
    def success_relpath(self) -> str:
        return f"final/{self.profile}/_SUCCESS.json"


OCR_SOURCES: dict[str, OcrSource] = {
    source.profile: source
    for source in (
        OcrSource("M", "cebf10c3599f9e802462db068848744bdffa7250be225e09399b47603180f426", ok=207_780, skipped=1_331),
        OcrSource("N", "f3185beafa54d9b5c8556348fbfc7322ebe824bee10eeb0e936fa7444302c93d", ok=44_572, skipped=0),
        OcrSource("S", "79e532317893dac1ebade9461c9da0d7a9216aa34da5fd4b8986e02dfb76f239", ok=620_976, skipped=141),
    )
}
OCR_DOCUMENTS = sum(source.ok for source in OCR_SOURCES.values())  # 873,328

OCR_COLUMNS = (
    "frame_id",
    "id",
    "video_id",
    "category",
    "profile",
    "n",
    "keyframe_name",
    "submit_keyframe_id",
    "frame_idx",
    "pts_time",
    "fps",
    "r2_key",
    "keyframe_pipeline_signature",
    "pipeline_signature",
    "postprocess_version",
    "status",
    "error",
    "text_clean",
    "text_clean_fold",
    "text_clean_hash",
    "text_ticker",
    "text_ticker_fold",
    "text_banner",
    "banner_camera",
    "banner_date",
    "text_hud",
    "race_stage",
    "race_time",
    "clock",
    "hour",
    "text_nfc",
    "boxes_json",
)

#: Batch-2 OCR fields the L mapping lacks (OCR handoff §11.2). The index is
#: `dynamic: false`, which would store them without indexing them, so they are
#: added to the mapping before the first bulk.
OCR_EXTRA_PROPERTIES: dict[str, Any] = {
    "profile": keyword(),
    "text_ticker": text_with_keyword(),
    "text_ticker_fold": text_with_keyword(),
    "text_banner": text_with_keyword(),
    "banner_camera": text_with_keyword(),
    "banner_date": {"type": "date", "format": "yyyy-MM-dd"},
    "text_hud": text_with_keyword(),
    "race_stage": {"type": "integer"},
    "race_time": keyword(),
    "text_clean_hash": keyword(),
    "pipeline_signature": keyword(),
    "keyframe_pipeline_signature": keyword(),
    "postprocess_version": keyword(),
    "boxes": {"type": "nested", "properties": {"region": keyword()}},
}

# ---------------------------------------------------------------------------
# speech source: https://huggingface.co/buckets/Baonenha1/aic26-media/resolve/Speech/speech_out_batch2.zip
# ---------------------------------------------------------------------------

SPEECH_ZIP_SHA256 = "2c8255168871a94cf7353241a16358fd7f8063904189d0c5422abb2aef0df2c0"
SPEECH_MEMBER_DIR = "speech_out_batch2"
SPEECH_MANIFEST = f"{SPEECH_MEMBER_DIR}/speech_manifest_batch2.csv"
SPEECH_VIDEOS = 316
SPEECH_DOCUMENTS = 21_782
ENGLISH_SCORE_MAX = 0.3
#: English-narrated news videos, speech handoff §6.1; the manifest must name exactly these.
ENGLISH_VIDEOS = frozenset(
    {"M02_V001"}
    | {f"M04_V{number:03d}" for number in (16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 28)}
)
SPEECH_EXTRA_PROPERTIES: dict[str, Any] = {"lang": keyword()}


# ---------------------------------------------------------------------------
# OCR documents
# ---------------------------------------------------------------------------


def _optional(value: Any) -> Any:
    """Parquet writes a missing string as ""; the batch-2-only fields store null instead."""
    return None if value == "" else value


def build_ocr_document(row: dict[str, Any], keyframe: KeyframeRow) -> dict[str, Any]:
    """The L document (`keyframe_mapping_v2.write_ocr_keyframes`) plus the batch-2 fields."""
    return {
        "ocr_id": row["id"],
        "submit_keyframe_id": keyframe.submit_keyframe_id,
        "frame_id": keyframe.frame_id,
        "keyframe_id": keyframe.keyframe_id,
        "video_id": keyframe.video_id,
        "category": keyframe.category,
        "submit_category": keyframe.category,
        # The record's own image_path is Colab staging; the R2 key locates the JPEG.
        "image_path": keyframe.r2_key,
        "status": row["status"],
        "error": row["error"] or "",
        "keyframe_n": keyframe.n,
        "keyframe_name": keyframe.keyframe_name,
        "frame_idx": keyframe.frame_idx,
        "pts_time": keyframe.pts_time,
        "fps": keyframe.fps,
        "text_clean": row["text_clean"] or "",
        "text_clean_fold": row["text_clean_fold"] or "",
        "text_nfc": row["text_nfc"] or "",
        "clock": row["clock"] or "",
        "hour": row["hour"],
        "boxes": json.loads(row["boxes_json"] or "[]"),
        "profile": keyframe.profile,
        "text_ticker": _optional(row["text_ticker"]),
        "text_ticker_fold": _optional(row["text_ticker_fold"]),
        "text_banner": _optional(row["text_banner"]),
        "banner_camera": _optional(row["banner_camera"]),
        "banner_date": _optional(row["banner_date"]),
        "text_hud": _optional(row["text_hud"]),
        "race_stage": row["race_stage"],
        "race_time": _optional(row["race_time"]),
        "text_clean_hash": _optional(row["text_clean_hash"]),
        "pipeline_signature": row["pipeline_signature"],
        "keyframe_pipeline_signature": row["keyframe_pipeline_signature"],
        "postprocess_version": row["postprocess_version"],
    }


def _check_boxes(boxes_json: str | None, where: str) -> None:
    boxes = json.loads(boxes_json or "[]")
    if not isinstance(boxes, list):
        raise ValueError(f"{where}: boxes_json is not a list")
    for box in boxes:
        if not isinstance(box, dict) or set(box) - {"text", "box", "region"} or not isinstance(box.get("text"), str):
            raise ValueError(f"{where}: malformed box {box!r}")
        if box.get("region") not in BOX_REGIONS:
            raise ValueError(f"{where}: box region {box.get('region')!r}")
        coords = box.get("box")
        if coords is not None and not (
            isinstance(coords, list) and len(coords) == 4 and all(type(value) is int for value in coords)
        ):
            raise ValueError(f"{where}: box coordinates {coords!r}")


def check_ocr_row(row: dict[str, Any], keyframe: KeyframeRow | None, profile: str) -> None:
    """Prove one OCR row describes exactly the registry JPEG it names."""
    where = f"OCR {profile} {row.get('frame_id')!r}"
    if keyframe is None or keyframe.profile != profile:
        raise ValueError(f"{where}: frame_id is not a keyframe of registry {profile}")
    expected = {
        "id": f"{keyframe.category}/{keyframe.video_id}/f{keyframe.frame_idx:08d}",
        "video_id": keyframe.video_id,
        "category": keyframe.category,
        "profile": keyframe.profile,
        "n": keyframe.n,
        "keyframe_name": f"{keyframe.keyframe_name}.jpg",
        "submit_keyframe_id": keyframe.submit_keyframe_id,
        "frame_idx": keyframe.frame_idx,
        "pts_time": keyframe.pts_time,
        "fps": keyframe.fps,
        "r2_key": keyframe.r2_key,
        "keyframe_pipeline_signature": PROFILES[keyframe.profile].pipeline_signature,
        "pipeline_signature": OCR_PIPELINE_SIGNATURE,
    }
    for name, wanted in expected.items():
        if row[name] != wanted:
            raise ValueError(f"{where}: {name}={row[name]!r}, registry says {wanted!r}")
    if row["status"] == "skipped":
        if keyframe.quality >= MIN_QUALITY:
            raise ValueError(f"{where}: skipped as black, but registry quality is {keyframe.quality}")
        return
    if row["status"] != "ok":
        raise ValueError(f"{where}: status {row['status']!r}")
    # Post-processing never ran on a skipped frame, so only an OCR'd row names its version.
    if row["postprocess_version"] != OCR_POSTPROCESS_VERSION:
        raise ValueError(f"{where}: postprocess_version={row['postprocess_version']!r}")
    if row["hour"] is not None and not 0 <= row["hour"] <= 23:
        raise ValueError(f"{where}: hour {row['hour']!r}")
    if row["race_stage"] is not None and not 1 <= row["race_stage"] <= 30:
        raise ValueError(f"{where}: race_stage {row['race_stage']!r}")
    if row["banner_date"] and not _ISO_DATE.fullmatch(row["banner_date"]):
        raise ValueError(f"{where}: banner_date {row['banner_date']!r}")
    _check_boxes(row["boxes_json"], where)


def iter_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing pyarrow. Install it with: uv pip install pyarrow") from exc

    parquet = pq.ParquetFile(path)
    missing = set(OCR_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for batch in parquet.iter_batches(batch_size=20_000, columns=list(OCR_COLUMNS)):
        columns = batch.to_pydict()
        for position in range(batch.num_rows):
            yield {name: columns[name][position] for name in OCR_COLUMNS}


def iter_ocr_documents(
    rows: Iterable[dict[str, Any]], registry: dict[str, KeyframeRow], source: OcrSource
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(video_id, document)`` per OCR'd frame; raise unless the rows cover the registry exactly."""
    seen: set[str] = set()
    status: Counter[str] = Counter()
    for row in rows:
        frame_id = row["frame_id"]
        if frame_id in seen:
            raise ValueError(f"OCR {source.profile}: duplicate frame_id {frame_id}")
        seen.add(frame_id)
        keyframe = registry.get(frame_id)
        check_ocr_row(row, keyframe, source.profile)
        status[row["status"]] += 1
        if row["status"] == "ok":
            yield keyframe.video_id, build_ocr_document(row, keyframe)
    expected_frames = sum(1 for row in registry.values() if row.profile == source.profile)
    if len(seen) != expected_frames:
        raise ValueError(f"OCR {source.profile}: {len(seen):,} rows, registry holds {expected_frames:,} keyframes")
    if status["ok"] != source.ok or status["skipped"] != source.skipped:
        raise ValueError(
            f"OCR {source.profile}: ok/skipped {status['ok']:,}/{status['skipped']:,}, "
            f"expected {source.ok:,}/{source.skipped:,}"
        )


def verify_ocr_files(root: Path) -> dict[str, str]:
    """Check the pinned parquets and their _SUCCESS markers; return the SHA-256 per profile."""
    marker = json.loads((root / "final/_SUCCESS.json").read_text(encoding="utf-8"))
    if marker.get("pipeline_signature") != OCR_PIPELINE_SIGNATURE:
        raise SystemExit(f"{root}/final/_SUCCESS.json: pipeline_signature {marker.get('pipeline_signature')!r}")
    digests: dict[str, str] = {}
    for name, source in OCR_SOURCES.items():
        success = json.loads((root / source.success_relpath).read_text(encoding="utf-8"))
        wanted = {
            "pipeline_signature": OCR_PIPELINE_SIGNATURE,
            "keyframe_signature": PROFILES[name].pipeline_signature,
            "postprocess_version": OCR_POSTPROCESS_VERSION,
            "expected": PROFILES[name].rows,
            "ok": source.ok,
            "skipped_black": source.skipped,
            "permanent_error": 0,
        }
        for key, value in wanted.items():
            if success.get(key) != value:
                raise SystemExit(f"{root / source.success_relpath}: {key}={success.get(key)!r}, expected {value!r}")
        path = root / source.relpath
        digest = sha256_file(path)
        if digest != source.sha256:
            raise SystemExit(
                f"{path}: SHA-256 {digest[:12]}… differs from the audited snapshot {source.sha256[:12]}…; "
                "the OCR final/ changed — re-audit before uploading."
            )
        digests[name] = digest
    return digests


# ---------------------------------------------------------------------------
# speech documents
# ---------------------------------------------------------------------------


def video_keyframes(rows: Sequence[KeyframeRow]) -> VideoKeyframes:
    """One registry video as the v2 mapper's time index (rows ordered by n, n = position + 1)."""
    return VideoKeyframes(
        rows[0].video_id,
        rows[0].category,
        array("i", (row.frame_idx for row in rows)),
        array("d", (row.pts_time for row in rows)),
        array("d", (row.fps for row in rows)),
    )


def anchor_fields(video: VideoKeyframes, label: str, keyframe: Keyframe) -> dict[str, Any]:
    """`keyframe_mapping_v2.keyframe_fields`, with the category the registry gives."""
    keyframe_id = f"{video.video_id}/{keyframe.n:03d}"
    return {
        f"{label}_keyframe_id": keyframe_id,
        f"{label}_frame_id": f"{video.video_id}@f{keyframe.frame_idx:08d}",
        f"{label}_submit_keyframe_id": f"{video.category}/{keyframe_id}",
        f"{label}_keyframe_n": keyframe.n,
        f"{label}_keyframe_pts_time": keyframe.pts_time,
        f"{label}_keyframe_frame_idx": keyframe.frame_idx,
    }


def map_time_span(video: VideoKeyframes, start: float, end: float) -> dict[str, Any]:
    """`keyframe_mapping_v2.map_time_span`, with `anchor_fields` for the anchors."""
    center = (start + end) / 2.0
    mapped: dict[str, Any] = {
        "duration": round(end - start, 6),
        "center_time": round(center, 6),
    }
    for label, time_s in (("start", start), ("center", center), ("end", end)):
        mapped.update(anchor_fields(video, label, video.keyframe(video.nearest_position(time_s))))
    mapped["keyframe_id"] = mapped["center_keyframe_id"]
    mapped["frame_id"] = mapped["center_frame_id"]
    mapped["submit_keyframe_id"] = mapped["center_submit_keyframe_id"]
    mapped["keyframe_n"] = mapped["center_keyframe_n"]
    mapped["keyframe_name"] = f"{mapped['center_keyframe_n']:03d}"
    mapped["keyframe_pts_time"] = mapped["center_keyframe_pts_time"]
    mapped["keyframe_frame_idx"] = mapped["center_keyframe_frame_idx"]
    mapped["submit_category"] = video.category
    return mapped


def build_speech_documents(
    video: VideoKeyframes, member: str, segments: list[dict[str, Any]], lang: str
) -> list[dict[str, Any]]:
    """The L documents (`keyframe_mapping_v2.write_speech_segments`) of one video, plus `lang`."""
    documents: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        where = f"{member}[{index}]"
        if segment.get("video_id") != video.video_id:
            raise ValueError(f"{where}: video_id {segment.get('video_id')!r}, file is {video.video_id}")
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{where}: invalid time span") from exc
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            raise ValueError(f"{where}: invalid time span {start}..{end}")
        score = segment.get("avg_word_score")
        document = {
            "segment_id": make_time_span_id(video.video_id, start, end, f"s{index:06d}"),
            "video_id": video.video_id,
            "start": start,
            "end": end,
            "text": segment.get("text", ""),
            "avg_word_score": score,
            # An English segment's score grades a Vietnamese aligner, not the transcript.
            "confidence_bucket": confidence_bucket(score) if lang == "vi" else "missing",
            "segment_role": segment_role(start, end),
            "word_count": len(segment.get("words") or []),
            "avg_logprob": segment.get("avg_logprob"),
            "no_speech_prob": segment.get("no_speech_prob"),
            "source_segment_idx": index,
            "source_speech_file": member,
            "lang": lang,
        }
        document.update(map_time_span(video, start, end))
        documents.append(document)
    return documents


@dataclass(frozen=True)
class SpeechVideo:
    video_id: str
    member: str
    segments: int
    lang: str


def load_speech_manifest(archive: zipfile.ZipFile) -> dict[str, SpeechVideo]:
    with archive.open(SPEECH_MANIFEST) as handle:
        rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))
    videos: dict[str, SpeechVideo] = {}
    for row in rows:
        video_id = row["video_id"]
        if row["status"] != "ok" or video_id in videos:
            raise ValueError(f"{SPEECH_MANIFEST}: {video_id} status={row['status']!r} or listed twice")
        lang = "en" if float(row["mean_avg_word_score"]) < ENGLISH_SCORE_MAX else "vi"
        videos[video_id] = SpeechVideo(
            video_id, f"{SPEECH_MEMBER_DIR}/{video_id}.speech.json", int(row["n_segments"]), lang
        )
    english = {video.video_id for video in videos.values() if video.lang == "en"}
    if english != ENGLISH_VIDEOS:
        raise ValueError(f"English videos {sorted(english)} differ from the handoff's {sorted(ENGLISH_VIDEOS)}")
    members = {name for name in archive.namelist() if name.endswith(".speech.json")}
    if members != {video.member for video in videos.values()} or len(videos) != SPEECH_VIDEOS:
        raise ValueError(f"{len(members)} transcripts in the zip, {len(videos)} manifest rows, expected {SPEECH_VIDEOS}")
    return videos


def iter_speech_documents(
    archive: zipfile.ZipFile, manifest: dict[str, SpeechVideo], videos: dict[str, VideoKeyframes]
) -> Iterator[tuple[str, dict[str, Any]]]:
    for video_id in sorted(manifest):
        entry = manifest[video_id]
        video = videos.get(video_id)
        if video is None:
            raise ValueError(f"{entry.member}: {video_id} is not a video of the keyframe registry")
        segments = json.loads(archive.read(entry.member))
        if len(segments) != entry.segments:
            raise ValueError(f"{entry.member}: {len(segments)} segments, manifest says {entry.segments}")
        for document in build_speech_documents(video, entry.member, segments, entry.lang):
            yield video_id, document


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


@dataclass
class Job:
    name: str
    index: str
    id_field: str
    base_properties: dict[str, Any]
    extra_properties: dict[str, Any]
    source_sha256: dict[str, str]
    #: A fresh, fully validated pass over the source: `(video_id, document)`.
    documents: Callable[[], Iterator[tuple[str, dict[str, Any]]]]
    total: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_group: dict[str, int] = field(default_factory=dict)
    samples: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def categories(self) -> list[str]:
        return sorted(self.by_category)


def mapping_problems(properties: dict[str, Any], documents: Iterable[dict[str, Any]]) -> set[str]:
    """Fields a document carries that the mapping would store without indexing."""
    stray: set[str] = set()
    boxes = (properties.get("boxes") or {}).get("properties") or {}
    for document in documents:
        stray.update(name for name in document if name not in properties)
        for box in document.get("boxes") or []:
            stray.update(f"boxes.{name}" for name in box if name not in boxes)
    return stray


def merged_properties(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for name, spec in extra.items():
        if name in merged and "properties" in spec:
            merged[name].setdefault("properties", {}).update(spec["properties"])
        else:
            merged[name] = spec
    return merged


def audit_job(job: Job, group_of: Callable[[dict[str, Any]], str]) -> None:
    """One full pass: validate every source row, count, and keep samples to compare after upload."""
    properties = merged_properties(job.base_properties, job.extra_properties)
    by_category: Counter[str] = Counter()
    by_group: Counter[str] = Counter()
    ids: set[str] = set()
    stray: set[str] = set()
    last: dict[str, dict[str, Any]] = {}
    first: dict[str, dict[str, Any]] = {}
    for _, document in job.documents():
        doc_id = document[job.id_field]
        if doc_id in ids:
            raise ValueError(f"{job.name}: duplicate _id {doc_id}")
        ids.add(doc_id)
        group = group_of(document)
        first.setdefault(group, document)
        last[group] = document
        by_category[document["submit_category"]] += 1
        by_group[group] += 1
        stray |= mapping_problems(properties, [document])
    if stray:
        raise ValueError(f"{job.name}: fields outside the mapping (stored, never indexed): {sorted(stray)}")
    if set(by_category) & EXPECTED_INFOSHOTPP_CATEGORIES:
        raise ValueError(f"{job.name}: a batch-2 category collides with L21–L30")
    job.total = len(ids)
    job.by_category = dict(sorted(by_category.items()))
    job.by_group = dict(sorted(by_group.items()))
    job.samples = {
        document[job.id_field]: document
        for group in sorted(first)
        for document in (first[group], last[group])
    }


def build_ocr_job(args: argparse.Namespace, rows_by_profile: dict[str, list[KeyframeRow]]) -> Job:
    digests = verify_ocr_files(args.ocr_root)
    registry = {row.frame_id: row for rows in rows_by_profile.values() for row in rows}

    def documents() -> Iterator[tuple[str, dict[str, Any]]]:
        for name, source in OCR_SOURCES.items():
            yield from iter_ocr_documents(iter_parquet_rows(args.ocr_root / source.relpath), registry, source)

    job = Job(
        name="ocr",
        index=args.ocr_index,
        id_field="submit_keyframe_id",
        base_properties=ocr_mapping_v2()["mappings"]["properties"],
        extra_properties=OCR_EXTRA_PROPERTIES,
        source_sha256=digests,
        documents=documents,
    )
    audit_job(job, lambda document: document["profile"])
    if job.total != OCR_DOCUMENTS:
        raise ValueError(f"OCR: {job.total:,} documents, expected {OCR_DOCUMENTS:,}")
    return job


def build_speech_job(args: argparse.Namespace, rows_by_profile: dict[str, list[KeyframeRow]]) -> Job:
    digest = sha256_file(args.speech_zip)
    if digest != SPEECH_ZIP_SHA256:
        raise SystemExit(
            f"{args.speech_zip}: SHA-256 {digest[:12]}… differs from the handoff's {SPEECH_ZIP_SHA256[:12]}…"
        )
    videos: dict[str, VideoKeyframes] = {}
    for rows in rows_by_profile.values():
        start = 0
        for end in range(1, len(rows) + 1):
            if end == len(rows) or rows[end].video_id != rows[start].video_id:
                videos[rows[start].video_id] = video_keyframes(rows[start:end])
                start = end

    def documents() -> Iterator[tuple[str, dict[str, Any]]]:
        with zipfile.ZipFile(args.speech_zip) as archive:
            yield from iter_speech_documents(archive, load_speech_manifest(archive), videos)

    job = Job(
        name="speech",
        index=args.speech_index,
        id_field="segment_id",
        base_properties=speech_mapping_v2()["mappings"]["properties"],
        extra_properties=SPEECH_EXTRA_PROPERTIES,
        source_sha256={"speech_out_batch2.zip": digest},
        documents=documents,
    )
    audit_job(job, lambda document: document["lang"])
    if job.total != SPEECH_DOCUMENTS:
        raise ValueError(f"speech: {job.total:,} documents, expected {SPEECH_DOCUMENTS:,}")
    return job


# ---------------------------------------------------------------------------
# bulks
# ---------------------------------------------------------------------------


def plan_bulks(
    items: Iterable[tuple[str, dict[str, Any]]], batch_size: int, max_bytes: int
) -> Iterator[tuple[list[dict[str, Any]], list[str]]]:
    """Yield ``(documents, videos finished by this bulk)``; items must be grouped by video.

    A video is reported only with the bulk that carries its last document, so a
    checkpoint written after that bulk succeeds never claims a video whose tail
    has not been sent. A bulk closes at `batch_size` documents or `max_bytes`.
    """
    batch: list[dict[str, Any]] = []
    finished: list[str] = []
    seen: set[str] = set()
    current: str | None = None
    size = 0
    for video_id, document in items:
        if video_id != current:
            if video_id in seen:
                raise ValueError(f"Documents of {video_id} are not contiguous")
            if current is not None:
                finished.append(current)
            seen.add(video_id)
            current = video_id
        document_bytes = len(json_bytes(document)) + 128  # +128 ≈ the action line
        if batch and (len(batch) >= batch_size or size + document_bytes > max_bytes):
            yield batch, finished
            batch, finished, size = [], [], 0
        batch.append(document)
        size += document_bytes
    if current is not None:
        finished.append(current)
    if batch:
        yield batch, finished


# ---------------------------------------------------------------------------
# remote
# ---------------------------------------------------------------------------


def flatten_types(properties: dict[str, Any], prefix: str = "") -> dict[str, str | None]:
    types: dict[str, str | None] = {}
    for name, spec in properties.items():
        path = f"{prefix}{name}"
        types[path] = spec.get("type", "object" if "properties" in spec else None)
        types.update(flatten_types(spec.get("properties") or {}, f"{path}."))
    return types


def remote_properties(client: ElasticClient, index: str) -> dict[str, Any]:
    result = client.json_request("GET", f"{index}/_mapping", timeout=60)
    return (next(iter(result.values())).get("mappings") or {}).get("properties") or {}


def mapping_plan(client: ElasticClient, job: Job) -> list[str]:
    """Check the live mapping; return the extra field paths it still lacks."""
    actual = flatten_types(remote_properties(client, job.index))
    for path, wanted in flatten_types(job.base_properties).items():
        if actual.get(path) != wanted:
            raise SystemExit(f"{job.index}: field {path} is {actual.get(path)!r}, expected {wanted!r}")
    missing = []
    for path, wanted in flatten_types(job.extra_properties).items():
        if path not in actual:
            missing.append(path)
        elif actual[path] != wanted:
            raise SystemExit(f"{job.index}: field {path} already exists as {actual[path]!r}, expected {wanted!r}")
    return missing


def remote_counts(client: ElasticClient, job: Job) -> dict[str, int]:
    total = client.count(job.index)
    base = count_where(client, job.index, terms_query(EXPECTED_INFOSHOTPP_CATEGORIES))
    batch2 = count_where(client, job.index, terms_query(job.categories))
    return {"total": total, "base_l21_l30": base, "batch2": batch2, "other": total - base - batch2}


def preflight_remote(client: ElasticClient, job: Job) -> tuple[dict[str, int], list[str]]:
    if not client.index_exists(job.index):
        raise SystemExit(f"Index {job.index} does not exist; it is the L21–L30 index this script extends.")
    missing = mapping_plan(client, job)
    client.refresh(job.index)
    counts = remote_counts(client, job)
    print(f"[{job.index}] remote before upload: {json.dumps(counts)}; mapping lacks {missing or 'nothing'}", flush=True)
    if counts["base_l21_l30"] != BASE_DOCUMENTS[job.name]:
        raise SystemExit(
            f"{job.index} holds {counts['base_l21_l30']:,} L21–L30 documents, expected "
            f"{BASE_DOCUMENTS[job.name]:,}. Refusing to extend an index that is not the verified L upload."
        )
    if counts["other"]:
        raise SystemExit(f"{job.index} holds {counts['other']:,} documents outside L21–L30 and batch 2; refusing.")
    return counts, missing


def extend_mapping(client: ElasticClient, job: Job, missing: list[str]) -> None:
    """Add the batch-2 fields. Only new fields are sent, so nothing already mapped can change."""
    if not missing:
        return
    new = {path.split(".", 1)[0] for path in missing}
    payload = {"properties": {name: job.extra_properties[name] for name in sorted(new)}}
    client.json_request("PUT", f"{job.index}/_mapping", payload=payload, timeout=120)
    still = mapping_plan(client, job)
    if still:
        raise SystemExit(f"{job.index}: mapping still lacks {still} after the update")
    print(f"[{job.index}] mapping extended with {missing}", flush=True)


def smoke_queries(job: Job) -> list[tuple[str, dict[str, Any], int]]:
    """`(label, query, expected count)`: every document is indexed on the fields the backend reads.

    The counts on `profile`/`lang` span every batch-2 document, so they also prove the
    added fields were mapped before the documents arrived.
    """
    queries: list[tuple[str, dict[str, Any], int]] = []
    group_field = "profile" if job.name == "ocr" else "lang"
    for group, count in job.by_group.items():
        queries.append((f"{group_field}={group}", {"term": {group_field: group}}, count))
    text_field = "text_clean" if job.name == "ocr" else "text"
    for doc_id, document in job.samples.items():
        text = document[text_field] or document.get("text_nfc") or ""
        words = text.split()[:6]
        if not words:
            continue
        clause = {"match_phrase": {text_field if document[text_field] else "text_nfc": " ".join(words)}}
        queries.append(
            (f"{text_field} of {doc_id}", {"bool": {"must": [clause], "filter": [{"term": {job.id_field: doc_id}}]}}, 1)
        )
    if job.name == "ocr":
        queries.append(("nested boxes.region", {"nested": {"path": "boxes", "query": {"term": {"boxes.region": "hud"}}}}, -1))
        queries.append(("banner_date", {"range": {"banner_date": {"gte": "2026-01-01"}}}, -1))
        queries.append(("race_stage", {"term": {"race_stage": 6}}, -1))
    return queries


def verify_upload(client: ElasticClient, job: Job, before: dict[str, int]) -> dict[str, Any]:
    client.refresh(job.index)
    counts = remote_counts(client, job)
    by_category = category_counts(client, job.index, job.categories)
    problems = [
        f"{category}: remote {by_category.get(category, 0):,}, source {expected:,}"
        for category, expected in job.by_category.items()
        if by_category.get(category, 0) != expected
    ]
    if counts["batch2"] != job.total:
        problems.append(f"batch-2 total {counts['batch2']:,}, expected {job.total:,}")
    if counts["base_l21_l30"] != before["base_l21_l30"]:
        problems.append(f"L21–L30 count moved {before['base_l21_l30']:,} -> {counts['base_l21_l30']:,}")
    if counts["other"]:
        problems.append(f"{counts['other']:,} documents outside L21–L30 and batch 2")

    result = client.json_request("POST", f"{job.index}/_mget", payload={"ids": list(job.samples)}, timeout=60)
    for doc_id, doc in zip(job.samples, result["docs"]):
        if not doc.get("found"):
            problems.append(f"sample {doc_id} not found")
        elif doc["_source"] != job.samples[doc_id]:
            problems.append(f"sample {doc_id} differs from the source document")

    smoke: dict[str, int] = {}
    for label, query, expected in smoke_queries(job):
        found = count_where(client, job.index, query)
        smoke[label] = found
        if (expected >= 0 and found != expected) or (expected < 0 and found == 0):
            problems.append(f"query {label}: {found:,} hits, expected {expected if expected >= 0 else '> 0'}")
    if problems:
        raise SystemExit(f"[{job.index}] verification FAILED:\n  " + "\n  ".join(problems))
    return {
        "index": job.index,
        "counts": counts,
        "by_group": job.by_group,
        "samples": list(job.samples),
        "queries": smoke,
        "source_sha256": job.source_sha256,
    }


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


def state_binding(job: Job, endpoint_hash: str) -> dict[str, Any]:
    return {
        "index": job.index,
        "endpoint_identity_sha256": endpoint_hash,
        "source_sha256": job.source_sha256,
        "registry_sha256": {name: profile.registry_sha256 for name, profile in PROFILES.items()},
        "documents": job.total,
    }


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise SystemExit(f"{path}: unknown schema; use a different --state-file")
        return state
    return {"schema_version": STATE_SCHEMA_VERSION, "created_at": utc_now(), "jobs": {}}


def job_state(state: dict[str, Any], job: Job, binding: dict[str, Any], path: Path) -> dict[str, Any]:
    entry = state["jobs"].get(job.name)
    if entry is None:
        entry = state["jobs"][job.name] = {"binding": binding, "completed_videos": [], "complete": False}
    elif entry.get("binding") != binding:
        raise SystemExit(
            f"{path} holds {job.name} state for another index/endpoint/source snapshot. "
            "Use a different --state-file (re-indexing is idempotent)."
        )
    return entry


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def upload(client: ElasticClient, job: Job, args: argparse.Namespace, state: dict[str, Any], endpoint_hash: str) -> dict[str, Any]:
    before, missing = preflight_remote(client, job)
    entry = job_state(state, job, state_binding(job, endpoint_hash), args.state_file)
    atomic_write_json(args.state_file, state)
    extend_mapping(client, job, missing)
    completed = set(entry["completed_videos"])
    if before["batch2"] and not completed:
        print(
            f"Note: {before['batch2']:,} batch-2 documents exist without local state; "
            "re-indexing all of them (same _id, idempotent).",
            flush=True,
        )
    todo = ((video_id, document) for video_id, document in job.documents() if video_id not in completed)
    print(f"[{job.index}] uploading {job.name} ({len(completed)} videos already checkpointed)", flush=True)
    started = time.monotonic()
    sent = 0
    for number, (docs, finished) in enumerate(plan_bulks(todo, args.batch_size, args.max_bulk_bytes), start=1):
        client.bulk_ndjson(build_bulk_payload(job.index, docs, id_field=job.id_field))
        sent += len(docs)
        completed.update(finished)
        entry["completed_videos"] = sorted(completed)
        entry["updated_at"] = utc_now()
        atomic_write_json(args.state_file, state)
        if number % 25 == 0:
            rate = sent / max(time.monotonic() - started, 0.001)
            print(f"  [{job.name}] sent {sent:,} ({rate:,.0f} docs/s)", flush=True)
    verification = verify_upload(client, job, before)
    entry["complete"] = True
    entry["verified_at"] = utc_now()
    atomic_write_json(args.state_file, state)
    print(
        f"[{job.index}] VERIFIED: {verification['counts']['total']:,} documents "
        f"(L21–L30 {verification['counts']['base_l21_l30']:,} + batch 2 {verification['counts']['batch2']:,}), "
        f"sent {sent:,} in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    return {"before": before, **verification}


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    for index in (args.ocr_index, args.speech_index):
        assert_writable_index(index, allow_non_v2=False)
    rows_by_profile = load_registries(root=args.registry_root)
    builders = {"ocr": build_ocr_job, "speech": build_speech_job}
    jobs = []
    for name in args.only:
        job = builders[name](args, rows_by_profile)
        print(
            f"[{name}] source PASS: {job.total:,} documents -> {job.index}; by {json.dumps(job.by_group)}; "
            f"{len(job.by_category)} categories",
            flush=True,
        )
        jobs.append(job)

    if args.dry_run:
        for job in jobs:
            sample = next(iter(job.samples.values()))
            print(json.dumps(sample, ensure_ascii=False)[:1500])
        print(f"DRY RUN PASSED in {time.monotonic() - started:.1f}s; Elastic was not contacted.", flush=True)
        return 0

    endpoint = read_secret(args.endpoint_file)
    client = ElasticClient(endpoint, read_secret(args.api_key_file))
    info = client.json_request("GET", "/", expected=(200,), timeout=60)
    print(f"Connected to Elasticsearch {(info.get('version') or {}).get('number')}", flush=True)
    if args.check_remote:
        for job in jobs:
            preflight_remote(client, job)
        print("REMOTE CHECK PASSED (read-only); nothing was written.", flush=True)
        return 0

    endpoint_hash = endpoint_identity(endpoint)
    state = load_state(args.state_file)
    verification = (
        json.loads(args.verification_file.read_text(encoding="utf-8")) if args.verification_file.exists() else {}
    )
    verification.setdefault("jobs", {})
    for job in jobs:
        verification["jobs"][job.name] = {"verified_at": utc_now(), **upload(client, job, args, state, endpoint_hash)}
        atomic_write_json(
            args.verification_file,
            {
                **verification,
                "schema_version": 1,
                "status": "PASS",
                "endpoint_identity_sha256": endpoint_hash,
                "registry_sha256": {name: profile.registry_sha256 for name, profile in PROFILES.items()},
            },
        )
    print(f"UPLOAD VERIFIED for {', '.join(args.only)} in {time.monotonic() - started:.1f}s.", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=["ocr", "speech"], default=["ocr", "speech"])
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT, help="Local copy of the OCR prefix (holds final/)")
    parser.add_argument("--speech-zip", type=Path, default=DEFAULT_SPEECH_ZIP)
    parser.add_argument("--ocr-index", default=DEFAULT_OCR_INDEX)
    parser.add_argument("--speech-index", default=DEFAULT_SPEECH_INDEX)
    parser.add_argument("--endpoint-file", type=Path, default=Path("API_KEY/elastic_endpoint.txt"))
    parser.add_argument("--api-key-file", type=Path, default=Path("API_KEY/elastic_apikey.txt"))
    parser.add_argument("--state-file", type=Path, default=Path(".elastic_upload_batch2_ocr_speech_state.json"))
    parser.add_argument(
        "--verification-file", type=Path, default=Path("elastic_upload_batch2_ocr_speech_verification.json")
    )
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--max-bulk-bytes", type=int, default=DEFAULT_MAX_BULK_BYTES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and build every document; no network.")
    mode.add_argument("--check-remote", action="store_true", help="Read-only checks against the indices; no writes.")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 10_000:
        parser.error("--batch-size must be between 1 and 10000")
    args.only = list(dict.fromkeys(args.only))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print("Interrupted. Finished videos are checkpointed; rerun the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
