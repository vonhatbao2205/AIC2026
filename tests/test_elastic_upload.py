import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from elastic_upload import (
    auth_headers,
    build_bulk_payload,
    chunked,
    drop_first,
    iter_infoshotpp_keyframes,
    iter_od_documents,
    normalize_ocr_record,
    normalize_od_record,
    od_committed_shard_ids,
    od_frames_mapping,
)


class FakeR2:
    """In-memory stand-in for R2Client: {key: bytes}."""

    def __init__(self, objects):
        self.objects = objects
        self.fetched = []

    def list_keys(self, prefix):
        return (key for key in sorted(self.objects) if key.startswith(prefix))

    def get_object(self, key):
        self.fetched.append(key)
        return self.objects[key]


def od_document(shard_frame, video_id="K01_V001"):
    return {
        "document_id": f"{video_id}:{shard_frame}",
        "keyframe_id": f"{video_id}:{shard_frame}",
        "video_id": video_id,
        "frame_name": shard_frame,
        "status": "ok",
    }


def od_fixture(prefix, shards, *, hashes=None, corrupt_checksum=False):
    objects = {}
    for shard_id, frame_names in shards.items():
        body = b"".join(
            json.dumps(od_document(name), ensure_ascii=False).encode("utf-8") + b"\n"
            for name in frame_names
        )
        payload = gzip.compress(body)
        objects[f"{prefix}/shards/part-{shard_id:06d}.jsonl.gz"] = payload
        marker = {
            "shard_id": shard_id,
            "document_count": len(frame_names),
            "error_count": 0,
            "sha256": "deadbeef" if corrupt_checksum else hashlib.sha256(payload).hexdigest(),
            **(hashes or {}),
        }
        objects[f"{prefix}/complete/part-{shard_id:06d}.complete.json"] = json.dumps(marker).encode("utf-8")
    return FakeR2(objects)


class ElasticUploadTest(unittest.TestCase):
    def test_infoshotpp_map_preserves_ordinal_and_decode_frame_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            category = root / "L25"
            category.mkdir()
            (category / "L25_V001.csv").write_text(
                "n,pts_time,fps,frame_idx\r\n"
                "1,0.120,25.0,3\r\n"
                "2,0.119,25.0,8\r\n",
                encoding="utf-8",
            )

            rows = list(iter_infoshotpp_keyframes(root))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["submit_keyframe_id"], "L25/L25_V001/002")
        self.assertEqual(rows[1]["keyframe_id"], "L25_V001/002")
        self.assertEqual(rows[1]["frame_id"], "L25_V001@f00000008")
        self.assertEqual(rows[1]["keyframe_n"], 2)
        self.assertEqual(rows[1]["frame_idx"], 8)
        self.assertEqual(rows[1]["pts_time"], 0.119)

    def test_infoshotpp_map_rejects_non_contiguous_n(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            category = root / "L21"
            category.mkdir()
            (category / "L21_V001.csv").write_text(
                "n,pts_time,fps,frame_idx\n2,0.1,25,3\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "expected 1"):
                list(iter_infoshotpp_keyframes(root))

    def test_auth_headers_wrap_raw_api_key(self):
        headers = auth_headers("abc123")

        self.assertEqual(headers["Authorization"], "ApiKey abc123")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_auth_headers_accept_existing_api_key_prefix(self):
        headers = auth_headers("ApiKey abc123")

        self.assertEqual(headers["Authorization"], "ApiKey abc123")

    def test_build_bulk_payload_uses_document_id_and_ndjson_trailing_newline(self):
        payload = build_bulk_payload(
            "speech_segments_v1",
            [{"segment_id": "s1", "text": "xin chào"}],
            id_field="segment_id",
        )
        lines = payload.decode("utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0]), {"index": {"_index": "speech_segments_v1", "_id": "s1"}})
        self.assertEqual(json.loads(lines[1]), {"segment_id": "s1", "text": "xin chào"})
        self.assertTrue(payload.endswith(b"\n"))

    def test_normalize_ocr_record_uses_submit_category_l26(self):
        record = normalize_ocr_record(
            {
                "id": "L26_a/L26_V001/001",
                "category": "L26_a",
                "video": "L26_V001",
                "text_clean": "abc",
            }
        )

        self.assertEqual(record["ocr_id"], "L26_a/L26_V001/001")
        self.assertEqual(record["video_id"], "L26_V001")
        self.assertEqual(record["keyframe_n"], 1)
        self.assertEqual(record["submit_category"], "L26")
        self.assertEqual(record["submit_keyframe_id"], "L26/L26_V001/001")

    def test_drop_first_skips_already_uploaded_records(self):
        self.assertEqual(list(drop_first(iter([1, 2, 3, 4]), 2)), [3, 4])
        self.assertEqual(list(drop_first(iter([1, 2]), 5)), [])


PREFIX = "Derived/ObjectDetection/aic26-od-v5/cfg/input-man/runtime-rt"


class ObjectDetectionUploadTest(unittest.TestCase):
    def test_normalize_od_record_adds_submit_identity(self):
        record = normalize_od_record(od_document("007", video_id="L26_V001"))

        self.assertEqual(record["submit_keyframe_id"], "L26/L26_V001/007")
        self.assertEqual(record["submit_category"], "L26")
        self.assertEqual(record["keyframe_n"], 7)
        self.assertEqual(record["document_id"], "L26_V001:007")

    def test_normalize_od_record_rejects_unexpected_identity(self):
        with self.assertRaises(ValueError):
            normalize_od_record({"document_id": "x", "video_id": "K01_V001", "frame_name": "abc"})

    def test_submit_identity_fields_are_in_the_strict_mapping(self):
        properties = od_frames_mapping()["mappings"]["properties"]

        self.assertEqual(od_frames_mapping()["mappings"]["dynamic"], "strict")
        for field in ("submit_keyframe_id", "submit_category", "keyframe_n"):
            self.assertIn(field, properties)
        self.assertEqual(properties["detections"]["type"], "nested")
        self.assertEqual(properties["object_counts"]["type"], "nested")

    def test_only_committed_shards_are_listed(self):
        r2 = od_fixture(PREFIX, {0: ["001"], 2: ["002"]})
        r2.objects[f"{PREFIX}/shards/part-000001.jsonl.gz"] = b"interrupted, no marker"

        self.assertEqual(od_committed_shard_ids(r2, PREFIX), [0, 2])

    def test_iter_od_documents_streams_every_committed_document(self):
        r2 = od_fixture(PREFIX, {0: ["001", "002"], 1: ["003"]})

        documents = list(iter_od_documents(r2, PREFIX, [0, 1]))

        self.assertEqual([doc["document_id"] for doc in documents], ["K01_V001:001", "K01_V001:002", "K01_V001:003"])

    def test_resume_skips_whole_shards_without_downloading_them(self):
        r2 = od_fixture(PREFIX, {0: ["001", "002"], 1: ["003", "004"]})

        documents = list(iter_od_documents(r2, PREFIX, [0, 1], skip=3))

        self.assertEqual([doc["document_id"] for doc in documents], ["K01_V001:004"])
        self.assertNotIn(f"{PREFIX}/shards/part-000000.jsonl.gz", r2.fetched)

    def test_checksum_mismatch_aborts_before_indexing(self):
        r2 = od_fixture(PREFIX, {0: ["001"]}, corrupt_checksum=True)

        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            list(iter_od_documents(r2, PREFIX, [0]))

    def test_namespace_mismatch_aborts_before_indexing(self):
        r2 = od_fixture(PREFIX, {0: ["001"]}, hashes={"runtime_hash": "other"})

        with self.assertRaisesRegex(RuntimeError, "namespace mismatch"):
            list(iter_od_documents(r2, PREFIX, [0], expected_hashes={"runtime_hash": "expected"}))

    def test_chunked_closes_a_batch_on_the_byte_budget(self):
        records = [{"id": index, "blob": "x" * 400} for index in range(10)]

        batches = list(chunked(iter(records), batch_size=1000, max_bytes=1024))

        self.assertGreater(len(batches), 1)
        self.assertEqual(sum(len(batch) for batch in batches), 10)
        for batch in batches:
            payload = build_bulk_payload("idx", batch, id_field="id")
            self.assertLessEqual(len(payload), 1024 + 512)


if __name__ == "__main__":
    unittest.main()
