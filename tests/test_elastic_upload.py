import json
import unittest

from elastic_upload import (
    auth_headers,
    build_bulk_payload,
    drop_first,
    normalize_ocr_record,
)


class ElasticUploadTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
