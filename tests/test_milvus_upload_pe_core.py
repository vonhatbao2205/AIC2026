import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa

from milvus_upload_pe_core import (
    EXPECTED_DIM,
    MapRow,
    endpoint_identity,
    load_or_create_state,
    upsert_shard_parallel,
    validate_and_build_entities,
)


class TestPeCoreMilvusUpload(unittest.TestCase):
    def _batch(self, *, keyframe_n: int = 1):
        del keyframe_n
        vector = np.zeros(EXPECTED_DIM, dtype=np.float32)
        vector[7] = 1.0
        arrays = [
            pa.array(["L25_V001@f00000123"]),
            pa.array(["L25_V001"]),
            pa.array(["L25"]),
            pa.array([123], type=pa.int64()),
            pa.array([4.92], type=pa.float64()),
            pa.array([25.0], type=pa.float32()),
            pa.array(["infoshootpp/keyframes/L25/L25_V001/f00000123.jpg"]),
            pa.FixedSizeListArray.from_arrays(pa.array(vector), EXPECTED_DIM),
        ]
        return pa.RecordBatch.from_arrays(
            arrays,
            names=[
                "frame_id",
                "video_id",
                "category",
                "frame_idx",
                "pts_time",
                "fps",
                "image_relpath",
                "embedding",
            ],
        )

    def test_final_map_ordinal_becomes_application_primary_key(self):
        remaining = {
            "L25_V001": {
                123: MapRow(keyframe_n=1001, frame_idx=123, pts_time=4.92, fps=25.0)
            }
        }

        entities, samples, first_id, last_id = validate_and_build_entities(
            self._batch(), "L25", remaining, build_entities=True
        )

        self.assertEqual(remaining, {})
        self.assertEqual(first_id, "L25_V001@f00000123")
        self.assertEqual(last_id, first_id)
        self.assertEqual(len(samples), 1)
        self.assertEqual(entities[0]["id"], "L25/L25_V001/1001")
        self.assertEqual(entities[0]["keyframe_id"], "L25_V001/1001")
        self.assertEqual(entities[0]["frame_id"], "L25_V001@f00000123")
        self.assertEqual(entities[0]["frame_idx"], 123)
        self.assertEqual(entities[0]["embedding"].shape, (EXPECTED_DIM,))

    def test_missing_frame_idx_join_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No unique InfoShot\\+\\+ map row"):
            validate_and_build_entities(
                self._batch(),
                "L25",
                {"L25_V001": {124: MapRow(1, 124, 4.96, 25.0)}},
                build_entities=False,
            )

    def test_upload_state_is_atomic_and_binding_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state, created = load_or_create_state(path, {"collection": "new", "rows": 1})
            self.assertTrue(created)
            self.assertEqual(json.loads(path.read_text())["binding"]["collection"], "new")
            resumed, created = load_or_create_state(path, {"collection": "new", "rows": 1})
            self.assertFalse(created)
            self.assertEqual(resumed, state)
            with self.assertRaisesRegex(ValueError, "binding differs"):
                load_or_create_state(path, {"collection": "other", "rows": 1})

    def test_endpoint_identity_does_not_contain_hostname(self):
        identity = endpoint_identity("https://example.invalid:19530")
        self.assertEqual(len(identity), 64)
        self.assertNotIn("example", identity)

    def test_parallel_upload_uses_each_client_and_uploads_all_batches(self):
        class FakeClient:
            def __init__(self):
                self.ids = []

            def upsert(self, collection_name, data, timeout):
                self.ids.extend(item["id"] for item in data)
                return {"upsert_count": len(data)}

        boxes = [[FakeClient()], [FakeClient()]]
        batches = [[{"id": "a"}], [{"id": "b"}], [{"id": "c"}], [{"id": "d"}]]
        upsert_shard_parallel(
            boxes,
            "unused",
            "unused",
            "collection",
            batches,
            max_retries=0,
            retry_sleep=0,
            timeout=1,
        )
        self.assertEqual(boxes[0][0].ids, ["a", "c"])
        self.assertEqual(boxes[1][0].ids, ["b", "d"])


if __name__ == "__main__":
    unittest.main()
