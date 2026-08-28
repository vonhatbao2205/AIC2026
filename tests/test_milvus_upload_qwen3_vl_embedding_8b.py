import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa

from milvus_upload_qwen3_vl_embedding_8b import (
    DEFAULT_COLLECTION,
    EXPECTED_DIM,
    EXPECTED_MODEL_ID,
    MapRow,
    endpoint_identity,
    expected_schema,
    load_or_create_state,
    validate_and_build_entities,
)


class TestQwen3VlMilvusUpload(unittest.TestCase):
    def _batch(self):
        vector = np.zeros(EXPECTED_DIM, dtype=np.float32)
        vector[17] = 1.0
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

    def test_qwen_contract_is_native_4096_and_has_own_collection(self):
        self.assertEqual(EXPECTED_MODEL_ID, "Qwen/Qwen3-VL-Embedding-8B")
        self.assertEqual(EXPECTED_DIM, 4096)
        self.assertIn("qwen3vl8b", DEFAULT_COLLECTION)

    def test_map_join_builds_application_identity_and_4096_vector(self):
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
        self.assertEqual(entities[0]["embedding"].shape, (4096,))

    def test_schema_uses_float_vector_4096(self):
        class FakeDataType:
            VARCHAR = "varchar"
            INT64 = "int64"
            DOUBLE = "double"
            FLOAT = "float"
            FLOAT_VECTOR = "float_vector"

        schema = expected_schema(FakeDataType)
        self.assertEqual(schema["embedding"], ("float_vector", {"dim": 4096}))

    def test_upload_state_and_endpoint_identity_are_target_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            binding = {"collection": DEFAULT_COLLECTION, "embedding_dim": EXPECTED_DIM}
            state, created = load_or_create_state(path, binding)
            self.assertTrue(created)
            self.assertEqual(state["binding"], binding)
            resumed, created = load_or_create_state(path, binding)
            self.assertFalse(created)
            self.assertEqual(resumed, state)

        identity = endpoint_identity("https://example.invalid:19530")
        self.assertEqual(len(identity), 64)
        self.assertNotIn("example", identity)


if __name__ == "__main__":
    unittest.main()
