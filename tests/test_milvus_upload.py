import unittest

import numpy as np

from milvus_upload import (
    build_audio_entity,
    build_image_entity,
    parse_keyframe_image_path,
    upload_entities,
    vector_dtype_name,
)


class TestMilvusUploadHelpers(unittest.TestCase):
    def test_parse_keyframe_image_path_normalizes_submit_category(self):
        parsed = parse_keyframe_image_path("Keyframes_L26_a/keyframes/L26_V001/007.jpg")

        self.assertEqual(parsed["category"], "L26")
        self.assertEqual(parsed["video_id"], "L26_V001")
        self.assertEqual(parsed["keyframe_n"], 7)
        self.assertEqual(parsed["keyframe_id"], "L26_V001/007")
        self.assertEqual(parsed["submit_keyframe_id"], "L26/L26_V001/007")

    def test_build_image_entity_preserves_join_fields(self):
        vector = np.array([1.0, 2.0], dtype=np.float32)

        entity = build_image_entity("Keyframes_L21/keyframes/L21_V001/001.jpg", vector, "float16")

        self.assertEqual(entity["id"], "L21/L21_V001/001")
        self.assertEqual(entity["video_id"], "L21_V001")
        self.assertEqual(entity["keyframe_n"], 1)
        self.assertEqual(entity["embedding"].dtype, np.float16)

    def test_build_audio_entity_preserves_window_join_fields(self):
        record = {
            "window_id": "K01_V001_a000001_000002500_000007500",
            "video_id": "K01_V001",
            "start": 2.5,
            "end": 7.5,
            "glap_idx": 1,
            "keyframe_id": "K01_V001/002",
            "submit_keyframe_id": "K01/K01_V001/002",
            "keyframe_n": 2,
            "top1_label": "Speech",
        }
        vector = np.array([0.25, 0.5], dtype=np.float16)

        entity = build_audio_entity(record, vector, "float16")

        self.assertEqual(entity["id"], record["window_id"])
        self.assertEqual(entity["submit_keyframe_id"], "K01/K01_V001/002")
        self.assertEqual(entity["glap_idx"], 1)
        self.assertEqual(entity["embedding"].dtype, np.float16)

    def test_vector_dtype_name_validates_supported_names(self):
        self.assertEqual(vector_dtype_name("float32"), "FLOAT_VECTOR")
        self.assertEqual(vector_dtype_name("float16"), "FLOAT16_VECTOR")

        with self.assertRaises(ValueError):
            vector_dtype_name("float64")

    def test_upload_entities_retries_failed_batches_with_new_client(self):
        class FakeClient:
            attempts = 0

            def upsert(self, collection_name, data):
                self.__class__.attempts += 1
                if self.__class__.attempts == 1:
                    raise RuntimeError("temporary stream timeout")
                self.rows = list(data)

        clients = [FakeClient(), FakeClient()]

        uploaded = upload_entities(
            clients[0],
            "vectors",
            iter([{"id": "a"}, {"id": "b"}]),
            batch_size=2,
            op="upsert",
            label="test",
            max_retries=2,
            retry_sleep=0,
            client_factory=lambda: clients.pop(0),
        )

        self.assertEqual(uploaded, 2)
        self.assertEqual(FakeClient.attempts, 2)


if __name__ == "__main__":
    unittest.main()
