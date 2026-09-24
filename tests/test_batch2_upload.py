import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa

from batch2_keyframes import PROFILES, KeyframeRow, profile_for_video, validate_rows
from elastic_upload import iter_infoshotpp_keyframes, keyframe_mapping
from elastic_upload_batch2_keyframe_map import build_document, plan_bulks
from milvus_upload_pe_core import EXPECTED_DIM
from milvus_upload_pe_core_batch2 import (
    DuplicateCounter,
    check_base_verification,
    in_expr,
    validate_and_build_entities,
)


def registry_columns(video_id, category, frames, *, profile="N", r2_key=None):
    """Registry columns for one video; `frames` is a list of (n, frame_idx)."""
    signature = PROFILES[profile].pipeline_signature
    return {
        "n": [n for n, _ in frames],
        "frame_id": [f"{video_id}@f{idx:08d}" for _, idx in frames],
        "video_id": [video_id] * len(frames),
        "category": [category] * len(frames),
        "frame_idx": [idx for _, idx in frames],
        "pts_time": [idx / 25.0 for _, idx in frames],
        "fps": [25.0] * len(frames),
        "quality": [0.9] * len(frames),
        "keyframe_name": [f"{n:03d}.jpg" for n, _ in frames],
        "submit_keyframe_id": [f"{category}/{video_id}/{n:03d}" for n, _ in frames],
        "r2_bucket": ["aic26-infoshot-keyframes"] * len(frames),
        "r2_key": [r2_key or f"Keyframes/Keyframes_{category}/{video_id}/{n:03d}.jpg" for n, _ in frames],
        "pipeline_signature": [signature] * len(frames),
    }


def keyframe(video_id, n, frame_idx, *, quality=0.9):
    profile, category = profile_for_video(video_id)
    return KeyframeRow(
        profile=profile.name,
        frame_id=f"{video_id}@f{frame_idx:08d}",
        video_id=video_id,
        category=category,
        n=n,
        frame_idx=frame_idx,
        pts_time=frame_idx / 30.0,
        fps=30.0,
        quality=quality,
        r2_key=f"Keyframes/Keyframes_{category}/{video_id}/{n:03d}.jpg",
    )


class TestBatch2Registry(unittest.TestCase):
    def test_hyphenated_video_ids_resolve_to_their_folder(self):
        self.assertEqual(profile_for_video("M03_V005")[1], "M03")
        self.assertEqual(profile_for_video("N001-V001")[1], "N001")
        self.assertEqual(profile_for_video("S01-V012")[1], "S01")
        with self.assertRaisesRegex(ValueError, "Not a batch-2 video id"):
            profile_for_video("L21_V001")

    def test_rows_are_validated_and_ordered_by_ordinal(self):
        rows = validate_rows(PROFILES["N"], registry_columns("N001-V001", "N001", [(2, 375), (1, 0)]))
        self.assertEqual([row.n for row in rows], [1, 2])
        self.assertEqual(rows[1].submit_keyframe_id, "N001/N001-V001/002")
        self.assertEqual(rows[1].r2_key, "Keyframes/Keyframes_N001/N001-V001/002.jpg")

    def test_frame_idx_named_jpeg_is_rejected(self):
        columns = registry_columns(
            "N001-V001", "N001", [(1, 0)], r2_key="Keyframes/Keyframes_N001/N001-V001/f00000000.jpg"
        )
        with self.assertRaisesRegex(ValueError, "r2_key"):
            validate_rows(PROFILES["N"], columns)

    def test_gap_in_ordinal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "n jumps 1 -> 3"):
            validate_rows(PROFILES["N"], registry_columns("N001-V001", "N001", [(1, 0), (3, 40)]))

    def test_video_of_another_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not belong to profile N"):
            validate_rows(PROFILES["N"], registry_columns("M01_V001", "M01", [(1, 0)]))


class TestElasticKeyframeMap(unittest.TestCase):
    def test_document_has_the_l_document_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "L21" / "L21_V001.csv"
            path.parent.mkdir()
            path.write_text("n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n", encoding="utf-8")
            l_document = next(iter_infoshotpp_keyframes(Path(directory)))
        document = build_document(keyframe("S01-V001", 7, 120))
        self.assertEqual(list(document), list(l_document))
        self.assertEqual(set(document), set(keyframe_mapping()["mappings"]["properties"]))
        self.assertEqual(document["submit_keyframe_id"], "S01/S01-V001/007")
        self.assertEqual(document["submit_category"], "S01")
        self.assertEqual(document["frame_idx"], 120)

    def test_video_is_checkpointed_only_with_its_last_document(self):
        rows = [keyframe("N001-V001", n, n * 10) for n in (1, 2, 3)] + [keyframe("N001-V002", 1, 0)]
        plan = list(plan_bulks(rows, batch_size=2))
        self.assertEqual([len(docs) for docs, _ in plan], [2, 2])
        self.assertEqual([finished for _, finished in plan], [[], ["N001-V001", "N001-V002"]])

    def test_interleaved_videos_are_rejected(self):
        rows = [keyframe("N001-V001", 1, 0), keyframe("N001-V002", 1, 0), keyframe("N001-V001", 2, 10)]
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            list(plan_bulks(rows, batch_size=10))


class TestMilvusBatch2Join(unittest.TestCase):
    def _batch(self, frame_idx=120, image_relpath="Keyframes/Keyframes_S01/S01-V001/007.jpg"):
        vector = np.zeros(EXPECTED_DIM, dtype=np.float32)
        vector[3] = 1.0
        arrays = [
            pa.array([f"S01-V001@f{frame_idx:08d}"]),
            pa.array(["S01-V001"]),
            pa.array(["S01"]),
            pa.array([frame_idx], type=pa.int64()),
            pa.array([frame_idx / 30.0], type=pa.float64()),
            pa.array([30.0], type=pa.float32()),
            pa.array([image_relpath]),
            pa.FixedSizeListArray.from_arrays(pa.array(vector), EXPECTED_DIM),
        ]
        names = ["frame_id", "video_id", "category", "frame_idx", "pts_time", "fps", "image_relpath", "embedding"]
        return pa.RecordBatch.from_arrays(arrays, names=names)

    def test_registry_ordinal_becomes_the_primary_key(self):
        row = keyframe("S01-V001", 7, 120)
        remaining = {row.frame_id: row}
        entities, samples, first, last = validate_and_build_entities(self._batch(), remaining, build_entities=True)
        self.assertEqual(remaining, {})
        self.assertEqual((first, last), ("S01-V001@f00000120", "S01-V001@f00000120"))
        self.assertEqual(entities[0]["id"], "S01/S01-V001/007")
        self.assertEqual(entities[0]["keyframe_id"], "S01-V001/007")
        self.assertEqual(entities[0]["keyframe_n"], 7)
        self.assertEqual(entities[0]["image_path"], "Keyframes/Keyframes_S01/S01-V001/007.jpg")
        self.assertEqual(samples[0].submit_keyframe_id, "S01/S01-V001/007")

    def test_frame_without_embedded_registry_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no embedded registry row"):
            validate_and_build_entities(self._batch(), {}, build_entities=False)

    def test_vector_row_pointing_at_another_jpeg_is_rejected(self):
        row = keyframe("S01-V001", 7, 120)
        with self.assertRaisesRegex(ValueError, "differs from registry"):
            validate_and_build_entities(
                self._batch(image_relpath="Keyframes/Keyframes_S01/S01-V001/008.jpg"),
                {row.frame_id: row},
                build_entities=False,
            )

    def test_duplicates_are_counted_within_one_video_only(self):
        counter = DuplicateCounter()
        vector = np.ones(4, dtype=np.float32)
        counter.add("S", "S01-V006", vector)
        counter.add("S", "S01-V006", vector)
        counter.add("S", "S01-V007", vector)
        self.assertEqual(counter.by_profile["S"], 1)

    def test_filter_expression_quotes_categories(self):
        self.assertEqual(in_expr("category", ["N001", "M01"]), 'category in ["M01", "N001"]')

    def test_base_collection_must_be_the_verified_l_upload(self):
        base = {
            "status": "PASS",
            "collection": "aic26_image_peg14_infoshotpp_v1",
            "endpoint_identity_sha256": "a" * 64,
            "row_count": 1_339_055,
            "semantic_fingerprint": "2cce817755d48b2a59c50a2a345d5c5a42f59fc46b4b6bd1ce3668eca1b03141",
            "embedding_dim": 1280,
            "metric_type": "COSINE",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verification.json"
            path.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(check_base_verification(path, base["collection"], "a" * 64), base)
            with self.assertRaisesRegex(ValueError, "endpoint_identity_sha256"):
                check_base_verification(path, base["collection"], "b" * 64)


if __name__ == "__main__":
    unittest.main()
