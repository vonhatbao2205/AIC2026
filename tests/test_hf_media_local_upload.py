import csv
import tempfile
import unittest
from pathlib import Path

from HF_BUCKET.aic26_hf_media_local_upload import (
    Corpus,
    build_video_inventory,
    commit_binding,
    inventory_digest,
    keyframe_media_key,
    load_map_rows,
    process_video,
    valid_video_commit,
)


class FakeRemote:
    def __init__(self):
        self.bucket = "owner/media"
        self.prefix = ""
        self.uploaded_pairs = []
        self.json_objects = {}

    def key(self, relative):
        return relative

    def read_json(self, path):
        return self.json_objects.get(path)

    def upload_pairs(self, pairs, *, label):
        del label
        self.uploaded_pairs.extend(pairs)

    def verify_exact(self, expected, *, prefix, label):
        del prefix, label
        actual = {remote: local.stat().st_size for local, remote in self.uploaded_pairs}
        assert actual == expected

    def upload_json(self, path, value, *, label):
        del label
        self.json_objects[path] = value


class TestLocalHfMediaUpload(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "infoshootpp"
        self.map_path = self.root / "map-keyframes/L25/L25_V001.csv"
        self.image_dir = self.root / "keyframes/L25/L25_V001"
        self.map_path.parent.mkdir(parents=True)
        self.image_dir.mkdir(parents=True)
        with self.map_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["n", "pts_time", "fps", "frame_idx"])
            writer.writerow([1, 1.0, 25.0, 25])
            # Tiny pts_time regression is intentional and must remain valid.
            writer.writerow([2, 0.999, 25.0, 50])
        (self.image_dir / "f00000025.jpg").write_bytes(b"\xff\xd8first")
        (self.image_dir / "f00000050.jpg").write_bytes(b"\xff\xd8second")
        self.corpus = Corpus(
            source_root=self.root,
            source_state_dir=self.root.parent,
            source_bucket="owner/source",
            source_prefix="infoshootpp-v1",
            source_dataset_manifest_sha256="a" * 64,
            shards={
                "L25": {
                    "sha256": "b" * 64,
                    "tar_bytes": 1234,
                    "source_bytes": 15,
                }
            },
            map_paths={"L25_V001": self.map_path},
            category_videos={"L25": ("L25_V001",)},
            category_frames={"L25": 2},
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_map_projection_keeps_ordinal_and_frame_idx_distinct(self):
        rows = load_map_rows(self.map_path)
        self.assertEqual(rows[1].n, 2)
        self.assertEqual(rows[1].frame_idx, 50)
        self.assertEqual(rows[1].source_frame_name, "f00000050.jpg")
        self.assertEqual(rows[1].media_frame_name, "002.jpg")
        self.assertEqual(
            keyframe_media_key("L25", "L25_V001", rows[1].n),
            "Keyframes/Keyframes_L25/L25_V001/002.jpg",
        )

    def test_local_inventory_is_exact_and_does_not_modify_source(self):
        inventory = build_video_inventory(
            self.corpus, "L25", "L25_V001", verify_jpeg_magic=True
        )
        self.assertEqual(inventory.file_sizes, (7, 8))
        self.assertEqual(inventory.total_bytes, 15)
        self.assertEqual(
            inventory.inventory_sha256,
            inventory_digest(inventory.mapping, inventory.file_sizes),
        )
        self.assertTrue((self.image_dir / "f00000025.jpg").exists())

    def test_process_video_uploads_aliases_and_colab_compatible_commit(self):
        remote = FakeRemote()
        binding = commit_binding(self.corpus, remote.bucket, remote.prefix, "L25")
        commit = process_video(
            remote,
            self.corpus,
            "L25",
            "L25_V001",
            binding,
            verify_jpeg_magic=True,
            verify_completed_on_resume=False,
        )
        self.assertEqual(
            [remote_path for _, remote_path in remote.uploaded_pairs],
            [
                "Keyframes/Keyframes_L25/L25_V001/001.jpg",
                "Keyframes/Keyframes_L25/L25_V001/002.jpg",
            ],
        )
        inventory = build_video_inventory(
            self.corpus, "L25", "L25_V001", verify_jpeg_magic=True
        )
        self.assertTrue(valid_video_commit(commit, inventory, binding))
        self.assertEqual(commit["file_sizes"], [7, 8])
        self.assertTrue((self.image_dir / "f00000050.jpg").exists())

    def test_extra_local_jpeg_is_rejected(self):
        (self.image_dir / "f99999999.jpg").write_bytes(b"\xff\xd8extra")
        with self.assertRaisesRegex(Exception, "inventory mismatch"):
            build_video_inventory(
                self.corpus, "L25", "L25_V001", verify_jpeg_magic=True
            )


if __name__ == "__main__":
    unittest.main()
