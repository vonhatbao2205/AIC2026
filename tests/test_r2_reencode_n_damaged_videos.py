import json
import tempfile
import unittest
from pathlib import Path

from r2_reencode_n_damaged_videos import TARGET_ROOT, add_override, max_gap, target_key, write_overrides


class TestReencodeChecks(unittest.TestCase):
    def test_max_gap_measures_the_worst_missing_moment(self):
        actual = [0.2, 0.24, 0.28, 0.4]
        self.assertAlmostEqual(max_gap([0.2, 0.28], actual), 0.0)
        # 0.33 s has no frame within 2 ms: the nearest is 0.28 s.
        self.assertAlmostEqual(max_gap([0.2, 0.33], actual), 0.05)

    def test_overrides_file_accumulates_and_can_be_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_overrides.json"
            add_override(path, "N027-V003")
            add_override(path, "N004-V001")
            self.assertEqual(json.loads(path.read_text())["roots"][TARGET_ROOT], ["N004-V001", "N027-V003"])
            write_overrides(path, {"N097-V003"})
            self.assertEqual(json.loads(path.read_text())["roots"][TARGET_ROOT], ["N097-V003"])

    def test_copies_go_to_their_own_root(self):
        self.assertEqual(target_key("N027-V003"), "Videos_Web_v2/Videos_N027/N027-V003.mp4")


if __name__ == "__main__":
    unittest.main()
