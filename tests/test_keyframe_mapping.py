import unittest

from keyframe_mapping import (
    Keyframe,
    KeyframeIndex,
    confidence_bucket,
    make_submit_keyframe_id,
    map_time_span,
    make_time_span_id,
    submit_category,
)


class KeyframeMappingTest(unittest.TestCase):
    def setUp(self):
        self.index = KeyframeIndex(
            {
                "K01_V001": [
                    Keyframe(n=1, pts_time=0.0, fps=25.0, frame_idx=0),
                    Keyframe(n=2, pts_time=5.0, fps=25.0, frame_idx=125),
                    Keyframe(n=3, pts_time=12.0, fps=25.0, frame_idx=300),
                ]
            }
        )

    def test_nearest_keyframe_clamps_and_prefers_previous_on_tie(self):
        self.assertEqual(self.index.nearest("K01_V001", -2.0).n, 1)
        self.assertEqual(self.index.nearest("K01_V001", 99.0).n, 3)
        self.assertEqual(self.index.nearest("K01_V001", 2.5).n, 1)
        self.assertEqual(self.index.nearest("K01_V001", 9.0).n, 3)

    def test_map_time_span_adds_start_center_and_end_keyframes(self):
        mapped = map_time_span(self.index, "K01_V001", start=4.0, end=11.0)

        self.assertEqual(mapped["center_time"], 7.5)
        self.assertEqual(mapped["start_keyframe_id"], "K01_V001/002")
        self.assertEqual(mapped["center_keyframe_id"], "K01_V001/002")
        self.assertEqual(mapped["end_keyframe_id"], "K01_V001/003")
        self.assertEqual(mapped["center_keyframe_pts_time"], 5.0)
        self.assertEqual(mapped["center_keyframe_frame_idx"], 125)

    def test_confidence_bucket(self):
        self.assertEqual(confidence_bucket(None), "missing")
        self.assertEqual(confidence_bucket(0.39), "low")
        self.assertEqual(confidence_bucket(0.49), "mid")
        self.assertEqual(confidence_bucket(0.50), "high")

    def test_time_span_id_is_stable_and_millisecond_based(self):
        self.assertEqual(
            make_time_span_id("K01_V001", 3.642, 28.962, "s000001"),
            "K01_V001_s000001_000003642_000028962",
        )

    def test_submit_keyframe_id_normalizes_l26_shards(self):
        self.assertEqual(submit_category("K01_V001"), "K01")
        self.assertEqual(submit_category("L26_V001"), "L26")
        self.assertEqual(make_submit_keyframe_id("L26_V001", 1), "L26/L26_V001/001")


if __name__ == "__main__":
    unittest.main()
