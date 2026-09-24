import tempfile
import unittest
from pathlib import Path

from r2_publish_n_web_videos import replace_sei_with_filler, target_key


def nal(payload: bytes) -> bytes:
    return len(payload).to_bytes(4, "big") + payload


class TestSeiToFiller(unittest.TestCase):
    def test_only_sei_units_become_same_length_filler(self):
        sps = b"\x27\x4d\x40\x28\x8d"
        sei = b"\x06\x05\x2d\x40\xf7\x16\x6b"  # payload_size overruns the unit, as on the cameras
        idr = b"\x25\x88\x84\x00\x33"
        packet = nal(sps) + nal(sei) + nal(idr)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            path.write_bytes(b"HEAD" + packet + b"TAIL")
            stats = replace_sei_with_filler(path, [["0", "0", "40", str(len(packet)), "4", "K__"]])
            data = path.read_bytes()

        self.assertEqual(len(data), 8 + len(packet))
        self.assertEqual(data[:4] + data[-4:], b"HEADTAIL")
        patched = data[4:-4]
        self.assertEqual(patched[: 4 + len(sps)], nal(sps))
        self.assertEqual(patched[-(4 + len(idr)):], nal(idr))
        filler = patched[4 + len(sps): 4 + len(sps) + 4 + len(sei)]
        self.assertEqual(filler, nal(b"\x0c" + b"\xff" * (len(sei) - 2) + b"\x80"))
        self.assertEqual((stats["nal_6"], stats["sei_replaced"]), (1, 1))

    def test_a_corrupt_nal_length_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            path.write_bytes((99).to_bytes(4, "big") + b"\x06\x05")
            with self.assertRaisesRegex(ValueError, "bad NAL length 99"):
                replace_sei_with_filler(path, [["0", "0", "40", "6", "0", "K__"]])

    def test_copies_mirror_the_original_layout_under_their_own_prefix(self):
        self.assertEqual(
            target_key("Videos/Videos_N001/N001-V001.mp4"), "Videos_Web/Videos_N001/N001-V001.mp4"
        )


if __name__ == "__main__":
    unittest.main()
