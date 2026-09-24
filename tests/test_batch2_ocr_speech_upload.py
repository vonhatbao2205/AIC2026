import csv
import io
import json
import unittest
import zipfile

import keyframe_mapping_v2
from batch2_keyframes import PROFILES, KeyframeRow, profile_for_video
from elastic_upload_batch2_ocr_speech import (
    OCR_EXTRA_PROPERTIES,
    OCR_PIPELINE_SIGNATURE,
    OCR_POSTPROCESS_VERSION,
    SPEECH_EXTRA_PROPERTIES,
    SPEECH_MANIFEST,
    OcrSource,
    build_ocr_document,
    build_speech_documents,
    check_ocr_row,
    flatten_types,
    iter_ocr_documents,
    load_speech_manifest,
    map_time_span,
    mapping_problems,
    merged_properties,
    plan_bulks,
    video_keyframes,
)
from elastic_upload_v2 import ocr_mapping_v2, speech_mapping_v2


def keyframe(video_id, n, frame_idx, *, quality=0.9, fps=30.0):
    profile, category = profile_for_video(video_id)
    return KeyframeRow(
        profile=profile.name,
        frame_id=f"{video_id}@f{frame_idx:08d}",
        video_id=video_id,
        category=category,
        n=n,
        frame_idx=frame_idx,
        pts_time=frame_idx / fps,
        fps=fps,
        quality=quality,
        r2_key=f"Keyframes/Keyframes_{category}/{video_id}/{n:03d}.jpg",
    )


def ocr_row(kf, **overrides):
    row = {
        "frame_id": kf.frame_id,
        "id": f"{kf.category}/{kf.video_id}/f{kf.frame_idx:08d}",
        "video_id": kf.video_id,
        "category": kf.category,
        "profile": kf.profile,
        "n": kf.n,
        "keyframe_name": f"{kf.n:03d}.jpg",
        "submit_keyframe_id": kf.submit_keyframe_id,
        "frame_idx": kf.frame_idx,
        "pts_time": kf.pts_time,
        "fps": kf.fps,
        "r2_key": kf.r2_key,
        "keyframe_pipeline_signature": PROFILES[kf.profile].pipeline_signature,
        "pipeline_signature": OCR_PIPELINE_SIGNATURE,
        "postprocess_version": OCR_POSTPROCESS_VERSION,
        "status": "ok",
        "error": "",
        "text_clean": "CHAGEE",
        "text_clean_fold": "chagee",
        "text_clean_hash": "0123456789abcdef",
        "text_ticker": "",
        "text_ticker_fold": "",
        "text_banner": "AN DUONG VUONG 10.Jun 2026 11:03:21",
        "banner_camera": "AN DUONG VUONG",
        "banner_date": "2026-06-10",
        "text_hud": "",
        "race_stage": None,
        "race_time": "",
        "clock": "11:03:21",
        "hour": 11,
        "text_nfc": "AN DUONG VUONG 10.Jun 2026 11:03:21 CHAGEE",
        "boxes_json": json.dumps(
            [
                {"text": "AN DUONG VUONG", "box": [2, 3, 341, 35], "region": "banner"},
                {"text": "CHAGEE", "box": [44, 210, 95, 242], "region": "scene"},
                {"text": "no bbox", "box": None, "region": "unknown"},
            ]
        ),
    }
    row.update(overrides)
    return row


def segment(video_id, start, end, *, score=0.55, text="xin chào"):
    return {
        "video_id": video_id,
        "start": start,
        "end": end,
        "text": text,
        "words": [{"start": start, "end": end, "word": "xin", "score": score}],
        "avg_logprob": None,
        "no_speech_prob": None,
        "avg_word_score": score,
    }


class TestBatch2Ocr(unittest.TestCase):
    def test_document_is_the_l_document_plus_the_batch2_fields(self):
        kf = keyframe("N001-V001", 41, 4635)
        doc = build_ocr_document(ocr_row(kf), kf)
        l_fields = set(ocr_mapping_v2()["mappings"]["properties"]) - {"ts"}
        # `boxes` is an L field; the extra mapping only adds `region` inside it.
        self.assertEqual(set(doc) - (set(OCR_EXTRA_PROPERTIES) - {"boxes"}), l_fields)
        self.assertEqual(doc["submit_keyframe_id"], "N001/N001-V001/041")
        self.assertEqual(doc["submit_category"], "N001")
        self.assertEqual(doc["keyframe_name"], "041")
        self.assertEqual(doc["image_path"], "Keyframes/Keyframes_N001/N001-V001/041.jpg")
        self.assertEqual(doc["banner_date"], "2026-06-10")
        self.assertIsNone(doc["text_ticker"])  # empty parquet strings become null in the new fields
        self.assertIsNone(doc["race_time"])
        self.assertEqual(doc["clock"], "11:03:21")
        properties = merged_properties(ocr_mapping_v2()["mappings"]["properties"], OCR_EXTRA_PROPERTIES)
        self.assertEqual(mapping_problems(properties, [doc]), set())

    def test_row_must_describe_the_registry_jpeg(self):
        kf = keyframe("M03_V005", 9, 225)
        check_ocr_row(ocr_row(kf), kf, "M")
        with self.assertRaisesRegex(ValueError, "n=10"):
            check_ocr_row(ocr_row(kf, n=10), kf, "M")
        with self.assertRaisesRegex(ValueError, "pts_time"):
            check_ocr_row(ocr_row(kf, pts_time=kf.pts_time + 0.04), kf, "M")
        with self.assertRaisesRegex(ValueError, "not a keyframe of registry S"):
            check_ocr_row(ocr_row(kf), kf, "S")

    def test_only_black_frames_may_be_skipped(self):
        bright = keyframe("M10_V029", 900, 90000)
        with self.assertRaisesRegex(ValueError, "skipped as black"):
            check_ocr_row(ocr_row(bright, status="skipped"), bright, "M")
        black = keyframe("M10_V029", 901, 90100, quality=0.01)
        check_ocr_row(ocr_row(black, status="skipped", postprocess_version="", boxes_json="[]"), black, "M")
        with self.assertRaisesRegex(ValueError, "postprocess_version"):
            check_ocr_row(ocr_row(bright, postprocess_version=""), bright, "M")

    def test_malformed_boxes_are_rejected(self):
        kf = keyframe("S01-V006", 1, 0)
        for boxes in (
            [{"text": "x", "box": [1, 2, 3, 4], "region": "sky"}],
            [{"text": "x", "box": [1.5, 2, 3, 4], "region": "scene"}],
            [{"text": "x", "box": [1, 2, 3, 4], "region": "scene", "score": 1}],
        ):
            with self.assertRaises(ValueError):
                check_ocr_row(ocr_row(kf, boxes_json=json.dumps(boxes)), kf, "S")

    def test_rows_must_cover_the_registry_and_skip_only_what_ocr_skipped(self):
        frames = [keyframe("N001-V001", 1, 0), keyframe("N001-V001", 2, 375)]
        registry = {kf.frame_id: kf for kf in frames}
        source = OcrSource("N", "unused", ok=2, skipped=0)
        docs = list(iter_ocr_documents([ocr_row(kf) for kf in frames], registry, source))
        self.assertEqual([video for video, _ in docs], ["N001-V001", "N001-V001"])
        with self.assertRaisesRegex(ValueError, "1 rows, registry holds 2"):
            list(iter_ocr_documents([ocr_row(frames[0])], registry, source))
        with self.assertRaisesRegex(ValueError, "duplicate frame_id"):
            list(iter_ocr_documents([ocr_row(frames[0])] * 2, registry, source))

    def test_region_is_added_inside_the_nested_boxes(self):
        types = flatten_types(merged_properties(ocr_mapping_v2()["mappings"]["properties"], OCR_EXTRA_PROPERTIES))
        self.assertEqual(types["boxes"], "nested")
        self.assertEqual(types["boxes.text"], "text")
        self.assertEqual(types["boxes.region"], "keyword")
        self.assertEqual(types["banner_date"], "date")


class TestBatch2Speech(unittest.TestCase):
    def test_hyphenated_video_is_anchored_in_its_registry_category(self):
        video = video_keyframes([keyframe("S01-V001", n, n * 30) for n in range(1, 6)])
        mapped = map_time_span(video, 1.0, 3.0)
        self.assertEqual(mapped["center_submit_keyframe_id"], "S01/S01-V001/002")
        self.assertEqual(mapped["start_submit_keyframe_id"], "S01/S01-V001/001")
        self.assertEqual(mapped["end_submit_keyframe_id"], "S01/S01-V001/003")
        self.assertEqual(mapped["submit_keyframe_id"], mapped["center_submit_keyframe_id"])
        self.assertEqual(mapped["submit_category"], "S01")

    def test_underscore_video_is_anchored_exactly_as_the_l_mapper(self):
        video = video_keyframes([keyframe("M01_V001", n, n * 25 + 3) for n in range(1, 40)])
        for start, end in ((0.0, 4.2), (10.5, 31.0), (30.0, 99.0)):
            self.assertEqual(map_time_span(video, start, end), keyframe_mapping_v2.map_time_span(video, start, end))

    def test_english_narration_is_not_graded_by_the_vietnamese_aligner(self):
        video = video_keyframes([keyframe("M04_V016", n, n * 25) for n in range(1, 5)])
        member = "speech_out_batch2/M04_V016.speech.json"
        [english] = build_speech_documents(video, member, [segment("M04_V016", 0.5, 2.0, score=0.1)], "en")
        [vietnamese] = build_speech_documents(video, member, [segment("M04_V016", 0.5, 2.0, score=0.1)], "vi")
        self.assertEqual((english["lang"], english["confidence_bucket"]), ("en", "missing"))
        self.assertEqual(english["avg_word_score"], 0.1)
        self.assertEqual(vietnamese["confidence_bucket"], "low")

    def test_document_is_the_l_document_plus_lang(self):
        video = video_keyframes([keyframe("S01-V006", n, n * 30) for n in range(1, 5)])
        docs = build_speech_documents(
            video,
            "speech_out_batch2/S01-V006.speech.json",
            [segment("S01-V006", 0.2, 1.4), {**segment("S01-V006", 2.0, 2.5, score=None), "words": []}],
            "vi",
        )
        self.assertEqual(docs[0]["segment_id"], "S01-V006_s000000_000000200_000001400")
        self.assertEqual(docs[1]["confidence_bucket"], "missing")
        self.assertEqual(docs[1]["word_count"], 0)
        l_fields = set(speech_mapping_v2()["mappings"]["properties"])
        self.assertEqual(set(docs[0]) - set(SPEECH_EXTRA_PROPERTIES), l_fields)
        properties = merged_properties(speech_mapping_v2()["mappings"]["properties"], SPEECH_EXTRA_PROPERTIES)
        self.assertEqual(mapping_problems(properties, docs), set())

    def test_segment_of_another_video_is_rejected(self):
        video = video_keyframes([keyframe("M01_V001", 1, 0)])
        with self.assertRaisesRegex(ValueError, "video_id 'M01_V002'"):
            build_speech_documents(video, "x.speech.json", [segment("M01_V002", 0.0, 1.0)], "vi")

    def test_manifest_must_name_the_english_videos_of_the_handoff(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            text = io.StringIO()
            writer = csv.DictWriter(text, fieldnames=["video_id", "status", "n_segments", "mean_avg_word_score"])
            writer.writeheader()
            writer.writerow({"video_id": "M02_V001", "status": "ok", "n_segments": 1, "mean_avg_word_score": 0.1})
            archive.writestr(SPEECH_MANIFEST, text.getvalue())
        with zipfile.ZipFile(buffer) as archive:
            with self.assertRaisesRegex(ValueError, "English videos"):
                load_speech_manifest(archive)


class TestBulks(unittest.TestCase):
    def test_video_is_checkpointed_only_with_its_last_document(self):
        items = [("A", {"i": 1}), ("A", {"i": 2}), ("A", {"i": 3}), ("B", {"i": 4})]
        bulks = list(plan_bulks(items, 2, 1 << 20))
        self.assertEqual([[doc["i"] for doc in docs] for docs, _ in bulks], [[1, 2], [3, 4]])
        self.assertEqual([finished for _, finished in bulks], [[], ["A", "B"]])

    def test_byte_cap_closes_a_bulk(self):
        items = [("A", {"text": "x" * 400}) for _ in range(3)]
        self.assertEqual([len(docs) for docs, _ in plan_bulks(items, 100, 1100)], [2, 1])

    def test_interleaved_videos_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            list(plan_bulks([("A", {}), ("B", {}), ("A", {})], 10, 1 << 20))


if __name__ == "__main__":
    unittest.main()
