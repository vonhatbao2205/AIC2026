import pytest

from app.identity import (
    canonical_submit_keyframe_id,
    category_from_video_id,
    group_from_video_id,
    make_submit_keyframe_id,
    normalize_category,
    parse_submit_keyframe_id,
    video_id_prefix,
)


def test_normalize_l26_shards():
    assert normalize_category("L26_a") == "L26"
    assert normalize_category("L26_b") == "L26"
    assert normalize_category("L26_a1") == "L26"
    assert normalize_category("L26") == "L26"
    assert normalize_category("K01") == "K01"


def test_parse_submit_keyframe_id_basic():
    p = parse_submit_keyframe_id("K01/K01_V001/001")
    assert p.category == "K01"
    assert p.video_id == "K01_V001"
    assert p.keyframe_n == 1
    assert p.keyframe_name == "001"
    assert p.submit_keyframe_id == "K01/K01_V001/001"


def test_parse_normalizes_shard_and_pads():
    p = parse_submit_keyframe_id("L26_a/L26_V001/14")
    assert p.category == "L26"
    assert p.keyframe_n == 14
    assert p.submit_keyframe_id == "L26/L26_V001/014"


def test_parse_strips_jpg():
    p = parse_submit_keyframe_id("K01/K01_V001/007.jpg")
    assert p.keyframe_n == 7
    assert p.submit_keyframe_id == "K01/K01_V001/007"


def test_parse_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_submit_keyframe_id("not-an-id")
    with pytest.raises(ValueError):
        parse_submit_keyframe_id("K01/K01_V001/abc")


def test_make_submit_keyframe_id():
    assert make_submit_keyframe_id("K01_V001", 1) == "K01/K01_V001/001"
    assert make_submit_keyframe_id("L26_V001", 14) == "L26/L26_V001/014"


def test_group_and_category_helpers():
    assert group_from_video_id("K01_V001") == "K01"
    assert category_from_video_id("L26_V001") == "L26"


def test_canonical_idempotent():
    assert canonical_submit_keyframe_id("L26_a/L26_V001/14") == "L26/L26_V001/014"
    once = canonical_submit_keyframe_id("L26_a/L26_V001/14")
    assert canonical_submit_keyframe_id(once) == once


def test_batch2_hyphenated_video_ids_keep_their_folder():
    # BTC names the camera (N) and cycling (S) series with a hyphen.
    assert group_from_video_id("M01_V001") == "M01"
    assert group_from_video_id("N001-V001") == "N001"
    assert category_from_video_id("S01-V012") == "S01"
    assert make_submit_keyframe_id("N001-V001", 2) == "N001/N001-V001/002"
    parsed = parse_submit_keyframe_id("S01/S01-V007/75123")
    assert (parsed.category, parsed.video_id, parsed.keyframe_n) == ("S01", "S01-V007", 75123)
    assert [video_id_prefix(cat) for cat in ("L21", "K01", "M10", "N100", "S01")] == [
        "L21_", "K01_", "M10_", "N100-", "S01-",
    ]
