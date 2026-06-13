import pytest

from app.identity import (
    canonical_submit_keyframe_id,
    category_from_video_id,
    group_from_video_id,
    make_submit_keyframe_id,
    normalize_category,
    parse_submit_keyframe_id,
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
