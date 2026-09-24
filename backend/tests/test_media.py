from app.media import MediaUrlBuilder

BASE = "https://media.example.com"


def test_keyframe_url_from_submit_id():
    b = MediaUrlBuilder(BASE)
    assert (
        b.keyframe_url_from_submit_id("K01/K01_V001/001")
        == "https://media.example.com/Keyframes/Keyframes_K01/K01_V001/001.jpg"
    )


def test_keyframe_url_from_video_and_n():
    b = MediaUrlBuilder(BASE)
    assert (
        b.keyframe_url("L26_V001", 14)
        == "https://media.example.com/Keyframes/Keyframes_L26/L26_V001/014.jpg"
    )


def test_video_url():
    b = MediaUrlBuilder(BASE)
    assert b.video_url("K01_V001") == "https://media.example.com/Videos/Videos_K01/K01_V001.mp4"


def test_base_url_trailing_slash_normalized():
    b = MediaUrlBuilder(BASE + "/")
    assert b.video_url("K01_V001") == "https://media.example.com/Videos/Videos_K01/K01_V001.mp4"


def test_shard_submit_id_resolves_to_video_group():
    # Submit id may carry a normalized L26 category; media group follows video_id.
    b = MediaUrlBuilder(BASE)
    url = b.keyframe_url_from_submit_id("L26/L26_V001/014")
    assert "Keyframes_L26/L26_V001/014.jpg" in url


def test_split_keyframe_hf_and_video_r2_origins():
    b = MediaUrlBuilder("https://hf.test/resolve", "https://r2.test")

    assert b.keyframe_url("L26_V001", 12).startswith("https://hf.test/resolve/Keyframes/")
    assert b.video_url("L26_V001") == "https://r2.test/Videos/Videos_L26/L26_V001.mp4"


def test_batch2_hyphenated_video_ids_resolve_to_their_folder():
    b = MediaUrlBuilder(BASE)
    assert (
        b.keyframe_url_from_submit_id("N001/N001-V001/002")
        == "https://media.example.com/Keyframes/Keyframes_N001/N001-V001/002.jpg"
    )
    assert b.video_url("S01-V007") == "https://media.example.com/Videos/Videos_S01/S01-V007.mp4"


def test_traffic_camera_videos_play_from_their_repaired_copies():
    b = MediaUrlBuilder(BASE)
    assert b.video_url("N001-V001") == "https://media.example.com/Videos_Web/Videos_N001/N001-V001.mp4"
    # Keyframes and every other series keep the original layout.
    assert "/Keyframes/Keyframes_N001/" in b.keyframe_url("N001-V001", 1)
    assert b.video_url("M01_V001") == "https://media.example.com/Videos/Videos_M01/M01_V001.mp4"


def test_republished_videos_follow_the_overrides_file(tmp_path, monkeypatch):
    import json
    from app import media

    path = tmp_path / "video_overrides.json"
    monkeypatch.setattr(media, "_OVERRIDES_PATH", path)
    monkeypatch.setattr(media, "_overrides", (-1.0, {}))
    b = MediaUrlBuilder(BASE)
    assert "/Videos_Web/Videos_N027/" in b.video_url("N027-V003")  # no file yet

    path.write_text(json.dumps({"roots": {"Videos_Web_v2": ["N027-V003"]}}), encoding="utf-8")
    assert b.video_url("N027-V003") == "https://media.example.com/Videos_Web_v2/Videos_N027/N027-V003.mp4"
    assert "/Videos_Web/Videos_N027/N027-V001.mp4" in b.video_url("N027-V001")
