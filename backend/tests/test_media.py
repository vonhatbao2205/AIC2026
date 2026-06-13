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
