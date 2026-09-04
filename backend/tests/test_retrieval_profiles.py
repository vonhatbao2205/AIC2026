from dataclasses import replace

from app.config import Settings, get_settings
from app.media import MediaUrlBuilder


def configured_settings() -> Settings:
    return Settings(
        milvus_endpoint_1="https://btc.milvus.test",
        milvus_token_1="btc-token",
        milvus_endpoint_2="https://infoshot.milvus.test",
        milvus_token_2="infoshot-token",
        media_base_url="https://media.r2.test",
        keyframe_media_base_url_2="https://keyframe.r2.test",
        keyframe_media_fallback_base_url_2="https://huggingface.test/bucket/resolve",
        idx_keyframe_map_1="btc-map",
        idx_keyframe_map_2="infoshot-map",
        idx_ocr_1="btc-ocr",
        idx_ocr_2="infoshot-ocr-v2",
        idx_speech_1="btc-speech",
        idx_speech_2="infoshot-speech-v2",
        idx_audio_1="btc-audio",
        idx_audio_2="infoshot-audio-v2",
        milvus_image_collection_1="btc-images",
        milvus_image_collection_2="infoshot-images",
    )


def test_btc_profile_keeps_full_dataset_endpoints():
    profile = configured_settings().for_retrieval_database("btc")

    assert profile.milvus_endpoint == "https://btc.milvus.test"
    assert profile.idx_keyframe_map == "btc-map"
    assert (profile.idx_ocr, profile.idx_speech, profile.idx_audio) == (
        "btc-ocr",
        "btc-speech",
        "btc-audio",
    )
    assert profile.ocr_missing_categories == ()
    assert profile.milvus_image_collection == "btc-images"
    assert profile.keyframe_media_base_url == "https://media.r2.test"
    assert profile.has_glap is False  # no encoder configured in this fixture


def test_infoshot_profile_uses_r2_keyframes_with_hf_fallback():
    profile = configured_settings().for_retrieval_database("infoshotpp")
    media = MediaUrlBuilder(profile.keyframe_media_base_url, profile.media_base_url)

    assert profile.milvus_endpoint == "https://infoshot.milvus.test"
    assert profile.idx_keyframe_map == "infoshot-map"
    assert (profile.idx_ocr, profile.idx_speech, profile.idx_audio) == (
        "infoshot-ocr-v2",
        "infoshot-speech-v2",
        "infoshot-audio-v2",
    )
    assert profile.ocr_missing_categories == ("L26",)
    assert profile.milvus_image_collection == "infoshot-images"
    assert profile.has_glap is False
    assert media.keyframe_url("L26_V001", 1).startswith("https://keyframe.r2.test/")
    assert profile.keyframe_media_fallback_base_url == (
        "https://huggingface.test/bucket/resolve"
    )
    assert media.video_url("L26_V001").startswith("https://media.r2.test/")


def test_infoshot_keyframe_origins_are_loaded_from_env(monkeypatch):
    monkeypatch.setenv("KEYFRAME_MEDIA_BASE_URL_2", "https://keyframe.r2.test/")
    monkeypatch.setenv(
        "KEYFRAME_MEDIA_FALLBACK_BASE_URL_2",
        "https://huggingface.test/bucket/resolve/",
    )
    get_settings.cache_clear()

    profile = get_settings().for_retrieval_database("infoshotpp")

    assert profile.keyframe_media_base_url == "https://keyframe.r2.test"
    assert profile.keyframe_media_fallback_base_url == (
        "https://huggingface.test/bucket/resolve"
    )


def test_video_origin_is_split_per_profile_like_the_keyframe_origin():
    """The HF bucket carries the 873 InfoShot++ videos (L21-L30) only.

    Sending profile 1 there would 404 every K01-K20 video, so each profile keeps
    its own video origin and only InfoShot++ moves to Hugging Face.
    """
    settings = replace(
        configured_settings(),
        video_media_base_url_2="https://hf.test/buckets/team/aic26-media/resolve",
    )
    btc = settings.for_retrieval_database("btc")
    infoshot = settings.for_retrieval_database("infoshotpp")

    assert MediaUrlBuilder(btc.keyframe_media_base_url, btc.video_media_base_url).video_url(
        "K01_V001"
    ) == "https://media.r2.test/Videos/Videos_K01/K01_V001.mp4"
    assert MediaUrlBuilder(
        infoshot.keyframe_media_base_url, infoshot.video_media_base_url
    ).video_url("L26_V001") == (
        "https://hf.test/buckets/team/aic26-media/resolve/Videos/Videos_L26/L26_V001.mp4"
    )


def test_profile_one_video_origin_can_be_moved_to_hugging_face_later():
    """One env var flips BTC over once K01-K20 land in the bucket."""
    settings = replace(configured_settings(), video_media_base_url_1="https://hf.test/resolve")
    btc = settings.for_retrieval_database("btc")

    assert MediaUrlBuilder(btc.keyframe_media_base_url, btc.video_media_base_url).video_url(
        "K01_V001"
    ) == "https://hf.test/resolve/Videos/Videos_K01/K01_V001.mp4"


def test_unknown_profile_is_rejected():
    try:
        configured_settings().for_retrieval_database("wrong")
    except ValueError as exc:
        assert "Unknown retrieval database" in str(exc)
    else:
        raise AssertionError("unknown retrieval profile was accepted")


def test_live_dres_submission_is_locked_off_by_default():
    """Answers are collected in the app's Submission tab and exported as CSV.

    Nothing may reach the evaluation server unless someone sets DRES_ENABLED, so
    a fully-credentialled config still reports DRES as unavailable.
    """
    credentialled = Settings(
        mock_mode=False,
        dres_base_url="http://dres.test",
        dres_username="fourier1",
        dres_password="secret",
    )
    assert credentialled.dres_enabled is False
    assert credentialled.has_dres is False
    assert replace(credentialled, dres_enabled=True).has_dres is True
