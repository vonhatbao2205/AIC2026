from app.config import Settings
from app.media import MediaUrlBuilder


def configured_settings() -> Settings:
    return Settings(
        milvus_endpoint_1="https://btc.milvus.test",
        milvus_token_1="btc-token",
        milvus_endpoint_2="https://infoshot.milvus.test",
        milvus_token_2="infoshot-token",
        media_base_url="https://media.r2.test",
        keyframe_media_base_url_2="https://huggingface.test/bucket/resolve",
        idx_keyframe_map_1="btc-map",
        idx_keyframe_map_2="infoshot-map",
        milvus_image_collection_1="btc-images",
        milvus_image_collection_2="infoshot-images",
    )


def test_btc_profile_keeps_full_dataset_endpoints():
    profile = configured_settings().for_retrieval_database("btc")

    assert profile.milvus_endpoint == "https://btc.milvus.test"
    assert profile.idx_keyframe_map == "btc-map"
    assert profile.milvus_image_collection == "btc-images"
    assert profile.keyframe_media_base_url == "https://media.r2.test"
    assert profile.has_glap is False  # no encoder configured in this fixture


def test_infoshot_profile_splits_hf_keyframes_from_r2_video():
    profile = configured_settings().for_retrieval_database("infoshotpp")
    media = MediaUrlBuilder(profile.keyframe_media_base_url, profile.media_base_url)

    assert profile.milvus_endpoint == "https://infoshot.milvus.test"
    assert profile.idx_keyframe_map == "infoshot-map"
    assert profile.milvus_image_collection == "infoshot-images"
    assert profile.has_glap is False
    assert media.keyframe_url("L26_V001", 1).startswith("https://huggingface.test/")
    assert media.video_url("L26_V001").startswith("https://media.r2.test/")


def test_unknown_profile_is_rejected():
    try:
        configured_settings().for_retrieval_database("wrong")
    except ValueError as exc:
        assert "Unknown retrieval database" in str(exc)
    else:
        raise AssertionError("unknown retrieval profile was accepted")
