import pytest

from app.services.timeline_service import TimelineService


@pytest.mark.asyncio
async def test_timeline_shape(settings):
    svc = TimelineService(settings)
    tl = await svc.build("K01_V001")
    assert tl["video_id"] == "K01_V001"
    assert tl["fps"] == 25.0
    assert set(tl) == {
        "retrieval_database",
        "video_id",
        "fps",
        "duration",
        "video_url",
        "keyframes",
    }
    kf = tl["keyframes"][0]
    assert set(kf) >= {"submit_keyframe_id", "keyframe_n", "frame_idx", "pts_time", "keyframe_url"}
    assert kf["keyframe_url"].startswith("https://media.test/Keyframes/")


@pytest.mark.asyncio
async def test_timeline_echoes_the_profile_that_answered(settings):
    """`/api/videos/{id}/timeline` is profile-scoped via a query parameter, so the
    body has to say which profile actually served it — the two profiles hold
    different keyframe maps and different media origins for the same video id."""
    settings.keyframe_media_base_url_2 = "https://hf.test/resolve"
    settings.video_media_base_url_2 = "https://hf.test/resolve"

    btc = await TimelineService(settings.for_retrieval_database("btc")).build("K01_V001")
    infoshotpp = await TimelineService(
        settings.for_retrieval_database("infoshotpp")
    ).build("K01_V001")

    assert btc["retrieval_database"] == "btc"
    assert infoshotpp["retrieval_database"] == "infoshotpp"
    assert btc["video_url"].startswith("https://media.test/")
    assert infoshotpp["video_url"].startswith("https://hf.test/resolve/")


@pytest.mark.asyncio
async def test_timeline_keyframes_sorted(settings):
    svc = TimelineService(settings)
    tl = await svc.build("K01_V001")
    ns = [k["keyframe_n"] for k in tl["keyframes"]]
    assert ns == sorted(ns)


@pytest.mark.asyncio
async def test_timeline_no_heavy_tracks(settings):
    """OCR/speech/audio/heatmap tracks were dropped for speed."""
    svc = TimelineService(settings)
    tl = await svc.build("K01_V001")
    for removed in ("speech_segments", "ocr_markers", "audio_windows", "heatmap"):
        assert removed not in tl


@pytest.mark.asyncio
async def test_video_keyframes_are_paged_past_the_result_window(settings):
    """An S01 race video has up to 75,123 keyframes, more than one Elastic page."""
    from dataclasses import replace

    from app.adapters.elastic_client import ElasticClient

    client = ElasticClient(replace(settings, mock_mode=False, elastic_endpoint="https://es.test", elastic_api_key="k"))
    total = 7
    bodies = []

    async def fake_search(index, body):
        bodies.append(body)
        start = body.get("search_after", [0])[0]
        stop = min(start + body["size"], total)
        return {"hits": {"hits": [
            {"_source": {"keyframe_n": n}, "sort": [n]} for n in range(start + 1, stop + 1)
        ]}}

    client._search = fake_search
    frames = await client.get_video_keyframes("S01-V007", page_size=3)

    assert [frame["keyframe_n"] for frame in frames] == list(range(1, total + 1))
    assert [body.get("search_after") for body in bodies] == [None, [3], [6]]
