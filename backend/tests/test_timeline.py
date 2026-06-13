import pytest

from app.services.timeline_service import TimelineService


@pytest.mark.asyncio
async def test_timeline_shape(settings):
    svc = TimelineService(settings)
    tl = await svc.build("K01_V001")
    assert tl["video_id"] == "K01_V001"
    assert tl["fps"] == 25.0
    assert set(tl) == {"video_id", "fps", "duration", "video_url", "keyframes"}
    kf = tl["keyframes"][0]
    assert set(kf) >= {"submit_keyframe_id", "keyframe_n", "frame_idx", "pts_time", "keyframe_url"}
    assert kf["keyframe_url"].startswith("https://media.test/Keyframes/")


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
