"""V-KIS sketch search: validation of the drawing and the PE image channel."""
from __future__ import annotations

import base64

import pytest

from app.canvas import parse_canvas_image

PNG_1PX = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
JPEG_HEAD = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 16).decode()


class TestParseCanvasImage:
    def test_inline_png_and_jpeg_are_accepted(self):
        assert parse_canvas_image(PNG_1PX) == PNG_1PX
        assert parse_canvas_image(f"  {PNG_1PX}\n") == PNG_1PX
        assert parse_canvas_image(JPEG_HEAD) == JPEG_HEAD

    def test_anything_that_is_not_inline_image_bytes_is_rejected(self):
        # The string is forwarded to the PE server, so a URL here would turn that
        # server into a fetcher for whatever the browser asked for.
        assert parse_canvas_image("https://evil.example/x.png") is None
        assert parse_canvas_image("data:text/html;base64,PHNjcmlwdD4=") is None
        assert parse_canvas_image(None) is None
        assert parse_canvas_image("") is None

    def test_payload_must_decode_to_the_declared_format(self):
        assert parse_canvas_image("data:image/png;base64,not base64!!") is None
        # A JPEG body labelled as PNG is a malformed request, not a drawing.
        assert parse_canvas_image("data:image/png;base64," + JPEG_HEAD.split(",", 1)[1]) is None


@pytest.mark.asyncio
async def test_sketch_search_returns_grouped_canvas_image_hits(settings):
    from app.services.canvas_service import CanvasService

    result = await CanvasService(settings).search({"canvas": {"image": PNG_1PX}})

    assert result["canvas"] == {"has_image": True}
    assert result["groups"], "a sketch must return candidates in mock mode"
    frames = [frame for group in result["groups"] for frame in group["frames"]]
    assert {channel for frame in frames for channel in frame["channels"]} == {"canvas_image"}
    assert all(frame["frame_idx"] is not None for frame in frames), "hits must be submittable"
    evidence = frames[0]["evidence"][0]
    assert evidence["type"] == "canvas_image"
    assert result["warnings"] == []


@pytest.mark.asyncio
async def test_sketch_encodes_the_drawing_once_and_searches_the_profile_pe_collection(settings):
    from app.services.canvas_service import CanvasService

    profile = settings.for_retrieval_database("infoshotpp")
    service = CanvasService(profile)
    sent: list[list[str]] = []
    searched: list[str] = []

    async def encode_image(images):
        sent.append(images)
        return [[1.0] + [0.0] * 1279]

    original = service.search_service.milvus._search_image_collection

    def spy(collection_name, *args, **kwargs):
        searched.append(collection_name)
        return original(collection_name, *args, **kwargs)

    service.search_service.pe.encode_image = encode_image
    service.search_service.milvus._search_image_collection = spy
    result = await service.search({"canvas": {"image": PNG_1PX}, "top_k": 5})

    assert sent == [[PNG_1PX]]
    assert searched == [profile.milvus_image_collection]
    assert sum(len(group["frames"]) for group in result["groups"]) <= 5


@pytest.mark.asyncio
async def test_manual_scope_narrows_a_sketch(settings):
    from app.services.canvas_service import CanvasService

    result = await CanvasService(settings).search(
        {"canvas": {"image": PNG_1PX}, "scope": {"mode": "manual", "categories": ["K01"]}}
    )

    assert result["groups"]
    assert all(group["video_id"].startswith("K01") for group in result["groups"])


@pytest.mark.asyncio
async def test_empty_or_invalid_sketch_returns_a_warning_not_an_error(settings):
    from app.services.canvas_service import CanvasService

    service = CanvasService(settings)

    async def fail(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("encode_image called without a valid drawing")

    service.search_service.pe.encode_image = fail
    for canvas in ({}, {"image": "https://evil.example/x.png"}):
        result = await service.search({"canvas": canvas})
        assert result["groups"] == []
        assert result["canvas"] == {"has_image": False}
        assert result["warnings"]


@pytest.mark.asyncio
async def test_missing_encode_image_route_is_reported_as_a_warning(settings):
    from app.adapters.pe_encoder import PeImageEncoderMissing
    from app.services.canvas_service import CanvasService

    service = CanvasService(settings)

    async def missing(*args, **kwargs):
        raise PeImageEncoderMissing("PE server has no /encode-image endpoint")

    service.search_service.pe.encode_image = missing
    result = await service.search({"canvas": {"image": PNG_1PX}})

    assert result["groups"] == []
    assert any("/encode-image" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_a_dead_encoder_is_a_warning_not_a_500(settings):
    from app.services.canvas_service import CanvasService

    service = CanvasService(settings)

    async def down(*args, **kwargs):
        raise ConnectionError("tunnel closed")

    service.search_service.pe.encode_image = down
    result = await service.search({"canvas": {"image": PNG_1PX}})

    assert result["groups"] == []
    assert any("tunnel closed" in warning for warning in result["warnings"])
