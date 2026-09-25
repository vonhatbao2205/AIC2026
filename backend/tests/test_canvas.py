"""V-KIS sketch search: validation of the drawing and the PE image channel."""
from __future__ import annotations

import base64

import pytest

import math
import struct
import zlib

from app.canvas import (
    BLANK_REFERENCE_IMAGES,
    BLANK_REFERENCE_RGB,
    blank_direction,
    parse_canvas_image,
    solid_png_data_url,
    without_blank_direction,
)

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

    assert result["canvas"] == {"has_image": True, "suppress_blank": True}
    assert result["groups"], "a sketch must return candidates in mock mode"
    frames = [frame for group in result["groups"] for frame in group["frames"]]
    assert {channel for frame in frames for channel in frame["channels"]} == {"canvas_image"}
    assert all(frame["frame_idx"] is not None for frame in frames), "hits must be submittable"
    evidence = frames[0]["evidence"][0]
    assert evidence["type"] == "canvas_image"
    assert result["warnings"] == []


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _spy_service(profile):
    """A CanvasService whose PE encoder and Milvus search record what they get."""
    from app.services.canvas_service import CanvasService

    service = CanvasService(profile)
    sent: list[list[str]] = []
    searched: list[tuple[str, list[float]]] = []

    async def encode_image(images):
        sent.append(list(images))
        # Deterministic, distinct unit vectors: the sketch leans on dimension 0,
        # each uniform reference on dimension 0 plus one of its own.
        out = []
        for index, _ in enumerate(images):
            vector = [0.0] * 1280
            vector[0] = 1.0
            vector[1 + index] = 1.0 if index else 0.5
            norm = math.sqrt(_dot(vector, vector))
            out.append([value / norm for value in vector])
        return out

    original = service.search_service.milvus._search_image_collection

    def spy(collection_name, vector, *args, **kwargs):
        searched.append((collection_name, list(vector)))
        return original(collection_name, vector, *args, **kwargs)

    service.search_service.pe.encode_image = encode_image
    service.search_service.milvus._search_image_collection = spy
    return service, sent, searched


@pytest.mark.asyncio
async def test_sketch_query_loses_the_blank_frame_direction_and_caches_the_references(settings):
    profile = settings.for_retrieval_database("infoshotpp")
    service, sent, searched = _spy_service(profile)

    result = await service.search({"canvas": {"image": PNG_1PX}, "top_k": 5})
    await service.search({"canvas": {"image": PNG_1PX}})

    # The uniform references ride in the first request only, then are reused.
    assert sent == [[PNG_1PX, *BLANK_REFERENCE_IMAGES], [PNG_1PX]]
    assert [collection for collection, _ in searched] == [profile.milvus_image_collection] * 2
    direction = service._blank_direction
    for _, vector in searched:
        assert abs(_dot(vector, direction)) < 1e-9, "the blank direction must be projected out"
        assert math.isclose(_dot(vector, vector), 1.0, rel_tol=1e-9)
    assert result["canvas"]["suppress_blank"] is True
    assert sum(len(group["frames"]) for group in result["groups"]) <= 5


@pytest.mark.asyncio
async def test_suppression_off_searches_the_raw_sketch_vector(settings):
    service, sent, searched = _spy_service(settings.for_retrieval_database("infoshotpp"))

    result = await service.search({"canvas": {"image": PNG_1PX}, "suppress_blank": False})

    assert sent == [[PNG_1PX]]
    raw = searched[0][1]
    assert raw[0] > 0.8 and raw[1] > 0.4, "the raw PE vector reaches Milvus unchanged"
    assert result["canvas"] == {"has_image": True, "suppress_blank": False}


class TestBlankSuppression:
    def test_reference_pngs_are_real_one_pixel_images_of_their_colour(self):
        for rgb, url in zip(BLANK_REFERENCE_RGB, BLANK_REFERENCE_IMAGES):
            assert parse_canvas_image(url) == url  # passes the same gate as a sketch
            raw = base64.b64decode(url.split(",", 1)[1])
            width, height = struct.unpack(">II", raw[16:24])
            idat_len = struct.unpack(">I", raw[33:37])[0]
            pixels = zlib.decompress(raw[41:41 + idat_len])
            assert (width, height) == (1, 1)
            assert tuple(pixels[1:4]) == rgb
        assert solid_png_data_url((1, 2, 3)) != solid_png_data_url((3, 2, 1))

    def test_projection_is_orthogonal_and_unit_length(self):
        direction = blank_direction([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert math.isclose(_dot(direction, direction), 1.0)

        query = without_blank_direction([0.6, 0.0, 0.8], direction)

        assert abs(_dot(query, direction)) < 1e-12
        assert math.isclose(_dot(query, query), 1.0)

    def test_partial_strength_keeps_part_of_the_component(self):
        direction = [1.0, 0.0]
        half = without_blank_direction([0.8, 0.6], direction, strength=0.5)
        assert 0 < _dot(half, direction) < 0.8

    def test_a_query_that_is_the_blank_direction_is_returned_unchanged(self):
        direction = [0.0, 1.0]
        assert without_blank_direction([0.0, 2.0], direction) == [0.0, 1.0]

    def test_blank_direction_needs_references(self):
        with pytest.raises(ValueError):
            blank_direction([])


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
