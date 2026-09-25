"""V-KIS sketch query: the operator's drawing, validated before it reaches PE.

For V-KIS the operator redraws the clip from memory — sky, road, a red car, a
person on the left — and the drawing itself is the query: it is flattened onto
its background in the browser, sent as one PNG, encoded by PE-Core-G14's image
tower and searched against the profile's PE keyframe collection. There is no
object detection behind it: layout, dominant colours and rough shapes are what a
sketch carries, and PE's image space is where keyframes with that composition
already live.

The PE server squashes every input to 448×448 (no centre crop), exactly as it did
for the indexed keyframes, so the 16:9 canvas maps onto 16:9 keyframes the same
way. It also drops alpha with `convert("RGB")`, which turns transparent pixels
black — the browser therefore sends an opaque, flattened image.

A sketch is mostly large flat areas, and to PE a flat image first means "a blank
frame": measured on the InfoShot++ index, a solid-green sketch returned near-white
title cards and fades (0/10 green frames in the top 10). The query is therefore
stripped of the direction PE gives uniform images before searching (see
`without_blank_direction`), which leaves the colour and layout the operator drew.

Everything here is pure so it can be unit-tested without the encoder.
"""
from __future__ import annotations

import base64
import binascii
import math
import struct
import zlib

_PREFIXES: dict[str, bytes] = {
    "data:image/png;base64,": b"\x89PNG\r\n\x1a\n",
    "data:image/jpeg;base64,": b"\xff\xd8\xff",
}


def parse_canvas_image(value: object) -> str | None:
    """The drawing as an inline PNG/JPEG data URL, or None when it is not one.

    The string is forwarded to the PE server, so anything that is not
    self-contained image bytes — an http(s) URL above all — is rejected here
    rather than letting the encoder fetch an arbitrary host. The payload must
    also decode and start with the format's magic bytes: a malformed body would
    otherwise surface as an opaque 500 from the GPU worker.
    """
    text = str(value or "").strip()
    for prefix, magic in _PREFIXES.items():
        if not text.startswith(prefix):
            continue
        try:
            raw = base64.b64decode(text[len(prefix):], validate=True)
        except (binascii.Error, ValueError):
            return None
        return text if raw.startswith(magic) else None
    return None


# ---- blank-frame suppression --------------------------------------------

#: Uniform images whose PE embeddings span "a blank frame". Their mean is the
#: direction removed from a sketch query.
BLANK_REFERENCE_RGB: tuple[tuple[int, int, int], ...] = ((255, 255, 255), (0, 0, 0), (128, 128, 128))

#: How much of that direction to remove (1.0 = all of it). Chosen on 8 synthetic
#: sketches against the live InfoShot++ index, scoring each top-10 keyframe by
#: how close its 4x4 colour layout is to the drawing: 0.5 / 0.75 / 1.0 improved
#: 4 / 5 / 7 of the 8, and the median distance fell from 184 to 108 at 1.0. The
#: one loss was an object drawn on the default white canvas, whose raw hits were
#: white frames; the operator can turn suppression off for a genuinely plain scene.
BLANK_SUPPRESSION = 1.0


def solid_png_data_url(rgb: tuple[int, int, int]) -> str:
    """A 1x1 PNG of one colour, as a data URL.

    PE squashes every input to 448x448 before encoding, so one pixel encodes
    exactly like a full canvas of that colour (cosine 1.0 on the live server)
    at a fraction of the payload, and without an imaging library here.
    """
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00" + bytes(rgb)))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


BLANK_REFERENCE_IMAGES: tuple[str, ...] = tuple(solid_png_data_url(rgb) for rgb in BLANK_REFERENCE_RGB)


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm > 0 else list(vector)


def blank_direction(reference_vectors: list[list[float]]) -> list[float]:
    """Unit mean of the uniform-image embeddings."""
    if not reference_vectors:
        raise ValueError("no reference vectors")
    dim = len(reference_vectors[0])
    return _unit([sum(vector[i] for vector in reference_vectors) / len(reference_vectors) for i in range(dim)])


def without_blank_direction(
    query: list[float], direction: list[float], strength: float = BLANK_SUPPRESSION
) -> list[float]:
    """The query with `strength` of its component along `direction` removed, re-normalized.

    Ranking by the result scores a frame by `q·x − s(q·b)(b·x)`: frames that look
    like a blank image lose exactly the similarity they owed to looking blank,
    and nothing else moves. A query that IS the blank direction (nothing left
    after the projection) is returned unchanged rather than amplified noise.
    """
    dot = sum(q * b for q, b in zip(query, direction))
    projected = [q - strength * dot * b for q, b in zip(query, direction)]
    if math.sqrt(sum(value * value for value in projected)) < 1e-3:
        return _unit(query)
    return _unit(projected)
