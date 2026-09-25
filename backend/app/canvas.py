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

Everything here is pure so it can be unit-tested without the encoder.
"""
from __future__ import annotations

import base64
import binascii

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
