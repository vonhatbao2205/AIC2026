"""Vietnamese→English translation for the visual (PE) query.

Mirrors the `translate_to_english` helper from ui-streamlit.ipynb (which used
googletrans), but hits Google's free translate endpoint directly via httpx so we
avoid the flaky googletrans dependency. Always degrades to the original text on
any error, and caches results in-process.

This is the fallback used when the LLM parser is OFF, so a Vietnamese query still
becomes an English visual query for the English-centric PE-Core-G14 encoder.
"""
from __future__ import annotations

import httpx

_ENDPOINT = "https://translate.googleapis.com/translate_a/single"
_cache: dict[str, str] = {}


def _looks_english(text: str) -> bool:
    """Cheap heuristic: no Vietnamese diacritics and mostly ASCII => treat as EN."""
    vi_marks = "ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ"
    low = text.lower()
    if any(c in low for c in vi_marks):
        return False
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(1, len(text))
    return ascii_ratio > 0.98


async def translate_vi_to_en(text: str, *, timeout: float = 5.0) -> str:
    """Translate `text` to English. Returns the original text unchanged on any
    failure or if it already looks English."""
    text = (text or "").strip()
    if not text or _looks_english(text):
        return text
    if text in _cache:
        return _cache[text]
    try:
        params = {"client": "gtx", "sl": "auto", "tl": "en", "dt": "t", "q": text}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(_ENDPOINT, params=params)
            resp.raise_for_status()
            data = resp.json()
        # data[0] is a list of [translated_segment, original_segment, ...]
        translated = "".join(seg[0] for seg in data[0] if seg and seg[0]).strip()
        result = translated or text
    except Exception:  # noqa: BLE001 - translation is best-effort
        result = text
    _cache[text] = result
    return result
