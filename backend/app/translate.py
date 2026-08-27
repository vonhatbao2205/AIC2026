"""Vietnamese→English translation for the visual (PE) query.

This is the fallback used when the LLM parser is OFF, so a Vietnamese query still
becomes an English visual query for the English-centric PE-Core-G14 encoder.

Getting this wrong is expensive and invisible: PE embeds Vietnamese text into a
space it was never trained for, so the operator sees plausible-looking keyframes
that simply do not match the query. That is exactly what happened when Google's
free endpoint started answering `429 Too Many Requests` — every search silently
fell back to the untranslated Vietnamese. Two lessons are baked in here:

1. **More than one provider.** One undocumented free endpoint is not a dependency
   worth betting a competition run on. Google is tried on two different hosts,
   then the NIM model the project already uses for parsing/expansion.
2. **Failure is reported.** `translate_vi_to_en_status` returns whether the text
   actually got translated, so callers can warn instead of quietly searching with
   the wrong query.
"""
from __future__ import annotations

from typing import Any

import httpx

# Chrome's dictionary-extension endpoint. Kept first because the `gtx` client on
# translate.googleapis.com is the one that gets rate-limited in practice.
_CLIENTS5_ENDPOINT = "https://clients5.google.com/translate_a/t"
_GTX_ENDPOINT = "https://translate.googleapis.com/translate_a/single"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_cache: dict[str, str] = {}


def _looks_english(text: str) -> bool:
    """Cheap heuristic: no Vietnamese diacritics and mostly ASCII => treat as EN."""
    vi_marks = "ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ"
    low = text.lower()
    if any(c in low for c in vi_marks):
        return False
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(1, len(text))
    return ascii_ratio > 0.98


async def _via_clients5(client: httpx.AsyncClient, text: str) -> str:
    """`[["translated text", "vi"]]` — one entry per segment."""
    response = await client.get(
        _CLIENTS5_ENDPOINT,
        params={"client": "dict-chrome-ex", "sl": "auto", "tl": "en", "q": text},
        headers={"User-Agent": _USER_AGENT},
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, str):  # some responses collapse to a bare string
        return data.strip()
    parts = [
        segment[0]
        for segment in data
        if isinstance(segment, (list, tuple)) and segment and isinstance(segment[0], str)
    ]
    return "".join(parts).strip()


async def _via_gtx(client: httpx.AsyncClient, text: str) -> str:
    """`[[[translated, original, ...], ...], ...]` — the original endpoint."""
    response = await client.get(
        _GTX_ENDPOINT,
        params={"client": "gtx", "sl": "auto", "tl": "en", "dt": "t", "q": text},
        headers={"User-Agent": _USER_AGENT},
    )
    response.raise_for_status()
    data = response.json()
    return "".join(segment[0] for segment in data[0] if segment and segment[0]).strip()


async def _via_llm(text: str, settings: Any, timeout: float) -> str:
    """Last resort: the NIM model already configured for parsing/expansion.

    Worth the extra second only when both Google hosts are refusing — but it is
    what keeps translation working at all when they are.
    """
    if settings is None or not getattr(settings, "has_llm", False):
        return ""
    body = {
        "model": settings.nvidia_fast_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Translate the user's Vietnamese text to English. Output ONLY the "
                    "English translation, no quotes, no notes, no explanation."
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }
    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=max(20.0, timeout)) as client:
        response = await client.post(
            f"{settings.nvidia_base_url.rstrip('/')}/chat/completions",
            json=body,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip().strip('"')


async def translate_vi_to_en_status(
    text: str, *, timeout: float = 5.0, settings: Any = None
) -> tuple[str, bool]:
    """Translate to English. Returns `(text, translated_ok)`.

    `translated_ok` is False only when the text needed translating and every
    provider failed — the caller then knows the visual query is still Vietnamese
    and that PE results will be poor.
    """
    text = (text or "").strip()
    if not text or _looks_english(text):
        return text, True
    if text in _cache:
        return _cache[text], True

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for provider in (_via_clients5, _via_gtx):
            try:
                translated = await provider(client, text)
            except Exception:  # noqa: BLE001 - try the next provider
                continue
            if translated and translated.casefold() != text.casefold():
                _cache[text] = translated
                return translated, True

    try:
        translated = await _via_llm(text, settings, timeout)
    except Exception:  # noqa: BLE001 - translation is best-effort
        translated = ""
    if translated and translated.casefold() != text.casefold():
        _cache[text] = translated
        return translated, True

    # Deliberately NOT cached: a rate limit is temporary and the next search
    # should try again rather than being pinned to the untranslated text.
    return text, False


async def translate_vi_to_en(text: str, *, timeout: float = 5.0, settings: Any = None) -> str:
    """Translate `text` to English, returning it unchanged if that is impossible."""
    translated, _ = await translate_vi_to_en_status(text, timeout=timeout, settings=settings)
    return translated
