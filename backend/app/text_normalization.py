"""Text normalization shared by query-time retrieval code."""
from __future__ import annotations

import unicodedata


def fold_vietnamese(text: str) -> str:
    """Lowercase Vietnamese text and remove diacritics, including đ/Đ.

    Unicode NFD removes combining tone/letter marks but deliberately leaves
    ``đ`` intact.  OCR preprocessing converts it to ``d``, so queries must do
    the same or the folded index silently misses words such as "đại".
    """
    decomposed = unicodedata.normalize("NFD", text or "")
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return unicodedata.normalize("NFC", without_marks.replace("đ", "d").replace("Đ", "D")).lower()
