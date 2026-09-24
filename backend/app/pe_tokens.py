"""Count PE-Core text tokens exactly as the PE text tower does.

PE-Core-G14-448 reads a query through the OpenAI CLIP BPE tokenizer with a
context of 72 tokens: <start_of_text>, at most 70 BPE tokens, <end_of_text>.
Anything past that is cut without an error (`SimpleTokenizer.__call__`
truncates and writes <end_of_text> into the last slot), so the tail of a long
description never reaches the vector — and the operator never finds out.

This is a dependency-free port of `perception_models/core/vision_encoder/
tokenizer.py` (OpenAI CLIP, MIT) for counting: the same vocabulary file
(pinned below), the same split pattern and the same byte-level BPE. Two
pieces of the original need third-party packages and are replaced:

* `regex`'s ``\\p{L}`` / ``\\p{N}`` classes -> `unicodedata` categories, which
  is what those classes are;
* `ftfy.fix_text` -> NFC plus straight quotes, which is all it changes on
  typed text (its mojibake repair has nothing to repair in a search box).
"""
from __future__ import annotations

import gzip
import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

#: `model.context_length` of facebook/PE-Core-G14-448 (model-setup-backend.ipynb).
PE_CONTEXT_LENGTH = 72
#: What a query can use: the context minus <start_of_text> and <end_of_text>.
PE_TEXT_TOKENS = PE_CONTEXT_LENGTH - 2

VOCAB_PATH = Path(__file__).with_name("pe_bpe_simple_vocab_16e6.txt.gz")
VOCAB_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
_SPECIAL_TOKENS = ("<start_of_text>", "<end_of_text>")
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"', "‟": '"'})
_WHITESPACE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _byte_encoder() -> dict[int, str]:
    """CLIP's reversible byte -> printable unicode table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


@lru_cache(maxsize=1)
def _bpe_ranks() -> dict[tuple[str, str], int]:
    data = VOCAB_PATH.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != VOCAB_SHA256:
        raise RuntimeError(f"{VOCAB_PATH.name} is not the CLIP vocabulary PE uses (sha256 {digest[:12]}…)")
    merges = gzip.decompress(data).decode("utf-8").split("\n")[1 : 49152 - 256 - 2 + 1]
    return {tuple(merge.split()): rank for rank, merge in enumerate(merges)}


@lru_cache(maxsize=20_000)
def _bpe(token: str) -> tuple[str, ...]:
    """Byte-level BPE of one pre-token; returns its sub-word pieces."""
    ranks = _bpe_ranks()
    word = tuple(token[:-1]) + (token[-1] + "</w>",)
    while len(word) > 1:
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        bigram = min(pairs, key=lambda pair: ranks.get(pair, float("inf")))
        if bigram not in ranks:
            break
        first, second = bigram
        merged: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                merged.append(first + second)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        word = tuple(merged)
    return word


def clean(text: str) -> str:
    """The tokenizer's "lower" cleaning: fix, unescape twice, collapse spaces, lower-case."""
    text = unicodedata.normalize("NFC", text or "").translate(_QUOTES)
    text = html.unescape(html.unescape(text)).strip()
    return _WHITESPACE.sub(" ", text).strip().lower()


def _kind(char: str) -> str:
    category = unicodedata.category(char)
    if category[0] == "L":
        return "letter"
    if category[0] == "N":
        return "number"
    return "space" if char.isspace() else "other"


def pre_tokens(text: str) -> list[tuple[int, str]]:
    """`(offset, piece)` for the CLIP split pattern over already-cleaned text:
    special tokens | contractions | letters+ | one number | other+ ."""
    pieces: list[tuple[int, str]] = []
    i, n = 0, len(text)
    while i < n:
        kind = _kind(text[i])
        if kind == "space":
            i += 1
            continue
        special = next((s for s in _SPECIAL_TOKENS if text.startswith(s, i)), None)
        contraction = next((c for c in _CONTRACTIONS if text.startswith(c, i)), None)
        if special or contraction:
            match = special or contraction
            pieces.append((i, match))
            i += len(match)
            continue
        j = i + 1
        if kind == "letter":
            while j < n and _kind(text[j]) == "letter":
                j += 1
        elif kind == "other":
            while j < n and _kind(text[j]) == "other":
                j += 1
        pieces.append((i, text[i:j]))
        i = j
    return pieces


def _token_count(piece: str) -> int:
    if piece in _SPECIAL_TOKENS:
        return 1
    encoder = _byte_encoder()
    return len(_bpe("".join(encoder[b] for b in piece.encode("utf-8"))))


@dataclass(frozen=True)
class PeTokenReport:
    """How much of one text PE reads."""

    text: str  # what was counted, as the tokenizer sees it (cleaned, lower-case)
    tokens: int  # BPE tokens, without <start_of_text>/<end_of_text>
    kept: str  # the part PE reads
    dropped: str  # the part cut off (empty when it fits)

    @property
    def limit(self) -> int:
        return PE_TEXT_TOKENS

    @property
    def truncated(self) -> bool:
        return self.tokens > PE_TEXT_TOKENS

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": self.tokens,
            "limit": PE_TEXT_TOKENS,
            "truncated": self.truncated,
            "kept": self.kept,
            "dropped": self.dropped,
        }


def count_pe_tokens(text: str) -> PeTokenReport:
    """Tokens PE-Core spends on `text`, and where its 70-token window ends.

    The split point is reported on word boundaries: a word whose tokens straddle
    the limit is counted as dropped, since PE only reads its first fragment.
    """
    cleaned = clean(text)
    total = 0
    cut: int | None = None
    for offset, piece in pre_tokens(cleaned):
        count = _token_count(piece)
        if cut is None and total + count > PE_TEXT_TOKENS:
            cut = offset
        total += count
    if cut is None:
        return PeTokenReport(cleaned, total, cleaned, "")
    return PeTokenReport(cleaned, total, cleaned[:cut].rstrip(), cleaned[cut:].strip())


def pe_query_report(queries: list[str]) -> dict[str, Any]:
    """The token report of every query sent to the PE text encoder."""
    reports = [count_pe_tokens(query).to_dict() for query in queries if query and query.strip()]
    return {
        "context_length": PE_CONTEXT_LENGTH,
        "limit": PE_TEXT_TOKENS,
        "queries": reports,
        "truncated": sum(1 for report in reports if report["truncated"]),
    }


# ---- before translation -----------------------------------------------
#: English PE tokens per CLIP pre-token of a Vietnamese query once it has been
#: translated (p05, median, p95). A word, a digit and a punctuation run each end
#: up as about one English token. Measured on 120 organiser queries from
#: TKIS/QA/TRAKE_queries.xlsx through the backend's own translator: 41 of them
#: were still over the limit in English, and every one of those had >= 45
#: pieces, which the p95 edge below flags.
TRANSLATED_TOKENS_PER_PIECE = (0.75, 0.90, 1.18)
#: At or above this share of the window a query is close enough to warn.
NEAR_LIMIT = 60


def estimate_translated_tokens(text: str) -> dict[str, Any]:
    """PE tokens a Vietnamese query will cost AFTER VI->EN translation, estimated.

    The console asks this on every pause in typing, and translating each pause
    through the free endpoints the search itself relies on would get them
    rate-limited. The estimate needs no network; the search reports the exact
    count of what it actually sent (`pe_query_report`).
    """
    cleaned = clean(text)
    pieces = pre_tokens(cleaned)
    low, mid, high = (round(len(pieces) * ratio) for ratio in TRANSLATED_TOKENS_PER_PIECE)

    def tail_from(ratio: float) -> tuple[str, str]:
        """Split where `ratio` tokens per piece would fill the window."""
        first_out = int(PE_TEXT_TOKENS / ratio)
        if len(pieces) <= first_out:
            return cleaned, ""
        cut = pieces[first_out][0]
        return cleaned[:cut].rstrip(), cleaned[cut:].strip()

    # Where PE probably stops (median), and what is at risk if the translation
    # comes out long (p95).
    kept, dropped = tail_from(TRANSLATED_TOKENS_PER_PIECE[1])
    _, at_risk = tail_from(TRANSLATED_TOKENS_PER_PIECE[2])
    return {
        "text": cleaned,
        "tokens": mid,
        "range": [low, high],
        "kept": kept,
        "dropped": dropped,
        "at_risk": at_risk,
    }


def live_pe_report(text: str, *, translated: bool) -> dict[str, Any]:
    """What the search box shows while typing: exact when PE will read the text as
    typed, estimated when the search will translate it first."""
    base = {"context_length": PE_CONTEXT_LENGTH, "limit": PE_TEXT_TOKENS, "exact": not translated}
    if not (text or "").strip():
        return {
            **base, "text": "", "tokens": 0, "range": None, "kept": "", "dropped": "", "at_risk": "", "status": "empty",
        }
    if translated:
        report = estimate_translated_tokens(text)
        low, high = report["range"]
        if low > PE_TEXT_TOKENS:
            status = "over"
        elif high > PE_TEXT_TOKENS:
            status = "may_exceed"
        else:
            status = "near" if high >= NEAR_LIMIT else "ok"
        return {**base, **report, "status": status}
    exact = count_pe_tokens(text)
    if exact.truncated:
        status = "over"
    else:
        status = "near" if exact.tokens >= NEAR_LIMIT else "ok"
    return {**base, **exact.to_dict(), "range": None, "at_risk": "", "status": status}
