"""PE-Core's 70-token text window: exact counting, the typing-time estimate, the reports."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pe_tokens import (
    PE_TEXT_TOKENS,
    count_pe_tokens,
    estimate_translated_tokens,
    live_pe_report,
    pe_query_report,
)
from app.services.search_service import SearchService

client = TestClient(app)

#: Counts from the official `perception_models` SimpleTokenizer (with ftfy and
#: regex) on the same strings; the port must agree on every one.
OFFICIAL_COUNTS = [
    ("A man in a blue shirt riding a motorbike past a Highlands Coffee shop at 7 pm.", 18),
    ("Một người đàn ông mặc áo xanh đi xe máy ngang qua ngã tư Nguyễn Trãi - Cống Quỳnh lúc 19:05", 70),
    ("mot nguoi dan ong mac ao xanh", 10),
    ("It's the camera's view; they're waiting, we'll see — “quoted” ‘text’ and 69 km/h!", 28),
    ("The Thing superhero, orange rocky stone-skinned muscular man", 11),
    (
        "Đoạn video không chuyển cảnh, ghi lại cảnh một tay đua xe đạp mặc áo vàng đang dẫn đầu đoàn đua "
        "trên đường phố, đồng hồ tốc độ hiển thị 69 km/h và quãng đường còn lại 12,5 km. Phía sau là xe hỗ "
        "trợ kỹ thuật màu trắng.",
        192,
    ),
    ("   multiple    spaces\tand\nnewlines &amp; html &lt;tags&gt; 2026-06-15 ①②③ ½ Ⅻ", 33),
    ("日本語のテキスト and emoji 🚗🚦 mixed", 16),
    ("ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểễệỉịọỏốồổỗộớờởỡợụủứừ", 203),
]


@pytest.mark.parametrize(("text", "tokens"), OFFICIAL_COUNTS)
def test_counts_match_the_official_tokenizer(text, tokens):
    assert count_pe_tokens(text).tokens == tokens


def test_the_cut_falls_on_a_word_boundary():
    report = count_pe_tokens("cat " * 75)
    assert report.tokens == 75 and report.truncated
    assert report.kept == " ".join(["cat"] * PE_TEXT_TOKENS)
    assert report.dropped == " ".join(["cat"] * 5)
    exactly = count_pe_tokens("cat " * PE_TEXT_TOKENS)
    assert not exactly.truncated and exactly.dropped == ""


@pytest.mark.parametrize(
    ("words", "status"),
    [(100, "over"), (70, "may_exceed"), (55, "near"), (30, "ok")],
)
def test_translation_estimate_flags_by_calibrated_range(words, status):
    report = live_pe_report(" ".join(["người"] * words), translated=True)
    assert report["status"] == status and report["exact"] is False
    low, high = report["range"]
    assert low <= report["tokens"] <= high


def test_estimate_points_at_the_text_at_risk():
    report = estimate_translated_tokens(" ".join(f"từ{i}" for i in range(40)) + " cuối cùng là đoạn này")
    # "từ12" is one letter piece and one digit piece per digit: ~3 pieces a word.
    assert report["dropped"].endswith("cuối cùng là đoạn này")
    assert report["at_risk"].endswith("cuối cùng là đoạn này")


def test_exact_when_pe_reads_the_text_as_typed():
    report = live_pe_report("a man in a blue shirt", translated=False)
    assert report == {
        "context_length": 72, "limit": 70, "exact": True, "text": "a man in a blue shirt", "tokens": 6,
        "truncated": False, "kept": "a man in a blue shirt", "dropped": "", "range": None, "at_risk": "", "status": "ok",
    }
    assert live_pe_report("  ", translated=True)["status"] == "empty"


def test_endpoint_joins_hints_like_the_search():
    # Mock mode never translates, so PE would read the Vietnamese itself.
    body = client.post("/api/pe/tokens", json={"query": "mặc áo xanh", "previous_hints": ["một người"], "translate": True}).json()
    assert body["exact"] is True
    assert body["text"] == "một người mặc áo xanh"
    assert body["tokens"] == count_pe_tokens("một người mặc áo xanh").tokens


def test_query_report_counts_every_pe_query():
    report = pe_query_report(["short query", "word " * 80, ""])
    assert [q["tokens"] for q in report["queries"]] == [2, 80]
    assert report["truncated"] == 1 and report["limit"] == 70
    assert SearchService.pe_tokens(["word " * 80], ("qwen3_vl",)) is None


@pytest.mark.asyncio
async def test_search_reports_what_pe_read(settings):
    service = SearchService(settings)
    long_query = "một người đàn ông mặc áo xanh đứng cạnh chiếc xe máy màu đỏ " * 6
    result = await service.search({"query": long_query, "image_models": ["pe"]})
    report = result["pe_tokens"]
    assert report["truncated"] == 1
    assert report["queries"][0]["tokens"] > PE_TEXT_TOKENS and report["queries"][0]["dropped"]
