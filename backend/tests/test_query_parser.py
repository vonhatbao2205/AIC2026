from app.query_parser import (
    apply_manual_overrides,
    ensure_image_pe,
    heuristic_parse,
)


def test_heuristic_enables_image_pe_by_default():
    p = heuristic_parse("một người đàn ông đi xe máy dưới mưa")
    assert p["channels"]["image_pe"]["enabled"] is True
    assert p["query_type"] == "T-KIS"


def test_heuristic_ocr_likely_on_digits_and_clock():
    p = heuristic_parse("đồng hồ hiển thị 18:29:57 trên màn hình")
    assert p["channels"]["ocr"]["enabled"] is True
    assert p["channels"]["ocr"]["time_filters"]["hour"] == 18
    assert p["channels"]["ocr"]["time_filters"]["clock"] == "18:29:57"


def test_ocr_fold_matches_ingestion_for_vietnamese_d_stroke():
    p = heuristic_parse("dòng chữ Đại học 2021")
    assert p["channels"]["ocr"]["queries_folded"] == ["dong chu dai hoc 2021"]


def test_ocr_keeps_the_number_and_drops_the_prose_around_it():
    """Regression: a scene description was sent to OCR as an all-token query.

    `search_ocr` ANDs `queries_vi`, so this 28-token sentence asked Elasticsearch
    for a frame containing every word of it — 0 of the 779,995 OCR frames do. The
    only thing actually printed on the answer frames is the speed readout "69".
    """
    p = heuristic_parse(
        "Đoạn video không chuyển cảnh thể hiện phần đua nước rút về đích của các "
        "tay đua. Vận tốc cao nhất đo được đến con số 69 km/h."
    )
    ocr = p["channels"]["ocr"]

    assert ocr["enabled"] is True
    assert ocr["numbers"] == ["69"]
    assert ocr["queries_vi"] == []
    assert ocr["queries_folded"] == []


def test_ocr_keeps_a_short_query_as_literal_screen_text():
    ocr = heuristic_parse('chữ "THPT 2021" trên màn hình')["channels"]["ocr"]
    assert ocr["enabled"] is True
    assert ocr["exact_phrases"] == ["THPT 2021"]


def test_digits_alone_no_longer_wake_the_ocr_channel():
    """Counting words are not screen text.

    "Có 1 người … giữa 2 người" used to route to OCR and put `1` and `2` — the
    two commonest tokens in the whole OCR corpus — into a hard filter. Measured
    against the live index that cost 2-16 s per search on a channel that could
    not contribute anything to the answer.
    """
    scene = heuristic_parse(
        "Có 1 người mặc áo trắng ngồi giữa 2 người mặc áo đen, phía sau là kệ sách"
    )["channels"]["ocr"]
    assert scene["enabled"] is False

    # A bare code+year is no longer enough on its own either; quote it or say so.
    assert heuristic_parse("THPT 2021")["channels"]["ocr"]["enabled"] is False


def test_a_named_number_is_a_cue_but_a_counted_one_is_not():
    named = heuristic_parse("xe mang số 69 về đích")["channels"]["ocr"]
    assert named["enabled"] is True
    assert named["numbers"] == ["69"]

    # "một số" means "some" — the word is there, the meaning is not.
    assert heuristic_parse("một số người đi bộ trên phố")["channels"]["ocr"]["enabled"] is False
    assert heuristic_parse("có 2 con chó chạy trong công viên")["channels"]["ocr"]["enabled"] is False


def test_ocr_quoted_span_wins_over_the_sentence_around_it():
    ocr = heuristic_parse(
        'Người dẫn chương trình đứng trước tấm bảng có dòng chữ "Chào mừng năm học mới" '
        "trong sân trường đông học sinh"
    )["channels"]["ocr"]
    assert ocr["exact_phrases"] == ["Chào mừng năm học mới"]
    assert ocr["queries_vi"] == []


def test_ocr_ignores_a_digit_welded_to_a_code():
    ocr = heuristic_parse("logo VTV3 ở góc màn hình")["channels"]["ocr"]
    assert ocr["numbers"] == []  # the 3 belongs to VTV3, it is not a number to search
    assert ocr["queries_vi"] == ["logo VTV3 ở góc màn hình"]


def test_ocr_stays_off_when_a_cue_word_yields_nothing_searchable():
    """A cue word alone is not evidence: enabling OCR here would cost a request
    and return nothing, because no part of the sentence is printed on a frame."""
    ocr = heuristic_parse(
        "Cảnh quay có một tấm biển hiệu rất mờ ở phía xa trong con hẻm nhỏ vào "
        "buổi tối mà người xem hoàn toàn không đọc được nội dung của nó"
    )["channels"]["ocr"]
    assert ocr["enabled"] is False


def test_heuristic_speech_likely():
    p = heuristic_parse("thủ tướng phát biểu về kinh tế")
    assert p["channels"]["speech"]["enabled"] is True


def test_heuristic_audio_likely_maps_labels():
    p = heuristic_parse("có tiếng piano và tiếng vỗ tay")
    audio = p["channels"]["audio"]
    assert audio["enabled"] is True
    assert "Piano" in audio["sound_labels_en"]
    assert "Applause" in audio["sound_labels_en"]


def test_heuristic_trake_splits_events():
    p = heuristic_parse("người chạy đến cửa sau đó mở cửa rồi bước vào", query_type_hint="auto")
    assert p["query_type"] == "TRAKE"
    assert p["trake"]["enabled"] is True
    assert len(p["trake"]["events"]) >= 2


def test_heuristic_qa_detection():
    p = heuristic_parse("biển số xe màu gì?")
    assert p["query_type"] == "QA"
    assert p["qa"]["enabled"] is True


def test_manual_overrides_force_and_disable():
    p = heuristic_parse("xe máy dưới mưa")
    p = apply_manual_overrides(p, {"force_channels": ["audio"], "disable_channels": ["image_pe"]})
    assert p["channels"]["audio"]["enabled"] is True
    assert p["channels"]["image_pe"]["enabled"] is False


def test_force_channel_populates_query_text():
    # Heuristic does not route "hạ long" to OCR, so ocr.queries_vi is empty.
    p = heuristic_parse("hạ long")
    assert p["channels"]["ocr"]["queries_vi"] == []
    # Forcing OCR must fill the query so it searches the text, not match-all.
    p = apply_manual_overrides(p, {"force_channels": ["ocr"]})
    assert p["channels"]["ocr"]["enabled"] is True
    assert p["channels"]["ocr"]["queries_vi"] == ["hạ long"]
    assert p["channels"]["ocr"]["queries_folded"] == ["ha long"]


def test_force_speech_and_audio_populate_query():
    p = heuristic_parse("hạ long")
    p = apply_manual_overrides(p, {"force_channels": ["speech", "audio"]})
    assert p["channels"]["speech"]["queries_vi"] == ["hạ long"]
    assert p["channels"]["audio"]["queries_en"] == ["hạ long"]


def test_ensure_image_pe_when_all_disabled():
    p = heuristic_parse("xe máy")
    for c in p["channels"].values():
        c["enabled"] = False
    p = ensure_image_pe(p)
    assert p["channels"]["image_pe"]["enabled"] is True
    assert p["channels"]["image_pe"]["queries_en"]
