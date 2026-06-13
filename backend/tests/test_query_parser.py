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
