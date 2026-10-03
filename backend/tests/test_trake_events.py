"""TRAKE events are split on the organisers' E1, E2, … markers and nothing else."""
from app.trake_events import marked_events_block, split_marked_events

# Numbers in the prose, as in the statements that were split wrong: every "(4)"
# used to open a new event.
NUMBERED = (
    "Đoạn video múa lân (1) con lân màu vàng, tìm các sự kiện sau:\n"
    "E1: Lân quay vòng trên cột số 4. Có 2 người (4), (5), (6) đứng xem.\n"
    "E2: Khoảnh khắc 4 chân chạm đất (2) lần, giống như ở E1.\n"
    "E3: Khoảnh khắc lân cúi chào (E1) lần (3)."
)


def test_numbers_and_mentions_in_an_event_stay_in_it():
    assert split_marked_events(NUMBERED) == [
        "Lân quay vòng trên cột số 4. Có 2 người (4), (5), (6) đứng xem.",
        "Khoảnh khắc 4 chân chạm đất (2) lần, giống như ở E1.",
        "Khoảnh khắc lân cúi chào (E1) lần (3).",
    ]


def test_a_statement_without_markers_is_not_split_on_its_numbers():
    assert split_marked_events("Có 2 con rồng (1) và (2) con lân, cột số 4. Sau đó (3) người chào.") is None
    assert split_marked_events("1) chạy đà. 2) giậm nhảy. 3) tiếp đất.") is None
    assert split_marked_events("E1: chỉ một sự kiện.") is None


def test_the_markers_the_organisers_write():
    # Colon, space only (a real pack), the event pack joined into one line.
    assert split_marked_events("Cảnh xưởng. E1: Cánh tay rô bốt lắp khung. E2: Công nhân quay tay xoay.") == [
        "Cánh tay rô bốt lắp khung.", "Công nhân quay tay xoay.",
    ]
    assert split_marked_events("Mở đầu.\nE1 Khoảnh khắc đầu tiên.\nE2 Khoảnh khắc thứ hai.") == [
        "Khoảnh khắc đầu tiên.", "Khoảnh khắc thứ hai.",
    ]
    # Typed by hand: the numbers are not checked, the markers are counted.
    assert len(split_marked_events("E1: một. E2: hai. E2: ba. E4: bốn.")) == 4


def test_the_events_are_spelt_out_for_the_models():
    block = marked_events_block(NUMBERED)
    assert "exactly 3" in block
    assert block.endswith("E3: Khoảnh khắc lân cúi chào (E1) lần (3).")
    assert marked_events_block("không có sự kiện nào được đánh dấu (1) (2)") == ""
