import pytest

from app.services.submit_service import (
    DuplicateSubmitError,
    SubmitFormatError,
    SubmitService,
    build_answer_sets,
    resolve_answer_mode,
)


@pytest.fixture
def submit_service(settings, tmp_path):
    return SubmitService(settings, history_path=tmp_path / "history.json")


@pytest.mark.asyncio
async def test_submit_stores_local_history(submit_service):
    entry = await submit_service.submit(
        task_id="q001",
        query_type="T-KIS",
        payload={"video_id": "K01_V001", "frame_idx": 150, "timestamp": 6.0},
    )
    assert entry["status"] == "local"  # no DRES in mock mode
    assert entry["task_id"] == "q001"
    assert entry["dedup_keys"] == ["K01_V001:150"]
    assert submit_service.history("q001")


@pytest.mark.asyncio
async def test_duplicate_submit_blocked_same_video_frame(submit_service):
    payload = {"video_id": "K01_V001", "frame_idx": 150, "timestamp": 6.0}
    await submit_service.submit(task_id="q001", query_type="T-KIS", payload=payload)
    with pytest.raises(DuplicateSubmitError):
        await submit_service.submit(task_id="q001", query_type="T-KIS", payload=payload)


@pytest.mark.asyncio
async def test_different_frame_idx_not_duplicate(submit_service):
    await submit_service.submit(
        task_id="q001", query_type="T-KIS", payload={"video_id": "K01_V001", "frame_idx": 150, "timestamp": 6.0}
    )
    entry = await submit_service.submit(
        task_id="q001", query_type="T-KIS", payload={"video_id": "K01_V001", "frame_idx": 151, "timestamp": 6.04}
    )
    assert entry["status"] == "local"


@pytest.mark.asyncio
async def test_duplicate_allowed_when_flag_set(submit_service):
    payload = {"video_id": "K01_V001", "frame_idx": 150, "timestamp": 6.0}
    await submit_service.submit(task_id="q001", query_type="T-KIS", payload=payload)
    entry = await submit_service.submit(
        task_id="q001", query_type="T-KIS", payload=payload, allow_duplicate=True
    )
    assert entry["was_duplicate"] is True


@pytest.mark.asyncio
async def test_same_frame_different_task_not_duplicate(submit_service):
    payload = {"video_id": "K01_V001", "frame_idx": 150, "timestamp": 6.0}
    await submit_service.submit(task_id="q001", query_type="T-KIS", payload=payload)
    entry = await submit_service.submit(task_id="q002", query_type="T-KIS", payload=payload)
    assert entry["status"] == "local"


@pytest.mark.asyncio
async def test_trake_sequence_submit_and_dedup(submit_service):
    payload = {
        "video_id": "L21_V029",
        "events": [
            {"event_index": 1, "frame_idx": 100, "pts_time": 4.0},
            {"event_index": 2, "frame_idx": 11232, "pts_time": 374.4},
        ],
    }
    entry = await submit_service.submit(task_id="q004", query_type="TRAKE", payload=payload)
    assert entry["dedup_keys"] == ["L21_V029|100,11232"]
    with pytest.raises(DuplicateSubmitError):
        await submit_service.submit(task_id="q004", query_type="TRAKE", payload=payload)


@pytest.mark.asyncio
async def test_qa_dedup_covers_the_segment_and_the_text(submit_service):
    """A QA answer is (segment, text): only the exact pair again is wasted."""
    base = {"video_id": "L02_V003", "frame_idx": 888, "timestamp": 35.5, "answer": "15 người"}
    await submit_service.submit(task_id="q002", query_type="QA", payload=base)

    # New answer on the same frame → a different guess.
    entry = await submit_service.submit(
        task_id="q002", query_type="QA", payload={**base, "answer": "16 người"}
    )
    assert entry["status"] == "local"
    # Same answer pinned to another segment → also a different guess.
    entry = await submit_service.submit(
        task_id="q002", query_type="QA",
        payload={**base, "frame_idx": 1200, "timestamp": 48.0},
    )
    assert entry["status"] == "local"
    with pytest.raises(DuplicateSubmitError):
        # Same segment AND same text (case/space-insensitive) → blocked.
        await submit_service.submit(
            task_id="q002", query_type="QA", payload={**base, "answer": "15  NGƯỜI "}
        )


# ---- DRES v2 answer format -------------------------------------------------


def test_kis_answer_is_media_item_with_ms_window():
    sets = build_answer_sets(
        task_name="tkis-00",
        query_type="T-KIS",
        payload={"video_id": "L21_V029", "frame_idx": 11232, "timestamp": 374.4},
        pad_ms=500,
    )
    assert sets == [
        {
            "taskName": "tkis-00",
            "answers": [{"mediaItemName": "L21_V029", "start": 373900, "end": 374900}],
        }
    ]


def test_kis_answer_never_carries_text():
    """DRES reads `text` first: a KIS answer with text would be judged as text."""
    answers = build_answer_sets(
        task_name="tkis-00",
        query_type="T-KIS",
        payload={"video_id": "L21_V029", "timestamp": 10.0, "answer": "leftover"},
        pad_ms=0,
    )[0]["answers"]
    assert answers == [{"mediaItemName": "L21_V029", "start": 10000, "end": 10000}]


def test_media_item_name_drops_the_file_extension():
    answers = build_answer_sets(
        task_name="t", query_type="V-KIS",
        payload={"video_id": "Videos_L30/L30_V095.mp4", "timestamp": 1.0},
    )[0]["answers"]
    assert answers[0]["mediaItemName"] == "L30_V095"


def test_ms_window_falls_back_to_frame_idx_and_fps():
    answers = build_answer_sets(
        task_name="t", query_type="T-KIS",
        payload={"video_id": "L21_V029", "frame_idx": 250, "fps": 25.0}, pad_ms=0,
    )[0]["answers"]
    assert answers[0] == {"mediaItemName": "L21_V029", "start": 10000, "end": 10000}


def test_explicit_ms_window_wins_and_is_ordered():
    answers = build_answer_sets(
        task_name="t", query_type="T-KIS",
        payload={"video_id": "L21_V029", "timestamp": 5.0, "start_ms": 9000, "end_ms": 3000},
    )[0]["answers"]
    assert answers[0] == {"mediaItemName": "L21_V029", "start": 3000, "end": 9000}


def test_qa_answer_carries_the_segment_and_the_text():
    """BTC's payload example: mediaItemName + start + end + text in ONE answer."""
    sets = build_answer_sets(
        task_name="qa-00",
        query_type="QA",
        payload={"video_id": "L02_V003", "frame_idx": 888, "timestamp": 35.5, "answer": "15 người"},
        pad_ms=500,
    )
    assert sets == [{
        "taskName": "qa-00",
        "answers": [{
            "mediaItemName": "L02_V003", "start": 35000, "end": 36000, "text": "15 người",
        }],
    }]


def test_qa_text_only_mode_drops_the_segment():
    answers = build_answer_sets(
        task_name="qa-00", query_type="QA",
        payload={"video_id": "L02_V003", "timestamp": 35.5, "answer": "15 người"},
        answer_mode="text",
    )[0]["answers"]
    assert answers == [{"text": "15 người"}]


def test_qa_without_a_segment_is_rejected():
    with pytest.raises(SubmitFormatError):
        build_answer_sets(task_name="qa-00", query_type="QA", payload={"answer": "15 người"})


def test_qa_without_answer_is_rejected_before_dres():
    with pytest.raises(SubmitFormatError):
        build_answer_sets(task_name="qa-00", query_type="QA", payload={"video_id": "L02_V003"})


def test_kis_without_video_id_is_rejected():
    with pytest.raises(SubmitFormatError):
        build_answer_sets(task_name="tkis-00", query_type="T-KIS", payload={"timestamp": 1.0})


def test_kis_without_any_time_is_rejected():
    with pytest.raises(SubmitFormatError):
        build_answer_sets(task_name="tkis-00", query_type="T-KIS", payload={"video_id": "L21_V029"})


def test_trake_is_one_ordered_answer_per_event():
    answers = build_answer_sets(
        task_name="trake-00",
        query_type="TRAKE",
        payload={
            "video_id": "L03_V005",
            "events": [
                {"event_index": 2, "frame_idx": 400, "pts_time": 16.0},
                {"event_index": 1, "frame_idx": 100, "pts_time": 4.0},
            ],
        },
        pad_ms=250,
    )[0]["answers"]
    assert answers == [
        {"mediaItemName": "L03_V005", "start": 3750, "end": 4250},
        {"mediaItemName": "L03_V005", "start": 15750, "end": 16250},
    ]


def test_answer_mode_override_forces_text_or_item():
    text_mode = build_answer_sets(
        task_name="t", query_type="T-KIS",
        payload={"video_id": "L21_V029", "timestamp": 1.0, "answer": "Hà Nội"}, answer_mode="text",
    )[0]["answers"]
    assert text_mode == [{"text": "Hà Nội"}]

    item_mode = build_answer_sets(
        task_name="t", query_type="QA", payload={"video_id": "L21_V029"}, answer_mode="item",
    )[0]["answers"]
    assert item_mode == [{"mediaItemName": "L21_V029"}]


def test_answer_mode_defaults_per_query_type():
    assert resolve_answer_mode("T-KIS") == "temporal"
    assert resolve_answer_mode("V-KIS") == "temporal"
    assert resolve_answer_mode("TRAKE") == "temporal"
    assert resolve_answer_mode("QA") == "temporal_text"


def test_task_name_omitted_when_unknown():
    """DRES infers the open task; sending taskName=null would be a format error."""
    sets = build_answer_sets(
        task_name=None, query_type="T-KIS", payload={"video_id": "L21_V029", "timestamp": 1.0}
    )
    assert "taskName" not in sets[0]


# ---- clearing the local log ------------------------------------------------


@pytest.mark.asyncio
async def test_clear_history_scoped_to_one_task(submit_service):
    kis = {"video_id": "K01_V001", "frame_idx": 150, "timestamp": 6.0}
    await submit_service.submit(task_id="q001", query_type="T-KIS", payload=kis)
    await submit_service.submit(task_id="q002", query_type="T-KIS", payload=kis)

    result = submit_service.clear_history("q001")

    assert result["deleted"] == 1
    assert submit_service.history("q001") == []
    assert len(submit_service.history("q002")) == 1
    # The removed entries are recoverable next to the history file.
    assert (submit_service.history_path.parent / result["backup"]).exists()


@pytest.mark.asyncio
async def test_clear_history_frees_the_dedup_guard(submit_service):
    """Clearing the log is also what un-blocks re-submitting the same answer."""
    payload = {"video_id": "K01_V001", "frame_idx": 150, "timestamp": 6.0}
    await submit_service.submit(task_id="q001", query_type="T-KIS", payload=payload)
    with pytest.raises(DuplicateSubmitError):
        await submit_service.submit(task_id="q001", query_type="T-KIS", payload=payload)

    submit_service.clear_history()

    entry = await submit_service.submit(task_id="q001", query_type="T-KIS", payload=payload)
    assert entry["was_duplicate"] is False


@pytest.mark.asyncio
async def test_clear_history_survives_a_reload(submit_service):
    await submit_service.submit(
        task_id="q001", query_type="T-KIS",
        payload={"video_id": "K01_V001", "frame_idx": 150, "timestamp": 6.0},
    )
    submit_service.clear_history()

    reloaded = SubmitService(submit_service.s, history_path=submit_service.history_path)
    assert reloaded.history() == []


def test_clear_empty_history_is_a_no_op(submit_service):
    assert submit_service.clear_history() == {"deleted": 0, "remaining": 0, "backup": None}


@pytest.mark.asyncio
async def test_delete_specific_history_entries(submit_service):
    """The operator picks which attempts to drop; the rest stay untouched."""
    ids = []
    for frame in (100, 200, 300):
        entry = await submit_service.submit(
            task_id="q001", query_type="T-KIS",
            payload={"video_id": "K01_V001", "frame_idx": frame, "timestamp": frame / 25},
        )
        ids.append(entry["id"])

    result = submit_service.clear_history(ids=[ids[0], ids[2]])

    assert result["deleted"] == 2
    assert [h["id"] for h in submit_service.history()] == [ids[1]]
    assert (submit_service.history_path.parent / result["backup"]).exists()


@pytest.mark.asyncio
async def test_deleting_one_entry_only_frees_that_answer(submit_service):
    a = await submit_service.submit(
        task_id="q001", query_type="T-KIS",
        payload={"video_id": "K01_V001", "frame_idx": 100, "timestamp": 4.0},
    )
    await submit_service.submit(
        task_id="q001", query_type="T-KIS",
        payload={"video_id": "K01_V001", "frame_idx": 200, "timestamp": 8.0},
    )

    submit_service.clear_history(ids=[a["id"]])

    # The deleted attempt can be re-submitted; the kept one is still guarded.
    await submit_service.submit(
        task_id="q001", query_type="T-KIS",
        payload={"video_id": "K01_V001", "frame_idx": 100, "timestamp": 4.0},
    )
    with pytest.raises(DuplicateSubmitError):
        await submit_service.submit(
            task_id="q001", query_type="T-KIS",
            payload={"video_id": "K01_V001", "frame_idx": 200, "timestamp": 8.0},
        )


def test_deleting_unknown_ids_changes_nothing(submit_service):
    assert submit_service.clear_history(ids=["nope"])["deleted"] == 0
