import pytest

from app.services.submit_service import DRES_TASK_TYPE, DuplicateSubmitError, SubmitService


@pytest.fixture
def submit_service(settings, tmp_path):
    return SubmitService(settings, history_path=tmp_path / "history.json")


@pytest.mark.asyncio
async def test_submit_stores_local_history(submit_service):
    entry = await submit_service.submit(
        task_id="q001",
        query_type="T-KIS",
        payload={"video_id": "K01_V001", "frame_idx": 150},
    )
    assert entry["status"] == "local"  # no DRES env in mock
    assert entry["task_id"] == "q001"
    assert entry["dedup_keys"] == ["K01_V001:150"]
    assert submit_service.history("q001")


@pytest.mark.asyncio
async def test_duplicate_submit_blocked_same_video_frame(submit_service):
    await submit_service.submit(task_id="q001", query_type="T-KIS", payload={"video_id": "K01_V001", "frame_idx": 150})
    with pytest.raises(DuplicateSubmitError):
        await submit_service.submit(task_id="q001", query_type="T-KIS", payload={"video_id": "K01_V001", "frame_idx": 150})


@pytest.mark.asyncio
async def test_different_frame_idx_not_duplicate(submit_service):
    await submit_service.submit(task_id="q001", query_type="T-KIS", payload={"video_id": "K01_V001", "frame_idx": 150})
    entry = await submit_service.submit(task_id="q001", query_type="T-KIS", payload={"video_id": "K01_V001", "frame_idx": 151})
    assert entry["status"] == "local"


@pytest.mark.asyncio
async def test_duplicate_allowed_when_flag_set(submit_service):
    await submit_service.submit(task_id="q001", query_type="T-KIS", payload={"video_id": "K01_V001", "frame_idx": 150})
    entry = await submit_service.submit(
        task_id="q001", query_type="T-KIS", payload={"video_id": "K01_V001", "frame_idx": 150}, allow_duplicate=True
    )
    assert entry["was_duplicate"] is True


@pytest.mark.asyncio
async def test_same_frame_different_task_not_duplicate(submit_service):
    await submit_service.submit(task_id="q001", query_type="T-KIS", payload={"video_id": "K01_V001", "frame_idx": 150})
    entry = await submit_service.submit(task_id="q002", query_type="T-KIS", payload={"video_id": "K01_V001", "frame_idx": 150})
    assert entry["status"] == "local"


@pytest.mark.asyncio
async def test_trake_sequence_submit_and_dedup(submit_service):
    payload = {
        "video_id": "L21_V029",
        "events": [
            {"event_index": 1, "frame_idx": 100},
            {"event_index": 2, "frame_idx": 11232},
        ],
    }
    entry = await submit_service.submit(task_id="q004", query_type="TRAKE", payload=payload)
    assert entry["dedup_keys"] == ["L21_V029|100,11232"]
    with pytest.raises(DuplicateSubmitError):
        await submit_service.submit(task_id="q004", query_type="TRAKE", payload=payload)


def test_dres_body_kis_format(submit_service):
    body = submit_service._dres_body("q1", "T-KIS", {"video_id": "L21_V029", "frame_idx": 11232, "timestamp": 374.4})
    assert body == {"task_id": "q1", "task_type": "TKIS", "video_id": "L21_V029", "frame_idx": 11232, "timestamp": 374.4}


def test_dres_body_qa_includes_answer(submit_service):
    body = submit_service._dres_body("q2", "QA", {"video_id": "L02_V003", "frame_idx": 888, "answer": "15 người"})
    assert body["task_type"] == "QA"
    assert body["answer"] == "15 người"
    assert body["frame_idx"] == 888


def test_dres_body_trake_events(submit_service):
    body = submit_service._dres_body(
        "q4", "TRAKE",
        {"video_id": "L03_V005", "events": [{"event_index": 1, "frame_idx": 100}, {"event_index": 2, "frame_idx": 400}]},
    )
    assert body["task_type"] == "TRAKE"
    assert body["video_id"] == "L03_V005"
    assert body["events"] == [{"event_index": 1, "frame_idx": 100}, {"event_index": 2, "frame_idx": 400}]
    assert "frame_idx" not in body


def test_task_type_mapping():
    assert DRES_TASK_TYPE["T-KIS"] == "TKIS"
    assert DRES_TASK_TYPE["V-KIS"] == "VKIS"
