"""Wire-level tests for the DRES v2 client, against a mocked DRES server."""
import json

import httpx
import pytest

from app.adapters.dres_client import DresClient, DresError
from app.config import Settings
from app.services.submit_service import SubmitService


def make_client(handler) -> DresClient:
    settings = Settings(
        mock_mode=False,
        dres_base_url="http://dres.test",
        dres_username="fourier1",
        dres_password="secret",
    )
    client = DresClient(settings)
    transport = httpx.MockTransport(handler)
    client._pool.get = lambda: httpx.AsyncClient(transport=transport)  # type: ignore[method-assign]
    return client


LOGIN_OK = {"id": "u1", "username": "fourier1", "role": "PARTICIPANT", "sessionId": "S1"}


@pytest.mark.asyncio
async def test_login_posts_credentials_and_caches_the_session():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/v2/login":
            return httpx.Response(200, json=LOGIN_OK)
        assert request.url.params["session"] == "S1"
        return httpx.Response(200, json=[])

    client = make_client(handler)
    await client.evaluations()
    await client.evaluations()

    assert [c.url.path for c in calls] == [
        "/api/v2/login",
        "/api/v2/client/evaluation/list",
        "/api/v2/client/evaluation/list",
    ]
    assert client.user == {"id": "u1", "username": "fourier1", "role": "PARTICIPANT"}


@pytest.mark.asyncio
async def test_expired_session_triggers_one_relogin_and_retry():
    """A long competition run must survive DRES expiring the session token."""
    sessions = iter(["S1", "S2"])
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/login":
            return httpx.Response(200, json={**LOGIN_OK, "sessionId": next(sessions)})
        session = request.url.params["session"]
        seen.append(session)
        if session == "S1":
            return httpx.Response(401, json={"status": False, "description": "expired"})
        return httpx.Response(200, json=[{"id": "e1", "name": "tkis"}])

    client = make_client(handler)
    evaluations = await client.evaluations()

    assert seen == ["S1", "S2"]
    assert evaluations[0]["name"] == "tkis"


@pytest.mark.asyncio
async def test_submit_sends_answer_sets_and_reads_the_verdict():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/login":
            return httpx.Response(200, json=LOGIN_OK)
        assert request.url.path == "/api/v2/submit/e1"
        assert request.url.params["session"] == "S1"
        bodies.append(json.loads(request.read()))
        return httpx.Response(
            200, json={"status": True, "submission": "CORRECT", "description": "Correct"}
        )

    client = make_client(handler)
    result = await client.submit(
        "e1", [{"taskName": "tkis-00", "answers": [{"mediaItemName": "L30_V095", "start": 1000, "end": 2000}]}]
    )

    assert bodies == [
        {"answerSets": [
            {"taskName": "tkis-00",
             "answers": [{"mediaItemName": "L30_V095", "start": 1000, "end": 2000}]}
        ]}
    ]
    assert result["verdict"] == "CORRECT"
    assert result["http_status"] == 200


@pytest.mark.asyncio
async def test_submit_202_is_accepted_without_a_verdict():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/login":
            return httpx.Response(200, json=LOGIN_OK)
        return httpx.Response(202, json={"status": True, "description": "queued"})

    result = await make_client(handler).submit("e1", [{"answers": [{"text": "x"}]}])
    assert result["http_status"] == 202
    assert result["verdict"] is None


@pytest.mark.asyncio
async def test_submit_rejection_raises_with_the_server_description():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/login":
            return httpx.Response(200, json=LOGIN_OK)
        return httpx.Response(400, json={"status": False, "description": "Task not running"})

    with pytest.raises(DresError) as exc:
        await make_client(handler).submit("e1", [{"answers": [{"text": "x"}]}])
    assert "Task not running" in str(exc.value)


@pytest.mark.asyncio
async def test_current_task_is_none_when_no_task_is_open():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/login":
            return httpx.Response(200, json=LOGIN_OK)
        return httpx.Response(404, json={"status": False, "description": "no task"})

    assert await make_client(handler).current_task("e1") is None


# ---- evaluation routing ----------------------------------------------------

EVALUATIONS = [
    {
        "id": "e-tkis", "name": "tkis", "status": "ACTIVE",
        "taskTemplates": [{"name": "tkis-00", "taskGroup": "T-KIS Group", "taskType": "Textual KIS"}],
    },
    {
        "id": "e-qa", "name": "qa", "status": "ACTIVE",
        "taskTemplates": [{"name": "qa-00", "taskGroup": "QA Group", "taskType": "Question Answering"}],
    },
]


def routing_service(tmp_path) -> SubmitService:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v2/login":
            return httpx.Response(200, json=LOGIN_OK)
        if path == "/api/v2/client/evaluation/list":
            return httpx.Response(200, json=EVALUATIONS)
        if path.startswith("/api/v2/client/evaluation/currentTask/"):
            name = "tkis-00" if path.endswith("e-tkis") else "qa-00"
            return httpx.Response(200, json={"name": name, "taskGroup": "g", "taskType": "t"})
        if path.endswith("/state"):
            return httpx.Response(200, json={"taskStatus": "RUNNING", "timeLeft": 120})
        return httpx.Response(404, json={"status": False, "description": "?"})

    client = make_client(handler)
    return SubmitService(client.s, client, history_path=tmp_path / "h.json")


@pytest.mark.asyncio
async def test_evaluation_is_auto_picked_from_the_query_type(tmp_path):
    service = routing_service(tmp_path)
    assert await service.resolve_evaluation("T-KIS") == "e-tkis"
    assert await service.resolve_evaluation("QA") == "e-qa"
    # An explicit choice always wins over auto-routing.
    assert await service.resolve_evaluation("QA", "e-tkis") == "e-tkis"


@pytest.mark.asyncio
async def test_prepare_resolves_the_open_task_name(tmp_path):
    service = routing_service(tmp_path)
    prepared = await service.prepare(
        query_type="T-KIS",
        payload={"video_id": "L30_V095", "frame_idx": 250, "timestamp": 10.0},
        pad_ms=500,
    )
    assert prepared["evaluation_id"] == "e-tkis"
    assert prepared["task_name"] == "tkis-00"
    assert prepared["body"]["answerSets"] == [
        {"taskName": "tkis-00", "answers": [{"mediaItemName": "L30_V095", "start": 9500, "end": 10500}]}
    ]
    assert prepared["url"] == "http://dres.test/api/v2/submit/e-tkis"


@pytest.mark.asyncio
async def test_submit_posts_to_dres_and_records_the_verdict(tmp_path):
    posted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v2/login":
            return httpx.Response(200, json=LOGIN_OK)
        if path == "/api/v2/client/evaluation/list":
            return httpx.Response(200, json=EVALUATIONS)
        if path.startswith("/api/v2/client/evaluation/currentTask/"):
            return httpx.Response(200, json={"name": "qa-00", "taskGroup": "g", "taskType": "t"})
        if path.endswith("/state"):
            return httpx.Response(200, json={"taskStatus": "RUNNING", "timeLeft": 60})
        if path == "/api/v2/submit/e-qa":
            posted.append(json.loads(request.read()))
            return httpx.Response(200, json={"status": True, "submission": "CORRECT", "description": "ok"})
        return httpx.Response(404, json={"status": False, "description": "?"})

    client = make_client(handler)
    service = SubmitService(client.s, client, history_path=tmp_path / "h.json")

    entry = await service.submit(
        query_type="QA",
        payload={"video_id": "L30_V095", "frame_idx": 250, "timestamp": 10.0, "answer": "HTV7"},
    )

    assert posted == [{"answerSets": [{"taskName": "qa-00", "answers": [
        {"mediaItemName": "L30_V095", "start": 9500, "end": 10500, "text": "HTV7"}
    ]}]}]
    assert entry["status"] == "dres_ok"
    assert entry["verdict"] == "CORRECT"
    assert entry["task_id"] == "e-qa/qa-00"


# ---- answer shape vs. the open task's type ---------------------------------


def shape_service(tmp_path, task_type: str) -> SubmitService:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v2/login":
            return httpx.Response(200, json=LOGIN_OK)
        if path == "/api/v2/client/evaluation/list":
            return httpx.Response(200, json=EVALUATIONS)
        if path.startswith("/api/v2/client/evaluation/currentTask/"):
            return httpx.Response(200, json={"name": "task-00", "taskGroup": "g", "taskType": task_type})
        if path.endswith("/state"):
            return httpx.Response(200, json={"taskStatus": "RUNNING", "timeLeft": 60})
        return httpx.Response(404, json={"status": False, "description": "?"})

    client = make_client(handler)
    return SubmitService(client.s, client, history_path=tmp_path / "h.json")


@pytest.mark.asyncio
async def test_qa_task_warns_when_the_answer_is_sent_as_a_media_segment(tmp_path):
    """The typed answer would be silently dropped — DRES reads `text` first."""
    service = shape_service(tmp_path, "Question Answering")

    prepared = await service.prepare(
        query_type="QA",
        payload={"video_id": "L30_V090", "timestamp": 22.0, "answer": "Nguyễn Thắm"},
        answer_mode="temporal",
    )

    assert prepared["answer_mode_mismatch"] is True
    assert any("SẼ BỊ BỎ" in w for w in prepared["warnings"])
    assert prepared["task_type"] == "Question Answering"


@pytest.mark.asyncio
async def test_kis_task_warns_when_the_answer_is_sent_as_text(tmp_path):
    service = shape_service(tmp_path, "Textual KIS")

    prepared = await service.prepare(
        query_type="T-KIS",
        payload={"video_id": "L30_V090", "timestamp": 22.0, "answer": "Nguyễn Thắm"},
        answer_mode="text",
    )

    assert prepared["answer_mode_mismatch"] is True
    assert any("chấm sai" in w for w in prepared["warnings"])


@pytest.mark.asyncio
async def test_auto_mode_matches_every_task_type(tmp_path):
    qa = await shape_service(tmp_path, "Question Answering").prepare(
        query_type="QA", payload={"video_id": "L30_V090", "timestamp": 22.0, "answer": "Nguyễn Thắm"},
    )
    assert qa["answer_mode"] == "temporal_text"
    assert qa["answer_mode_mismatch"] is False
    assert qa["body"]["answerSets"][0]["answers"] == [
        {"mediaItemName": "L30_V090", "start": 21500, "end": 22500, "text": "Nguyễn Thắm"}
    ]

    kis = await shape_service(tmp_path, "Textual KIS").prepare(
        query_type="T-KIS", payload={"video_id": "L30_V090", "timestamp": 22.0}, pad_ms=500,
    )
    assert kis["answer_mode"] == "temporal"
    assert kis["answer_mode_mismatch"] is False


# ---- task statement (đề bài) ----------------------------------------------


def hint_service(tmp_path, hint_response, calls: list[str] | None = None) -> SubmitService:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if calls is not None:
            calls.append(path)
        if path == "/api/v2/login":
            return httpx.Response(200, json=LOGIN_OK)
        if path == "/api/v2/client/evaluation/list":
            return httpx.Response(200, json=EVALUATIONS)
        if path.startswith("/api/v2/client/evaluation/currentTask/"):
            return httpx.Response(200, json={"name": "tkis-00", "taskGroup": "g", "taskType": "Textual KIS"})
        if path.endswith("/state"):
            return httpx.Response(
                200, json={"taskStatus": "RUNNING", "taskTemplateId": "tpl-1", "timeElapsed": 5}
            )
        if path.endswith("/template/task/tpl-1/hint"):
            return hint_response
        return httpx.Response(404, json={"status": False, "description": "?"})

    client = make_client(handler)
    return SubmitService(client.s, client, history_path=tmp_path / "h.json")


@pytest.mark.asyncio
async def test_task_hint_returns_the_statement_text(tmp_path):
    service = hint_service(tmp_path, httpx.Response(200, json={
        "taskId": "tpl-1",
        "sequence": [
            {"contentType": "TEXT", "content": "Câu hai", "offset": 30},
            {"contentType": "TEXT", "content": "Câu một", "offset": 0},
            {"contentType": "EMPTY", "content": "", "offset": 0},
        ],
        "loop": False,
    }))

    hint = await service.task_hint("T-KIS")

    # Ordered by offset, so a progressive hint reads in the order teams see it.
    assert hint["text"] == "Câu một\n\nCâu hai"
    assert hint["task_name"] == "tkis-00"
    assert hint["task_status"] == "RUNNING"
    assert hint["task_template_id"] == "tpl-1"
    assert [e["content_type"] for e in hint["elements"]] == ["TEXT", "TEXT"]


@pytest.mark.asyncio
async def test_task_hint_keeps_media_hints_for_vkis(tmp_path):
    service = hint_service(tmp_path, httpx.Response(200, json={
        "taskId": "tpl-1",
        "sequence": [{"contentType": "IMAGE", "content": "base64data", "offset": 0}],
        "loop": True,
    }))

    hint = await service.task_hint("T-KIS")

    assert hint["elements"] == [{"content_type": "IMAGE", "content": "base64data", "offset": 0}]
    assert hint["loop"] is True


@pytest.mark.asyncio
async def test_task_hint_drops_oversized_media_with_a_warning(tmp_path):
    service = hint_service(tmp_path, httpx.Response(200, json={
        "taskId": "tpl-1",
        "sequence": [
            {"contentType": "VIDEO", "content": "x" * 13_000_000, "offset": 0},
            {"contentType": "TEXT", "content": "Đề bài", "offset": 0},
        ],
        "loop": False,
    }))

    hint = await service.task_hint("T-KIS")

    assert hint["text"] == "Đề bài"
    assert [e["content_type"] for e in hint["elements"]] == ["TEXT"]
    assert any("quá lớn" in w for w in hint["warnings"])


@pytest.mark.asyncio
async def test_task_hint_is_cached_per_task_template(tmp_path):
    calls: list[str] = []
    service = hint_service(tmp_path, httpx.Response(200, json={
        "taskId": "tpl-1",
        "sequence": [{"contentType": "TEXT", "content": "Đề bài", "offset": 0}],
        "loop": False,
    }), calls)

    await service.task_hint("T-KIS")
    await service.task_hint("T-KIS")

    hint_calls = [c for c in calls if c.endswith("/hint")]
    assert len(hint_calls) == 1  # a hint never changes while its task is open

    await service.task_hint("T-KIS", force=True)
    assert len([c for c in calls if c.endswith("/hint")]) == 2


@pytest.mark.asyncio
async def test_task_hint_says_so_when_dres_withholds_it(tmp_path):
    service = hint_service(tmp_path, httpx.Response(403, json={"status": False, "description": "nope"}))

    hint = await service.task_hint("T-KIS")

    assert hint["text"] == ""
    assert any("chưa cho xem" in w for w in hint["warnings"])


@pytest.mark.asyncio
async def test_dres_failure_still_records_the_attempt_locally(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v2/login":
            return httpx.Response(200, json=LOGIN_OK)
        if path == "/api/v2/client/evaluation/list":
            return httpx.Response(200, json=EVALUATIONS)
        if path.startswith("/api/v2/client/evaluation/currentTask/"):
            return httpx.Response(200, json={"name": "tkis-00", "taskGroup": "g", "taskType": "t"})
        if path.endswith("/state"):
            return httpx.Response(200, json={"taskStatus": "RUNNING", "timeLeft": 60})
        return httpx.Response(500, json={"status": False, "description": "boom"})

    client = make_client(handler)
    service = SubmitService(client.s, client, history_path=tmp_path / "h.json")

    entry = await service.submit(
        query_type="T-KIS", payload={"video_id": "L30_V095", "timestamp": 10.0}
    )
    assert entry["status"] == "dres_error"
    assert "boom" in entry["dres"]["error"]
    assert service.history(entry["task_id"])
