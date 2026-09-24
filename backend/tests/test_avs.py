import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.avs import generate_avs
from app.main import app
from app.models import SearchRequest, SubmitRequest
from app.query_parser import heuristic_parse
from app.services.submit_service import DuplicateSubmitError, SubmitFormatError, SubmitService
from .test_answer_gen import frame, group
from .test_dres_client import LOGIN_OK, make_client


def candidates():
    return [group("A", 1, [frame("A", 1, 100, 1), frame("A", 2, 125, .9), frame("A", 3, 500, .8)]),
            group("B", .5, [frame("B", 1, 100, .5), frame("B", 2, 900, .4)])]


def test_avs_only_retrieved_moments_deduped_and_diverse():
    answers, _ = generate_avs(candidates(), limit=100)
    assert [(a["video_id"], a["frame_idx"]) for a in answers] == [("A", 100), ("B", 100), ("A", 500), ("B", 900)]
    assert all(a["kind"] == "anchor" and a["offset_frames"] == 0 for a in answers)
    # Underfilled queues stay underfilled; no artificial KIS offsets.
    assert len(answers) == 4


def test_avs_excludes_accepted_neighborhoods_and_does_not_guess_time():
    groups = candidates()
    groups[1]["frames"][1].update(pts_time=None, fps=None)
    answers, _ = generate_avs(groups, taken=[("A", 110)])
    assert [(a["video_id"], a["frame_idx"]) for a in answers] == [("B", 100), ("A", 500)]
    assert generate_avs([], limit=100)[0] == []


def test_avs_api_reuses_current_groups_and_ignores_kis_offsets():
    SearchRequest(query="people then dogs", query_type_hint="AVS")
    assert heuristic_parse("people then dogs", "AVS")["query_type"] == "AVS"
    with TestClient(app) as client:
        result = client.post("/api/answers/generate", json={"query_type_hint": "AVS", "groups": candidates(),
            "params": {"offsets": [25, 75]}, "limit": 100})
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["reused_results"] and body["query_type"] == "AVS"
    assert len(body["answers"]) == 4
    assert all(a["kind"] == "anchor" for a in body["answers"])


@pytest.mark.asyncio
async def test_avs_dres_wire_flow_routing_preview_submit_and_task_change(tmp_path, allow_test_submit):
    posted = []
    current = {"name": "avs-01", "taskType": "Ad-hoc Video Search"}
    def handler(req):
        path = req.url.path
        if path.endswith("/login"):
            return httpx.Response(200, json=LOGIN_OK)
        if path.endswith("/evaluation/list"):
            return httpx.Response(200, json=[{"id": "e1", "name": "finals", "status": "ACTIVE",
                "taskTemplates": [{"name": "avs-01", "taskType": "Ad-hoc Video Search"}]},
                {"id": "e2", "name": "textual KIS", "status": "ACTIVE"}])
        if "/currentTask/" in path:
            return httpx.Response(200, json=current if path.endswith("e1") else {"name": "kis-01"})
        if path.endswith("/state"):
            return httpx.Response(200, json={"taskStatus": "RUNNING"})
        assert path == "/api/v2/submit/e1"
        posted.append(json.loads(req.read()))
        return httpx.Response(200, json={"status": True, "submission": "CORRECT"})
    client = make_client(handler)
    service = SubmitService(client.s, client, history_path=tmp_path / "history.json")
    payload = {"video_id": "A", "frame_idx": 101, "timestamp": 4.04, "fps": 25}
    prepared = await service.prepare(query_type="AVS", payload=payload, pad_ms=0)
    assert prepared["evaluation_id"] == "e1"
    assert prepared["body"] == {"answerSets": [{"taskName": "avs-01", "answers": [{"mediaItemName": "A", "start": 4040, "end": 4040}]}]}
    assert posted == []  # preview cannot submit
    kwargs = dict(query_type="AVS", payload=payload, evaluation_id="e1", task_name="avs-01",
                  expected_task_name="avs-01", require_dres=True, pad_ms=0)
    result = await service.submit(**kwargs)
    assert result["status"] == "dres_ok" and result["verdict"] == "CORRECT"
    assert posted == [prepared["body"]]
    with pytest.raises(DuplicateSubmitError):
        await service.submit(**kwargs)
    current["name"] = "avs-02"
    with pytest.raises(SubmitFormatError, match="task changed"):
        await service.submit(**kwargs)
    assert len(posted) == 1


@pytest.mark.asyncio
async def test_direct_submit_never_silently_saves_locally(settings, tmp_path):
    service = SubmitService(settings, history_path=tmp_path / "history.json")
    with pytest.raises(SubmitFormatError, match="unavailable"):
        await service.submit(query_type="AVS", payload={"video_id": "A", "timestamp": 4},
                             require_dres=True, expected_task_name="avs-01")
    assert not service.history()
    SubmitRequest(query_type="AVS", payload={"video_id": "A", "timestamp": 4})
