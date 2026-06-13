"""End-to-end API tests against the FastAPI app in mock mode."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_mock_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "mock"
    assert body["ok"] is True
    # capabilities flag off for unbuilt modules
    assert body["capabilities"]["vlm_rerank"] is False
    assert body["capabilities"]["audio_vector_search"] is False


def test_parse_endpoint():
    r = client.post("/api/query/parse", json={"query": "thủ tướng phát biểu 18:00"})
    assert r.status_code == 200
    parsed = r.json()
    assert parsed["channels"]["image_pe"]["enabled"] is True


def test_search_endpoint_grouped():
    r = client.post("/api/search", json={"query": "bản tin thời sự"})
    assert r.status_code == 200
    body = r.json()
    assert body["groups"]
    assert body["groups"][0]["frames"][0]["submit_keyframe_id"].count("/") == 2


def test_simple_search_flat_ordered_list():
    r = client.post("/api/search/simple", json={"query": "a flooded street", "top_k": 12})
    assert r.status_code == 200
    body = r.json()
    assert "results" in body and isinstance(body["results"], list)
    assert len(body["results"]) <= 12
    scores = [x["score"] for x in body["results"]]
    assert scores == sorted(scores, reverse=True)  # ordered by similarity
    first = body["results"][0]
    assert first["image_id"] == first["submit_keyframe_id"]
    assert first["keyframe_url"].endswith(".jpg")
    assert "image_path" not in str(body)


def test_simple_search_empty_query():
    r = client.post("/api/search/simple", json={"query": "  "})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_keyframe_endpoint_path_with_slashes():
    r = client.get("/api/keyframes/K01/K01_V001/001")
    assert r.status_code == 200
    body = r.json()
    assert body["submit_keyframe_id"] == "K01/K01_V001/001"
    assert body["image_id"] == body["submit_keyframe_id"]
    assert body["keyframe_url"].endswith("/Keyframes_K01/K01_V001/001.jpg")


def test_keyframe_endpoint_normalizes_shard():
    r = client.get("/api/keyframes/L26_a/L26_V001/14")
    assert r.status_code == 200
    assert r.json()["submit_keyframe_id"] == "L26/L26_V001/014"


def test_timeline_endpoint():
    r = client.get("/api/videos/K01_V001/timeline")
    assert r.status_code == 200
    body = r.json()
    assert body["fps"] == 25.0
    assert body["keyframes"]


def test_snap_endpoint():
    r = client.post("/api/videos/K01_V001/snap", json={"raw_time": 5.0})
    assert r.status_code == 200
    body = r.json()
    assert body["submit_keyframe_id"].count("/") == 2
    assert "delta_frames" in body and "far" in body


def test_submit_and_duplicate_guard(tmp_path, monkeypatch):
    # isolate history file
    from app.main import submit_service

    submit_service.history_path = tmp_path / "h.json"
    submit_service._history = []
    payload = {
        "task_id": "qX",
        "query_type": "T-KIS",
        "payload": {"video_id": "K01_V001", "frame_idx": 150},
    }
    r1 = client.post("/api/submit", json=payload)
    assert r1.status_code == 200
    r2 = client.post("/api/submit", json=payload)
    assert r2.status_code == 409
    assert r2.json()["detail"]["error"] == "duplicate_submit"
