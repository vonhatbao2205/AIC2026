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
    assert body["capabilities"]["qa_nvila"] is True
    assert body["capabilities"]["qa_visual_verification"] is True
    assert body["capabilities"]["qa_web_grounding"] is True


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


def test_qa_nvila_analysis_is_grounded_to_canonical_candidates():
    payload = {
        "question": "Bản tin trên màn hình tên gì?",
        "candidates": [
            {
                "submit_keyframe_id": "K01/K01_V001/1",
                "frame_idx": 0,
                "pts_time": 0,
                "retrieval_score": 0.9,
                "evidence": [{"type": "ocr", "text": "Bản tin thời sự HTV7"}],
            }
        ],
    }
    r = client.post("/api/qa/analyze", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "mock-nvila-8b"
    assert body["candidate_answers"][0]["answer"] == "Bản tin thời sự"
    assert body["hotspots"][0]["submit_keyframe_id"] == "K01/K01_V001/001"
    assert body["hotspots"][0]["keyframe_url"].endswith("/K01_V001/001.jpg")
    assert body["best_submit_keyframe_id"] == "K01/K01_V001/001"


def test_qa_nvila_analysis_rejects_arbitrary_image_identity():
    r = client.post(
        "/api/qa/analyze",
        json={"question": "What?", "candidates": [{"submit_keyframe_id": "https://evil.test/a.jpg"}]},
    )
    assert r.status_code == 400


def test_qa_web_grounding_merges_answer_but_keeps_canonical_frame():
    r = client.post(
        "/api/qa/analyze",
        json={
            "question": "Tên cửa hàng nổi tiếng thế giới là gì?",
            "web_grounding": "on",
            "candidates": [{
                "submit_keyframe_id": "K20/K20_V013/229",
                "frame_idx": 18270,
                "pts_time": 609.0,
                "retrieval_score": .9,
                "evidence": [{"type": "speech", "text": "logo Disney lấy cảm hứng từ lâu đài"}],
            }],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["best_answer"] == "WALTDISNEY"
    assert body["best_submit_keyframe_id"] == "K20/K20_V013/229"
    assert body["candidate_answers"][0]["source"] == "web"
    assert body["web_grounding"]["used"] is True
    assert body["web_grounding"]["sources"]
    assert body["web_grounding"]["visual_verification"]["used"] is True


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


def test_canvas_palette_only_offers_what_od_can_answer():
    r = client.get("/api/canvas/palette")
    assert r.status_code == 200
    palette = r.json()

    assert len(palette["colors"]) == 16  # aic16-lab-v1
    labels = {item["label"]: item for item in palette["labels"]}
    assert labels["person"]["colorable"] is True
    # OD extracts no colour for crowd, so the picker must not offer one.
    assert labels["crowd"]["colorable"] is False


def test_canvas_search_returns_groups_with_layout_evidence():
    r = client.post(
        "/api/search/canvas",
        json={
            "canvas": {
                "objects": [
                    {"id": "q1", "label": "car", "bbox": [0.05, 0.52, 0.35, 0.86], "color": "red"},
                    {"id": "q2", "label": "person", "bbox": [0.66, 0.35, 0.82, 0.86], "color": "blue"},
                ],
                "mode": "rough",
            }
        },
    )
    assert r.status_code == 200
    body = r.json()

    assert body["canvas"]["queries_en"], "canvas must produce PE text"
    frames = [frame for group in body["groups"] for frame in group["frames"]]
    target = next(f for f in frames if f["submit_keyframe_id"] == "K01/K01_V001/002")
    layout = next(e for e in target["evidence"] if e["type"] == "object_layout")
    assert layout["coverage"] == 1.0
    assert {m["label"] for m in layout["matches"]} == {"car", "person"}


def test_canvas_search_rejects_an_oversized_canvas():
    r = client.post(
        "/api/search/canvas",
        json={"canvas": {"objects": [{"label": "person", "bbox": [0, 0, 0.1, 0.1]}] * 13}},
    )
    assert r.status_code == 422
