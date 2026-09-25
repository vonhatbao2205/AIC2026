"""End-to-end API tests against the FastAPI app in mock mode."""
import pytest
from fastapi.testclient import TestClient

from app import paths
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
    assert body["model"] == "mock-deepseek-vision"
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
        "payload": {"video_id": "K01_V001", "frame_idx": 150, "timestamp": 6.0},
    }
    r1 = client.post("/api/submit", json=payload)
    assert r1.status_code == 200
    # The stored entry carries the exact DRES v2 body (ms window, no text).
    assert r1.json()["answer_sets"][0]["answers"][0]["mediaItemName"] == "K01_V001"
    r2 = client.post("/api/submit", json=payload)
    assert r2.status_code == 409
    assert r2.json()["detail"]["error"] == "duplicate_submit"


def test_submit_rejects_a_frame_with_no_resolvable_time():
    """No pts_time and no fps means no ms window — caught before DRES sees it."""
    from app.main import submit_service

    submit_service._history = []
    r = client.post(
        "/api/submit",
        json={"task_id": "qX", "query_type": "T-KIS", "payload": {"video_id": "K01_V001", "frame_idx": 150}},
    )
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "invalid_format"


def test_submit_preview_returns_the_exact_dres_body():
    r = client.post(
        "/api/submit/preview",
        json={
            "task_id": "qX",
            "query_type": "QA",
            "payload": {"video_id": "K01_V001", "frame_idx": 150, "timestamp": 6.0, "answer": "HTV7"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer_mode"] == "qa_text"
    # The final-round QA form: answer, video and the frame's instant in one text.
    assert body["body"]["answerSets"][0]["answers"] == [{"text": "QA-HTV7-K01_V001-6000"}]


def test_dres_status_reports_disabled_in_mock_mode():
    r = client.get("/api/dres/status")
    assert r.status_code == 200
    assert r.json()["configured"] is False


SKETCH_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_canvas_palette_endpoint_is_gone():
    # The object palette served the OD vocabulary; the sketch needs none.
    assert client.get("/api/canvas/palette").status_code == 404


@pytest.mark.parametrize("database", ["btc", "infoshotpp"])
def test_canvas_sketch_searches_the_selected_profile(database):
    r = client.post(
        "/api/search/canvas",
        json={"retrieval_database": database, "canvas": {"image": SKETCH_PNG}},
    )
    assert r.status_code == 200
    body = r.json()

    assert body["retrieval_database"] == database
    assert body["canvas"] == {"has_image": True, "suppress_blank": True}
    assert body["groups"], "a sketch must return candidates in mock mode"
    frames = [frame for group in body["groups"] for frame in group["frames"]]
    assert all(frame["channels"] == ["canvas_image"] for frame in frames)


def test_canvas_sketch_is_routed_to_its_own_profile_service(monkeypatch):
    from app import main

    called: list[str] = []
    for name, service in main.canvas_services.items():
        async def fake(req, name=name):
            called.append(name)
            return {"groups": [], "warnings": [], "canvas": {"has_image": True}}
        monkeypatch.setattr(service, "search", fake)

    client.post("/api/search/canvas", json={"retrieval_database": "infoshotpp", "canvas": {"image": SKETCH_PNG}})

    assert called == ["infoshotpp"]


def test_canvas_search_requires_a_drawing():
    assert client.post("/api/search/canvas", json={"canvas": {}}).status_code == 422
    assert client.post("/api/search/canvas", json={"canvas": {"image": ""}}).status_code == 422


def test_canvas_search_rejects_an_oversized_drawing():
    r = client.post(
        "/api/search/canvas",
        json={"canvas": {"image": "data:image/png;base64," + "A" * 4_000_000}},
    )
    assert r.status_code == 422


@pytest.mark.parametrize("database", ["btc", "infoshotpp"])
def test_health_offers_the_sketch_on_both_profiles(database):
    body = client.get(f"/api/health?retrieval_database={database}").json()

    assert body["capabilities"]["canvas_sketch_search"] is True
    assert "canvas_object_search" not in body["capabilities"]
    assert "object_index" not in body["services"]


@pytest.mark.skipif(paths.static_dir() is None, reason="frontend not built (npm run build)")
def test_bundled_ui_is_served_with_a_spa_fallback():
    """The packaged app is one origin: the console and the API share a port."""
    assert client.get("/").status_code == 200
    # Unknown UI paths fall back to the shell so the app boots anywhere.
    assert client.get("/deep/link").status_code == 200
    # …but a mistyped API path must stay a 404, not become an HTML page.
    assert client.get("/api/definitely-not-a-route").status_code == 404
