"""QA copilot visual passes on DeepSeek V4.1 Flash: request shape, fallbacks, caching."""
import json

import pytest

from app.adapters.deepseek_vision_qa import ANALYZE_SYSTEM_PROMPT, DeepSeekVisionQaClient
from app.adapters.nvila_client import NvilaQaClient
from app.adapters.qa_vision import QaVisionUnavailable, build_qa_vision_client
from app.config import Settings


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b"", headers=None, text=""):
        self.status_code = status
        self._payload = payload
        self.content = content
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, replies, image_ok=True):
        self.replies = list(replies)
        self.posts = []
        self.image_ok = image_ok

    async def get(self, url, timeout=None):
        if not self.image_ok:
            raise RuntimeError("download failed")
        return FakeResponse(content=b"\xff\xd8jpeg", headers={"content-type": "image/jpeg"})

    async def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json})
        reply = self.replies.pop(0)
        return FakeResponse(payload={"choices": [{"message": {"content": reply}}]})


def live():
    return Settings(mock_mode=False, deepseek_api_key="sk-test")


def wire(client, fake):
    client._http.get = lambda: fake
    return fake


def candidates(n=2):
    return [
        {
            "candidate_id": f"C{i:02d}", "submit_keyframe_id": f"L30/L30_V072/{i:03d}", "video_id": "L30_V072",
            "keyframe_n": i, "frame_idx": 100 * i, "pts_time": 4.0 * i, "retrieval_score": 0.03,
            "image_url": f"https://keyframe.test/{i}.jpg",
            "evidence": [{"type": "ocr", "text": "CLB FANA trao quà xã Giang Ly"}] if i == 1 else [],
        }
        for i in range(1, n + 1)
    ]


ANSWER = {
    "answerable": True,
    "hotspots": [{"candidate_id": "C01", "relevance": 0.95, "answer_support": "banner reads Giang Ly"}],
    "candidate_answers": [{"answer": "Giang Ly", "confidence": 0.93, "supporting_candidate_ids": ["C01"], "reason": "banner"}],
    "best_answer": "Giang Ly",
    "best_candidate_id": "C01",
    "uncertainty": "",
}


@pytest.mark.asyncio
async def test_analyze_sends_labelled_frames_to_deepseek_with_thinking():
    client = DeepSeekVisionQaClient(live())
    fake = wire(client, FakeClient([json.dumps(ANSWER)]))
    result = await client.analyze("Hỏi xã này có tên là gì?", candidates(), max_answers=3)
    body = fake.posts[0]["json"]
    assert fake.posts[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert body["model"] == "deepseek-flash"
    assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "high"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["content"] == ANALYZE_SYSTEM_PROMPT
    parts = body["messages"][1]["content"]
    # Each frame's label comes right before its image, so ids cannot drift from pictures.
    assert parts[1]["type"] == "text" and parts[1]["text"].startswith("C01 · video L30_V072 · t=00:04.00")
    assert "OCR: CLB FANA" in parts[1]["text"]
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[2]["image_url"]["detail"] == "high"
    assert parts[3]["text"].startswith("C02") and "(no text cue)" in parts[3]["text"]
    assert result["best_answer"] == "Giang Ly" and result["model"] == "deepseek-flash"


@pytest.mark.asyncio
async def test_a_frame_the_backend_cannot_fetch_goes_as_its_url():
    client = DeepSeekVisionQaClient(live())
    fake = wire(client, FakeClient([json.dumps(ANSWER)], image_ok=False))
    await client.analyze("q", candidates(1))
    assert fake.posts[0]["json"]["messages"][1]["content"][2]["image_url"]["url"] == "https://keyframe.test/1.jpg"


@pytest.mark.asyncio
async def test_empty_json_is_retried_and_results_are_cached():
    client = DeepSeekVisionQaClient(live())
    fake = wire(client, FakeClient(["", json.dumps(ANSWER)]))
    first = await client.analyze("q", candidates())
    again = await client.analyze("q", candidates())
    assert len(fake.posts) == 2  # one retry, then the cache answers
    assert first["best_answer"] == "Giang Ly" and again["cached"] is True


@pytest.mark.asyncio
async def test_failures_surface_as_unavailable():
    client = DeepSeekVisionQaClient(live())
    wire(client, FakeClient(["no json", "still none"]))
    with pytest.raises(QaVisionUnavailable, match="not a JSON object"):
        await client.analyze("q", candidates())
    with pytest.raises(QaVisionUnavailable, match="DEEPSEEK_API_KEY"):
        await DeepSeekVisionQaClient(Settings(mock_mode=False)).analyze("q", candidates())


@pytest.mark.asyncio
async def test_verification_looks_at_hotspots_and_cited_frames():
    client = DeepSeekVisionQaClient(live())
    verdicts = {"verdicts": [{"answer": "Giang Ly", "status": "supported", "visual_confidence": 0.9,
                              "supporting_candidate_ids": ["C02"], "reason": "sign"}], "uncertainty": ""}
    fake = wire(client, FakeClient([json.dumps(verdicts)]))
    result = await client.verify_grounded(
        "q", candidates(3),
        [{"answer": "Giang Ly", "confidence": 0.8, "supporting_candidate_ids": ["C03"], "reason": "web",
          "web_sources": [{"title": "vi.wikipedia.org"}]}],
        hotspot_ids=["C02"],
    )
    parts = fake.posts[0]["json"]["messages"][1]["content"]
    labels = [part["text"][:3] for part in parts if part["type"] == "text" and part["text"][:1] == "C"]
    assert labels == ["C02", "C03"]
    assert '"answer": "Giang Ly"' in parts[0]["text"] and "vi.wikipedia.org" in parts[0]["text"]
    assert result["verdicts"][0]["status"] == "supported"


def test_backend_selection():
    assert isinstance(build_qa_vision_client(Settings()), DeepSeekVisionQaClient)
    assert isinstance(build_qa_vision_client(Settings(qa_vision_backend="nvila")), NvilaQaClient)
    assert Settings(deepseek_api_key="k").has_qa_vision is True
    assert Settings(qa_vision_backend="nvila", deepseek_api_key="k").has_qa_vision is False
