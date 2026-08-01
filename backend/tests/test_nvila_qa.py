"""Pure NVILA/Google QA tests; no TestClient or network required."""

import pytest

from app.config import Settings
from app.main import (
    _apply_visual_verification,
    _merge_qa_analyses,
    _normalize_qa_analysis,
    analyze_qa,
)
from app.models import QaAnalyzeRequest


def _candidate(candidate_id: str = "C01") -> dict:
    return {
        "candidate_id": candidate_id,
        "submit_keyframe_id": "K01/K01_V001/001",
        "video_id": "K01_V001",
        "keyframe_n": 1,
        "frame_idx": 125,
        "pts_time": 5.0,
        "image_url": "https://media.test/Keyframes_K01/K01_V001/001.jpg",
    }


def test_nvila_config_requires_url_and_token():
    assert Settings(nvila_base_url="https://worker.test").has_nvila is False
    assert Settings(nvila_token="secret").has_nvila is False
    assert Settings(nvila_base_url="https://worker.test", nvila_token="secret").has_nvila is True


def test_normalizer_drops_hallucinated_and_ungrounded_answers():
    result = {
        "answerable": True,
        "best_answer": "hallucinated",
        "best_candidate_id": "C99",
        "candidate_answers": [
            {
                "answer": "hallucinated",
                "confidence": 0.99,
                "supporting_candidate_ids": ["C99"],
            },
            {
                "answer": "grounded",
                "confidence": 0.8,
                "supporting_candidate_ids": ["C01"],
            },
        ],
        "hotspots": [
            {"candidate_id": "C99", "relevance": 1.0},
            {"candidate_id": "C01", "relevance": 0.8},
        ],
    }

    normalized = _normalize_qa_analysis("question", result, [_candidate()])

    assert [item["answer"] for item in normalized["candidate_answers"]] == ["grounded"]
    assert normalized["best_answer"] == "grounded"
    assert normalized["best_candidate_id"] == "C01"
    assert [item["candidate_id"] for item in normalized["hotspots"]] == ["C01"]


def test_normalizer_marks_response_unanswerable_without_grounded_option():
    result = {
        "answerable": True,
        "best_answer": "unsupported",
        "candidate_answers": [{"answer": "unsupported", "supporting_candidate_ids": []}],
    }

    normalized = _normalize_qa_analysis("question", result, [_candidate()])

    assert normalized["answerable"] is False
    assert normalized["best_answer"] == ""
    assert normalized["candidate_answers"] == []


def test_normalizer_drops_zero_confidence_and_prompt_placeholders():
    result = {
        "answerable": True,
        "candidate_answers": [
            {"answer": "Disney", "confidence": 0, "supporting_candidate_ids": ["C01"]},
            {"answer": "visual candidate answer", "confidence": .9, "supporting_candidate_ids": ["C01"]},
            {"answer": "WALTDISNEY", "confidence": .8, "supporting_candidate_ids": ["C01"]},
        ],
        "hotspots": [
            {"candidate_id": "C01", "relevance": 0, "answer_support": "logo"},
            {"candidate_id": "C01", "relevance": .8, "answer_support": "short visible cue"},
            {"candidate_id": "C01", "relevance": .7, "answer_support": "Disney logo in the frame"},
        ],
        "uncertainty": "short caveat",
    }

    normalized = _normalize_qa_analysis("Tên là gì?", result, [_candidate()])

    assert [item["answer"] for item in normalized["candidate_answers"]] == ["WALTDISNEY"]
    assert len(normalized["hotspots"]) == 1
    assert normalized["uncertainty"] == ""


def test_merge_combines_visual_and_google_answers_without_changing_frames():
    visual = _normalize_qa_analysis(
        "Tên là gì?",
        {
            "candidate_answers": [{
                "answer": "Disney",
                "confidence": .65,
                "supporting_candidate_ids": ["C01"],
                "reason": "Visible logo",
            }],
        },
        [_candidate()],
    )
    grounded = _normalize_qa_analysis(
        "Tên là gì?",
        {
            "candidate_answers": [{
                "answer": "WALT DISNEY",
                "confidence": .88,
                "supporting_candidate_ids": ["C01", "C99"],
                "reason": "Canonical name resolved by search",
            }],
        },
        [_candidate()],
        source="google",
        sources=[{"title": "Disney", "url": "https://thewaltdisneycompany.com/"}],
    )

    merged = _merge_qa_analyses(visual, grounded, 5)

    assert [item["answer"] for item in merged["candidate_answers"]] == ["WALT DISNEY", "Disney"]
    assert merged["best_candidate_id"] == "C01"
    assert merged["best_submit_keyframe_id"] == "K01/K01_V001/001"
    assert all(
        frame["candidate_id"] == "C01"
        for answer in merged["candidate_answers"]
        for frame in answer["supporting_frames"]
    )


def test_pass3_drops_contradiction_and_caps_insufficient_web_answer():
    grounded = _normalize_qa_analysis(
        "Tên là gì?",
        {
            "candidate_answers": [
                {"answer": "Wrong", "confidence": .95, "supporting_candidate_ids": ["C01"]},
                {"answer": "Maybe", "confidence": .88, "supporting_candidate_ids": ["C01"]},
            ],
        },
        [_candidate()],
        source="google",
    )
    verified, meta = _apply_visual_verification(
        grounded,
        {
            "model": "NVILA-8B",
            "latency_ms": 12,
            "verdicts": [
                {
                    "answer": "Wrong",
                    "status": "contradicted",
                    "visual_confidence": .91,
                    "supporting_candidate_ids": ["C01"],
                    "reason": "The visible logo conflicts with this answer.",
                },
                {
                    "answer": "Maybe",
                    "status": "insufficient",
                    "visual_confidence": .60,
                    "supporting_candidate_ids": ["C01"],
                    "reason": "The exact expansion is not visible.",
                },
            ],
        },
        [_candidate()],
    )

    assert [option["answer"] for option in verified["candidate_answers"]] == ["Maybe"]
    assert verified["candidate_answers"][0]["confidence"] == .49
    assert verified["candidate_answers"][0]["visual_verification"]["status"] == "insufficient"
    assert meta["rejected_answers"] == ["Wrong"]
    visual = _normalize_qa_analysis(
        "Tên là gì?",
        {"candidate_answers": [{
            "answer": "Wrong",
            "confidence": .99,
            "supporting_candidate_ids": ["C01"],
        }]},
        [_candidate()],
    )
    merged = _merge_qa_analyses(visual, verified, 5)
    assert [option["answer"] for option in merged["candidate_answers"]] == ["Maybe"]


def test_pass3_supported_answer_uses_nvila_frame_and_fused_confidence():
    grounded = _normalize_qa_analysis(
        "Tên là gì?",
        {"candidate_answers": [{
            "answer": "WALTDISNEY",
            "confidence": .88,
            "supporting_candidate_ids": ["C01"],
        }]},
        [_candidate()],
        source="google",
    )
    verified, meta = _apply_visual_verification(
        grounded,
        {"verdicts": [{
            "answer": "WALT DISNEY",
            "status": "supported",
            "visual_confidence": .90,
            "supporting_candidate_ids": ["C01", "C99"],
            "reason": "Disney clue is visually consistent with the canonical entity.",
        }]},
        [_candidate()],
    )

    option = verified["candidate_answers"][0]
    assert option["confidence"] == .886
    assert option["supporting_candidate_ids"] == ["C01"]
    assert option["visual_verification"]["status"] == "supported"
    assert meta["used"] is True


@pytest.mark.asyncio
async def test_endpoint_pipeline_can_force_mock_google_grounding():
    result = await analyze_qa(QaAnalyzeRequest.model_validate({
        "question": "Tên cửa hàng nổi tiếng thế giới là gì?",
        "web_grounding": "on",
        "candidates": [{
            "submit_keyframe_id": "K20/K20_V013/229",
            "frame_idx": 18270,
            "pts_time": 609.0,
            "retrieval_score": .9,
            "evidence": [{"type": "speech", "text": "logo Disney lấy cảm hứng từ lâu đài"}],
        }],
    }))

    assert result["best_answer"] == "WALTDISNEY"
    assert result["best_submit_keyframe_id"] == "K20/K20_V013/229"
    assert result["candidate_answers"][0]["source"] == "google"
    assert result["web_grounding"]["used"] is True
    assert result["candidate_answers"][0]["visual_verification"]["status"] == "supported"
    assert result["web_grounding"]["visual_verification"]["used"] is True
