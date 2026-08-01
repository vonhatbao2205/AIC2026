"""Pure NVILA/web-grounding QA tests; no TestClient or network required."""

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


def test_merge_combines_visual_and_web_answers_without_changing_frames():
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
        source="web",
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
        source="web",
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

def test_pass3_demotes_but_keeps_a_contradicted_visual_answer():
    """Pass 3 only inspects the hotspot ∪ web-cited subset, so it may contradict
    an answer without having seen the frame pass 1 grounded it on. It must veto
    the web claim outright yet leave the visual option visible and labelled."""
    visual = _normalize_qa_analysis(
        "Tên là gì?",
        {"candidate_answers": [
            {"answer": "Wrong", "confidence": .99, "supporting_candidate_ids": ["C01"]},
            {"answer": "Other", "confidence": .70, "supporting_candidate_ids": ["C01"]},
        ]},
        [_candidate()],
    )
    grounded = _normalize_qa_analysis(
        "Tên là gì?",
        {"candidate_answers": [{
            "answer": "Wrong",
            "confidence": .95,
            "supporting_candidate_ids": ["C01"],
        }]},
        [_candidate()],
        source="web",
    )
    verified, meta = _apply_visual_verification(
        grounded,
        {"verdicts": [{
            "answer": "Wrong",
            "status": "contradicted",
            "visual_confidence": .91,
            "supporting_candidate_ids": ["C01"],
            "reason": "The visible logo conflicts with this answer.",
        }]},
        [_candidate()],
    )

    assert verified["candidate_answers"] == []  # the web claim is gone
    assert meta["rejected_answers"] == ["Wrong"]

    merged = _merge_qa_analyses(visual, verified, 5)
    by_answer = {option["answer"]: option for option in merged["candidate_answers"]}
    assert set(by_answer) == {"Other", "Wrong"}
    # Demoted below the undisputed option instead of silently deleted.
    assert merged["candidate_answers"][0]["answer"] == "Other"
    assert by_answer["Wrong"]["confidence"] == .30
    assert by_answer["Wrong"]["visual_verification"]["status"] == "contradicted"


def test_pass3_downgrade_does_not_mislabel_a_directly_grounded_answer():
    """An `insufficient` verdict describes the WEB claim. Showing it next to a
    high pass-1 visual confidence would contradict itself as the operator picks."""
    visual = _normalize_qa_analysis(
        "Tên là gì?",
        {"candidate_answers": [{
            "answer": "Walt Disney",
            "confidence": .94,
            "supporting_candidate_ids": ["C01"],
        }]},
        [_candidate()],
    )
    verified, _ = _apply_visual_verification(
        _normalize_qa_analysis(
            "Tên là gì?",
            {"candidate_answers": [{
                "answer": "Walt Disney",
                "confidence": .80,
                "supporting_candidate_ids": ["C01"],
            }]},
            [_candidate()],
            source="web",
        ),
        {"verdicts": [{"answer": "Walt Disney", "status": "insufficient", "visual_confidence": .2}]},
        [_candidate()],
    )

    option = _merge_qa_analyses(visual, verified, 5)["candidate_answers"][0]

    assert option["confidence"] == .94
    assert option["source"] == "hybrid"
    assert "visual_verification" not in option


def test_web_sources_are_attributed_per_answer_not_broadcast():
    sources = [
        {"title": "thewaltdisneycompany.com", "url": "https://vertex.test/redirect/aaa"},
        {"title": "pixar.fandom.com", "url": "https://vertex.test/redirect/bbb"},
    ]
    normalized = _normalize_qa_analysis(
        "Tên là gì?",
        {"candidate_answers": [
            {
                "answer": "Walt Disney",
                "confidence": .9,
                "supporting_candidate_ids": ["C01"],
                "source_domains": ["thewaltdisneycompany.com"],
            },
            {
                "answer": "Pixar",
                "confidence": .6,
                "supporting_candidate_ids": ["C01"],
                "source_domains": ["https://pixar.fandom.com/wiki/Pixar"],
            },
        ]},
        [_candidate()],
        source="web",
        sources=sources,
    )

    cited = {
        option["answer"]: [source["title"] for source in option["web_sources"]]
        for option in normalized["candidate_answers"]
    }
    assert cited == {
        "Walt Disney": ["thewaltdisneycompany.com"],
        "Pixar": ["pixar.fandom.com"],
    }


def test_unattributable_sources_are_cited_nowhere_unless_unambiguous():
    sources = [
        {"title": "a.test", "url": "https://vertex.test/redirect/aaa"},
        {"title": "b.test", "url": "https://vertex.test/redirect/bbb"},
    ]
    two = _normalize_qa_analysis(
        "Tên là gì?",
        {"candidate_answers": [
            {"answer": "First", "confidence": .9, "supporting_candidate_ids": ["C01"]},
            {"answer": "Second", "confidence": .6, "supporting_candidate_ids": ["C01"]},
        ]},
        [_candidate()],
        source="web",
        sources=sources,
    )
    assert all(option["web_sources"] == [] for option in two["candidate_answers"])

    # A lone grounded answer unambiguously owns the whole search.
    one = _normalize_qa_analysis(
        "Tên là gì?",
        {"candidate_answers": [
            {"answer": "First", "confidence": .9, "supporting_candidate_ids": ["C01"]},
        ]},
        [_candidate()],
        source="web",
        sources=sources,
    )
    assert [source["title"] for source in one["candidate_answers"][0]["web_sources"]] == ["a.test", "b.test"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Có bao nhiêu người trong hai bức ảnh?", False),   # "hai" is not "ai"
        ("Vụ tai nạn xảy ra ở làn đường nào?", False),      # "tai" is not "ai"
        ("Chiếc áo có màu gì?", False),
        ("Người phát biểu là ai?", True),                   # trailing punctuation
        ("Tên thương hiệu trên biển hiệu là gì?", True),
        ("Which company owns the logo?", True),
    ],
)
def test_auto_grounding_only_fires_on_real_entity_questions(monkeypatch, question, expected):
    from app import main as main_module

    monkeypatch.setattr(main_module.settings, "mock_mode", False)
    monkeypatch.setattr(main_module.settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(main_module.settings, "deepseek_grounding_enabled", True)
    visual = {
        "question": question,
        "answerable": True,
        "candidate_answers": [{"answer": "A", "confidence": .92}],
    }
    candidates = [{"evidence": [{"type": "speech", "text": "clue"}]}]

    assert main_module._should_ground_with_web("auto", visual, candidates) is expected


def test_pass3_supported_answer_uses_nvila_frame_and_fused_confidence():
    grounded = _normalize_qa_analysis(
        "Tên là gì?",
        {"candidate_answers": [{
            "answer": "WALTDISNEY",
            "confidence": .88,
            "supporting_candidate_ids": ["C01"],
        }]},
        [_candidate()],
        source="web",
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
async def test_endpoint_pipeline_can_force_mock_web_grounding():
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
    assert result["candidate_answers"][0]["source"] == "web"
    assert result["web_grounding"]["used"] is True
    assert result["candidate_answers"][0]["visual_verification"]["status"] == "supported"
    assert result["web_grounding"]["visual_verification"]["used"] is True
