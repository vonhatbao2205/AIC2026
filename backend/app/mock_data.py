"""Deterministic in-memory fixtures for mock mode.

Small synthetic corpus across a few videos so the UI/tests have realistic shapes
without touching Elastic/Milvus. Keyed by submit_keyframe_id / video_id.
"""
from __future__ import annotations

from .identity import make_submit_keyframe_id

FPS = 25.0
KEYFRAME_INTERVAL_S = 4.0  # synthetic spacing between keyframes

MOCK_VIDEOS = ["K01_V001", "K01_V002", "L26_V001"]


def _keyframes_for(video_id: str, count: int = 30) -> list[dict]:
    frames = []
    for n in range(1, count + 1):
        pts = round((n - 1) * KEYFRAME_INTERVAL_S, 3)
        frames.append(
            {
                "submit_keyframe_id": make_submit_keyframe_id(video_id, n),
                "video_id": video_id,
                "keyframe_n": n,
                "keyframe_name": f"{n:03d}",
                "frame_idx": int(round(pts * FPS)),
                "pts_time": pts,
                "fps": FPS,
            }
        )
    return frames


MOCK_KEYFRAMES: dict[str, list[dict]] = {v: _keyframes_for(v) for v in MOCK_VIDEOS}

# OCR text seeded on a few frames.
MOCK_OCR = {
    "K01/K01_V001/001": {"text_clean": "HTV7 HD Bản tin thời sự", "clock": "18:29:57", "hour": 18},
    "K01/K01_V001/006": {"text_clean": "Thủ tướng Nhật Bản", "clock": None, "hour": None},
    "L26/L26_V001/003": {"text_clean": "Giá vàng hôm nay tăng mạnh", "clock": None, "hour": None},
}

# Speech segments per video.
MOCK_SPEECH = {
    "K01_V001": [
        {
            "segment_id": "K01_V001_s0",
            "start": 3.6,
            "end": 28.9,
            "text": "kính chào quý vị đây là chương trình thời sự thủ tướng nhật bản",
            "confidence_bucket": "high",
            "segment_role": "intro",
            "center_time": 16.3,
        },
        {
            "segment_id": "K01_V001_s1",
            "start": 40.0,
            "end": 52.0,
            "text": "lũ gây sạt lở cô lập hoàn toàn bốn bản",
            "confidence_bucket": "high",
            "segment_role": "body",
            "center_time": 46.0,
        },
    ],
    "L26_V001": [
        {
            "segment_id": "L26_V001_s0",
            "start": 5.0,
            "end": 15.0,
            "text": "giá vàng trong nước hôm nay tăng mạnh",
            "confidence_bucket": "mid",
            "segment_role": "body",
            "center_time": 10.0,
        }
    ],
}

# Audio windows per video.
MOCK_AUDIO = {
    "K01_V001": [
        {
            "window_id": "K01_V001_a0",
            "start": 0.0,
            "end": 5.0,
            "top1_label": "Theme music",
            "top1_score": 0.82,
            "tag_labels": ["Theme music", "Music", "Speech"],
            "caption": "news intro jingle",
            "caption_quality": "useful",
            "audio_stoplist_hit": False,
            "top1_is_stoplisted": False,
            "center_time": 2.5,
        },
        {
            "window_id": "K01_V001_a1",
            "start": 44.0,
            "end": 49.0,
            "top1_label": "Speech",
            "top1_score": 0.7,
            "tag_labels": ["Speech"],
            "caption": None,
            "caption_quality": "none",
            "audio_stoplist_hit": False,
            "top1_is_stoplisted": False,
            "center_time": 46.5,
        },
    ],
    "L26_V001": [
        {
            "window_id": "L26_V001_a0",
            "start": 8.0,
            "end": 13.0,
            "top1_label": "Music",
            "top1_score": 0.6,
            "tag_labels": ["Music", "Bell"],
            "caption": "bản tin tài chính",
            "caption_quality": "vietnamese_asr",
            "audio_stoplist_hit": False,
            "top1_is_stoplisted": False,
            "center_time": 10.5,
        }
    ],
}


def keyframe_record(submit_keyframe_id: str) -> dict | None:
    for frames in MOCK_KEYFRAMES.values():
        for f in frames:
            if f["submit_keyframe_id"] == submit_keyframe_id:
                return f
    return None
