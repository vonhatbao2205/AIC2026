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


def _detection(label: str, bbox: tuple[float, float, float, float], conf: float, color: str | None = None) -> dict:
    x1, y1, x2, y2 = bbox
    return {
        "instance_id": f"{label}-{int(x1 * 100)}-{int(y1 * 100)}",
        "label": label,
        "canonical_label": label,
        "conf": conf,
        "bbox_norm": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "center_norm": {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2},
        "size_norm": {"width": x2 - x1, "height": y2 - y1, "area": (x2 - x1) * (y2 - y1)},
        "position": "center",
        "position_tags": ["center"],
        "dominant_color": color,
        "color_names": [color] if color else [],
        "color_reliable": bool(color),
    }


# Object-detection fixtures for the V-KIS canvas channel. Frame 002 of K01_V001 is
# the intended answer for the canvas used in tests: a red car on the left and a
# person on the right, which is exactly the layout a drawn query describes.
MOCK_OBJECT_FRAMES: dict[str, list[dict]] = {
    "K01/K01_V001/002": [
        _detection("car", (0.05, 0.50, 0.36, 0.86), 0.91, "red"),
        _detection("person", (0.66, 0.34, 0.83, 0.88), 0.87, "blue"),
        _detection("building", (0.00, 0.02, 1.00, 0.55), 0.44),
    ],
    "K01/K01_V001/007": [
        _detection("person", (0.10, 0.30, 0.28, 0.90), 0.79, "black"),
        _detection("microphone", (0.30, 0.44, 0.36, 0.58), 0.61),
    ],
    "K01/K01_V002/004": [
        _detection("car", (0.60, 0.55, 0.92, 0.88), 0.72, "white"),
        _detection("person", (0.20, 0.36, 0.38, 0.88), 0.66, "red"),
    ],
    "L26/L26_V001/003": [
        _detection("screen", (0.25, 0.15, 0.78, 0.70), 0.83, "blue"),
        _detection("person", (0.05, 0.40, 0.22, 0.92), 0.58, "white"),
    ],
}


def object_frames(labels: list[str], *, size: int = 400) -> list[dict]:
    """Mock OD candidates: frames holding at least one of the drawn labels."""
    wanted = {label.lower() for label in labels}
    out = []
    for submit_keyframe_id, detections in MOCK_OBJECT_FRAMES.items():
        if wanted and not any(d["canonical_label"] in wanted for d in detections):
            continue
        _, video_id, frame_name = submit_keyframe_id.split("/")
        out.append(
            {
                "submit_keyframe_id": submit_keyframe_id,
                "video_id": video_id,
                "keyframe_n": int(frame_name),
                "width": 1280,
                "height": 720,
                "detection_count": len(detections),
                "detections": detections,
            }
        )
    return out[:size]
