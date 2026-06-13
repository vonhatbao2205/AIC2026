"""Keyframe timeline for one video.

Returns the BTC keyframe list (sorted, with frame_idx/pts_time) used by the
frontend filmstrip and the TRAKE pause-frame picker (snap to nearest keyframe).
The heavier OCR/speech/audio tracks were intentionally dropped for speed.
"""
from __future__ import annotations

from typing import Any

from ..adapters.elastic_client import ElasticClient
from ..config import Settings
from ..media import MediaUrlBuilder

DEFAULT_FPS = 25.0


class TimelineService:
    def __init__(self, settings: Settings):
        self.s = settings
        self.elastic = ElasticClient(settings)
        self.media = MediaUrlBuilder(settings.media_base_url)

    async def build(self, video_id: str) -> dict[str, Any]:
        keyframes_raw = await self.elastic.get_video_keyframes(video_id)

        keyframes = []
        fps = DEFAULT_FPS
        for kf in sorted(keyframes_raw, key=lambda k: k.get("keyframe_n", 0)):
            fps = kf.get("fps") or fps
            keyframes.append(
                {
                    "submit_keyframe_id": kf["submit_keyframe_id"],
                    "keyframe_n": kf["keyframe_n"],
                    "frame_idx": kf.get("frame_idx"),
                    "pts_time": kf.get("pts_time"),
                    "keyframe_url": self.media.keyframe_url(video_id, kf["keyframe_n"]),
                }
            )

        duration = keyframes[-1]["pts_time"] if keyframes else None

        return {
            "video_id": video_id,
            "fps": fps,
            "duration": duration,
            "video_url": self.media.video_url(video_id),
            "keyframes": keyframes,
        }
