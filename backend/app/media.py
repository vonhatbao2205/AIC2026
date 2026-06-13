"""Cloudflare R2 media URL builders.

Layout (from CLOUDFLARE_R2_MEDIA_HANDOFF.md):
  keyframe: {base}/Keyframes/Keyframes_{group}/{video_id}/{frame_3_digits}.jpg
  video:    {base}/Videos/Videos_{group}/{video_id}.mp4

URLs are built ONLY from submit_keyframe_id / video_id + keyframe_n.
`image_path` / `source_path` are never used.
"""
from __future__ import annotations

from .identity import group_from_video_id, parse_submit_keyframe_id


class MediaUrlBuilder:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def keyframe_url(self, video_id: str, keyframe_n: int) -> str:
        group = group_from_video_id(video_id)
        frame_name = f"{keyframe_n:03d}.jpg"
        return f"{self.base_url}/Keyframes/Keyframes_{group}/{video_id}/{frame_name}"

    def keyframe_url_from_submit_id(self, submit_keyframe_id: str) -> str:
        parsed = parse_submit_keyframe_id(submit_keyframe_id)
        # Media group follows the video id, not the (normalized) submit category,
        # and these agree for our data because shard videos still resolve to one group.
        return self.keyframe_url(parsed.video_id, parsed.keyframe_n)

    def video_url(self, video_id: str) -> str:
        group = group_from_video_id(video_id)
        return f"{self.base_url}/Videos/Videos_{group}/{video_id}.mp4"
