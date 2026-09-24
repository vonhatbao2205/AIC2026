"""Media URL builders (Cloudflare R2 and/or a Hugging Face public bucket).

Layout (from CLOUDFLARE_R2_MEDIA_HANDOFF.md; the HF migration reproduces it
key-for-key, so only the base URL differs between the two origins):
  keyframe: {keyframe_base}/Keyframes/Keyframes_{group}/{video_id}/{frame_3_digits}.jpg
  video:    {video_base}/Videos/Videos_{group}/{video_id}.mp4

URLs are built ONLY from submit_keyframe_id / video_id + keyframe_n.
`image_path` / `source_path` are never used.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .identity import group_from_video_id, parse_submit_keyframe_id

#: Series whose playable copies live under their own prefix. The N traffic-camera
#: recordings carry malformed SEI that Chrome refuses to decode; the repaired
#: copies (same frames and timestamps, see `r2_publish_n_web_videos.py`) sit at
#: `Videos_Web/`, a new key so the year-long immutable CDN cache of the broken
#: originals can never mask them.
_WEB_VIDEO_SERIES = frozenset({"N"})

#: Individual videos republished under yet another root, written by
#: `r2_reencode_n_damaged_videos.py` as each one is uploaded: N recordings whose
#: cameras also dropped frames, re-encoded so Chrome never stops on the damage.
#: Re-read when the file changes, so a published video switches over without a
#: restart; a missing or unreadable file means no overrides.
_OVERRIDES_PATH = Path(__file__).with_name("video_overrides.json")
_overrides_lock = threading.Lock()
_overrides: tuple[float, dict[str, str]] = (-1.0, {})


def _video_root_overrides() -> dict[str, str]:
    global _overrides
    try:
        mtime = _OVERRIDES_PATH.stat().st_mtime
    except OSError:
        return {}
    with _overrides_lock:
        if mtime != _overrides[0]:
            try:
                roots = json.loads(_OVERRIDES_PATH.read_text(encoding="utf-8")).get("roots") or {}
                mapping = {video: root for root, videos in roots.items() for video in videos}
            except (OSError, ValueError, AttributeError):
                mapping = _overrides[1]
            _overrides = (mtime, mapping)
        return _overrides[1]


class MediaUrlBuilder:
    def __init__(self, keyframe_base_url: str, video_base_url: str | None = None):
        self.keyframe_base_url = keyframe_base_url.rstrip("/")
        self.video_base_url = (video_base_url or keyframe_base_url).rstrip("/")
        # Kept for callers/tests that inspect the old attribute.
        self.base_url = self.keyframe_base_url

    def keyframe_url(self, video_id: str, keyframe_n: int) -> str:
        group = group_from_video_id(video_id)
        frame_name = f"{keyframe_n:03d}.jpg"
        return f"{self.keyframe_base_url}/Keyframes/Keyframes_{group}/{video_id}/{frame_name}"

    def keyframe_url_from_submit_id(self, submit_keyframe_id: str) -> str:
        parsed = parse_submit_keyframe_id(submit_keyframe_id)
        # Media group follows the video id, not the (normalized) submit category,
        # and these agree for our data because shard videos still resolve to one group.
        return self.keyframe_url(parsed.video_id, parsed.keyframe_n)

    def video_url(self, video_id: str) -> str:
        group = group_from_video_id(video_id)
        root = _video_root_overrides().get(video_id) or (
            "Videos_Web" if group[:1] in _WEB_VIDEO_SERIES else "Videos"
        )
        return f"{self.video_base_url}/{root}/Videos_{group}/{video_id}.mp4"
