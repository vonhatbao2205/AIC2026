"""Elastic adapter for the OD frame index (`od-frame-v5`).

Candidate generation only: it returns frames whose detections could plausibly
satisfy the canvas, together with the detection geometry. Ranking is NOT done
here — a nested Elastic query scores each clause independently, so two drawn
people would both match the same detected person. The one-to-one assignment in
`app.canvas` decides the final order.

Identity note: OD documents are keyed by `document_id = "<video_id>:<frame_name>"`,
which is not what the rest of the app joins on. The uploader stores
`submit_keyframe_id` alongside it, and this adapter falls back to deriving it so a
frame is never dropped just because the index predates that field.
"""
from __future__ import annotations

from typing import Any


from .. import mock_data
from ..config import Settings
from ..identity import group_from_video_id, make_submit_keyframe_id
from ..scope import elastic_filter_clause
from .http_pool import PooledHttpClient, failure_reason

# Detection subfields the assignment needs. Fetching the full `detections` array
# would drag along per-source scores and the raw colour histogram — several times
# the payload for data the matcher never reads.
_DETECTION_FIELDS = [
    "detections.instance_id",
    "detections.label",
    "detections.canonical_label",
    "detections.conf",
    "detections.bbox_norm",
    "detections.center_norm",
    "detections.size_norm",
    "detections.position",
    "detections.position_tags",
    "detections.dominant_color",
    "detections.color_names",
    "detections.color_reliable",
]

# Frame-level fields only. Detections arrive through `inner_hits` so the response
# carries just the detections that could match a drawn object: the assignment
# never pairs an object with a different canonical label, and pulling every
# detection of 400 frames instead cost ~7s per canvas search.
_SOURCE_FIELDS = [
    "submit_keyframe_id",
    "document_id",
    "video_id",
    "frame_name",
    "detection_count",
    "width",
    "height",
]

# Enough nested hits to cover a crowd scene without unbounded payloads; the
# assignment only needs the best few instances per drawn label anyway.
_INNER_HITS_SIZE = 24


class ObjectElasticClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.mock = settings.mock_mode or not settings.has_elastic
        self._base = (settings.elastic_endpoint or "").rstrip("/")
        self._http = PooledHttpClient()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"ApiKey {self.s.elastic_api_key}",
            "Content-Type": "application/json",
        }

    async def health(self) -> dict[str, Any]:
        if self.mock:
            return {"ok": True, "mode": "mock"}
        try:
            resp = await self._http.get().get(
                f"{self._base}/{self.s.idx_objects}/_count", headers=self._headers(), timeout=10.0
            )
            resp.raise_for_status()
            return {"ok": True, "documents": resp.json().get("count", 0)}
        except Exception as exc:  # noqa: BLE001 - surfaced as a health warning
            return {"ok": False, "error": failure_reason(exc)}

    async def search_canvas_candidates(
        self,
        labels: list[str],
        *,
        colors: dict[str, str] | None = None,
        positions: dict[str, str] | None = None,
        exclude_labels: list[str] | None = None,
        categories: tuple[str, ...] = (),
        size: int = 400,
        min_conf: float = 0.20,
        min_label_ratio: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Frames that contain enough of the drawn labels, with their detections.

        `minimum_should_match` is a ratio, not "all": an operator drawing five
        objects from memory will usually get one of them wrong, and demanding all
        five would return nothing exactly when help is needed most.
        """
        labels = [label for label in dict.fromkeys(labels) if label]
        if not labels:
            return []
        if self.mock:
            frames = mock_data.object_frames(labels, size=size)
            if not categories:
                return frames
            wanted = set(categories)
            return [f for f in frames if group_from_video_id(str(f.get("video_id") or "")) in wanted]

        required = max(1, int(len(labels) * min_label_ratio))
        # Frame-level `canonical_labels` gates candidates without paying for a
        # nested query.
        gate = {
            "bool": {
                "should": [{"term": {"canonical_labels": label}} for label in labels],
                "minimum_should_match": required,
            }
        }
        # One nested clause carries both jobs: it ranks which frames survive the
        # `size` cut (colour/position matches score higher) and returns exactly
        # the detections the assignment can use.
        boosts: list[dict[str, Any]] = []
        for label in labels:
            color = (colors or {}).get(label)
            if color:
                boosts.append({
                    "bool": {
                        "filter": [
                            {"term": {"detections.canonical_label": label}},
                            {"term": {"detections.dominant_color": color}},
                            {"term": {"detections.color_reliable": True}},
                        ]
                    }
                })
            position = (positions or {}).get(label)
            if position:
                boosts.append({
                    "bool": {
                        "filter": [
                            {"term": {"detections.canonical_label": label}},
                            {"term": {"detections.position_tags": position}},
                        ]
                    }
                })

        # Excluded labels ride along in the same nested clause: the scorer applies
        # their presence as a penalty, and it can only do that if those detections
        # are actually returned. They never gate candidate selection — a missed
        # detection must not be read as proof of absence.
        wanted_labels = list(dict.fromkeys([*labels, *(exclude_labels or [])]))
        nested = {
            "nested": {
                "path": "detections",
                "score_mode": "max",
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"detections.canonical_label": wanted_labels}},
                            {"range": {"detections.conf": {"gte": min_conf}}},
                        ],
                        "should": boosts,
                    }
                },
                "inner_hits": {
                    "name": "objects",
                    "size": _INNER_HITS_SIZE,
                    "_source": _DETECTION_FIELDS,
                },
            }
        }

        scope = elastic_filter_clause(categories)
        body = {
            "size": size,
            "track_total_hits": False,
            "timeout": "5s",
            "_source": _SOURCE_FIELDS,
            "query": {
                "bool": {
                    "filter": [{"term": {"status": "ok"}}, gate, *([scope] if scope else [])],
                    "must": [nested],
                }
            },
        }
        resp = await self._http.get().post(
            f"{self._base}/{self.s.idx_objects}/_search",
            json=body,
            headers=self._headers(),
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()

        out: list[dict[str, Any]] = []
        for hit in data.get("hits", {}).get("hits", []):
            frame = _normalize_frame(hit)
            if frame:
                out.append(frame)
        return out


def _normalize_frame(hit: dict[str, Any]) -> dict[str, Any] | None:
    source = hit.get("_source", {})
    video_id = source.get("video_id")
    frame_name = str(source.get("frame_name") or "")
    if not video_id or not frame_name.isdigit():
        return None
    keyframe_n = int(frame_name)
    inner = hit.get("inner_hits", {}).get("objects", {}).get("hits", {}).get("hits", [])
    return {
        "submit_keyframe_id": source.get("submit_keyframe_id")
        or make_submit_keyframe_id(video_id, keyframe_n),
        "video_id": video_id,
        "keyframe_n": keyframe_n,
        "width": source.get("width"),
        "height": source.get("height"),
        "detection_count": source.get("detection_count"),
        "detections": [nested.get("_source", {}) for nested in inner],
    }
