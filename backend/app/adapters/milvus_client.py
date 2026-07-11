"""Milvus/Zilliz adapter for PE-Core-G14 image vector search.

Searches `aic26_image_peg14_v1` (COSINE). Hit ids ARE submit_keyframe_id, so we
parse video_id / keyframe_n from the id. In mock mode returns deterministic
image hits keyed off the query string.
"""
from __future__ import annotations

from typing import Any

from ..config import Settings
from ..identity import parse_submit_keyframe_id
from .. import mock_data


_FLOAT16_MIN_NORMAL = 2**-14


def _sanitize_float16_vector(vector: list[float]) -> list[float]:
    """Zero values that Milvus rejects as float16 underflow.

    The audio collection uses FLOAT16_VECTOR. Some normalized GLAP text
    embeddings contain values below the smallest normal float16 magnitude;
    replacing those negligible components with zero keeps cosine search stable.
    """
    return [0.0 if 0.0 < abs(value) < _FLOAT16_MIN_NORMAL else float(value) for value in vector]


class MilvusClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.mock = settings.mock_mode or not settings.has_milvus
        self._client = None

    def _connect(self):
        if self._client is None:
            from pymilvus import MilvusClient as _MC  # imported lazily

            self._client = _MC(uri=self.s.milvus_endpoint, token=self.s.milvus_token)
        return self._client

    async def health(self) -> dict[str, Any]:
        if self.mock:
            return {"ok": True, "mode": "mock"}
        try:
            client = self._connect()
            has = client.has_collection(self.s.milvus_image_collection)
            return {"ok": bool(has), "collection": self.s.milvus_image_collection}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def search_image(
        self, vector: list[float], *, top_k: int = 100, video_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return image hits: submit_keyframe_id, video_id, keyframe_n, score (0..1).

        `video_id` restricts the search to one video (used by TRAKE pass-2 to find a
        video's best frame for a specific event)."""
        if self.mock:
            return self._mock_image(top_k=top_k, video_id=video_id)

        client = self._connect()
        results = client.search(
            collection_name=self.s.milvus_image_collection,
            data=[vector],
            limit=top_k,
            output_fields=["submit_keyframe_id", "video_id", "keyframe_n"],
            search_params={"metric_type": "COSINE"},
            filter=f'video_id == "{video_id}"' if video_id else "",
        )
        out: list[dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            kf_id = entity.get("submit_keyframe_id") or (hit.get("id") if isinstance(hit, dict) else None)
            if not kf_id:
                continue
            try:
                parsed = parse_submit_keyframe_id(kf_id)
            except ValueError:
                continue
            distance = hit.get("distance") if isinstance(hit, dict) else getattr(hit, "distance", 0.0)
            out.append(
                {
                    "submit_keyframe_id": parsed.submit_keyframe_id,
                    "video_id": entity.get("video_id") or parsed.video_id,
                    "keyframe_n": entity.get("keyframe_n") or parsed.keyframe_n,
                    "score": float(distance or 0.0),  # COSINE similarity
                }
            )
        return out

    def search_audio(self, vector: list[float], *, top_k: int = 100) -> list[dict[str, Any]]:
        """GLAP audio-vector search over `aic26_audio_glap_v1` (COSINE). Hits carry
        submit_keyframe_id / video_id / keyframe_n / top1_label (window-level)."""
        if self.mock:
            return self._mock_audio(top_k=top_k)
        client = self._connect()
        results = client.search(
            collection_name=self.s.milvus_audio_collection,
            data=[_sanitize_float16_vector(vector)],
            limit=top_k,
            output_fields=["submit_keyframe_id", "video_id", "keyframe_n", "top1_label", "start", "end"],
            search_params={"metric_type": "COSINE"},
        )
        out: list[dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            kf_id = entity.get("submit_keyframe_id")
            if not kf_id:
                continue
            try:
                parsed = parse_submit_keyframe_id(kf_id)
            except ValueError:
                continue
            distance = hit.get("distance") if isinstance(hit, dict) else getattr(hit, "distance", 0.0)
            out.append(
                {
                    "submit_keyframe_id": parsed.submit_keyframe_id,
                    "video_id": entity.get("video_id") or parsed.video_id,
                    "keyframe_n": entity.get("keyframe_n") or parsed.keyframe_n,
                    "top1_label": entity.get("top1_label"),
                    "window_id": hit.get("id") if isinstance(hit, dict) else None,
                    "start": entity.get("start"),
                    "end": entity.get("end"),
                    "score": float(distance or 0.0),
                }
            )
        return out

    def _mock_image(self, *, top_k: int, video_id: str | None = None) -> list[dict[str, Any]]:
        out = []
        rank = 0
        items = (
            [(video_id, mock_data.MOCK_KEYFRAMES.get(video_id, []))]
            if video_id
            else list(mock_data.MOCK_KEYFRAMES.items())
        )
        for video_id, frames in items:
            for f in frames[:8]:
                out.append(
                    {
                        "submit_keyframe_id": f["submit_keyframe_id"],
                        "video_id": video_id,
                        "keyframe_n": f["keyframe_n"],
                        "score": round(0.95 - rank * 0.01, 4),
                    }
                )
                rank += 1
        out.sort(key=lambda x: -x["score"])
        return out[:top_k]

    def _mock_audio(self, *, top_k: int) -> list[dict[str, Any]]:
        out, rank = [], 0
        for video_id, frames in mock_data.MOCK_KEYFRAMES.items():
            for f in frames[:4]:
                out.append(
                    {
                        "submit_keyframe_id": f["submit_keyframe_id"],
                        "video_id": video_id,
                        "keyframe_n": f["keyframe_n"],
                        "top1_label": "Music",
                        "window_id": f"{video_id}_a{rank:06d}",
                        "start": f["pts_time"],
                        "end": f["pts_time"] + 5,
                        "score": round(0.9 - rank * 0.02, 4),
                    }
                )
                rank += 1
        out.sort(key=lambda x: -x["score"])
        return out[:top_k]
