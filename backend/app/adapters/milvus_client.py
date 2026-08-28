"""Milvus/Zilliz adapter for PE-Core-G14 image vector search.

Searches `aic26_image_peg14_v1` (COSINE). Hit ids ARE submit_keyframe_id, so we
parse video_id / keyframe_n from the id. In mock mode returns deterministic
image hits keyed off the query string.
"""
from __future__ import annotations

from typing import Any

from ..config import Settings
from ..identity import parse_submit_keyframe_id
from ..scope import milvus_filter_expr
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

    async def health_qwen_image(self) -> dict[str, Any]:
        """Check the independent InfoShot++ Qwen3-VL image collection."""
        if not self.s.is_infoshotpp:
            return {
                "ok": False,
                "mode": "disabled",
                "reason": "Qwen3-VL embeddings are indexed only for InfoShot++",
            }
        if self.s.mock_mode:
            return {
                "ok": True,
                "mode": "mock",
                "collection": self.s.milvus_qwen3_vl_image_collection,
            }
        if not self.s.has_milvus:
            return {
                "ok": False,
                "mode": "disabled",
                "reason": "MILVUS_ENDPOINT_2/TOKEN_2 are not configured",
            }
        try:
            client = self._connect()
            collection = self.s.milvus_qwen3_vl_image_collection
            has = client.has_collection(collection)
            return {"ok": bool(has), "collection": collection}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def search_image(
        self,
        vector: list[float],
        *,
        top_k: int = 100,
        video_id: str | None = None,
        categories: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        """Return image hits: submit_keyframe_id, video_id, keyframe_n, score (0..1).

        `video_id` restricts the search to one video (used by TRAKE pass-2 to find a
        video's best frame for a specific event). `categories` restricts it to a set
        of dataset folders — pushed into the Milvus filter so the top_k the operator
        asked for is filled from INSIDE the scope, rather than retrieved globally
        and then thinned out by a post-filter."""
        return self._search_image_collection(
            self.s.milvus_image_collection,
            vector,
            top_k=top_k,
            video_id=video_id,
            categories=categories,
        )

    def search_qwen_image(
        self,
        vector: list[float],
        *,
        top_k: int = 100,
        video_id: str | None = None,
        categories: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        """Search the 4096-d Qwen3-VL InfoShot++ collection.

        This is intentionally a separate method from PE search: callers can
        never pass a Qwen vector to PE's collection (or vice versa) by choosing
        the wrong optional argument.
        """
        if not self.s.is_infoshotpp:
            raise RuntimeError("Qwen3-VL image collection is available only for InfoShot++")
        if not self.s.mock_mode and not self.s.has_milvus:
            raise RuntimeError("InfoShot++ Milvus endpoint/token are not configured")
        return self._search_image_collection(
            self.s.milvus_qwen3_vl_image_collection,
            vector,
            top_k=top_k,
            video_id=video_id,
            categories=categories,
        )

    def _search_image_collection(
        self,
        collection_name: str,
        vector: list[float],
        *,
        top_k: int,
        video_id: str | None,
        categories: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        if self.mock:
            return self._mock_image(top_k=top_k, video_id=video_id, categories=categories)

        client = self._connect()
        clauses = [
            expr
            for expr in (
                f'video_id == "{video_id}"' if video_id else "",
                milvus_filter_expr(categories),
            )
            if expr
        ]
        results = client.search(
            collection_name=collection_name,
            data=[vector],
            limit=top_k,
            output_fields=["submit_keyframe_id", "video_id", "keyframe_n"],
            search_params={"metric_type": "COSINE"},
            filter=" and ".join(clauses),
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

    def get_image_vectors(self, submit_keyframe_ids: list[str]) -> dict[str, list[float]]:
        """Fetch stored PE-G14 embeddings by primary key.

        The image collection's PK `id` IS the submit_keyframe_id (see
        milvus_upload.py), so a marked frame can seed an image-to-image kNN
        without re-encoding the picture.
        """
        return self._get_image_vectors_from_collection(
            self.s.milvus_image_collection, submit_keyframe_ids
        )

    def get_qwen_image_vectors(
        self, submit_keyframe_ids: list[str]
    ) -> dict[str, list[float]]:
        """Fetch stored Qwen vectors for model-consistent relevance feedback."""
        if not self.s.is_infoshotpp:
            raise RuntimeError("Qwen3-VL image collection is available only for InfoShot++")
        if not self.s.mock_mode and not self.s.has_milvus:
            raise RuntimeError("InfoShot++ Milvus endpoint/token are not configured")
        return self._get_image_vectors_from_collection(
            self.s.milvus_qwen3_vl_image_collection, submit_keyframe_ids
        )

    def _get_image_vectors_from_collection(
        self, collection_name: str, submit_keyframe_ids: list[str]
    ) -> dict[str, list[float]]:
        ids = [kf_id for kf_id in dict.fromkeys(submit_keyframe_ids) if kf_id]
        if not ids:
            return {}
        if self.mock:
            return {kf_id: [0.1] * 8 for kf_id in ids}
        client = self._connect()
        rows = client.get(
            collection_name=collection_name,
            ids=ids,
            output_fields=["id", "submit_keyframe_id", "embedding"],
        )
        out: dict[str, list[float]] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            kf_id = row.get("submit_keyframe_id") or row.get("id")
            vector = row.get("embedding")
            if not kf_id or vector is None:
                continue
            out[str(kf_id)] = [float(value) for value in vector]
        return out

    def search_audio(
        self, vector: list[float], *, top_k: int = 100, categories: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]:
        """GLAP audio-vector search over `aic26_audio_glap_v1` (COSINE). Hits carry
        submit_keyframe_id / video_id / keyframe_n / top1_label (window-level)."""
        if self.mock:
            return self._mock_audio(top_k=top_k, categories=categories)
        client = self._connect()
        results = client.search(
            collection_name=self.s.milvus_audio_collection,
            data=[_sanitize_float16_vector(vector)],
            limit=top_k,
            output_fields=["submit_keyframe_id", "video_id", "keyframe_n", "top1_label", "start", "end"],
            search_params={"metric_type": "COSINE"},
            filter=milvus_filter_expr(categories),
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

    def _mock_image(
        self, *, top_k: int, video_id: str | None = None, categories: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]:
        out = []
        rank = 0
        items = (
            [(video_id, mock_data.MOCK_KEYFRAMES.get(video_id, []))]
            if video_id
            else list(mock_data.MOCK_KEYFRAMES.items())
        )
        for video_id, frames in _in_categories(items, categories):
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

    def _mock_audio(self, *, top_k: int, categories: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        out, rank = [], 0
        for video_id, frames in _in_categories(list(mock_data.MOCK_KEYFRAMES.items()), categories):
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


def _in_categories(
    items: list[tuple[str, list[dict[str, Any]]]], categories: tuple[str, ...]
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Mock-mode stand-in for the Milvus category filter (same prefix rule)."""
    if not categories:
        return items
    wanted = set(categories)
    return [(vid, frames) for vid, frames in items if vid and vid.split("_")[0] in wanted]
