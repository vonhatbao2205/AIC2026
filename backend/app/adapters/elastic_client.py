"""Elastic adapter for OCR / speech / audio / keyframe-map indices.

Search methods return raw scored hits (dicts). The search service converts them
to ranked `ChannelHit`s. Quality demotions from `app.scoring` are applied here so
they influence per-channel ordering (and therefore RRF rank).

In mock mode, returns deterministic matches from `app.mock_data`.
"""
from __future__ import annotations

from typing import Any


from .. import mock_data
from ..config import Settings
from ..scoring import audio_score_multiplier, speech_score_multiplier
from .http_pool import PooledHttpClient


class ElasticClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.mock = settings.mock_mode or not settings.has_elastic
        self._base = (settings.elastic_endpoint or "").rstrip("/")
        self._http = PooledHttpClient()

    # ---- HTTP plumbing -------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"ApiKey {self.s.elastic_api_key}",
            "Content-Type": "application/json",
        }

    async def _search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base}/{index}/_search"
        resp = await self._http.get().post(url, json=body, headers=self._headers(), timeout=20.0)
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> dict[str, Any]:
        if self.mock:
            return {"ok": True, "mode": "mock"}
        try:
            resp = await self._http.get().get(f"{self._base}/", headers=self._headers(), timeout=10.0)
            resp.raise_for_status()
            info = resp.json()
            return {"ok": True, "cluster": info.get("cluster_name")}
        except Exception as exc:  # noqa: BLE001 - surfaced as health warning
            return {"ok": False, "error": str(exc)}

    # ---- OCR -----------------------------------------------------------
    async def search_ocr(
        self,
        queries_vi: list[str],
        queries_folded: list[str] | None = None,
        *,
        exact_phrases: list[str] | None = None,
        hour: int | None = None,
        clock: str | None = None,
        size: int = 100,
    ) -> list[dict[str, Any]]:
        # No query text and no time filter => nothing to search (never match_all).
        if not queries_vi and not (queries_folded or []) and not (exact_phrases or []) and hour is None and not clock:
            return []
        if self.mock:
            return self._mock_ocr(queries_vi + (queries_folded or []), hour=hour, size=size)

        should: list[dict] = []
        for q in queries_vi:
            should.append({"match": {"text_clean": {"query": q, "boost": 2.0}}})
            should.append({"match": {"text_nfc": {"query": q}}})
        for q in (queries_folded or []):
            should.append({"match": {"text_clean_fold": {"query": q, "fuzziness": "AUTO"}}})
        for p in (exact_phrases or []):
            should.append({"match_phrase": {"text_clean": {"query": p, "boost": 3.0}}})

        if not should:
            return []  # never fall through to match_all on an empty query

        filt: list[dict] = []
        if hour is not None:
            filt.append({"term": {"hour": hour}})
        if clock:
            filt.append({"term": {"clock": clock}})

        body = {
            "size": size,
            "query": {"bool": {"should": should, "minimum_should_match": 1, "filter": filt}},
        }
        data = await self._search(self.s.idx_ocr, body)
        out = []
        for h in data.get("hits", {}).get("hits", []):
            src = h.get("_source", {})
            out.append(
                {
                    "submit_keyframe_id": src.get("submit_keyframe_id"),
                    "video_id": src.get("video_id"),
                    "keyframe_n": src.get("keyframe_n"),
                    "score": float(h.get("_score") or 0.0),
                    "text_clean": src.get("text_clean") or src.get("text_nfc") or "",
                    "clock": src.get("clock"),
                    "hour": src.get("hour"),
                    "boxes": src.get("boxes") or [],
                }
            )
        return out

    # ---- Speech --------------------------------------------------------
    async def search_speech(
        self,
        queries_vi: list[str],
        *,
        exact_phrases: list[str] | None = None,
        size: int = 100,
    ) -> list[dict[str, Any]]:
        if not queries_vi and not (exact_phrases or []):
            return []
        if self.mock:
            return self._mock_speech(queries_vi, size=size)

        should: list[dict] = [{"match": {"text": {"query": q}}} for q in queries_vi]
        for p in (exact_phrases or []):
            should.append({"match_phrase": {"text": {"query": p, "boost": 3.0}}})
        if not should:
            return []
        body = {
            "size": size,
            "query": {"bool": {"should": should, "minimum_should_match": 1}},
        }
        data = await self._search(self.s.idx_speech, body)
        out = []
        for h in data.get("hits", {}).get("hits", []):
            src = h.get("_source", {})
            mult = speech_score_multiplier(src.get("confidence_bucket"), src.get("segment_role"))
            out.append(
                {
                    "submit_keyframe_id": src.get("submit_keyframe_id")
                    or src.get("center_submit_keyframe_id"),
                    "video_id": src.get("video_id"),
                    "keyframe_n": src.get("keyframe_n") or src.get("center_keyframe_n"),
                    "score": float(h.get("_score") or 0.0) * mult,
                    "text": src.get("text") or "",
                    "start": src.get("start"),
                    "end": src.get("end"),
                    "confidence_bucket": src.get("confidence_bucket"),
                    "segment_role": src.get("segment_role"),
                }
            )
        out.sort(key=lambda x: -x["score"])
        return out

    # ---- Audio ---------------------------------------------------------
    async def search_audio(
        self,
        queries_en: list[str],
        sound_labels: list[str] | None = None,
        *,
        size: int = 100,
    ) -> list[dict[str, Any]]:
        if not queries_en and not (sound_labels or []):
            return []
        if self.mock:
            return self._mock_audio(queries_en + (sound_labels or []), size=size)

        should: list[dict] = []
        for lbl in (sound_labels or []):
            should.append({"term": {"top1_label": {"value": lbl, "boost": 3.0}}})
            should.append({"terms": {"tag_labels": [lbl]}})
        for q in queries_en:
            should.append({"match": {"caption": {"query": q}}})
            should.append({"match": {"top1_label": {"query": q, "boost": 2.0}}})
            should.append({"match": {"tag_labels": {"query": q}}})
        if not should:
            return []
        body = {
            "size": size,
            "query": {"bool": {"should": should, "minimum_should_match": 1}},
        }
        data = await self._search(self.s.idx_audio, body)
        wanted = {s.lower() for s in (sound_labels or [])}
        out = []
        for h in data.get("hits", {}).get("hits", []):
            src = h.get("_source", {})
            top1 = (src.get("top1_label") or "")
            matched_top1 = top1.lower() in wanted if wanted else False
            mult = audio_score_multiplier(
                audio_stoplist_hit=bool(src.get("audio_stoplist_hit")),
                top1_is_stoplisted=bool(src.get("top1_is_stoplisted")),
                caption_quality=src.get("caption_quality"),
                matched_on_top1=matched_top1,
            )
            if mult <= 0:
                continue
            out.append(
                {
                    "submit_keyframe_id": src.get("submit_keyframe_id")
                    or src.get("center_submit_keyframe_id"),
                    "video_id": src.get("video_id"),
                    "keyframe_n": src.get("keyframe_n") or src.get("center_keyframe_n"),
                    "score": float(h.get("_score") or 0.0) * mult,
                    "top1_label": top1,
                    "caption": src.get("caption"),
                    "caption_quality": src.get("caption_quality"),
                    "start": src.get("start"),
                    "end": src.get("end"),
                    "window_id": src.get("window_id"),
                }
            )
        out.sort(key=lambda x: -x["score"])
        return out

    # ---- Keyframe map / timeline --------------------------------------
    async def get_keyframe(self, submit_keyframe_id: str) -> dict[str, Any] | None:
        if self.mock:
            return mock_data.keyframe_record(submit_keyframe_id)
        body = {"size": 1, "query": {"term": {"submit_keyframe_id": submit_keyframe_id}}}
        data = await self._search(self.s.idx_keyframe_map, body)
        hits = data.get("hits", {}).get("hits", [])
        return hits[0]["_source"] if hits else None

    async def get_keyframes_by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Batch-fetch keyframe-map docs by submit_keyframe_id (the doc _id) in a
        single _mget request. Returns {submit_keyframe_id: source}."""
        if not ids:
            return {}
        if self.mock:
            out: dict[str, dict[str, Any]] = {}
            for i in ids:
                rec = mock_data.keyframe_record(i)
                if rec:
                    out[i] = rec
            return out
        url = f"{self._base}/{self.s.idx_keyframe_map}/_mget"
        resp = await self._http.get().post(url, json={"ids": ids}, headers=self._headers(), timeout=20.0)
        resp.raise_for_status()
        data = resp.json()
        out2: dict[str, dict[str, Any]] = {}
        for doc in data.get("docs", []):
            if doc.get("found"):
                out2[doc["_id"]] = doc.get("_source", {})
        return out2

    async def get_video_keyframes(self, video_id: str, *, size: int = 5000) -> list[dict[str, Any]]:
        if self.mock:
            return list(mock_data.MOCK_KEYFRAMES.get(video_id, []))
        body = {
            "size": size,
            "query": {"term": {"video_id": video_id}},
            "sort": [{"keyframe_n": "asc"}],
        }
        data = await self._search(self.s.idx_keyframe_map, body)
        return [h["_source"] for h in data.get("hits", {}).get("hits", [])]

    async def get_video_speech(self, video_id: str, *, size: int = 2000) -> list[dict[str, Any]]:
        if self.mock:
            return list(mock_data.MOCK_SPEECH.get(video_id, []))
        body = {
            "size": size,
            "query": {"term": {"video_id": video_id}},
            "sort": [{"start": "asc"}],
        }
        data = await self._search(self.s.idx_speech, body)
        return [h["_source"] for h in data.get("hits", {}).get("hits", [])]

    async def get_video_ocr(self, video_id: str, *, size: int = 5000) -> list[dict[str, Any]]:
        if self.mock:
            out = []
            for kf_id, rec in mock_data.MOCK_OCR.items():
                if kf_id.split("/")[1] == video_id:
                    out.append({"submit_keyframe_id": kf_id, **rec})
            return out
        body = {
            "size": size,
            "query": {
                "bool": {
                    "must": [{"term": {"video_id": video_id}}],
                    "should": [{"exists": {"field": "text_clean"}}],
                }
            },
        }
        data = await self._search(self.s.idx_ocr, body)
        return [h["_source"] for h in data.get("hits", {}).get("hits", [])]

    async def get_video_audio(self, video_id: str, *, size: int = 5000) -> list[dict[str, Any]]:
        if self.mock:
            return list(mock_data.MOCK_AUDIO.get(video_id, []))
        body = {
            "size": size,
            "query": {"term": {"video_id": video_id}},
            "sort": [{"start": "asc"}],
        }
        data = await self._search(self.s.idx_audio, body)
        return [h["_source"] for h in data.get("hits", {}).get("hits", [])]

    # ---- Mock helpers --------------------------------------------------
    def _mock_ocr(self, queries: list[str], *, hour: int | None, size: int) -> list[dict]:
        terms = [q.lower() for q in queries if q]
        out = []
        for kf_id, rec in mock_data.MOCK_OCR.items():
            text = rec["text_clean"].lower()
            if hour is not None and rec.get("hour") != hour:
                continue
            if terms and not any(t in text or any(w in text for w in t.split()) for t in terms):
                continue
            vid = kf_id.split("/")[1]
            out.append(
                {
                    "submit_keyframe_id": kf_id,
                    "video_id": vid,
                    "keyframe_n": int(kf_id.split("/")[2]),
                    "score": 5.0,
                    "text_clean": rec["text_clean"],
                    "clock": rec.get("clock"),
                    "hour": rec.get("hour"),
                    "boxes": [],
                }
            )
        return out[:size]

    def _mock_speech(self, queries: list[str], *, size: int) -> list[dict]:
        terms = [w.lower() for q in queries for w in q.split()]
        out = []
        for video_id, segs in mock_data.MOCK_SPEECH.items():
            for seg in segs:
                text = seg["text"].lower()
                if terms and not any(t in text for t in terms):
                    continue
                kf = mock_data.make_submit_keyframe_id(
                    video_id, _nearest_kf_n(video_id, seg["center_time"])
                )
                mult = speech_score_multiplier(seg.get("confidence_bucket"), seg.get("segment_role"))
                out.append(
                    {
                        "submit_keyframe_id": kf,
                        "video_id": video_id,
                        "keyframe_n": int(kf.split("/")[2]),
                        "score": 4.0 * mult,
                        "text": seg["text"],
                        "start": seg["start"],
                        "end": seg["end"],
                        "confidence_bucket": seg.get("confidence_bucket"),
                        "segment_role": seg.get("segment_role"),
                    }
                )
        out.sort(key=lambda x: -x["score"])
        return out[:size]

    def _mock_audio(self, queries: list[str], *, size: int) -> list[dict]:
        terms = [q.lower() for q in queries if q]
        out = []
        for video_id, windows in mock_data.MOCK_AUDIO.items():
            for w in windows:
                hay = " ".join([w.get("top1_label") or "", *(w.get("tag_labels") or []), w.get("caption") or ""]).lower()
                if terms and not any(t in hay for t in terms):
                    continue
                matched_top1 = any(t in (w.get("top1_label") or "").lower() for t in terms)
                mult = audio_score_multiplier(
                    audio_stoplist_hit=bool(w.get("audio_stoplist_hit")),
                    top1_is_stoplisted=bool(w.get("top1_is_stoplisted")),
                    caption_quality=w.get("caption_quality"),
                    matched_on_top1=matched_top1,
                )
                if mult <= 0:
                    continue
                kf = mock_data.make_submit_keyframe_id(
                    video_id, _nearest_kf_n(video_id, w["center_time"])
                )
                out.append(
                    {
                        "submit_keyframe_id": kf,
                        "video_id": video_id,
                        "keyframe_n": int(kf.split("/")[2]),
                        "score": 4.0 * mult,
                        "top1_label": w.get("top1_label"),
                        "caption": w.get("caption"),
                        "caption_quality": w.get("caption_quality"),
                        "start": w["start"],
                        "end": w["end"],
                        "window_id": w["window_id"],
                    }
                )
        out.sort(key=lambda x: -x["score"])
        return out[:size]


def _nearest_kf_n(video_id: str, t: float) -> int:
    frames = mock_data.MOCK_KEYFRAMES.get(video_id, [])
    if not frames:
        return 1
    nearest = min(frames, key=lambda f: abs(f["pts_time"] - t))
    return nearest["keyframe_n"]
