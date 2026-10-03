"""Evidence with provenance, de-duplication and bounded state for one run."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .compiler import compile_constraints


class EvidenceBoard:
    def __init__(self, query: str, query_type: str):
        self.query = query
        self.query_type = query_type
        self.constraints = compile_constraints(query, query_type)
        self.candidates: dict[str, dict] = {}
        self.retrieval_order: list[str] = []
        self.inspected: dict[str, set[str]] = {}
        self.history: list[dict] = []
        self.local_text: dict[tuple, dict] = {}
        self.version = 0
        self.closed = False

    def add_frame(self, frame: dict, *, source: str, at: float = 0.0) -> dict | None:
        if self.closed:
            return None
        key = frame.get("submit_keyframe_id")
        if not key or not frame.get("video_id"):
            return None
        if key not in self.candidates:
            if len(self.candidates) >= 300:
                # Reserve room for a late rescue even after broad exploration.
                victim = next((k for k in reversed(self.candidates) if not self.candidates[k]["reports"]), None)
                if source not in {"codex", "claude"} or victim is None:
                    return None
                self.candidates.pop(victim)
                if victim in self.retrieval_order:
                    self.retrieval_order.remove(victim)
            self.candidates[key] = {"frame": dict(frame), "sources": [], "evidence": {},
                                    "reports": {}, "first_seen_s": at, "scores": {}, "probability": None}
        c = self.candidates[key]
        c["frame"].update({k: v for k, v in frame.items() if v is not None})
        if source not in c["sources"]:
            c["sources"].append(source)
        for item in frame.get("evidence") or []:
            if item.get("text"):
                self.add_evidence(key, str(item.get("type") or "retrieval"), item["text"],
                                  str(item.get("source_id") or key))
        if frame.get("overlay"):
            self.add_evidence(key, "metadata", json.dumps(frame["overlay"], ensure_ascii=False), key)
        t = c["frame"].get("pts_time")
        if t is not None:
            for item in self.local_text.values():
                if item["video_id"] == frame["video_id"] and item["start"] - 5 <= t <= item["end"] + 5:
                    self.add_evidence(key, item["kind"], item["text"], item["source_id"])
        return c

    def remember_text(self, video_id, start, end, kind, text, source_id):
        if self.closed or not str(text).strip():
            return
        self.local_text[(kind, source_id)] = {"video_id": video_id, "start": start, "end": end,
                                            "kind": kind, "text": text, "source_id": source_id}
        while len(self.local_text) > 1000:
            self.local_text.pop(next(iter(self.local_text)))
        for key, candidate in self.candidates.items():
            frame = candidate["frame"]
            t = frame.get("pts_time")
            if frame["video_id"] == video_id and t is not None and start - 5 <= t <= end + 5:
                self.add_evidence(key, kind, text, source_id)

    def seed(self, result: dict, *, source: str = "retrieval", at: float = 0.0):
        if self.closed:
            return
        for group in result.get("groups") or []:
            for frame in group.get("frames") or []:
                c = self.add_frame(frame, source=source, at=at)
                key = frame.get("submit_keyframe_id")
                if c is not None and source == "retrieval" and key not in self.retrieval_order:
                    self.retrieval_order.append(key)
        self.version += 1

    def add_evidence(self, key: str, kind: str, text: str, source_id: str):
        if self.closed or key not in self.candidates or not str(text).strip():
            return
        # The same text/frame rewrapped by different tools remains one source.
        digest = hashlib.sha256(f"{kind}:{source_id}".encode()).hexdigest()[:16]
        c = self.candidates[key]
        if digest not in c["evidence"] and len(c["evidence"]) >= 24:
            # Reserve visual observations even when a dense OCR window filled
            # the board first. Later text cannot evict inspected observations.
            if kind != "agent_visual_observation":
                return
            victim = next((k for k, e in c["evidence"].items() if e["kind"] != "agent_visual_observation"), None)
            if victim is None:
                return
            del c["evidence"][victim]
        if digest not in c["evidence"] or c["evidence"][digest]["text"] != str(text)[:1800]:
            c["evidence"][digest] = {"id": digest, "kind": kind, "source_id": source_id, "text": str(text)[:1800]}
            c["probability"] = None
            c["scores"] = {}
            self.version += 1

    def report(self, candidate: dict, at: float):
        agent, key = candidate["agent"], candidate["submit_keyframe_id"]
        c = self.add_frame(candidate, source=agent, at=at)
        if c is None:
            return
        c["reports"][agent] = {k: candidate.get(k) for k in ("confidence", "reason", "answer", "event", "time")}
        # Updating an answer/event changes what must be verified, even when the
        # accompanying visual observation has not changed.
        c["probability"] = None
        c["scores"] = {}
        # A claim after looking at a real frame is textual visual evidence. A
        # claim from retrieval alone stays a report, not verified evidence.
        if key in self.inspected.get(agent, set()) and candidate.get("reason"):
            self.add_evidence(key, "agent_visual_observation", candidate["reason"], f"{agent}:{key}")
        self.version += 1

    def ranked(self, *, verified: bool = True) -> list[dict]:
        # Reciprocal-rank fusion for unverified baselines. Self-confidence only
        # orders an agent's list; it is never interpreted as calibrated truth.
        rrf: dict[str, float] = {}
        for i, key in enumerate(self.retrieval_order, 1):
            rrf[key] = 1 / (60 + i)
        for agent in ("codex", "claude"):
            keys = sorted((k for k, c in self.candidates.items() if agent in c["reports"]),
                          key=lambda k: -float(self.candidates[k]["reports"][agent].get("confidence") or 0))
            for i, key in enumerate(keys, 1):
                rrf[key] = rrf.get(key, 0) + 1 / (60 + i)
        def verification_order(candidate):
            if not verified or candidate["probability"] is None:
                return (0, 0)
            scores = list(candidate["scores"].values())
            # Unknown is abstention, not a reason to promote a shortlisted
            # candidate over an unassessed one. Preserve RRF in that case.
            if scores and all(s["supported"] > max(s["unknown"], s["refuted"]) for s in scores):
                return (-1, -candidate["probability"])
            if any(s["refuted"] > max(s["unknown"], s["supported"]) for s in scores):
                return (1, -candidate["probability"])
            return (0, 0)
        keys = sorted(self.candidates, key=lambda k: (
            *verification_order(self.candidates[k]), -rrf.get(k, 0), self.candidates[k]["first_seen_s"], k))
        return [{**self.candidates[k]["frame"], "submit_keyframe_id": k,
                 "agent_sources": [s for s in self.candidates[k]["sources"] if s in {"codex", "claude"}],
                 "probability": self.candidates[k]["probability"],
                 "raw_probability": self.candidates[k].get("raw_probability"),
                 "constraint_scores": self.candidates[k]["scores"],
                 "fusion_score": rrf.get(k, 0), "first_seen_s": self.candidates[k]["first_seen_s"]} for k in keys]

    def shortlist(self, limit: int) -> list[str]:
        # One leading proposal per agent, not two from whichever agent has
        # higher self-confidence. Include the current verified leader as well.
        ranked = self.ranked(verified=False)
        base = [f["submit_keyframe_id"] for f in ranked]
        proposed = []
        for agent in ("codex", "claude"):
            keys = [k for k in base if agent in self.candidates[k]["reports"]]
            if keys:
                proposed.append(max(keys, key=lambda k: float(self.candidates[k]["reports"][agent].get("confidence") or 0)))
        verified = [f["submit_keyframe_id"] for f in self.ranked() if f["probability"] is not None]
        ordered = list(dict.fromkeys(proposed + verified[:1] + base))
        if self.query_type == "TRAKE" and ordered:
            # Cover the leading video's events before spending the remaining
            # slots on alternatives. The context stays bounded to 12 frames.
            leader = self.candidates[ordered[0]]["frame"]["video_id"]
            event_keys = []
            for constraint in self.constraints:
                if constraint.event is not None:
                    key = next((k for k in ordered if self.candidates[k]["frame"]["video_id"] == leader
                                and self.candidates[k]["frame"].get("event") == constraint.event), None)
                    if key:
                        event_keys.append(key)
            limit = min(12, max(limit, len(event_keys)))
            return list(dict.fromkeys(event_keys + ordered))[:limit]
        # Prefer distinct moments over near-duplicate adjacent retrieval frames.
        selected = list(dict.fromkeys(proposed + verified[:1]))[:limit]
        for key in ordered:
            if key in selected:
                continue
            f = self.candidates[key]["frame"]
            same_moment = any(self.candidates[k]["frame"]["video_id"] == f["video_id"]
                              and self.candidates[k]["frame"].get("pts_time") is not None
                              and f.get("pts_time") is not None
                              and abs(self.candidates[k]["frame"]["pts_time"] - f["pts_time"]) <= 8 for k in selected)
            if not same_moment:
                selected.append(key)
            if len(selected) >= limit:
                break
        return list(dict.fromkeys(selected + ordered))[:limit]

    def disagreement(self) -> bool | None:
        tops = []
        for agent in ("codex", "claude"):
            rows = [c for c in self.candidates.values() if agent in c["reports"]]
            if not rows:
                return None
            tops.append(max(rows, key=lambda c: float(c["reports"][agent].get("confidence") or 0))["frame"])
        a, b = tops
        return not (a["video_id"] == b["video_id"] and a.get("pts_time") is not None
                    and b.get("pts_time") is not None and abs(a["pts_time"] - b["pts_time"]) <= 8)

    def state(self, limit: int = 5) -> dict[str, Any]:
        keys = self.shortlist(limit)
        def evidence(key):
            items = list(self.candidates[key]["evidence"].values())
            visual = [e for e in items if e["kind"] == "agent_visual_observation"]
            # Balance modalities and collapse repeated text only for context
            # selection; the stored provenance remains available for auditing.
            selected = visual[-4:]
            seen = {e["text"].casefold().strip() for e in selected}
            pools = {}
            for e in reversed(items):
                if e["kind"] != "agent_visual_observation":
                    pools.setdefault(e["kind"], []).append(e)
            while len(selected) < 8 and any(pools.values()):
                for pool in pools.values():
                    while pool:
                        e = pool.pop(0)
                        text = e["text"].casefold().strip()
                        if text not in seen:
                            selected.append(e)
                            seen.add(text)
                            break
                    if len(selected) >= 8:
                        break
            return [{**e, "text": e["text"][:600]} for e in selected]
        return {"query": self.query, "query_type": self.query_type,
                "constraints": [c.to_dict() for c in self.constraints],
                "agent_disagreement": self.disagreement(),
                "candidates": [{"key": k, "video_id": self.candidates[k]["frame"]["video_id"],
                                "time": self.candidates[k]["frame"].get("pts_time"),
                                "event": self.candidates[k]["frame"].get("event"),
                                "answer": self.candidates[k]["frame"].get("answer"),
                                "retrieval_scores": self.candidates[k]["frame"].get("per_channel_score", {}),
                                "evidence": evidence(k),
                                "reports": self.candidates[k]["reports"],
                                "constraint_scores": self.candidates[k]["scores"]} for k in keys],
                "recent_tools": self.history[-8:]}
