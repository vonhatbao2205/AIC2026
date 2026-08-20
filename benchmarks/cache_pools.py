"""Chụp lại kho ứng viên (candidate pool) của từng truy vấn có ground truth.

Bộ sinh 100 đáp án chỉ ăn đầu ra của bước gom frame theo video. Tách đôi như vậy
để việc dò tham số không phải gọi lại Milvus/Elastic hàng nghìn lần: chạy live
đúng một lần cho mỗi truy vấn, ghi toàn bộ frame đã hợp nhất xuống đĩa, sau đó
`run_answer_gen.py` dò tham số hoàn toàn offline trên file này.

Cách dùng:
    backend/.venv/bin/python benchmarks/cache_pools.py --profile infoshotpp --side L
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from app.config import get_settings  # noqa: E402
from app.services.search_service import SearchService  # noqa: E402
from app.services.trake_service import TrakeService  # noqa: E402
from gt_loader import load_qa, load_tkis, load_trake  # noqa: E402

CACHE_DIR = ROOT / "benchmarks" / "cache"

# Sâu hơn mức thanh trượt của console: kho ứng viên phải đủ để mô phỏng lại mọi
# giá trị pool_depth nhỏ hơn khi dò tham số offline.
CACHE_TOP_K = 500
CACHE_MAX_VIDEOS = 100


def _side_ok(videos: set[str], side: str) -> bool:
    if side == "all":
        return True
    return bool(videos) and all(v[:1].upper() == side.upper() for v in videos)


def collect_queries(side: str, kinds: set[str]) -> list[dict]:
    """Các truy vấn có GT, đã lọc theo phía dữ liệu (L21-L30 hay K01-K20)."""
    out: list[dict] = []
    if "kis" in kinds:
        tkis, _ = load_tkis()
        for q in tkis:
            # Một dòng TRAKE bị soạn nhầm vào sheet T-KIS; nó không có query id thật.
            if not q.query_id.endswith("-kis"):
                continue
            if not _side_ok(q.gt_videos, side):
                continue
            out.append(
                {
                    "query_id": q.query_id,
                    "kind": "kis",
                    "query_type_hint": "T-KIS",
                    "text": q.text,
                    "gt": [[v, f] for v, f in q.gt],
                    "dbc": q.dbc,
                }
            )
    if "qa" in kinds:
        for q in load_qa():
            if not q.gt or not _side_ok(q.gt_videos, side):
                continue
            out.append(
                {
                    "query_id": q.query_id,
                    "kind": "qa",
                    "query_type_hint": "QA",
                    "text": q.text,
                    "gt": [[v, f] for v, f, _ in q.gt],
                    "gt_answers": sorted({a for _, _, a in q.gt if a}),
                }
            )
    if "trake" in kinds:
        for q in load_trake():
            if not q.alternatives:
                continue
            videos = {v for alt in q.alternatives.values() for _, v, _ in alt}
            if not _side_ok(videos, side):
                continue
            out.append(
                {
                    "query_id": q.query_id,
                    "kind": "trake",
                    "query_type_hint": "TRAKE",
                    "text": q.text,
                    "n_events": q.n_events,
                    "alternatives": {
                        k: [[o, v, f] for o, v, f in alt] for k, alt in q.alternatives.items()
                    },
                }
            )
    return out


def slim_frame(frame: dict) -> dict:
    """Chỉ giữ những trường bộ sinh đáp án thật sự đọc — file cache nhỏ đi ~10 lần."""
    return {
        "submit_keyframe_id": frame.get("submit_keyframe_id"),
        "keyframe_n": frame.get("keyframe_n"),
        "frame_idx": frame.get("frame_idx"),
        "fps": frame.get("fps"),
        "pts_time": frame.get("pts_time"),
        "score": frame.get("score"),
        "channels": frame.get("channels") or [],
    }


def slim_group(group: dict) -> dict:
    return {
        "video_id": group.get("video_id"),
        "video_score": group.get("video_score"),
        "max_score": group.get("max_score"),
        "mean_top_score": group.get("mean_top_score"),
        "frame_count": group.get("frame_count"),
        "ambiguous": group.get("ambiguous"),
        "channels": group.get("channels") or [],
        "frames": [slim_frame(f) for f in (group.get("frames") or [])],
    }


async def run_one(
    search: SearchService, trake: TrakeService, query: dict, scope_mode: str, profile: str
) -> dict:
    payload = {
        "retrieval_database": profile,
        "query": query["text"],
        "query_type_hint": query["query_type_hint"],
        "scope": {"mode": scope_mode, "categories": []},
        "previous_hints": [],
        "manual_overrides": {"force_channels": [], "disable_channels": []},
        "use_llm": False,
        "expand": False,
        "top_k": CACHE_TOP_K,
        "max_videos": CACHE_MAX_VIDEOS,
    }
    t0 = time.perf_counter()
    res = await search.search(payload)
    record = {
        **query,
        "scope": res.get("scope"),
        "warnings": res.get("warnings") or [],
        "latency_ms": res.get("latency_ms"),
        "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
        "groups": [slim_group(g) for g in res.get("groups") or []],
    }
    if query["kind"] == "trake":
        # TRAKE cần thêm chuỗi sự kiện do DP dựng, không chỉ frame rời rạc.
        tres = await trake.search_trake({**payload, "top_k": max(CACHE_TOP_K, 400)})
        record["trake_sequences"] = tres.get("sequences") or []
        record["trake_events"] = [
            {"index": e.get("event_index"), "text": e.get("description") or e.get("text")}
            for e in (tres.get("events") or [])
        ]
    return record


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="infoshotpp", choices=["infoshotpp", "btc"])
    ap.add_argument("--side", default="L", choices=["L", "K", "all"])
    ap.add_argument("--scope", default="auto", choices=["auto", "all"])
    ap.add_argument("--kinds", default="kis,qa,trake")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    queries = collect_queries(args.side, kinds)
    if args.limit:
        queries = queries[: args.limit]
    print(f"{len(queries)} truy vấn có GT (side={args.side}, kinds={sorted(kinds)})", flush=True)

    settings = get_settings().for_retrieval_database(args.profile)
    search = SearchService(settings)
    trake = TrakeService(settings, search)

    records = []
    for i, query in enumerate(queries, start=1):
        try:
            record = await run_one(search, trake, query, args.scope, args.profile)
        except Exception as exc:  # ghi lại chứ không nuốt lỗi
            print(f"  [{i}/{len(queries)}] {query['query_id']} LỖI: {exc}", flush=True)
            records.append({**query, "error": str(exc), "groups": []})
            continue
        groups = record["groups"]
        gt_videos = {v for v, _ in (record.get("gt") or [])} or {
            v for alt in (record.get("alternatives") or {}).values() for _, v, _ in alt
        }
        rank = next(
            (j for j, g in enumerate(groups, start=1) if g["video_id"] in gt_videos), None
        )
        records.append(record)
        print(
            f"  [{i}/{len(queries)}] {query['query_id']}: {len(groups)} video, "
            f"gt_video_rank={rank} ({record['wall_ms']:.0f} ms)",
            flush=True,
        )

    out = {
        "profile": args.profile,
        "side": args.side,
        "scope_mode": args.scope,
        "top_k": CACHE_TOP_K,
        "max_videos": CACHE_MAX_VIDEOS,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "queries": records,
    }
    dest = (
        Path(args.out)
        if args.out
        else CACHE_DIR / f"pools_{args.profile}_{args.side}_{args.scope}.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    size_mb = dest.stat().st_size / 1e6
    print(f"→ {dest} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    asyncio.run(main())
