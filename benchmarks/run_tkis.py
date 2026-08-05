"""Chạy lại toàn bộ T-KIS có ground truth trên hệ thống AIC26 đang chạy live.

Cách dùng:
    python benchmarks/run_tkis.py --config full
    python benchmarks/run_tkis.py --config pe_only --out runs/pe_only.json

Mỗi lần chạy ghi ra một file JSON gồm: cấu hình, thời điểm, và với từng query là
danh sách video đã xếp hạng, hạng của ground truth, độ trễ theo từng giai đoạn.
Có file này thì mới giải thích được vì sao một query đúng hay sai, thay vì chỉ
còn lại một con số tổng.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_loader import ROOT, load_tkis  # noqa: E402

API = "http://127.0.0.1:8000"

# Ablation: tắt dần các kênh để biết kênh nào thực sự đóng góp.
CONFIGS = {
    "full": {"disable_channels": []},
    "no_audio": {"disable_channels": ["audio"]},
    "no_speech": {"disable_channels": ["speech", "audio"]},
    "pe_only": {"disable_channels": ["ocr", "speech", "audio"]},
    "full_llm": {"disable_channels": [], "use_llm": True},
}

CUTOFFS = (1, 5, 10, 20, 50, 100)


def search(text: str, cfg: dict, top_k: int = 100) -> dict:
    payload = {
        "query": text,
        "top_k": top_k,
        "max_videos": 100,
        "use_llm": cfg.get("use_llm", False),
        "manual_overrides": {
            "force_channels": [],
            "disable_channels": cfg.get("disable_channels", []),
        },
    }
    t0 = time.perf_counter()
    r = requests.post(f"{API}/api/search", json=payload, timeout=300)
    wall_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    data["_wall_ms"] = wall_ms
    return data


def rank_of(groups: list[dict], gt_videos: set[str]) -> int | None:
    """Hạng (1-based) của video ground truth đầu tiên trong danh sách nhóm."""
    for i, g in enumerate(groups, start=1):
        if g.get("video_id") in gt_videos:
            return i
    return None


def frame_rank_of(groups: list[dict], gt: list[tuple[str, int]], tol: int) -> int | None:
    """Hạng của frame đúng khi duyệt phẳng theo thứ tự hiển thị cho người dùng."""
    pos = 0
    for g in groups:
        for f in g.get("frames", []):
            pos += 1
            fidx = f.get("frame_idx")
            if fidx is None:
                continue
            for gv, gf in gt:
                if g.get("video_id") == gv and abs(int(fidx) - gf) <= tol:
                    return pos
    return None


def summarize(records: list[dict], key: str) -> dict:
    ranks = [r[key] for r in records]
    n = len(ranks)
    hits = {f"H@{k}": sum(1 for r in ranks if r is not None and r <= k) for k in CUTOFFS}
    mrr = sum(1.0 / r for r in ranks if r is not None) / n if n else 0.0
    found = [r for r in ranks if r is not None]
    return {
        "n": n,
        "hits": hits,
        "hit_rate": {k: v / n for k, v in hits.items()} if n else {},
        "mrr": mrr,
        "found": len(found),
        "mean_rank_found": statistics.mean(found) if found else None,
        "median_rank_found": statistics.median(found) if found else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full", choices=list(CONFIGS))
    ap.add_argument("--tol", type=int, default=150, help="dung sai frame khi tính frame-level")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    queries, reasons = load_tkis()
    if args.limit:
        queries = queries[: args.limit]

    print(f"[{args.config}] {len(queries)} query T-KIS có GT (loại: {reasons})", flush=True)

    records, failures = [], []
    for i, q in enumerate(queries, start=1):
        try:
            data = search(q.text, cfg)
        except Exception as exc:  # ghi lại chứ không bỏ qua im lặng
            failures.append({"query_id": q.query_id, "error": str(exc)})
            print(f"  [{i}/{len(queries)}] {q.query_id} LỖI: {exc}", flush=True)
            continue

        groups = data.get("groups", [])
        vr = rank_of(groups, q.gt_videos)
        fr = frame_rank_of(groups, q.gt, args.tol)
        lat = data.get("latency_ms", {})
        rec = {
            "query_id": q.query_id,
            "phase": q.phase,
            "text": q.text,
            "gt_videos": sorted(q.gt_videos),
            "gt_frames": q.gt,
            "video_rank": vr,
            "frame_rank": fr,
            "n_groups": len(groups),
            "top_videos": [g.get("video_id") for g in groups[:10]],
            "channels_used": sorted({c for g in groups for c in g.get("channels", [])}),
            "query_type": (data.get("parsed") or {}).get("query_type"),
            "latency_ms": lat,
            "wall_ms": data.get("_wall_ms"),
            "workbook_recall_note": q.recall_note,
        }
        records.append(rec)
        print(
            f"  [{i}/{len(queries)}] {q.query_id}: video_rank={vr} frame_rank={fr} "
            f"({lat.get('total_ms', 0):.0f} ms)",
            flush=True,
        )

    out = {
        "config": args.config,
        "config_detail": cfg,
        "tolerance_frames": args.tol,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_queries": len(records),
        "failures": failures,
        "video_level": summarize(records, "video_rank"),
        "frame_level": summarize(records, "frame_rank"),
        "latency": {
            "total_ms": sorted(r["latency_ms"].get("total_ms", 0) for r in records),
            "parse_ms": sorted(r["latency_ms"].get("parse_ms", 0) for r in records),
            "fusion_ms": sorted(r["latency_ms"].get("fusion_ms", 0) for r in records),
            "wall_ms": sorted(r["wall_ms"] or 0 for r in records),
        },
        "records": records,
    }

    dest = Path(args.out) if args.out else ROOT / "benchmarks" / "runs" / f"tkis_{args.config}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    v, f = out["video_level"], out["frame_level"]
    print(f"\n=== {args.config} (n={v['n']}) ===")
    print("video-level:", {k: f"{r:.2%}" for k, r in v["hit_rate"].items()}, f"MRR={v['mrr']:.4f}")
    print("frame-level:", {k: f"{r:.2%}" for k, r in f["hit_rate"].items()}, f"MRR={f['mrr']:.4f}")
    print("→", dest)


if __name__ == "__main__":
    main()
