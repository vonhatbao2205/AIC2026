"""Tách ảnh hưởng của việc định tuyến sai khỏi độ khó tự nhiên của query.

Trong lần chạy đầu, 22/67 query T-KIS bị bộ phân tích gán nhầm là TRAKE và nhóm
query đó có H@1 thấp hơn hẳn. Có hai cách giải thích:

  (a) định tuyến sai làm giảm chất lượng truy hồi;
  (b) những query đó vốn khó hơn (mô tả cả một quá trình chứ không phải một
      khoảnh khắc), nên dù định tuyến đúng vẫn khó.

Script này chạy lại đúng nhóm query đó nhưng ép `query_type_hint="T-KIS"`. Nếu
kết quả cải thiện thì (a) đúng; nếu gần như không đổi thì (b) đúng.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_loader import ROOT, load_tkis  # noqa: E402

API = "http://127.0.0.1:8000"


def search(text: str, hint: str) -> dict:
    t0 = time.perf_counter()
    r = requests.post(
        f"{API}/api/search",
        json={"query": text, "query_type_hint": hint, "top_k": 100, "max_videos": 100},
        timeout=300,
    )
    wall = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    d = r.json()
    d["_wall_ms"] = wall
    return d


def rank_of(groups: list[dict], gt_videos: set[str]) -> int | None:
    for i, g in enumerate(groups, start=1):
        if g.get("video_id") in gt_videos:
            return i
    return None


def stats(ranks: list[int | None]) -> dict:
    n = len(ranks)
    if not n:
        return {}
    return {
        "n": n,
        "H@1": sum(1 for r in ranks if r == 1) / n,
        "H@5": sum(1 for r in ranks if r and r <= 5) / n,
        "H@10": sum(1 for r in ranks if r and r <= 10) / n,
        "H@100": sum(1 for r in ranks if r and r <= 100) / n,
        "MRR": sum(1 / r for r in ranks if r) / n,
    }


def main() -> None:
    baseline = json.load(open(ROOT / "benchmarks" / "runs" / "tkis_full.json", encoding="utf-8"))
    misrouted = {r["query_id"] for r in baseline["records"] if r["query_type"] != "T-KIS"}
    base_rank = {r["query_id"]: r["video_rank"] for r in baseline["records"]}
    base_type = {r["query_id"]: r["query_type"] for r in baseline["records"]}

    queries = [q for q in load_tkis()[0] if q.query_id in misrouted]
    print(f"Chạy lại {len(queries)} query bị định tuyến sai, ép hint = T-KIS\n")

    records = []
    for i, q in enumerate(queries, start=1):
        try:
            d = search(q.text, "T-KIS")
        except Exception as exc:
            print(f"  [{i}] {q.query_id} LỖI: {exc}")
            continue
        vr = rank_of(d.get("groups", []), q.gt_videos)
        old = base_rank[q.query_id]
        records.append(
            {
                "query_id": q.query_id,
                "auto_type": base_type[q.query_id],
                "rank_auto": old,
                "rank_forced_tkis": vr,
                "forced_type": (d.get("parsed") or {}).get("query_type"),
                "wall_ms": d.get("_wall_ms"),
            }
        )
        arrow = "=" if old == vr else ("↑" if (vr and (not old or vr < old)) else "↓")
        print(f"  [{i}/{len(queries)}] {q.query_id}: {old} → {vr} {arrow}")

    before = stats([r["rank_auto"] for r in records])
    after = stats([r["rank_forced_tkis"] for r in records])

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n": len(records),
        "auto_routing": before,
        "forced_tkis": after,
        "records": records,
    }
    dest = ROOT / "benchmarks" / "runs" / "routing_ablation.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Nhóm query bị định tuyến sai (n={len(records)}) ===")
    for name, s in (("định tuyến tự động", before), ("ép hint T-KIS", after)):
        print(
            f"  {name:22s} H@1={s['H@1']:6.1%} H@5={s['H@5']:6.1%} "
            f"H@10={s['H@10']:6.1%} H@100={s['H@100']:6.1%} MRR={s['MRR']:.3f}"
        )
    improved = sum(
        1
        for r in records
        if r["rank_forced_tkis"] and (not r["rank_auto"] or r["rank_forced_tkis"] < r["rank_auto"])
    )
    worse = sum(
        1
        for r in records
        if r["rank_auto"] and (not r["rank_forced_tkis"] or r["rank_forced_tkis"] > r["rank_auto"])
    )
    print(f"  tốt lên: {improved}, kém đi: {worse}, còn lại không đổi")
    print("→", dest)


if __name__ == "__main__":
    main()
