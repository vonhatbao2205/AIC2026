"""Chạy lại các truy vấn TRAKE có ground truth trên hệ thống đang chạy live.

Khác với T-KIS, một kết quả TRAKE chỉ được coi là đúng khi vừa đúng video, vừa
khớp từng event theo đúng thứ tự thời gian. Workbook có query cho phép nhiều
phương án chuỗi (alternative), nên điểm của một query là điểm cao nhất trên các
phương án đó.

Cách dùng:
    python benchmarks/run_trake.py
    python benchmarks/run_trake.py --mode greedy   # đối chứng với chọn tham lam
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_loader import ROOT, load_trake  # noqa: E402

API = "http://127.0.0.1:8000"

# Nhiều mức dung sai để thấy kết quả nhạy thế nào với việc chấm chặt hay lỏng.
TOLERANCES = (25, 50, 100, 250, 500)


def search_trake(text: str, top_k: int = 400, use_llm: bool = False) -> dict:
    t0 = time.perf_counter()
    r = requests.post(
        f"{API}/api/search/trake",
        json={"query": text, "top_k": top_k, "use_llm": use_llm},
        timeout=900,
    )
    wall = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    d = r.json()
    d["_wall_ms"] = wall
    return d


def score_sequence(seq: dict, alt: list[tuple[int, str, int]], tol: int) -> dict:
    """Chấm một sequence dự đoán với một phương án GT."""
    if seq.get("video_id") != alt[0][1]:
        return {"video_ok": False, "matched": 0, "total": len(alt), "accuracy": 0.0}

    by_event = {f.get("event_index"): f for f in seq.get("frames", [])}
    matched = 0
    offsets = []
    for order, _video, gt_frame in alt:
        f = by_event.get(order)
        if f is None or f.get("frame_idx") is None:
            continue
        off = abs(int(f["frame_idx"]) - gt_frame)
        offsets.append(off)
        if off <= tol:
            matched += 1
    return {
        "video_ok": True,
        "matched": matched,
        "total": len(alt),
        "accuracy": matched / len(alt) if alt else 0.0,
        "offsets": offsets,
    }


def best_over_alternatives(seq: dict, alts: dict, tol: int) -> dict:
    best = {"video_ok": False, "matched": 0, "total": 0, "accuracy": 0.0}
    for _key, alt in alts.items():
        s = score_sequence(seq, alt, tol)
        if (s["accuracy"], s["video_ok"]) > (best["accuracy"], best["video_ok"]):
            best = s
    return best


def video_rank(sequences: list[dict], gt_videos: set[str]) -> int | None:
    for i, s in enumerate(sequences, start=1):
        if s.get("video_id") in gt_videos:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=400)
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    queries = load_trake()
    with_gt = [q for q in queries if q.alternatives]
    finals = [q for q in queries if not q.alternatives]
    print(f"TRAKE: {len(with_gt)} query có GT, {len(finals)} query chung kết chưa có GT")

    records = []
    for i, q in enumerate(with_gt, start=1):
        try:
            d = search_trake(q.text, args.top_k, args.use_llm)
        except Exception as exc:
            print(f"  [{i}] {q.query_id} LỖI: {exc}", flush=True)
            records.append({"query_id": q.query_id, "error": str(exc)})
            continue

        seqs = d.get("sequences", [])
        gt_videos = {v for alt in q.alternatives.values() for _o, v, _f in alt}
        vr = video_rank(seqs, gt_videos)

        top1 = seqs[0] if seqs else {}
        per_tol = {}
        for tol in TOLERANCES:
            s_top1 = best_over_alternatives(top1, q.alternatives, tol) if top1 else {}
            # Chuỗi tốt nhất trong 10 kết quả đầu — phản ánh việc người vận hành
            # thường xem vài kết quả đầu chứ không chỉ nhìn kết quả số 1.
            best10 = max(
                (best_over_alternatives(s, q.alternatives, tol) for s in seqs[:10]),
                key=lambda x: x["accuracy"],
                default={},
            )
            per_tol[tol] = {"top1": s_top1, "best_top10": best10}

        rec = {
            "query_id": q.query_id,
            "phase": q.phase,
            "n_events": q.n_events,
            "n_alternatives": len(q.alternatives),
            "gt_videos": sorted(gt_videos),
            "video_rank": vr,
            "n_sequences": len(seqs),
            "top1_video": top1.get("video_id"),
            "top1_coverage": top1.get("coverage"),
            "top1_confident_coverage": top1.get("confident_coverage"),
            "top1_filled": top1.get("filled_events"),
            "top1_complete": top1.get("complete"),
            "per_tolerance": {str(k): v for k, v in per_tol.items()},
            "wall_ms": d.get("_wall_ms"),
        }
        records.append(rec)
        acc100 = per_tol[100]["top1"].get("accuracy", 0.0)
        print(
            f"  [{i}/{len(with_gt)}] {q.query_id}: video_rank={vr} "
            f"top1={top1.get('video_id')} cov={top1.get('coverage')}/{q.n_events} "
            f"acc@tol100={acc100:.2f} ({d.get('_wall_ms', 0):.0f} ms)",
            flush=True,
        )

    # Chung kết: không có GT nên chỉ ghi lại kết quả để người vận hành đối chiếu.
    final_records = []
    for i, q in enumerate(finals, start=1):
        try:
            d = search_trake(q.text, args.top_k, args.use_llm)
        except Exception as exc:
            final_records.append({"query_id": q.query_id, "error": str(exc)})
            continue
        seqs = d.get("sequences", [])
        top1 = seqs[0] if seqs else {}
        final_records.append(
            {
                "query_id": q.query_id,
                "n_events": q.n_events,
                "top1_video": top1.get("video_id"),
                "top1_coverage": top1.get("coverage"),
                "top1_confident_coverage": top1.get("confident_coverage"),
                "top1_filled": top1.get("filled_events"),
                "top1_complete": top1.get("complete"),
                "top1_frames": [
                    {"event_index": f.get("event_index"), "frame_idx": f.get("frame_idx")}
                    for f in top1.get("frames", [])
                ],
                "wall_ms": d.get("_wall_ms"),
            }
        )
        print(
            f"  [final {i}] {q.query_id}: {top1.get('video_id')} "
            f"cov={top1.get('coverage')}/{q.n_events} fill={top1.get('filled_events')}",
            flush=True,
        )

    valid = [r for r in records if "error" not in r]
    summary = {}
    for tol in TOLERANCES:
        t = str(tol)
        summary[t] = {
            "top1_video_correct": sum(1 for r in valid if r["per_tolerance"][t]["top1"].get("video_ok")),
            "top1_all_events": sum(
                1 for r in valid if r["per_tolerance"][t]["top1"].get("accuracy", 0) == 1.0
            ),
            "best10_all_events": sum(
                1 for r in valid if r["per_tolerance"][t]["best_top10"].get("accuracy", 0) == 1.0
            ),
            "mean_event_accuracy_top1": (
                sum(r["per_tolerance"][t]["top1"].get("accuracy", 0) for r in valid) / len(valid)
                if valid
                else 0.0
            ),
        }

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "top_k": args.top_k,
        "use_llm": args.use_llm,
        "n_with_gt": len(valid),
        "video_rank_hits": {
            f"H@{k}": sum(1 for r in valid if r["video_rank"] and r["video_rank"] <= k)
            for k in (1, 3, 5, 10, 20, 50)
        },
        "per_tolerance_summary": summary,
        "records": records,
        "finals_no_gt": final_records,
    }

    dest = Path(args.out) if args.out else ROOT / "benchmarks" / "runs" / "trake.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== TRAKE (n={len(valid)}) ===")
    print("video rank:", out["video_rank_hits"])
    for tol in TOLERANCES:
        s = summary[str(tol)]
        print(
            f"  tol±{tol}: video đúng {s['top1_video_correct']}/{len(valid)}, "
            f"đủ event ở top1 {s['top1_all_events']}/{len(valid)}, "
            f"trong top10 {s['best10_all_events']}/{len(valid)}, "
            f"acc TB {s['mean_event_accuracy_top1']:.3f}"
        )
    print("→", dest)


if __name__ == "__main__":
    main()
