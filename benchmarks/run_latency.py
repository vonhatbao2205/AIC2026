"""Đo độ trễ backend theo từng cấu hình kênh, lặp lại giống benchmark cũ.

Giao thức: một lượt khởi động (không tính), sau đó N request cho mỗi cấu hình,
lấy top-100 mỗi kênh. Đo cả thời gian backend tự báo (`latency_ms.total_ms`) và
thời gian tường (wall) đo từ phía client để thấy phần overhead mạng.
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
from gt_loader import ROOT  # noqa: E402

API = "http://127.0.0.1:8000"

# Một truy vấn có đủ tín hiệu để mọi kênh đều có việc làm.
QUERY = (
    "Phóng viên đứng trước một tòa nhà lớn, có dòng chữ trên màn hình và "
    "tiếng nhạc nền, sau đó máy quay lia sang đám đông"
)

CONFIGS = {
    "pe_only": ["ocr", "speech", "audio"],
    "pe_ocr": ["speech", "audio"],
    "pe_speech": ["ocr", "audio"],
    "pe_audio": ["ocr", "speech"],
    "full_hybrid": [],
}


def one(disable: list[str]) -> tuple[float, dict]:
    t0 = time.perf_counter()
    r = requests.post(
        f"{API}/api/search",
        json={
            "query": QUERY,
            "top_k": 100,
            "max_videos": 100,
            "manual_overrides": {"force_channels": [], "disable_channels": disable},
        },
        timeout=300,
    )
    wall = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return wall, r.json().get("latency_ms", {})


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(int(round(p * (len(s) - 1))), len(s) - 1)
    return s[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=25)
    ap.add_argument("--reverse", action="store_true", help="đảo thứ tự để kiểm tra ảnh hưởng warm-up")
    ap.add_argument("--interleave", action="store_true", help="chạy xen kẽ các cấu hình để loại bỏ ảnh hưởng thứ tự")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    configs = dict(reversed(list(CONFIGS.items()))) if args.reverse else CONFIGS

    samples: dict[str, dict[str, list[float]]] = {
        name: {"wall": [], "total": [], "fusion": []} for name in configs
    }

    if args.interleave:
        # Chạy xen kẽ: mỗi vòng gọi lần lượt tất cả cấu hình. Cách này loại bỏ
        # lợi thế của cấu hình chạy sau (cache đã nóng) mà kiểu chạy theo khối
        # không tránh được.
        for name, disable in configs.items():
            one(disable)  # warm-up
        for _ in range(args.n):
            for name, disable in configs.items():
                w, lat = one(disable)
                samples[name]["wall"].append(w)
                samples[name]["total"].append(lat.get("total_ms", 0.0))
                samples[name]["fusion"].append(lat.get("fusion_ms", 0.0))
    else:
        for name, disable in configs.items():
            one(disable)  # warm-up, không tính vào thống kê
            for _ in range(args.n):
                w, lat = one(disable)
                samples[name]["wall"].append(w)
                samples[name]["total"].append(lat.get("total_ms", 0.0))
                samples[name]["fusion"].append(lat.get("fusion_ms", 0.0))

    results = {}
    for name, disable in configs.items():
        walls = samples[name]["wall"]
        totals = samples[name]["total"]
        fusions = samples[name]["fusion"]
        results[name] = {
            "n": args.n,
            "disabled": disable,
            "backend_total_ms": {
                "median": statistics.median(totals),
                "mean": statistics.mean(totals),
                "p95": pct(totals, 0.95),
                "min": min(totals),
                "max": max(totals),
            },
            "wall_ms": {
                "median": statistics.median(walls),
                "p95": pct(walls, 0.95),
                "min": min(walls),
                "max": max(walls),
            },
            "fusion_ms": {"median": statistics.median(fusions)},
        }
        b = results[name]["backend_total_ms"]
        print(
            f"{name:12s} median={b['median']/1000:.2f}s p95={b['p95']/1000:.2f}s "
            f"min-max={b['min']/1000:.2f}-{b['max']/1000:.2f}s",
            flush=True,
        )

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_per_config": args.n,
        "query": QUERY,
        "order": list(configs),
        "interleaved": args.interleave,
        "note": (
            "Bỏ một request warm-up cho mỗi cấu hình. Đã gồm enrichment/RRF/grouping, "
            "không gồm LLM parse. Cấu hình chạy đầu tiên vẫn chịu chi phí khởi động "
            "kết nối/cache nên cần đọc kèm thứ tự chạy."
        ),
        "results": results,
    }
    if args.out:
        name = args.out
    elif args.interleave:
        name = "latency_interleaved.json"
    else:
        name = "latency_reversed.json" if args.reverse else "latency_forward.json"
    dest = ROOT / "benchmarks" / "runs" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("→", dest)


if __name__ == "__main__":
    main()
