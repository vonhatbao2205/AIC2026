"""Tính đúng các con số xuất hiện trong chương Thực nghiệm của báo cáo.

Tập đánh giá chính là những truy vấn T-KIS mô tả **một khoảnh khắc đơn**. Các
truy vấn mô tả một chuỗi nhiều mốc nối tiếp được tách ra báo cáo riêng, vì về bản
chất đó là bài toán chuỗi sự kiện chứ không phải T-KIS.

Tiêu chí phân loại dựa trên hình dạng câu hỏi và không phụ thuộc thứ hạng hệ
thống trả về, nên việc chia tập không bị ảnh hưởng bởi kết quả.

    python benchmarks/report_metrics.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_loader import ROOT  # noqa: E402

RUNS = ROOT / "benchmarks" / "runs"
CUTOFFS = (1, 5, 10, 20, 50, 100)

# Nhãn cấu hình dùng trong bảng ablation của báo cáo.
CONFIG_LABELS = {
    "pe_only": "Chỉ ảnh",
    "no_speech": "+ OCR",
    "no_audio": "+ lời nói",
    "full": "+ âm thanh (đầy đủ)",
}


def single_moment_ids() -> set[str]:
    """Truy vấn mô tả một khoảnh khắc đơn.

    Bộ phân tích đánh dấu những câu liệt kê nhiều mốc nối tiếp là dạng chuỗi sự
    kiện; đây chính là nhóm cần tách ra. Việc gán nhãn này chỉ đọc nội dung câu
    hỏi, không dùng tới thứ hạng nào.
    """
    base = json.loads((RUNS / "tkis_full.json").read_text(encoding="utf-8"))
    return {r["query_id"] for r in base["records"] if r["query_type"] != "TRAKE"}


def multi_event_ids() -> set[str]:
    base = json.loads((RUNS / "tkis_full.json").read_text(encoding="utf-8"))
    return {r["query_id"] for r in base["records"] if r["query_type"] == "TRAKE"}


def stats(records: list[dict], key: str) -> dict:
    n = len(records)
    ranks = [r[key] for r in records]
    found = [r for r in ranks if r]
    return {
        "n": n,
        **{f"R@{k}": sum(1 for r in ranks if r and r <= k) / n for k in CUTOFFS},
        "MRR": sum(1 / r for r in ranks if r) / n,
        "found": len(found),
        "median": statistics.median(found) if found else None,
        "mean": statistics.mean(found) if found else None,
    }


def subset(config: str, ids: set[str]) -> list[dict]:
    d = json.loads((RUNS / f"tkis_{config}.json").read_text(encoding="utf-8"))
    return [r for r in d["records"] if r["query_id"] in ids]


def row(label: str, s: dict) -> str:
    cells = " & ".join(f"{s[f'R@{k}']*100:.2f}".replace(".", ",") for k in CUTOFFS)
    return f"{label} & {cells} & {s['MRR']:.4f}".replace(".", ",")


def main() -> None:
    single, multi = single_moment_ids(), multi_event_ids()
    print(f"Tập chính (một khoảnh khắc): {len(single)} truy vấn")
    print(f"Tách riêng (mô tả chuỗi):    {len(multi)} truy vấn\n")

    print("--- Bảng kết quả chính (cấu hình đầy đủ) ---")
    recs = subset("full", single)
    for level, key in (("Video", "video_rank"), ("Khung hình", "frame_rank")):
        s = stats(recs, key)
        print(" ", row(level, s))
        print(
            f"      tìm thấy {s['found']}/{s['n']}, trung vị {s['median']}, "
            f"trung bình {s['mean']:.2f}"
        )

    print("\n--- Bảng ablation kênh ---")
    for level, key in (("Video", "video_rank"), ("Khung hình", "frame_rank")):
        print(f"  [{level}]")
        for cfg, label in CONFIG_LABELS.items():
            print("   ", row(label, stats(subset(cfg, single), key)))

    print("\n--- Nhóm mô tả chuỗi (báo cáo riêng) ---")
    print(" ", row("Mô tả chuỗi", stats(subset("full", multi), "video_rank")))
    print(" ", row("Một khoảnh khắc", stats(subset("full", single), "video_rank")))


if __name__ == "__main__":
    main()
