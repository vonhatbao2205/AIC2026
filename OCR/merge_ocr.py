#!/usr/bin/env python3
"""Gộp tất cả ocr_extracted/*.jsonl -> 1 file ocr.jsonl duy nhất.
- Dedup theo `id` (giữ bản gặp đầu tiên; dữ liệu hiện tại không trùng).
- KHÔNG lọc/sửa gì — giữ nguyên record gốc (raw, boxes, text_nfc, text_fold...).
"""
import glob, json, os, collections

SRC_DIR = os.path.join(os.path.dirname(__file__), "ocr_extracted")
OUT     = os.path.join(os.path.dirname(__file__), "ocr.jsonl")   # nằm NGOÀI ocr_extracted để khỏi tự đọc lại

files = sorted(glob.glob(os.path.join(SRC_DIR, "*.jsonl")))
seen = set()
n_write = n_dup = n_bad = 0
cat = collections.Counter()

with open(OUT, "w", encoding="utf-8") as fout:
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rid = json.loads(s)["id"]
                except Exception:
                    n_bad += 1
                    continue
                if rid in seen:
                    n_dup += 1
                    continue
                seen.add(rid)
                fout.write(s + "\n")          # ghi NGUYÊN dòng gốc (không re-serialize)
                n_write += 1
                cat[rid.split("/")[0]] += 1

print(f"Đọc {len(files)} file từ {SRC_DIR}")
print(f"==> Ghi {n_write:,} record DUY NHẤT -> {OUT}")
print(f"    Bỏ qua: {n_dup:,} trùng id | {n_bad} dòng hỏng")
print(f"    {len(cat)} loại | dung lượng: {os.path.getsize(OUT)/1e6:.1f} MB")
print("\nPhân bố theo loại:")
for c in sorted(cat):
    print(f"  {c:8s} {cat[c]:>7,}")
