#!/usr/bin/env python3
"""Post-process Phần B (lọc nhiễu) + C (giữ nguyên gốc).
Đọc ocr.jsonl (KHÔNG sửa), ghi ocr_clean.jsonl với mỗi record:
  - GIỮ NGUYÊN mọi trường cũ (raw, boxes, text_nfc, text_fold, ...)  [Phần C]
  - THÊM text_clean / text_clean_fold  = đã bỏ logo đài / đồng hồ / watermark / nút YouTube / ký tự lẻ / toạ độ sót  [Phần B]
  - THÊM clock / hour = đồng hồ phát sóng tách ra field riêng (không vứt, để filter sáng/chiều nếu cần)
"""
import json, re, unicodedata, os, collections

IN  = os.path.join(os.path.dirname(__file__), "ocr.jsonl")
OUT = os.path.join(os.path.dirname(__file__), "ocr_clean.jsonl")

def nfc(s):  return unicodedata.normalize("NFC", s) if s else s
def fold(s):
    if not s: return s
    d = unicodedata.normalize("NFD", s)
    d = "".join(c for c in d if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", d.replace("đ", "d").replace("Đ", "D")).lower()

COORD = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)")          # mảnh toạ độ (x,y) còn sót
TIME  = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")          # box toàn timestamp
CLOCK = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")             # đồng hồ HH:MM:SS đầy đủ
LOGO  = {"hd", "hctv", "subscribe", "subscribed", "giay"}  # dạng đã fold

def is_logo_word(w):
    return w in LOGO or w.startswith("htv")              # htv, htv7, htv9, htv50...

def clean_box_text(t):
    t = COORD.sub(" ", t or "")                          # bỏ toạ độ sót (0.06% frame vỡ format)
    return re.sub(r"\s+", " ", t).strip()

def is_noise(t):
    """Box TOÀN BỘ là logo/đồng hồ/ký tự lẻ -> rác. Câu có chữ thật -> giữ."""
    s = (t or "").strip()
    if not s: return True
    if len(s) == 1: return True
    if TIME.match(s): return True
    w = fold(s).split()
    return bool(w) and all(is_logo_word(x) or len(x) == 1 or TIME.match(x) for x in w)

def extract_clock(boxes):
    cands = [b for b in boxes if CLOCK.match((b.get("text") or "").strip())]
    tr = [b for b in cands if b.get("box") and b["box"][0] > 700 and b["box"][1] < 230]  # góc trên-phải
    pick = tr or cands
    if pick:
        c = pick[0]["text"].strip()
        m = re.match(r"^(\d{1,2}):", c)
        return c, (int(m.group(1)) if m else None)
    return "", None

n = 0; cb = ca = 0; n_clock = 0; n_empty = 0
ex = []
with open(IN, encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        n += 1
        boxes = r.get("boxes") or []
        kept = [clean_box_text(b.get("text", "")) for b in boxes]
        kept = [t for t in kept if t and not is_noise(t)]
        clean = " ".join(kept)
        # Phần C: giữ nguyên hết, chỉ THÊM trường mới
        r["text_clean"] = nfc(clean)
        r["text_clean_fold"] = fold(clean)
        clk, hr = extract_clock(boxes)
        r["clock"] = clk
        r["hour"] = hr
        cb += len(r.get("text_nfc", "")); ca += len(clean)
        if clk: n_clock += 1
        if not clean: n_empty += 1
        if r.get("text_nfc") and r["text_nfc"] != clean and len(ex) < 5:
            ex.append((r["id"], r["text_nfc"], clean, clk))
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Đọc {IN}\n→ ghi {OUT}")
print(f"{n:,} record | giảm nhiễu {100*(1-ca/max(1,cb)):.1f}%  (text {cb:,} → {ca:,} ký tự)")
print(f"tách được clock: {n_clock:,} | text_clean rỗng (frame chỉ logo/giờ, không có tin): {n_empty:,}")
print(f"ocr.jsonl GIỮ NGUYÊN {os.path.getsize(IN)/1e6:.1f}MB | ocr_clean.jsonl {os.path.getsize(OUT)/1e6:.1f}MB")
print("\n--- ví dụ before → after ---")
for i, (idv, before, after, clk) in enumerate(ex, 1):
    print(f"{i}. {idv}  [clock={clk}]")
    print(f"   trước: {before[:140]}")
    print(f"   sau  : {after[:140]}")
