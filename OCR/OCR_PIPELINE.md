# OCR Pipeline — AIC 2026 (Keyframe OCR tiếng Việt)

Tài liệu tổng hợp: kiến trúc pipeline, model, post-processing, file sản phẩm, và việc cần làm tiếp (Elasticsearch...).
Cập nhật: 2026-06.

---

## 1. Mục tiêu & bối cảnh

- **Bài toán:** OCR chữ tiếng Việt (+ một ít tiếng Anh/CJK) trên **keyframe video tin tức** cho HCMC AI Challenge 2026 (Known-Item Search).
- **OCR là 1 kênh truy hồi** trong hệ thống (bên cạnh visual embedding PE/SigLIP, ASR, object...). Cứu các query có **chữ trên màn hình**: tiêu đề bản tin, ticker, tên người/địa danh, biển hiệu, số liệu.
- **Loại chữ chính:** lower-third / ticker tin tức (nền sạch, giá trị cao) + scene text + logo/watermark (nhiễu).

---

## 2. Quyết định model: **HunyuanOCR** (không dùng PaddleOCR-VL)

| | HunyuanOCR (`tencent/HunyuanOCR`, 1B) | PaddleOCR-VL (0.9B) |
|---|---|---|
| Text spotting (scene/video) | **70.92** | 53.38 |
| Tối ưu cho | **scene text, video frame, subtitle** | document parsing (PDF/bảng) |
| Tiếng Việt | hỗ trợ tường minh | gián tiếp (Latin) |

→ Dataset là **TV broadcast** (overlay/scene text, không phải tài liệu) → HunyuanOCR hợp hơn. Chất lượng thực tế: **ticker + caption tên đọc gần như hoàn hảo** kể cả dấu chồng (ế/ữ/ợ). Prompt dùng chế độ **spotting** (text + toạ độ).

---

## 3. Kiến trúc pipeline

```
Keyframes BTC                          (Keyframes_<CAT>/keyframes/<CAT>_V###/###.jpg)
      │
      ▼
HunyuanOCR (spotting prompt)           "text(x1,y1),(x2,y2)text(...)..."
      │
      ▼
parse_spotting()                       tách text ⟷ toạ độ  → boxes[{text, box}]
      │
      ├─ normalize_nfc()  → text_nfc        (chuẩn hoá Unicode NFC)
      └─ fold_diacritics()→ text_fold       (bỏ dấu + lowercase)
      │
      ▼
JSONL per-category (resume theo `id`)  ocr_extracted/<CAT>.jsonl (+ shard .gpu0/.gpu1)
      │
      ▼  merge_ocr.py (dedup theo id)
ocr.jsonl                              382,299 record DUY NHẤT, 34 loại
      │
      ▼  postprocess_ocr.py  (Phần B + C)
ocr_clean.jsonl                        + text_clean, text_clean_fold, clock, hour
```

### Hạ tầng tính toán (2 nguồn)
- **Kaggle T4×2** — engine `transformers`, chạy 2-GPU data-parallel. (vLLM KHÔNG chạy trên T4/Turing.) → ra các file shard `<CAT>.gpu0.jsonl` / `.gpu1.jsonl`.
- **Colab A100** — engine **vLLM batched** (nhanh hơn ~1 bậc, ~0.19s/ảnh ở batch 128–256). → ra các file `<CAT>.jsonl`.
- Notebook: `hunyuanocr_colab_a100.ipynb` (A100, dùng chính), `hunyuanocr_kaggle_pipeline.ipynb` (T4×2).

### Cơ chế resume
- Skip theo field **`id` = `<loại>/<video>/<frame>`** (không theo tên file). Đọc gộp mọi `<CAT>*.jsonl` → bỏ qua ảnh đã có id → chạy tiếp phần thiếu. Chịu được ngắt giữa chừng (flush mỗi batch) + file ghi dở.

### Gotchas đã gặp (để khỏi vấp lại)
- **vLLM trên T4 (Turing sm75): bất khả thi** (FlashInfer JIT/runtime lỗi). T4 → dùng `transformers`.
- **transformers HunyuanOCR phải load `bfloat16`** (generate() ép `pixel_values→bf16`; fp16 → lỗi dtype ViT).
- **transformers: `device_map={"":0}`** (1 GPU). `"auto"` trên T4×2 → lỗi cross-device.
- **vLLM trên Colab A100:** lỗi `libcudart.so.13` (lệch CUDA) → fix bằng **clean reinstall**: `pip uninstall -y vllm torch torchvision torchaudio` → `pip install vllm` → **Restart runtime**.
- **vLLM tự bound khâu encode ảnh (encoder-cache budget)** → batch lớn (128–256) **không OOM** trên A100; cứ đẩy batch cao cho nhanh. (`BATCH_SIZE` = độ sâu hàng đợi + cỡ flush, KHÔNG phải số ảnh chạy đồng thời.)
- **Resume bug đã sửa:** glob đệ quy `/content/**` quét cả cây keyframes → treo 20+ phút. Đổi sang glob nông.

---

## 4. Định dạng dữ liệu (schema)

### `ocr.jsonl` (mỗi dòng = 1 keyframe)
| field | nghĩa |
|---|---|
| `id` | `<loại>/<video>/<frame>` — **định danh duy nhất, khớp keyframe BTC** (dùng khi submit) |
| `category`, `video` | vd `K01`, `K01_V001` |
| `image_path` | đường dẫn ảnh |
| `status` | `ok` / `error` (hiện 100% ok) |
| `raw` | output thô HunyuanOCR (có cả toạ độ) — bản đầy đủ để đối chiếu |
| `boxes` | `[{text, box:[x1,y1,x2,y2]}]` — text tách khỏi toạ độ |
| `text_nfc` | text sạch toạ độ, chuẩn **NFC** (có cả logo/giờ) |
| `text_fold` | `text_nfc` **bỏ dấu + lowercase** |
| `ts` | timestamp xử lý |

### `ocr_clean.jsonl` = `ocr.jsonl` + thêm (không xoá gì)
| field | nghĩa |
|---|---|
| `text_clean` | đã bỏ logo/đồng hồ/watermark/nút YouTube/ký tự lẻ/toạ độ sót — **dùng để search** |
| `text_clean_fold` | `text_clean` bỏ dấu + lowercase — **search không dấu/fuzzy** |
| `clock` | đồng hồ phát sóng HH:MM:SS tách riêng (vd `18:30:51`) |
| `hour` | giờ (int, vd `18`) — để filter sáng/chiều |

---

## 5. Post-processing

### Phần A — KỸ THUẬT (đã làm sẵn trong lúc OCR, có trong `ocr.jsonl`)
1. **Chuẩn hoá NFC** (`text_nfc`) — bắt buộc, nếu không search miss âm thầm (tiếng Việt 2 cách gõ dấu). ✅ 100%.
2. **Bỏ dấu + lowercase** (`text_fold`) — cho search không dấu/fuzzy. ✅ 100%.
3. **Tách toạ độ khỏi text** (`boxes`) — toạ độ là số, để trong text là nhiễu BM25. ✅ 99.94% (0.06% frame degenerate còn sót, được dọn nốt ở Phần B).

### Phần B — LỌC NHIỄU (đã làm → `text_clean`, script `postprocess_ocr.py`)
Bỏ các box **TOÀN BỘ là rác** (câu có chữ thật thì giữ nguyên):
- **Đồng hồ phát sóng** `06:32:08` (nhảy từng giây, có ở mọi frame → IDF≈0)
- **Logo đài** `HTV/HTV7/HTV9/HD/HCTV`
- **Watermark** `giây` / `160 giây`
- **Nút YouTube** `SUBSCRIBE/SUBSCRIBED`
- **Ký tự lẻ** (1 ký tự, garble) + **mảnh toạ độ sót**

> ⚠️ **Chỉ bỏ box toàn rác.** Thời gian/chữ NẰM TRONG câu tin được giữ (vd "...đến 7 giờ ngày 4/10", "Tháng 10/2024"). Chữ nước ngoài (自民党, ...) cũng giữ.

### Phần C — AN TOÀN
- **Không sửa file gốc.** `ocr.jsonl` nguyên vẹn; `postprocess_ocr.py` đọc rồi ghi file mới `ocr_clean.jsonl`.
- **Chỉ THÊM** trường, không xoá. Đồng hồ không vứt mà tách ra `clock`/`hour`.

### Quyết định lọc còn TREO (team chốt, rồi chỉnh `postprocess_ocr.py`)
- Đồng hồ: bỏ khỏi search? (hiện: bỏ + tách ra `clock`).
- Logo nhãn hàng (`BURBERRY/GUCCI/MILO`): hiện **giữ** (khó tự động).
- Frame bảng/slide dày (giá trị thấp): index thường / hạ trọng số / bỏ?
- Lọc theo **vị trí box** (góc = logo) thay vì từ khoá? (chính xác hơn, cần kích thước ảnh).
- Gộp text lặp (degeneration ~0.2%)?

---

## 6. Thống kê hiện trạng

- **382,299 keyframe đã OCR** | 34 loại (K01–K20, L21–L30 với L26 chia a–e) | **0 lỗi, 0 trùng id**.
- Post-process: **giảm ~10.7% nhiễu** | tách clock cho **197,629** frame | **24,971** frame `text_clean` rỗng (cảnh chỉ logo/giờ, không có chữ tin — bình thường, vẫn giữ để tra theo id).
- Loại lớn nhất: **L25 = 37,445** (ôn thi, nhiều slide), L26_a–e ~14k–17k mỗi cái.
- ⚠️ **L23 chỉ 2,326** (nhỏ bất thường) — **cần verify đã chạy hết chưa.**
- ~0.1–0.2% frame **slide/bảng siêu dày** bị lặp+cụt token (degeneration) — giá trị thấp, chấp nhận được.

---

## 7. File sản phẩm

| File | Nội dung |
|---|---|
| `ocr_extracted/*.jsonl` | output thô per-category (shard Kaggle `.gpuX` + Colab `.jsonl`) |
| `ocr.jsonl` | **gộp + dedup** → 382,299 record (nguồn gốc, không lọc) |
| `ocr_clean.jsonl` | **đã post-process** (thêm text_clean/clock/hour) → để đẩy Elasticsearch |
| `merge_ocr.py` | gộp ocr_extracted/ → ocr.jsonl (rerun được) |
| `postprocess_ocr.py` | ocr.jsonl → ocr_clean.jsonl (rerun được) |
| `hunyuanocr_colab_a100.ipynb` | notebook OCR trên Colab A100 (vLLM) |
| `hunyuanocr_kaggle_pipeline.ipynb` | notebook OCR trên Kaggle T4×2 (transformers) |

---

## 8. VIỆC CẦN LÀM TIẾP

### 8.1. Đưa vào Elasticsearch (ưu tiên)
**Index:** dùng `ocr_clean.jsonl`.

**Trường để index/search:**
- `text_clean` → analyzer tiếng Việt + BM25 (search **có dấu**).
- `text_clean_fold` → search **không dấu** + **fuzzy** (Levenshtein ≤1–2) cho lỗi OCR.
- (tuỳ chọn) `text_nfc` giữ làm bản đầy đủ.

**Trường keyword/filter (không phân tích):**
- `id`, `category`, `video`, `image_path` (→ **identifier submit**).
- `hour` (integer, filter sáng/chiều), `clock`.
- `boxes` (lưu để sau lọc logo theo vị trí nếu cần).

**Gợi ý mapping:**
- `text_clean`: `type text`, analyzer tiếng Việt (vd plugin `vi_analyzer` hoặc ICU), kèm sub-field `.fold` (không dấu) và `.keyword`.
- Bật **fuzziness** ở query (AUTO) để chịu lỗi OCR.
- Bulk index bằng `elasticsearch.helpers.bulk` (382k doc).

### 8.2. Tích hợp vào hệ thống retrieval
- **OCR-aware activation:** chỉ bật kênh OCR khi query "OCR-likely" (số, tên riêng, brand, dấu trích dẫn) — LLM phân loại query. (Tránh lỗi "metadata tạ" của năm trước.)
- **Fusion:** gộp rank OCR với PE/SigLIP semantic bằng **RRF (k=60)**, không hardcode trọng số.

### 8.3. Dọn dẹp / kiểm tra
- **Verify L23** (2,326) đã đủ chưa; chạy lại nếu thiếu.
- Team chốt **chính sách lọc Phần B** (mục 5) → chỉnh `postprocess_ocr.py` → rerun.
- (Tuỳ chọn) bước **gộp token lặp** + strip toạ độ mạnh hơn cho ~0.2% frame degenerate.

### 8.4. Nice-to-have
- Lọc **logo theo vị trí box** (góc trên-phải) thay vì từ khoá → chính xác hơn, vẫn bắt được "HTV" khi nó nằm trong câu tin.
- Tận dụng `boxes` cho reading-order / lọc logo đài / verify entity.
