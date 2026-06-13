# Audio Event Extraction Pipeline — Handoff (nhánh audio)

> Đọc file này đầu mỗi session để hiểu nhánh audio mà không phải dò lại lịch sử.
> Bù cho nhánh **speech** (xem [PIPELINE.md](PIPELINE.md)): bắt **sự kiện âm thanh non-speech**
> (piano, vó ngựa, chó sủa, chuông, còi xe, nhạc...) mà ASR không bao giờ bắt được.

---

## 1. Bối cảnh & phạm vi

Hệ retrieval video tiếng Việt (thi AIC), kiến trúc **2 nhánh độc lập**:
- **speech** (ASR → text) — đã xong, xem PIPELINE.md.
- **audio** (sound event) — ✅ **EXTRACT XONG TOÀN BỘ 1478/1478 video, integrity sạch** (chi tiết §7). File này.
- **fusion temporal** + **index** + **mapping giây→keyframe (TRAKE)** — *chưa làm* (§8).

Giải đúng query hỗn hợp kiểu *"một người nói cảm ơn và bên cạnh có tiếng piano"*: speech bắt "cảm ơn", audio bắt "piano", fusion join theo thời gian.

---

## 2. Model (SOTA giữa 2026) & lý do

| Tầng | Model | Vai trò | Ghi chú T4 |
|---|---|---|---|
| **GLAP** | `mispeech/GLAP` (Dasheng-0.6B + text enc đa ngữ) | embedding audio-text 1024-d → cosine với text query, **đa ngôn ngữ** | nhẹ, chạy mọi transformers |
| **CED** | `mispeech/ced-base` (ViT ~86M) | tag AudioSet 527 nhãn ("Piano","Dog","Church bell"...) + score | nhẹ; **VỠ trên transformers 5.0** (xem §4) |
| **caption** | `mispeech/midashenglm-7b` (Qwen2.5-Omni-3B + Dasheng, int4) | sinh caption giàu nghĩa, **CHỈ** trên window non-speech | ~7B int4 ≈ 6-8GB, **vừa T4**; load ~5 phút |

**Đã loại:** Audio Flamingo 3 (chất lượng nhất nhưng **36GB VRAM** → không vừa T4). LAION-CLAP (English-only; GLAP đa ngữ + Dasheng mạnh hơn ở environment/music).

---

## 3. Pipeline & cách chạy

**File:** `audio_pipeline.py` + `audio_pipeline.ipynb` (self-contained Kaggle, mirror nhánh speech).

Luồng mỗi video (window 5s, hop 2.5s):
```
video → ffmpeg(16kHz mono) → window
   ├─ MỌI window:  GLAP embedding (→ .glap.npy)  +  CED top-5 tag
   └─ GATE (top-1 tag là non-speech, score ≥ CAPTION_GATE): MiDashengLM caption (int4)
→ <id>.audio.json + <id>.glap.npy
```

**Chạy Kaggle:** bật Internet + GPU **T4×2**, sửa `DATASET_GLOB` (đệ quy `**`), rồi:
1. Config → 2. **Ghi file pipeline** (writefile) → 3. **Setup** → **Restart kernel** (đổi transformers) → 4. Run.
- **KHÔNG cần convert model / cuDNN wrapper** (khác speech — không có CTranslate2).
- Multi-GPU: `--num_gpus 2` chia video round-robin, mỗi worker ghim 1 GPU + **load riêng 1 MiDashengLM** (vẫn vừa 16GB).
- Checkpoint: skip video đã có `.audio.json`.
- **Benchmark/validate dùng 1 GPU** (cố ý, để đo throughput đơn vị); **Run full dùng `NUM_GPUS`=2**.

---

## 4. Stack version đã pin (QUAN TRỌNG NHẤT — đừng đổi tùy tiện)

Nhánh audio dùng **torch stock của Kaggle** (KHÔNG hạ như speech) → là **môi trường riêng**, chạy notebook/session **tách biệt** với speech. Cell Setup:

```bash
pip install glap_model || pip install "git+https://github.com/xiaomi-research/dasheng-glap.git"
pip install soundfile librosa accelerate bitsandbytes
pip install "transformers==4.53.3"     # <-- PIN MẤU CHỐT
```

**`transformers==4.53.3` là khe duy nhất thỏa cả 2 ràng buộc:**

| Ràng buộc | Vì sao |
|---|---|
| transformers **< 5.0** | Kaggle mặc định **5.0** → custom code CED ra **tag rác** (Maraca/Mosquito đồng đều ~0.66, không có Speech). Pin <5.0 → CED đúng. |
| transformers **≥ 4.52** | MiDashengLM dựa **Qwen2.5-Omni** (merge vào transformers 4.52) → bản cũ hơn không load được caption. |

GLAP chạy được cả 5.0 lẫn 4.53.3 → không phải ràng buộc. **Nếu Kaggle đổi base image, có thể phải chỉnh lại pin này.**

Phụ thuộc khác: torch ~2.10 (stock Kaggle), numpy ~2.4 (stock), bitsandbytes (int4), soundfile (libsndfile), librosa (fallback resample).

---

## 5. Các bug đã gặp & cách sửa (để session sau khỏi mò lại)

| Triệu chứng | Nguyên nhân | Fix (đã nằm trong code) |
|---|---|---|
| `No module named 'glap_model'` mà Setup vẫn báo xong | `pip -q` giấu lỗi cài | bỏ `-q` + fallback git + **hard-check import** ngay tại Setup |
| CED tag rác (Maraca/Mosquito ~0.66 đều) | transformers 5.0 phá CED | **pin `transformers==4.53.3`** |
| caption lỗi `...FloatTensor != HalfTensor...cudnn_batch_norm` | audio float32 vs encoder fp16 | `apply_chat_template(...).to(device=model.device, dtype=model.dtype)` |
| caption lỗi `MiDashengLMProcessor has no attribute batch_decode` | decode sai API | decode bằng **`AutoTokenizer.batch_decode`** riêng, KHÔNG phải processor |
| caption bung trên ~mọi window (sẽ vỡ thời gian) | gate cũ lấy "tag non-speech ĐẦU TIÊN" → "Speech synthesizer" (0.6) lọt | gate theo **TOP-1 only** + mở rộng `SPEECH_STOP` (speech + ambience) |
| benchmark `delta <= 0` lần 1 | runA tải model ~14GB, runB cache | thêm **warm-up** (tải/cache trước khi tính giờ) |
| benchmark `delta <= 0` lần 2 | audio quá nhanh, proc(1 video) < nhiễu load | runB xử lý **12 video** (`B_SET`) cho delta đủ lớn |
| writefile cell bị ghi đè (mất `audio_pipeline.py`) | script patch match "pip" trong chữ "pi**p**eline" | patch notebook **match theo header markdown**, không theo nội dung |

**Cấu hình caption quan trọng** (trong `audio_pipeline.py`):
- `CAPTION_GATE = 0.30` — nâng lên `0.50` để caption ít hơn / chạy nhanh hơn.
- `SPEECH_STOP` — tập nhãn top-1 KHÔNG kích hoạt caption (speech variants + Inside-room + Silence).
- `max_new_tokens=64` — vài caption bị cắt cụt; tăng nếu muốn caption đầy đủ (chậm hơn).

---

## 6. Hiệu năng (benchmark thật, T4×2)

| | caption ON | caption OFF |
|---|---|---|
| Tốc độ 1 GPU | **29.9x** realtime | 88.8x realtime |
| Load model/worker | ~5 phút (MiDashengLM int4) | nhanh |
| Wall-time 2 GPU (325h, 1478 video) | **~5h30m** | ~1h55m |
| Session 12h cần | ~0.50 (gói 1 session) | ~0.18 |

→ Caption làm **×3 thời gian**. Cả 2 đều gói gọn 1 session.

**MiDashengLM ~14GB tải lại MỖI session** (HF cache không persistent) → stage thành Kaggle Dataset + set `HF_HOME` để khỏi tải lại.

---

## 7. Output & chất lượng đã validate

**Vị trí:** `/kaggle/working/audio_out/`
- `<video_id>.audio.json`: list window
  ```json
  {"video_id","start","end",
   "tags":[{"label","score"}],   // CED top-5, multi-label sigmoid
   "glap_idx": 0,                 // hàng trong .glap.npy
   "caption": null | "..."}       // chỉ window non-speech được gate
  ```
- `<video_id>.glap.npy`: float16 `[N_window × 1024]`, hàng i ↔ `glap_idx` i.
- Cùng convention `{video_id,start,end}` với `.speech.json` → fusion join thẳng.

**Đọc L21_V001 + L21_V023 (bản tin) — kết luận:**
- ✅ **Tags bám sự kiện thật theo giây**: vó ngựa (Clip-clop/Horse 105s), rắn (Snake/Hiss 110s), chuông nhà thờ (Church bell 138s), chó sủa (Dog/Bow-wow 142-167s), trống (Drum kit/Hi-hat 45s), xe/trực thăng. Giá trị cốt lõi — ASR mù hết.
- ✅ **Caption bắn ở NHẠC HIỆU/sting** (intro chưa có narration → top-1 = Music/Television). V023 0-5s caption bắt đúng *"piano and gong, dark and ominous, suspenseful"* — chính xác use-case "tiếng piano". **Mỗi video news thường ≥1 caption hữu ích ở intro** → caption KHÔNG vô dụng trên news (tóm đoạn nhạc mà tag thô "Music" không diễn tả nổi).
- ⚠️ **"Speech" gần như luôn top-1** sau intro (narration đè) → caption chỉ bắn ~1% (đoạn nhạc). Sự kiện non-speech vẫn nằm trong **tags top-5** → retrieval qua tags+GLAP vẫn bắt được.
- 💡 **Quyết định caption:** nghiêng GIỮ `USE_CAPTION=True` (5.5h) — bắt được nhạc hiệu/sting (piano/brass/gong) đáng giá. Chỉ tắt (2h) nếu chắc corpus thuần tin-có-dẫn, không MV/phóng sự không lời.

**✅ EXTRACT HOÀN TẤT — TOÀN CORPUS 1478/1478 video, 466,996 window (số cuối):**
- **Integrity sạch**: 1478 `.audio.json` + 1478 `.glap.npy` khớp 1-1, 0 file 0-byte, 0 JSON rỗng, 0 JSON hỏng.
- **Phân bố nhóm**: K-series 605 (K01–K20, ~28-32/nhóm), L-series 873 (L21–L30; **L26=498** áp đảo, L30=96, L25=88).
- K-series đa dạng (nấu ăn `Sizzle/Frying/Chopping`, hành động `Machine gun`, nhạc `Drum/Flute/Timpani`) → caption bắn **7.54%** (gấp ~8× L-news ~1%). Top-1: Speech 431k (92%), Music 29k (6.3%).
- ⚠️ **Tag nhiễu KHỔNG LỒ**: `Speech synthesizer` **103,651 window (22.2%)**, `Mantra` **38,467 (8.2%)** — CED nhầm giọng đọc tiếng Việt thành synth/tụng kinh. **BẮT BUỘC stoplist.**
- Caption (35,222 tổng): **63.3% mô tả thật** (instrument/mood — vàng), **31.6% echo generic** (*"contains non-speech sounds, music and instruments"*), **5.1% ASR tiếng Việt SAI MODE** (phiên âm lời nói thay vì mô tả → trùng speech branch, sai ngôn ngữ cho index caption-English).
- 💡 **Non-music event → TAG đáng tin hơn CAPTION** (bếp/va chạm: tag "Frying (food)" chính xác, caption hay generic/sai). Caption mạnh nhất ở **nhạc** (brass/flute/drum/folk/piano+gong...).

**Quy tắc cho bước INDEX (§8.1) — rút từ 466k window:**
- **Stoplist tag (BẮT BUỘC)**: `Speech synthesizer`, `Mantra` (~142k window rác). Cân nhắc thêm `Cash register`/`Snake`/`Plop` khi lẻ loi + điểm thấp.
- **Tin tag top-1 (~0.70); tag phụ chỉ tin khi score ≥ ~0.55 HOẶC cụm tag liên quan** (Drum+Hi-hat, Bell+Ding, Horse+Clip-clop). Weight theo score, đừng coi top-5 ngang nhau.
- **Lọc caption**: DROP caption có **dấu tiếng Việt** (5.1% ASR sai mode) + DROP/hạ trọng số **echo generic** (31.6%) → giữ ~63% mô tả thật.
- **GLAP là đường retrieval thứ 2** — không dính nhiễu tag/caption, encode audio thật.

---

## 8. Việc tiếp theo (chưa làm)

1. **Build index**: GLAP `.npy` → FAISS/Qdrant (cosine); tags + caption → Elasticsearch (BM25/filter, weight theo score). Query VN: decompose → dịch sound-event sang English cho GLAP (chế độ mạnh nhất).
2. **Mapping giây → keyframe (cho TRAKE)**: đọc `map-keyframes` CSV của BTC (`pts_time, fps, frame_idx`) → `time_to_keyframe(video_id, t)`. **Tách rời, CPU-only, chạy sau extract** (đừng bake vào extraction — chưa chốt format PDF mục 2.1). Audio và video chung mốc t=0 nên `pts_time` khớp thẳng `start/end` giây.
3. **Fusion temporal**: join `.audio.json` ↔ `.speech.json` ↔ keyframe theo `(video_id, ±3s)` + RRF + co-occurrence (tránh false positive chỉ-piano hoặc chỉ-"cảm ơn"). Lưu ý TRAKE là **chuỗi sự kiện** → cần align thứ tự keyframe đơn điệu tăng, không chỉ map từng sự kiện.
