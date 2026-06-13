# Speech Extraction Pipeline — Handoff (SOTA tiếng Việt)

> Đọc file này đầu mỗi session để hiểu pipeline hiện tại mà không cần dò lại lịch sử.
> Nhánh **speech** của hệ thống video retrieval tiếng Việt (bối cảnh thi AIC).

---

## 1. Bối cảnh & phạm vi

Bài toán: retrieval video theo query âm thanh, ví dụ *"một người nói cảm ơn và bên cạnh có tiếng piano"*. Kiến trúc **2 nhánh độc lập**:

- **Nhánh speech** (ASR → text) ← **đang làm, đã chạy được end-to-end**
- **Nhánh audio** (CLAP + audio tagging cho sound event) ← *chưa làm*
- **Fusion temporal** (RRF + co-occurrence ±vài giây) ← *chưa làm* — chỗ 2 nhánh "gặp nhau"

Thiết kế gốc & lý do nằm trong `chat-export-2026-06-02T05-02-43.md`.

### Quyết định model đã chốt
- **PhoWhisper-large** (CT2) + **WhisperX forced alignment** (wav2vec2-vi) cho **toàn bộ corpus**.
- **KHÔNG dùng Chunkformer**: corpus không phải long-form (video 0.5–45 phút, median 16, chỉ 1.2% ≥30m, max 45m). WhisperX + VAD chunking thừa sức; Chunkformer chỉ thắng khi audio dài hàng giờ.
- **KHÔNG dùng LLM post-correct**: tiếng Anh trong video cực ít → ROI ~0, bỏ để khỏi tốn chục giờ. Đã gỡ khỏi code.

---

## 2. Cách chạy trên Kaggle (notebook self-contained)

File chính: **`speech_pipeline.ipynb`** — chạy 1 mạch từ trên xuống, tự ghi `speech_pipeline.py` + `kaggle_run.py` ra đĩa (không cần upload file rời).

**Chuẩn bị:** bật **Internet** (Settings), chọn accelerator **GPU T4 × 2**.

**Thứ tự cell:**
1. **Config** — sửa `DATASET_GLOB` (mặc định `/kaggle/input/**/*.mp4`, **đệ quy** vì dataset lồng nhiều tầng `.../Videos_K20/video/*.mp4`).
2. **Ghi file** — `%%writefile` ra 2 file `.py`.
3. **Setup** — pip cài stack đã pin (xem §4) + convert PhoWhisper-large → CT2. **Sau cell này phải Restart kernel** (vì torch bị cài lại).
4. *(tùy chọn)* **Benchmark** — ước lượng tổng thời gian (phương pháp 2-lần-chạy, tách thời gian load model).
5. **Run** — chạy đa GPU full corpus.
6. **Kiểm tra** — xem phân bố `avg_word_score` + segment mẫu.

> ⚠️ Mỗi session mới phải chạy lại Setup (pip ~10 phút). Restart kernel sau Setup. Convert sẽ skip nếu thư mục `phowhisper-large-ct2` còn (xem §6 stage model).

---

## 3. Kiến trúc code

**`speech_pipeline.py`** — luồng mỗi video:
```
video -> ffmpeg(16kHz mono) -> PhoWhisper-large CT2 (ASR) -> WhisperX forced-align (word ts)
      -> lọc hallucination -> chuẩn hóa -> JSON {video_id,start,end,text,words[]}
```
- `SpeechExtractor`: load model 1 lần/process. 2 backend qua `--backend`:
  - `whisperx` (mặc định): faster-whisper chạy ASR + align wav2vec2-vi → word-timestamp chính xác.
  - `faster-whisper`: timestamp cross-attention, nhẹ, không đụng torch/pyannote (backup nếu whisperx vỡ).
- **Multi-GPU**: `--num_gpus 2` chia video round-robin, mỗi worker `CUDA_VISIBLE_DEVICES` riêng, dùng `spawn`.
- **Checkpoint**: skip video đã có `.speech.json` → hết session cứ chạy lại, nối tiếp.
- **`--min_word_score`** (mặc định **0.0 = TẮT**): lọc segment theo điểm align. Để 0.0 lúc extract (xem §7).

**`kaggle_run.py`** — wrapper bắt buộc: nạp `LD_LIBRARY_PATH` trỏ **tất cả** `nvidia/*/lib` (cudnn + cublas...) **trước khi** launch python, vá lỗi `libcudnn_*.so not found` của CTranslate2. **Đừng chạy thẳng `python speech_pipeline.py`** — phải qua wrapper này.

`kaggle_setup.sh` — **legacy, không dùng** (setup đã nằm trong notebook).

---

## 4. Stack version đã pin (QUAN TRỌNG — đừng đổi tùy tiện)

Đây là tổ hợp **hard-won** sau nhiều vòng debug trên Kaggle. Cell Setup cài chính xác:

```bash
pip install "numpy<2" torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2
pip install "whisperx==3.3.1" "pyannote.audio==3.3.2" "ctranslate2<4.5" \
    transformers==4.48.3 nvidia-cudnn-cu12 elasticsearch
pip install "numpy<2"   # ép lại phòng dep kéo numpy lên 2
```

**Lý do từng pin** (gặp lại lỗi nào thì tra đây):

| Pin | Vì sao |
|---|---|
| `torchaudio==2.2.2` | torchaudio mới của Kaggle **gỡ** `AudioMetaData` + `list_audio_backends` mà pyannote cần → `AttributeError`. 2.2.2 còn các API đó. |
| `torch==2.2.2` | Phải khớp torchaudio 2.2.2. |
| `torchvision==0.17.2` | Khớp triple torch 2.2.2; bản mới gọi `torch.library.register_fake` (chỉ có ở torch≥2.4) → vỡ khi `transformers` đụng torchvision. |
| `numpy<2` | torch 2.2 build cho numpy 1.x; numpy 2 gây `Failed to initialize NumPy: _ARRAY_API not found`. |
| `transformers==4.48.3` | transformers mới **chặn `torch.load`** nếu torch<2.6 (CVE-2025-32434). Aligner wav2vec2-vi chỉ có file `.bin` (không safetensors) → không load được. 4.48.3 chưa có cái chặn đó. |
| `ctranslate2<4.5` | ct2 4.x dùng **cuDNN 9** (`libcudnn.so.9`), khác soname cuDNN 8 (`.so.8`) mà torch bundle → **chạy chung 1 process không xung đột**. |
| `vad_method` fallback | Bản whisperx này **không có** param `vad_method`. Code `load_model` đã bọc try/except: thiếu thì bỏ qua → dùng VAD bundled công khai của whisperx (pyannote segmentation, **không cần HF token**, tự auto-upgrade lightning checkpoint). |

**Mâu thuẫn lõi cần nhớ:** pyannote/torchaudio cần torch **cũ** (≤2.4), còn transformers/ct2 mới thích torch **mới**. Cách thoát = hạ torch + hạ transformers, ct2 cuDNN khác soname nên sống chung được. Nếu Kaggle đổi base image, có thể phải chỉnh lại các pin này.

---

## 5. Hiệu năng & ngân sách thời gian

Benchmark thật (beam=5, batch=8, float16, T4):
- **11.2x realtime / 1 GPU**, load model **29s**.
- Corpus: **1478 video, 324.8 giờ audio**.
- Ước lượng **~14h30m wall trên 2 GPU** → **~1.3 session** (12h/session, tính 11h hữu dụng).
→ Sẽ cần **2 session**, checkpoint nối tiếp. Nằm trong quota 30h/tuần.

**Tăng tốc nếu muốn gói 1 session:** `COMPUTE_TYPE='int8_float16'` (T4 chạy int8 tốt, WER chênh không đáng kể) — tác động đúng chỗ nghẽn (compute). Tăng `batch_size` chỉ giúp khiêm tốn (~5-15%) vì T4 đã compute-bound. Đo lại bằng cell benchmark trước khi quyết.

---

## 6. Output

- Vị trí: **`/kaggle/working/speech_out/<video_id>.speech.json`** (tuyệt đối, persistent, lưu khi Save Version).
- `<video_id>` = **tên file mp4 bỏ đuôi** (vd `K20_V001.mp4` → `K20_V001.speech.json`). Dir **phẳng**, key theo stem (tên file đang unique nên không trùng).
- Format mỗi file = list segment:
  ```json
  {"video_id","start","end","text",
   "words":[{"start","end","word","score"}],
   "avg_word_score": 0.56,          // whisperx
   "avg_logprob": null, "no_speech_prob": null}  // chỉ có ở backend faster-whisper
  ```
- Format này để sẵn cho **fusion temporal** join với nhánh audio + keyframe.

**Stage model để khỏi convert lại mỗi session:** sau khi có `phowhisper-large-ct2/`, Save Version → tạo Kaggle Dataset từ nó → session sau bỏ comment dòng `!cp -r /kaggle/input/phowhisper-ct2/phowhisper-large-ct2 .` trong cell Convert.

---

## 7. Chất lượng đã validate (43 file đầu, 2097 segment)

- **`avg_word_score`**: cụm chặt **0.51–0.61, median 0.565**, min 0.217. → forced-alignment **khớp dấu tiếng Việt tốt** (không bị nuke). Mức 0.5–0.6 là lành mạnh cho wav2vec2-vi.
- **Score tương quan chất lượng**: segment lảm nhảm (audio kém/chồng tiếng) rơi xuống **0.217 = thấp nhất**; segment tốt 0.55–0.59. → dùng làm tín hiệu lọc được.
- **ASR**: PhoWhisper-large xuất sắc trên tin tức VN (danh từ riêng, số đọc chuẩn). Vài lỗi nhỏ kiểu Whisper (*nước biển→biệt*) nhưng không ảnh hưởng BM25.

**Chính sách lọc:** giữ **`MIN_WORD_SCORE=0.0`** khi extract (lưu trọn dữ liệu + score). **Lọc ở tầng index** (rẻ, chạy lại được): drop/hạ trọng số segment có `avg_word_score < ~0.40` (dưới bulk 0.51, trên rác 0.2x). **Đừng** bake threshold vào extraction — đổi ý là phải chạy lại 325h.

---

## 8. Việc tiếp theo (chưa làm)

1. **Build index**: Elasticsearch + VN analyzer (BM25), lọc theo `avg_word_score`. Code khung `ES_*` đã có trong `speech_pipeline.py`.
2. **Nhánh audio**: CLAP embedding + audio tagging (CED/BEATs) → vector + tags → Milvus/Qdrant. Query VN dịch sang EN trước khi encode CLAP.
3. **Fusion temporal**: join speech ↔ audio ↔ keyframe theo `(video_id, cửa sổ ±vài giây)`, RRF (k=60) + yêu cầu co-occurrence (tránh false positive chỉ-piano hoặc chỉ-"cảm ơn").
