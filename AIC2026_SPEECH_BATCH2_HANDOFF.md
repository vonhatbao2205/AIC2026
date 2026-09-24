# AIC 2026 — Speech Batch 2 Handoff (S01 + M01–M10)

> Snapshot ngày **2026-09-24**, sau khi trích speech batch 2 xong.
> Đây là data contract cho các task retrieval, backend và frontend cần tìm kiếm hoặc hiển thị
> lời nói trong video. Không cần chạy lại notebook để dùng dữ liệu.
> File này không chứa token hay secret.

---

## 1. Trạng thái bàn giao

```text
Bucket:           Baonenha1/aic26-media   (Hugging Face Storage Bucket, PUBLIC)
Prefix:           Speech/speech_out_batch2/
Public base URL:  https://huggingface.co/buckets/Baonenha1/aic26-media/resolve
Status:           COMPLETE — 316/316 video status=ok, 0 lỗi, 0 retry
Notebook:         extract-audio-batch2-colab.ipynb   (thư mục Output_Speech)
Phiên chạy:       2026-09-24 08:45–10:34 UTC (15:45–17:34 giờ VN), 1 phiên, NVIDIA A100-SXM4-80GB
```

| Nhóm | Nội dung | Video | Giờ audio | Giờ có speech | Segment | Từ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S01 | đua xe đạp (Cúp HTV 2026) | 12 | 46,60 | 32,30 | 5.481 | 396.581 |
| M01–M10 | tin tức | 304 | 99,55 | 88,93 | 16.301 | 1.330.632 |
| **Tổng** | | **316** | **146,14** | **121,23** | **21.782** | **1.727.213** |

Ngoài phạm vi file này:

- **N001–N100** (camera giao thông, 298 video): video **không có track audio** nên không có JSON speech.
- **Batch 1** (K01–K20, L21–L30, 1.478 video): chạy bằng `extract-audio.ipynb` trên Kaggle, kết quả
  ở `speech_out.zip` (thư mục `Output_Speech`). Batch 1 **chưa** có trên HF bucket.

---

## 2. Layout trên bucket

```text
Baonenha1/aic26-media/
└── Speech/
    ├── speech_out_batch2/
    │   ├── M01_V001.speech.json … M10_V031.speech.json      304 file
    │   ├── S01-V001.speech.json … S01-V012.speech.json       12 file
    │   ├── speech_manifest_batch2.csv                        1 dòng/video (mục 5)
    │   ├── _events_batch2.jsonl                              nhật ký từng video (resume/audit)
    │   ├── _run_batch2.log                                   log orchestrator
    │   └── _session.json                                     GPU + tham số ASR của phiên chạy
    ├── speech_out_batch2.zip                                 316 JSON + manifest
    └── _cache/phowhisper-large-ct2/                          model CT2 để chạy lại (mục 8)
```

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| 316 file `.speech.json` | 213.179.656 | — |
| `speech_manifest_batch2.csv` | 49.259 | `dbe7151a21ea56523e2a94c9f2c412dd030f81748ef349b345e60589c5454485` |
| `speech_out_batch2.zip` | 30.157.643 | `2c8255168871a94cf7353241a16358fd7f8063904189d0c5422abb2aef0df2c0` |

Zip có 317 entry, layout giống `speech_out.zip` của batch 1:
`speech_out_batch2/<video_id>.speech.json` + `speech_out_batch2/speech_manifest_batch2.csv`.

Prefix `Speech/` nằm ngoài snapshot media: không được liệt kê trong `manifest/dataset_manifest.json`
và không làm thay đổi `manifest/_SUCCESS.json` (xem `AIC2026_HF_MEDIA_HANDOFF.md`).

---

## 3. Truy cập

Bucket public, đọc không cần token:

```text
https://huggingface.co/buckets/Baonenha1/aic26-media/resolve/Speech/speech_out_batch2/M01_V001.speech.json
https://huggingface.co/buckets/Baonenha1/aic26-media/resolve/Speech/speech_out_batch2/speech_manifest_batch2.csv
https://huggingface.co/buckets/Baonenha1/aic26-media/resolve/Speech/speech_out_batch2.zip
```

Lấy trọn bộ nhanh nhất là tải zip (30 MB). Liệt kê file qua API:

```text
GET https://huggingface.co/api/buckets/Baonenha1/aic26-media/tree/Speech?recursive=true
```

Muốn **ghi** vào bucket thì cần `HF_TOKEN` có quyền write. Không đưa token vào frontend hay source code.

---

## 4. Schema JSON

Mỗi video có một file `<video_id>.speech.json`. File là **mảng segment** theo thứ tự thời gian,
**cùng schema với batch 1**:

```json
[
  {
    "video_id": "M01_V001",
    "start": 4.688,
    "end": 26.671,
    "text": "đây là chương trình sáu mươi giây của đài phát thanh và truyền hình thành phố hồ chí minh …",
    "words": [
      {"start": 4.688, "end": 4.868, "word": "đây", "score": 0.497},
      {"start": 4.888, "end": 4.968, "word": "là", "score": 0.499}
    ],
    "avg_logprob": null,
    "no_speech_prob": null,
    "avg_word_score": 0.5552
  }
]
```

| Trường | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `video_id` | str | giữ nguyên tên file video: `S01-V001` dùng `-`, `M01_V001` dùng `_` |
| `start`, `end` | float (giây) | biên segment, lấy từ VAD; mỗi segment ≤ 30 s (trung vị 21,8 s) |
| `text` | str | transcript của segment |
| `words[]` | list | từng từ, căn bằng wav2vec2 tiếng Việt |
| `words[].start/end` | float \| **null** | thời điểm của từ; `null` khi aligner không đặt được (mục 6.5) |
| `words[].score` | float | độ tin cậy căn chỉnh của từ, 0–1 |
| `avg_word_score` | float \| **null** | trung bình `score` của segment; `null` khi `words` rỗng |
| `avg_logprob`, `no_speech_prob` | null | luôn `null` với backend whisperx (giữ để tương thích schema) |

Quy ước thời gian: `start`/`end` tính bằng giây từ đầu file video. Cả 316 video đều có
`start_time = 0` ở stream video và audio, nên giá trị này **trùng với `pts_time` của video**. Dùng trực
tiếp để seek (`<video>.currentTime = start`) hoặc ghép với keyframe theo thời gian.

Đặc điểm text (giống batch 1):

- Phần lớn là chữ thường, ít dấu câu (thường chỉ có dấu chấm cuối câu).
- **Số viết bằng chữ** trong audio tiếng Việt: `hai ngàn không trăm hai mươi sáu`, `mười ba giờ`.
  Query có chữ số (`2026`, `13h`) sẽ không khớp nếu index không chuẩn hoá số.
- Tên riêng nước ngoài có thể bị phiên âm sai (batch 1 có `siêu bão ra thôn`).

---

## 5. Manifest `speech_manifest_batch2.csv`

316 dòng, một dòng/video, dựng từ plan + JSON thực có ở cuối lần chạy.

| Cột | Ý nghĩa |
| --- | --- |
| `video_id` | như tên file JSON |
| `group` | `S01`, `M01` … `M10` |
| `r2_key` | key video trên R2 `aic26-media`, dạng `Videos/Videos_{group}/{video_id}.mp4` |
| `public_url` | URL phát video qua custom domain `https://video.baoencoder.site` (mục 7) |
| `duration_sec` | thời lượng video (ffprobe) |
| `video_codec` | codec video **tại thời điểm chạy** (`h264` hoặc `av1`, mục 6.4) |
| `audio_codec` | luôn `aac` |
| `audio_start_sec` | `start_time` của stream audio; luôn `0.0` |
| `status` | luôn `ok` |
| `n_segments`, `n_words` | số segment, số từ |
| `speech_sec` | tổng thời lượng các segment |
| `mean_avg_word_score` | trung bình `avg_word_score` của video; **< 0,3 = audio tiếng Anh** (mục 6.1) |
| `error` | rỗng ở cả 316 dòng |

```python
import pandas as pd

BASE = "https://huggingface.co/buckets/Baonenha1/aic26-media/resolve/Speech/speech_out_batch2"
speech = pd.read_csv(f"{BASE}/speech_manifest_batch2.csv", dtype={"video_id": str})
assert len(speech) == 316 and speech["video_id"].is_unique and (speech["status"] == "ok").all()
speech["lang"] = speech["mean_avg_word_score"].lt(0.3).map({True: "en", False: "vi"})
```

---

## 6. Chất lượng và lưu ý khi dùng

### 6.1 Có 12 video tin tức với audio tiếng Anh

```text
M02_V001
M04_V016  M04_V017  M04_V018  M04_V019  M04_V020
M04_V022  M04_V023  M04_V024  M04_V025  M04_V026  M04_V028
```

Đây là bản tiếng Anh của chương trình "60 giây" HTV. Chữ trên màn hình vẫn là tiếng Việt, nhưng
**lời đọc là tiếng Anh**. Đã kiểm độc lập bằng Whisper-base trên các đoạn 30 s: M04_V028 (giây 60 và
400) và M02_V001 (giây 120) nhận ra `en` với xác suất 0,98, còn video đối chứng M04_V001 ra `vi` 1,00.
M04_V021 và M04_V027 nằm giữa dải này nhưng là tiếng Việt bình thường.

Hệ quả:

- `text` là **tiếng Anh**, nội dung đúng ("greetings ladies and gentlemen … the sixty second program of
  ho chi minh city television …"). Query tiếng Việt sẽ không khớp: cần dịch query, index riêng cho
  tiếng Anh, hoặc dịch transcript trước khi index.
- `avg_word_score` chỉ khoảng 0,09–0,105, do aligner tiếng Việt chấm từ tiếng Anh. Đây **không** phải lỗi
  nhận dạng. Biên segment (`start`/`end`, lấy từ VAD) vẫn tin được; timestamp **từng từ** chỉ nên coi là
  gần đúng.
- Phân biệt được chắc chắn bằng `mean_avg_word_score < 0.3`: 12 video này nằm ở 0,09–0,105, mọi video
  tiếng Việt ≥ 0,478.

Toàn bộ batch 2 chạy với `language="vi"` giống batch 1. Nếu cần timestamp từ chính xác cho 12 video này,
có thể chạy lại riêng chúng với aligner tiếng Anh. Hiện chưa làm.

### 6.2 Điểm căn chỉnh của audio tiếng Việt tương đương batch 1

| | Segment | Decile `avg_word_score` (10%…90%) | Median |
| --- | ---: | --- | ---: |
| Batch 1 (K + L) | 48.819 | 0,468 · 0,503 · 0,524 · 0,540 · 0,553 · 0,564 · 0,575 · 0,588 · 0,604 | 0,553 |
| Batch 2, video tiếng Việt | 20.268 | 0,483 · 0,513 · 0,531 · 0,545 · 0,557 · 0,568 · 0,579 · 0,591 · 0,607 | 0,557 |

Batch 2 chạy với `min_word_score = 0.0` (không lọc theo điểm) giống batch 1. Không khuyến nghị lọc theo
`avg_word_score` khi index: điểm này đo độ khớp căn chỉnh, không đo độ đúng của transcript.

### 6.3 M10_V029: chỉ có speech trong 18 phút đầu

Video dài 92 phút (5.523 s), nhưng segment cuối kết thúc ở giây 1.104,7. Track audio vẫn dài đủ
5.523 s; đo âm lượng tại các giây 1.500, 3.000 và 5.000 đều ra **−91 dB (im lặng tuyệt đối)**. Không có
speech sau phút ~18 là đúng với dữ liệu, không phải lỗi pipeline.

### 6.4 S01: speech thưa hơn, 6 video chạy trên bản AV1

- Chỉ khoảng 69% thời lượng S01 là speech (M là 89%), vì là tường thuật đua xe có nhạc, tiếng đám đông
  và đoạn không lời. Lọc theo từ khoá thấy khoảng 12 segment là câu tiếng Anh ngắn, rải rác trong
  S01-V001/V003/V004/V006/V007/V010/V011.
- Lúc chạy, S01 đang được transcode AV1 → H.264 trên R2: V006–V011 được xử lý từ bản **AV1** (cột
  `video_codec`), các video còn lại từ bản H.264. Transcode chỉ động vào video: stream AAC của bản gốc và
  bản mới trùng codec, sample rate, thời lượng, bitrate và `start_time = 0` (đã so trên S01-V001 và
  S01-V004). Vì vậy transcript và timestamp **vẫn đúng cho bản H.264 hiện tại**. Tại thời điểm viết
  handoff, S01-V006 đã sang H.264; V007–V011 vẫn là AV1.

### 6.5 Trường có thể `null`

- **8 từ** có `start = end = null`, `score = 0`. Tất cả là ký hiệu (`£`, `€`, `–`, `�`) trong các video
  tiếng Anh M02_V001, M04_V016, M04_V019, M04_V025.
- **1 segment** có `words = []` và `avg_word_score = null`: S01-V006, giây 9.691,7–9.692,2 (0,5 s nhưng
  text là một câu đầy đủ, nhiều khả năng là hallucination gần cuối video).

Consumer phải xử lý `null` ở `words[].start/end` và `avg_word_score`.

### 6.6 Đã kiểm trên toàn bộ 316 file

- Mọi segment có đúng 8 trường, mọi từ có đúng 4 trường, như batch 1.
- `video_id` trong segment khớp tên file; segment không chồng lấn nhau và không dài quá 30 s.
- Mỗi video có từ 28 đến 666 segment (trung vị 51), không video nào 0 segment.

---

## 7. Map sang video

`video_id` trong speech trùng `video_id` của video trên R2 bucket `aic26-media`:

```text
r2_key     = Videos/Videos_{group}/{video_id}.mp4
public_url = https://video.baoencoder.site/{r2_key}
```

Lưu ý khi phát video batch 2 qua custom domain (kiểm ngày 2026-09-24):

- CDN có thể còn **cache 404 cũ** cho URL không query string (đã gặp với `M01_V001`).
  `batch2_content_index.csv` né bằng cách thêm `?v=<ETag>`; cách triệt để là Purge Cache URL đó trên Cloudflare.
- File > 512 MB (toàn bộ S01, vài video N) được trả `200` và bỏ qua `Range`, nên seek chậm. URL r2.dev của
  bucket (`https://pub-c707387e6f0c4d4b9e37f00506a26f04.r2.dev`) trả `206` đúng Range với mọi kích thước.

`AIC2026_R2_VIDEO_HANDOFF.md` (22/09) ghi batch 2 nằm ở bucket `aic2026`, video N là `.mov` và không có M.
Thông tin đó đã cũ: batch 2 (S01, N, M) hiện nằm ở `aic26-media`, tất cả là `.mp4` faststart.

Ví dụ ghép speech với video:

```python
import requests

BASE = "https://huggingface.co/buckets/Baonenha1/aic26-media/resolve/Speech/speech_out_batch2"
video_id, group = "M01_V001", "M01"
segments = requests.get(f"{BASE}/{video_id}.speech.json", timeout=60).json()
seg = segments[0]
url = f"https://video.baoencoder.site/Videos/Videos_{group}/{video_id}.mp4#t={seg['start']:.2f}"
```

---

## 8. Pipeline, tham số và chạy lại

Giữ **y hệt batch 1** để hai batch cùng chất lượng và định dạng:

| Thành phần | Giá trị |
| --- | --- |
| ASR | `vinai/PhoWhisper-large` commit `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb`, CTranslate2 `float16` |
| Aligner | `nguyenvulebinh/wav2vec2-base-vietnamese-250h` |
| Stack | venv Python 3.12: `whisperx 3.3.1`, `torch 2.2.2`, `transformers 4.48.3`, `numpy<2`, `ctranslate2<4.5`, cài với `uv --exclude-newer 2026-06-01` (resolve ra `numpy 1.26.4`, `ctranslate2 4.4.0`) |
| VAD | pyannote đóng gói sẵn trong whisperx 3.3.1 (onset 0,5, offset 0,363, chunk 30 s) |
| Decode | `language="vi"`, beam 5, best_of 5, temperature 0 → 1, `condition_on_previous_text=False` |
| Lọc | `min_word_score = 0.0`; bỏ segment rỗng, lặp từ, câu kiểu "cảm ơn các bạn đã theo dõi" |
| Audio | ffmpeg → WAV 16 kHz mono PCM s16le |

Chỉ khác batch 1 ở phần hạ tầng: `batch_size` 32 thay vì 16 (mỗi đoạn VAD đều được pad đủ 30 s nên
batch không đổi transcript), 4 process ASR chia một GPU, và video được tải thẳng từ R2 trong lúc ASR chạy.

Số liệu phiên chạy:

```text
Benchmark (1 process, 2 video M):  36x realtime, VRAM đỉnh 15,3 GiB
Chạy chính (314 video):            4 process, 6 luồng tải, 1 giờ 46 phút, 82x realtime tổng,
                                   ~21x mỗi process, VRAM đỉnh 64,7/80 GiB, GPU ~100%
```

Chạy lại:

- Mở `extract-audio-batch2-colab.ipynb` trên Colab (A100 + High-RAM). Secrets: `HF_TOKEN` (write),
  khuyến nghị thêm `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` của account chứa R2 `aic26-media`. Rồi **Run all**.
- Notebook tải các JSON đã có trên bucket về trước, rồi **bỏ qua** những video đó. Muốn chạy lại một video
  thì xoá `Speech/speech_out_batch2/<video_id>.speech.json` trên bucket trước.
- Model CT2 được cache ở `Speech/_cache/phowhisper-large-ct2/` (`model.bin` 3.086.913.037 bytes), nên
  phiên sau không phải convert lại.
- Kết quả được đồng bộ lên bucket mỗi 2 phút và khi cell kết thúc. Cell cuối dựng lại manifest và zip.

---

## 9. Liên hệ với các handoff khác

- `AIC2026_R2_VIDEO_HANDOFF.md`: video trên R2. Phần batch 2 đã cũ (mục 7).
- `AIC2026_HF_MEDIA_HANDOFF.md`: snapshot media K + L trên cùng HF bucket. Prefix `Speech/` là dữ liệu
  bổ sung, không nằm trong snapshot đó.
- `Media_Upload/batch2_content/`: index nội dung hình (contact sheet, OCR) của 614 video batch 2. Có thể
  ghép với speech theo `video_id`.

---

## 10. Checklist cho agent

Retrieval / backend:

```text
[ ] Đọc speech từ Speech/speech_out_batch2/ (hoặc zip), không chạy lại ASR.
[ ] Giữ nguyên video_id có "-" (S01) và "_" (M); ghép với video, keyframe theo video_id.
[ ] Gắn lang="en" cho 12 video ở mục 6.1 (mean_avg_word_score < 0.3); xử lý query tiếng Anh/dịch cho nhóm này.
[ ] Chuẩn hoá số khi index: transcript tiếng Việt viết số bằng chữ.
[ ] Xử lý null ở words[].start/end và avg_word_score.
[ ] Không lọc segment theo avg_word_score.
[ ] Nhóm N không có speech; batch 1 (K, L) lấy từ speech_out.zip.
```

Frontend:

```text
[ ] Seek theo segment.start (giây, trùng pts_time của video).
[ ] Highlight theo từ chỉ khi words[].start != null; với 12 video tiếng Anh nên highlight theo segment.
[ ] Video batch 2: dùng URL có ?v=<ETag> hoặc URL đã purge cache để tránh 404 cũ của CDN.
```
