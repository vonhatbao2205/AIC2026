# Elastic Retrieval Handoff — Speech, Audio, OCR, Keyframes

> Đọc file này trước khi bắt đầu build backend/frontend retrieval.
> File này ghi lại những gì đã chuẩn bị, đã upload lên Elastic Cloud, và cách nối các nhánh metadata với nhau.

---

## 1. Mục tiêu

Hệ retrieval hiện có 4 nguồn metadata đã sẵn sàng trên Elastic Cloud:

- **speech**: ASR transcript từ `speech_out/*.speech.json`.
- **audio**: audio event tags/caption từ `results_audio_event/audio_out/*.audio.json`.
- **OCR**: keyframe OCR từ `ocr_clean.jsonl`.
- **keyframe map**: cầu nối `time seconds <-> keyframe submit id` từ `map-keyframes-s1-s2/**/*.csv`.

Embedding vector **không nằm trong Elastic**. Image PE-G14 và audio GLAP đã được upload lên Zilliz/Milvus; xem chi tiết trong `MILVUS_EMBEDDING_HANDOFF.md`.

---

## 2. Files Và Scripts Đã Tạo

### Mapper

Script:

```bash
python keyframe_mapping.py \
  --map-dir map-keyframes-s1-s2 \
  --audio-dir results_audio_event/audio_out \
  --speech-dir speech_out \
  --out-dir elastic_staging
```

Tạo staging:

```text
elastic_staging/keyframe_map.jsonl
elastic_staging/audio_windows_mapped.jsonl
elastic_staging/speech_segments_mapped.jsonl
elastic_staging/mapping_summary.json
```

Raw data **không bị sửa**.

### Elastic Uploader

Script:

```bash
python elastic_upload.py \
  --endpoint-file elastic_endpoint.txt \
  --api-key-file elastic_apikey.txt \
  --staging-dir elastic_staging \
  --ocr-path ocr_clean.jsonl \
  --index-prefix aic26 \
  --batch-size 2000 \
  --summary-path elastic_staging/elastic_upload_summary.json
```

Uploader dùng Python stdlib `urllib`, không cần package ngoài. Có retry cho lỗi network tạm thời và có resume:

```bash
python elastic_upload.py \
  --endpoint-file elastic_endpoint.txt \
  --api-key-file elastic_apikey.txt \
  --staging-dir elastic_staging \
  --ocr-path ocr_clean.jsonl \
  --index-prefix aic26 \
  --batch-size 2000 \
  --resume-existing \
  --only audio_windows ocr_keyframes
```

Không ghi API key vào log. Không commit/ghi secret vào Markdown.

### Tests

```bash
python -m unittest tests/test_keyframe_mapping.py tests/test_elastic_upload.py
```

Hiện có 10 tests pass.

### Milvus/Zilliz Embedding Uploader

Script:

```bash
.venv/bin/python milvus_upload.py \
  --endpoint-file milvus_endpoint.txt \
  --token-file milvus_token.txt \
  --image-vector-dtype float32 \
  --audio-vector-dtype float16 \
  --batch-size 1024 \
  --max-retries 10 \
  --retry-sleep 15
```

Upload xong 2 collection:

```text
aic26_image_peg14_v1  382,299 vectors  FLOAT_VECTOR    dim 1280
aic26_audio_glap_v1   466,996 vectors  FLOAT16_VECTOR  dim 1024
```

Chi tiết schema, retry, checkpoint, và verify nằm trong `MILVUS_EMBEDDING_HANDOFF.md`.

---

## 3. Elastic Cloud Indices Đã Upload

Index prefix: `aic26`

| Index | Nguồn | Remote count đã verify |
|---|---:|---:|
| `aic26_keyframe_map_v1` | `elastic_staging/keyframe_map.jsonl` | `382,299` |
| `aic26_speech_segments_v1` | `elastic_staging/speech_segments_mapped.jsonl` | `48,819` |
| `aic26_audio_windows_v1` | `elastic_staging/audio_windows_mapped.jsonl` | `466,996` |
| `aic26_ocr_keyframes_v1` | `ocr_clean.jsonl` qua transform | `382,299` |

Verification cuối:

```text
aic26_keyframe_map_v1       382299 ok True
aic26_speech_segments_v1     48819 ok True
aic26_audio_windows_v1      466996 ok True
aic26_ocr_keyframes_v1      382299 ok True
```

Search mẫu đã verify:

- Speech `match text: "thủ tướng"` trả hit có `submit_keyframe_id`.
- Audio `term top1_label: "Music"` trả hit có `submit_keyframe_id`.
- OCR `match text_clean: "Biển Đông"` trả hit có `submit_keyframe_id`.

---

## 4. Key Identity Fields

### `submit_keyframe_id`

Đây là ID chuẩn nên dùng cho backend/frontend và submit:

```text
<category>/<video_id>/<keyframe_3_digits>
K01/K01_V001/001
L26/L26_V001/014
```

Lưu ý: OCR raw có thể có category shard như `L26_a`, nhưng submit chuẩn dùng `L26`. Uploader đã normalize OCR:

```text
L26_a/L26_V001/001 -> L26/L26_V001/001
```

### Không Dùng `image_path` Làm ID

Không dùng `image_path` trong Elastic hoặc Milvus để truy xuất video/keyframe. Với OCR, `image_path` là đường dẫn tuyệt đối từ môi trường extraction, ví dụ:

```text
/kaggle/input/datasets/.../Keyframes_K01/keyframes/K01_V001/001.jpg
```

Đường dẫn này phụ thuộc máy/Kaggle/dataset mount, không ổn định cho backend/frontend và không nên dùng làm route, cache key, submit ID, hoặc join key.

Khóa nên dùng:

```text
submit_keyframe_id      # public/display/submit id, vd K01/K01_V001/001
keyframe_id             # internal compact id, vd K01_V001/001
video_id + keyframe_n   # join chắc giữa OCR/keyframe map/vector
window_id               # riêng audio window
```

Nếu API frontend muốn field tên `image_id`, hãy đặt:

```text
image_id = submit_keyframe_id
```

`image_path` chỉ nên giữ để debug/provenance, không dùng để truy xuất keyframe/video.

### Internal Keyframe Fields

Audio/speech records có cả 3 mốc:

```text
start_submit_keyframe_id
center_submit_keyframe_id
end_submit_keyframe_id
submit_keyframe_id              # alias của center_submit_keyframe_id
```

Khi UI cần một thumbnail/keyframe đại diện, dùng `submit_keyframe_id`. Khi fusion cần khoảng thời gian, dùng cả start/center/end.

### Document IDs

Elastic document IDs:

```text
keyframe_map:    submit_keyframe_id
speech_segments: segment_id
audio_windows:   window_id
ocr_keyframes:   submit_keyframe_id
```

---

## 5. Index Schema Tóm Tắt

### `aic26_keyframe_map_v1`

Mỗi doc = 1 keyframe.

Fields chính:

```text
submit_keyframe_id
keyframe_id              # internal: K01_V001/001
video_id
submit_category
keyframe_n
keyframe_name            # "001"
pts_time
fps
frame_idx
```

Dùng để:

- map OCR/keyframe về `pts_time`;
- map speech/audio theo giây sang keyframe;
- lấy timing cho UI/fusion.

### `aic26_speech_segments_v1`

Mỗi doc = 1 ASR segment.

Fields chính:

```text
segment_id
video_id
start
end
duration
center_time
text
avg_word_score
confidence_bucket        # high | mid | low | missing
segment_role             # intro | preview | body
word_count
submit_keyframe_id
start_submit_keyframe_id
center_submit_keyframe_id
end_submit_keyframe_id
```

Confidence policy:

```text
avg_word_score < 0.40        -> low
0.40 <= score < 0.50         -> mid
score >= 0.50                -> high
None                         -> missing
```

Query/backend khuyến nghị:

- Drop hoặc rất hạ điểm `confidence_bucket=low`.
- Hạ điểm `mid`.
- Hạ trọng số `segment_role=intro/preview` khi dùng làm temporal anchor.

### `aic26_audio_windows_v1`

Mỗi doc = 1 audio window 5s/hop 2.5s.

Fields chính:

```text
window_id
video_id
start
end
duration
center_time
glap_idx
tags                    # nested [{label, score}]
tag_labels
tag_scores
top1_label
top1_score
caption
has_caption
caption_quality          # none | useful | generic | vietnamese_asr
audio_stoplist_hit
top1_is_stoplisted
submit_keyframe_id
start_submit_keyframe_id
center_submit_keyframe_id
end_submit_keyframe_id
```

Important audio noise:

```text
Speech synthesizer
Mantra
```

Mapper marks:

```text
audio_stoplist_hit = true if any tag label is in stoplist
top1_is_stoplisted = true if top1 label is stoplisted
```

Query/backend khuyến nghị:

- Không coi top-5 tags ngang nhau.
- Tin top1 hơn; tag phụ chỉ nên tin khi score cao hoặc match cụm liên quan.
- Hạ/drop `caption_quality=generic` và `vietnamese_asr`.
- GLAP vector search ở Milvus là kênh thứ hai; Elastic audio chỉ cho tags/caption/filter.

### `aic26_ocr_keyframes_v1`

Mỗi doc = 1 keyframe OCR.

Uploader bỏ field `raw` để giảm dung lượng, giữ text đã clean.

Fields chính:

```text
ocr_id                  # raw OCR id, vd L26_a/L26_V001/001
submit_keyframe_id      # normalized, vd L26/L26_V001/001
video_id
category                # raw category/shard
submit_category
image_path
status
keyframe_n
keyframe_name
text_clean
text_clean_fold
text_nfc
clock
hour
boxes                   # nested [{text, box:[x1,y1,x2,y2]}]
```

Query/backend khuyến nghị:

- Search có dấu trên `text_clean`.
- Search không dấu/fuzzy trên `text_clean_fold`.
- Filter giờ bằng `hour`, `clock`.
- OCR chỉ nên bật mạnh khi query có dấu hiệu OCR-likely: tên riêng, số, quote, brand, chữ trên màn hình.

---

## 6. Backend Retrieval Flow Gợi Ý

### Text Query

Với query tiếng Việt bình thường:

1. Search speech:
   - `match text`.
   - filter/hạ điểm theo `confidence_bucket`, `segment_role`.
2. Search OCR nếu query OCR-likely:
   - `match text_clean`.
   - `match text_clean_fold` với fuzzy nếu cần.
3. Search audio tags/caption nếu query có sound-event:
   - `tag_labels`, `top1_label`, `caption`.
   - Hạ/drop stoplist/generic caption.
4. Search Milvus GLAP cho sound-event semantic:
   - Query tiếng Việt nên dịch sound-event sang English trước khi encode GLAP.
   - Milvus hit join về metadata bằng `window_id` hoặc `(video_id, start/end)`.
5. Search Milvus PE-G14 cho image/text-image semantic:
   - Dùng đúng encoder PE-Core-G14-448.
   - Milvus hit join về OCR/keyframe/thumbnail bằng `submit_keyframe_id`.
6. Fuse bằng RRF, rồi temporal co-occurrence.

### Temporal Fusion

Mỗi hit đều có:

```text
video_id
start/end or pts_time
submit_keyframe_id
```

Speech/audio:

```text
center_time = (start + end) / 2
submit_keyframe_id = nearest keyframe to center_time
```

OCR:

```text
submit_keyframe_id -> keyframe_map -> pts_time
```

Fusion condition cơ bản:

```text
same video_id
abs(speech.center_time - audio.center_time) <= 3s or 5s
abs(keyframe.pts_time - speech/audio center_time) <= window
```

Với TRAKE/sequence queries, cần giữ thứ tự keyframe tăng dần, không chỉ match từng event độc lập.

---

## 7. Frontend Integration Notes

Frontend không cần biết raw extraction files. Backend nên trả mỗi result gồm:

Chi tiết cách build `keyframe_url` và `video_url` từ Cloudflare R2 nằm trong `CLOUDFLARE_R2_MEDIA_HANDOFF.md`.

```json
{
  "video_id": "K01_V001",
  "submit_keyframe_id": "K01/K01_V001/144",
  "keyframe_n": 144,
  "pts_time": 345.0,
  "score": 0.91,
  "channels": ["speech", "audio", "ocr"],
  "evidence": [
    {"type": "speech", "text": "...", "start": 342.1, "end": 350.0},
    {"type": "audio", "top1_label": "Music", "start": 345.0, "end": 350.0},
    {"type": "ocr", "text_clean": "..."}
  ]
}
```

UI nên hiển thị:

- keyframe thumbnail theo `submit_keyframe_id`;
- video id/category;
- timestamp;
- badges nguồn match: `speech`, `audio`, `ocr`, `vector`;
- evidence snippets ngắn.

Không expose API key Elastic ra frontend. Frontend gọi backend, backend gọi Elastic/Milvus.

---

## 8. Credentials Và Security

Local files hiện có:

```text
elastic_endpoint.txt
elastic_apikey.txt
api-key.txt
milvus_endpoint.txt
milvus_token.txt
```

Không in API key/token ra log. Không đưa API key/token vào Markdown. Backend nên đọc secret từ env:

```text
ELASTIC_ENDPOINT
ELASTIC_API_KEY
MILVUS_ENDPOINT
MILVUS_TOKEN
```

Nếu dùng file trong dev, nhớ không commit secret.

---

## 9. Commands Hữu Ích

Run tests:

```bash
python -m unittest tests/test_keyframe_mapping.py tests/test_elastic_upload.py
.venv/bin/python -m unittest tests/test_milvus_upload.py
```

Regenerate staging:

```bash
python keyframe_mapping.py \
  --map-dir map-keyframes-s1-s2 \
  --audio-dir results_audio_event/audio_out \
  --speech-dir speech_out \
  --out-dir elastic_staging
```

Upload all indices:

```bash
python elastic_upload.py \
  --endpoint-file elastic_endpoint.txt \
  --api-key-file elastic_apikey.txt \
  --staging-dir elastic_staging \
  --ocr-path ocr_clean.jsonl \
  --index-prefix aic26 \
  --batch-size 2000 \
  --summary-path elastic_staging/elastic_upload_summary.json
```

Resume selected indices:

```bash
python elastic_upload.py \
  --endpoint-file elastic_endpoint.txt \
  --api-key-file elastic_apikey.txt \
  --staging-dir elastic_staging \
  --ocr-path ocr_clean.jsonl \
  --index-prefix aic26 \
  --batch-size 2000 \
  --resume-existing \
  --only audio_windows ocr_keyframes
```

---

## 10. Current Known Gaps

- Need backend query encoder integration for PE-Core-G14 image search and GLAP audio search.
- Need backend query/fusion layer.
- Need frontend result UI.
- Need Vietnamese analyzer/tokenizer decision for Elastic. Current mappings use built-in `text` fields, not custom Vietnamese tokenizer.
- Speech post-processing beyond confidence/role is not fully implemented:
  - no 8-12s re-segmentation yet;
  - no Vietnamese number normalization yet;
  - no synonym/alias expansion yet.
- OCR L23 was noted in `OCR_PIPELINE.md` as small (`2,326` frames); verify if this matters for final submission.

---

## 11. Re-upload Khi Elastic Cloud Hết Hạn Hoặc Đổi Cluster

Nếu Elastic Cloud cũ hết hạn, không cần chạy lại extraction. Chỉ cần có lại các file local:

```text
elastic_staging/keyframe_map.jsonl
elastic_staging/audio_windows_mapped.jsonl
elastic_staging/speech_segments_mapped.jsonl
ocr_clean.jsonl
```

Nếu mất `elastic_staging/`, regenerate bằng:

```bash
python keyframe_mapping.py \
  --map-dir map-keyframes-s1-s2 \
  --audio-dir results_audio_event/audio_out \
  --speech-dir speech_out \
  --out-dir elastic_staging
```

Sau đó tạo Elastic Cloud project/cluster mới, lấy endpoint và API key mới, rồi cập nhật 2 file:

```text
elastic_endpoint.txt
elastic_apikey.txt
```

Kiểm tra kết nối:

```bash
python - <<'PY'
from pathlib import Path
from elastic_upload import ElasticClient, read_secret
client = ElasticClient(read_secret(Path("elastic_endpoint.txt")), read_secret(Path("elastic_apikey.txt")))
info = client.json_request("GET", "/", expected=(200,), timeout=30)
print("connected", info.get("cluster_name"), (info.get("version") or {}).get("number"))
PY
```

Upload lại toàn bộ lên cloud mới:

```bash
python elastic_upload.py \
  --endpoint-file elastic_endpoint.txt \
  --api-key-file elastic_apikey.txt \
  --staging-dir elastic_staging \
  --ocr-path ocr_clean.jsonl \
  --index-prefix aic26 \
  --batch-size 2000 \
  --summary-path elastic_staging/elastic_upload_summary.json
```

Với cloud mới trống, **không cần** `--resume-existing`.

Nếu upload bị ngắt giữa chừng, chạy lại phần còn thiếu bằng:

```bash
python elastic_upload.py \
  --endpoint-file elastic_endpoint.txt \
  --api-key-file elastic_apikey.txt \
  --staging-dir elastic_staging \
  --ocr-path ocr_clean.jsonl \
  --index-prefix aic26 \
  --batch-size 2000 \
  --resume-existing \
  --only audio_windows ocr_keyframes
```

Có thể thay `--only ...` theo index đang bị dở:

```text
keyframe_map
speech_segments
audio_windows
ocr_keyframes
```

Nếu muốn giữ song song nhiều Elastic project, đổi prefix:

```bash
--index-prefix aic26_test2
```

Khi đó index sẽ thành:

```text
aic26_test2_keyframe_map_v1
aic26_test2_speech_segments_v1
aic26_test2_audio_windows_v1
aic26_test2_ocr_keyframes_v1
```

---

## 12. Mental Model

Think of the system as:

```text
Zilliz/Milvus
  PE-G14 image vectors
  GLAP audio vectors

Elastic Cloud
  speech text
  audio tags/caption text
  OCR text
  keyframe time map

Fusion backend
  turns query into channel searches
  joins all hits through video_id + time + submit_keyframe_id
  ranks with RRF/co-occurrence

Frontend
  shows keyframe-centric results with evidence snippets
```

The common join key for display and submit is:

```text
submit_keyframe_id
```
