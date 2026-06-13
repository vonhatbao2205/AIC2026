# Milvus/Zilliz Embedding Handoff

> Đọc file này khi build backend vector retrieval hoặc khi cần upload lại embeddings lên Milvus/Zilliz.
> File này không chứa endpoint/token thật. Secret vẫn nằm ở `milvus_endpoint.txt` và `milvus_token.txt`.

---

## 1. Trạng Thái Hiện Tại

Đã upload xong 2 collection embedding lên Zilliz/Milvus:

| Collection | Nguồn | Dtype Milvus | Dim | Remote count đã verify |
|---|---|---:|---:|---:|
| `aic26_image_peg14_v1` | `peG14.pkl` | `FLOAT_VECTOR` | `1280` | `382,299` |
| `aic26_audio_glap_v1` | `results_audio_event/audio_out/*.glap.npy` + `elastic_staging/audio_windows_mapped.jsonl` | `FLOAT16_VECTOR` | `1024` | `466,996` |

Quyết định dtype:

- Image PE-Core-G14 giữ `float32` vì đây là retrieval chính và embedding gốc trong `peG14.pkl` là `float32`.
- Audio GLAP giữ `float16` vì `.glap.npy` gốc đã là `float16`; convert lên `float32` sẽ tốn thêm dung lượng nhưng không khôi phục thêm thông tin.

Verification cuối:

```text
aic26_image_peg14_v1 remote_count 382299 expected 382299 complete True
aic26_audio_glap_v1 remote_count 466996 expected 466996 complete True
```

Search mẫu cũng đã verify:

```text
image top-1: L21/L21_V001/001, distance 1.0
audio top-1: K01_V001_a000000_000000000_000005000, distance 1.0
```

---

## 2. Files Và Script Đã Tạo

Script upload:

```text
milvus_upload.py
```

Test:

```text
tests/test_milvus_upload.py
```

Checkpoint resume:

```text
milvus_upload_checkpoint.json
```

Runtime local:

```text
.venv/
.uv-cache/
```

Package đã cài trong `.venv`:

```text
pymilvus
numpy
```

Run tests:

```bash
.venv/bin/python -m unittest \
  tests/test_milvus_upload.py \
  tests/test_keyframe_mapping.py \
  tests/test_elastic_upload.py
```

Kết quả verify gần nhất:

```text
Ran 15 tests
OK
```

---

## 3. Image Collection: `aic26_image_peg14_v1`

Nguồn:

```text
peG14.pkl
```

Pickle có metadata:

```text
model_name: PE-Core-G14-448
embedding_dim: 1280
total_images: 382299
```

Format chính:

```text
data["embeddings"][image_path] -> np.ndarray shape (1280,), dtype float32
```

Ví dụ key:

```text
Keyframes_L21/keyframes/L21_V001/001.jpg
```

Milvus schema:

```text
id                  VARCHAR primary key
keyframe_id         VARCHAR
submit_keyframe_id  VARCHAR
category            VARCHAR
video_id            VARCHAR
keyframe_n          INT64
image_path          VARCHAR
embedding           FLOAT_VECTOR dim=1280 metric=COSINE index=AUTOINDEX
```

ID chuẩn:

```text
id = submit_keyframe_id = <category>/<video_id>/<keyframe_3_digits>
L21/L21_V001/001
L26/L26_V001/007
```

Lưu ý normalize `L26_a`, `L26_b`, ... về `L26` theo `video_id`. Submit vẫn dùng `L26`.

Backend join:

- Dùng `submit_keyframe_id` để join với Elastic OCR/keyframe map/result UI.
- Dùng `video_id + keyframe_n` khi cần join chắc với OCR/keyframe map.
- Không dùng `image_path` để truy xuất video/keyframe. Field này chỉ là provenance/debug từ file embedding; OCR `image_path` trong Elastic còn là absolute Kaggle path. Nếu backend/frontend cần field tên `image_id`, hãy set `image_id = submit_keyframe_id`.

---

## 4. Audio Collection: `aic26_audio_glap_v1`

Nguồn vector:

```text
results_audio_event/audio_out/<video_id>.glap.npy
```

Nguồn metadata/join:

```text
elastic_staging/audio_windows_mapped.jsonl
```

GLAP shape:

```text
np.ndarray shape (N, 1024), dtype float16
```

Milvus schema:

```text
id                  VARCHAR primary key
video_id            VARCHAR
start               FLOAT
end                 FLOAT
glap_idx            INT64
keyframe_id         VARCHAR
submit_keyframe_id  VARCHAR
keyframe_n          INT64
top1_label          VARCHAR
embedding           FLOAT16_VECTOR dim=1024 metric=COSINE index=AUTOINDEX
```

ID chuẩn:

```text
id = window_id
K01_V001_a000000_000000000_000005000
```

Backend join:

- Milvus audio hit join Elastic audio metadata bằng `id == window_id`.
- Có thể join nhanh về keyframe bằng `submit_keyframe_id`.
- Có thể fuse temporal bằng `video_id`, `start`, `end`.

---

## 5. Command Upload Đã Dùng

Tạo `.venv`:

```bash
UV_CACHE_DIR=/home/bao/Projects/ExtractAudio/.uv-cache uv venv .venv
```

Cài SDK:

```bash
UV_CACHE_DIR=/home/bao/Projects/ExtractAudio/.uv-cache \
  uv pip install --python .venv/bin/python pymilvus numpy
```

Upload end-to-end:

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

Script có checkpoint, nên nếu rớt giữa chừng chỉ cần chạy lại cùng command. `upsert` được dùng mặc định nên retry overlap không tạo duplicate.

---

## 6. Lỗi Đã Gặp Và Cách Xử Lý

Trong lúc upload Zilliz serverless có vài lần timeout:

```text
StatusCode.UNAVAILABLE
Stream removed (recvmsg:Connection timed out)
```

Đã sửa `milvus_upload.py` để:

- retry từng batch;
- reconnect bằng client alias mới;
- backoff theo `--retry-sleep`;
- ghi checkpoint sau mỗi batch thành công.

Khi remote count thấp hơn checkpoint do batch commit chưa chắc chắn, đã lùi checkpoint về remote count rồi resume. Vì dùng `upsert`, upload lại vùng overlap là an toàn.

---

## 7. Verify Sau Upload

Kiểm tra count:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from milvus_upload import connect_client, get_row_count

expected = {
    "aic26_image_peg14_v1": 382299,
    "aic26_audio_glap_v1": 466996,
}

client = connect_client(Path("milvus_endpoint.txt"), Path("milvus_token.txt"))
for name, exp in expected.items():
    count = get_row_count(client, name)
    print(name, count, exp, count == exp)
PY
```

Kiểm tra vector type/schema:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from milvus_upload import connect_client

client = connect_client(Path("milvus_endpoint.txt"), Path("milvus_token.txt"))
for name in ["aic26_image_peg14_v1", "aic26_audio_glap_v1"]:
    desc = client.describe_collection(collection_name=name)
    print(name)
    for field in desc.get("fields") or []:
        if field.get("name") == "embedding":
            print(field.get("type"), field.get("params"))
PY
```

Kết quả mong đợi:

```text
aic26_image_peg14_v1 embedding type 101 dim 1280
aic26_audio_glap_v1 embedding type 102 dim 1024
```

Trong PyMilvus:

```text
101 = FLOAT_VECTOR
102 = FLOAT16_VECTOR
```

---

## 8. Backend Retrieval Notes

Image retrieval:

1. Encode text/query hoặc image query bằng đúng model PE-Core-G14-448.
2. Search `aic26_image_peg14_v1`.
3. Join hit về Elastic/keyframe/OCR bằng `submit_keyframe_id`.

Audio retrieval:

1. Encode sound-event query bằng đúng GLAP text/audio encoder.
2. Search `aic26_audio_glap_v1`.
3. Join hit về Elastic audio metadata bằng `id == window_id`.
4. Fuse temporal với speech/OCR bằng `video_id`, `start/end`, `submit_keyframe_id`.

Backend không nên expose Milvus token cho frontend. Frontend gọi backend; backend gọi Elastic + Milvus/Zilliz.

---

## 9. Re-upload Khi Đổi Zilliz/Milvus Cluster

Nếu cluster hết hạn hoặc đổi endpoint/token:

1. Cập nhật:

```text
milvus_endpoint.txt
milvus_token.txt
```

2. Xóa hoặc đổi checkpoint nếu upload vào cluster mới:

```bash
mv milvus_upload_checkpoint.json milvus_upload_checkpoint.old.json
```

3. Chạy lại upload:

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

Nếu chỉ muốn upload một collection:

```bash
.venv/bin/python milvus_upload.py --only image --image-vector-dtype float32
.venv/bin/python milvus_upload.py --only audio --audio-vector-dtype float16
```

Nếu cần recreate collection do đổi dtype/schema:

```bash
.venv/bin/python milvus_upload.py \
  --only image \
  --image-vector-dtype float32 \
  --drop-existing \
  --ignore-checkpoint
```

---

## 10. Mental Model

```text
Zilliz/Milvus
  image vectors: PE-Core-G14-448, keyframe-level, primary retrieval
  audio vectors: GLAP, audio-window-level, semantic sound retrieval

Elastic Cloud
  speech transcript metadata
  audio tag/caption metadata
  OCR text metadata
  keyframe time map

Fusion backend
  queries Elastic + Milvus
  joins through submit_keyframe_id, window_id, video_id, and time
  returns ranked keyframes with evidence
```
