# Handoff — PE-Core-G14-448 cho AIC 2026 Retrieval

Tài liệu này là nguồn handoff độc lập cho AI/kỹ sư tiếp theo tích hợp collection
PE-Core-G14-448 vào hệ thống retrieval. Không cần đọc lại lịch sử hội thoại để sử
dụng artifact, nhưng nên đọc các manifest production được dẫn ở phần dưới trước khi
thay đổi pipeline.

Trạng thái cập nhật: **2026-08-14**.

## 1. Trạng thái chốt

Đã encode hoàn tất toàn bộ **1.339.055 keyframe InfoShot++** thuộc L21–L30 bằng
`PE-Core-G14-448`.

```text
complete:             true
total_committed_rows: 1,339,055
expected_total_rows:  1,339,055
embedding_dim:        1,280
storage_dtype:        float32
normalized:           true
```

Artifact production trên Hugging Face Storage Bucket:

```text
hf://buckets/Baonenha1/DATA-AIC-Keyframe/derived/pe-core-g14-448-v1/
```

Bản đã tải về máy:

```text
/home/bao/Projects/EncoderModel/pe-core-g14-448-v1/
```

Notebook/source đã dùng để encode:

- [`PE_Core_G14_448_AIC2026_Colab.ipynb`](PE_Core_G14_448_AIC2026_Colab.ipynb)
- [`PE_Core_G14_448_AIC2026_Colab.py`](PE_Core_G14_448_AIC2026_Colab.py)
- Notebook legacy dùng làm parity reference:
  [`perceptiong14.ipynb`](perceptiong14.ipynb)
- Mô tả corpus keyframe:
  [`AIC2026_InfoShotPP_Kaggle_Keyframe.md`](AIC2026_InfoShotPP_Kaggle_Keyframe.md)

## 2. Identity contract bắt buộc

Mọi query vector dùng với collection này phải tuân thủ cùng PE model space:

```text
model:                  PE-Core-G14-448
upstream repository:    https://github.com/facebookresearch/perception_models.git
implementation git SHA: 3e352cca660658d4b5c90f42a7808b11469e4c66
checkpoint SHA-256:     516a20fb30394cc8080344c32d7e49e0e575cf2c6d141977207bf94b3bec192f
image resolution:       448 × 448
embedding dimension:    1,280
text context length:    72
semantic fingerprint:   2cce817755d48b2a59c50a2a345d5c5a42f59fc46b4b6bd1ce3668eca1b03141
```

Không thay một trong các thành phần sau rồi ghi vector vào cùng collection:

- model/config khác, đặc biệt `PE-Core-L14-336`;
- checkpoint khác;
- resize/crop khác;
- output không normalize;
- dimension hoặc dtype khác;
- implementation upstream khác mà chưa chạy parity test.

> **Cảnh báo quan trọng:** một số cell UI cũ trong `perceptiong14.ipynb` có biến
> `PERCEPTION_MODEL_NAME = "PE-Core-L14-336"`. Đây không phải model của artifact
> production này. Không copy cấu hình đó vào query service. Collection hiện tại là
> **G14-448, 1280-d**.

## 3. Nguồn dữ liệu

Nguồn keyframe:

```text
bucket: Baonenha1/DATA-AIC-Keyframe
prefix: infoshootpp-v1
```

Canonical key:

```text
frame_id = "{video_id}@f{frame_idx:08d}"
```

Số lượng production:

| Category        |                Rows | Parquet shards |
| --------------- | ------------------: | -------------: |
| L21             |             111.031 |             14 |
| L22             |             130.134 |             16 |
| L23             |              40.808 |              5 |
| L24             |              69.826 |              9 |
| L25             |             154.211 |             19 |
| L26             |             559.060 |             69 |
| L27             |              33.201 |              5 |
| L28             |              90.042 |             11 |
| L29             |              83.697 |             11 |
| L30             |              67.045 |              9 |
| **Tổng** | **1.339.055** |  **168** |

Các checksum nguồn được khóa trong notebook và `run_config.json`. Không join dữ
liệu bằng số thứ tự ảnh kiểu `001.jpg`; dùng `frame_id`, `video_id` và `frame_idx`.

## 4. Semantic preprocessing và inference

### 4.1 Image-side preprocessing

Contract đã dùng để tạo collection:

```text
Pillow decode
→ convert("RGB")
→ official PE bilinear resize, squash thành 448 × 448
→ không center crop
→ tensor [0, 1]
→ normalize mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
→ PE-Core-G14-448 vision tower
→ L2 normalize trong FP32
→ lưu float32
```

Pipeline production tối ưu I/O bằng cách resize trên CPU rồi truyền `uint8` sang
GPU; GPU mới đổi sang FP32 và normalize. Preprocess parity đã được kiểm với official
transform trước khi encode.

Không được thay squash resize bằng center crop, letterbox, tiling hoặc dynamic
resolution nếu muốn giữ cùng image embedding contract.

### 4.2 Numeric policy

```text
model weights:        FP32
forward:              CUDA autocast FP16
model mode:           eval + inference_mode
TF32:                 disabled
quantization:         none
output normalization: FP32
storage:              FP32
```

Production đã audit FP16 so với pure FP32 trên 64 ảnh phân bố đều. Cả ba runtime
đều vượt ngưỡng:

```text
required min cosine:  >= 0.999
required mean cosine: >= 0.9999

observed worst min cosine:  0.9997063
observed worst mean cosine: 0.9999517
```

`torch.compile` đã được benchmark nhưng bị từ chối vì min cosine khoảng
`0.9999866–0.9999893`, thấp hơn contract compile `0.99999`. Vector production được
encode bằng **eager runner**, không phải compiled runner.

### 4.3 Throughput thực tế

Các runtime production:

- NVIDIA A100-SXM4-80GB;
- NVIDIA A100-SXM4-40GB;
- batch size được autotune và đều chọn `64`;
- `11` DataLoader workers;
- throughput benchmark khoảng `29–31 ảnh/giây`;
- batch lớn hơn dùng thêm VRAM nhưng không tăng throughput đáng kể.

Batch size và worker là execution config, không làm thay đổi semantic vector khi
pipeline deterministic/parity vẫn giữ nguyên.

## 5. Cấu trúc artifact

```text
pe-core-g14-448-v1/
├── embeddings/
│   ├── L21/part-00000.parquet
│   ├── ...
│   └── L30/part-00008.parquet
├── commits/
│   ├── L21/part-00000.json
│   ├── ...
│   └── L30/part-00008.json
├── executions/
│   └── <runtime timestamp>.json
├── success/
│   └── Lxx.json
├── run_config.json
├── embedding_dataset_manifest.json
└── _SUCCESS.json
```

Tổng local hiện tại:

```text
352 files
168 Parquet files
168 commit JSON files
10 category success files
3 execution records
6.510 GiB tổng dung lượng
```

Mỗi shard có tối đa `8.192` rows; shard cuối của category có thể nhỏ hơn. Parquet
không compression để ưu tiên tốc độ đọc và giữ fixed-size float vectors.

### 5.1 Parquet schema

| Column            | Arrow type                         | Ý nghĩa                                            |
| ----------------- | ---------------------------------- | ---------------------------------------------------- |
| `frame_id`      | `string`                         | Canonical ID, ví dụ`L21_V001@f00000000`          |
| `video_id`      | `string`                         | Ví dụ`L21_V001`                                  |
| `category`      | `string`                         | `L21` … `L30`                                   |
| `frame_idx`     | `int64`                          | Decode-order frame index thật                       |
| `pts_time`      | `float64`                        | Timestamp PTS theo giây                             |
| `fps`           | `float32`                        | FPS của video                                       |
| `image_relpath` | `string`                         | Đường dẫn JPEG tương đối trong corpus nguồn |
| `embedding`     | `fixed_size_list<float32>[1280]` | Unit vector PE-G14                                   |

Arrow metadata trên mỗi Parquet còn chứa:

```text
schema_version
semantic_fingerprint
model_name
checkpoint_sha256
embedding_dim
embedding_dtype
l2_normalized
```

### 5.2 Commit protocol

Encoder dùng transaction boundary theo thứ tự:

```text
write local Parquet
→ tính SHA-256
→ upload Parquet
→ verify remote size
→ upload commit JSON
→ mới xem shard là hoàn tất
```

Resume chỉ tin shard có commit hợp lệ, đúng fingerprint, đúng row range và đúng
remote byte size. Không có distributed lock giữa hai notebook; nếu encode tiếp về
sau, chỉ chạy song song trên category/shard không giao nhau.

## 6. Những gì đã xác minh sau khi tải về

Local artifact đã được audit ngày 2026-08-14:

```text
manifest complete:           true
manifest total rows:         1,339,055
commit files:                168
rows cộng từ commit:         1,339,055
Parquet size mismatches:     0
Parquet SHA-256 mismatches:  0
Parquet bytes đã hash:       6,989,867,810
```

Không chỉ kiểm size: toàn bộ 168 Parquet đã được đọc và hash lại, tất cả khớp
`parquet_sha256` trong commit tương ứng.

Để startup retrieval fail-fast, luôn kiểm `_SUCCESS.json` và semantic fingerprint:

```python
import json
from pathlib import Path

ROOT = Path("/home/bao/Projects/EncoderModel/pe-core-g14-448-v1")
EXPECTED_ROWS = 1_339_055
EXPECTED_DIM = 1_280
EXPECTED_FINGERPRINT = (
    "2cce817755d48b2a59c50a2a345d5c5a42f59fc46b4b6bd1ce3668eca1b03141"
)

success = json.loads((ROOT / "_SUCCESS.json").read_text(encoding="utf-8"))
assert success["complete"] is True
assert success["total_committed_rows"] == EXPECTED_ROWS
assert success["expected_total_rows"] == EXPECTED_ROWS
assert success["semantic_fingerprint"] == EXPECTED_FINGERPRINT
assert success["retrieval_contract"]["dimension"] == EXPECTED_DIM
assert success["retrieval_contract"]["image_vectors_l2_normalized"] is True
```

Nếu check này fail, không build hoặc phục vụ index từ prefix đó.

## 7. Contract retrieval

Image vectors đã là unit vectors. Metric đúng là:

```text
cosine similarity
```

hoặc tương đương:

```text
inner product trên query và image vector đã L2-normalize
```

Không dùng L2 distance trực tiếp trừ khi hệ thống hiểu rõ quan hệ với unit vector.
Không dùng dot product với query chưa normalize.

### 7.1 Query-side encoder

Query phải dùng **text tower của PE-Core-G14-448**, official text tokenizer và
context length `72`. Offline collection chỉ cần vision tower, nhưng query service
cần full CLIP hoặc một text-only extraction đã được parity-test.

Ví dụ contract theo API tại upstream SHA đã pin:

```python
import torch
import torch.nn.functional as F

from core.vision_encoder import pe
from core.vision_encoder import transforms as pe_transforms

MODEL_NAME = "PE-Core-G14-448"

model = pe.CLIP.from_config(MODEL_NAME, pretrained=True)
model.requires_grad_(False).eval().cuda()
tokenizer = pe_transforms.get_text_tokenizer(model.context_length)


@torch.inference_mode()
def encode_text_queries(texts: list[str]):
    tokens = tokenizer(texts).cuda(non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _, text_features, _ = model(None, tokens)
    return F.normalize(text_features.float(), p=2, dim=-1).cpu().numpy()
```

Startup assertions:

```python
query = encode_text_queries(["a person riding a bicycle"])
assert query.shape == (1, 1280)
assert abs(float((query * query).sum()) - 1.0) < 2e-5
```

Ghi chú:

- checkout đúng upstream SHA trước khi load model;
- xác minh checkpoint SHA nếu query service tự quản checkpoint;
- PE không phải instruction-aware model như Qwen3-VL; không tự thêm prompt kiểu
  `Represent the user's input.` vào query PE;
- benchmark query tiếng Việt trực tiếp so với bản dịch tiếng Anh trên validation
  set AIC. Không mặc định một lựa chọn nếu chưa có metric;
- nếu đổi precision query, chạy recall/parity test trước khi deploy.

### 7.5 Filtering và temporal dedup

InfoShot++ là high-recall và khá dày. Top-K thô thường chứa nhiều frame gần nhau
trong cùng video. Retrieval layer nên:

1. lấy top candidate lớn hơn số kết quả cần hiển thị, ví dụ 100–300;
2. áp dụng filter `category`/`video_id` nếu query yêu cầu;
3. temporal NMS hoặc grouping theo `(video_id, pts_time)`;
4. giữ frame có score cao nhất trong mỗi cửa sổ thời gian;
5. trả thêm neighboring frames khi UI cần duyệt timeline.

Window NMS cụ thể phải tune trên AIC validation; không hard-code một giá trị nếu chưa
đo miss rate cho event ngắn.

## 8. So sánh với `peG14.pkl` legacy

File legacy:

```text
/home/bao/Projects/EncoderModel/peG14.pkl
```

Nó chứa:

```text
382,299 vectors tổng
204,978 vectors K01–K20
177,321 vectors L21–L30
model metadata: PE-Core-G14-448
dimension: 1280
dtype: float32
```

Collection mới không phải bản chuyển đổi từng hàng của pickle cũ. Pickle dùng ID
ordinal kiểu:

```text
Keyframes_L25/keyframes/L25_V001/375.jpg
```

Collection mới dùng frame index thật:

```text
L25_V001@f00038907
```

Do đó không join mù bằng ordinal.

### 8.1 Audit compatibility đã thực hiện

Đã quét toàn bộ norm của 382.299 vector cũ và 1.339.055 vector mới:

| Metric       |      Legacy |         New |
| ------------ | ----------: | ----------: |
| Mean L2 norm | 1.000000006 | 1.000000006 |
| Min L2 norm  | 0.999999855 | 0.999999835 |
| Max L2 norm  | 1.000000149 | 1.000000153 |
| Bad shape    |           0 |           0 |
| NaN/Inf      |           0 |           0 |

Phân bố coordinate trên L21–L30:

```text
centroid cosine:                 0.991342
per-dimension mean correlation: 0.991354
per-dimension std correlation:  0.972546
```

Paired-video audit lấy 240 old vectors trải trên 30 video, tìm nearest vector trong
cùng video ở collection mới:

```text
mean nearest cosine:   0.979895
median:                0.985619
p95:                   0.999997
max:                   0.999999881

>= 0.999:  22 / 240
>= 0.990:  79 / 240
>= 0.950: 224 / 240
>= 0.900: 235 / 240
```

Random cross-video control chỉ có mean cosine `0.509950`, max `0.769332`.

Ví dụ gần như exact:

```text
legacy: L21_V016/001.jpg
new:    L21_V016@f00000000
cosine: 0.999999881
```

Kết luận: hai bộ embedding nằm trong cùng PE-G14 coordinate space và tương thích
retrieval. Phần không đạt cosine gần 1 chủ yếu do hai pipeline chọn keyframe khác
nhau, không phải do embedding dimension/model space bị lệch.

### 8.2 Chính sách migration

- Dùng collection mới làm source of truth cho L21–L30.
- Không concatenate L21–L30 legacy vào index mới vì sẽ trùng video và dùng hệ ID
  khác.
- K01–K20 legacy không có counterpart trong collection mới. Nếu cần giữ, migrate
  riêng sang schema canonical và audit mapping/frame metadata trước khi merge index.
- Pickle có thể thực thi code khi load. Chỉ load file trusted hoặc dùng restricted
  unpickler; Parquet là format production được ưu tiên.

## 9. Multi-encoder integration

PE-G14 nên có index riêng. Nếu kết hợp với SigLIP2 hoặc Qwen3-VL:

- không concatenate raw vectors khác dimension nếu không có projection được train;
- retrieve candidate độc lập theo từng encoder;
- normalize/calibrate score trên validation set;
- fusion bằng weighted score, rank fusion hoặc reranker;
- lưu score riêng (`pe_score`, `siglip_score`, `qwen_score`) để debug;
- luôn giữ `frame_id` làm join key giữa các encoder.

Một baseline an toàn:

```text
PE top 200
∪ SigLIP top 200
∪ Qwen top 200
→ join bằng frame_id
→ rank fusion / learned weighting
→ temporal dedup
→ top K cuối
```

Không giả định score cosine của các model khác nhau có cùng calibration.

## 10. Secret và vận hành

- Không ghi HF token vào notebook public, log, manifest hoặc tài liệu này.
- `HF_TOKEN.txt` chỉ là credential local; không commit vào Git.
- Nếu token từng xuất hiện trong chat/log không kiểm soát, rotate token trên Hugging
  Face.
- Retrieval chỉ cần read permission với bucket; encode/upload cần write permission.
- Master remote prefix là immutable về mặt quy ước. Nếu đổi semantic config, tạo
  prefix version mới thay vì ghi đè `v1`.

## 11. Checklist cho AI/kỹ sư tiếp theo

Trước khi tích hợp:

- [ ] Đọc `_SUCCESS.json` và xác nhận `complete=true`.
- [ ] Xác nhận đủ `1.339.055` rows.
- [ ] Xác nhận semantic fingerprint chính xác.
- [ ] Dùng `PE-Core-G14-448`, không dùng L14.
- [ ] Checkout upstream SHA đã pin.
- [ ] Query output là `float32`, shape `[B, 1280]`, L2 norm ≈ 1.
- [ ] Dùng cosine hoặc inner product.
- [ ] Dùng `frame_id` làm canonical join key.
- [ ] Có temporal dedup cho kết quả cùng video.
- [ ] Không trộn vector có semantic fingerprint khác.

## 12. Source of truth ưu tiên

Khi có khác biệt giữa tài liệu và artifact, ưu tiên theo thứ tự:

1. `pe-core-g14-448-v1/_SUCCESS.json`;
2. `pe-core-g14-448-v1/embedding_dataset_manifest.json`;
3. `pe-core-g14-448-v1/run_config.json`;
4. commit JSON của từng shard;
5. notebook production `PE_Core_G14_448_AIC2026_Colab.ipynb`;
6. tài liệu handoff này;
7. notebook legacy `perceptiong14.ipynb`.

Không dùng notebook legacy làm nguồn cấu hình model nếu nó mâu thuẫn với semantic
fingerprint/manifest production.
