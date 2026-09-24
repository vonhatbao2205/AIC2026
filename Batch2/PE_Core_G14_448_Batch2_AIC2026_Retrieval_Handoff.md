# Handoff — PE-Core-G14-448 vector cho keyframe batch 2 (M, N, S01)

Tài liệu bàn giao hai collection vector `PE-Core-G14-448` của **data batch 2** cho AI/kỹ sư tích hợp
retrieval (index, search backend, frontend). Đọc kèm:

- [`PE_Core_G14_448_AIC2026_Retrieval_Handoff.md`](PE_Core_G14_448_AIC2026_Retrieval_Handoff.md): handoff của
  collection L. Query encoder, numeric policy và cách fusion ở đó áp dụng nguyên cho batch 2.
- [`AIC2026_Batch2_M_N_Keyframe_R2_HANDOFF.md`](AIC2026_Batch2_M_N_Keyframe_R2_HANDOFF.md): nguồn keyframe batch 2
  (registry, layout R2, lưu ý dữ liệu).

> Snapshot ngày **2026-09-24**. Mọi số liệu dưới đây lấy từ artifact đã tải về và audit độc lập (§6), không lấy
> từ log notebook.

## 1. Trạng thái chốt

| | M + N (tin tức + camera giao thông) | S01 (đua xe đạp) |
|---|---|---|
| Vector | **252.352** (M 207.780 + N 44.572) | **620.976** |
| Video | 304 M + 298 N | 12 |
| Bỏ frame `quality < 0,05` | 1.331 (M), 0 (N) | 141 |
| `_SUCCESS.json` | `complete=true`, 2026-09-23 19:28 UTC | `complete=true`, 2026-09-24 09:14 UTC |
| Parquet shard | 130 (M 3/category, N 1/category) | 82 |
| Dung lượng | 1,23 GiB (374 file) | 3,02 GiB (183 file) |
| HF prefix | `derived/pe-core-g14-448-batch2-mn-v1` | `derived/pe-core-g14-448-batch2-s01-v1` |
| Bản local | `pe-core-g14-448-batch2-mn-v1/` | `pe-core-g14-448-batch2-s01-v1/` |

Tổng batch 2: **873.328 vector**, 1280-d, float32, đã L2-normalize. Bucket HF:
`hf://buckets/Baonenha1/DATA-AIC-Keyframe/`. Bản local nằm trong `/home/bao/Projects/EncoderModel/`.

Hai collection batch 2 cùng **embedding space** với collection L `derived/pe-core-g14-448-v1` (1.339.055 vector),
nên có thể nạp chung một index. Tổng L + batch 2 là **2.212.383 vector**, và `frame_id` không trùng giữa các bộ.

Notebook encode: [`PE_Core_G14_448_Batch2_MN_R2_AIC2026_Colab.ipynb`](PE_Core_G14_448_Batch2_MN_R2_AIC2026_Colab.ipynb)
(source [`.py`](PE_Core_G14_448_Batch2_MN_R2_AIC2026_Colab.py)). Một notebook phục vụ cả hai collection qua
`CONFIG["source_set"]` = `"MN"` hoặc `"S01"`.

## 2. Identity contract

Model, preprocessing, precision và storage giống hệt collection L:

```text
model:                     PE-Core-G14-448 (vision tower)
implementation git SHA:    3e352cca660658d4b5c90f42a7808b11469e4c66 (facebookresearch/perception_models)
checkpoint SHA-256:        516a20fb30394cc8080344c32d7e49e0e575cf2c6d141977207bf94b3bec192f
preprocess:                Pillow → RGB → official bilinear squash 448×448 → mean/std 0.5, không crop
inference:                 FP32 weights + autocast FP16, eager (không torch.compile), L2 normalize FP32
storage:                   Parquet fixed_size_list<float32>[1280], shard 8.192 rows, không nén
embedding contract SHA-256: d940db4aa194deeea4daeb05c80a5a90e3e3b498e28d06afed74894aa4dd1d09
```

`embedding contract` là SHA-256 của `semantic_config` **bỏ phần `source`**. Cả L, M/N và S01 đều ra đúng giá trị
này; đây là điều kiện để trộn vector vào cùng index.

`semantic_fingerprint` (hash của **toàn bộ** `semantic_config`, gồm cả nguồn keyframe) thì khác nhau theo
collection. Dùng nó để kiểm đúng artifact, không dùng để kiểm tương thích giữa các collection:

| Collection | `semantic_fingerprint` |
|---|---|
| L `pe-core-g14-448-v1` | `2cce817755d48b2a59c50a2a345d5c5a42f59fc46b4b6bd1ce3668eca1b03141` |
| `pe-core-g14-448-batch2-mn-v1` | `c0386a1e2f26cb02941dadcf928b335dc5d073b7e18ff7410033a85ac0ed37e1` |
| `pe-core-g14-448-batch2-s01-v1` | `7c4d7a3ec7b5f657b08cc5bf26391c9ba4fd1c6fb02a840fbdf31b72605d3fea` |

Không ghi vector từ model/checkpoint/preprocess khác vào các prefix này. Nếu nguồn keyframe đổi (chạy lại
pipeline keyframe → signature mới), encode vào prefix mới (`...-v2`), không ghi đè `v1`.

## 3. Nguồn keyframe và phạm vi

Work queue là `frame_registry.parquet` của từng profile trên R2 `aic26-infoshot-keyframes`, được pin bằng
SHA-256. Mỗi vector ứng với đúng một JPEG trong registry, trừ các frame `quality < 0,05` (frame đen/fade).

| Profile | Metadata prefix | Pipeline signature | Registry rows | Registry SHA-256 |
|---|---|---|---:|---|
| M | `manifest/infoshoot_m` | `e6c789322adb` | 209.111 | `7b34dc36…f460ae` |
| N | `manifest/infoshoot_n` | `773e4b7465c2` | 44.572 | `523b1a11…6b89fa` |
| S | `manifest/infoshoot_s` | `db5918a110cf` | 621.117 | `b3d7c630…024a54` |

Frame bị loại: 1.331 frame M (gồm 1.035 frame đuôi đen của `M10_V029`, `n ≥ 676`) và 141 frame S01. N không có
frame nào bị loại. Muốn biết frame nào bị loại, lọc registry bằng `quality < 0.05`.

### 3.1 M/N theo category

| Category | Vector | | Category | Vector |
|---|---:|---|---|---:|
| M01 | 17.987 | | M06 | 20.170 |
| M02 | 18.821 | | M07 | 22.507 |
| M03 | 18.175 | | M08 | 23.824 |
| M04 | 19.233 | | M09 | 24.028 |
| M05 | 19.995 | | M10 | 23.040 |
| N001–N100 | 44.572 (100 category, 103–663 vector/category) | | | |

### 3.2 S01 theo video

| Video | Vector | Shard | | Video | Vector | Shard |
|---|---:|---:|---|---|---:|---:|
| S01-V001 | 36.728 | 5 | | S01-V007 | 75.099 | 10 |
| S01-V002 | 38.621 | 5 | | S01-V008 | 65.731 | 9 |
| S01-V003 | 29.740 | 4 | | S01-V009 | 45.631 | 6 |
| S01-V004 | 51.454 | 7 | | S01-V010 | 56.934 | 7 |
| S01-V005 | 55.274 | 7 | | S01-V011 | 62.472 | 8 |
| S01-V006 | 34.621 | 5 | | S01-V012 | 68.671 | 9 |

## 4. Lần chạy

| | M + N | S01 |
|---|---|---|
| Runtime | 1 × A100-SXM4-80GB | 2 × A100-SXM4-80GB song song (`s01_a`, `s01_b`, mỗi runtime 6 video) |
| Thời gian | 2026-09-23 17:03 → 19:28 UTC | 2026-09-24 06:20 → 09:14 UTC |
| Batch / workers | 64 / 11 | 64 / 11 |
| Throughput | ~29 ảnh/s | ~30 ảnh/s mỗi runtime |
| Numeric audit FP16 vs FP32 (64 ảnh) | min 0,99960, mean 0,99994 | min 0,99923 / 0,99981, mean 0,99994 / 0,99996 |

Ngưỡng audit là min ≥ 0,999 và mean ≥ 0,9999. Mọi runtime đều đạt.

## 5. Cấu trúc artifact

```text
pe-core-g14-448-batch2-mn-v1/            pe-core-g14-448-batch2-s01-v1/
├── run_config.json                      ├── run_config.json          # + parallel_plan s01_a / s01_b
├── executions/<timestamp>.json          ├── executions/colab_<session>-<timestamp>.json
│                                        ├── audits/colab_<session>-<timestamp>.json
├── embeddings/{M01..M10,N001..N100}/    ├── embeddings/{S01-V001..S01-V012}/
│   └── part-00000.parquet …             │   └── part-00000.parquet …
├── commits/<category>/part-*.json       ├── commits/<video_id>/part-*.json
├── success/<category>.json              ├── success/<video_id>.json
├── embedding_dataset_manifest.json      ├── embedding_dataset_manifest.json
└── _SUCCESS.json                        └── _SUCCESS.json
```

Thư mục con dưới `embeddings/` là **work unit**: category với M/N, video với S01 (một video S01 có 30–75
nghìn keyframe). Đừng suy category từ tên thư mục; đọc cột `category`, cột này luôn là category thật (`M01`,
`N001`, `S01`). Loader nên glob `embeddings/*/*.parquet`.

### 5.1 Parquet schema

Giống hệt collection L:

| Cột | Kiểu | Ví dụ |
|---|---|---|
| `frame_id` | string | `S01-V001@f00000002` — khoá chính, `{video_id}@f{frame_idx:08d}` |
| `video_id` | string | `M01_V001`, `N001-V001`, `S01-V001` |
| `category` | string | `M01`, `N001`, `S01` |
| `frame_idx` | int64 | chỉ số frame theo thứ tự decode trong video gốc |
| `pts_time` | float64 | giây, dùng để seek video |
| `fps` | float32 | FPS trung bình |
| `image_relpath` | string | **key R2** của JPEG, ví dụ `Keyframes/Keyframes_S01/S01-V001/001.jpg` |
| `embedding` | fixed_size_list<float32>[1280] | unit vector |

Arrow metadata của mỗi file: `schema_version`, `semantic_fingerprint`, `model_name`, `checkpoint_sha256`,
`embedding_dim`, `embedding_dtype`, `l2_normalized`, `source_r2_bucket` (`aic26-infoshot-keyframes`) và
`source_pipeline_signature` (signature snapshot keyframe của profile). Parquet vector không có cột
`pipeline_signature` (giữ schema giống L); nếu index cần trường này, lấy từ Arrow metadata hoặc bảng §3.

**Khác L ở `image_relpath`:**

| | Ví dụ | Nơi chứa ảnh | Tên file |
|---|---|---|---|
| L | `infoshootpp/keyframes/L21/L21_V001/f00000000.jpg` | corpus HF `infoshootpp-v1` | theo `frame_idx` |
| Batch 2 | `Keyframes/Keyframes_M01/M01_V001/001.jpg` | R2 `aic26-infoshot-keyframes` | theo thứ tự `n` |

Backend phải resolve ảnh theo collection. Với batch 2:

```text
ảnh:   https://pub-5010c807b73743ba82d6c40852ae9e6b.r2.dev/<image_relpath>
video: https://video.baoencoder.site/Videos/Videos_<category>/<video_id>.mp4?v=<ETag hiện tại>, seek tới pts_time
```

Thêm `?v=<ETag>` để tránh bản CDN cache cũ (xem handoff keyframe §2).

### 5.2 Commit và success

Mỗi shard có một commit JSON, gồm row range, `first_frame_id`/`last_frame_id`, `parquet_sha256`, `parquet_bytes`,
norm min/max/mean và `source_pipeline_signature`. Commit S01 có thêm `work_unit` (video) và `session_name`.
Protocol giống L: Parquet được upload và verify trước, commit ghi sau cùng.

Trong S01, `audits/colab_s01_b-….json` ghi `complete=false` (617.841 rows). Điều này là **bình thường**: runtime
`s01_b` xong trước khi `s01_a` encode xong S01-V012. Nguồn sự thật là `_SUCCESS.json`.

## 6. Đã kiểm chứng (2026-09-24)

Audit chạy độc lập trên bản local, so với bản sao registry keyframe đã audit
(`~/Projects/ExtractKeyframe/keyframe_batch2/infoshoot_{m,n,s}/final/`, SHA-256 khớp bảng §3).
**Cả hai collection đều 0 lỗi.**

- Tính lại `semantic_fingerprint` từ `run_config.json`: khớp `_SUCCESS.json`. Embedding contract tính lại khớp L.
- Mọi Parquet có SHA-256 và số byte khớp commit. Shard id liên tục, row range đúng.
- **Từng dòng** khớp registry: đúng tập frame (registry trừ `quality < 0,05`), đúng thứ tự, và `frame_idx`,
  `pts_time`, `fps`, `image_relpath` = `r2_key`, `category`, `video_id` đều khớp.
- S01: tính lại được `selected_image_manifest_sha256` (danh sách `frame_id` + key + size + MD5 JPEG).
- Vector: float32, 1280-d, không NaN/Inf. Sai số norm tối đa 1,6e-7 (S01) và 1,5e-7 (M/N).
- `frame_id` duy nhất trên toàn bộ L + M/N + S01 (2.212.383 dòng).
- **Vector đúng với ảnh của nó.** Ở S01, 1.200 nhóm vector trùng byte khớp chính xác 1.200 nhóm JPEG trùng
  MD5 (§8.1). Ảnh giống hệt cho vector giống hệt, ảnh khác nhau không bao giờ cho vector trùng. Nếu có lệch
  dòng giữa vector và `frame_id`, điều này không thể xảy ra.
- Tính liên tục theo thời gian. Cosine giữa hai keyframe liên tiếp cùng video:

  | | median | p5 | p1 |
  |---|---:|---:|---:|
  | S01 | 0,983 | 0,948 | 0,832 |
  | M/N | 0,934 | 0,596 | 0,514 |

- Không có dấu hiệu vector sụp về một hướng: cần 190 (S01) và 217 (M/N) thành phần chính mới giữ 90 %
  phương sai.
- **Cùng không gian với L.** Median cosine tới láng giềng gần nhất ở collection khác (mẫu 2.500 vector/bộ):

  | Truy vấn từ | → L | → M | → N | → S01 |
  |---|---:|---:|---:|---:|
  | M (tin tức) | **0,712** | — | 0,581 | 0,655 |
  | S01 | 0,681 | **0,859** | 0,693 | — |
  | N | **0,817** | 0,807 | — | 0,766 |

  Tin tức M gần L (cũng là tin tức) nhất. S01 gần M vì bản tin HTV7 (category M06) phát lại hình ảnh chính giải
  Cúp Truyền hình 2026.
- Kiểm bằng mắt: với 5 frame S01 ngẫu nhiên, láng giềng gần nhất đúng ngữ nghĩa. Cận cảnh tay đua ra cận cảnh
  tay đua (cosine 0,94), flycam cao tốc ra flycam cao tốc (0,97), màn chia đôi đồ hoạ ra màn chia đôi (0,94).
  Láng giềng trong M là các bản tin M06 phát lại chặng đua tương ứng (0,90–0,94).

**Chưa làm:** encode lại vài ảnh bằng chính model để so vector. Việc này cần tải checkpoint khoảng 9,7 GB. Audit
FP16 so với FP32 trên Colab (§4) và các kiểm tra trên đã bao phủ phần lớn rủi ro.

## 7. Tích hợp retrieval

### 7.1 Query và metric

Giống hệt L (handoff L §7): query dùng **text tower PE-Core-G14-448**, official tokenizer, `context_length=72`,
output L2-normalize FP32. Metric là **COSINE**, hoặc inner product trên unit vector. Không thêm instruction prompt
vào query PE. Code encode query nằm ở handoff L §7.1.

### 7.2 Gộp vào index

- Nạp L, M/N và S01 vào **cùng một index** (ví dụ Milvus `FLOAT_VECTOR(1280)` + `COSINE`) với `frame_id` làm
  primary key.
- Nên thêm các trường scalar `category`, `video_id`, `pts_time`, `frame_idx` và một trường nguồn, ví dụ
  `source_set ∈ {L, M, N, S01}`. Trường nguồn dùng để lọc theo nhóm dữ liệu và để resolve ảnh đúng convention
  (§5.1).
- Đừng nạp collection nào không qua được check khi khởi động dưới đây.

```python
import hashlib
import json
from pathlib import Path

BASE = Path("/home/bao/Projects/EncoderModel")
EMBEDDING_CONTRACT = "d940db4aa194deeea4daeb05c80a5a90e3e3b498e28d06afed74894aa4dd1d09"
COLLECTIONS = {  # prefix: (rows, semantic_fingerprint)
    "pe-core-g14-448-v1": (1_339_055, "2cce817755d48b2a59c50a2a345d5c5a42f59fc46b4b6bd1ce3668eca1b03141"),
    "pe-core-g14-448-batch2-mn-v1": (252_352, "c0386a1e2f26cb02941dadcf928b335dc5d073b7e18ff7410033a85ac0ed37e1"),
    "pe-core-g14-448-batch2-s01-v1": (620_976, "7c4d7a3ec7b5f657b08cc5bf26391c9ba4fd1c6fb02a840fbdf31b72605d3fea"),
}


def embedding_contract(semantic_config):
    body = {k: v for k, v in semantic_config.items() if k != "source"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


for name, (rows, fingerprint) in COLLECTIONS.items():
    success = json.loads((BASE / name / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert success["complete"] is True, name
    assert success["total_committed_rows"] == success["expected_total_rows"] == rows, name
    assert success["semantic_fingerprint"] == fingerprint, name
    assert embedding_contract(success["semantic_config"]) == EMBEDDING_CONTRACT, name
    assert success["retrieval_contract"]["dimension"] == 1280, name
```

Đọc shard batch 2 để build index:

```python
import numpy as np
import pyarrow.parquet as pq

R2_PUBLIC = "https://pub-5010c807b73743ba82d6c40852ae9e6b.r2.dev"


def iter_batch2(name):
    for path in sorted((BASE / name / "embeddings").glob("*/*.parquet")):
        table = pq.read_table(path)
        vectors = table.column("embedding").combine_chunks().values.to_numpy().reshape(-1, 1280)
        meta = table.drop(["embedding"]).to_pandas()
        meta["image_url"] = R2_PUBLIC + "/" + meta["image_relpath"]
        yield meta, vectors  # vectors: float32 [rows, 1280], đã L2-normalize
```

### 7.3 Dedup kết quả

S01 dày **3,7 keyframe/giây** (coverage 0,5 s theo InfoShot++ v4.6). Hai keyframe liên tiếp có median cosine 0,983,
nên top-K thô sẽ đầy các frame cách nhau vài trăm mili giây. Temporal NMS/grouping theo `(video_id, pts_time)` là
**bắt buộc**, nhất là với S01 (xem handoff L §7.5). Nên lấy nhiều candidate hơn (300 hoặc hơn) trước khi dedup.
Cửa sổ NMS cần tune riêng: một cửa sổ hợp với M/N (0,58 keyframe/giây) sẽ quá hẹp với S01.

### 7.4 Kết hợp encoder khác

Fusion theo `frame_id` bằng rank/RRF, không cộng cosine giữa các model (handoff L §9). Vector Qwen3-VL cho
batch 2 chưa chạy xong. Khi có, join bằng `frame_id` như trên.

## 8. Vấn đề đã biết

### 8.1 S01 có 9.450 vector trùng byte

Có 10.650 dòng S01, chia thành 1.200 nhóm, mà JPEG giống hệt từng byte. Vì vậy vector cũng giống hệt; bỏ đi
thì 9.450 dòng là dư. Đặc điểm:

- mỗi nhóm nằm trọn trong một video, không có nhóm nào trải qua nhiều video;
- nhóm lớn nhất có 46 frame;
- nguồn chủ yếu là `coverage_floor`;
- nhiều nhất ở S01-V006 (2.302 dòng), sau đó V011, V008, V010. V001 không có.

Nhóm lớn nhất là màn chờ tĩnh "TRỰC TIẾP – Cúp Truyền hình 2026 – Chặng 06" chiếu khoảng 15 phút đầu S01-V006. M/N
không có vector trùng.

Không phải lỗi embedding, nhưng truy vấn khớp màn đó sẽ trả về hàng trăm ảnh y hệt. Khuyến nghị: khi build
index, giữ **một frame canonical cho mỗi nhóm** (frame sớm nhất) để tìm kiếm, và lưu bảng
`frame_id → canonical_frame_id` để UI vẫn mở được timeline:

```python
import hashlib

import pandas as pd

parts = []
for meta, vectors in iter_batch2("pe-core-g14-448-batch2-s01-v1"):
    keys = [hashlib.blake2b(v.tobytes(), digest_size=16).hexdigest() for v in vectors]
    parts.append(meta[["frame_id", "video_id", "pts_time"]].assign(vector_key=keys))
frames = pd.concat(parts, ignore_index=True).sort_values(["video_id", "pts_time"], kind="mergesort")
frames["canonical_frame_id"] = frames.groupby("vector_key")["frame_id"].transform("first")
duplicates = frames[frames.frame_id != frames.canonical_frame_id]  # 9.450 dòng
```

Muốn giảm khối lượng S01 sâu hơn: handoff keyframe §11.6 gợi ý xét 212.980 keyframe chỉ có nguồn
`coverage_floor` (34 %). Làm việc đó ở bước hậu xử lý riêng, không sửa collection `v1`.

### 8.2 Lưu ý thừa hưởng từ dữ liệu keyframe

- `S01-V002`: video trên R2 đã chuyển mã AV1 → H.264 **sau khi** trích keyframe. `frame_idx`/`pts_time` vẫn
  seek đúng (cùng số frame, cùng timeline), chỉ ETag nguồn trong registry khác video hiện tại (handoff keyframe
  §11.5). Nếu các video AV1 còn lại được chuyển mã sau này, quy tắc tương tự áp dụng; vector không bị ảnh hưởng.
- Ảnh M còn băng ticker, ảnh N còn banner camera, vì vector được encode trên **full frame** giống L. Query về chữ
  trên ticker có thể khớp tin khác với hình (handoff keyframe §11.4).
- Frontend chưa có media manifest cho batch 2. Cần nối `media_manifest_{M,N,S}.csv` vào manifest chung (handoff
  keyframe §5.2).

## 9. Chạy lại hoặc encode thêm

- Notebook resume theo shard: chạy lại với cùng `source_set` và `active_session` chỉ encode shard còn thiếu. Với
  hai collection hiện tại (đã complete), chạy lại chỉ kiểm lại rồi bỏ qua.
- `source_set="S01"` với hai session `s01_a`/`s01_b` (mỗi runtime 6 video); `source_set="MN"` với session `mn`.
  Assignment được ghi trong `run_config.json` (`parallel_plan`); notebook từ chối nếu hai session trùng work unit.
- Nếu snapshot keyframe đổi, notebook sẽ dừng vì SHA-256 registry không khớp. Khi đó cập nhật `SOURCE_SETS` và
  **đổi output prefix**; không ghi vào prefix `v1`.

## 10. Checklist tích hợp

- [ ] Check khi khởi động (§7.2) pass cho L, M/N, S01.
- [ ] Đủ 1.339.055 + 252.352 + 620.976 vector; `frame_id` là primary key.
- [ ] Query dùng text tower PE-Core-G14-448, context 72, L2-normalize; metric COSINE.
- [ ] Resolve ảnh theo nguồn: batch 2 là `R2_PUBLIC/<image_relpath>`, L theo convention cũ.
- [ ] Temporal NMS/grouping theo `(video_id, pts_time)`, tune riêng cho S01.
- [ ] Quyết định xử lý 9.450 vector trùng của S01 (§8.1).
- [ ] Không trộn vector có embedding contract khác `d940db4a…`.

## 11. Source of truth

Khi tài liệu và artifact khác nhau, ưu tiên theo thứ tự:

1. `_SUCCESS.json` của từng prefix;
2. `run_config.json` (semantic config, `parallel_plan`);
3. commit JSON của từng shard;
4. registry keyframe `manifest/infoshoot_{m,n,s}/final/frame_registry.parquet` trên R2;
5. notebook `PE_Core_G14_448_Batch2_MN_R2_AIC2026_Colab.ipynb`;
6. tài liệu này.
