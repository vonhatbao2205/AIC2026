# Handoff — TARA (Tarsier2-7B) clip vector cho video batch 2 (M, N, S01)

Tài liệu bàn giao hai collection vector `TARA` (video-clip embedding, 3584-d) của **data batch 2** cho AI/kỹ sư
tích hợp retrieval (index, search backend, frontend). Đọc kèm:

- [`PE_Core_G14_448_Batch2_AIC2026_Retrieval_Handoff.md`](PE_Core_G14_448_Batch2_AIC2026_Retrieval_Handoff.md):
  vector keyframe PE-Core của cùng 614 video. Fusion với TARA ở §7.4.
- [`AIC2026_Batch2_M_N_Keyframe_R2_HANDOFF.md`](AIC2026_Batch2_M_N_Keyframe_R2_HANDOFF.md): layout R2, URL video,
  lưu ý dữ liệu batch 2.

> Snapshot ngày **2026-09-25**, commit cuối được lấy: `S01-V007/part-00003` lúc 14:33:58 UTC. Mọi số liệu dưới
> đây lấy từ artifact đã tải về và audit độc lập (§6), không lấy từ log notebook.

## 1. Trạng thái chốt

| | M + N (tin tức + camera giao thông) | S01 (đua xe đạp) |
|---|---|---|
| Clip vector | **283.454** (M 128.945 + N 154.509), đủ 100 % | **45.756 / 60.562** (75,6 %), **snapshot một phần** |
| Video | 304 M + 298 N (148,6 giờ) | 12 video (46,6 giờ): 9 đủ, V007 một phần, V010/V011 chưa có (§8.1) |
| Clip scale (hop 50 %) | M 8/24/72 s · N **4/8/16 s** | 8/24/72 s |
| `success/<unit>.json` | 110/110 category | 9/12 video |
| `_SUCCESS.json` | **chưa có** trên HF, dữ liệu vẫn đủ (§8.4) | chưa có (collection chưa xong) |
| Parquet shard | 331 (M 132, N 199) | 50 |
| Dung lượng | 3,81 GiB (787 file) | 0,62 GiB (115 file) |
| HF prefix | `derived/tara-tarsier2-7b-3584-batch2-clip-v1` | `derived/tara-tarsier2-7b-3584-batch2-s01-clip-v1` |
| Bản local | `tara-tarsier2-7b-3584-batch2-clip-v1/` | `tara-tarsier2-7b-3584-batch2-s01-clip-v1/` |

Tổng batch 2 trong snapshot: **329.210 clip vector**, 3584-d, float32, đã L2-normalize. Bucket HF:
`hf://buckets/Baonenha1/aic26-media/`. Bản local nằm trong `/home/bao/Projects/EncoderModel/` và chỉ gồm shard
đã commit.

Hai collection batch 2 cùng **embedding space** với collection L `derived/tara-tarsier2-7b-3584-clip-v1`
(168.536 clip, 873 video L21–L30, `complete=true`), nên có thể nạp chung một index. Tổng L + batch 2 là
**497.746 clip**, và `clip_id` không trùng giữa các bộ.

Notebook encode:

- M/N: [`TARA_Tarsier2_7B_Batch2_R2_AIC2026_Colab.ipynb`](TARA_Tarsier2_7B_Batch2_R2_AIC2026_Colab.ipynb)
  (source [`.py`](TARA_Tarsier2_7B_Batch2_R2_AIC2026_Colab.py)), 4 session `session_1`…`session_4`.
- S01: [`TARA_Tarsier2_7B_S01_R2_AIC2026_Colab.ipynb`](TARA_Tarsier2_7B_S01_R2_AIC2026_Colab.ipynb)
  (source [`.py`](TARA_Tarsier2_7B_S01_R2_AIC2026_Colab.py)), 2 session `s01_a`/`s01_b`, work unit là video.
- L: [`TARA_Tarsier2_7B_Clip_AIC2026_Colab.ipynb`](TARA_Tarsier2_7B_Clip_AIC2026_Colab.ipynb).

## 2. Identity contract

Model, preprocessing, precision và storage giống hệt collection L:

```text
model:                      bpiyush/TARA @ 3e7cb730d86ae10da7eb1c85f17e45ece5a5e353 (base Tarsier2-7b-0115)
checkpoint:                 4 shard safetensors, tensor payload 16.582.751.232 byte,
                            index SHA-256 59a64242446007de3eb6502c7d2ab52cc2721a0037fb920ae716b9218a4ad8af
video prompt:               "<video>\nSummary above video in one word:"
frame:                      8 frame uniform trong cửa sổ (sampler v1, gồm frame đầu/cuối), mỗi frame nhân đôi
                            → 16 image slot; max_pixels 460.800, giữ aspect ratio, không crop
inference:                  BF16 + FlashAttention-2; pooling = hidden state cuối của token cuối; L2 normalize FP32
storage:                    Parquet fixed_size_list<float32>[3584], shard 1.024 rows, không nén
embedding contract SHA-256: 611ef1c9b8dac812723d0bd81bacff65346025a0ed6159e3a8cc7f571618e0b3
```

`embedding contract` là SHA-256 của `semantic_config` **bỏ `source` và `clip_plan`**. Khác PE, ở đây bỏ cả clip
plan vì N dùng cửa sổ ngắn hơn L. Cả L, M/N và S01 đều ra đúng giá trị này; đây là điều kiện để trộn vector vào
cùng index.

`semantic_fingerprint` (hash của **toàn bộ** `semantic_config`) khác nhau theo collection. Dùng nó để kiểm đúng
artifact, không dùng để kiểm tương thích giữa các collection. Lưu ý L và M/N cùng bắt đầu bằng `ef1…`:

| Collection | `semantic_fingerprint` |
|---|---|
| L `tara-tarsier2-7b-3584-clip-v1` | `ef197331649dfb93e7057304f294d8bd1cd98bf22b1902bca48daf6dc4874819` |
| `tara-tarsier2-7b-3584-batch2-clip-v1` | `ef1af4ae8bbcd96fba72d78842ea11a93d837cf981e485a619a3e461723b70ee` |
| `tara-tarsier2-7b-3584-batch2-s01-clip-v1` | `e99c2efaf4395c4e850d6659413906f87c31dc62f5925804b5d6b42fe9643c0a` |

Không ghi vector từ model/revision/prompt/`n_frames`/`max_pixels` khác vào các prefix này. Đổi scale hoặc đổi
video nguồn thì encode vào prefix mới (`...-v2`), không ghi đè `v1`.

## 3. Nguồn video và clip plan

Video lấy từ R2 `aic26-media`, key `Videos/Videos_<category>/<video_id>.mp4`. Manifest 614 video được pin ngày
2026-09-23 (size, ETag, `duration_sec` của luồng video, `fps`, `nb_frames`), nhúng trong notebook section 4,
SHA-256 `ea094415db6184987ddd2070a5980663f5716b527dd42e42eb1f316a27733e45`.

11 video S01 gốc là AV1, `decord` không đọc được. Notebook chỉ nhận bản H.264 do
[`AIC2026_S01_AV1_to_H264_R2_Colab.ipynb`](AIC2026_S01_AV1_to_H264_R2_Colab.ipynb) tạo (đúng từng frame, metadata
`source-etag` khớp bản AV1 đã pin). S01-V012 vốn là H.264. Tại snapshot, **V010 và V011 vẫn là AV1** nên chưa được
encode.

Clip plan chỉ suy ra từ `duration_sec` trong manifest nên deterministic. Với mỗi scale (window `W`, hop `H`) và
video dài `D`:

```text
D <= W  →  1 clip [0, D]
D >  W  →  start_i = min(i*H, D-W), i = 0..ceil((D-W)/H);  end_i = start_i + W   (clip cuối kết thúc đúng tại D)
```

Thứ tự row: `video (natural order) → scale (event, sequence, scene) → start_time`.

| Nhóm | `event` | `sequence` | `scene` |
|---|---|---|---|
| M, S01 (giống L) | 8 s, hop 4 s | 24 s, hop 12 s | 72 s, hop 36 s |
| N (camera giao thông) | **4 s**, hop 2 s | **8 s**, hop 4 s | **16 s**, hop 8 s |

Tên scale giống L nhưng **độ dài cửa sổ của N khác**. Cần độ dài thật thì dùng `end_time - start_time` hoặc Arrow
metadata `scales` của file.

Số clip theo scale trong snapshot:

| | event | sequence | scene |
|---|---:|---:|---:|
| M | 89.437 | 29.707 | 9.801 |
| N | 88.367 | 44.130 | 22.012 |
| S01 | 32.942 | 9.612 | 3.202 |

### 3.1 M/N theo category

| Category | Clip | | Category | Clip |
|---|---:|---|---|---:|
| M01 | 11.186 | | M06 | 12.333 |
| M02 | 11.623 | | M07 | 13.667 |
| M03 | 11.023 | | M08 | 14.581 |
| M04 | 11.537 | | M09 | 14.992 |
| M05 | 12.329 | | M10 | 15.674 |
| N001–N100 | 154.509 (100 category, 532–2.605 clip/category) | | | |

### 3.2 S01 theo video

| Video | Session | Clip có / kế hoạch | Shard | Trạng thái |
|---|---|---:|---:|---|
| S01-V001 | `s01_a` | 3.584 / 3.584 | 4 | đủ |
| S01-V002 | `s01_a` | 3.666 / 3.666 | 4 | đủ |
| S01-V003 | `s01_b` | 3.164 / 3.164 | 4 | đủ |
| S01-V004 | `s01_b` | 5.118 / 5.118 | 5 | đủ |
| S01-V005 | `s01_b` | 5.123 / 5.123 | 6 | đủ |
| S01-V006 | `s01_b` | 3.511 / 3.511 | 4 | đủ |
| S01-V007 | `s01_b` | **4.096 / 7.310** | 4/8 | **một phần**: chỉ `event`, 0 → 16.388 s (video dài 20.246 s) |
| S01-V008 | `s01_a` | 6.241 / 6.241 | 7 | đủ |
| S01-V009 | `s01_a` | 4.829 / 4.829 | 5 | đủ |
| S01-V010 | `s01_a` | **0 / 5.512** | 0/6 | **không có**, video vẫn AV1 trên R2 |
| S01-V011 | `s01_b` | **0 / 6.080** | 0/6 | **không có**, video vẫn AV1 trên R2 |
| S01-V012 | `s01_a` | 6.424 / 6.424 | 7 | đủ |

## 4. Lần chạy

| | M + N | S01 |
|---|---|---|
| Runtime | 4 × A100-SXM4-80GB song song (`session_1`…`session_4`) | `s01_a`: A100-SXM4-80GB; `s01_b`: A100-SXM4-40GB (lần resume 14:49 UTC chuyển sang 80GB, chưa có shard trong snapshot) |
| Thời gian | 2026-09-23 19:03 → 2026-09-25 13:43 UTC, nhiều lần resume (8 execution) | 2026-09-24 20:09 UTC → snapshot (5 execution) |
| Batch / workers | 8 / 1 | 8 / 1 (80GB), 2 / 1 (40GB) |
| Throughput | median 0,60 clip/s mỗi runtime (0,45–0,65) | median 0,46 clip/s (0,31–0,56) |

**Quality gate trên Colab** chạy mỗi lần khởi động, trước khi ghi vector. Mỗi lần so 6 clip thật giữa đường
production (reader cache, bỏ `lm_head`, batch) với đường chính chủ `TARA` (`sample_video` + `generate`). Ngưỡng:
min cosine ≥ 0,999, mean ≥ 0,9995, batch so với single ≥ 0,999, và `frame_indices` phải khớp tuyệt đối. Cả 13
execution đều pass, min cosine ≥ 0,9999998 và không lệch frame index nào.

**Self-test** (notebook M/N section 11, chạy ở `session_1`–`session_3`):

- *Known-answer* với `assets/folding_paper.mp4`: folding / cutting / unfolding = 0,646 / 0,393 / 0,300. Số chính
  chủ trong README TARA là 0,6488 / 0,3952 / 0,3009, lệch tối đa 0,0028.
- *Arrow-of-time* (đảo thứ tự 8 frame): median `1 − cos` = 0,149 (M03_V006), 0,178 (M08_V004), 0,052
  (N043-V004). Vector có nhạy với chiều thời gian.
- *Temporal pairs giao thông*: **chưa chạy** vì bảng `TEMPORAL_PAIRS` để trống.

## 5. Cấu trúc artifact

```text
tara-tarsier2-7b-3584-batch2-clip-v1/          tara-tarsier2-7b-3584-batch2-s01-clip-v1/
├── run_config.json                            ├── run_config.json    # + embedding_contract_sha256, parallel_plan
├── executions/colab_session_<n>-<ts>.json     ├── executions/colab_s01_<a|b>-<ts>.json
├── audits/colab_session_<n>-<ts>.json         │
├── self_tests/colab_session_<n>-<ts>.json     │
├── embeddings/{M01..M10,N001..N100}/          ├── embeddings/{S01-V001..S01-V009,S01-V012}/  # + V007 (4 shard)
│   └── part-00000.parquet …                   │   └── part-00000.parquet …
├── commits/<category>/part-*.json             ├── commits/<video_id>/part-*.json
└── success/<category>.json   (110)            └── success/<video_id>.json   (9)
    # chưa có embedding_dataset_manifest.json / _SUCCESS.json (§8.4)
```

Thư mục con dưới `embeddings/` là **work unit**: category với M/N, video với S01. Đừng suy category từ tên thư
mục; đọc cột `category`, cột này luôn là category thật (`M01`, `N001`, `S01`). Loader nên glob
`embeddings/*/*.parquet`.

### 5.1 Parquet schema

Giống hệt collection L:

| Cột | Kiểu | Ví dụ / ý nghĩa |
|---|---|---|
| `clip_id` | string | `N001-V001@event@t000012000`, khoá chính, `{video_id}@{scale}@t{start_ms:09d}` |
| `video_id` | string | `M01_V001`, `N001-V001`, `S01-V001` (cùng convention với PE) |
| `category` | string | `M01`, `N001`, `S01` |
| `scale` | string | `event` \| `sequence` \| `scene` |
| `scale_index` | int8 | 0 \| 1 \| 2 |
| `start_time`, `end_time` | float64 | giây, **identity của plan** |
| `fps` | float32 | `avg_frame_rate` trong manifest |
| `duration_sec` | float64 | duration luồng video trong manifest |
| `video_relpath` | string | **key R2** `aic26-media`, ví dụ `Videos/Videos_N001/N001-V001.mp4` |
| `frame_indices` | fixed_size_list<int32>[8] | frame index thật model đã nhìn thấy |
| `embedding` | fixed_size_list<float32>[3584] | unit vector |

Arrow metadata của mỗi file: `schema_version`, `semantic_fingerprint`, `model_id`, `model_revision`,
`checkpoint_index_sha256`, `video_eol_prompt`, `n_frames`, `max_pixels`, `pooling`, `embedding_dim`,
`embedding_dtype`, `l2_normalized`, `scales_by_group`, `scales`, `source_r2_bucket`, `video_manifest_sha256`.

`start_time`/`end_time` xác định row, còn `frame_indices` là **ground truth** của cửa sổ thật:
`frame_indices[0]/fps .. frame_indices[-1]/fps`. Hai cách tính lệch nhau tối đa 0,12 s trong snapshot. Clip cuối
của video bị lùi về `D - W` nên `start_ms` có thể lẻ (`…@t000612009`). N039 có 3 clip cuối bị kéo lùi vì decoder
đọc ít frame hơn manifest (`clamped_clips_this_run` trong `success/N039.json`). Khi cần định vị thời điểm (KIS,
TRAKE), dùng `frame_indices`.

Resolve media giống PE batch 2:

```text
video: https://video.baoencoder.site/<video_relpath>?v=<ETag hiện tại>, seek tới frame_indices[0]/fps
```

`video_etags` trong commit/success ghi ETag của đúng bytes đã encode. Với S01 đó là bản H.264. Thêm `?v=<ETag>` để
tránh bản CDN cache cũ.

### 5.2 Commit và success

Mỗi shard có một commit JSON gồm `row_start`/`row_stop`/`rows`, `first_clip_id`/`last_clip_id`, `parquet_path`,
`parquet_bytes`, `parquet_sha256`, norm min/max/mean, `video_etags` và `execution` (GPU, batch, session,
version). Protocol: upload Parquet → verify size → ghi commit sau cùng. **Parquet không có commit không được tính**
(bản local chỉ tải shard đã commit).

`success/<unit>.json` được ghi khi một work unit đủ shard. File này gồm `rows`, `shards`, `videos`,
`clamped_clips_this_run`, `frame_substitutions_this_run`, `videos_eof_trimmed_this_run` và `video_etags`.

## 6. Đã kiểm chứng (2026-09-25)

Audit chạy độc lập trên bản local, đối chiếu với manifest video đã pin (trích từ notebook, SHA-256 khớp
`run_config.json`). **Cả hai collection đều 0 lỗi.**

- Tính lại `semantic_fingerprint` từ `run_config.json`: khớp mọi commit và Arrow metadata của mọi Parquet.
  Embedding contract tính lại ra `611ef1c9…` cho cả L, M/N và S01.
- 381 Parquet có SHA-256 và số byte khớp commit. Shard id liên tục từ 0, row range đúng. M/N đủ shard ở 110/110
  category.
- **Từng dòng** khớp clip plan tính lại từ manifest: `clip_id`, `video_id`, `category`, `scale`, `scale_index`,
  `start_time`, `end_time` đúng giá trị và đúng thứ tự; `fps`, `duration_sec`, `video_relpath` khớp manifest.
- `clip_id` duy nhất trên 329.210 dòng. Prefix `video_id` (`M…`, `N…`, `S01…`) không giao với L (`L…`).
- `frame_indices`: đủ 8 giá trị, không giảm, nằm trong `[0, nb_frames)`. Frame đầu/cuối cách `start_time`/
  `end_time` tối đa 0,116 s (M/N) và 0,034 s (S01).
- Vector: float32, 3584-d, không NaN/Inf. Sai số norm tối đa 1,4e-7.
- Tính liên tục theo thời gian. Cosine giữa hai clip liên tiếp cùng video, cùng scale:

  | | median | p5 | p1 |
  |---|---:|---:|---:|
  | M | 0,788 | 0,465 | 0,359 |
  | N | 0,985 | 0,930 | 0,870 |
  | S01 | 0,955 | 0,730 | 0,576 |

  N rất cao vì camera cố định: các clip của cùng một camera gần nhau hơn nhiều so với tin tức (§7.3).
- Không có dấu hiệu vector sụp về một hướng. Số thành phần chính giữ 90 % phương sai (5.000 vector `event`/bộ):
  M 504, N 86, S01 125, L 326. N thấp vì chỉ có 298 góc camera cố định.
- **Cùng không gian với L.** Median cosine tới láng giềng gần nhất (scale `event`, 2.500 vector/bộ, mẫu L lấy từ 20
  shard):

  | Truy vấn từ | → L | → M | → N | → S01 |
  |---|---:|---:|---:|---:|
  | M (tin tức) | **0,646** | (0,753) | 0,291 | 0,505 |
  | N | **0,776** | 0,732 | (0,975) | 0,656 |
  | S01 | **0,869** | 0,791 | 0,520 | (0,949) |

  Số trong ngoặc là trong cùng bộ, đã bỏ chính nó. Tin tức M gần L hơn nhiều so với N (0,646 so với 0,291), nghĩa
  là khoảng cách phản ánh nội dung chứ không phản ánh collection.

**Chưa làm:** encode lại vài clip bằng chính model để so vector (cần checkpoint 16,6 GB và GPU); quality gate trên
Colab (§4) đã so trực tiếp với đường chính chủ. Chưa đo chất lượng text→video trên query AIC, và chưa đo temporal
pairs cho camera giao thông.

## 7. Tích hợp retrieval

### 7.1 Query và metric

Query dùng **text side của chính TARA** (`TARA.encode_text`) rồi L2-normalize FP32. `encode_text` tự bọc query
vào EOL prompt `<sent>\nSummary above sentence in one word:`, nên không thêm instruction nào khác. Metric là
**COSINE**, hoặc inner product trên unit vector. TARA fine-tune trên dữ liệu tiếng Anh: **dịch query VI → EN
trước khi encode**.

Môi trường giống notebook section 1: Python 3.10, `torch==2.5.1+cu121`, `transformers==4.45.0`,
`flash-attn==2.8.3`, `numpy==1.26.4`, `pillow==10.0.0`. Weights BF16 khoảng 16,6 GB, nên cần GPU ≥ 24 GB.

```python
import sys

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

model_dir = snapshot_download("bpiyush/TARA", revision="3e7cb730d86ae10da7eb1c85f17e45ece5a5e353")
sys.path.insert(0, model_dir)
from modeling_tara import TARA  # noqa: E402

model = TARA.from_pretrained(
    model_dir, device_map={"": 0}, attn_implementation="flash_attention_2", low_cpu_mem_usage=True
)
model.model.eval()


@torch.inference_mode()
def encode_query(texts):
    z = model.encode_text(list(texts)).float()
    return F.normalize(z, p=2, dim=-1).cpu().numpy()  # float32 [n, 3584]
```

`modeling_tara` import `shared.utils`, gói này kéo theo matplotlib/cv2/sklearn/seaborn. Nếu import lỗi, cài các
package đó hoặc dùng stub `ensure_shared_utils` trong worker của notebook (section 7).

### 7.2 Gộp vào index

- Nạp L, M/N và S01 vào **cùng một index** (ví dụ Milvus `FLOAT_VECTOR(3584)` + `COSINE`) với `clip_id` làm
  primary key.
- Nên thêm các trường scalar `category`, `video_id`, `scale`, `start_time`, `end_time`, `frame_indices` (hoặc frame
  đầu/cuối) và một trường nguồn, ví dụ `source_set ∈ {L, M, N, S01}`.
- RAM cho vector thô: 14 KB/clip, tức 4,4 GiB cho batch 2 và 2,25 GiB cho L. Index HNSW tốn thêm.
- Bỏ 1.591 clip đuôi đen của M10_V029 (§8.2). Loader dưới đây đã lọc sẵn.
- Đừng nạp collection nào không qua được check khi khởi động dưới đây. M/N chưa có `_SUCCESS.json` (§8.4), nên
  check này dựa trên commit.

```python
import hashlib
import json
from pathlib import Path

BASE = Path("/home/bao/Projects/EncoderModel")
EMBEDDING_CONTRACT = "611ef1c9b8dac812723d0bd81bacff65346025a0ed6159e3a8cc7f571618e0b3"
SNAPSHOT = {  # prefix: (rows, semantic_fingerprint) — snapshot 2026-09-25
    "tara-tarsier2-7b-3584-batch2-clip-v1": (283_454, "ef1af4ae8bbcd96fba72d78842ea11a93d837cf981e485a619a3e461723b70ee"),
    "tara-tarsier2-7b-3584-batch2-s01-clip-v1": (45_756, "e99c2efaf4395c4e850d6659413906f87c31dc62f5925804b5d6b42fe9643c0a"),
}


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def embedding_contract(semantic_config):
    body = {k: v for k, v in semantic_config.items() if k not in ("source", "clip_plan")}
    return hashlib.sha256(canonical(body)).hexdigest()


for name, (rows, fingerprint) in SNAPSHOT.items():
    root = BASE / name
    run_config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(canonical(run_config["semantic_config"])).hexdigest() == fingerprint, name
    assert run_config["semantic_fingerprint"] == fingerprint, name
    assert embedding_contract(run_config["semantic_config"]) == EMBEDDING_CONTRACT, name
    commits = [json.loads(p.read_text(encoding="utf-8")) for p in root.glob("commits/*/*.json")]
    assert all(c["semantic_fingerprint"] == fingerprint for c in commits), name
    assert sum(c["rows"] for c in commits) == rows, name
    for c in commits:
        parquet = root / c["parquet_path"].split("/", 2)[2]
        assert parquet.stat().st_size == c["parquet_bytes"], parquet
    assert len(list(root.glob("embeddings/*/*.parquet"))) == len(commits), name
```

Với L, check `_SUCCESS.json` của `derived/tara-tarsier2-7b-3584-clip-v1`: `complete=true`, 168.536 rows,
fingerprint `ef197331…`.

Đọc shard batch 2 để build index:

```python
import pyarrow.compute as pc
import pyarrow.parquet as pq

DIM = 3584
M10_BLACK_TAIL_START = 1112.0  # §8.2


def iter_tara(name):
    for path in sorted((BASE / name / "embeddings").glob("*/*.parquet")):
        table = pq.read_table(path)
        vectors = table.column("embedding").combine_chunks().values.to_numpy().reshape(-1, DIM)
        meta = table.drop_columns(["embedding"])
        keep = pc.invert(pc.and_(
            pc.equal(meta.column("video_id"), "M10_V029"),
            pc.greater_equal(meta.column("start_time"), M10_BLACK_TAIL_START),
        ))
        yield meta.filter(keep), vectors[keep.to_numpy(zero_copy_only=False)]  # float32 unit vectors
```

Chạy trên snapshot này, loader trả về 327.619 clip (329.210 − 1.591).

### 7.3 Chọn scale và dedup

- Cửa sổ chồng 50 % và có 3 scale, nên **một khoảnh khắc nằm trong khoảng 6 clip** (2 clip mỗi scale). Top-K thô
  sẽ đầy các cửa sổ chồng nhau. Temporal NMS/grouping theo `(video_id, [start_time, end_time])` là **bắt buộc**.
  Nên lấy 300 candidate trở lên trước khi dedup.
- Hành động ngắn, một chuyển động (N: "a car turns left") thì ưu tiên `event`. Tương tác/maneuver ("a motorcycle
  overtakes a car") thì dùng `sequence`, `scene` làm ngữ cảnh. Muốn đơn giản thì search mọi scale rồi NMS.
- N: hai clip liên tiếp có median cosine 0,985, và láng giềng trong N là 0,975. Kết quả N vì thế dễ bị một camera
  chiếm hết top-K. Nên group theo `video_id` và dùng cửa sổ NMS rộng hơn M.
- Định vị trả lời: dùng `frame_indices` (§5.1). Với KIS có thể lấy frame giữa cửa sổ; với TRAKE lấy mốc từ
  `frame_indices` của clip `event` tốt nhất.

### 7.4 Kết hợp encoder khác

TARA là vector **clip**, còn PE/Qwen là vector **keyframe** (`frame_id`). `video_id` dùng chung convention.
Join theo thời gian: keyframe PE thuộc clip TARA nếu cùng `video_id` và `pts_time ∈ [start_time, end_time]`. Fuse
bằng rank/RRF ở mức đoạn thời gian, **không cộng cosine giữa các model** (score distribution khác nhau).
S01-V010, S01-V011 và phần cuối S01-V007 chỉ có PE (§8.1).

## 8. Vấn đề đã biết

### 8.1 S01 chỉ là snapshot một phần

Collection S01 chưa xong. Còn thiếu 14.806 clip:

- **S01-V010 và S01-V011: không có vector TARA nào** (8,9 giờ video). Hai video này vẫn là AV1 trên R2 vì notebook
  transcode dừng sau V009.
- **S01-V007: chỉ có 4.096 clip `event` đầu, từ 0 đến 16.388 s.** Video không có clip `sequence`/`scene` nào,
  và khoảng 1,07 giờ cuối không có TARA.

Backend không được giả định mọi video đều đủ 3 scale. Query nhắm vào các đoạn này phải dựa vào PE/Qwen. Khi có
shard mới, `clip_id` và shard đã commit không đổi, nên chỉ cần upsert dòng mới (§9).

### 8.2 M10_V029: 1.591 vector đuôi đen trùng byte

M10_V029 dài 5.523 s, nhưng từ khoảng 1.112 s trở đi video chỉ còn màn đen (handoff PE cũng ghi nhận 1.035
keyframe đuôi đen của video này). Các clip có `start_time ≥ 1112` gồm 1.102 `event`, 367 `sequence` và 122
`scene`. Tất cả **1.591 clip này có vector giống hệt từng byte**. Một query bất kỳ khớp "màn đen" sẽ trả về
hàng trăm kết quả vô nghĩa, nên **bỏ khỏi index** (loader §7.2 đã lọc).

### 8.3 Vector trùng khác (vô hại)

- **N, 120 cặp:** clip cuối của video bị lùi về `D - W` nên trùng frame với clip liền trước, ví dụ
  `N003-V001@sequence@t000612000` và `…@t000612009`. `frame_indices` giống nhau thì vector giống nhau. NMS (§7.3)
  tự loại.
- **S01, 166 nhóm (508 dòng dư):** màn chờ phát lặp chu kỳ 100 s ở đầu V002, V004, V006, V008 và V012. Nhiều nhất
  là V006, 245 dòng, trong khoảng 20–904 s. Có thể giữ một clip canonical cho mỗi nhóm giống cách PE §8.1.

### 8.4 M/N chưa có `_SUCCESS.json`

Dữ liệu M/N **đã đủ**: 110/110 `success/*.json`, 283.454 dòng, đã audit ở §6. Audit cuối trên HF (`session_2`,
2026-09-25 13:44 UTC) lại chạy bằng **bản notebook cũ**, bản này còn tính S01 vào collection M/N. Vì vậy các file
`audits/*.json` ghi `expected_total_rows=344.016`, S01 0 dòng, `complete=false`; bỏ qua các file này.

Cách sửa: Run All bản notebook hiện tại với một session bất kỳ. Mọi shard sẽ được skip, và section 10 ghi
`_SUCCESS.json` + `embedding_dataset_manifest.json`. Việc này không chặn retrieval, vì check khởi động §7.2 dựa
trên commit.

## 9. Chạy lại hoặc bổ sung

- **Hoàn tất S01:** chạy `AIC2026_S01_AV1_to_H264_R2_Colab.ipynb` cho V010, V011. Sau đó Run All notebook S01 với
  `s01_a` (V010) và `s01_b` (phần còn lại của V007 và V011). Tổng 14.806 clip, tức khoảng 7 GPU-giờ ở 0,6 clip/s.
  Chạy song song thì nhánh `s01_b` (9.294 clip) quyết định, mất khoảng 4,5–5 giờ. Notebook resume theo shard;
  shard đã commit được skip.
- **Đồng bộ lại bản local:** tải commit mới, rồi chỉ tải Parquet mà commit trỏ tới và verify `parquet_sha256`.
  Cập nhật `SNAPSHOT` ở §7.2 theo tổng `rows` mới.

  ```python
  from huggingface_hub import download_bucket_files, list_bucket_tree

  BUCKET, PREFIX = "Baonenha1/aic26-media", "derived/tara-tarsier2-7b-3584-batch2-s01-clip-v1"
  local = BASE / PREFIX.split("/")[-1]
  items = {i.path: i for i in list_bucket_tree(BUCKET, prefix=PREFIX, recursive=True, token=HF_TOKEN)
           if getattr(i, "type", "file") == "file"}
  download_bucket_files(BUCKET, token=HF_TOKEN, files=[
      (i, str(local / p[len(PREFIX) + 1:])) for p, i in items.items() if p.endswith(".json")])
  commits = [json.loads(p.read_text()) for p in local.glob("commits/*/*.json")]
  download_bucket_files(BUCKET, token=HF_TOKEN, files=[
      (items[c["parquet_path"]], str(local / c["parquet_path"][len(PREFIX) + 1:])) for c in commits
      if not (local / c["parquet_path"][len(PREFIX) + 1:]).exists()])
  ```

- Đổi model, prompt, `n_frames`, `max_pixels` hoặc scale là tạo collection **mới** (`...-v2`), không ghi vào `v1`.

## 10. Checklist tích hợp

- [ ] Check khi khởi động (§7.2) pass cho M/N và S01; `_SUCCESS.json` của L `complete=true`.
- [ ] Đủ 168.536 (L) + 283.454 (M/N) + 45.756 (S01) clip, trừ 1.591 clip đuôi đen M10_V029; `clip_id` là primary
      key.
- [ ] Query: dịch VI → EN, `TARA.encode_text`, L2-normalize FP32; metric COSINE.
- [ ] Temporal NMS/grouping theo `(video_id, thời gian)` trên cả 3 scale; cửa sổ riêng cho N.
- [ ] Định vị kết quả bằng `frame_indices`, không chỉ `start_time`/`end_time`.
- [ ] S01-V010, S01-V011 và phần cuối S01-V007 fallback sang PE/Qwen (§8.1).
- [ ] Fusion với PE/Qwen bằng RRF theo đoạn thời gian, không cộng cosine.
- [ ] Không trộn vector có embedding contract khác `611ef1c9…`.

## 11. Source of truth

Khi tài liệu và artifact khác nhau, ưu tiên theo thứ tự:

1. commit JSON + Parquet của từng shard (M/N chưa có `_SUCCESS.json`; L dùng `_SUCCESS.json`);
2. `run_config.json` (semantic config, `expected_clip_counts`, `parallel_plan`);
3. `success/<unit>.json`;
4. manifest video đã pin (notebook section 4, SHA-256 `ea094415…`);
5. notebook `TARA_Tarsier2_7B_Batch2_R2_AIC2026_Colab.ipynb` và `TARA_Tarsier2_7B_S01_R2_AIC2026_Colab.ipynb`;
6. tài liệu này.
