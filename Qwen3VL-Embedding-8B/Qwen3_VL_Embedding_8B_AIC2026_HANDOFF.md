# Qwen3-VL-Embedding-8B AIC 2026 — handoff

Cập nhật: **2026-08-28** (Asia/Ho_Chi_Minh)

## 1. Trạng thái cuối

Collection `qwen3-vl-embedding-8b-4096-v3` đã **hoàn tất toàn bộ**:

```text
expected rows : 1.339.055
committed rows: 1.339.055
Parquet shards: 658/658
categories    : L21..L30 complete
global        : complete=true
```

Global audit hoàn tất tại:

```text
2026-08-27T20:06:02.874403+00:00
```

`embedding_dataset_manifest.json` và `_SUCCESS.json` đã được ghi trên Hugging Face
Storage Bucket. Không còn session/category nào cần encode tiếp. Không ghi thêm vector
vào prefix v3 trừ khi có chủ đích sửa/rebuild toàn collection với một version mới.

Notebook production cuối cùng:

- `Qwen3_VL_Embedding_8B_AIC2026_Colab_2026_01_FA2.ipynb`
- Mirror để review/diff: `Qwen3_VL_Embedding_8B_AIC2026_Colab_2026_01_FA2.py`

Tên file còn `2026_01` chỉ vì lịch sử. Nội dung hiện tại đã được chuyển sang contract
**Colab Latest ngày 2026-08-27**, không còn target runtime 2026.01.

Các artifact có tên `..._SDPA.*` đã bị xóa khỏi workspace và không phải production
path. Notebook cũ `Qwen3_VL_Embedding_8B_AIC2026_Colab.ipynb` chỉ giữ làm reference;
không dùng nó để ghi thêm vào v3.

SHA-256 của production artifact hiện tại:

```text
4662af8b86730d9791e5665e96379663e85e4e6e87dd1f83bf555e638b8964d4  Qwen3_VL_Embedding_8B_AIC2026_Colab_2026_01_FA2.ipynb
2a303f4a90e10ff335ac524912e3ceef35244195eb689993a46603eaf7681f93  Qwen3_VL_Embedding_8B_AIC2026_Colab_2026_01_FA2.py
```

## 2. Source, remote output và local copy

Source:

```text
hf://buckets/Baonenha1/DATA-AIC-Keyframe/infoshootpp-v1/
```

Output production hoàn chỉnh:

```text
hf://buckets/Baonenha1/DATA-AIC-Keyframe/derived/qwen3-vl-embedding-8b-4096-v3/
```

Local copy đã tải về workspace theo layout giống `pe-core-g14-448-v1`:

```text
/home/bao/Projects/EncoderModel/qwen3-vl-embedding-8b-4096-v3/
```

Layout:

```text
qwen3-vl-embedding-8b-4096-v3/
├── run_config.json
├── embeddings/Lxx/part-*.parquet
├── commits/Lxx/part-*.json
├── success/Lxx.json
├── executions/*.json
├── audits/*.json
├── embedding_dataset_manifest.json
└── _SUCCESS.json
```

Inventory remote/local đã đối chiếu ngày 2026-08-28:

```text
remote files       : 1.392
local files        : 1.392
remote bytes       : 22.075.685.318
local bytes        : 22.075.685.318
missing local      : 0
extra local        : 0
size mismatch      : 0
Parquet files      : 658
commit JSON        : 658
category success   : 10
execution JSON     : 59
audit JSON         : 4
all JSON files     : 734
```

Lệnh sync có thể chạy lại an toàn khi cần phục hồi local copy:

```bash
HF_TOKEN="$(tr -d '\r\n' < HF_TOKEN.txt)" \
.venv-embedding-audit/bin/hf buckets sync \
"hf://buckets/Baonenha1/DATA-AIC-Keyframe/derived/qwen3-vl-embedding-8b-4096-v3" \
"./qwen3-vl-embedding-8b-4096-v3" \
--verbose
```

Không print nội dung `HF_TOKEN.txt` vào terminal log, notebook output hoặc handoff.

## 3. Completion theo category và session

| Category | Rows | Shards | Owner cuối | Trạng thái |
|---|---:|---:|---|---|
| L21 | 111.031 | 55 | session_1 | complete |
| L22 | 130.134 | 64 | session_1 | complete |
| L23 | 40.808 | 20 | session_1 | complete |
| L24 | 69.826 | 35 | session_1 | complete |
| L25 | 154.211 | 76 | session_3 | complete |
| L26 | 559.060 | 273 | session_2 | complete |
| L27 | 33.201 | 17 | session_3 | complete |
| L28 | 90.042 | 44 | session_3 | complete |
| L29 | 83.697 | 41 | session_3 | complete |
| L30 | 67.045 | 33 | session_3 | complete |
| **Tổng** | **1.339.055** | **658** | 3 sessions | **complete** |

Parallel plan authoritative trong remote `run_config.json`:

```python
"session_1": ["L21", "L22", "L23", "L24"]
"session_2": ["L26"]
"session_3": ["L25", "L27", "L28", "L29", "L30"]
```

Các mốc cuối đã quan sát:

- Session 1 hoàn tất L21–L24.
- Session 3 hoàn tất L25, L27–L30; tại thời điểm đó global có 929.499 rows.
- Session 2 hoàn tất phần còn lại của L26 và tạo global `_SUCCESS.json`.

Transaction boundary vẫn là commit JSON:

1. Ghi Parquet local.
2. Upload Parquet và verify size.
3. Upload commit JSON sau cùng.
4. Chỉ shard có commit hợp lệ mới được xem là hoàn tất/resume-skip.

## 4. Semantic embedding contract đã khóa

Semantic fingerprint:

```text
8047495b9ffdd325610b7ea8ecefb15389564632a7e2e20c23ae6995e1454ac3
```

Model và implementation:

```text
model_id                      : Qwen/Qwen3-VL-Embedding-8B
model revision                : 2c4565515e0f265c6511776e7193b22c0968ddc7
Qwen implementation git SHA   : 393e2978d27852b0d0230d6994f37f9c15bed73c
Qwen implementation SHA-256   : 0e447b6a80e05527f3fd437ab4f1617a6e7d0df102b8164348bbce787a41b267
checkpoint index SHA-256       : f8656aa4a0a666568f6f6befe980a2ba02d9702eb909bd4c211d6f5db1f980ea
checkpoint tensor payload      : 16.289.587.680 bytes
checkpoint weight files        : 16.289.679.624 bytes
safetensors container overhead : 91.944 bytes
transformers                   : 5.14.1
qwen-vl-utils                  : 0.0.14
embedding dimension            : 4096
MRL dimension                  : None
```

Image/document side:

```text
system: Represent the user's input.
user  : <IMAGE>
```

Preprocessing:

```text
official qwen-vl-utils process_vision_info
official chat template + add_generation_prompt=True
preserve aspect ratio
patch size: 16
smart resize factor: 32
min_pixels: 4.096
max_pixels: 1.843.200
no center crop
processor do_resize=False sau process_vision_info
color: RGB via qwen_vl_utils.to_rgb
image mean/std: [0.5, 0.5, 0.5]
spatial merge size: 2
temporal patch size: 2
max_length: 8192
right padding, truncation=True
pixel values pre-cast BF16 sau official preprocessing
```

Inference/storage:

```text
architecture: full multimodal Qwen3VLForEmbedding
weights/compute: BF16
attention: flash_attention_2
torch.inference_mode()
không torch.autocast()
pooling: last attended token
cast pooled vector sang FP32
L2-normalize trong FP32
storage: native 4096-d FP32
Parquet compression: none
shard_rows: 2.048
```

Parquet schema đã xác minh trên toàn bộ 658 shard:

```text
frame_id: string
video_id: string
category: string
frame_idx: int64
pts_time: float64
fps: float32
image_relpath: string
embedding: fixed_size_list<float32>[4096]
```

Không đổi image-side instruction rồi trộn vector vào v3. Nếu thay model revision,
preprocessing, pooling, dimension hoặc normalization, phải dùng output prefix mới.

## 5. Colab Latest và FlashAttention-2

Contract runtime cuối được notebook kiểm tra fail-fast:

```text
target_colab_runtime : latest
Python expected      : 3.13
Python observed      : 3.13.15
PyTorch expected     : 2.11
PyTorch observed     : 2.11.0+cu128
torch CUDA           : 12.8
CXX11 ABI            : True
GPU production       : NVIDIA A100-SXM4-40GB (sm80)
flash-attn runtime   : 2.8.3
Pillow               : 11.3.0
attention            : flash_attention_2
```

`Latest (recommended)` trong UI Colab không đồng nghĩa với past runtime `2026.07`.
Lỗi cũ xuất hiện vì notebook từng so `Latest` với contract cố định `2026.07`/Python
3.12. Contract hiện tại đã sửa đúng sang `latest` + Python 3.13 + Torch 2.11.

Upstream chưa có wheel phù hợp toàn bộ contract cp313/Torch 2.11/CUDA 12.8 tại lúc
chạy, nên notebook build source chính thức `flash-attn==2.8.3`:

```text
FLASH_ATTENTION_FORCE_BUILD=TRUE
FLASH_ATTN_CUDA_ARCHS=80
MAX_JOBS<=4
NVCC_THREADS=1
```

Build dùng Ninja với 73 compile step; `[1/73]` phải chạy đến compile/link/wheel hoàn
tất. Dấu hiệu thành công cuối:

```text
[FA2 BUILD DONE] return_code=0
[FA2 INSTALL DONE] return_code=0
FA2 final status: 2.8.3
```

Setup cell hiện stream stdout/stderr chi tiết và heartbeat mỗi 15 giây, gồm PID,
elapsed, thời gian từ output cuối, CPU/RSS của process tree, số child process, system
CPU và RAM còn trống. Full log nằm dưới:

```text
/content/flash_attn_build_logs/
```

Các heartbeat từng thấy `cpu_tree=300–400%`, 17–18 child và RAM còn hơn 60 GiB là
compile đang hoạt động bình thường, không phải runtime idle.

## 6. FA2 wheel cache trên Google Drive

Wheel đã build được giữ lại để session khác không compile lại:

```text
Drive root:
/content/drive/MyDrive/AIC2026/flash_attn_wheels/

Contract directory:
flash-attn-2.8.3_cp313_torch-2.11_cu128_abiTRUE_sm80_9cc7cb1fcf45/

Wheel:
flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl

bytes : 59.660.426
SHA-256: 63bb4b733bb08c90346ff392a98f5f398f1c4ccb7cb6b505be29c355071bdff6
contract SHA-256:
9cc7cb1fcf45a8ed6d711592e7da375505a258780ac4cdf6ee339441ec7c5e97
```

Cache contract chứa exact Python/tag/SOABI, Torch, torch CUDA, CXX11 ABI, Linux
architecture, compute capability và FA arch. Restore chỉ được chấp nhận sau khi
manifest, byte count và SHA-256 đều khớp.

Log save đã xác nhận:

```text
[FA2 LOCAL CACHE HIT]
[FA2 DRIVE CACHE SAVE]
fa2_drive_cache: SAVED_VERIFIED
```

Session mới cùng contract sẽ có dạng:

```text
[FA2 DRIVE CACHE COPY]
[FA2 DRIVE CACHE HIT_VERIFIED]
FA2 wheel origin: drive_cache
[FA2 INSTALL DONE] return_code=0
```

Nếu Drive lỗi hoặc cache sai contract/hash, notebook cảnh báo rồi fallback build source.
Manifest được ghi sau wheel như commit marker. Yêu cầu đổi cache sang HF Bucket từng
được nêu nhưng đã hủy; code production hiện vẫn dùng Google Drive.

## 7. Resume state, autotune và quality gates

Cell production đã port logic canonical từ
`Qwen3_VL_Embedding_8B_AIC2026_Colab.ipynb` sang notebook Latest:

- Resume dựa trên commit JSON hợp lệ, không chỉ nhìn Parquet.
- Remote `parallel_plan` là source of truth nếu local plan khác.
- Batch autotune preprocess trực tiếp danh sách N image path cho từng candidate.
- Đã bỏ helper nhân tensor synthetic của bản trung gian.
- Official-wrapper quality gate và numeric batch-vs-batch1 gate khớp notebook gốc.
- Giữ thêm legacy cross-runtime gate vì v3 chứa vector từ hai runtime.

Thứ tự gate:

1. Official preprocess parity.
2. Batch benchmark/VRAM validation.
3. Production forward so với official wrapper.
4. Numeric batch-vs-batch1 audit.
5. Legacy cross-runtime audit so với vector v3 đã commit.

Config production cuối:

```python
"batch_size": 32
"batch_candidates": None
"num_workers": 4
"prefetch_factor": 1
"persistent_workers": False
"max_vram_fraction": 0.94
"fail_on_non_a100": True
```

A100 40 GB chạy batch 32. Không khuyến nghị batch 64 trên A100 40 GB vì benchmark
worst-case từng dùng khoảng 36,3 GiB reserved VRAM.

## 8. Numeric quality gates đã đạt

Trong 59 execution JSON có 57 production startup thực sự chạy quality gate. Hai
execution còn lại có audit rỗng vì category của session đã hoàn tất nên notebook
cố ý skip autotune/gate; đây không phải failed production write.

Kết quả thấp nhất trên các execution có gate:

```text
official-wrapper gate:
  min của production_vs_official_wrapper_min_cosine  = 0.9999985099
  min của production_vs_official_wrapper_mean_cosine = 0.9999986291
  required min cosine                                 = 0.999

numeric batch-vs-batch1 gate:
  min của production_vs_batch1_min_cosine  = 0.9991896152
  min của production_vs_batch1_mean_cosine = 0.9996383190
  required min/mean                         = 0.999 / 0.9995
  all pass                                  = true
```

Latest runtime Torch 2.11/FA2 2.8.3 còn chạy cross-runtime audit với shard legacy:

```text
sample_count       : 16
legacy FA2         : 2.8.3.post1
runtime FA2        : 2.8.3
runtime Torch      : 2.11.0+cu128
min cosine         : 0.9996109605
mean cosine        : 0.9997297525
required min/mean  : 0.999 / 0.9995
result             : PASS
```

Collection v3 giữ FA2 `2.8.3.post1` trong immutable semantic config vì đó là baseline
tạo fingerprint. Actual runtime luôn nằm trong execution/commit provenance.

## 9. Audit toàn bộ local collection ngày 2026-08-28

Không chỉ tin `_SUCCESS.json`: toàn bộ 20,56 GiB collection đã được đọc và kiểm tra.

Các check đã chạy:

- Parse toàn bộ 734 JSON.
- So inventory path/size của từng file với HF Bucket.
- Kiểm tra commit/shard index liên tục và row range không gap/overlap.
- Tính SHA-256 của toàn bộ 658 Parquet và so với từng commit JSON.
- Kiểm tra Parquet byte count, schema, schema metadata và row count.
- Đọc toàn bộ 1.339.055 vector FP32 4096-d.
- Kiểm tra first/last frame ID của từng shard so với commit.
- Kiểm tra `frame_id == video_id@fXXXXXXXX`, category và video prefix.
- Kiểm tra duplicate, null, NaN/Inf, `pts_time`, `fps` và L2 norm.

Kết quả:

```text
audit result                 : PASS
Parquet SHA-256 mismatch     : 0
schema mismatch              : 0
row/range mismatch           : 0
actual rows                  : 1.339.055
unique frame_id              : 1.339.055
duplicate frame_id           : 0
bad identity rows            : 0
non-finite embedding values  : 0
invalid numeric metadata     : 0
L2 norm min                  : 0.9999998211860657
L2 norm mean                 : 0.9999999911577315
L2 norm max                  : 1.0000001192092896
```

Manifest hashes local:

```text
f3b3ce02f3ccfb3f40660afb80dd2d3a98abfa1a1b377992ace0f788ff17905e  _SUCCESS.json
f3b3ce02f3ccfb3f40660afb80dd2d3a98abfa1a1b377992ace0f788ff17905e  embedding_dataset_manifest.json
bfdd29e1586aed353fd4c9795053e19066f3b896342027dc0af8028da4868911  run_config.json
```

`_SUCCESS.json` và `embedding_dataset_manifest.json` giống hệt nhau.

## 10. Runtime provenance của committed rows

658 commit ghi rõ hai execution stack:

| Runtime | Rows | Compute | Attention | GPU |
|---|---:|---|---|---|
| Torch 2.8.0+cu126 | 925.403 | BF16 | FA2 | A100 40 GB |
| Torch 2.11.0+cu128 | 413.652 | BF16 | FA2 | A100 40 GB |
| **Tổng** | **1.339.055** |  |  |  |

Đây là provenance có chủ đích của một collection resume qua runtime mới. Không nên
nói mọi shard được tạo bởi cùng binary stack. Tuy nhiên official-wrapper, numeric,
cross-runtime gate đều đạt và toàn bộ vector đã qua audit norm/integrity.

## 11. Retrieval contract

Image vectors đã khóa bằng instruction generic:

```text
Represent the user's input.
```

Query baseline:

```text
system: Retrieve images or text relevant to the user's query.
user: <text query>
```

Query phải dùng:

```text
model: Qwen/Qwen3-VL-Embedding-8B
revision: 2c4565515e0f265c6511776e7193b22c0968ddc7
official Qwen3VLProcessor chat template
add_generation_prompt=True
last attended-token pooling
FP32 L2 normalization
dimension=4096
```

Index gợi ý:

```text
Milvus FLOAT_VECTOR dimension=4096
metric=COSINE
```

Có thể A/B test query instruction mà không encode lại ảnh. Không được thay image-side
instruction rồi mix vector. Khi ensemble với PE-Core-G14-448 hoặc SigLIP2, không cộng
raw cosine trực tiếp vì score distribution khác nhau; dùng rank fusion/RRF hoặc học
calibration trên validation set.

## 12. Những vấn đề đã xử lý

- Xóa production path SDPA gây nhầm lẫn.
- Sửa notebook từ runtime cố định `2026.07` sang Colab `latest` đúng Python 3.13.
- Build FA2 source có stdout/stderr và heartbeat chi tiết, không còn tình trạng không
  biết compile hay idle.
- Lưu wheel đã build lên Google Drive, verify manifest/size/SHA trước khi reuse.
- Không build lại nếu FA2 đã import đúng hoặc có wheel cache hợp lệ.
- Port resume state, batch autotune và official-wrapper/numeric gate canonical.
- Giữ legacy cross-runtime audit trước khi Torch 2.11 ghi thêm vào v3.
- Bỏ general PyTorch SDPA toggles; model bắt buộc
  `attention_implementation="flash_attention_2"`.
- Không dùng `torch.autocast`; model đã load BF16, pooling xong mới FP32 normalize.
- Pin Pillow 11.3.0 để tránh lỗi `cannot import name '_Ink'`.
- Remote plan hợp lệ là source of truth; prefix có object nhưng thiếu `run_config.json`
  vẫn fail-closed.
- DataLoader production dùng workers=4, prefetch=1, persistent workers off.
- Global manifest chỉ được ghi khi đủ commit cho L21–L30.

## 13. Quy tắc bảo toàn collection hoàn chỉnh

- Coi `_SUCCESS.json`, `embedding_dataset_manifest.json`, commit JSON và Parquet hash
  là source of truth, không dựa vào screenshot/progress bar.
- Không xóa hoặc overwrite object trong prefix v3.
- Không chạy lại notebook với semantic config khác trên cùng prefix.
- Nếu cần sửa embedding contract, tạo prefix/version mới.
- Giữ `run_config.json` và semantic fingerprint khi xây index/query service.
- Backup local có thể phục hồi bằng `hf buckets sync`; chạy audit lại sau khi copy giữa
  disk/máy khác.
- Không commit hoặc log `HF_TOKEN.txt`.

## 14. Validation production artifact local

Tại thời điểm cập nhật handoff:

```text
notebook cells       : 23
code cells           : 11
saved outputs        : 0
mirror py_compile    : OK
notebook JSON        : OK
concatenated code    : compile OK
ipynb/Python mirror  : synchronized
```

Lệnh kiểm tra cơ bản:

```bash
python3 -m py_compile Qwen3_VL_Embedding_8B_AIC2026_Colab_2026_01_FA2.py
jq empty Qwen3_VL_Embedding_8B_AIC2026_Colab_2026_01_FA2.ipynb
```

Collection hiện đã hoàn chỉnh và local audit PASS. Bước tiếp theo thuộc retrieval/index
pipeline, không còn là embedding production/resume.
