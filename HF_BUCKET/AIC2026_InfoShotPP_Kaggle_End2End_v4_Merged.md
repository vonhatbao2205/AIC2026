# AIC 2026 — InfoShot++ v4.6 Kaggle T4×2 end-to-end

Tài liệu này mô tả pipeline được triển khai trong
`AIC2026_InfoShotPP_Kaggle_End2End_v4_Merged.ipynb`. Nguồn chuẩn của tài liệu là
code và `CONFIG` đang thực thi trong notebook hiện tại.

## 1. Mục tiêu và trạng thái hiện tại

Pipeline nhận video AIC, phát hiện các thay đổi ngắn và trạng thái nội dung quan trọng,
chọn keyframe có độ phủ thời gian cao, sau đó ghi JPEG và registry phục vụ retrieval.

Trạng thái production hiện tại:

- Chạy trên Kaggle với hai GPU T4.
- Hỗ trợ `K01..K20` và `L21..L30`.
- Mặc định `run_mode="batch"`.
- Selector là `google/siglip-base-patch16-224`.
- Probe semantic chạy ở 10 FPS.
- Motion Farneback giữ bật ở 5 FPS.
- BTC safety union đang tắt: output chỉ đến từ InfoShot++, microburst và coverage.
- L25 có bước nén static-state riêng cho video giảng bài/slide tĩnh.
- Selector/final embeddings không được lưu mặc định.
- Output được chia theo session, tự archive và dừng trước quota Kaggle.

## 2. Luồng end-to-end

```text
DATA_ROOT
   ↓
Fast discovery không đệ quy + audit ID/category/partition
   ↓
ffprobe metadata: codec, FPS, duration, frame count, resolution
   ↓
PASS 1 — một lượt decode low-resolution
   ├─ native sentinel trên GPU 0
   │  ├─ global grayscale change
   │  ├─ local tile change
   │  ├─ broadcast text-edge change
   │  └─ full-frame localized text-edge change
   ├─ Farneback motion 5 FPS trên CPU
   ├─ quality/pHash/text hash trên CPU
   └─ SigLIP probe 10 FPS trên GPU 0 + GPU 1
          ↓
Native events + GLRT semantic boundaries
          ↓
Content-driven segments
          ↓
COMMON + MULTI-UNIQUE + short-shot candidates
          ↓
Native-FPS microburst refinement quanh từng event
          ↓
Priority-aware temporal/text/semantic dedup
          ↓
Coverage gap-fill: target 0,4 s; hard bound khoảng 0,5 s
          ↓
L25-only verified static-state consolidation
          ↓
Materialize frame chính xác từ video gốc
          ↓
JPEG + registry + map-keyframes + stats + success marker
          ↓
Session archive/quota guard + global registry + manifest
```

## 3. Input, discovery và chuẩn hoá ID

### 3.1 Dataset mặc định

```text
/kaggle/input/datasets/loinguyen57/aic-dataset-2025-s1
```

Layout được hỗ trợ trực tiếp:

```text
DATA_ROOT/
├── Videos_K01/video/K01_V001.mp4
├── Videos_L21_a/video/L21_V001.mp4
├── Videos_L26_d/video/L26_V222.mp4
└── ...
```

Discovery không dùng `os.walk()` hay `rglob()` trên toàn dataset. Nó chỉ:

1. Đọc các folder cấp một khớp `Videos_Kxx...` hoặc `Videos_Lxx...`.
2. Đọc file video trực tiếp trong folder con `video/`.
3. Nếu không có `video/`, đọc file trực tiếp trong `Videos_*`.

Nhờ vậy nó không duyệt qua hàng trăm nghìn JPEG trong `Keyframes_*`.

### 3.2 Exclusion và canonical ID

Hai folder duplicate bị loại theo path component chính xác:

```text
Videos_L25_a1
Videos_L25_b
```

Canonical regex hỗ trợ cả K và L:

```regex
([KL]\d{2})(?:_?([a-z])\d*)?_V(\d+)
```

Ví dụ:

```text
K01_a_V001 → K01_V001
L26_d_V222 → L26_V222
L30_V004   → L30_V004
```

Partition được lấy từ tên folder `Videos_*`; `main` đại diện cho folder không có hậu
tố. Nếu hai file còn lại cùng canonical ID sau exclusion, audit dừng để tránh ghi đè.

### 3.3 Audit

- Full corpus dự kiến có 1.478 video sau exclusion.
- Full-corpus batch bị chặn nếu số video/category không khớp.
- Session chỉ chọn một số category vẫn được chạy nếu các category được yêu cầu tồn tại.
- Merge cuối có thể bắt buộc đủ 1.478 video và toàn bộ category.

## 4. Cấu hình vận hành quan trọng

### 4.1 Chế độ chạy

```python
"run_mode": "batch"  # audit | smoke | batch | merge
```

- `audit`: chỉ discover/audit, không tải model.
- `smoke`: chạy full các video chỉ định và hiển thị diagnostics/gallery.
- `batch`: chạy session extraction có resume và quota guard.
- `merge`: gộp các Kaggle Dataset shard, không cần attach video nguồn/model.

### 4.2 Chọn category và partition

```python
"session_categories": "L21,L22,L23",
"session_partitions": "",
```

Quy tắc:

- CSV string hoặc list đều hợp lệ.
- Khoảng trắng sau dấu phẩy được bỏ tự động.
- Partition rỗng nghĩa là mọi partition.
- `main` nghĩa là folder không hậu tố.
- Partition filter áp dụng chung cho mọi category trong session.

Ví dụ chỉ chạy `Videos_L25_a` và `Videos_L26_a`:

```python
"session_categories": "L25,L26",
"session_partitions": "a",
```

Ví dụ chỉ chạy `Videos_L26_d` và `Videos_L26_e`:

```python
"session_categories": "L26",
"session_partitions": "d,e",
```

### 4.3 Chia nhỏ session

```python
"range_start": 0,
"range_end": None,
"num_shards": 1,
"shard_index": 0,
```

- `range_start/range_end`: cắt một khoảng trong danh sách video đã sort tự nhiên.
- `num_shards/shard_index`: chia round-robin deterministic.
- `session_name`: có thể đặt tên thủ công; nếu `None`, notebook tự sinh.

Ví dụ session ID:

```text
l21-l22-l23
l25-l26__p-a
l26__p-d-e__r0-50
l26__s00of02
```

## 5. Metadata và định danh frame

`ffprobe` đọc:

- codec;
- average/rate FPS;
- frame count;
- duration;
- width/height.

Nếu `ffprobe` lỗi, pipeline fallback sang OpenCV. FPS không hợp lệ làm video dừng thay
vì đoán. `frame_idx` là decode-order index thật; `pts_time` giữ timestamp PTS độc lập.

Các đối tượng chính:

- `VideoMeta`: metadata video.
- `Event`: trigger có vùng thời gian, source, priority và score.
- `FrameCandidate`: frame ứng viên với quality, hash, text state và embedding.

Priority:

- P0: text event, extreme native event, short shot và protected evidence.
- P1: COMMON, UNIQUE và semantic boundary.
- P2: native event thường, motion event thường và coverage floor.

## 6. Encoder đa GPU

Selector mặc định:

```python
"backend": "siglip"
"model_name": "google/siglip-base-patch16-224"
"batch_per_gpu": 128
```

Với hai T4, tổng selector batch mặc định là 256 ảnh. Hai model replica xử lý hai nửa
batch bằng hai worker độc lập; pipeline không dùng `DataParallel`.

SigLIP preprocessing chạy trên GPU:

1. Copy BGR uint8 đã pin sang từng GPU.
2. Chuyển BGR sang RGB.
3. Scale, bicubic resize và normalize.
4. Chạy FP16 inference.
5. L2-normalize embedding.

Nếu CUDA OOM, batch/GPU tự giảm dần tới `min_batch_per_gpu=16`. ResNet-50 và PE vẫn có
implementation cho ablation, nhưng extraction production hiện dùng SigLIP.

## 7. Pass 1: fused native sentinel + semantic probe

### 7.1 Decode và scheduling

- PyAV/software decode chạy trên CPU.
- Frame được resize về `sentinel_width=320`.
- Native batch mặc định: 768 frame.
- Có hai batch prefetch.
- Native CUDA preprocessing chạy trên GPU 0.
- SigLIP chia đều cho GPU 0 và GPU 1.
- Farneback, hash và một số quality metrics vẫn chạy CPU, overlap với GPU.

GPU thường chạy theo burst. Sau Pass 1, các stage CPU/decode/JPEG có thể làm GPU hiển
thị idle; điều đó không có nghĩa GPU bị bỏ qua.

### 7.2 Native signals

Mỗi decoded frame tạo các raw signal:

- `global_raw`: mean absolute grayscale difference toàn frame.
- `local_raw`: mean của 4 tile thay đổi mạnh nhất trên grid 8×8.
- `text_raw`: edge-map change tại top 20% và bottom 40% của frame.
- `text_local_raw`: mean của 3 tile edge-change mạnh nhất trên grid 12×20 toàn frame.
- `motion_raw`: median Farneback optical-flow magnitude ở 5 FPS.

Text edge sử dụng Sobel, ngưỡng percentile 75 với floor 24 và morphological close theo
chiều ngang. Full-frame text signal bảo vệ chữ nhỏ giữa ảnh, công thức, handwriting,
speedometer, bảng điểm và highlight.

### 7.3 Probe semantic 10 FPS

Tại `probe_fps=10`, pipeline lưu:

- normalized SigLIP embedding;
- quality và sharpness;
- pHash 64-bit;
- text hash 64-bit;
- text density;
- `frame_idx` và `pts_time`.

Probe/sentinel cache được lưu dưới dạng NPZ float16 khi cần. Fingerprint gồm metadata
video, model và mọi cấu hình làm thay đổi Pass 1, trong đó có `sentinel_top_tiles`.

## 8. Native event detection

Mỗi raw signal được chuẩn hoá bằng rolling median/MAD với cửa sổ khoảng 2 giây:

```text
z = (x - rolling_median) / robust_scale
```

Một event chỉ được nhận khi đồng thời vượt z-threshold và absolute raw floor. Sau đó
temporal peak NMS loại các peak quá gần nhau.

Threshold production:

| Signal | z threshold | raw floor |
|---|---:|---:|
| global | 3.5 | 0.030 |
| local | 3.5 | 0.055 |
| broadcast text | 3.0 | 0.025 |
| localized text | 3.0 | 0.060 |
| motion | 3.5 | 0.65 |

Text event hoặc event có `z >= 6` nhận P0. Event thường nhận P2.

## 9. GLRT segmentation và InfoShot candidates

### 9.1 Semantic boundary

InfoShot GLRT chạy trên embedding probe bằng cumulative feature sums, không tạo affinity
matrix `T×T`. Các window từ 3 đến 15 probe frame được đánh giá; GPU 0 thực hiện phép
tính khi CUDA khả dụng.

Boundary được lấy từ:

- GLRT local peak;
- native-global event rất mạnh;
- TransNetV2 nếu được bật.

Boundary luôn phải nằm bên trong timeline và để lại ít nhất `min_probe_frames=3` ở hai
phía. Defensive spacing loại boundary 0, boundary cuối, duplicate và boundary quá sát
mép. Segment dài hơn 300 probe frame, xấp xỉ 30 giây, bị bắt buộc chia tiếp tại boundary
mạnh nhất phù hợp.

TransNetV2 hiện tắt mặc định.

### 9.2 COMMON

Trong mỗi segment, pipeline tính:

- typicality: mức giống với phần còn lại của segment;
- local volatility: mức khác với các probe lân cận;
- quality: độ nét/phơi sáng và penalty clipping/black frame.

COMMON score:

```text
0.7 × typicality - 0.3 × volatility
```

Frame quality quá thấp bị tránh nếu segment còn lựa chọn khác.

### 9.3 MULTI-UNIQUE

UNIQUE score:

```text
0.5 × (1 - typicality) + 0.5 × volatility
```

Segment có ít nhất 3 probe luôn giữ một primary UNIQUE khác COMMON. Các UNIQUE peak bổ
sung phải qua raw score, robust z và temporal NMS.

### 9.4 Short shot

Segment ngắn không quá 800 ms tạo protected short-shot event P0. Very-short shot không
quá 500 ms có thể giữ tối đa 3 frame trong microburst.

## 10. BTC safety union

Code vẫn hỗ trợ đọc `frame_idx/pts_time` từ mapping BTC, nhưng cấu hình hiện tại là:

```python
"btc_union": {
    "enabled": False,
    "require_mapping": False,
    "prefer_btc": True,
}
```

Do đó pipeline hiện tại:

- không index CSV BTC;
- không đưa keyframe có sẵn vào candidate pool;
- không copy JPEG BTC;
- tạo output độc lập từ video nguồn.

## 11. Native-FPS microburst refinement

Sau khi có events, pipeline decode tuần tự video lần hai. Mỗi event mở một cửa sổ:

- event thường: ±350 ms;
- COMMON: ±100 ms;
- semantic boundary: ±250 ms;
- short shot: toàn segment ngắn.

Trong mỗi cửa sổ, background median được tạo từ các frame. Mỗi frame được chấm bằng:

- global/local visual delta;
- global/local text-edge delta;
- text density;
- quality;
- khoảng cách tới event center.

`text_local_event` đặt trọng số lớn nhất vào localized text delta. Mỗi event thường giữ
tối đa 2 frame; frame bổ sung phải đủ score và khác pHash hoặc text hash. Candidate sau
refinement giữ nguyên priority của event gốc, nên motion P2 không bị nâng thành P0.

## 12. Candidate union và safe dedup

Candidate pool gồm:

- COMMON;
- primary/additional UNIQUE;
- native global/local/text/motion triggers;
- semantic boundary;
- short shot;
- microburst refined frames;
- BTC nếu tùy chọn này được bật.

Candidate cùng `frame_idx` được gộp sources và scores trước. Hai frame chỉ được coi là
duplicate khi đồng thời:

- cùng semantic segment;
- cách nhau không quá 1.000 ms;
- text hash Hamming không quá 4;
- text-density delta không quá 0,015;
- pHash Hamming không quá 4;
- SigLIP cosine ít nhất 0,985.

Candidate được xét theo P0 → P1 → P2, BTC preference nếu có, quality rồi frame index.
Dedup vì vậy ưu tiên protected evidence thay vì frame coverage thông thường.

## 13. Coverage gap-fill

Coverage chạy sau adaptive dedup và không thay thế candidate đã chọn. Nếu một temporal
gap lớn hơn 0,5 giây, pipeline chèn probe frame trước target `left + 0,4 s`, cân bằng:

- độ gần target;
- probe quality.

Thiết lập:

```python
"max_gap_sec": 0.50
"target_step_sec": 0.40
"verify_tolerance_sec": 0.02
```

Coverage frame có source `coverage_floor`, priority P2. Với category thường, validation
yêu cầu gap không quá khoảng 0,52 giây.

## 14. L25 verified static-state consolidation

L25 là video giảng bài/slide, nên coverage 0,4 giây có thể tạo nhiều frame gần như giống
nhau. Stage này chạy riêng cho L25 sau coverage.

### 14.1 Static-state guards

Candidate L25 được decode lại ở width 640. Hai candidate liên tiếp chỉ thuộc cùng static
state khi tất cả guard cùng đạt:

| Guard | Ngưỡng |
|---|---:|
| input gap | ≤ 0,60 s |
| pHash Hamming | ≤ 2 |
| text-hash Hamming | ≤ 2 |
| text-density delta | ≤ 0,005 |
| SigLIP cosine | ≥ 0,995 |
| global grayscale delta | ≤ 0,012 |
| top local grayscale delta | ≤ 0,025 |
| global edge delta | ≤ 0,010 |
| top local edge delta | ≤ 0,025 |
| motion raw | ≤ 0,35 |

Local grayscale/edge grid 12×20 bảo vệ thay đổi nhỏ như thêm công thức, khoanh đáp án,
highlight hoặc handwriting.

### 14.2 Frame được giữ

Một run chỉ được compress khi có ít nhất 5 frame và kéo dài ít nhất 1,5 giây. Pipeline
luôn giữ:

- frame đầu run;
- frame cuối run;
- frame quality tốt nhất;
- mọi P0/protected trigger;
- heartbeat để gap static không quá 3,6 giây.

Dynamic interval vẫn phải đạt coverage khoảng 0,52 giây. Nếu protected-frame hoặc
conditional coverage verification thất bại, stage fallback nguyên bộ frame trước
compression, không chấp nhận giảm recall để tiết kiệm dung lượng.

Registry đánh dấu `l25_static_verified`, `l25_static_run_id` và các source role như
`l25_static_first`, `l25_static_best`, `l25_static_heartbeat`.

## 15. Materialize JPEG và registry

Sau khi selection hoàn tất, pipeline decode video lần cuối đến frame target xa nhất.
Với mỗi target:

1. Decode frame RGB/BGR chính xác theo decode-order index.
2. Resize cạnh lớn nhất về tối đa 1.280 px.
3. Tính lại quality, pHash, text hash và text density trên ảnh lưu.
4. Ghi JPEG quality 92.
5. Tạo row registry.

Đường dẫn canonical:

```text
keyframes/L23/L23_V015/f00000123.jpg
```

Frame ID:

```text
L23_V015@f00000123
```

Registry chứa tối thiểu:

- video/category/frame/time/FPS/segment;
- sources và priority;
- quality, pHash, text hash, text density;
- scores JSON;
- L25 static-state metadata;
- image path và source video path.

`map-keyframes/<video_id>.csv` giữ schema legacy đúng thứ tự:

```text
n,pts_time,fps,frame_idx
```

Mặc định:

```python
"save_selector_embeddings": False
"save_final_embeddings": False
"final_embeddings.backend": "none"
```

Vì vậy extraction không re-encode toàn bộ retained frame bằng SigLIP/PE.

## 16. Lifecycle một video và resume

`process_video()` thực hiện theo thứ tự:

1. Kiểm tra `_SUCCESS.json` nếu `skip_existing=True`.
2. Probe metadata.
3. Load hoặc chạy Pass 1.
4. Tạo sentinel events.
5. Build segments và InfoShot candidates.
6. Merge events và chạy microburst.
7. Optional BTC union, hiện tắt.
8. Safe dedup.
9. Coverage gap-fill.
10. L25 consolidation nếu phù hợp.
11. Materialize và ghi registry.
12. Ghi stats và success marker.

Success marker chỉ được chấp nhận khi:

- pipeline signature khớp;
- registry còn tồn tại;
- loose JPEG hoặc verified archive còn tồn tại, trừ registry-only mode.

Smoke output có hậu tố `__smoke` và không tạo production success marker, vì vậy batch
sau đó vẫn xử lý video thật.

## 17. Smoke mode và kiểm tra chất lượng

Smoke chạy full mọi video trong `smoke_video_paths` theo thứ tự. Một video lỗi được ghi
vào `errors/` nhưng không chặn video smoke tiếp theo.

Diagnostics gồm:

- retained count/density;
- source distribution;
- temporal gap statistics;
- 15 gap lớn nhất;
- contact sheet sample đều toàn timeline;
- gallery phân trang;
- raw-frame inspection quanh `smoke_inspect_times`;
- khoảng cách từ raw frame đến keyframe gần nhất.

Để smoke một video:

```python
"run_mode": "smoke",
"smoke_video_paths": [
    "/kaggle/input/.../Videos_L23_a/video/L23_V015.mp4",
],
```

## 18. Batch mode, output và lỗi

Batch sort video tự nhiên, áp dụng range/shard rồi xử lý tuần tự. Sau mỗi video:

- cập nhật CSV run summary;
- xóa error cũ nếu video thành công;
- xóa completed Pass-1 cache theo storage policy;
- kiểm tra archive/quota;
- chạy garbage collection và giải phóng CUDA cache;
- cập nhật `session_state.json`.

Lỗi một video được ghi vào:

```text
errors/<video_id>.error.json
```

Batch tiếp tục với video kế tiếp.

## 19. Output layout

Với session `L21,L22,L23`:

```text
/kaggle/working/infoshootpp/l21-l22-l23/
├── cache/
│   ├── probe/
│   └── sentinel/
├── keyframes/<category>/<video_id>/*.jpg
├── archives/
│   ├── <video_id>.keyframes.zip
│   └── <video_id>.archive.json
├── registry/
│   ├── <video_id>.csv
│   ├── <video_id>.jsonl
│   ├── <video_id>.stats.json
│   └── <video_id>._SUCCESS.json
├── map-keyframes/<video_id>.csv
├── selector-embeddings/
├── final-embeddings/
├── runs/
│   ├── run_l21-l22-l23.csv
│   └── session_state.json
├── errors/
├── frame_registry.parquet
├── frame_registry.jsonl
├── run_summary.csv
└── run_manifest.json
```

Embedding folders được tạo nhưng trống với cấu hình mặc định. Cache của video thành công
bị xóa trong batch; cache của video lỗi hoặc đang chạy có thể còn để phục vụ resume.

## 20. Storage guard và archive

Thiết lập hiện tại:

```python
"archive_trigger_gb": 12.0
"quota_gb": 20.0
"reserve_for_next_video_gb": 2.0
"delete_images_after_verify": True
"delete_pass1_cache_after_success": True
```

Sau khi session đạt 12 GiB:

1. Loose JPEG của video hoàn tất được đóng gói thành
   `archives/<video_id>.keyframes.zip`.
2. ZIP được CRC-verify và đối chiếu member list.
3. Registry được cập nhật với archive path/member và URI `zip://...`.
4. Metadata archive được ghi atomically.
5. Chỉ sau các bước trên, loose JPEG mới bị xoá.

JPEG đã nén nên ZIP chủ yếu giảm file count, không giảm nhiều byte. Hàng rào dung lượng
thật là stop line 18 GiB. Khi đạt stop line, notebook dừng trước video kế và ghi:

```json
{
  "stopped_for_storage": true,
  "next_range_start": 123,
  "next_video_id": "L22_V..."
}
```

Session tiếp theo dùng `next_range_start` để chạy tiếp.

## 21. Global registry, manifest và registry bundle

Sau extraction, notebook tạo:

- `frame_registry.parquet`;
- `frame_registry.jsonl`;
- `run_summary.csv`;
- `run_manifest.json` chứa config, signature, counts và trạng thái session.

Nếu `export_registry_zip=True`, nó còn tạo:

```text
/kaggle/working/infoshootpp_v4_6_<session_id>_registry.zip
```

Registry ZIP chứa metadata, mapping, summaries, run state và archive metadata. Nó không
chứa loose JPEG, selector/final embeddings hoặc `*.keyframes.zip`. Muốn giữ asset thật,
phải Save Version/commit toàn bộ thư mục session dưới `/kaggle/working/infoshootpp/`.

## 22. Validation

Validation kiểm tra:

- registry tồn tại và không rỗng;
- `frame_idx` không trùng và tăng dần;
- map-keyframes đúng 4 cột và đúng row count;
- image/archive được registry tham chiếu còn tồn tại;
- dynamic coverage không quá khoảng 0,52 giây;
- L25 verified-static gap không quá khoảng 3,62 giây;
- L25 dynamic gap vẫn theo bound thường.

Notebook còn hỗ trợ annotation CSV:

```text
video_id,event_start,event_end,event_type[,clear_start,clear_end]
```

để tính temporal EventHitRecall, ClearHitRecall và Complete Video Recall. Readable text
recall thật sự vẫn cần OCR/transcript ground truth ở stage riêng.

## 23. Merge nhiều Kaggle session

Sau khi mỗi session được Save Version thành Kaggle Dataset:

1. Tạo notebook merge mới.
2. Attach mọi Dataset shard.
3. Đặt `run_mode="merge"`.
4. Điền `merge.input_roots`.

Ví dụ:

```python
"run_mode": "merge",
"merge": {
    "input_roots": [
        "/kaggle/input/infoshootpp-l21-l23",
        "/kaggle/input/infoshootpp-l24-l26",
        "/kaggle/input/infoshootpp-l27-l30",
    ],
    "output_root": "/kaggle/working/infoshootpp_merged",
    "copy_assets": False,
    "strict_pipeline_signature": True,
    "strict_session_complete": True,
    "expected_categories": None,
    "expected_total_videos": 1478,
    "enforce_expected_total": True,
}
```

Merge kiểm tra:

- session manifest và trạng thái complete;
- pipeline signature nhất quán;
- duplicate video ID giữa các shard;
- registry/image/archive hợp lệ;
- category và tổng video dự kiến.

`copy_assets=False` là chế độ khuyên dùng. Global registry thêm source session/root và
asset relative path, tham chiếu trực tiếp vào attached Kaggle Dataset shard mà không copy
hàng trăm GB JPEG. `copy_assets=True` chỉ dùng khi output quota đủ.

## 24. Cách chạy trên Kaggle

### 24.1 Smoke kiểm tra chất lượng

1. Chọn accelerator `GPU T4 ×2`.
2. Bật Internet lần đầu tải SigLIP, hoặc attach model cache.
3. Đặt `run_mode="smoke"`.
4. Điền một hay nhiều đường dẫn trong `smoke_video_paths`.
5. Run All.
6. Kiểm tra source distribution, gap, contact sheet và gallery.

### 24.2 Batch extraction

1. Đặt `run_mode="batch"`.
2. Chọn `session_categories` và `session_partitions`.
3. Run All.
4. Đọc `runs/session_state.json` và `run_manifest.json`.
5. Nếu storage stop, dùng `next_range_start` cho session kế.
6. Save Version/Commit output thành Kaggle Dataset shard.

### 24.3 Merge

1. Attach toàn bộ Dataset shard.
2. Chuyển sang `run_mode="merge"`.
3. Điền `merge.input_roots`.
4. Run All và chỉ chấp nhận merge khi audit đạt.

## 25. Hiệu năng và giới hạn hiện tại

- Pass 1 tận dụng hai GPU nhưng PyAV decode vẫn là CPU software decode.
- GPU 0 làm native preprocessing; cả GPU 0/1 encode SigLIP.
- Farneback motion, hash, microburst và JPEG encode vẫn chủ yếu dùng CPU.
- Non-L25 thường decode video ba lượt: Pass 1, microburst và materialize.
- L25 có thêm một lượt decode candidate-state trước materialize.
- GPU idle sau dòng `GPU pass1` là bình thường vì downstream thiên về CPU/I/O.
- `save_selector_embeddings=False` tránh một lượt SigLIP cuối nhưng không giảm số lần
  decode cần để ghi JPEG.
- Coverage 0,4 giây tạo output dày ở video động; L25 consolidation chỉ áp dụng cho
  verified static run, không áp dụng rộng cho action/news video.
- ZIP JPEG giảm file count nhiều hơn giảm dung lượng; phải dựa vào quota stop và chia
  session để tránh vượt Kaggle output quota.

## 26. Cấu hình production tham khảo

Chạy toàn bộ partition `a` của L25 và L26:

```python
"run_mode": "batch",
"session_categories": "L25,L26",
"session_partitions": "a",
"range_start": 0,
"range_end": None,
"num_shards": 1,
"shard_index": 0,
```

Chạy mọi partition của L21:

```python
"run_mode": "batch",
"session_categories": "L21",
"session_partitions": "",
```

Chạy riêng L26_d và L26_e:

```python
"run_mode": "batch",
"session_categories": "L26",
"session_partitions": "d,e",
```

Trong mọi trường hợp production hiện tại nên giữ:

```python
"motion_fps": 5.0
"btc_union.enabled": False
"save_selector_embeddings": False
"save_final_embeddings": False
"session_storage.enabled": True
```

## 27. Snapshot merge local ban đầu trước L25 retrieval post-processing

Ngày 12/08/2026, toàn bộ output từ 16 notebook version đã được audit và merge vật lý
trên máy local. Pre-merge audit phát hiện 900 candidate video; trong đó 27 video L24
từ `vnhtbo_ver5` trùng ID với `baovo123_ver4`. Theo quy tắc ưu tiên nguồn L24 đầy đủ,
27 bản từ `vnhtbo_ver5` bị loại và toàn bộ 43 video L24 được lấy từ
`baovo123_ver4`.

Kết quả ngay sau bước merge vật lý ban đầu (đây là mốc lịch sử trước khi thay L25):

- 873 video duy nhất.
- 1.567.052 JPEG keyframe.
- 873 map-keyframes CSV.
- 873 registry CSV, kèm stats và success marker tương ứng.
- Không có video thiếu, duplicate được chọn, mismatch số ảnh hoặc lỗi hậu merge.
- Pipeline signature của các session nguồn nhất quán.

Thống kê theo category trước và sau merge khớp tuyệt đối:

| Category | Số video | Số keyframe |
|---|---:|---:|
| L21 | 29 | 111.031 |
| L22 | 31 | 130.134 |
| L23 | 25 | 40.808 |
| L24 | 43 | 69.826 |
| L25 | 88 | 382.208 |
| L26 | 498 | 559.060 |
| L27 | 16 | 33.201 |
| L28 | 24 | 90.042 |
| L29 | 23 | 83.697 |
| L30 | 96 | 67.045 |
| **Tổng** | **873** | **1.567.052** |

### 27.1 Quy trình merge an toàn dung lượng

Mỗi archive video được xử lý tuần tự:

```text
đọc thống kê pre-merge
    ↓
giải nén một archive vào staging
    ↓
đếm và đối chiếu toàn bộ JPEG với registry/map/stats
    ↓
atomic move vào thư mục đích
    ↓
copy và chuẩn hoá metadata
    ↓
ghi checkpoint merge_state.json
    ↓
chỉ sau đó mới xoá archive nguồn vừa hoàn tất
```

Cách này tránh phải giữ đồng thời toàn bộ archive và toàn bộ ảnh đã giải nén. Tổng cộng
773 archive đã được xác minh rồi xoá; nội dung của chúng hiện nằm trong output merge.
Muốn khôi phục chính các file ZIP đã xoá phải tải lại notebook output. Hai version nguồn
chứa tổng cộng 100 video loose JPEG được move trực tiếp trên cùng filesystem. 27 archive
L24 bị loại do trùng vẫn được giữ trong thư mục nguồn và không xuất hiện trong dataset
merge.

Merge có checkpoint theo từng video nên có thể chạy lại để tiếp tục nếu tiến trình bị
gián đoạn. Thư mục `.staging` sau khi hoàn tất không còn dữ liệu tạm.

### 27.2 Cấu trúc output local

Output hoàn chỉnh nằm tại `infoshootpp_merged_873/` và có cấu trúc:

```text
infoshootpp_merged_873/
├── keyframes/
│   ├── L21/L21_V001/*.jpg
│   ├── L21/L21_V002/*.jpg
│   ├── L22/L22_V001/*.jpg
│   └── ...
├── map-keyframes/
│   ├── L21/L21_V001.csv
│   ├── L21/L21_V002.csv
│   ├── L22/L22_V001.csv
│   └── ...
├── registry/
│   └── Lxx/
│       ├── Lxx_Vxxx.csv
│       ├── Lxx_Vxxx.stats.json
│       └── Lxx_Vxxx._SUCCESS.json
├── metadata/
│   ├── premerge_video_statistics.csv
│   ├── premerge_category_statistics.csv
│   ├── duplicate_resolution.csv
│   ├── premerge_summary.json
│   ├── merge_state.json
│   ├── postmerge_video_statistics.csv
│   ├── postmerge_category_statistics.csv
│   └── postmerge_summary.json
└── merge_manifest.json
```

Registry sau merge đã được viết lại `image_path` để trỏ trực tiếp tới JPEG trong cấu
trúc mới; các cột tham chiếu archive cũ được làm rỗng.

### 27.3 Báo cáo kiểm chứng

Các file dùng để đối chiếu:

- [Thống kê từng video trước merge](infoshootpp_merged_873/metadata/premerge_video_statistics.csv)
- [Thống kê category trước merge](infoshootpp_merged_873/metadata/premerge_category_statistics.csv)
- [Quyết định xử lý 27 duplicate L24](infoshootpp_merged_873/metadata/duplicate_resolution.csv)
- [Tóm tắt pre-merge](infoshootpp_merged_873/metadata/premerge_summary.json)
- [Checkpoint của 873 video](infoshootpp_merged_873/metadata/merge_state.json)
- [Thống kê từng video sau merge](infoshootpp_merged_873/metadata/postmerge_video_statistics.csv)
- [Thống kê category sau merge](infoshootpp_merged_873/metadata/postmerge_category_statistics.csv)
- [Tóm tắt post-merge](infoshootpp_merged_873/metadata/postmerge_summary.json)
- [Manifest merge cuối](infoshootpp_merged_873/merge_manifest.json)

Post-merge audit tại thời điểm đó đã đếm lại độc lập từng thư mục keyframe, map CSV và registry CSV.
Kết quả `audit_ok=true`, `problems=[]`, tổng sau merge là 873 video và 1.567.052
keyframe, khớp tuyệt đối với thống kê pre-merge. Tại thời điểm hoàn tất, output chiếm
khoảng 242 GiB và filesystem còn khoảng 67 GiB trống.

> **Lưu ý:** 1.567.052 keyframe và 382.208 keyframe L25 trong Section 27 là snapshot
> lịch sử của master high-recall. Dataset vận hành hiện tại đã thay L25 bằng retrieval
> set hậu xử lý được mô tả ở Section 28; tổng hiện tại là **1.339.055 keyframe**.

## 28. L25 retrieval post-processing và corpus hiện hành

Sau merge ban đầu, L25 được hậu xử lý riêng vì đây chủ yếu là video bài giảng/ôn thi.
Master InfoShot++ high-recall giữ coverage dày để không bỏ sót nội dung, nhưng với slide
tĩnh, `coverage_floor` tạo rất nhiều JPEG gần như giống nhau. Retrieval set mới dùng BTC
làm backbone và chỉ giữ phần InfoShot++ có detector evidence thật:

```text
L25 retrieval set
    = BTC L25 backbone
    + InfoShot++ adaptive residual có evidence
    - coverage-derived-only frames
    - exact/strict cross-source duplicate đã xác minh
```

Stage này chỉ áp dụng cho L25. L21–L24 và L26–L30 không bị thay đổi.

### 28.1 Phân loại coverage-derived-only

Các source bắt đầu bằng `l25_static_` chỉ là metadata role do static consolidation thêm
sau coverage, không phải detector evidence độc lập. Định nghĩa được dùng là:

```python
META_SOURCES = {s for s in sources if s.startswith("l25_static_")}

coverage_derived_only = (
    "coverage_floor" in sources
    and not (sources - {"coverage_floor"} - META_SOURCES)
)
```

Vì vậy các frame sau bị loại:

```text
coverage_floor
coverage_floor|l25_static_anchor
coverage_floor|l25_static_anchor|l25_static_heartbeat
coverage_floor|l25_static_anchor|l25_static_first
```

Nhưng frame có genuine adaptive evidence như `text_event`, `text_local_event`,
`semantic_unique`, `semantic_boundary`, `motion_event`, `short_shot`, `microburst`,
`native_global` hoặc `native_local` luôn được giữ. Post-processing không thêm coverage
mới và không còn bắt buộc gap tối đa 0,5 giây; gap chỉ được report.

### 28.2 Số liệu đầu vào và filter L25

Audit dữ liệu gốc cho kết quả khớp toàn bộ sanity reference:

| Thành phần | Số frame |
|---|---:|
| InfoShot++ L25 ban đầu | 382.208 |
| Literal `sources == coverage_floor` | 250.418 |
| Có token `coverage_floor` | 256.530 |
| Coverage-derived-only bị loại | 256.530 |
| InfoShot++ adaptive residual còn lại | 125.678 |
| Coverage đồng thời có genuine adaptive evidence | 0 |

Toàn bộ 125.678 adaptive residual được bảo vệ khỏi dedup InfoShot↔InfoShot. Near-dedup
chỉ xét cặp BTC↔InfoShot trong cửa sổ thời gian 0,6 giây, và chỉ merge khi tất cả guard
ảnh/text/local cùng đạt. Candidate InfoShot mang một trong các protected source sau
không được near-merge, kể cả khi ảnh rất giống:

```text
text_event
text_local_event
semantic_unique
semantic_unique_primary
semantic_boundary
short_shot
motion_event
```

Điều này bảo vệ các thay đổi nhỏ nhưng retrieval-critical như số HUD `68 → 69`, một
dòng công thức, dấu tick, circle, highlight hoặc subtitle mới.

### 28.3 BTC backbone và canonicalization

BTC L25 có 88 video, 37.445 JPEG và 37.445 mapping row. Mapping gốc dùng đúng schema:

```text
n,pts_time,fps,frame_idx
```

CSV BTC có 362 duplicate `frame_idx` group/row trên 81 video. `frame_idx` trong CSV vẫn
được coi là authoritative như đáp án BTC; mỗi group được canonicalize thành một record,
giữ toàn bộ `btc_n` provenance và chọn một JPEG representative thực. Có 17 visual
mismatch group trên 16 video; theo quyết định review, pipeline dùng ảnh representative
bên trái trong contact sheet và vẫn ghi đầy đủ mismatch vào audit.

Canonical identity cuối:

```python
frame_id = f"{video_id}@f{frame_idx:08d}"
```

Nếu BTC và InfoShot có cùng `frame_idx`, chúng được exact-merge và provenance trở thành
`btc+infoshoot`. Với near pair khác `frame_idx`, pipeline chỉ merge khi toàn bộ strict
guards đều pass; semantic cosine, pHash hoặc global similarity đơn lẻ không đủ để xóa.

### 28.4 Kết quả dedup và L25 retrieval set

| Chỉ số | Giá trị |
|---|---:|
| BTC input | 37.445 |
| BTC canonical sau internal duplicate collapse | 37.083 |
| BTC duplicate groups/rows collapsed | 362 |
| Exact BTC↔InfoShot overlap | 8.541 |
| Strict near-duplicate overlap | 9 |
| Near pairs được tính strict metrics | 7.050 |
| Protected near pairs giữ cả hai | 45.656 |
| **L25 retrieval set cuối** | **154.211** |

Phân bố provenance của 154.211 frame cuối:

| `source_kind` | Số frame |
|---|---:|
| `btc` | 28.533 |
| `infoshoot` | 117.128 |
| `btc+infoshoot` | 8.550 |

Physical representative thực tế:

| `representative_source` | Số frame |
|---|---:|
| BTC | 31.481 |
| InfoShot++ | 122.730 |

So với L25 high-recall ban đầu, retrieval set giảm 227.997 frame. Registry vẫn giữ
`representative_btc_n`, `btc_n_all`, `infoshoot_frame_id`, `source_kind`, evidence
`sources`, hash/text metrics và physical representative; legacy map vẫn đúng thứ tự bốn
cột `n,pts_time,fps,frame_idx`.

### 28.5 Cài đặt transactional vào corpus master

L25 mới được stage và audit trước khi swap. Quy trình thay thế:

```text
audit post-process output + bind SHA-256 với đúng master upstream
    ↓
stage 154.211 ảnh + map + registry chuẩn hoá
    ↓
kiểm từng registry row ↔ map row ↔ JPEG
    ↓
transactional swap keyframes/L25, registry/L25, map-keyframes/L25
    ↓
audit lại toàn bộ L21–L30
    ↓
chỉ khi full audit pass mới xoá cây L25 cũ 382.208 frame
```

Registry L25 trong master dùng schema union 38 cột để giữ compatibility với extractor cũ
và provenance mới. Metadata extraction cũ chỉ được join bằng `infoshoot_frame_id`, không
join mù theo final `frame_id`, tránh gắn nhầm metadata coverage cũ vào BTC-only frame.
Mọi operational `image_path` được viết lại vào corpus master.

Master hiện là mixed pipeline:

```text
L21–L24, L26–L30 : InfoShot++ v4.6 signature ab304005fc8f
L25               : retrieval post-process signature 7999f9cb189ee115
corpus signature  : mixed:ab304005fc8f+7999f9cb189ee115
```

Source post-processing tạm `l25_retrieval_merged/` đã được xóa sau khi xác nhận master
độc lập. Một số path tuyệt đối nằm trong Parquet provenance lịch sử vì vậy không còn là
operational path; registry CSV trong master mới là nguồn đường dẫn vận hành.

### 28.6 Số keyframe hiện tại

Dataset vận hành hiện tại nằm tại `infoshootpp/`, vẫn đủ 873 video:

| Category | Số video | Số keyframe hiện tại |
|---|---:|---:|
| L21 | 29 | 111.031 |
| L22 | 31 | 130.134 |
| L23 | 25 | 40.808 |
| L24 | 43 | 69.826 |
| L25 | 88 | **154.211** |
| L26 | 498 | 559.060 |
| L27 | 16 | 33.201 |
| L28 | 24 | 90.042 |
| L29 | 23 | 83.697 |
| L30 | 96 | 67.045 |
| **Tổng hiện tại** | **873** | **1.339.055** |

Full-corpus audit sau replacement đã pass:

- 873 video và 1.339.055 JPEG.
- Registry/map/JPEG count khớp theo từng video.
- Không duplicate canonical `frame_id` hoặc `frame_idx` trong một video.
- `run_summary.csv` còn đủ 873 row và tổng `final_frames=1.339.055`.
- `run_manifest.json`, `merge_manifest.json`, `merge_state.json` và post-merge metadata
  đều đã cập nhật về current state.
- Các `premerge_*` artifact và nested pre-merge audit vẫn giữ nguyên như lịch sử.

Artifact kiểm chứng hiện tại:

- [L25 install audit](infoshootpp/metadata/l25_postprocess/install_audit.json)
- [L25 install manifest](infoshootpp/metadata/l25_postprocess/install_manifest.json)
- [L25 post-process audit](infoshootpp/metadata/l25_postprocess/audit.json)
- [L25 post-process run summary](infoshootpp/metadata/l25_postprocess/run_summary.csv)
- [Current run summary 873 video](infoshootpp/run_summary.csv)
- [Current run manifest](infoshootpp/run_manifest.json)
- [Current post-merge category statistics](infoshootpp/metadata/postmerge_category_statistics.csv)

Tại thời điểm cập nhật, toàn bộ `infoshootpp/` chiếm khoảng 206 GiB; riêng payload JPEG
keyframe khoảng 202,17 GiB.

## 29. Bản phân phối trên Hugging Face Storage Bucket

Ngày 13/08/2026, corpus hiện hành sau L25 post-processing đã được upload hoàn tất lên
Hugging Face Storage Bucket:

```text
Bucket ID     : Baonenha1/DATA-AIC-Keyframe
Bucket URI    : hf://buckets/Baonenha1/DATA-AIC-Keyframe/
Dataset prefix: infoshootpp-v1/
Upload status : complete
Video         : 873
Keyframe      : 1.339.055
```

Đây là bản phân phối của **current corpus** trong Section 28, không phải snapshot
1.567.052 frame trước L25 post-processing. Mọi remote object đã được kiểm tra size sau
upload trước khi TAR tạm trên máy local bị xóa.

### 29.1 Shard layout trên Bucket

JPEG đã nén sẵn nên các shard dùng TAR không compression để đóng gói/giải nén nhanh và
không tốn CPU vô ích. Mỗi category là một shard:

```text
hf://buckets/Baonenha1/DATA-AIC-Keyframe/infoshootpp-v1/
├── keyframes/
│   ├── L21.tar
│   ├── L22.tar
│   ├── L23.tar
│   ├── L24.tar
│   ├── L25.tar
│   ├── L26.tar
│   ├── L27.tar
│   ├── L28.tar
│   ├── L29.tar
│   └── L30.tar
├── metadata/
│   └── infoshootpp_metadata.tar
├── manifests/
│   ├── dataset_manifest.json
│   └── keyframe_category_shards.jsonl
└── README_DOWNLOAD.md
```

Mỗi keyframe TAR giữ nguyên cấu trúc member:

```text
infoshootpp/keyframes/Lxx/Lxx_Vxxx/f########.jpg
```

Do tất cả TAR đều bắt đầu bằng `infoshootpp/`, có thể giải nén nhiều category vào cùng
một thư mục cha để tái tạo corpus mà không cần move/rename sau đó.

### 29.2 Nội dung metadata TAR

`infoshootpp_metadata.tar` có kích thước 888.432.640 byte và SHA-256:

```text
65f99d79320d1e2d5d7222379e5117d96b1d5d1163b58a2462433fdd1c11c857
```

TAR này chứa mọi dữ liệu không phải JPEG:

```text
infoshootpp/
├── map-keyframes/             # 873 legacy CSV
├── registry/                  # 873 registry + stats/success
├── metadata/                  # audit, postmerge, L25 provenance
├── run_summary.csv            # 873 video
├── run_manifest.json
└── merge_manifest.json
```

Vì vậy `map-keyframes/` và `registry/` đã được upload đầy đủ nhưng không xuất hiện thành
873 object CSV riêng trên giao diện Bucket. Cách đóng gói này giảm mạnh file count và số
HTTP request. Một task embedding/OCR nên tải metadata TAR trước; sau đó chỉ tải các
category TAR thực sự cần xử lý.

### 29.3 Manifest và checksum các category shard

| Shard | Video | Keyframe | TAR bytes | SHA-256 |
|---|---:|---:|---:|---|
| L21.tar | 29 | 111.031 | 18.262.026.240 | `9e641def2d2da0e75cd45f0d6e686c07a91b1f0d483832960cce0f79bc734146` |
| L22.tar | 31 | 130.134 | 21.711.011.840 | `4fd4a5bdbfe927ebfa7e3d964fb2cf3c10f34fef4a4262de69aa8c6b20a0c390` |
| L23.tar | 25 | 40.808 | 7.851.427.840 | `0316081ce228d9e4c8c8bc84e6dcd47a4106fc38f670a13c0deee2d2a0717084` |
| L24.tar | 43 | 69.826 | 15.279.052.800 | `c71e663d346b062803a34981e909f367d67f0d51496820ceb518afa903a9d28e` |
| L25.tar | 88 | 154.211 | 23.876.280.320 | `ce8afc5a0dbf9700d0545c64c10371e6366c60ecf327adae9fafbabdba421a4b` |
| L26.tar | 498 | 559.060 | 79.997.450.240 | `82ed494c26248472856c134fcd37b982f1ea9fc500f37534f77297fa492d67af` |
| L27.tar | 16 | 33.201 | 6.732.646.400 | `4c4b40b82e0c9e6a9beddc511445514d039a4f24ffa407b14bc981ea563db6fd` |
| L28.tar | 24 | 90.042 | 17.562.224.640 | `a2c27d67e139482ecbe4b9df0ef7aec8ed515a74cccfa0528eeffcfed6aaaef9` |
| L29.tar | 23 | 83.697 | 17.598.791.680 | `3c7d9bdeb8e5878b8aacaea009e3391a1de252cc56b076ea3e99a2f60d050808` |
| L30.tar | 96 | 67.045 | 10.603.458.560 | `597d0db9a1da4ec321a0c1534c7fa75a161d81057c6a777516f8cad05ea84992` |
| **Tổng category TAR** | **873** | **1.339.055** | **219.474.370.560** | xem manifest |

Tổng category TAR khoảng 204,40 GiB; cộng metadata TAR khoảng 205,23 GiB. Source corpus
dùng để tạo shard được nhận dạng bởi:

```text
run_manifest.json SHA-256:
11b5a0bfaf805f4a615ed5c77d796b30c2214783ccd4f59b9fa6123366981d9e

run_summary.csv SHA-256:
8ccd83d2113ad6f9d5e4a1b29000ed83ec7476edbdbc0fa327691cbe224d9a93
```

`manifests/dataset_manifest.json` và
`manifests/keyframe_category_shards.jsonl` trên Bucket là source of truth để download
có chọn lọc và verify checksum.

### 29.4 Tải toàn bộ trên Colab/Kaggle

Cài Hugging Face CLI có Storage Bucket support:

```bash
pip install -U "huggingface_hub==1.24.0"
```

Tải toàn bộ prefix:

```bash
hf buckets sync \
  hf://buckets/Baonenha1/DATA-AIC-Keyframe/infoshootpp-v1/ \
  ./DATA-AIC-Keyframe
```

Giải nén metadata và các keyframe TAR song song:

```bash
mkdir -p ./data

tar -xf ./DATA-AIC-Keyframe/metadata/infoshootpp_metadata.tar \
  -C ./data

find ./DATA-AIC-Keyframe/keyframes -name 'L*.tar' -print0 \
  | xargs -0 -n1 -P4 sh -c 'tar -xf "$1" -C ./data' _
```

Kết quả:

```text
./data/infoshootpp/
├── keyframes/
├── map-keyframes/
├── registry/
├── metadata/
├── run_summary.csv
└── run_manifest.json
```

Để verify một shard trước khi giải nén:

```bash
sha256sum ./DATA-AIC-Keyframe/keyframes/L25.tar
```

### 29.5 Tải chọn lọc cho embedding hoặc OCR

Ví dụ chỉ xử lý L25:

```bash
mkdir -p ./download ./data

hf buckets cp \
  hf://buckets/Baonenha1/DATA-AIC-Keyframe/infoshootpp-v1/metadata/infoshootpp_metadata.tar \
  ./download/infoshootpp_metadata.tar

hf buckets cp \
  hf://buckets/Baonenha1/DATA-AIC-Keyframe/infoshootpp-v1/keyframes/L25.tar \
  ./download/L25.tar

tar -xf ./download/infoshootpp_metadata.tar -C ./data
tar -xf ./download/L25.tar -C ./data
```

Downstream task nên dùng các khóa sau:

- `frame_id = {video_id}@f{frame_idx:08d}` làm canonical join key;
- `video_id`, `frame_idx`, `pts_time`, `fps` để map về timeline video;
- `sources` và `source_kind` để phân tích provenance/evidence;
- `map-keyframes/Lxx/Lxx_Vxxx.csv` chỉ là legacy mapping bốn cột;
- `registry/Lxx/Lxx_Vxxx.csv` là metadata chi tiết và nguồn identity chính.

`image_path` trong registry là absolute path của máy đã materialize corpus. Sau khi tải
sang Colab/Kaggle, không dùng trực tiếp prefix tuyệt đối đó; resolve JPEG bằng quy tắc:

```python
image_path = (
    DATA_ROOT
    / "infoshootpp" / "keyframes"
    / category / video_id
    / f"f{frame_idx:08d}.jpg"
)
```

Output embedding hoặc OCR không nên đổi tên/xóa keyframe. Nên ghi thành dataset mới và
join ngược bằng `frame_id`, ví dụ:

```text
embeddings/Lxx/*.parquet
ocr/Lxx/*.parquet
```

Các cột tối thiểu nên giữ trong output downstream:

```text
frame_id, video_id, category, frame_idx, pts_time, image_relpath
```

Embedding bổ sung vector/model/version; OCR bổ sung text, confidence, bounding boxes,
language và OCR model/version. Cách này cho phép retry theo category hoặc video mà không
làm thay đổi corpus keyframe nguồn.

### 29.6 Upload implementation và resume state

Upload được thực hiện bởi:

- [Bucket uploader](upload_infoshootpp_to_hf_bucket.py)
- [Uploader requirements](requirements-hf-bucket-upload.txt)
- [Local upload state](.hf_bucket_upload_state_DATA-AIC-Keyframe/upload_state.json)
- [Local dataset manifest](.hf_bucket_upload_state_DATA-AIC-Keyframe/dataset_manifest.json)
- [Local shard manifest](.hf_bucket_upload_state_DATA-AIC-Keyframe/keyframe_category_shards.jsonl)

Uploader dùng hai worker, retry lỗi mạng và scratch budget giữ tối thiểu 15 GiB dự
phòng. Mỗi object chỉ được ghi `uploaded` sau khi remote size khớp; TAR local sau đó mới
bị xóa. Upload state hiện là `complete`.

Hugging Face Storage Buckets là object storage không versioning. Không dùng lệnh sync
với `--delete`, và không xóa/ghi đè prefix `infoshootpp-v1/` nếu chưa tạo phiên bản dataset
mới hoặc xác minh lại manifest/checksum.
