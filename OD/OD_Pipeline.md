# Object Detection Pipeline — Technical Design, Decision History and Production Contract

## 1. Mục đích và trạng thái tài liệu

Đây là tài liệu duy nhất cho module Object Detection (OD), được hợp nhất từ review v3, v4, v5 và
tài liệu vận hành hiện tại. Tài liệu phục vụ:

- viết technical report và giải thích các quyết định thiết kế;
- triển khai notebook trên Google Colab A100;
- đọc keyframe từ Cloudflare R2 và resume an toàn;
- tạo metadata cho tìm vật thể, định vị, đếm và truy vấn màu;
- kiểm định, bulk index và phát hành Elasticsearch index;
- phân biệt rõ thiết kế mục tiêu, implementation hiện tại và bằng chứng từ run thực tế.

Source of truth thực thi là `build_nb.py`; file này sinh `OD_Kaggle_Notebook.ipynb`. Khi tài liệu và
notebook khác nhau, phải ưu tiên code rồi cập nhật lại tài liệu này.

Dataset production cuối cùng mang version/schema:

```text
pipeline_version = aic26-od-v5.1-gdino-augment
schema_version   = od-frame-v5
source_pipeline  = aic26-od-v5
```

## 2. Tóm tắt điều hành

Module xử lý khoảng 382.299 keyframe bằng YOLOE-26L open-vocabulary instance segmentation, làm giàu
kết quả thành metadata có contract rõ ràng, ghi các shard bất biến lên R2 và lập chỉ mục Elasticsearch
với nested mapping. Thiết kế ưu tiên bốn thuộc tính:

1. **Recall và khả năng truy vấn:** prompt theo genre, lưu bbox chuẩn hóa, màu, aliases và count.
2. **Tính nhất quán:** canonicalize synonym/role trước dedup và đếm.
3. **Khả năng phục hồi:** manifest, namespace hash, shard checksum và completion marker.
4. **Khả năng phát hành:** validation artifact, exact document-ID comparison và atomic alias switch.

Pipeline production đã chạy thành công theo hai tầng. YOLOE-26L segmentation xử lý toàn bộ keyframe;
sau đó Grounding DINO posthoc chạy trên các frame thỏa fallback/audit policy. Kết quả hai nguồn được
canonical-dedup, làm giàu lại metadata và ghi thành dataset v5.1 self-contained. Namespace YOLOE v5
được giữ bất biến làm nguồn, còn Elasticsearch sử dụng namespace v5.1 sau validation.

## 3. Phạm vi và yêu cầu hệ thống

### 3.1. Use case được hỗ trợ

- tìm frame có một loại vật thể;
- tìm vật thể theo vị trí ngôn ngữ như `top_left`, `bottom_right`;
- tìm vật thể gần một điểm user click;
- vẽ rectangle trên một ảnh và tìm vật thể có hình học tương tự ở frame khác;
- vẽ lại đúng bbox của detection match trên frontend;
- tìm theo màu đáng tin cậy;
- đếm số instance nhìn thấy của một số class rời rạc;
- resume job dài sau khi Colab ngắt;
- bulk index idempotent và chỉ chuyển alias sau validation.

### 3.2. Ngoài phạm vi

- OCR hoàn chỉnh cho logo/text region;
- ground-truth population counting trong crowd;
- phát hiện class hoàn toàn mới chưa có trong batch metadata tại query time;
- crop/vector embedding trong frame index;
- bảo đảm zero-shot precision/recall tuyệt đối;
- thay thế human/VLM verification cho top candidate quan trọng.

## 4. Dữ liệu đầu vào và đánh giá output legacy `objects/`

Đã audit toàn bộ 382.299 JSON trong `objects/`:

- 100% parse hợp lệ và năm mảng song song có cùng chiều dài;
- mỗi file luôn có đúng 100 candidate, kể cả nhiều candidate score rất thấp;
- score, box, OpenImages label ID và entity đều lưu dạng chuỗi;
- box có vẻ chuẩn hóa nhưng không có kích thước ảnh hoặc contract tự mô tả `xyxy/yxyx`;
- không có canonical label, model/source provenance, mask, màu, count semantics, spatial grid,
  runtime hash hoặc completion marker.

Kết luận: `objects/` là output Faster-RCNN/OpenImages legacy, chỉ phù hợp làm weak reference để so
agreement hoặc hỗ trợ audit. Nó **không tương thích** với schema v5 và không được resume/trộn vào
namespace production. Convert kiểu dữ liệu không thể tái tạo instance mask, color reliability và
runtime provenance đã thiếu.

## 5. Lịch sử phát triển và quyết định kỹ thuật

### 5.1. Prototype trước v3

Prototype chứng minh được khả năng detect nhưng chưa phù hợp production:

- chỉ đọc `/kaggle/input`, không đọc R2 trực tiếp;
- inference từng ảnh làm A100 thiếu tải;
- global union vocabulary rồi lọc theo genre sau inference;
- cùng một confidence threshold cho lưu detection và count;
- role như `reporter`, `cyclist`, `student` có thể đếm trùng với `person`;
- màu lấy median toàn bbox, dễ lấy background;
- chỉ có center grid, thiếu bbox normalized và position contract;
- append JSONL/crop embedding có thể tạo dangling reference khi runtime chết;
- không có explicit Elasticsearch mapping, resume identity hoặc release gate.

### 5.2. V3 — nền tảng R2, batch và metadata

V3 đưa vào các nền tảng chính:

- R2 manifest dựa trên `r2_key + size + etag`;
- config/manifest namespace;
- tải song song và prefetch shard kế tiếp;
- batch inference A100, OOM split;
- prompt YOLOE theo genre trước inference;
- canonical dedup cross-label/cross-model;
- bbox pixel/normalized, center, size, grid và position tags;
- strict nested Elasticsearch mapping;
- deterministic document ID và bulk idempotency.

Nhược điểm còn lại của v3 là runtime/model chưa được khóa đầy đủ, count threshold còn thô, màu vẫn
heuristic, query chưa rank tốt và completion semantics chưa đủ chặt.

### 5.3. V4 — retrieval ranking, count contract và commit marker

V4 xử lý các vấn đề production sau review:

| Vấn đề | Quyết định v4 |
|---|---|
| Query dùng toàn `filter`, `_score=0` | Dùng native nested sort cho query đơn giản; `script_score` cho geometry. |
| Rectangle chỉ cần overlap, box toàn frame dễ thắng | Rank bằng confidence, IoU, center, area và aspect similarity. |
| Một count threshold cho mọi class/model | Whitelist countable class và threshold theo source/canonical class. |
| Scene/substance bị đếm | Giữ searchable nhưng đặt `countable=false`. |
| Không diễn đạt được exact zero | Tách absence ở presence threshold và absence ở count threshold. |
| Lower-bound chỉ là cờ root | Thêm uncertainty/lower-bound theo từng canonical label. |
| Fallback chỉ nhìn top confidence | Policy kết hợp empty, confidence, count, genre anchor và audit sample. |
| Shard có error vẫn bị coi complete | Tách data shard khỏi completion marker; chỉ marker mới là commit. |
| Bulk retry request-level | Retry item-level cho lỗi tạm thời, fail hard với lỗi không retry được. |
| UI đọc index đang import dở | Import concrete index rồi atomic alias switch. |

### 5.4. V5 — runtime integrity, mask/Lab color và release gate

V5 tạo namespace/schema mới và thêm:

- runtime hash khóa model, weight, precision, package, CUDA/GPU và source code;
- pin/checksum MobileCLIP và Grounding DINO;
- cache prompt embedding của toàn bộ genre trước khi predictor chuyển sang FP16;
- prompt embedding checksum và segmentation dtype guard;
- mask-based CIE Lab color với reliability metadata;
- exact-zero semantics chặt hơn;
- cross-frame spatial retrieval evaluator;
- release report khóa exact R2/Elasticsearch ID set và marker checksum set.

## 6. Kiến trúc hiện tại

```text
Cloudflare R2 / Keyframes/
  │
  ├─ list objects → stable manifest → manifest_hash
  │
  ├─ stage 1.024 frame/shard bằng downloader song song
  │       └─ prefetch shard kế tiếp song song với GPU
  │
  ├─ YOLOE-26L segmentation + genre prompt
  │       ├─ bbox + score + class
  │       └─ instance mask
  │
  ├─ Grounding DINO posthoc theo fallback/audit policy
  │
  ├─ canonicalize → cross-label/source dedup
  │
  ├─ enrich metadata
  │       ├─ geometry/spatial
  │       ├─ masked Lab color
  │       └─ count/uncertainty
  │
  ├─ gzip JSONL shard → R2
  ├─ completion marker/checksum → R2 (commit point)
  ├─ validation + contact sheet + release report
  └─ Elasticsearch concrete index → exact-ID gate → atomic alias
```

Các module được notebook sinh ra:

| Module | Trách nhiệm |
|---|---|
| `od_r2.py` | R2 client, list/stage/upload, gzip JSONL streaming. |
| `od_pipeline_lib.py` | YOLOE/GDINO inference, dedup, color/count/spatial metadata. |
| `od_runner.py` | manifest, runtime binding, sharding, resume, validation và evaluator. |
| `od_elastic.py` | mapping, bulk import, release gate và query helper. |

## 7. Model, vocabulary và inference

### 7.1. YOLOE detector

- Weight bắt buộc: `yoloe-26l-seg.pt`.
- Task: open-vocabulary object detection + instance segmentation.
- Precision chính: FP16 trên CUDA.
- Không tự fallback sang YOLOE-11/v8 hoặc FP32 trong cùng namespace.
- Load/checksum hoặc warm-up lỗi thì fail trước khi commit shard.

YOLOE được chọn vì vừa cho open-vocabulary label vừa có instance mask. Mask không chỉ phục vụ hiển
thị mà còn giảm background contamination khi trích xuất màu.

### 7.2. Genre-conditional vocabulary

Base vocabulary được kết hợp với vocabulary riêng cho tám genre:

```text
news, cycling, lion_dance, exam, cooking, culture, mekong, positive
```

Genre hiện được ánh xạ theo batch K/L. Đây là metadata heuristic; nếu các video trong cùng batch khác
ngữ nghĩa, cần chuyển genre xuống cấp video/frame trong manifest.

Global union vocabulary đã bị loại vì các class không liên quan cạnh tranh trong cùng prediction,
tăng false positive và làm prompt không thực sự conditional. V5 tạo/cache toàn bộ genre embedding
khi model còn FP32, sau đó chỉ bind tensor đã cast sang đúng device/FP16 khi đổi genre.

### 7.3. Canonicalization và dedup

Role/subtype vẫn được giữ trong `label`/`aliases`, nhưng map về canonical entity trước count. Ví dụ:

```text
reporter/news anchor/police officer/cyclist/student/chef → person
motorcycle escort                                      → motorcycle
exam paper                                             → document
television/projector screen                            → screen
peloton/group of people                                → crowd
```

Detection được sort giảm dần theo confidence. Hai detection cùng canonical label và IoU từ 0,65 được
gộp bằng weighted box; confidence lấy max, aliases/sources hợp nhất, source score giữ max. Raw label
của candidate confidence cao nhất được giữ làm primary label.

### 7.4. Grounding DINO policy

Thiết kế mặc định `GDINO_MODE="policy"` gọi fallback khi:

- YOLOE không có detection;
- top confidence dưới 0,32;
- số detection thấp hơn mức tối thiểu theo genre;
- thiếu anchor class của genre chuyên biệt;
- frame thuộc deterministic audit sample (news 2%, genre khác 1%).

Prompt được chunk 20 label và inference batch mặc định 4. Production sử dụng policy này trong pass
posthoc: toàn bộ source shard được scan để lập target plan, nhưng chỉ target keyframe mới được tải và
đưa qua Grounding DINO.

### 7.5. Grounding DINO posthoc v5.1

Cell 9 là augmentation pass độc lập chạy sau khi tầng YOLOE đã hoàn tất:

1. xác minh tất cả shard nguồn có completion marker hợp lệ;
2. scan document YOLOE và tính lại policy từ detection đã lưu;
3. load/checksum Grounding DINO pinned revision hoàn toàn ở FP32 và chạy warm-up thật;
4. chỉ tải keyframe của target frame, infer theo genre và prompt chunk;
5. canonical-dedup với detection YOLOE, giữ metadata màu mask YOLOE khi hai nguồn match;
6. ghi lại đầy đủ mọi document sang namespace `aic26-od-v5.1-gdino-augment`, rồi commit marker cuối;
7. resume theo marker v5.1 nếu Colab hoặc R2 bị ngắt.

Frame không cần fallback được copy sang namespace mới với runtime/config provenance mới. Detection
GDINO-only không có segmentation mask nên màu chỉ dùng inner-box Lab và luôn
`color_reliable=false`. Policy posthoc dựa trên detection đã dedup/lọc được lưu; raw YOLO candidate
không được lưu nên đây là phép tái dựng bảo thủ, không thể giống tuyệt đối quyết định online trước
dedup.

Anchor check phải đọc `label`, `canonical_label` **và `aliases`**. Policy online chạy trên raw
detection trước dedup nên thấy đủ mọi label YOLOE phát ra; posthoc đọc detection đã dedup, nơi
`canonical_dedup` ghi đè `label` bằng raw label có confidence cao nhất và đẩy label còn lại xuống
`aliases`. Vì `chef`, `student`, `teacher`, `performer`, `farmer` đều canonical hóa về `person`,
bản `-v1` chỉ đọc `label`/`canonical_label` đã sinh `missing_genre_anchor_class` giả cho 49.580
frame — tập trung ở L26 cooking (79.590 frame) và L25 exam (37.445 frame), hai batch lớn nhất và
đúng là hai genre có anchor bị canonical hóa. `aliases` chỉ chứa label thật đã merge cộng VI alias
(không trùng tên anchor tiếng Anh) nên không tạo match giả. Đã sửa ở
`GDINO_POSTHOC_CODE_VERSION = aic26-gdino-posthoc-fp32-v2`.

## 8. Cấu hình A100 và throughput

Notebook source hiện mặc định:

| Tham số | Giá trị | Lý do |
|---|---:|---|
| `BATCH_SIZE` | 32 | Baseline an toàn cho A100; OOM tự chia đôi. |
| `IMAGE_SIZE` | 960 | Tăng khả năng thấy vật thể nhỏ so với 640. |
| `SHARD_SIZE` | 1024 | Checkpoint đủ thường mà không tạo quá nhiều object R2. |
| `DOWNLOAD_WORKERS` | 64 | Che độ trễ R2 bằng I/O song song. |
| `DETECTION_CONF` | 0,20 | Recall threshold để lưu candidate. |
| `YOLO_IOU` | 0,70 | Hạn chế loại người/vật đứng gần nhau. |
| `CANONICAL_DEDUP_IOU` | 0,65 | Gộp synonym/source trùng instance. |
| `MAX_DETECTIONS` | 300 | Giới hạn output và nhận biết saturation. |
| `GDINO_BATCH_SIZE` | 4 | Giới hạn VRAM cho fallback model. |

`BATCH_SIZE` là số ảnh trong một lần GPU inference; `SHARD_SIZE` là số document trong một checkpoint
R2. Với 382.299 frame và shard 1.024:

```text
ceil(382299 / 1024) = 374 shard
```

Tăng batch không làm giảm số shard. Nó chỉ giảm số forward pass trong mỗi shard và không chắc tăng
throughput nếu GPU đã bão hòa hoặc postprocess/color/R2 trở thành bottleneck.

### 8.1. Bằng chứng từ run Colab A100 hiện tại

Các log thực tế được ghi nhận trong quá trình vận hành:

- GPU: NVIDIA A100-SXM4 40 GB; YOLOE FP16;
- YOLOE source prefix:
  `Derived/ObjectDetection/aic26-od-v5/7af657b02cb0/input-6bbfa222abe6/runtime-cf50c70a8c3b6e43`;
- batch được người vận hành tăng lên 256;
- throughput ổn định quan sát được khoảng 17,9–18,4 frame/s;
- mỗi shard 1.024 frame mất khoảng 53–57 giây;
- các shard quan sát được đều `status={'ok': 1024}` và `committed=True`;
- một run batch nhỏ hơn trước đó đạt xấp xỉ 19,3–19,5 frame/s;
- Grounding DINO posthoc v2 đã pass checksum, warm-up, inference và hoàn tất các completion marker
  của namespace v5.1;
- dataset dùng để validate/index là output v5.1 trong `gdino_aug_summary["output_prefix"]`, không
  phải YOLOE source prefix ở trên.

Đây là quan sát vận hành, chưa phải benchmark kiểm soát: input mix, warm-up, R2 và metadata workload
có thể khác. Nó cho thấy batch 256 không tự động nhanh hơn batch nhỏ hơn.

## 9. R2 manifest, namespace và resume

### 9.1. Manifest identity

Manifest được tạo từ danh sách keyframe R2. Fingerprint cập nhật lần lượt bằng:

```text
r2_key<TAB>size<TAB>etag<NEWLINE>
```

Thay key, kích thước hoặc ETag sẽ đổi `manifest_hash`, ngăn resume nhầm input.

### 9.2. Output namespace

```text
Derived/ObjectDetection/aic26-od-v5/
  <config_hash>/
  input-<manifest_hash>/
  runtime-<runtime_hash>/
```

- `config_hash`: model config, inference config, metadata, vocab, aliases và input semantic config.
- `manifest_hash`: identity của tập keyframe.
- `runtime_hash`: model/weight thực tế, prompt/text encoder, precision, environment và code.

`BATCH_SIZE` nằm trong inference config nên thay đổi đúng cách sẽ đổi `config_hash` và bắt đầu
namespace mới. Resume chỉ xảy ra khi cả ba identity giống nhau.

`SHARD_SIZE` hiện nằm trong `run` config nhưng chưa tham gia `config_hash`. Đây là điểm cần cải thiện;
không được đổi shard size giữa run. Marker slice hash sẽ phát hiện nhiều mismatch, nhưng không nên dựa
vào đó như cơ chế migration.

### 9.3. Runtime identity

Runtime hash khóa:

- resolved YOLOE weight và SHA-256;
- MobileCLIP2-B file SHA, CLIP revision và source-code hash;
- SHA-256 của toàn bộ genre prompt embedding;
- Grounding DINO model/revision/weight SHA;
- precision thực tế;
- Python, Torch, CUDA, cuDNN, GPU;
- package versions và installed-environment lock hash;
- SHA-256 của `od_pipeline_lib.py`, `od_runner.py`, `od_r2.py`;
- segmentation dtype guard version.

### 9.4. Shard commit contract

Data shard có dạng `part-XXXXXX.jsonl.gz`. Upload data file chưa có nghĩa là complete. Commit point là
`part-XXXXXX.complete.json`, chỉ được upload khi `error_count=0`.

Marker chứa tối thiểu:

- config/manifest/runtime hash;
- resolved detector, detector/text/prompt/fallback checksums;
- segmentation guard version;
- manifest slice hash;
- document count và gzip SHA-256;
- first/last document ID và commit time.

Resume chỉ skip shard nếu marker parse được, đúng contract và raw shard tồn tại. Raw shard không có
marker, marker hỏng hoặc mismatch đều được coi là pending và chạy lại.

## 10. Xử lý lỗi và các bài học Colab

### 10.1. Dependency resolver và Pillow ABI

- Colab yêu cầu `requests==2.32.4`; cài 2.32.5 tạo dependency conflict.
- Colab preload native Pillow `_imaging` 11.3.0. Cài Python files Pillow 12.x trong cùng process gây
  `_Ink` import error hoặc core/Pillow version mismatch.
- Notebook pin `Pillow==11.3.0`, xóa toàn bộ `PIL.*` khỏi `sys.modules` và import lại sau pip.

### 10.2. Ultralytics CLIP AutoUpdate

YOLOE có thể tự cài CLIP và tải `mobileclip2_b.ts` ở lần `set_classes()` đầu, yêu cầu restart runtime
và làm thay đổi dependency sau khi job bắt đầu. V5 pin CLIP commit trong dependency cell, tạo prompt
và checksum text encoder trước runtime binding.

### 10.3. YOLOE text embedding precision binding

V5 tạo và cache toàn bộ MobileCLIP genre embedding trước khi predictor được bind vào runtime. Khi
đổi genre, embedding cache được chuyển sang đúng device/precision của detector; warm-up xác nhận
model và prompt đồng nhất trước khi khóa runtime hash.

### 10.4. Ultralytics segmentation mask dtype

Pipeline cài guard idempotent cho `process_mask/process_mask_native`: coefficient matrix nhỏ được
đưa về cùng precision FP32 với mask prototypes trước phép nhân, trong khi YOLOE backbone vẫn dùng
precision tối ưu trên CUDA. Self-test đi trực tiếp qua mask reconstruction kernel trước khi runtime
được bind.

### 10.5. Grounding DINO precision contract và production run

Grounding DINO posthoc chạy hoàn toàn ở FP32: model parameters, `pixel_values` và forward pass cùng
một dtype; autocast không được bật. Trước khi tạo namespace, cell bắt buộc kiểm revision, SHA-256 của
weight và chạy warm-up thật qua cả model lẫn postprocessor.

Run production đã hoàn tất thành công trên target subset. Cell giữ nguyên source YOLOE v5, merge
Grounding DINO rồi tạo dataset v5.1 self-contained. Runtime v5.1 khóa source config/runtime hash,
Grounding DINO revision/weight checksum, precision, policy và augmentation code version. Toàn bộ
source shard được scan và toàn bộ shard đích được ghi lại để dataset có thể index độc lập; GPU không
chạy lại YOLOE và Grounding DINO chỉ xử lý target frame.

### 10.6. Grounding DINO warm-up và `-inf` hợp lệ trong logits

`GroundingDinoContrastiveEmbedding` chủ động `masked_fill(-inf)` các text token không dùng rồi pad
tới `max_text_len=256`; `post_process_grounded_object_detection` biến các vị trí đó thành score 0
qua `sigmoid`. Vì vậy health check không yêu cầu mọi raw logit phải hữu hạn.

Warm-up hiện kiểm:

- `isnan(logits)` — NaN mới là lỗi thật;
- `isfinite(logits.sigmoid())` và tồn tại score dương — bắt trường hợp text branch chết hẳn;
- `isfinite(pred_boxes)` — box phải hữu hạn toàn bộ;
- chạy thật `post_process_grounded_object_detection` để chốt chữ ký API trước khi tạo namespace.

Từ transformers 5.x, `post_process_grounded_object_detection` cần `input_ids` để decode text label;
gọi thiếu sẽ raise `TypeError` giữa vòng lặp. Variant chữ ký được resolve một lần trong warm-up và
cache lại, thay vì nuốt `TypeError` ở mỗi vocab chunk của mỗi batch.

## 11. Frame document và detection schema

### 11.1. Frame-level fields

| Nhóm | Field tiêu biểu |
|---|---|
| Identity | `document_id`, `keyframe_id`, `video_id`, `frame_name/index`, `batch`, `genre`. |
| Media | `r2_key`, `image_url`, `width`, `height`, `input_etag`. |
| Provenance | pipeline/config/manifest/runtime hash, model và weight/text/prompt hash. |
| Summary | `labels`, `canonical_labels`, detection/count totals. |
| Retrieval | `object_text`, `spatial_tokens`, `count_tokens`. |
| Count safety | `counts_are_estimates`, `count_is_lower_bound`, `count_uncertain_labels`, notes. |
| Fallback | `fallback_status`, `fallback_reasons`, `fallback_error`. |

`document_id = <video_id>:<frame_name>` được dùng làm Elasticsearch `_id`, nên action `index` là
idempotent.

### 11.2. Detection-level fields

Mỗi phần tử `detections` gồm:

- identity: `instance_id`, raw `label`, `canonical_label`, aliases, label group;
- confidence/provenance: `conf`, `sources`, `source_scores`;
- count: `countable`, `counted`, threshold và source quyết định;
- geometry: `bbox_px`, `bbox_norm`, `center_norm`, normalized width/height/area;
- spatial: grid 7×7, horizontal/vertical zone, position, position tags;
- color: dominant/top colors, method, confidence, purity, entropy, Delta-E, coverage, reliability.

## 12. Spatial metadata và truy vấn vị trí

Pixel bbox được chuẩn hóa theo width/height:

```text
nx1=x1/W, ny1=y1/H, nx2=x2/W, ny2=y2/H
cx=(nx1+nx2)/2, cy=(ny1+ny2)/2
```

Center được lượng tử vào grid 7×7 và ba vùng ngang/dọc. Ví dụ một object ở góc trên trái có thể mang:

```text
horizontal_zone = left
vertical_zone   = top
position        = top_left
position_tags   = [left, top, top_left, r0c0]
```

Normalized coordinates làm query độc lập resolution và cho phép frontend vẽ bằng `bbox_norm` hoặc
`bbox_px` lấy từ `inner_hits`.

Hai rectangle giao nhau khi:

```text
det.x1 <= query.x2 AND det.x2 >= query.x1
det.y1 <= query.y2 AND det.y2 >= query.y1
```

Overlap chỉ là candidate filter; ranking hình học mới quyết định thứ tự.

## 13. Color extraction

Color chỉ chạy cho `COLOR_CLASSES` có màu bề mặt hữu ích và detection confidence từ 0,30. Quy trình:

1. chọn instance mask tương ứng detection;
2. scale mask theo chunk trên GPU;
3. crop bbox và resize mask 64×64;
4. chỉ chuyển batch boolean mask nhỏ về CPU;
5. với `person`, ưu tiên vùng thân/quần áo;
6. trim highlight/shadow cực trị nếu dynamic range lớn;
7. chuyển sRGB D65 sang CIE Lab;
8. lượng tử palette bằng Delta-E 1976 và giữ top colors;
9. tính purity, entropy, median Delta-E, coverage và confidence.

`color_reliable=true` chỉ khi dùng segmentation mask, đủ pixel/coverage và đạt các ngưỡng quality.
GDINO-only/mask thiếu dùng inner-box fallback nhưng luôn unreliable. Class ngoài whitelist hoặc
detection yếu được ghi skip reason, không giả tạo màu.

Query màu mặc định yêu cầu reliable dominant color. `color_mode="any"` có thể tăng recall cho vật thể
đa màu. `object_text` cũng chỉ nhận màu reliable để tránh background làm nhiễu BM25.

## 14. Count semantics

Pipeline tách recall threshold khỏi count threshold:

- detection được lưu từ confidence 0,20;
- count chỉ áp dụng `COUNTABLE_CLASSES` và threshold theo detector/canonical class;
- threshold YOLOE mặc định 0,35, ví dụ person 0,32;
- threshold GDINO mặc định 0,42, thường cao hơn vì calibration khác.

Scene/substance như smoke, fire, food, landscape vẫn searchable nhưng không có semantics instance
count ổn định và không được đếm.

Các field:

- `detection_count`: mọi detection qua recall/min-area;
- `counted_detection_count`: detection countable vượt threshold;
- `object_counts`: nested aggregate theo canonical label;
- `count_uncertain_labels`: class có instance trong vùng không chắc/lower-bound.

Crowd chỉ làm count `person` thành lower-bound. Detector chạm `max_det` làm mọi countable class có
nguy cơ bị truncate. Count là số instance **nhìn thấy bởi detector**, không phải population thật.

Query contract:

- `zero_mode="presence"`: không có detection từ recall threshold;
- `zero_mode="counted"`: không có detection vượt count threshold;
- exact dương với `strict=True`: loại lower-bound và frame còn candidate ngay dưới count threshold;
- `exclude_lower_bound=True` là mặc định.

## 15. Elasticsearch mapping và query contract

`detections` và `object_counts` bắt buộc là `nested`. Plain object sẽ flatten array và có thể ghép
label của instance A với bbox/màu của instance B.

Các query helper:

| Helper | Hành vi |
|---|---|
| `query_object` | Label-only dùng native nested filter/sort; hỗ trợ color và position. |
| `query_drawn_region` | Coarse overlap/center filter rồi geometry script ranking. |
| `query_near_point` | Rank theo confidence và khoảng cách center tới điểm click. |
| `query_count` | Exact/range/zero với count uncertainty contract. |

Label/color-only không chạy Painless để giảm latency. Position/point/drawing mới dùng script score.
Rectangle score mặc định:

```text
0.25 × confidence
+ 0.35 × IoU
+ 0.20 × centerProximity
+ 0.12 × areaSimilarity
+ 0.08 × aspectSimilarity
```

Candidate chỉ chạm biên có geometry score 0; area/aspect similarity phạt box phủ toàn frame. Weights
cần tune bằng query/GT đại diện, không phải hằng số tối ưu cho mọi dataset.

Bulk importer:

- đọc gzip shard trực tiếp từ R2;
- batch tối đa 500 docs/8 MiB;
- retry item-level 408/429/502/503/504;
- fail hard với lỗi không retry được;
- dùng deterministic `_id`;
- import vào concrete index, không ghi trực tiếp alias production.

## 16. Validation, metrics và release gate

Validation full kiểm:

- completion marker, raw shard existence và gzip checksum;
- config/manifest/runtime/model/prompt correspondence;
- exact document ID, duplicate/missing/extra;
- bbox normalized, confidence/source, color contract;
- countable/count consistency và per-label lower-bound;
- class/genre/confidence distributions và collapse warning;
- fallback status distribution.

Nó tạo deterministic contact sheet khoảng 200 ảnh. Khi có GT, evaluator báo:

- precision, recall, F1 theo class;
- count MAE, frame exact match, presence accuracy;
- `per_frame_box_recall_at_k`;
- cross-frame spatial Recall@10/50/100, median rank;
- query latency p50/p95/p99.

Full validation luôn tạo `validation/release-report.json` chứa validation status, namespace hashes,
expected/committed counts, sorted document-ID SHA và marker-set SHA.

Alias chỉ được chuyển khi:

1. import toàn bộ dataset, không dùng partial `max_shards`;
2. raw shard checksum còn đúng;
3. release report khớp namespace và R2 marker set;
4. exact sorted R2 document-ID hash bằng Elasticsearch `_id` hash;
5. concrete index count đúng;
6. atomic alias switch thành công.

Chỉ `_count` bằng nhau là chưa đủ vì hai tập ID khác nhau vẫn có thể cùng số lượng.

## 17. Guarantees và non-guarantees

### Pipeline bảo đảm

- không trộn output khác config/input/runtime khi thay đổi đúng qua config cell;
- không coi raw shard là complete nếu chưa có valid marker;
- không commit shard có frame `status=error`;
- bbox/màu/count của cùng instance không bị Elasticsearch cross-match;
- duplicate indexing không tạo document mới;
- alias không đổi trước exact-ID/checksum release gate.

### Pipeline không bảo đảm

- detector không bỏ sót vật thể bị che hoặc quá nhỏ;
- count không phải ground-truth trong crowd;
- palette color không mô tả hoàn hảo vật thể đa màu/ánh sáng phức tạp;
- prompt list hiện tại bao phủ mọi query tương lai;
- threshold/geometry weights đã tối ưu nếu chưa calibrate bằng GT.

## 18. Security và vận hành

Secrets bắt buộc được đọc từ Colab Secrets/environment:

```text
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_ACCOUNT_ID
R2_BUCKET
```

Tùy chọn gồm session token, public base URL và Elasticsearch credentials. Không print hoặc embed
secret trong notebook/output.

`R2/cloudflareR2_api.txt` là plaintext credential lịch sử và phải được rotate, xóa khỏi project lẫn
git history. Notebook không được đọc file này.

Khi Colab ngắt, chạy lại đúng notebook/config sẽ load cùng namespace và resume completion marker.
Không sửa `cfg` thủ công sau khi hash đã được tạo. Không đổi batch/shard size giữa một run đang chạy.

Với posthoc, chỉ chạy cell 9 sau khi cell 8 đã báo đủ toàn bộ shard. Kết quả trả về trong
`gdino_aug_cfg` và `gdino_aug_summary`; cell validation kế tiếp tự chọn `gdino_aug_cfg`. Nếu posthoc
bị ngắt, chạy lại chính cell 9: target plan và runtime identity deterministic nên marker v5.1 đã
commit được bỏ qua. Nếu Colab đã restart, chạy lại các cell setup/module/manifest nhưng không cần
load YOLOE; gán exact `Final R2 output` cũ vào `GDINO_SOURCE_PREFIX` để cell restore source
`config.json` từ R2.

## 19. Kiểm thử và bằng chứng còn thiếu

Regression/static suite hiện có 11 test cho:

- canonical dedup và source/class-specific count threshold;
- GDINO policy trigger;
- mask/Lab color contract và dependency pins;
- YOLO prompt dtype cache, warm-up và segmentation guard;
- ranked spatial/count queries;
- item-level bulk retry;
- completion-marker resume và runtime mismatch rejection.
- posthoc cell FP32, source-completion gate, v5.1 resume và không gọi lại YOLOE;
- posthoc anchor policy đọc `aliases` sau dedup và không đụng các trigger còn lại.

Test tĩnh không thay thế các bước cần chạy thật:

- full validation sau khi đủ 374 completion marker;
- contact-sheet review stratified theo genre/class;
- GT calibration cho count/color/geometry;
- Elasticsearch latency benchmark trên full index;
- kiểm tra distribution `fallback_status` và tỷ lệ detection bổ sung của Grounding DINO v5.1;
- đo batch size bằng benchmark kiểm soát thay vì suy từ một run duy nhất.

## 20. Gợi ý cấu trúc technical report

Một báo cáo module OD có thể dùng trực tiếp bố cục:

1. Problem statement và retrieval/count requirements.
2. Legacy baseline và gap analysis.
3. Model selection: YOLOE segmentation và policy-targeted Grounding DINO posthoc.
4. Genre vocabulary và canonical entity design.
5. R2/Colab/A100 execution architecture.
6. Geometry, color và count metadata algorithms.
7. Elasticsearch nested schema và ranking.
8. Reproducibility, resume và release integrity.
9. Experimental setup: GPU, batch, image size, dataset/shard count.
10. Results: throughput, detection/count/spatial metrics và query latency.
11. Runtime compatibility lessons: Pillow, CLIP, prompt/mask dtype và Grounding DINO health check.
12. Limitations, security và future work.

Các số liệu throughput trong mục 8.1 chỉ nên ghi là operational observation. Precision/recall/F1,
count MAE, spatial Recall@K và latency chỉ được báo như kết quả sau khi chạy evaluator/benchmark tương
ứng; không suy diễn từ `status=ok` hoặc throughput.

## 21. Hướng cải tiến ưu tiên

1. Đặt `SHARD_SIZE` vào namespace/config hash hoặc tạo explicit storage-layout hash.
2. Định lượng recall gain và chi phí của Grounding DINO posthoc v5.1 theo genre, class và trigger
   reason để tối ưu target policy.
3. Thu thập GT stratified theo genre, class, size, occlusion và color.
4. Calibrate confidence/count/color thresholds theo dữ liệu thật.
5. Benchmark batch 32/64/128/256 có kiểm soát trên cùng shard và tách GPU/I/O/postprocess time.
6. Thêm OCR cho text/logo và on-demand open-vocabulary worker cho class mới.
7. Đặt instance/crop embeddings ở index riêng nếu cần semantic similarity.
8. Rotate/xóa credential plaintext lịch sử.

## 22. Tham chiếu

- Ultralytics YOLOE: <https://docs.ultralytics.com/models/yoloe>
- Grounding DINO pinned revision:
  <https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb>
- Executable source: `build_nb.py`
- Generated notebook: `OD_Kaggle_Notebook.ipynb`
- Tests: `tests/test_od_pipeline_v4.py` (tên file lịch sử, đang test contract v5)
