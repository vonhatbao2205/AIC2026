# AIC 2026 — Keyframe batch 2 (M tin tức + N camera giao thông + S01 đua xe đạp) trên Cloudflare R2

Tài liệu bàn giao bộ keyframe của **data batch 2** cho các task downstream (embedding, OCR,
object detection, caption, retrieval backend/frontend). Bộ này được trích bằng ba notebook Colab
A100 và ghi thẳng lên cùng bucket R2 với keyframe InfoShot++ L21–L30.

> Snapshot ngày 2026-09-24. Toàn bộ số liệu trong tài liệu này lấy từ audit độc lập trên R2
> (`audit_batch2_r2_keyframes.py`), không lấy lại từ log của notebook.

## 1. Trạng thái bàn giao

| | M — tin tức | N — camera giao thông | S01 — đua xe đạp |
|---|---|---|---|
| Video | **304/304** (M01–M10) | **298/298** (N001–N100, 101 camera) | **12/12** (S01-V001–V012) |
| Thời lượng nguồn | 99,54 giờ | 49,15 giờ | 46,60 giờ |
| Keyframe | **209.111** | **44.572** | **621.117** |
| Mật độ | 0,584 keyframe/giây | 0,252 keyframe/giây | 3,703 keyframe/giây |
| Dung lượng JPEG | 34.491.134.216 byte (32,12 GiB) | 10.986.243.931 byte (10,23 GiB) | 107.376.353.122 byte (100,00 GiB) |
| Thuật toán | InfoShot++ nới lỏng cho tin tức | Chọn theo thay đổi tiền cảnh (camera cố định) | InfoShot++ v4.6 **nguyên bản** như L21–L30 |
| Pipeline version | `infoshootpp-aic-m-news-v1` | `keyframe-aic-n-traffic-v1` | `infoshootpp-aic-s01-cycling-v1` |
| Pipeline signature | `e6c789322adb` | `773e4b7465c2` | `db5918a110cf` |
| Selector | SigLIP2 `google/siglip2-base-patch16-224` | SigLIP2 `google/siglip2-base-patch16-224` | SigLIP2 `google/siglip2-base-patch16-224` |
| Metadata prefix | `manifest/infoshoot_m/` | `manifest/infoshoot_n/` | `manifest/infoshoot_s/` |
| `_SUCCESS.json` | 2026-09-23 15:01 UTC | 2026-09-23 08:13 UTC | 2026-09-24 05:37 UTC |
| Audit | **PASS** | **PASS** (1 cảnh báo, xem §11.3) | **PASS** (2 cảnh báo, xem §11.5, §11.7) |

Tổng batch 2: **614 video, 195,29 giờ, 874.800 keyframe, 142,36 GiB**. Riêng S01 chiếm 71 % số
keyframe và 70 % dung lượng.

Để so sánh mật độ:

| Bộ | Keyframe/giây |
|---|---:|
| K01–K20 InfoShot++ v4.6 (trước post-processing) | 3,51 |
| L21–L30 InfoShot++ v4.6 | 2,85 |
| Keyframe BTC trên L21–L30 | ~0,38 |
| **M batch 2** | **0,58** |
| **N batch 2** | **0,25** |
| **S01 batch 2** | **3,70** |

M và N được nới lỏng có chủ đích để tránh lặp lại tình trạng K quá dày. S01 thì ngược lại: đua xe đạp
cần độ chính xác nên giữ nguyên thuật toán v4.6 của L21–L30. S01 dày hơn cả L và K (trước post-processing),
chủ yếu do micro-burst và coverage 0,5 s (§9.4).

Các lưu ý dữ liệu cần xử lý ở downstream (chi tiết ở §11):

- `M10_V029` có 1.035 keyframe **đen hoàn toàn**, vì video nguồn chỉ có nội dung 18,5 phút đầu,
  73 phút còn lại là màu đen.
- Keyframe lưu là **full frame**. Ảnh M vẫn còn băng ticker đỏ, ảnh N vẫn còn banner camera; các vùng
  này chỉ bị bỏ khi chọn keyframe.
- Video nguồn `S01-V002` đã được chuyển mã AV1 → H.264 **sau khi** trích keyframe. Keyframe vẫn đúng
  (cùng số frame, cùng timeline), nhưng ETag nguồn trong registry khác object hiện tại (§11.5).
- S01 có 621.117 keyframe (100 GiB): cần tính trước chi phí embedding/OCR (§11.6).

## 2. Nguồn video

Video nằm trong bucket R2 `aic26-media`, được upload bởi
`CloudflareR2/AIC2026_R2_Video_Upload_Batch2_L21-L30_Colab.ipynb`.

```text
aic26-media/
├── Videos/
│   ├── Videos_M01/M01_V001.mp4 … Videos_M10/M10_V031.mp4      304 video, video_id dùng "_"
│   ├── Videos_N001/N001-V001.mp4 … Videos_N100/N100-V003.mp4   298 video, video_id dùng "-"
│   └── Videos_S01/S01-V001.mp4 … S01-V012.mp4                  12 video, video_id dùng "-"
└── Videos_AV1_Original/Videos_S01/S01-V001.mp4 … S01-V005.mp4  bản AV1 gốc được giữ lại (§2, S01)
```

- `video_id` giữ đúng tên file của BTC: `M01_V001` (gạch dưới), `N001-V001` và `S01-V001` (gạch ngang).
- **M:** HTV7 "60 giây", H.264 + AAC. Có 298 video 1920×1080, 4 video 1280×720
  (`M05_V004`, `M05_V005`, `M08_V004`, `M08_V025`) và 2 video 640×360 (`M03_V006`, `M03_V030`).
  FPS 24–30. Thời lượng 11,7–92,0 phút, trung vị 19,4 phút.
- **N:** 1920×1080, FPS thay đổi 10,9–30, không có audio, thường ~10 phút/đoạn. Mỗi camera thường có
  3 đoạn: sáng (~07–08h), trưa (~10–11h), tối (~18–19h). Ba đoạn `N043-V002/V003/V004` chỉ dài 16–25 giây.
- Metadata camera N (tên nút giao, ngày, giờ đầu/cuối đọc từ banner):
  `~/Projects/Media_Upload/batch2_content/batch2_camera_metadata.csv`.
- **S01:** đua xe đạp, 12 video 1920×1080, 30 FPS CFR, 2,4–5,6 giờ mỗi video (tổng 5.032.587 frame).
  Bản upload ban đầu gồm 11 AV1 + 1 H.264 (`S01-V012`). Trong lúc trích, người vận hành đã chuyển mã
  **tại chỗ** (ghi đè cùng key) `S01-V001`, `V002`, `V003` sang H.264. Bản AV1 gốc của V001–V005 được
  giữ ở `Videos_AV1_Original/`. Object đã chuyển mã mang metadata:
  `pipeline=aic2026-s01-av1-to-h264-v1`, `source-etag`, `source-size` (của bản AV1),
  `video-frames`, `origin` (key bản sao AV1).
  Chuyển mã giữ nguyên số frame, CFR 30 và mốc thời gian bắt đầu, nên `frame_idx`/`pts_time` đúng cho cả
  hai bản. Keyframe `S01-V001`, `V003`, `V012` được trích từ H.264; các video còn lại từ AV1 (§9.3).
- Public URL video: `https://video.baoencoder.site/Videos/Videos_<nhóm>/<video_id>.mp4`.
  CDN từng cache một phản hồi 404 cũ cho `M01_V001.mp4`, và video S01 đã chuyển mã có thể còn bản AV1 cũ
  trong cache. Khi đọc qua URL public, hãy thêm `?v=<ETag hiện tại>` (lấy từ S3 API hoặc
  `batch2_camera_metadata.csv`), hoặc đọc bằng S3 API.

## 3. Layout output trên R2

Bucket `aic26-infoshot-keyframes`, public base URL:

```text
https://pub-5010c807b73743ba82d6c40852ae9e6b.r2.dev
```

```text
aic26-infoshot-keyframes/
├── Keyframes/
│   ├── Keyframes_L21/ … Keyframes_L30/           # corpus L cũ, không bị đụng tới
│   ├── Keyframes_M01/M01_V001/001.jpg …           # batch 2 — tin tức
│   ├── Keyframes_N001/N001-V001/001.jpg …         # batch 2 — camera giao thông
│   └── Keyframes_S01/S01-V001/001.jpg …           # batch 2 — đua xe đạp
└── manifest/
    ├── media_manifest.csv, _SUCCESS.json          # của L, không bị sửa (vẫn là bản 2026-09-04)
    ├── infoshoot_m/
    │   ├── videos/<Mxx>/<video_id>/registry.csv | map.csv | stats.json
    │   ├── commits/<Mxx>/<video_id>.json         # ghi SAU CÙNG cho mỗi video
    │   ├── runs/m-m01-m10-x10/<timestamp>.csv
    │   └── final/
    │       ├── frame_registry.parquet             # 209.111 dòng
    │       ├── media_manifest_M.csv
    │       ├── map-keyframes.zip                  # map-keyframes/Mxx/<video_id>.csv
    │       ├── run_summary.csv, category_summary.csv, run_manifest.json
    │       └── _SUCCESS.json                      # ghi cuối cùng
    ├── infoshoot_n/                               # cùng cấu trúc, final/media_manifest_N.csv
    └── infoshoot_s/                               # cùng cấu trúc, final/media_manifest_S.csv
        ├── cache/S01/<video_id>/*.npz             # cache pass 1 tạm, xoá sau khi commit (§11.7)
        └── leases/S01/<video_id>.json             # giữ chỗ khi chạy nhiều runtime, xoá sau khi commit
```

- Tên JPEG trên R2 là `{n:03d}.jpg`, với `n` là thứ tự keyframe (bắt đầu từ 1) trong video, **không
  phải** `frame_idx`. Đây đúng là layout của keyframe L trong bucket này. `:03d` chỉ là độ rộng tối
  thiểu: video có hơn 999 keyframe sẽ có `1000.jpg`, `1710.jpg`; S01 lên tới `75123.jpg`.
- JPEG: quality 92, cạnh dài tối đa 1.280 px. Ảnh M phần lớn là 1280×720, riêng 1.182 frame
  640×360 (từ 2 video nguồn 640×360; notebook không upscale). Ảnh N và S01 toàn bộ là 1280×720.
  Ảnh S01 trung bình 169 KiB.
- Header: `Content-Type: image/jpeg`, `Cache-Control: public, max-age=86400`. Không dùng
  `immutable`, vì nếu chạy lại với tham số khác thì `001.jpg` có thể đổi nội dung. CORS trả
  `Access-Control-Allow-Origin: *`.
- Các file `final/` đều đọc được qua public URL, ví dụ
  `https://pub-5010c807b73743ba82d6c40852ae9e6b.r2.dev/manifest/infoshoot_s/final/frame_registry.parquet`.

## 4. Định danh và contract thời gian

| Trường | Ví dụ M | Ví dụ N | Ví dụ S01 | Dùng để |
|---|---|---|---|---|
| `frame_id` | `M03_V005@f00000009` | `N001-V001@f00000000` | `S01-V001@f00000002` | **Khoá chính** cho embedding/OCR/index |
| `video_id` | `M03_V005` | `N001-V001` | `S01-V001` | Lọc, phát video, chia shard |
| `category` / `group` | `M03` | `N001` | `S01` | Thư mục R2, chia session |
| `frame_idx` | `9` | `0` | `2` | Chỉ số frame theo thứ tự decode trong video gốc |
| `pts_time` | `0.36` | `0.2` | `0.066667` | Thời điểm (giây) để seek/phát video |
| `fps` | `25.0` | `24.99965871` | `30.0` | FPS trung bình từ ffprobe (N có VFR) |
| `n` / `keyframe_n` | `1` | `1` | `1` | Thứ tự keyframe; trùng với tên file R2 |
| `submit_keyframe_id` | `M03/M03_V005/001` | `N001/N001-V001/001` | `S01/S01-V001/001` | Tương thích định dạng submit/app cũ |
| `r2_key` | `Keyframes/Keyframes_M03/M03_V005/001.jpg` | `Keyframes/Keyframes_N001/N001-V001/001.jpg` | `Keyframes/Keyframes_S01/S01-V001/001.jpg` | Đọc ảnh |

Quy tắc:

- Dùng `frame_id` làm khoá join. `n` chỉ ổn định trong đúng snapshot này (pipeline signature ở §1).
- `frame_idx` là chỉ số frame theo thứ tự decode của PyAV, giống contract của K/L. Video N có timestamp
  không đều, nên không suy `frame_idx` từ `pts_time × fps`; luôn dùng cặp giá trị lưu trong registry.
- Video N được remux từ `.mov` với `-copyts`, nên `pts_time` của frame đầu thường là **0,2 s**, không
  phải 0.
- Nếu một video từng được chạy lại, số keyframe và `n` của video đó có thể thay đổi. Commit trên R2
  giữ `pipeline_signature` và ETag của video nguồn để phát hiện chuyện này.

## 5. Các file metadata

### 5.1 `final/frame_registry.parquet`

Mỗi dòng ứng với một JPEG trên R2. M, N và S01 có cùng 36 cột:

| Nhóm | Cột |
|---|---|
| Định danh | `n`, `frame_id`, `video_id`, `category`, `frame_idx`, `pts_time`, `timestamp`, `fps` |
| Lý do chọn | `segment_id`, `sources` (các nguồn nối bằng ký tự `\|`), `priority`, `scores_json` |
| Chất lượng/hash | `quality` (0–1), `phash_hex`, `text_hash_hex`, `text_density`, `caption_density` |
| Tên/ID ảnh | `keyframe_name`, `source_frame_name` (`f{frame_idx:08d}.jpg`), `submit_keyframe_id`, `legacy_keyframe_id` |
| Vị trí ảnh | `r2_bucket`, `r2_key`, `public_url`, `size_bytes`, `image_width`, `image_height`, `image_view` |
| Vùng phân tích | `analysis_crop_top_frac`, `analysis_crop_bottom_frac`, `analysis_mask_boxes` |
| Nguồn video | `source_video_bucket`, `source_video_key`, `source_video_size`, `source_video_etag` |
| Phiên bản | `pipeline_signature` |

`quality`, pHash và text hash được tính trên **analysis view**. Với M và N, đó là ảnh đã bỏ ticker/banner
(§6.1, §7.1), không phải JPEG full frame. S01 không crop (`analysis_crop_*` = 0, `analysis_mask_boxes` = `[]`).

`source_video_size`/`source_video_etag` là của **đúng bản video đã decode** khi trích. Với `S01-V002`, đó
là bản AV1, khác object hiện tại (§11.5).

### 5.2 `final/media_manifest_M.csv`, `final/media_manifest_N.csv`, `final/media_manifest_S.csv`

7 cột đầu trùng thứ tự với `manifest/media_manifest.csv` của L, là các cột frontend đang đọc:

```text
kind,group,video_id,frame_name,frame_idx,source_path,r2_key
```

Tiếp theo là `category, keyframe_n, keyframe_name, submit_keyframe_id, frame_id, pts_time, fps,
public_url, size_bytes, r2_bucket_uri, pipeline_signature`.

Notebook **không** ghi vào `manifest/media_manifest.csv` của L. Muốn frontend hiện batch 2, hãy nối 7 cột
đầu của ba file này vào manifest chung ở bước deploy. Nên làm trên một bản sao rồi mới thay file chung.

### 5.3 `final/map-keyframes.zip`

Gồm `map-keyframes/<nhóm>/<video_id>.csv` với schema legacy bốn cột, đúng thứ tự:

```text
n,pts_time,fps,frame_idx
```

M có 304 file, N có 298 file, S01 có 12 file. Số dòng mỗi file bằng số JPEG của video đó.

### 5.4 Thống kê và commit

- `final/run_summary.csv`: một dòng mỗi video, gồm codec/độ phân giải/số frame gốc, thời lượng, số probe,
  số event, thông số dedup/coverage, `retained_fps`, thời gian từng stage (`t_pass1_sec`, `t_refine_sec`,
  `t_materialize_sec`...).
- `final/category_summary.csv`, `final/run_manifest.json` (có toàn bộ `CONFIG` đã chạy),
  `final/_SUCCESS.json`.
- `commits/<nhóm>/<video_id>.json`: `binding` (pipeline signature + key/size/ETag video nguồn),
  `frame_count`, `file_sizes[]`, `file_md5[]`, `registry_sha256`. MD5 của từng JPEG trùng với ETag
  R2, nên có thể kiểm toàn vẹn mà không cần tải ảnh về.

## 6. Pipeline M — tin tức

Notebook: `AIC2026_InfoShotPP_M_News_R2_Colab.ipynb`.

### 6.1 Analysis view

Các vùng dưới đây chỉ bị bỏ khi **chọn** keyframe. JPEG lưu lên R2 vẫn là full frame.

- Bỏ phần dưới **10,2 %** khung hình. Băng ticker đỏ chạy chữ liên tục nằm ở 90,1–95,8 % chiều cao;
  nó làm sentinel và SigLIP2 tưởng khung hình đang đổi, trong khi nội dung thường là tin khác.
  Lower-third tiêu đề nằm ngay phía trên ticker vẫn được giữ.
- Che vùng logo HTV7 + đồng hồ (nhảy mỗi giây) ở `[0.800, 0.030, 0.955, 0.175]` theo tỉ lệ khung hình.

### 6.2 Luồng xử lý

```text
Video M (1 lượt decode, frame low-res của analysis view giữ trong RAM ≤ 12 GiB/video)
  ├─ native sentinel GPU: global/local/broadcast-text/localized-text change
  ├─ Farneback motion 5 FPS
  └─ SigLIP2 probe 10 FPS
        ↓
GLRT segment (≥ 0,5 s, ≤ 30 s) → COMMON + UNIQUE
        ↓
Sentinel event (NMS 0,8 s) + semantic/boundary/short-shot → gộp event trong 0,6 s
        ↓
Micro-burst native FPS: 1 frame/event, phạt frame wipe/dissolve,
encode SigLIP2 cho chính frame được chọn, thay frame probe tương ứng
        ↓
Dedup theo nội dung (≤ 3 s): cosine SigLIP2 + chữ ký lower-third không đổi
        ↓
Coverage: gap ≤ 2 s khi hình đổi; tỉa frame coverage nếu cảnh tĩnh (gap ≤ 6 s)
        ↓
Decode lại frame chính xác → JPEG 1280 px → upload R2 → commit
```

### 6.3 Tham số khác với InfoShot++ v4.6 (K/L/S01)

| Nhóm | v4.6 (K/L/S01) | M batch 2 |
|---|---|---|
| Coverage | gap ≤ 0,5 s ở mọi nơi | gap ≤ 2 s khi hình đổi, ≤ 6 s khi cảnh tĩnh |
| UNIQUE | luôn giữ primary UNIQUE | chỉ khi vượt ngưỡng (z ≥ 3, score ≥ 0,6), ≤ 2/segment, bỏ nếu cosine với COMMON > 0,93 |
| Sentinel | z 3–3,5, NMS 0,25 s | z 3,5–4,5, raw floor cao hơn, NMS 0,8 s |
| Micro-burst | tối đa 2 frame/event + giữ trigger thô | 1 frame/event, trigger thô chỉ khi burst rỗng |
| Dedup | ≤ 1 s, cùng segment, cosine ≥ 0,985, pHash ≤ 4 | ≤ 3 s, cosine ≥ 0,94 (khác segment 0,965), caption change ≤ 0,30 |
| Short shot | P0 | P1 |

Guard caption: lưới 8×32 ô trên dải lower-third. Một ô tính là đổi khi mật độ cạnh lệch ≥ 0,25; hai frame
khác nhau nếu hơn 30 % số ô đổi. Ngưỡng này được hiệu chỉnh trên M03_V005 (1.840 cặp frame cùng cảnh):
nhiễu do chuyển động có p95 = 0,25, còn khi tiêu đề/phụ đề/nhãn tên xuất hiện thì lệch 0,34–0,47.

## 7. Pipeline N — camera giao thông

Notebook: `AIC2026_Keyframe_N_Traffic_R2_Colab.ipynb`.

Video N là camera cố định, không có cắt cảnh, nên TransNetV2 và bước chia shot GLRT không dùng được.
Nội dung đáng tìm là xe cộ và người đi qua trên nền tĩnh.

### 7.1 Analysis view

Bỏ **4 %** trên cùng: banner tên camera + ngày + đồng hồ (0–3,5 % chiều cao). JPEG lưu vẫn còn banner,
nên OCR đọc được tên camera và giờ.

### 7.2 Luồng xử lý

```text
Video N: decode mọi frame (giữ frame_idx), chỉ phân tích probe 5 FPS
  └─ mỗi probe: lưới độ sáng 18×32 + độ nét + hash + SigLIP2
        ↓
Quét thời gian, so với KEYFRAME GIỮ GẦN NHẤT (không phải frame liền trước):
  ≥ 15 % số ô lệch ≥ 16 mức xám (sau khi bù median độ sáng từng frame) → keyframe mới
  · cách keyframe trước ≥ 2,5 s
  · heartbeat nếu 15 s không có thay đổi (đèn đỏ, đường vắng)
  · trong 0,6 s sau điểm đổi, chọn probe nét nhất
        ↓
Novelty SigLIP2: frame khác hẳn trung bình ±15 s (z ≥ 4, novelty ≥ 0,05, NMS 5 s),
bỏ nếu trong ±1,5 s đã có keyframe gần như giống hệt
        ↓
Decode lại frame chính xác → JPEG 1280 px → upload R2 → commit
```

Nguồn (`sources`) của keyframe N: `traffic_start`, `traffic_change`, `traffic_heartbeat`,
`traffic_novelty`.

Khi hiệu chỉnh, nhiễu cảm biến theo từng ô rất thấp (trung vị ~1,5 mức xám). Các video tối/mưa có mật độ
cao hơn là do xe máy dày đặc thật, không phải do nhiễu.

## 8. Pipeline S01 — đua xe đạp

Notebook: `AIC2026_InfoShotPP_S01_Cycling_R2_Colab.ipynb` (profile `sports`).

Đua xe đạp cần độ chính xác, nên S01 **không nới lỏng** như M/N mà
dùng đúng InfoShot++ v4.6 đã chạy cho L21–L30. Khác biệt duy nhất là selector: SigLIP → SigLIP2.

- **Analysis view:** tắt, phân tích trên full frame để giữ đúng hành vi v4.6 của L21–L30.
- **Luồng xử lý:** giống sơ đồ §6.2, nhưng giữ nguyên tham số v4.6 ở cột trái bảng §6.3: sentinel z 3–3,5,
  NMS 0,25 s; micro-burst tối đa 2 frame/event và giữ trigger thô; luôn giữ primary UNIQUE; dedup chỉ trong
  1 s, cùng segment, cosine ≥ 0,985 và pHash ≤ 4; coverage gap ≤ 0,5 s ở mọi nơi, không tỉa cảnh tĩnh;
  short shot P0.
- **Kiểm chứng tương đương:** trên cùng dữ liệu pass 1 của một clip S01 10 phút, bộ chọn của notebook cho
  ra đúng 2.647/2.647 `frame_idx` như code v4.6 gốc trong notebook Kaggle.
- **Video dài 2,4–5,6 giờ:**
  - Cache pass 1 được đẩy lên R2 (`infoshoot_s/cache/`), nên runtime Colab bị ngắt giữa video không phải
    decode lại pass 1. Cache bị xoá sau khi commit.
  - Mỗi video được "giữ chỗ" (`infoshoot_s/leases/`), nên có thể chạy nhiều runtime cùng lúc mà không
    trích trùng video.
  - Frame low-res không giữ được trong RAM (vượt trần 12 GiB/video), nên micro-burst phải decode lại
    video. Đây là lý do `t_refine_sec` lớn (§9.5).
- **Chịu được chuyển mã nguồn:**
  - Trước khi tải, notebook HEAD lại object để lấy đúng bản hiện tại, và kiểm lại ETag sau khi tải.
  - Commit cũ vẫn được giữ nếu object hiện tại là bản chuyển mã từ đúng bản đã trích: `source-etag` và
    `source-size` trỏ về bản đó, và `video-frames` bằng số frame đã decode.

## 9. Thống kê

### 9.1 M theo nhóm

| Nhóm | Video | Giờ | Keyframe | Keyframe/giây | GiB |
|---|---:|---:|---:|---:|---:|
| M01 | 29 | 8,64 | 18.012 | 0,579 | 2,70 |
| M02 | 31 | 8,98 | 18.836 | 0,583 | 2,96 |
| M03 | 31 | 8,52 | 18.212 | 0,594 | 2,66 |
| M04 | 28 | 8,90 | 19.265 | 0,601 | 3,03 |
| M05 | 31 | 9,52 | 20.032 | 0,585 | 3,08 |
| M06 | 30 | 9,52 | 20.204 | 0,590 | 3,17 |
| M07 | 32 | 10,55 | 22.535 | 0,593 | 3,43 |
| M08 | 30 | 11,25 | 23.856 | 0,589 | 3,67 |
| M09 | 31 | 11,57 | 24.050 | 0,577 | 3,78 |
| M10 | 31 | 12,09 | 24.109 | 0,554 | 3,65 |
| **Tổng** | **304** | **99,54** | **209.111** | **0,584** | **32,12** |

Theo video: trung vị 684 keyframe (thấp nhất 432, cao nhất 1.710 là `M10_V029`, xem §11.1); mật độ
trung vị 0,586/giây. Gap lớn nhất giữa hai keyframe liên tiếp là 6,0 s.

### 9.2 N theo dải camera

| Dải | Camera | Video | Giờ | Keyframe | Keyframe/giây |
|---|---:|---:|---:|---:|---:|
| N001–N010 | 10 | 30 | 5,01 | 4.716 | 0,261 |
| N011–N020 | 10 | 32 | 5,33 | 5.445 | 0,284 |
| N021–N030 | 10 | 30 | 5,00 | 4.365 | 0,242 |
| N031–N040 | 10 | 30 | 5,01 | 4.039 | 0,224 |
| N041–N050 | 10 | 31 | 4,57 | 3.823 | 0,233 |
| N051–N060 | 10 | 28 | 4,67 | 3.716 | 0,221 |
| N061–N070 | 10 | 27 | 4,51 | 3.787 | 0,233 |
| N071–N080 | 10 | 30 | 5,04 | 5.671 | 0,313 |
| N081–N090 | 10 | 30 | 5,00 | 4.438 | 0,247 |
| N091–N100 | 10 | 30 | 5,01 | 4.572 | 0,253 |
| **Tổng** | **100** | **298** | **49,15** | **44.572** | **0,252** |

Theo khung giờ trên banner:

| Khung giờ | Video | Keyframe | Keyframe/giây trung bình |
|---|---:|---:|---:|
| Sáng (trước 9h)¹ | 99 | 13.698 | 0,230 |
| Trưa (10–11h) | 101 | 14.805 | 0,253 |
| Tối (18–19h) | 98 | 16.069 | 0,272 |

¹ Gồm cả `N073-V002`, banner ghi 01:00–01:10.

Theo video: trung vị 158 keyframe, mật độ 0,067–0,378/giây. Ít keyframe nhất là ba đoạn ngắn
`N043-V004` (16,7 giây, 3 keyframe), `N043-V003` (21,9 giây, 6) và `N043-V002` (24,4 giây, 7).
Thống kê từng camera nằm trong `final/category_summary.csv`.

### 9.3 S01 theo video

| Video | Codec lúc trích | Giờ | Frame gốc | Keyframe | Keyframe/giây | GiB |
|---|---|---:|---:|---:|---:|---:|
| S01-V001 | H.264 | 2,76 | 297.899 | 36.736 | 3,70 | 5,10 |
| S01-V002 | AV1 | 2,82 | 304.650 | 38.657 | 3,81 | 6,05 |
| S01-V003 | H.264 | 2,43 | 262.950 | 29.747 | 3,39 | 4,86 |
| S01-V004 | AV1 | 3,94 | 425.400 | 51.461 | 3,63 | 7,67 |
| S01-V005 | AV1 | 3,94 | 425.736 | 55.280 | 3,90 | 8,57 |
| S01-V006 | AV1 | 2,70 | 291.748 | 34.633 | 3,56 | 6,24 |
| S01-V007 | AV1 | 5,62 | 607.382 | 75.123 | 3,71 | 11,73 |
| S01-V008 | AV1 | 4,80 | 518.534 | 65.732 | 3,80 | 10,61 |
| S01-V009 | AV1 | 3,72 | 401.286 | 45.638 | 3,41 | 8,46 |
| S01-V010 | AV1 | 4,24 | 457.950 | 56.937 | 3,73 | 9,48 |
| S01-V011 | AV1 | 4,68 | 505.202 | 62.480 | 3,71 | 9,96 |
| S01-V012 | H.264 | 4,94 | 533.850 | 68.693 | 3,86 | 11,28 |
| **Tổng** | | **46,60** | **5.032.587** | **621.117** | **3,70** | **100,00** |

Keyframe chiếm 12,3 % số frame gốc. Mật độ theo video khá đều (3,39–3,90/giây). Khoảng cách trung vị giữa
hai keyframe là 0,27–0,37 s, gap lớn nhất đúng 0,5 s ở cả 12 video (giới hạn coverage v4.6).

### 9.4 Phân bố nguồn

Một keyframe có thể mang nhiều nguồn, nên tổng lớn hơn số keyframe.

| M | Keyframe | | N | Keyframe | | S01 | Keyframe |
|---|---:|---|---|---:|---|---|---:|
| `microburst` | 154.509 | | `traffic_change` | 40.906 | | `microburst` | 245.526 |
| `text_event` | 109.125 | | `traffic_heartbeat` | 2.000 | | `coverage_floor` | 212.980 |
| `native_local` | 107.112 | | `traffic_novelty` | 1.422 | | `text_event` | 178.919 |
| `text_local_event` | 100.843 | | `traffic_start` | 298 | | `semantic_unique_primary` | 173.496 |
| `native_global` | 99.756 | | | | | `native_global` | 158.531 |
| `semantic_boundary` | 95.510 | | | | | `common` | 154.789 |
| `common` | 79.570 | | | | | `native_local` | 152.372 |
| `motion_event` | 73.536 | | | | | `text_local_event` | 112.215 |
| `semantic_unique` | 40.119 | | | | | `short_shot` | 103.777 |
| `coverage_floor` | 35.030 | | | | | `semantic_boundary` | 69.396 |
| `short_shot` | 2.501 | | | | | `motion_event` | 51.720 |
| | | | | | | `semantic_unique` | 47.068 |

Ở tin tức, text/native/motion event thường cùng bắn tại một điểm cắt cảnh, nên các nguồn này chồng lên
nhau nhiều. Ở S01, 212.980 keyframe (34,3 %) **chỉ** có nguồn `coverage_floor`: chúng tồn tại để giữ
gap ≤ 0,5 s, không phải vì có sự kiện. Nếu downstream cần giảm khối lượng, đây là nhóm nên xét trước (§11.6).

### 9.5 Thời gian chạy trên Colab A100

Mỗi profile chạy trên một runtime A100 riêng. Trong mỗi runtime, số video xử lý song song do notebook
tự chọn theo số CPU (`parallel_videos=None`, thường là 3).

| | M | N | S01 |
|---|---|---|---|
| Session | `m-m01-m10-x10` | `n-n001-n100-x100` | `s-s01` |
| Commit đầu → cuối (UTC) | 23/09 04:56 → 14:59 (**10,0 giờ**) | 23/09 04:57 → 08:10 (**3,2 giờ**) | 23/09 19:18 → 24/09 05:37 (**10,3 giờ**) |
| Thời gian trung bình/video | 286 s (pass 1: 144, micro-burst: 46, materialize: 95) | 115 s (pass 1: 62, materialize: 52) | 9.960 s (pass 1: 3.026, micro-burst: 3.845, materialize: 3.018) |
| RAM frame cache | 303/304 video (`M10_V029` vượt trần 12 GiB) | không dùng | 0/12 (video quá dài) |

Ghi chú S01:

- Lượt đầu commit 10 video. `S01-V001` và `S01-V003` lỗi tải vì object bị chuyển mã sang H.264 ngay trong
  lúc chạy. Notebook đã được sửa (§8), và lượt sau trích hai video này từ bản H.264.
- Decode H.264 nhanh hơn AV1 rõ rệt. V001/V003 (H.264) mất ~1.800 s xử lý cho mỗi giờ video, còn các video
  AV1 mất 2.300–2.960 s. Hai lượt không hoàn toàn so được với nhau: lượt V001/V003 chỉ chạy 2 video song song
  thay vì 3.

## 10. Kiểm chứng (audit độc lập)

Script chỉ đọc: [`audit_batch2_r2_keyframes.py`](audit_batch2_r2_keyframes.py). Script dùng S3 API với
credential trong `CloudflareR2/cloudflareR2_api.txt` (không in ra), chỉ gọi list/head/get.

Các điểm đã kiểm, cho từng profile M, N và S01:

1. Tải `final/*` và đọc toàn bộ commit JSON. Mọi commit đều `status=verified` và có cùng một pipeline
   signature, trùng với `_SUCCESS.json`.
2. Liệt kê video nguồn trong `aic26-media`: đủ 304 M, 298 N và 12 S01. Mỗi video có đúng một commit.
   Key/size/ETag trong commit khớp với object hiện tại, trừ `S01-V002` (nguồn đã chuyển mã sau khi trích).
   Video này chỉ được chấp nhận khi thoả cả ba điều kiện:
   - metadata `source-etag`/`source-size` của object hiện tại trỏ đúng bản AV1 đã trích;
   - `video-frames` = 304.650 = số frame đã decode;
   - bản sao ở `origin` (`Videos_AV1_Original/…`) vẫn còn đúng ETag đó.
3. Liệt kê **từng JPEG** trên R2: 209.111 JPEG M, 44.572 JPEG N và 621.117 JPEG S01. Mỗi video có đủ
   `001..N`, không thiếu, không thừa, không có object lạc. Size và **ETag = MD5** của từng file khớp với commit.
4. Registry: không trùng `frame_id`; `n` liên tục; `frame_idx` tăng nghiêm ngặt; `r2_key` đúng layout;
   `size_bytes` khớp commit; `source_video_etag` khớp ETag nguồn ghi trong commit.
5. `map-keyframes.zip`: đủ file, đúng header 4 cột, khớp registry theo từng video.
   `media_manifest_*.csv`: đúng 7 cột legacy, tập `r2_key` bằng đúng tập object trên R2.
6. Không còn object `errors/` hay `_preflight/` sót lại. Object `cache/`/`leases/` sót lại được báo là cảnh
   báo: S01 còn 4 file cache của clip smoke (§11.7).
7. Gap thời gian:
   - M lớn nhất 6,0 s (giới hạn 6,05 s).
   - S01 lớn nhất 0,5 s (giới hạn 0,52 s).
   - N lớn nhất 16,24 s; đây là cảnh báo, không phải lỗi, xem §11.3.
8. Public URL: mẫu ngẫu nhiên 8 JPEG mỗi profile đều trả HTTP 206, `image/jpeg`, CORS `*`, size đúng.
9. Corpus L không bị đụng tới: `manifest/_SUCCESS.json` và `manifest/media_manifest.csv` vẫn mang
   thời điểm sửa 2026-09-04. Bucket có đúng các nhóm keyframe: 10 L, 10 M, 100 N, 1 S.

Kết quả: `audit_ok = true` cho cả M, N và S01.

Artifact local (metadata, không có ảnh):

```text
keyframe_batch2/
├── infoshoot_m/final/   # bản sao final/* của M trên R2
├── infoshoot_n/final/   # bản sao final/* của N trên R2
├── infoshoot_s/final/   # bản sao final/* của S01 trên R2
└── audit/
    ├── audit_report.json   # gồm cả danh sách nguồn đã chuyển mã (transcoded_sources)
    ├── M_category_statistics.csv, M_video_statistics.csv, M_commits.csv
    ├── N_category_statistics.csv, N_video_statistics.csv, N_commits.csv
    └── S_category_statistics.csv, S_video_statistics.csv, S_commits.csv
```

Chạy lại audit (cả ba profile, ~10 phút vì phải liệt kê ~875k object):

```bash
uv run --with boto3 --with pandas --with pyarrow --with requests python audit_batch2_r2_keyframes.py
```

## 11. Vấn đề đã biết và khuyến nghị

### 11.1 `M10_V029` có đuôi đen 73 phút

- Video nguồn `Videos/Videos_M10/M10_V029.mp4` dài 5.523 giây, nhưng từ ~1.110 giây đến hết là
  màu đen tuyệt đối. Đây là bản BTC giữ nguyên byte (`convert-action: none`).
- Keyframe `n = 676..1710` của video này (1.035 frame, `pts_time ≥ 1109,6`) đều đen, `quality < 0,05`.
  Frame `n = 1..675` bình thường.
- **Khuyến nghị:** khi embedding/OCR/index, bỏ `M10_V029` với `n ≥ 676`, hoặc bỏ mọi frame
  `quality < 0,05` (§11.2). Nếu cần sạch hẳn trên R2, có thể cắt video này về 675 keyframe. Vì các frame
  đen nằm liền ở cuối, việc cắt không đánh số lại frame nào; chỉ cần xoá object 676–1710 và cập nhật
  registry/map/manifest/commit của video này. Việc này **chưa làm**, vì là thao tác xoá trên bucket.

### 11.2 Frame đen/trống do chuyển cảnh

- **M:** ngoài `M10_V029`, còn 294 keyframe M có `quality < 0,05`, rải trên 165 video, mỗi video 1–7 frame.
  Chúng chủ yếu là khoảnh khắc fade-to-black giữa hai tin (nguồn `semantic_unique`, `microburst`,
  `text_local_event`).
- **S01:** có 141 keyframe (0,02 %) dưới ngưỡng này, rải trên cả 12 video. Đó là các đoạn fade/đen ngắn;
  đoạn dài nhất là 17 keyframe liên tiếp ở `S01-V012`, `n = 42951..42967`.
- **N:** không có frame nào dưới ngưỡng này.

Filter gợi ý cho downstream:

```python
usable = registry[registry.quality >= 0.05]
```

### 11.3 `N030-V003` có một gap 16,24 s

Heartbeat N được thiết kế tối đa khoảng 15 + 0,6 + 0,2 = 15,8 s, và 138 video N có gap lớn nhất
15,6–15,8 s, đúng thiết kế. Riêng camera `N030-V003` (~12 fps) tự rớt frame: sau mốc 190,375 s, video
nguồn không có frame nào trong 0,88 s. Vì vậy heartbeat buộc phải lấy frame có thật tiếp theo ở 191,86 s.
Đây là thuộc tính của video nguồn, không mất dữ liệu nào.

### 11.4 Chữ không liên quan trong ảnh lưu

- **M:** ảnh còn băng ticker ở 90,1–95,8 % chiều cao. Chữ trên ticker thường nói về tin khác với hình
  đang chiếu. Khi OCR để retrieval, nên cắt theo `analysis_crop_bottom_frac` (0,102) hoặc gắn nhãn riêng
  cho text trong vùng ticker. Logo và đồng hồ HTV7 nằm ở vùng `analysis_mask_boxes`.
- **N:** banner (4 % trên cùng) chứa tên nút giao, ngày và giờ. OCR banner hữu ích làm metadata
  (camera, thời điểm thật), nhưng nên tách khỏi text của cảnh. Tên camera và giờ cũng đã có sẵn trong
  `batch2_camera_metadata.csv`.

### 11.5 `S01-V002`: nguồn được chuyển mã sau khi trích

- Keyframe `S01-V002` được trích từ bản AV1: ETag `a47585a807a2f5bc0e049361c1355089-105`,
  3.490.561.917 byte. Registry (`source_video_etag`) và commit ghi đúng bản này.
- Sau đó object `Videos/Videos_S01/S01-V002.mp4` bị ghi đè bằng bản H.264: ETag
  `dccebda912c83e761862159d0b339293-438`, 14.670.747.424 byte.
- Bản AV1 vẫn còn ở `Videos_AV1_Original/Videos_S01/S01-V002.mp4`.
- Hai bản có cùng 304.650 frame, CFR 30, cùng mốc bắt đầu. Vì vậy `frame_idx`/`pts_time` của keyframe
  seek đúng trên bản H.264 hiện tại. **Không cần trích lại.**
- Nếu downstream so ETag registry với object hiện tại, hãy chấp nhận trường hợp này bằng metadata
  `source-etag` của object. `transcoded_sources` trong `audit_report.json` ghi đủ bằng chứng.
- `S01-V004`/`V005` đã có bản sao AV1 trong `Videos_AV1_Original/`, nhưng tại thời điểm audit, key chính vẫn
  là AV1. Nếu sau này chúng được chuyển mã bằng cùng pipeline, áp dụng đúng quy tắc trên.

### 11.6 Khối lượng S01

S01 chiếm 621.117 keyframe (100 GiB), gấp ~2,4 lần M + N cộng lại, vì dùng thuật toán v4.6 không nới lỏng.
Khi lên kế hoạch embedding/OCR, hãy tính theo con số này.

Nếu cần giảm khối lượng, hãy làm post-processing trên dataset riêng, **không xoá trên R2**. Có thể làm
tương tự Pass 2 của K, ưu tiên xét 212.980 keyframe chỉ có nguồn `coverage_floor` (§9.4).

### 11.7 Cache smoke sót lại trên R2

Còn 4 object ở `manifest/infoshoot_s/cache/S01/{S01-V004,S01-V012}/{probe,sentinel}.npz` (~25 MB). Đây là
cache pass 1 của **clip smoke 600 giây**, được ghi vào key cache của video đầy đủ.

- Không ảnh hưởng dữ liệu: chữ ký cache gồm kích thước file và số frame, nên cache này không bao giờ được
  dùng cho video đầy đủ.
- Builder đã được sửa để clip smoke không ghi cache lên R2 nữa.
- 4 object này **chưa xoá**, vì là thao tác xoá trên bucket.

### 11.8 Khác

- Frontend chưa hiện batch 2 cho tới khi manifest chung được nối thêm (§5.2).
- Nếu sau này chạy lại một video với tham số khác, commit mới sẽ ghi đè `001.jpg…` và xoá JPEG thừa trong
  đúng thư mục video đó. Do `max-age=86400`, CDN có thể giữ ảnh cũ tới 1 ngày; nên purge cache nếu cần
  hiển thị ngay.

## 12. Hướng dẫn cho task downstream

### 12.1 Nguồn sự thật

Work queue cho embedding/OCR là:

```text
manifest/infoshoot_m/final/frame_registry.parquet   (209.111 dòng)
manifest/infoshoot_n/final/frame_registry.parquet   (44.572 dòng)
manifest/infoshoot_s/final/frame_registry.parquet   (621.117 dòng)
```

Mỗi dòng tương ứng đúng một JPEG. Không tự suy danh sách ảnh từ việc list bucket.

### 12.2 Đọc metadata và ảnh

Không cần credential, đọc thẳng qua public URL:

```python
import io, requests, pandas as pd
from PIL import Image

BASE = "https://pub-5010c807b73743ba82d6c40852ae9e6b.r2.dev"
reg = pd.read_parquet(f"{BASE}/manifest/infoshoot_m/final/frame_registry.parquet")
reg = reg[reg.quality >= 0.05]                                            # bỏ frame đen (§11.1–11.2)

row = reg.iloc[0]
image = Image.open(io.BytesIO(requests.get(row.public_url, timeout=60).content)).convert("RGB")
```

Khi tải số lượng lớn (hàng trăm nghìn ảnh, nhất là S01), `r2.dev` bị giới hạn tốc độ. Nên dùng S3 API của
R2 (boto3 `get_object`, ≥ 32 luồng) với credential chỉ đọc, và đọc theo `r2_key`:

```python
s3.get_object(Bucket="aic26-infoshot-keyframes", Key=row.r2_key)["Body"].read()
```

Muốn kiểm toàn vẹn sau khi tải: `hashlib.md5(data).hexdigest()` phải bằng ETag của object, hoặc bằng
`file_md5[n-1]` trong commit.

### 12.3 Output downstream

Không đổi tên hay xoá keyframe nguồn. Ghi output thành dataset riêng và join ngược bằng `frame_id`:

```text
embeddings/batch2/{M,N,S01}/*.parquet
ocr/batch2/{M,N,S01}/*.parquet
```

Cột tối thiểu nên giữ:

```text
frame_id, video_id, category, frame_idx, pts_time, r2_key, pipeline_signature
```

Giữ `pipeline_signature` để không trộn vector/OCR giữa hai snapshot, nếu sau này có video bị chạy lại.
Với S01 (621k frame), nên checkpoint theo `video_id` để resume được.

## 13. Tái lập

- Builder: [`build_infoshootpp_batch2_r2_colab.py`](build_infoshootpp_batch2_r2_colab.py) sinh ra
  ba notebook từ code dùng chung:
  - [`AIC2026_InfoShotPP_M_News_R2_Colab.ipynb`](AIC2026_InfoShotPP_M_News_R2_Colab.ipynb)
  - [`AIC2026_Keyframe_N_Traffic_R2_Colab.ipynb`](AIC2026_Keyframe_N_Traffic_R2_Colab.ipynb)
  - [`AIC2026_InfoShotPP_S01_Cycling_R2_Colab.ipynb`](AIC2026_InfoShotPP_S01_Cycling_R2_Colab.ipynb)
- Cấu hình đã chạy nằm trong `final/run_manifest.json` của mỗi profile, khớp với mặc định của notebook.
- Chạy lại toàn bộ bằng `run_mode="batch"` với cùng cấu hình: các video đã có commit khớp signature và
  ETag nguồn sẽ được bỏ qua. Video nguồn được chuyển mã từ đúng bản đã trích (metadata `source-etag`,
  cùng số frame) cũng được bỏ qua.
- Nếu đổi bất kỳ tham số thuật toán nào thì signature đổi, mọi video sẽ được xử lý lại và ghi đè key cũ.
  Khi đó cần thông báo cho downstream vì `n` có thể thay đổi.
- `run_mode="finalize"` chỉ dựng lại `final/*` từ commit và registry trên R2, không cần GPU.
