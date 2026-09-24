# AIC 2026 — OCR batch 2 (M tin tức + N camera giao thông + S01 đua xe đạp)

Tài liệu bàn giao kết quả OCR của **toàn bộ keyframe data batch 2** cho các task downstream (Elasticsearch,
retrieval backend, fusion, frontend). Keyframe nguồn được mô tả trong
[`AIC2026_Batch2_M_N_Keyframe_R2_HANDOFF.md`](AIC2026_Batch2_M_N_Keyframe_R2_HANDOFF.md).

> Snapshot ngày 2026-09-24. Số liệu lấy từ `audit/` và `raw/` trên HF Bucket, và từ `final/` dựng bằng cell
> Finalize của notebook (post-process `aic-ocr-clean-batch2-region-v3`). Mọi dòng trong `final/` đã được
> đối chiếu với `frame_registry.parquet` của keyframe (§9).

## 1. Trạng thái bàn giao

| | M — tin tức | N — camera giao thông | S01 — đua xe đạp | Tổng |
|---|---:|---:|---:|---:|
| Keyframe trong registry | 209.111 | 44.572 | 621.117 | **874.800** |
| OCR `status = ok` | 207.780 | 44.572 | 620.976 | **873.328** |
| `status = skipped` (frame đen, `quality < 0,05`) | 1.331 | 0 | 141 | 1.472 |
| `permanent_error` / missing / thừa | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | **0** |
| Shard (đơn vị checkpoint) | 10 (`M01`…`M10`) | 10 (`N001_N010`…`N091_N100`) | 12 (`S01-V001`…`S01-V012`) | 32 |
| Keyframe `pipeline_signature` | `e6c789322adb` | `773e4b7465c2` | `db5918a110cf` | |
| Audit từng shard | **PASS** | **PASS** | **PASS** | 32/32 |

- OCR `pipeline_signature`: **`ae38408251ce3f9b`**, dùng chung cho cả ba profile.
- OCR đủ 100 % registry: 873.328 frame `ok` + 1.472 frame `skipped` = 874.800. Không frame nào lỗi.
- Không có frame `no_text`: mọi frame đều có ít nhất chữ overlay (logo, ticker, banner, HUD).

Lưu ý quan trọng cho downstream (chi tiết ở §10):

- **Field search chính là `text_clean`.** Chữ overlay được tách ra field riêng: ticker (M, S01), banner camera (N),
  HUD đua (S01), logo/đồng hồ (M, S01). Hệ quả là **54,6 % frame M** có `text_clean` rỗng; trên các frame này chỉ có
  ticker và logo (§7.1).
- Join bằng **`frame_id`**. `video_id` của N và S01 dùng gạch ngang (`N001-V001`, `S01-V001`), nên không được suy
  category bằng `video_id.split("_")`.
- Index OCR hiện tại (`aic26_ocr_keyframes_v2`) có mapping `dynamic: false`. Các field mới của batch 2 chỉ được
  index khi đã thêm vào mapping (§11.2).

## 2. Model và cấu hình

Notebook: [`hunyuanocr_colab_a100_v3_7_batch2_MN_r2.ipynb`](hunyuanocr_colab_a100_v3_7_batch2_MN_r2.ipynb). Tên có `MN`
nhưng notebook xử lý cả M, N và S01. Notebook dựa trên v3.7 spatial-safe, bản đã OCR L21–L30 của batch 1.

| Thành phần | Giá trị |
|---|---|
| Model | `tencent/HunyuanOCR` 1.5, revision `de8f10ad2f00a0cefd790b526de8a65dcfdb3205` |
| Engine | vLLM `0.18.1`, transformers `4.57.6`, `bfloat16`, 1× A100 40 GB, batch 512 |
| Prompt (spotting) | `检测并识别图片中的文字，将文本坐标格式化输出。` (chữ + toạ độ) |
| Ảnh vào | JPEG keyframe full frame từ R2, cạnh dài ≤ 1.280 px (không resize thêm) |
| Sinh | greedy (`temperature=0`, `top_p=1`, `top_k=-1`), `repetition_penalty=1.08` |
| Token | 3.072 cho mọi frame; tự retry 8.192 nếu output bị cắt (`length`) hoặc spotting không có bbox |
| Chống lặp | cắt đuôi lặp (exact suffix + spatial-safe cho spotting); repetition rõ ràng thì không retry |
| Không có chữ | câu trả lời "图片中没有文字。" và tương đương được nhận là no-text hợp lệ |
| `pipeline_version` | `aic-hunyuanocr-colab-v3.7-batch2-r2` |
| Post-process | `aic-ocr-clean-batch2-region-v3` (§6) |

Model và sampling giống hệt OCR batch 1 (L21–L30,
`ocr/hunyuanocr-1.5-spotting-v2-cuda12-ar/`, signature `5da5af248888a0f7`). Hai batch có signature khác nhau vì
khác `pipeline_version` và cấu hình token (batch 1 có budget riêng cho L25). Mỗi batch nằm ở prefix riêng, không
trộn record giữa hai prefix.

### Thời gian chạy

| Session | Shard | Frame | Thời gian | Tốc độ |
|---|---|---:|---:|---:|
| M + N (1 A100) | M01–M10, N001_N010–N091_N100 | 253.683 | 319 phút (23/09 20:32 → 24/09 01:51 UTC) | 13,3 frame/s |
| S01 session 1 | V007, V012, V006, V003 | 208.196 | 271 phút | 12,8 frame/s |
| S01 session 2 | V008, V011, V002, V001 | 203.605 | không ghi lại¹ | |
| S01 session 3 | V010, V005, V004, V009 | 209.316 | 273 phút | 12,8 frame/s |

¹ `run_state` của session 2 bị một lượt chạy resume (1,4 phút, không còn frame nào cần OCR) ghi đè.
Checkpoint cuối của S01 lên Bucket lúc 24/09 12:07 UTC.

## 3. Layout output trên HF Bucket

```text
hf://buckets/Baonenha1/DATA-AIC-Keyframe/ocr/batch2/hunyuanocr-1.5-spotting-v3.7-r2/
├── run_manifest.json                 # cấu hình + OCR signature + sha256 của 3 registry keyframe
├── run_state/<hash>.json             # tóm tắt từng session (shard, thời gian, shard lỗi)
├── raw/<shard>.jsonl                 # 32 file, 2,46 GB: checkpoint append-only (M 585 MB, N 107 MB, S01 1,77 GB)
├── audit/<shard>.audit.json          # 32 file: expected / ok / skipped / missing / cờ chất lượng
└── final/
    ├── M/ocr_clean.jsonl             # 207.780 dòng, 730 MB  (chỉ status=ok)
    ├── M/ocr_clean.parquet           # 209.111 dòng, 95 MB   (mọi frame, gồm skipped)
    ├── M/_SUCCESS.json
    ├── N/ocr_clean.jsonl             # 44.572 dòng, 131 MB
    ├── N/ocr_clean.parquet           # 44.572 dòng, 8,4 MB
    ├── N/_SUCCESS.json
    ├── S/ocr_clean.jsonl             # 620.976 dòng, 2,17 GB
    ├── S/ocr_clean.parquet           # 621.117 dòng, 177 MB
    ├── S/_SUCCESS.json
    ├── audit_summary.json            # audit của 32 shard + tổng theo profile
    └── _SUCCESS.json                 # ghi SAU CÙNG, khi cả M, N, S đều đủ
```

- Profile S01 dùng tên thư mục `S`, giống cách đặt tên `infoshoot_s/` và `media_manifest_S.csv` của keyframe.
- Chỉ đọc `final/`. `raw/` là checkpoint: một frame có thể xuất hiện nhiều lần (retry), và record mới nhất theo
  `ts` mới là record hợp lệ. Finalize đã chọn sẵn record đó.
- `final/<profile>/_SUCCESS.json` chỉ được ghi khi profile đó đủ 100 % registry.

## 4. Định danh và join

| Field | Ví dụ M | Ví dụ N | Ví dụ S01 | Ghi chú |
|---|---|---|---|---|
| `frame_id` | `M03_V005@f00000009` | `N001-V001@f00000000` | `S01-V001@f00000002` | **khoá chính**, trùng registry keyframe |
| `id` | `M03/M03_V005/f00000009` | `N001/N001-V001/f00000000` | `S01/S01-V001/f00000002` | cùng format `id` OCR batch 1 |
| `video_id` | `M03_V005` | `N001-V001` | `S01-V001` | |
| `category` | `M03` | `N001` | `S01` | lấy từ registry, **không** tách từ `video_id` |
| `profile` / `shard` | `M` / `M03` | `N` / `N001_N010` | `S` / `S01-V001` | |
| `n` = `keyframe_n` | `1` | `1` | `1` | thứ tự keyframe, trùng tên JPEG |
| `submit_keyframe_id` | `M03/M03_V005/001` | `N001/N001-V001/001` | `S01/S01-V001/001` | dùng để submit |
| `keyframe_name` | `001.jpg` | `001.jpg` | `75123.jpg` | tên file trên R2 (index batch 1 lưu dạng `001`) |
| `frame_idx`, `pts_time`, `fps` | | | | copy nguyên từ registry |
| `r2_key` | `Keyframes/Keyframes_M03/M03_V005/001.jpg` | | | đọc ảnh |

Quy tắc:

- `frame_id`, `n`, `submit_keyframe_id`, `r2_key`, `frame_idx`, `pts_time` khớp registry keyframe **từng dòng** (§9).
- `n` chỉ ổn định trong đúng snapshot keyframe này. Hãy giữ `keyframe_pipeline_signature` cùng output.
- Hai field **không dùng** để truy xuất ảnh:
  - `image_path` là đường dẫn staging trên Colab (`/content/data/batch2/...`).
  - `public_url` copy từ registry, trỏ `r2.dev`.

  Hãy dựng URL từ `r2_key`: `https://keyframe.baoencoder.site/<r2_key>`.

## 5. Schema

### 5.1 Record `raw/*.jsonl` (giống v3.7)

| Nhóm | Field |
|---|---|
| Định danh | các field ở §4, cùng `r2_bucket`, `image_relpath`, `size_bytes`, `image_width`, `image_height`, `keyframe_quality`, `keyframe_pipeline_signature`, `analysis_crop_top_frac`, `analysis_crop_bottom_frac`, `analysis_mask_boxes` |
| Phiên bản | `pipeline_version`, `pipeline_signature` (OCR), `notebook_patch_version`, `model_id`, `model_version`, `model_revision`, `prompt_mode`, `ts` |
| Kết quả | `status` (`ok` / `skipped` / `permanent_error`), `skip_reason`, `error`, `raw`, `raw_original`, `boxes` (`[{text, box:[x1,y1,x2,y2]}]`), `text_nfc`, `text_fold` |
| Cờ sinh | `finish_reason`, `generated_tokens`, `generation_max_tokens`, `truncated`, `adaptive_retry`, `adaptive_retry_reason`, `adaptive_retry_error`, `adaptive_retry_skipped_repetition`, `repetition_detector_kind`, `repetition_detector_count`, `tail_repetition_*`, `spotting_malformed`, `no_text_detected`, `no_text_response` |

- Toạ độ `box` nằm trên **lưới 0..1000 theo cả hai trục**, độc lập kích thước ảnh. Đây là quy ước output của HunyuanOCR.
- `raw` là output model sau khi cắt đuôi lặp. `raw_original` lưu output trước khi cắt (hoặc output 3.072 token trước
  khi retry 8.192) và chỉ có giá trị khi khác `raw`.
- `text_nfc` = mọi box ghép lại, chuẩn NFC (gồm cả logo, ticker, banner). `text_fold` = `text_nfc` bỏ dấu + lowercase.

### 5.2 `final/<profile>/ocr_clean.jsonl`

Mỗi dòng là một record `raw` có `status = ok`, cộng thêm các field post-process sau:

| Field | Ý nghĩa |
|---|---|
| `boxes[].region` | `scene` · `ticker` · `logo_clock` · `banner` · `hud` · `unknown` (text không có bbox), xem §6 |
| `text_clean`, `text_clean_fold` | **field search chính**: chữ trong cảnh (`scene`/`unknown`), đã bỏ box nhiễu |
| `text_clean_hash` | 16 hex đầu của sha1(`text_clean`), dùng để gộp frame liên tiếp cùng chữ |
| `text_ticker`, `text_ticker_fold` | chữ trên băng ticker (M: băng tin đỏ · S01: MiniGame / bình luận khán giả) |
| `text_banner` | toàn bộ chữ trong banner camera N |
| `banner_camera` | tên nút giao N (box sát mép trái banner) |
| `banner_date` | ngày trên banner N, ISO `YYYY-MM-DD` |
| `text_hud` | S01: HUD đua ("CHẶNG n", đồng hồ đua, km còn lại, khoảng cách, marker, tốc độ) |
| `race_stage` | S01: số chặng (int) đọc từ HUD; `null` nếu không đọc được |
| `race_time` | S01: thời gian **đã đua** `HH:MM:SS`, không phải giờ thật |
| `clock`, `hour` | giờ thật. M: đồng hồ HTV7 (chuỗi như trên màn hình, giờ có thể 1 chữ số, vd `8:30:13`). N: giờ banner (`HH:MM:SS`). S01: không có |
| `ticker_policy`, `postprocess_version` | `separate`, `aic-ocr-clean-batch2-region-v3` |

### 5.3 `final/<profile>/ocr_clean.parquet`

Mỗi dòng ứng với một frame của registry, gồm cả frame `skipped`. Parquet có 49 cột, kiểu cố định:

```text
frame_id, id, video_id, category, profile, shard, n, keyframe_name, submit_keyframe_id, frame_idx, pts_time, fps,
r2_key, public_url, image_width, image_height, keyframe_quality, keyframe_pipeline_signature, pipeline_signature,
model_id, model_revision, status, skip_reason, error, no_text_detected,
text_clean, text_clean_fold, text_clean_hash, text_ticker, text_ticker_fold,
text_banner, banner_camera, banner_date, text_hud, race_stage, race_time, clock, hour,
text_nfc, text_fold, raw, boxes_json, n_boxes,
finish_reason, truncated, adaptive_retry, tail_repetition_detected, spotting_malformed, postprocess_version
```

- `int64` (nullable): `n`, `frame_idx`, `image_width`, `image_height`, `race_stage`, `hour`, `n_boxes`.
- `float64`: `pts_time`, `fps`, `keyframe_quality`.
- `bool`: `no_text_detected`, `truncated`, `adaptive_retry`, `tail_repetition_detected`, `spotting_malformed`.
- `string`: các cột còn lại. Chuỗi rỗng nghĩa là không có giá trị.
- `boxes_json` là chuỗi JSON của `boxes`, đã có `region`.
- Frame `skipped` có mọi field text rỗng.

## 6. Post-process theo vùng (v3)

HunyuanOCR đọc **toàn bộ khung hình**, gồm cả chữ overlay. Post-process gán mỗi box vào một vùng theo tâm box, rồi
chỉ đưa box `scene`/`unknown` không phải nhiễu vào `text_clean`. Post-process chỉ đọc `raw`, nên đổi quy tắc không
cần OCR lại, chỉ cần chạy lại Finalize (§12).

| Profile | Nguồn quy tắc | `banner` | `ticker` | `logo_clock` | `hud` |
|---|---|---|---|---|---|
| M | registry (`analysis_*`) | — | tâm y ≥ 0,898 (băng tin đỏ 90,1–95,8 %) | `[0.80, 0.03, 0.955, 0.175]` logo HTV7 + đồng hồ | — |
| N | registry | tâm y ≤ 0,04 | — | — | — |
| S01 | `PROFILES["S"]` trong notebook (registry S01 không crop) | — | tâm y ≥ 0,92 | `[0.80, 0.03, 0.955, 0.175]` logo "HTV TRỰC TIẾP"; `[0.80, 0.80, 0.98, 0.92]` sponsor | `[0.00, 0.00, 0.22, 0.20]` "CHẶNG n" + đồng hồ đua; box **dạng telemetry** trong `[0.22, 0.00, 0.72, 0.20]` và `[0.00, 0.76, 0.14, 0.90]` |

Toạ độ trong bảng tính theo tỉ lệ khung hình `[x1, y1, x2, y2]`. Vùng S01 được đo trên keyframe livestream HTV.

Quy tắc nhiễu (box bị bỏ khỏi `text_clean` khi **toàn bộ** box là nhiễu):

- Kế thừa v3.7:
  - mảnh toạ độ sót;
  - dấu câu/ký hiệu đứng một mình;
  - logo `HTV*`, `HD`, `HCTV`, `SUBSCRIBE`;
  - watermark `giây` / `60 giây`;
  - chuỗi giờ nằm ở góc trên phải.
- Chữ/số alphanumeric đứng một mình **được giữ**, giống v3.7.
- S01 thêm các cụm `TON DONG A`, `TON DONG`, `DONG A`, `TRỰC TIẾP`. Box chỉ gồm các cụm này (kèm logo HTV) là nhiễu
  ở bất kỳ vị trí nào. Còn title card như "TON DONG A CÚP TRUYỀN HÌNH 2026 … CHẶNG 06 …" vẫn được giữ.
- "Dạng telemetry" là text ≤ 16 ký tự chỉ gồm số, `km`, `km/h`, `+ - . , : ' "`, hoặc một chữ `S`/`P`. Ví dụ
  `106.4 Km`, `+2'56''`, `1`, `S`. Chữ thật trên cổng đua trong cùng dải (ví dụ "NON SÔNG LIỀN MỘT DẢI",
  "CÚP TRUYỀN HÌNH") vẫn ở `scene`.

Các field trích xuất:

- `clock`, `hour`:
  - M lấy chuỗi `H:MM:SS` hợp lệ đầu tiên trong vùng logo, nếu không có thì lấy ở góc trên phải (như v3.7).
  - N lấy từ banner.
  - Chỉ nhận giờ 00–23 và phút/giây 00–59.
- `banner_camera`: box banner sát mép trái (x1 ≤ 30/1000), đã bỏ ngày và giờ nếu OCR gộp chung box.
- `race_stage`: `CHẶNG n` trong HUD, chỉ nhận 1–30. Giá trị `0` là OCR đọc nhầm số 6/8/9 ở cỡ chữ nhỏ, nên được
  chuyển thành `null`.

## 7. Thống kê

### 7.1 Độ phủ (tính trên frame `ok`)

| | M | N | S01 |
|---|---:|---:|---:|
| `text_nfc` không rỗng | 100 % | 100 % | 100 % |
| **`text_clean` không rỗng** | **45,4 %** | **60,7 %** | **50,3 %** |
| `text_ticker` không rỗng | 86,3 % | — | 19,8 % |
| `banner_camera` / `banner_date` | — | 99,6 % / 99,6 % | — |
| `text_hud` không rỗng | — | — | 74,2 % |
| `race_stage` / `race_time` | — | — | 66,7 % / 70,3 % |
| `clock` / `hour` | 90,5 % | 99,6 % | — |
| Độ dài `text_clean` (trung vị / p90, ký tự) | 41 / 124 | 21 / 66 | 18 / 111 |
| Số `text_clean` khác nhau | 71.423 | 13.850 | 182.452 |
| Frame có `text_clean` giống frame liền trước cùng video | 48,5 % | 48,7 % | 55,2 % |
| `text_clean` chỉ gồm token ≤ 2 ký tự | 2,7 % | 3,8 % | 8,3 % |

Số box theo vùng:

| | `scene` | `ticker` | `logo_clock` | `banner` | `hud` | `unknown` |
|---|---:|---:|---:|---:|---:|---:|
| M | 535.925 | 376.827 | 419.184 | — | — | 54 |
| N | 95.835 | — | — | 106.694 | — | 6 |
| S01 | 1.935.043 | 223.539 | 1.861.325 | — | 2.441.370 | 92 |

### 7.2 Chất lượng sinh (từ `audit/`)

| | M | N | S01 |
|---|---:|---:|---:|
| Retry 8.192 token (`adaptive_retry`) | 129 | 11 | 224 |
| Vẫn bị cắt sau retry (`truncated`) | 88 | 10 | 144 |
| Còn đuôi lặp sau cleanup | 22 | 3 | 40 |
| Spotting không có bbox (`spotting_malformed`) | 2 | 0 | 0 |
| `permanent_error` | 0 | 0 | 0 |

Frame bị cắt là frame có chữ rất dày. Trung vị số box của chúng là 130 (M) và 146 (S01), trong khi frame bình
thường chỉ có 5 và 10 box. Phần chữ đã đọc được vẫn nằm trong `raw` và `boxes`.

### 7.3 M — đồng hồ phát sóng

187.989 frame có `hour`. 184.608 frame (98,2 %) là khung **18 giờ** (bản tin "60 giây" buổi tối). Phần còn lại rải ở
8 giờ (692 frame) và 10–16 giờ (~2.400 frame).

### 7.4 N — banner so với `batch2_camera_metadata.csv`

Metadata camera: `~/Projects/Media_Upload/batch2_content/batch2_camera_metadata.csv`.

| Kiểm tra (theo video, lấy giá trị phổ biến nhất) | Khớp |
|---|---:|
| `banner_date` = `overlay_date` | 297/298 |
| trung vị `hour` nằm trong khung giờ `first/last_overlay_time` | 297/298 |
| `banner_camera` bằng hoặc chứa nhau với `camera_location` | 275/298 |

Trong 23 video lệch tên camera, phần lớn là do **metadata sai**, còn OCR đúng:

- tên bị cụt: `…NGUYEN VAN` ↔ OCR `…NGUYEN VAN CU`;
- lỗi đọc trong metadata: `CORE Vien Da cau Calmette` ↔ `Cong Vien Da cau Calmette`,
  `Hong Bang-Phu Dong Thien Wong` ↔ `…Thien Vuong`, `NamKy KhoiNia-Ly TuTrong` ↔ `Nam Ky Khoi Nghia - Ly Tu Trong`.

Riêng `N016-V001` và `N016-V003`, metadata ghi `NTMK - Ton That Tung` vì suy từ nhóm ("OCR cùng nhóm N016"). Banner
thật là `NTMK - Nguyen Thuong Hien` và `DBPhu - Truong Dinh`, OCR đọc đúng; đã kiểm bằng ảnh. **Hãy dùng
`banner_camera` thay cho metadata CSV khi cần tên camera theo frame.**

Phân bố `hour` của N: 6 h (117), 7 h (1.277), 8 h (12.099), 10 h (914), 11 h (14.093), 18 h (882), 19 h (15.001).

### 7.5 S01 — chặng theo video

Mỗi video S01 là một chặng. `race_stage` phổ biến nhất trong mỗi video đúng bằng số thứ tự video:

| Video | Chặng | Frame có `race_stage` / frame `ok` | Tỉ lệ đúng chặng chính |
|---|---:|---:|---:|
| S01-V001 | 1 | 16.870 / 36.728 | 98,8 % |
| S01-V002 | 2 | 31.075 / 38.621 | 99,8 % |
| S01-V003 | 3 | 11.943 / 29.740 | 97,5 % |
| S01-V004 | 4 | 33.940 / 51.454 | 99,1 % |
| S01-V005 | 5 | 46.736 / 55.274 | 99,7 % |
| S01-V006 | 6 | 13.033 / 34.621 | 98,3 % |
| S01-V007 | 7 | 59.349 / 75.099 | 99,6 % |
| S01-V008 | 8 | 46.657 / 65.731 | 99,7 % |
| S01-V009 | 9 | 6.018 / 45.631 | 96,5 % |
| S01-V010 | 10 | 40.820 / 56.934 | 99,5 % |
| S01-V011 | 11 | 44.991 / 62.472 | 97,8 % |
| S01-V012 | 12 | 62.427 / 68.671 | 100,0 % |

Các giá trị lệch (1–3,5 %) thường là chặng kế tiếp hoặc chặng trước xuất hiện trong đồ hoạ giới thiệu.
`S01-V009` có ít frame đọc được chặng vì HUD ghi "CHẶNG 9" và OCR thường đọc thành "0" (§10.3).

Title card đầu chặng được giữ trong `text_clean`, gồm ngày, số chặng và địa điểm. Ví dụ:
`CÚP TRUYỀN HÌNH 2026 THỨ 4 NGÀY 08.04.2026 Vòng đua quanh CHẶNG 06 QUẢNG TRƯỜNG HỒ C…`.

## 8. Khác biệt so với OCR batch 1 (L21–L30)

| | Batch 1 (L21–L30) | Batch 2 |
|---|---|---|
| Prefix | `ocr/hunyuanocr-1.5-spotting-v2-cuda12-ar/` | `ocr/batch2/hunyuanocr-1.5-spotting-v3.7-r2/` |
| Nguồn ảnh | TAR InfoShot++ trên HF Bucket | JPEG trên R2 theo `frame_registry.parquet` |
| `id` | `Lxx/Lxx_Vyyy/f########` | cùng format |
| `submit_keyframe_id` | phải map qua `map-keyframes` | có sẵn trong record |
| Post-process | `aic-ocr-clean-v3.1-no-text`: ticker nằm **trong** `text_clean` | v3: ticker, banner, HUD, logo tách field riêng |
| Output final | `ocr_clean.jsonl` | `final/<profile>/ocr_clean.{jsonl,parquet}` |

M và L cùng là bản tin HTV. Nếu muốn kênh OCR xử lý M giống L, hãy search `text_clean` **và** `text_ticker` cho M
(§11.3).

## 9. Kiểm chứng

1. 32/32 file `audit/*.json` có `audit_ok = true`, `missing = 0`, `extra = 0`, `permanent_error = 0`. Signature OCR và
   keyframe khớp `run_manifest.json`.
2. 32 file `raw/*.jsonl` tải về có kích thước khớp listing trên Bucket.
3. Đối chiếu từng dòng `final/<profile>/ocr_clean.parquet` với `frame_registry.parquet` của keyframe
   (`keyframe_batch2/infoshoot_{m,n,s}/final/`):
   - tập `frame_id` bằng nhau, không thiếu, không thừa;
   - `frame_id` duy nhất;
   - `n`, `submit_keyframe_id`, `r2_key`, `frame_idx`, `pts_time`, `category` trùng từng dòng;
   - `id = category/video_id/f########`.
4. Mọi frame `skipped` đều có `keyframe_quality < 0,05`.
5. Số dòng `ocr_clean.jsonl` bằng số frame `ok`: 207.780 / 44.572 / 620.976.
6. So post-process v3 với bản v2 trước đó, thay đổi chỉ nằm ở field dự kiến:
   - M: 1.252 dòng `clock`/`hour` có giờ, phút hoặc giây sai (ví dụ `48:12:30`, `18:32:66`) được bỏ;
   - N: `banner_camera` ở 14.115 dòng không còn dính biển hiệu cảnh;
   - các field khác của M/N không đổi.

## 10. Vấn đề đã biết và khuyến nghị

### 10.1 M: phần lớn chữ là overlay

- 54,6 % frame M chỉ có ticker và logo/đồng hồ, nên `text_clean` rỗng. Nếu chỉ search `text_clean`, các frame
  này không bao giờ được kênh OCR trả về.
- Ticker có ở 86,3 % frame. Nội dung ticker thường là **tin khác** với hình đang chiếu (handoff keyframe §11.4).
  Khuyến nghị: index `text_ticker` với trọng số thấp, hoặc chỉ dùng khi `text_clean` không có kết quả.
- Nhóm `text_clean` phổ biến nhất là đồ hoạ chuyển mục: `60 TIẾP THEO`, `TIN CHÍNH`, `TIẾP THEO TRÊN HTV7 …`.
  Chúng có nhiều frame, nên IDF thấp.

### 10.2 N: dùng banner làm metadata thật

- `banner_camera` + `banner_date` + `hour` cho biết camera và thời điểm thật của từng frame. Dữ liệu này đáng tin hơn
  `batch2_camera_metadata.csv` (§7.4).
- Chữ trong cảnh của N chủ yếu là biển hiệu, biển số, biển báo ("HIGHLANDS COFFEE", "CHAGEE", "CẤP CỨU"…). 39,3 % frame
  không có chữ trong cảnh.

### 10.3 S01: HUD, chặng và nhiễu còn sót

- `race_stage` bị `null` ở 18.685 frame vì OCR đọc chữ số nhỏ 6/8/9 thành 0; nặng nhất là V009, V008, V006. Muốn lọc
  theo chặng thì nên dùng **ánh xạ video → chặng** ở §7.5 (`S01-V00k` = chặng k), và chỉ dùng `race_stage` khi cần
  bằng chứng theo frame.
- `race_time` là thời gian đã đua của chặng, không phải giờ trong ngày.
- 8,3 % frame S01 có `text_clean` chỉ gồm token rất ngắn: số áo, số tốc độ nằm ngoài vùng telemetry, "S S", "12".
  Các token này vô hại với BM25 (IDF thấp) nhưng không mang nghĩa.
- 55,2 % frame S01 có `text_clean` trùng frame liền trước, vì mật độ keyframe cao (3,7/giây). Nên gộp kết quả OCR
  theo (`video_id`, `text_clean_hash`) trong một cửa sổ thời gian ngắn, để một đoạn đua không chiếm hết top-k.

### 10.4 Chung

- 1.472 frame đen được `skipped`. Chúng **không có** trong `ocr_clean.jsonl` nhưng có trong parquet (`status = skipped`).
- 242 frame vẫn bị cắt sau 8.192 token và 65 frame còn đuôi lặp. Text của các frame này có thể thiếu ở cuối.
- Ký tự CJK/tiếng Anh trong ảnh được giữ nguyên, không dịch.

## 11. Hướng dẫn cho downstream

### 11.1 Đọc dữ liệu

Cần token HF có quyền đọc Bucket `Baonenha1/DATA-AIC-Keyframe`:

```bash
export HF_TOKEN=...          # không commit token
P=hf://buckets/Baonenha1/DATA-AIC-Keyframe/ocr/batch2/hunyuanocr-1.5-spotting-v3.7-r2
uvx --from 'huggingface_hub[hf-xet]==1.24.0' hf buckets cp $P/final/_SUCCESS.json ./
uvx --from 'huggingface_hub[hf-xet]==1.24.0' hf buckets sync $P/final ./ocr_batch2_final
```

```python
import pandas as pd

cols = ["frame_id", "submit_keyframe_id", "video_id", "category", "n", "pts_time", "status",
        "text_clean", "text_clean_fold", "text_ticker", "banner_camera", "race_stage", "hour"]
s01 = pd.read_parquet("ocr_batch2_final/S/ocr_clean.parquet", columns=cols)   # chọn cột: bỏ raw/boxes_json cho nhẹ RAM
s01 = s01[s01.status == "ok"]
```

Trước khi dùng, kiểm `final/<profile>/_SUCCESS.json` (có `expected`, `ok`, `skipped_black`) và `pipeline_signature`
bằng `ae38408251ce3f9b`.

### 11.2 Đưa vào Elasticsearch

Record batch 2 **đã có sẵn** `submit_keyframe_id`, `frame_idx`, `pts_time`. Không cần chạy `keyframe_mapping_v2.py`:
script này suy category bằng `video_id.split("_", 1)[0]`, nên sẽ loại toàn bộ record N và S01.

Map sang schema index OCR hiện tại (`aic26_ocr_keyframes_v2`):

| Field index | Lấy từ record batch 2 |
|---|---|
| `ocr_id` | `id` |
| `submit_keyframe_id`, `frame_id`, `video_id`, `frame_idx`, `pts_time`, `fps` | cùng tên |
| `category`, `submit_category` | `category` |
| `keyframe_n` | `n` |
| `keyframe_id` | `f"{video_id}/{n:03d}"` |
| `keyframe_name` | `f"{n:03d}"` (record có đuôi `.jpg`, index batch 1 không có) |
| `image_path` | `r2_key` (không dùng `image_path` của record) |
| `text_clean`, `text_clean_fold`, `text_nfc`, `clock`, `hour`, `status`, `error`, `ts`, `boxes` | cùng tên |

Mapping hiện tại có `dynamic: false`. Muốn search hoặc lọc bằng field mới thì phải thêm vào mapping rồi reindex:

```text
profile                keyword
text_ticker            text (+ .fold qua text_ticker_fold)
text_banner            text
banner_camera          keyword + text
banner_date            date (format yyyy-MM-dd)
text_hud               text
race_stage             integer
race_time              keyword
text_clean_hash        keyword
keyframe_pipeline_signature, pipeline_signature   keyword
boxes.region           keyword (trong nested boxes)
```

### 11.3 Truy vấn

Áp dụng bài học ranking OCR của batch 1 (`AIC2026/Vấn-đề-ranking-OCR.md`):

- Xếp tầng: cùng box khớp cụm (`boxes.text` `match_phrase`) > cả frame khớp cụm > đủ 100 % token
  (`operator: and`) > đủ 100 % token dạng bỏ dấu > fuzzy fallback.
- Không cộng điểm cùng một bằng chứng qua `text_clean` + `text_nfc` + `text_clean_fold` (dùng `dis_max`).
- **Không fuzzy số, năm, biển số, mã.** Với batch 2 điều này càng quan trọng, vì có km, giờ, số áo, biển số xe.
- Chỉ dùng `text_ticker` (M, S01) như kênh phụ, trọng số thấp.
- Filter theo field có cấu trúc thay vì full-text:
  - `profile`/`category`;
  - `hour` (M, N);
  - `banner_camera`, `banner_date` (N);
  - chặng S01 qua `video_id` hoặc `race_stage`.
- Gộp frame S01 liên tiếp cùng `text_clean_hash` trước khi fusion (RRF).

### 11.4 Join với embedding / keyframe

- Join bằng `frame_id`.
- Giữ `pipeline_signature` (OCR) và `keyframe_pipeline_signature` (keyframe) trong mọi bảng dẫn xuất. Nếu keyframe
  được trích lại (signature keyframe đổi), OCR của video đó phải chạy lại.

## 12. Tái lập

- Notebook: [`hunyuanocr_colab_a100_v3_7_batch2_MN_r2.ipynb`](hunyuanocr_colab_a100_v3_7_batch2_MN_r2.ipynb).
- Chạy OCR:
  - Colab A100, secrets `HF_TOKEN` (ghi Bucket) và `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`
    (đọc R2);
  - đặt `CATEGORIES` theo shard (`"M"`, `"N"`, `"S"`, hoặc danh sách video S01), rồi Run all;
  - resume theo `frame_id`, chỉ nhận record khớp cả OCR signature lẫn keyframe signature.
- Dựng lại `final/` (không cần GPU): `CATEGORIES = []`, `RUN_POSTPROCESS = True`, Run all.
  - Đổi quy tắc vùng hoặc nhiễu (`PROFILES`, cell post-process) thì chỉ cần làm bước này.
  - Hãy tăng `POSTPROCESS_VERSION` khi đổi quy tắc.
- Đổi model, prompt, token hoặc sampling làm OCR signature đổi. Khi đó notebook dừng vì prefix đã mang signature
  `ae38408251ce3f9b`, và cần một prefix mới để OCR lại toàn bộ.
