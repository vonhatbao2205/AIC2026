# AIC 2026 — Bàn giao video & keyframe batch 2 (M, N, S01) cho web viewer

Tài liệu cho nhóm làm web xem dữ liệu và keyframe batch 2. Nội dung: phát video
và ảnh ở đâu, theo quy tắc nào, video nào đã được xử lý lại và vì sao, cùng các
lỗi đã biết. Chi tiết về quá trình trích keyframe nằm ở
[`AIC2026_Batch2_M_N_Keyframe_R2_HANDOFF.md`](AIC2026_Batch2_M_N_Keyframe_R2_HANDOFF.md);
tài liệu này không lặp lại phần đó.

> Snapshot 2026-09-24. Mọi con số dưới đây đều lấy từ việc kiểm tra thật trên R2
> và trên Chrome 151, không lấy từ log.

## 1. Tóm tắt: URL cần dùng

| Nội dung | URL |
|---|---|
| Keyframe (mọi nhóm) | `https://keyframe.baoencoder.site/Keyframes/Keyframes_{group}/{video_id}/{n:03d}.jpg` |
| Video M, S01 (và L) | `https://video.baoencoder.site/Videos/Videos_{group}/{video_id}.mp4` |
| Video N | `https://video.baoencoder.site/Videos_Web/Videos_{group}/{video_id}.mp4` |
| Video N có frame hỏng (103 video, §5.3) | `https://video.baoencoder.site/Videos_Web_v2/Videos_{group}/{video_id}.mp4` |
| Metadata (registry, manifest, map) | `https://pub-5010c807b73743ba82d6c40852ae9e6b.r2.dev/manifest/infoshoot_{m,n,s}/final/…` |

**Không dùng bản gốc `Videos/Videos_Nxxx/…` cho video N.** Chrome không phát
được 286/298 file gốc (§5.1).

## 2. Phạm vi

| Nhóm | Nội dung | Video | Thư mục (`group`) | Keyframe |
|---|---|---:|---|---:|
| M | Thời sự "60 giây" HTV7 | 304 | `M01`…`M10` | 209.111 |
| N | Camera giao thông cố định (một thư mục = một camera) | 298 | `N001`…`N100` | 44.572 |
| S01 | Đua xe đạp Cúp Truyền hình 2026 | 12 | `S01` | 621.117 |

Tổng: 614 video, 874.800 keyframe. Keyframe L21–L30 cũng nằm trên cùng máy chủ
ảnh, theo cùng layout.

## 3. Định danh

| Trường | Ví dụ | Ghi chú |
|---|---|---|
| `video_id` | `M01_V001`, `N001-V001`, `S01-V001` | Đúng tên file của BTC. **M dùng `_`, N và S01 dùng `-`.** |
| `group` | `M01`, `N001`, `S01` | Tên thư mục trên R2. Lấy phần trước dấu `_` **hoặc** `-` đầu tiên. |
| `n` | `1`, `2`, … | Thứ tự keyframe trong video, bắt đầu từ 1; cũng là tên file ảnh `{n:03d}.jpg` (độ rộng tối thiểu 3, nên có cả `1000.jpg` hay `75123.jpg`). |
| `frame_idx` | `375` | Chỉ số frame theo thứ tự giải mã của video gốc. **Không phải** tên file ảnh. |
| `pts_time` | `15.201` | Giây, dùng để tua video tới đúng keyframe. |
| `submit_keyframe_id` | `N001/N001-V001/002` | `{group}/{video_id}/{n:03d}`, là id dùng trong app. |
| `frame_id` | `N001-V001@f00000375` | `{video_id}@f{frame_idx:08d}`, là khoá join với embedding. |

Hai lỗi thường gặp:
- Lấy `group` bằng `video_id.split("_")[0]`: với N và S01 kết quả là `N001-V001`, sai thư mục, nên ảnh và video trả 404.
- Suy thời gian bằng `frame_idx / fps`: N là VFR (10,9–30 fps, frame đầu thường ở 0,2 s). Luôn dùng `pts_time` trong registry.

## 4. Keyframe

### 4.1 Danh sách keyframe (nguồn sự thật)

Mỗi nhóm có một thư mục `final/` trên R2, đọc công khai qua
`https://pub-5010c807b73743ba82d6c40852ae9e6b.r2.dev/manifest/infoshoot_{m|n|s}/final/`:

| File | Dùng cho web | Nội dung |
|---|---|---|
| `media_manifest_{M,N,S}.csv` | **Nhẹ, nên dùng cho web** | 7 cột đầu `kind,group,video_id,frame_name,frame_idx,source_path,r2_key`, sau đó `category, keyframe_n, keyframe_name, submit_keyframe_id, frame_id, pts_time, fps, public_url, size_bytes, …` |
| `frame_registry.parquet` | Khi cần đủ thông tin | 36 cột, gồm `quality`, `sources`, `phash_hex`, `image_width/height`, … |
| `map-keyframes.zip` | Tương thích định dạng cũ | `map-keyframes/{group}/{video_id}.csv`, cột `n,pts_time,fps,frame_idx` |

Mỗi dòng ứng với đúng một ảnh JPEG trên R2. Không nên dựng danh sách bằng cách
list bucket (S01 có 621 nghìn object).

### 4.2 Ảnh

- JPEG chất lượng 92, cạnh dài tối đa 1.280 px (phần lớn 1280×720; riêng `M03_V006` và `M03_V030` là 640×360).
- Header: `Cache-Control: public, max-age=86400`, `Access-Control-Allow-Origin: *`; có hỗ trợ đọc theo đoạn (Range).
- Ảnh là **full frame**: ảnh M còn băng chữ chạy ticker ở đáy, ảnh N còn banner tên camera và giờ ở đỉnh.

### 4.3 Frame nên ẩn hoặc đánh dấu

- Frame có `quality < 0.05` là frame đen hoặc đang chuyển cảnh: 1.331 frame M (trong đó 1.035 frame là đoạn đuôi đen dài 73 phút của `M10_V029`, `n ≥ 676`) và 141 frame S01. Các frame này không có embedding.
- S01 dày tới 3,7 keyframe/giây và có 9.450 keyframe trùng từng byte với keyframe khác trong cùng video (ví dụ màn chờ "TRỰC TIẾP" đầu `S01-V006`). Filmstrip nên lấy mẫu thưa hoặc gom theo thời gian.

## 5. Video

### 5.1 Vì sao video N phải xử lý lại

Video N gốc là file `.mov` của camera, BTC phát trong các file zip. Lúc upload,
các file này chỉ được **remux** sang `.mp4`: giữ nguyên stream, không mã hoá lại.
Camera ghi các thông điệp SEI trong stream H.264 **khai báo kích thước lớn hơn
chỗ chứa thật**:

- FFmpeg và VLC bỏ qua được, nên xem trên máy vẫn thấy bình thường.
- **Chrome từ chối giải mã** ngay packet thứ 3 (`PIPELINE_ERROR_DECODE`). Player hiện màn hình đen trong khi đồng hồ vẫn chạy.
- 286/298 video N bị lỗi này. 12 video không bị là của `N010`, `N011`, `N023`, `N078` (có lẽ khác loại camera).

### 5.2 `Videos_Web/`: bản vá SEI, giữ nguyên từng frame (298 video)

- Mỗi SEI được ghi đè bằng NAL *filler data* (type 12) **có cùng độ dài**. Mọi decoder đều bỏ qua loại NAL này.
- Kích thước file, vị trí và kích thước từng sample, timestamp, và **toàn bộ frame giải mã ra đều giống bản gốc**. Đã kiểm trên N070-V002: 15.000/15.000 frame trùng MD5 và bảng packet trùng hoàn toàn.
- Bản gốc `Videos/Videos_Nxxx/…` giữ nguyên không đổi.
- Metadata trên object: `pipeline=aic2026-n-sei-filler-v1`, `source-key`, `source-etag`, `source-size`, `video-frames`, `sei-units-replaced`, `strict-decoded-frames`.
- Phải dùng key mới thay vì ghi đè: video được phục vụ với `Cache-Control: public, max-age=31536000, immutable`, nên CDN và trình duyệt sẽ giữ bản hỏng cũ tới một năm.

### 5.3 `Videos_Web_v2/`: 103 video mã hoá lại

Ngoài lỗi SEI, 103 bản ghi còn bị **mất hoặc hỏng frame tham chiếu ngay lúc camera ghi**
(`mmco: unref short failure`). Chrome che được phần lớn chỗ hỏng, nhưng ở một số
keyframe thì dừng hẳn, ví dụ `N027-V003` dừng ở 4:34. Các video này được mã hoá lại
thành stream sạch:

- **Cách làm:** giải mã bản gốc ở chế độ lenient (FFmpeg tự che frame hỏng, giống lúc trích keyframe), rồi mã hoá H.264 High bằng NVENC (CQ 25, riêng `N027-V003` là CQ 23), GOP 25, không B-frame, `yuv420p`. Video không có audio vì bản gốc cũng không có.
- **Timestamp giữ nguyên:** mỗi frame giữ đúng thời gian gốc, lệch ≤ 2 ms. **Mọi keyframe trong registry đều có frame trùng thời điểm** (lệch ≤ 2 ms; với N027-V003 là 0,0 ms trên 89/89 keyframe).
- **Số frame ít hơn bản gốc:** chỉ còn các frame giải mã được. Frame camera làm mất (ví dụ 1.222 trên 13.473 ở N027-V003) không còn; ở các đoạn đó player đứng hình ngắn.
- **Đã kiểm trên output:** giải mã strict sạch hoàn toàn. Chrome phát liên tục qua các đoạn từng bị dừng.
- **Metadata trên object:** `pipeline=aic2026-n-nvenc-reencode-v1`, `source-key`, `source-etag`, `source-size`, `video-frames`, `keyframes-checked`.
- **Danh sách:** xem Phụ lục A. File máy đọc được là `backend/app/video_overrides.json` trong repo ClueScope (`{"roots": {"Videos_Web_v2": [video_id, …]}}`), cũng chính là thứ backend dùng để chọn URL. Đã xong đủ **103/103** video (2026-09-25), không video nào lỗi.

### 5.4 Phát và tua

- Tua tới keyframe bằng `video.currentTime = pts_time`, trong đó `pts_time` lấy từ registry hoặc manifest.
- Header video: `Content-Type: video/mp4`, `Access-Control-Allow-Origin: *`, hỗ trợ Range với M, N và L.
- Video N dài khoảng 10 phút (riêng `N043-V002/V003/V004` chỉ 16–25 giây), 1920×1080, không có audio.

## 6. Vấn đề đã biết (chưa xử lý)

| Vấn đề | Ảnh hưởng | Gợi ý |
|---|---|---|
| **S01 không tua được.** `video.baoencoder.site` trả nguyên file (`200`) thay vì `206` khi được yêu cầu một đoạn, với phần lớn video S01 (mỗi file 3–30 GB). | Video S01 chỉ phát được từ đầu; nhảy tới 3:19:21 không được. | Cần sửa phía Cloudflare hoặc máy chủ gốc cho `Videos/Videos_S01/*`. Chưa có cách khắc phục phía client. |
| **S01 đang chuyển AV1 sang H.264 ngay tại chỗ** (cùng key; bản AV1 gốc lưu ở `Videos_AV1_Original/`). Lần kiểm 2026-09-24: `S01-V001`…`V007` đã là H.264 (metadata `pipeline=aic2026-s01-av1-to-h264-v1`, 8,9–29,9 GB/file), `S01-V012` là H.264 từ đầu, còn `S01-V008`…`V011` vẫn là AV1. | AV1 cần Chrome, Edge hoặc Firefox bản mới; Safari trên máy cũ không phát được. Video vừa chuyển mã có thể còn bản AV1 cũ trong cache CDN. | Đọc metadata `pipeline` của object để biết bản hiện tại; thêm `?v=<ETag>` vào URL để tránh bản cũ trong cache. Việc chuyển mã giữ nguyên số frame và timestamp, nên `pts_time`/`frame_idx` vẫn đúng. |
| **Cache một năm (`immutable`).** | Ghi đè một file video tại chỗ sẽ không có hiệu lực. | Khi cần thay video, đăng lên key mới, như `Videos_Web/` và `Videos_Web_v2/` đã làm. |
| Từng có response 404 cũ của `M01_V001.mp4` bị cache. Lần kiểm 2026-09-24 thì đã trả 206. | — | Nếu gặp lại, thêm `?v=<ETag>` vào URL hoặc purge cache của URL đó. |
| Chữ không liên quan trong ảnh: ticker M ở 90–96 % chiều cao, banner N ở 4 % trên cùng. | OCR hoặc tìm kiếm theo chữ có thể khớp nhầm tin khác. | Cắt vùng khi OCR; banner N lại hữu ích để lấy tên camera và giờ. |

## 7. Code mẫu (JavaScript)

```js
const KEYFRAME_BASE = "https://keyframe.baoencoder.site";
const VIDEO_BASE = "https://video.baoencoder.site";

/** "M01_V001" -> "M01", "N001-V001" -> "N001", "S01-V012" -> "S01" */
export function groupOf(videoId) {
  return videoId.split(/[_-]/)[0];
}

export function keyframeUrl(videoId, n) {
  return `${KEYFRAME_BASE}/Keyframes/Keyframes_${groupOf(videoId)}/${videoId}/${String(n).padStart(3, "0")}.jpg`;
}

/** `reencoded`: Set các video_id trong Phụ lục A / video_overrides.json. */
export function videoUrl(videoId, reencoded) {
  const group = groupOf(videoId);
  const root = reencoded.has(videoId) ? "Videos_Web_v2" : group.startsWith("N") ? "Videos_Web" : "Videos";
  return `${VIDEO_BASE}/${root}/Videos_${group}/${videoId}.mp4`;
}

/** Tua tới một keyframe; `row` là một dòng của media_manifest_*.csv. */
export function seekToKeyframe(video, row) {
  video.currentTime = Number(row.pts_time);
}
```

## 8. Đã kiểm chứng

- **Đủ bản sửa:** 298/298 video N gốc đều có bản `Videos_Web/` tương ứng, đúng pipeline, `source-etag` khớp ETag bản gốc hiện tại, cùng kích thước.
- **Phát được trên Chrome 151:** bản `Videos_Web/` của N001-V001, N002-V001, N043-V004, N055-V001, N070-V002, N078-V001, N100-V003 đều phát và tua tới khoảng 4:30 được. Bản gốc N070-V002 thì không phát được.
- **Mã hoá lại:** 103/103 bản `Videos_Web_v2/` đã publish; bản nào cũng giải mã strict sạch và mọi keyframe đều có frame trùng thời điểm (lệch ≤ 2 ms). Trên Chrome, N027-V003 phát liên tục qua 4:30–4:38 (trước đây dừng ở 4:34); N004-V001, N058-V001, N063-V002 và N097-V003 đều phát liên tục trong các đoạn thử.
- **Nguồn gốc lỗi:** file `.mov` gốc BTC của `N033-V002` (trùng kích thước và CRC32 với metadata trên R2) cũng không phát được trên Chrome (`PIPELINE_ERROR_DECODE`, 0 frame), và giải mã strict chỉ được 5.063/15.002 frame. Bản `.mp4` remux có 15.002 packet giống hệt về kích thước, thứ tự và keyframe. Vậy lỗi SEI có sẵn trong bản ghi của camera, bước upload không gây ra.
- **Keyframe:** `https://keyframe.baoencoder.site` trả 206 cho ảnh đầu và ảnh cuối của các video L, M, N, S01 đã thử (ví dụ `S01-V007/75123.jpg`).
- **Registry:** SHA-256 của ba file `frame_registry.parquet` khớp bản đã audit (xem handoff keyframe §3).

## Phụ lục A: 103 video N phát từ `Videos_Web_v2/`

Theo từng camera (`group`: video):

- N004: V001, V002, V003 · N007: V001, V002, V003 · N008: V001, V002, V003 · N009: V001, V002
- N015: V001, V002, V003 · N019: V003 · N021: V001, V002, V003 · N027: V001, V002, V003
- N028: V001, V002, V003 · N029: V001, V002, V003 · N030: V001, V002, V003 · N032: V001, V002, V003
- N039: V001, V002, V003 · N040: V001, V002, V003 · N041: V002, V003 · N043: V004, V005
- N045: V001, V002, V003 · N055: V001, V003 · N058: V001, V002, V003 · N059: V001, V002, V003
- N060: V001 · N063: V002 · N066: V001, V002, V003 · N067: V001, V002, V003
- N069: V001, V002, V003 · N073: V001, V002, V003 · N076: V001, V002, V003 · N077: V002, V003
- N079: V002 · N083: V001, V002, V003 · N085: V001, V002 · N087: V001, V002, V003
- N088: V001, V002, V003 · N089: V001, V002, V003 · N092: V001, V002, V003 · N093: V001, V002, V003
- N095: V001, V002, V003 · N096: V001, V002, V003 · N097: V001, V002, V003

Video id đầy đủ có dạng `{group}-{video}`, ví dụ `N027-V003`. 195 video N còn lại
phát từ `Videos_Web/`.

## Tái lập

- Bản vá SEI: [`r2_publish_n_web_videos.py`](../r2_publish_n_web_videos.py). Chạy lại được; video đã có bản đúng `source-etag` sẽ được bỏ qua.
- Bản mã hoá lại: [`r2_reencode_n_damaged_videos.py`](../r2_reencode_n_damaged_videos.py). Cần GPU có NVENC; A100 không có khối này. Script đối chiếu với registry batch 2 đã ghim.
  Chạy song song trên Colab L4 bằng [`r2_reencode_n_damaged_videos_colab.ipynb`](../r2_reencode_n_damaged_videos_colab.ipynb) (dựng bằng [`build_r2_reencode_n_colab.py`](../build_r2_reencode_n_colab.py)); `--sync-overrides` ghi lại `video_overrides.json` từ danh sách thực tế trên R2.
- Chọn URL trong app ClueScope: [`backend/app/media.py`](../backend/app/media.py) (`MediaUrlBuilder.video_url`).
