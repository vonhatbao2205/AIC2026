# Cloudflare R2 Media Handoff

> Đọc file này khi build frontend/backend cần hiển thị keyframe thumbnail hoặc phát video.
> File này tóm tắt `cloudflare.md` và `cloudflareR2.ipynb`. Không chứa R2 access key/secret.

---

## 1. Mục Tiêu

Media được upload từ Kaggle dataset lên Cloudflare R2 bucket:

```text
aic26-media
```

R2 bucket có 3 prefix chính:

```text
Keyframes/
Videos/
manifest/
```

Backend/frontend không đọc file local Kaggle. Chỉ build public URL từ:

```text
MEDIA_BASE_URL
submit_keyframe_id
video_id
keyframe_n / frame_name
```

---

## 2. Public Media Base URL

R2 bucket mặc định không public. Cần bật một trong hai:

```text
Dev/test: r2.dev public URL
Production: custom domain, ví dụ https://media.yourdomain.com
```

Backend nên đọc base URL từ env:

```text
MEDIA_BASE_URL=https://media.yourdomain.com
```

Không đưa R2 write token/API secret vào frontend. Frontend chỉ cần public media URL.

---

## 3. R2 Object Key Layout

### Keyframes

Upload source:

```text
<dataset>/Keyframes_<group>/keyframes/<video_id>/<frame_name>
```

R2 destination:

```text
Keyframes/Keyframes_<group>/<video_id>/<frame_name>
```

Ví dụ:

```text
Keyframes/Keyframes_K01/K01_V001/001.jpg
Keyframes/Keyframes_L21/L21_V001/001.jpg
Keyframes/Keyframes_L26/L26_V001/014.jpg
```

### Videos

Upload source:

```text
<dataset>/Videos_<group>/video/<video_id>.mp4
```

R2 destination:

```text
Videos/Videos_<group>/<video_id>.mp4
```

Ví dụ:

```text
Videos/Videos_K01/K01_V001.mp4
Videos/Videos_L21/L21_V001.mp4
Videos/Videos_L29/L29_V009.mp4
```

---

## 4. Group Rules

`group` lấy từ prefix video/category:

```text
K01_V001 -> K01
L21_V001 -> L21
L26_V001 -> L26
```

Helper trong notebook:

```python
def group_from_video_id(video_id: str) -> str:
    return video_id.split("_")[0]
```

Rule đặc biệt cho videos:

```text
SKIP video folders:
  Videos_L25_a1
  Videos_L25_b

USE:
  Videos_L25_a
```

Lý do: các folder L25 có biến thể duplicate. Notebook gom destination về:

```text
Videos/Videos_L25/<video_id>.mp4
```

nên phải skip `Videos_L25_a1` và `Videos_L25_b` để tránh trùng/nhầm.

---

## 5. Build URL Cho Backend/Frontend

### Từ `submit_keyframe_id`

Trong Elastic/Milvus, keyframe public ID chuẩn là:

```text
submit_keyframe_id = <group>/<video_id>/<frame_3_digits>
K01/K01_V001/001
L26/L26_V001/014
```

Build keyframe URL:

```python
def keyframe_url_from_submit_id(media_base_url: str, submit_keyframe_id: str) -> str:
    group, video_id, frame = submit_keyframe_id.split("/")
    frame_name = frame if frame.endswith(".jpg") else f"{frame}.jpg"
    return f"{media_base_url}/Keyframes/Keyframes_{group}/{video_id}/{frame_name}"
```

Ví dụ:

```text
MEDIA_BASE_URL=https://media.yourdomain.com
submit_keyframe_id=K01/K01_V001/001

=> https://media.yourdomain.com/Keyframes/Keyframes_K01/K01_V001/001.jpg
```

### Từ `video_id + keyframe_n`

Nếu backend có:

```text
video_id=K01_V001
keyframe_n=1
```

thì:

```python
def keyframe_url(media_base_url: str, video_id: str, keyframe_n: int) -> str:
    group = video_id.split("_")[0]
    frame_name = f"{keyframe_n:03d}.jpg"
    return f"{media_base_url}/Keyframes/Keyframes_{group}/{video_id}/{frame_name}"
```

### Video URL

```python
def video_url(media_base_url: str, video_id: str) -> str:
    group = video_id.split("_")[0]
    return f"{media_base_url}/Videos/Videos_{group}/{video_id}.mp4"
```

Ví dụ:

```text
video_id=K01_V001

=> https://media.yourdomain.com/Videos/Videos_K01/K01_V001.mp4
```

---

## 6. TypeScript Helpers

Frontend/backend TypeScript có thể dùng:

```ts
export function keyframeUrlFromSubmitId(baseUrl: string, submitKeyframeId: string): string {
  const [group, videoId, frame] = submitKeyframeId.split("/");
  const frameName = frame.endsWith(".jpg") ? frame : `${frame}.jpg`;
  return `${baseUrl}/Keyframes/Keyframes_${group}/${videoId}/${frameName}`;
}

export function keyframeUrl(baseUrl: string, videoId: string, keyframeN: number): string {
  const group = videoId.split("_")[0];
  const frameName = `${String(keyframeN).padStart(3, "0")}.jpg`;
  return `${baseUrl}/Keyframes/Keyframes_${group}/${videoId}/${frameName}`;
}

export function videoUrl(baseUrl: string, videoId: string): string {
  const group = videoId.split("_")[0];
  return `${baseUrl}/Videos/Videos_${group}/${videoId}.mp4`;
}
```

---

## 7. Không Dùng `image_path` / `source_path` Để Truy Xuất

Không dùng các path local này để build media URL:

```text
image_path
source_path
```

Lý do:

- OCR `image_path` có thể là absolute path từ Kaggle hoặc Colab, ví dụ `/kaggle/input/...` hoặc `/content/keyframes/...`.
- Manifest `source_path` là đường dẫn local trong Kaggle notebook.
- Các path này phụ thuộc môi trường upload, không ổn định cho backend/frontend.

Key nên dùng:

```text
submit_keyframe_id      # keyframe public id
video_id
keyframe_n
frame_name
r2_key                  # nếu đọc từ manifest
```

Nếu API muốn field tên `image_id`, hãy đặt:

```text
image_id = submit_keyframe_id
```

---

## 8. Manifest

Notebook tạo và upload:

```text
manifest/media_manifest.csv
```

Manifest rows có dạng:

```text
kind          # keyframe | video
group         # K01, L21, L26, ...
video_id      # K01_V001
frame_name    # 001.jpg với keyframe, null với video
frame_idx     # 1 với keyframe 001.jpg, null với video
source_path   # local Kaggle path, chỉ debug
r2_key        # object key trên R2
```

Backend có thể dùng manifest để validate media tồn tại hoặc làm fallback lookup. Frontend không cần tải toàn bộ manifest nếu backend đã trả URL trực tiếp.

Manifest URL:

```text
<MEDIA_BASE_URL>/manifest/media_manifest.csv
```

---

## 9. Upload Notebook Tóm Tắt

Notebook `cloudflareR2.ipynb` dùng AWS CLI vì R2 S3-compatible:

```bash
aws --endpoint-url "$R2_ENDPOINT" s3 sync <src> s3://$R2_BUCKET/<dst_prefix> \
  --size-only \
  --no-progress \
  --only-show-errors \
  --cache-control "public, max-age=31536000, immutable"
```

`--size-only` làm upload rerun-safe:

```text
Đã upload đủ size      -> skip
Chưa có trên R2        -> upload
Size khác / thiếu file -> upload lại
```

Không phải resume byte-by-byte trong một object đang upload dở. Với video lớn, nếu disconnect giữa file thì rerun sẽ upload lại file đó.

Lưu ý theo notebook hiện tại:

- Cell 9 upload keyframes đang active.
- Cell 10 upload videos đang comment trong `cloudflareR2.ipynb`; cần un-comment hoặc chạy cell tương đương để upload videos.
- `cloudflare.md` có đầy đủ version Cell 9 keyframes + Cell 10 videos.

---

## 10. Ongoing Multipart Upload

Nếu R2 UI hiện `Ongoing Multipart Upload`, đó là video/key lớn đang upload dở hoặc upload bị ngắt giữa chừng.

Không coi nó là object `.mp4` hoàn chỉnh. Nếu notebook vẫn đang chạy, chờ upload xong. Nếu notebook đã dừng, có thể abort multipart upload dở rồi chạy lại sync.

Kiểm tra multipart uploads:

```bash
aws --endpoint-url "$R2_ENDPOINT" s3api list-multipart-uploads \
  --bucket "$R2_BUCKET" \
  --prefix "Videos/" \
  --query 'Uploads[].{Key:Key,UploadId:UploadId,Initiated:Initiated}' \
  --output table
```

Không chạy 2 notebook cùng upload cùng video prefix/object key. Notebook thứ hai không biết notebook thứ nhất đang upload dở; nó chỉ skip object đã commit đủ size.

---

## 11. CORS Và Video Range

CORS policy nên cho frontend đọc:

```text
AllowedMethods: GET, HEAD
ExposeHeaders:
  ETag
  Accept-Ranges
  Content-Range
  Content-Length
  Content-Type
  cf-cache-status
```

Test video range:

```bash
curl -I -H "Range: bytes=0-1023" \
  "<MEDIA_BASE_URL>/Videos/Videos_K01/K01_V001.mp4"
```

Nếu header range hợp lý, frontend `<video>` có thể load/tua.

---

## 12. Backend Result Shape Gợi Ý

Backend nên trả media URL đã build sẵn để frontend không phải biết R2 layout quá nhiều:

```json
{
  "video_id": "K01_V001",
  "submit_keyframe_id": "K01/K01_V001/001",
  "keyframe_n": 1,
  "keyframe_url": "https://media.yourdomain.com/Keyframes/Keyframes_K01/K01_V001/001.jpg",
  "video_url": "https://media.yourdomain.com/Videos/Videos_K01/K01_V001.mp4",
  "evidence": []
}
```

Frontend vẫn có thể import helper URL builder nếu backend chỉ trả IDs, nhưng backend-built URLs giảm khả năng lệch convention.

---

## 13. Checklist Cho Agent

Backend:

```text
[ ] Read MEDIA_BASE_URL from env/config.
[ ] Treat submit_keyframe_id as image_id/public keyframe id.
[ ] Build keyframe URL from submit_keyframe_id or video_id + keyframe_n.
[ ] Build video URL from video_id.
[ ] Do not use image_path/source_path for retrieval.
[ ] Optionally validate against manifest/media_manifest.csv.
```

Frontend:

```text
[ ] Display keyframe thumbnail from keyframe_url.
[ ] Use video_url for playback.
[ ] Keep submit_keyframe_id for submit/copy.
[ ] Do not expose R2 write credentials.
[ ] Do not rely on Kaggle local paths.
```
