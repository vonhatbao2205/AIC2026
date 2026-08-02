Ok, mình guide end-to-end theo đúng setup của bạn:

```text
Kaggle input
  /kaggle/input/datasets/loinguyen57/aic-dataset-2025-s1
  /kaggle/input/datasets/loinguyen57/aic-dataset-2025-s2
  /kaggle/input/datasets/loinguyen57/aic-dataset-s2-video

        ↓ upload

Cloudflare R2 bucket
  Keyframes/
  Videos/
  manifest/
```

Và nhớ rule đặc biệt:

```text
SKIP video folders:
  Videos_L25_a1
  Videos_L25_b

USE:
  Videos_L25_a
```

## A. Setup bên Cloudflare R2

### 1. Tạo bucket

Trong màn hình R2 của bạn, bấm  **Create bucket** .

Đặt tên bucket:

```text
aic26-media
```

Cloudflare R2 hỗ trợ S3-compatible API, nên sau này mình sẽ dùng AWS CLI từ Kaggle để upload vào bucket này. Khi tạo API token, Cloudflare sẽ cho endpoint dạng `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`. ([Cloudflare Docs](https://developers.cloudflare.com/r2/get-started/s3/?utm_source=chatgpt.com "S3 · Cloudflare R2 docs"))

### 2. Tạo R2 API token

Vào:

```text
R2 Object Storage
→ Manage R2 API tokens
→ Create API token
```

Chọn:

```text
Permission: Object Read & Write
Bucket scope: aic26-media only
```

Sau khi tạo, copy lại 3 thứ:

```text
Account ID
Access Key ID
Secret Access Key
```

Secret chỉ hiện một lần, nên lưu lại ngay.

### 3. Bật public access để frontend đọc được media

Vào bucket `aic26-media` →  **Settings** .

Bạn có 2 lựa chọn:

```text
Dev/test nhanh: Enable r2.dev public URL
Production: Add Custom Domain, ví dụ media.yourdomain.com
```

Cloudflare nói bucket R2 mặc định không public; muốn expose ra Internet thì phải bật public bucket, qua custom domain hoặc `r2.dev`. `r2.dev` chỉ nên dùng dev/non-production vì có rate limit; custom domain mới dùng được cache/WAF/access control tốt hơn. ([Cloudflare Docs](https://developers.cloudflare.com/r2/data-access/public-buckets/?utm_source=chatgpt.com "Public buckets · Cloudflare R2 docs"))

Nếu bạn chưa có domain riêng, cứ bật `r2.dev` trước để test upload. Sau này đổi sang custom domain không ảnh hưởng object key.

### 4. Thêm CORS

Trong bucket `aic26-media` → **Settings** → **CORS Policy** → Add policy.

Dán JSON này:

```json
[
  {
    "AllowedOrigins": ["*"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": [
      "ETag",
      "Accept-Ranges",
      "Content-Range",
      "Content-Length",
      "Content-Type",
      "cf-cache-status"
    ],
    "MaxAgeSeconds": 3600
  }
]
```

CORS cần nếu frontend của bạn chạy ở domain khác với R2 media domain; Cloudflare cũng lưu ý với custom domain, CORS headers chỉ hiện khi request có `Origin` hợp lệ. ([Cloudflare Docs](https://developers.cloudflare.com/r2/buckets/cors/?utm_source=chatgpt.com "Configure CORS · Cloudflare R2 docs"))

---

# B. Setup Kaggle Secrets

Trong Kaggle Notebook:

```text
Add-ons → Secrets
```

Tạo 4 secret:

```text
R2_ACCOUNT_ID = <account id Cloudflare>
R2_ACCESS_KEY_ID = <access key id>
R2_SECRET_ACCESS_KEY = <secret access key>
R2_BUCKET = aic26-media
```

Không hardcode secret vào notebook.

---

# C. Code Kaggle upload end-to-end

Chạy từng cell theo thứ tự dưới đây.

## Cell 1 — Install AWS CLI

```python
!pip -q install awscli
```

## Cell 2 — Load R2 credentials

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()

os.environ["AWS_ACCESS_KEY_ID"] = secrets.get_secret("R2_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = secrets.get_secret("R2_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = "auto"

account_id = secrets.get_secret("R2_ACCOUNT_ID")
bucket = secrets.get_secret("R2_BUCKET")

os.environ["R2_ENDPOINT"] = f"https://{account_id}.r2.cloudflarestorage.com"
os.environ["R2_BUCKET"] = bucket

print("R2 endpoint:", os.environ["R2_ENDPOINT"])
print("R2 bucket:", os.environ["R2_BUCKET"])
```

## Cell 3 — Test connection tới R2

```python
!aws --endpoint-url "$R2_ENDPOINT" s3 ls "s3://$R2_BUCKET"
```

Nếu không lỗi là connect OK.

---

# D. Khai báo path Kaggle và rule exclude

## Cell 4 — Config dataset paths

```python
from pathlib import Path
import re
import subprocess
import pandas as pd
from collections import defaultdict

DATASET_ROOTS = [
    Path("/kaggle/input/datasets/loinguyen57/aic-dataset-2025-s1"),
    Path("/kaggle/input/datasets/loinguyen57/aic-dataset-2025-s2"),
    Path("/kaggle/input/datasets/loinguyen57/aic-dataset-s2-video"),
]

EXCLUDE_VIDEO_FOLDERS = {
    "Videos_L25_a1",
    "Videos_L25_b",
}

for root in DATASET_ROOTS:
    print(root, "exists:", root.exists())
```

Kết quả cả 3 dòng phải là `exists: True`.

## Cell 5 — Helper functions

```python
def get_keyframe_group(folder_name: str):
    """
    Keyframes_L21   -> L21
    Keyframes_L26_a -> L26
    Keyframes_K01   -> K01
    """
    m = re.match(r"Keyframes_([LK]\d+)", folder_name)
    return m.group(1) if m else None


def get_video_group(folder_name: str):
    """
    Videos_L21_a  -> L21
    Videos_L25_a  -> L25
    Videos_L25_a1 -> L25
    Videos_K01    -> K01
    """
    m = re.match(r"Videos_([LK]\d+)", folder_name)
    return m.group(1) if m else None


def r2_sync(src: Path, dst_prefix: str, dry_run: bool = False):
    cmd = [
        "aws",
        "--endpoint-url", os.environ["R2_ENDPOINT"],
        "s3", "sync",
        str(src),
        f"s3://{os.environ['R2_BUCKET']}/{dst_prefix}",
        "--size-only",
        "--no-progress",
        "--only-show-errors",
        "--cache-control", "public, max-age=31536000, immutable",
    ]

    if dry_run:
        cmd.append("--dryrun")

    print("\nSYNC:")
    print("  FROM:", src)
    print("  TO:  ", f"s3://{os.environ['R2_BUCKET']}/{dst_prefix}")
    subprocess.run(cmd, check=True)
```

---

# E. Scan dataset trước khi upload

## Cell 6 — In ra folder sẽ upload

```python
keyframe_folders = []
video_folders = []
skipped_video_folders = []

for dataset_root in DATASET_ROOTS:
    for folder in dataset_root.iterdir():
        if not folder.is_dir():
            continue

        kf_group = get_keyframe_group(folder.name)
        if kf_group and (folder / "keyframes").exists():
            keyframe_folders.append((folder, kf_group))

        vd_group = get_video_group(folder.name)
        if vd_group and (folder / "video").exists():
            if folder.name in EXCLUDE_VIDEO_FOLDERS:
                skipped_video_folders.append((folder, vd_group))
            else:
                video_folders.append((folder, vd_group))

print("Keyframe folders:", len(keyframe_folders))
for f, g in keyframe_folders:
    print(" ", g, f)

print("\nVideo folders to upload:", len(video_folders))
for f, g in video_folders:
    print(" ", g, f)

print("\nVideo folders skipped:", len(skipped_video_folders))
for f, g in skipped_video_folders:
    print(" ", g, f)
```

Bạn phải thấy:

```text
Video folders skipped:
  L25 .../Videos_L25_a1
  L25 .../Videos_L25_b
```

---

# F. Check collision trước khi upload video

Vì mình gom:

```text
Videos_L25_a/video/*.mp4 → Videos/Videos_L25/*.mp4
```

nên phải check xem có file trùng tên không.

## Cell 7 — Check duplicate destination keys

```python
video_dest_map = defaultdict(list)

for folder, group in video_folders:
    video_dir = folder / "video"

    for mp4 in video_dir.glob("*.mp4"):
        dest_key = f"Videos/Videos_{group}/{mp4.name}"
        video_dest_map[dest_key].append(str(mp4))

collisions = {k: v for k, v in video_dest_map.items() if len(v) > 1}

print("Video destination collisions:", len(collisions))

for k, v in list(collisions.items())[:50]:
    print("\nCOLLISION:", k)
    for src in v:
        print("  ", src)
```

Kỳ vọng:

```text
Video destination collisions: 0
```

Nếu khác 0 thì dừng lại, chưa upload.

---

# G. Dry run trước

## Cell 8 — Dry run keyframes + videos

```python
for folder, group in keyframe_folders:
    src = folder / "keyframes"
    dst = f"Keyframes/Keyframes_{group}/"
    r2_sync(src, dst, dry_run=True)

for folder, group in video_folders:
    src = folder / "video"
    dst = f"Videos/Videos_{group}/"
    r2_sync(src, dst, dry_run=True)
```

Dry run chỉ in ra file sẽ upload, không upload thật.

---

# H. Upload thật lên R2

Cấu trúc R2 sau upload sẽ là:

```text
Keyframes/
  Keyframes_L21/
    L21_V001/
      001.jpg
      002.jpg
  Keyframes_K01/
    K01_V001/
      001.jpg

Videos/
  Videos_L21/
    L21_V001.mp4
  Videos_K01/
    K01_V001.mp4

manifest/
  media_manifest.csv
```

## Cell 9 — Upload keyframes

```python
for folder, group in keyframe_folders:
    src = folder / "keyframes"
    dst = f"Keyframes/Keyframes_{group}/"
    r2_sync(src, dst, dry_run=False)
```

## Cell 10 — Upload videos

```python
for folder, group in video_folders:
    src = folder / "video"
    dst = f"Videos/Videos_{group}/"
    r2_sync(src, dst, dry_run=False)
```

Nếu Kaggle disconnect giữa chừng, chạy lại cell này được. `aws s3 sync --size-only` sẽ skip file đã upload đủ size.

---

# I. Tạo manifest cho backend

## Cell 11 — Generate `media_manifest.csv`

```python
rows = []

# Keyframes
for folder, group in keyframe_folders:
    src = folder / "keyframes"

    for img in src.glob("*/*.jpg"):
        video_id = img.parent.name
        frame_name = img.name
        frame_stem = img.stem
        frame_idx = int(frame_stem) if frame_stem.isdigit() else None

        rows.append({
            "kind": "keyframe",
            "group": group,
            "video_id": video_id,
            "frame_name": frame_name,
            "frame_idx": frame_idx,
            "source_path": str(img),
            "r2_key": f"Keyframes/Keyframes_{group}/{video_id}/{frame_name}",
        })

# Videos
for folder, group in video_folders:
    src = folder / "video"

    for mp4 in src.glob("*.mp4"):
        video_id = mp4.stem

        rows.append({
            "kind": "video",
            "group": group,
            "video_id": video_id,
            "frame_name": None,
            "frame_idx": None,
            "source_path": str(mp4),
            "r2_key": f"Videos/Videos_{group}/{mp4.name}",
        })

df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/media_manifest.csv", index=False)

print("Manifest rows:", len(df))
print(df["kind"].value_counts())
df.head()
```

## Cell 12 — Upload manifest lên R2

```python
!aws --endpoint-url "$R2_ENDPOINT" s3 cp \
  /kaggle/working/media_manifest.csv \
  "s3://$R2_BUCKET/manifest/media_manifest.csv" \
  --content-type "text/csv" \
  --cache-control "public, max-age=3600"
```

---

# J. Verify sau khi upload

## Cell 13 — List R2 folders

```python
!aws --endpoint-url "$R2_ENDPOINT" s3 ls "s3://$R2_BUCKET/" 
```

```python
!aws --endpoint-url "$R2_ENDPOINT" s3 ls "s3://$R2_BUCKET/Keyframes/" 
```

```python
!aws --endpoint-url "$R2_ENDPOINT" s3 ls "s3://$R2_BUCKET/Videos/" 
```

```python
!aws --endpoint-url "$R2_ENDPOINT" s3 ls "s3://$R2_BUCKET/manifest/" 
```

## Cell 14 — Check không upload nhầm L25 duplicate

```python
!aws --endpoint-url "$R2_ENDPOINT" s3 ls "s3://$R2_BUCKET/Videos/Videos_L25/" | head -20
```

Check số lượng file local từ `Videos_L25_a` so với R2:

```python
local_l25_a = None

for folder, group in video_folders:
    if folder.name == "Videos_L25_a":
        local_l25_a = folder / "video"

if local_l25_a:
    local_count = len(list(local_l25_a.glob("*.mp4")))
    print("Local Videos_L25_a count:", local_count)
else:
    print("Videos_L25_a not found in upload list")
```

---

# K. URL backend sẽ dùng

Nếu bạn dùng public URL/custom domain, ví dụ:

```text
https://media.yourdomain.com
```

thì backend build URL như sau:

```python
MEDIA_BASE_URL = "https://media.yourdomain.com"

def group_from_video_id(video_id: str) -> str:
    return video_id.split("_")[0]  # L21, L25, K01

def keyframe_url(video_id: str, frame_name: str) -> str:
    group = group_from_video_id(video_id)
    return f"{MEDIA_BASE_URL}/Keyframes/Keyframes_{group}/{video_id}/{frame_name}"

def video_url(video_id: str) -> str:
    group = group_from_video_id(video_id)
    return f"{MEDIA_BASE_URL}/Videos/Videos_{group}/{video_id}.mp4"
```

Ví dụ:

```python
print(keyframe_url("L21_V001", "001.jpg"))
print(video_url("K01_V001"))
```

Output:

```text
https://media.yourdomain.com/Keyframes/Keyframes_L21/L21_V001/001.jpg
https://media.yourdomain.com/Videos/Videos_K01/K01_V001.mp4
```

Nếu đang dùng `r2.dev`, thay `MEDIA_BASE_URL` bằng public dev URL của bucket.

---

# L. Test bằng browser hoặc curl

Sau khi bật public URL, test 1 file thật.

Ví dụ nếu public base URL là:

```text
https://pub-xxxx.r2.dev
```

test:

```bash
curl -I "https://pub-xxxx.r2.dev/manifest/media_manifest.csv"
```

Test ảnh:

```bash
curl -I "https://pub-xxxx.r2.dev/Keyframes/Keyframes_K01/K01_V001/001.jpg"
```

Test video range:

```bash
curl -I -H "Range: bytes=0-1023" "https://pub-xxxx.r2.dev/Videos/Videos_K01/K01_V001.mp4"
```

Nếu video trả được header hợp lý thì frontend `<video>` có thể load/tua.

---

## Checklist cuối cùng

```text
Cloudflare:
  [ ] Created bucket: aic26-media
  [ ] Created R2 API token: Object Read & Write, scoped to bucket
  [ ] Saved Account ID, Access Key ID, Secret Access Key
  [ ] Enabled r2.dev or custom domain
  [ ] Added CORS policy

Kaggle:
  [ ] Added 4 secrets
  [ ] Test aws s3 ls OK
  [ ] Scan folder OK
  [ ] Confirm skipped Videos_L25_a1 and Videos_L25_b
  [ ] Collision count = 0
  [ ] Dry run OK
  [ ] Upload keyframes
  [ ] Upload videos
  [ ] Generate/upload media_manifest.csv

Backend:
  [ ] Use R2 URL, not base64 images
  [ ] Build keyframe URL from video_id + frame_name
  [ ] Build video URL from video_id
```

Chạy thực tế thì bạn làm từ  **Cell 1 → Cell 14** . Nếu upload bị ngắt, chỉ cần chạy lại  **Cell 9** ,  **Cell 10** , rồi  **Cell 12** .
