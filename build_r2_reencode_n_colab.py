#!/usr/bin/env python3
"""Build the Colab (L4) notebook that re-encodes the damaged N videos.

The notebook embeds the exact modules this repo runs locally, so a Colab worker
and a local worker apply the same checks and write the same R2 keys:

    elastic_upload.py               (R2 credentials helper)
    batch2_keyframes.py             (pinned batch-2 registry)
    r2_publish_n_web_videos.py      (frame counting)
    r2_reencode_n_damaged_videos.py (the re-encode itself)

The Colab worker takes the list from the end (`--reverse`) while the local one
takes it from the start; each skips a video the other already published.

    python build_r2_reencode_n_colab.py --videos-file n_damaged.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODULES = (
    "elastic_upload.py",
    "batch2_keyframes.py",
    "r2_publish_n_web_videos.py",
    "r2_reencode_n_damaged_videos.py",
)
REGISTRY_URL = "https://pub-5010c807b73743ba82d6c40852ae9e6b.r2.dev/manifest/infoshoot_n/final/frame_registry.parquet"
# A release build, not `master`: master needs NVENC API 13.1 (driver >= 610), newer
# than Colab's 580 driver; n8.1 matches the local ffmpeg 8.1 and runs on 570+.
FFMPEG_DIR = "/content/ffmpeg-n8.1"
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def build(root: Path, videos: list[str]) -> dict:
    cells = [
        markdown(f"""
# AIC 2026 — Mã hoá lại video N bị hỏng frame (Colab L4)

Notebook này chạy `r2_reencode_n_damaged_videos.py` trên GPU **L4**. L4 có NVENC; **A100 thì không**, nên đừng chọn A100.

- Nguồn: bản gốc `aic26-media/Videos/Videos_Nxxx/<id>.mp4` (chỉ đọc).
- Đích: `aic26-media/Videos_Web_v2/Videos_Nxxx/<id>.mp4`.
- Mỗi video: giải mã lenient → mã hoá H.264 bằng NVENC, giữ nguyên timestamp → kiểm giải mã strict sạch, thời điểm từng frame ≤ 2 ms, mọi keyframe trong registry có frame tương ứng → upload.
- Máy local chạy danh sách từ đầu; notebook này chạy **từ cuối** (`--reverse`). Video bên kia đã publish sẽ tự bỏ qua. Chạy lại ô cuối là tiếp tục từ chỗ dừng.

**Chuẩn bị:** Runtime → Change runtime type → **L4 GPU**. Colab Secrets (biểu tượng chìa khoá) cần có `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ACCOUNT_ID`, đều bật *Notebook access*.

{len(videos)} video trong danh sách.
"""),
        markdown("## 1. GPU, ffmpeg có NVENC, thư viện"),
        code(f"""
%%bash
set -e
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cd /content
if [ ! -x {FFMPEG_DIR}/bin/ffmpeg ]; then
  curl -sSL -o ffmpeg.tar.xz "{FFMPEG_URL}"
  mkdir -p {FFMPEG_DIR} && tar -xJf ffmpeg.tar.xz -C {FFMPEG_DIR} --strip-components=1 && rm ffmpeg.tar.xz
fi
{FFMPEG_DIR}/bin/ffmpeg -hide_banner -version | head -1
pip install -q boto3 pyarrow
"""),
        code(f"""
import os, re, subprocess
os.environ["PATH"] = "{FFMPEG_DIR}/bin:" + os.environ["PATH"]
# Colab keeps the NVIDIA driver libraries (libnvidia-encode) here.
os.environ["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia:" + os.environ.get("LD_LIBRARY_PATH", "")
smoke = subprocess.run(
    ["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25",
     "-t", "2", "-c:v", "h264_nvenc", "-f", "null", "-"],
    capture_output=True, text=True,
)
if smoke.returncode != 0:
    print("\\n".join(l for l in smoke.stderr.splitlines() if re.search(r"nvenc|driver|cuda|cannot|error", l, re.I)))
    raise SystemExit("NVENC không chạy được — xem các dòng trên (runtime có phải L4 không?)")
print("NVENC OK, CPU:", os.cpu_count())
"""),
        markdown("## 2. Credential R2 (từ Colab Secrets, không in ra)"),
        code("""
import os
from google.colab import userdata
for name in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ACCOUNT_ID"):
    value = userdata.get(name)
    assert value, f"Thiếu Colab Secret {name}"
    os.environ[name] = value.strip()
print("R2 credentials loaded")
"""),
        markdown("## 3. Code (nguyên văn từ repo ClueScope)"),
    ]
    for module in MODULES:
        source = (root / module).read_text(encoding="utf-8")
        cells.append(code(f"%%writefile /content/{module}\n{source}"))
    cells += [
        markdown("## 4. Registry keyframe N (SHA-256 được script kiểm) và danh sách video"),
        code(f"""
import json, urllib.request
from pathlib import Path
registry = Path("/content/keyframe_batch2/infoshoot_n/final/frame_registry.parquet")
registry.parent.mkdir(parents=True, exist_ok=True)
if not registry.exists():
    # Cloudflare answers the default Python-urllib User-Agent with 403.
    request = urllib.request.Request("{REGISTRY_URL}", headers={{"User-Agent": "Mozilla/5.0 (aic26-colab)"}})
    with urllib.request.urlopen(request, timeout=120) as response:
        registry.write_bytes(response.read())
Path("/content/n_damaged.json").write_text(json.dumps({{"damaged": {json.dumps(videos)}}}))
print(registry.stat().st_size, "bytes registry;", {len(videos)}, "videos")
"""),
        markdown("## 5. Chạy (từ cuối danh sách). Chạy lại ô này để tiếp tục nếu bị ngắt."),
        code("""
%cd /content
!python -u r2_reencode_n_damaged_videos.py --videos-file /content/n_damaged.json --reverse --workers 4 \\
    --registry-root /content/keyframe_batch2 --work-dir /content/work \\
    --overrides-file /content/video_overrides.json --report /content/reencode_report.json
"""),
        code("""
import json
report = json.load(open("/content/reencode_report.json"))
print(report["summary"])
for row in report["results"]:
    if row["status"] == "failed":
        print(row["video_id"], row["error"][:200])
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "L4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("r2_reencode_n_damaged_videos_colab.ipynb"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    videos = json.loads(args.videos_file.read_text(encoding="utf-8"))["damaged"]
    args.output.write_text(json.dumps(build(root, videos), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(videos)} videos)")


if __name__ == "__main__":
    main()
