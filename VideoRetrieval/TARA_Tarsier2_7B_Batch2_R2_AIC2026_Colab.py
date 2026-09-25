# %% [markdown]
# # AIC 2026 — TARA (Tarsier2-7B) video-clip embedding cho batch 2 M/N/S01 từ R2 (Colab A100 80 GB)
#
# Notebook E2E này encode **614 video batch 2** trên bucket R2 `aic26-media` — M tin tức 304,
# N camera giao thông 298, S01 đua xe đạp 12 — thành **344.016 clip embedding 3584-d** bằng
# [`bpiyush/TARA`](https://huggingface.co/bpiyush/TARA), ghi Parquet shard và upload
# checkpoint-safe lên Hugging Face Storage Bucket.
#
# ## Khác gì so với notebook batch 1 (L21–L30)
#
# - **Model, venv, prompt EOL, `n_frames=8`, `max_pixels=460.800`, pooling last-token,
#   BF16 + FlashAttention-2, quality gate và schema Parquet giữ nguyên.** Worker TARA
#   (section 7) giữ nguyên từng byte.
# - **Clip scale theo nhóm video, hop 50%.** N (camera giao thông) dùng **4 / 8 / 16 s**: 8
#   frame trên cửa sổ 4 s ≈ một frame mỗi 0,57 s, đủ dày để bắt maneuver ngắn (rẽ, vượt, cắt
#   ngang). M (tin tức) và S01 (đua xe đạp) giữ **8 / 24 / 72 s** như collection L. Scale nằm
#   trong semantic fingerprint nên đây là collection **mới**, không trộn với L.
# - **Nguồn video là R2 `aic26-media/Videos/`**, tải bằng S3 API (ranged GET song song), kiểm
#   size + ETag multipart (part 32 MiB) với manifest 614 video đã pin SHA-256 (section 4).
# - **S01 AV1.** `decord 0.6.0` (reader chính chủ) không giải mã được AV1. Chạy trước
#   `AIC2026_S01_AV1_to_H264_R2_Colab.ipynb` để thay 11 video AV1 bằng bản H.264 đúng từng
#   frame; notebook này chỉ nhận bản H.264 có metadata `source-etag` khớp bản AV1 đã pin.
# - **Chạy được trên A100 và H100** (`gpu_profiles`, chọn theo tên GPU + VRAM). A100 80 GB đã đo:
#   GPU là nút thắt, 1 worker × batch 8 = 0,623 clip/s, 2 worker = 0,615 clip/s, nên đặt sẵn
#   1 worker × batch 8. A100 40 GB autotune batch. Nếu batch mặc định OOM trên runtime thực tế,
#   notebook tự autotune lại các batch nhỏ hơn thay vì dừng.
#   H100 chưa đo: lần chạy đầu autotune batch và đo 1 vs 2 worker cùng GPU, chỉ giữ 2 worker
#   nếu nhanh hơn ≥ 10%. Wheel FlashAttention 2.8.3 của venv có sẵn kernel sm_80 và sm_90.
# - **Encode theo chunk ≤ 1 shard (1.024 clip)** thay vì cả video. Video S01 dài 2,4–5,6 giờ
#   (8–17 nghìn clip); nếu Colab ngắt, mỗi worker chỉ mất tối đa một chunk.
#
# ## Session
#
# | Session | Category | Clip |
# |---|---|---:|
# | `session_1` | M01–M06 + N001–N010 | 85.787 |
# | `session_2` | M07–M10 + N011–N027 | 86.688 |
# | `session_3` | N028–N084 | 85.777 |
# | `session_4` | N085–N100 (S01 chạy bằng `TARA_Tarsier2_7B_S01_R2_AIC2026_Colab.ipynb`) | 85.764 |
#
# Đo trên A100 80 GB: ~0,62 clip/s, tức mỗi session ≈ 38–39 GPU-giờ (tổng ~154 GPU-giờ).
# Một session dài hơn giới hạn runtime Colab nên sẽ phải Run All lại nhiều lần; mọi shard
# đã commit được skip. Chạy các session trên nhiều runtime song song để rút thời gian.
#
# > Mỗi Colab runtime chỉ đổi `active_session` rồi Run All. Các session dùng chung
# > `output_prefix` nhưng encode category rời nhau; shard đã commit được skip.
#
# Official references:
#
# - https://huggingface.co/bpiyush/TARA
# - https://github.com/bpiyush/TARA
# - https://arxiv.org/abs/2512.13511

# %%
# =========================
# CONFIG — chỉ sửa cell này
# =========================

M_CATEGORIES = [f"M{i:02d}" for i in range(1, 11)]
N_CATEGORIES = [f"N{i:03d}" for i in range(1, 101)]
ALL_CATEGORIES = M_CATEGORIES + N_CATEGORIES + ["S01"]

CONFIG = {
    # Video batch 2 trên Cloudflare R2 (notebook chỉ đọc, không sửa/xóa).
    "source_r2_bucket": "aic26-media",
    "r2_part_size_mib": 32,          # quy ước upload video của repo: ETag multipart part 32 MiB
    "r2_download_concurrency": 16,   # số ranged GET song song cho mỗi video
    "r2_download_attempts": 3,

    # Derived output. Cùng HF bucket với collection L nhưng prefix hoàn toàn mới.
    "output_bucket_id": "Baonenha1/aic26-media",
    "output_prefix": "derived/tara-tarsier2-7b-3584-batch2-clip-v1",
    "reference_output_prefix": "derived/tara-tarsier2-7b-3584-clip-v1",
    # Prefix của lần chạy thử 4/8/16 cho mọi nhóm (đã bỏ); không bao giờ ghi vào đó nữa.
    "abandoned_output_prefixes": ["derived/tara-tarsier2-7b-3584-batch2-clip-4-8-16-v1"],

    # Chạy song song: tạo bốn bản Colab của notebook, mỗi bản chỉ đổi đúng dòng
    # active_session. Assignment phải rời nhau để không có hai runtime ghi cùng shard.
    "active_session": "session_1",  # session_1 | session_2 | session_3 | session_4
    "session_assignments": {
        "session_1": M_CATEGORIES[0:6] + N_CATEGORIES[0:10],   # 85.787 clip
        "session_2": M_CATEGORIES[6:10] + N_CATEGORIES[10:27], # 86.688 clip
        "session_3": N_CATEGORIES[27:84],                      # 85.777 clip
        # S01 để cuối: N chạy trước (~11 giờ) trong lúc notebook S01 chuyển AV1 → H.264.
        "session_4": N_CATEGORIES[84:] + ["S01"],              # 85.764 clip
    },
    # Category giao cho notebook riêng: notebook này bỏ qua khi encode và khi xét _SUCCESS.
    # Giữ trong session_assignments/clip plan để semantic fingerprint và run_config không đổi.
    # S01 → TARA_Tarsier2_7B_S01_R2_AIC2026_Colab.ipynb (work unit = video, 2 session song song).
    "delegated_categories": {"S01": "derived/tara-tarsier2-7b-3584-batch2-s01-clip-v1"},

    # ---- Model chính chủ, pin theo commit ----
    "model_id": "bpiyush/TARA",
    "model_revision": "3e7cb730d86ae10da7eb1c85f17e45ece5a5e353",
    "base_model": "omni-research/Tarsier2-7b-0115",
    "embedding_dim": 3584,
    "attention_implementation": "flash_attention_2",
    "model_weight_dtype": "bfloat16",
    "storage_dtype": "float32",
    "l2_normalize_fp32": True,

    # ---- Semantic preprocessing contract = default chính chủ của TARA ----
    # Không đổi bất kỳ key nào ở đây giữa các lần resume: đổi là sang embedding
    # space khác và mọi vector cũ trong collection thành vô nghĩa.
    "n_frames": 8,                 # tarsier2/default_config.yaml
    "max_pixels": 460_800,         # 1280 * 720 // 2
    "min_pixels": 0,
    "video_eol_prompt": "<video>\nSummary above video in one word:",
    "text_eol_prompt": "<sent>\nSummary above sentence in one word:",
    "expected_default_config": {
        "max_n_frames": 256,
        "n_frames": 8,
        "max_pixels": 460_800,
        "min_pixels": 0,
        "max_seq_len": 16_384,
        "is_training": False,
        "print_data_error": True,
        "do_image_padding": False,
        "do_image_crop": False,
        "do_image_resize": False,
        "video_sampling_strategy": {
            "video_sampler_version": "v1",
            "force_frames_n_divisible": 1,
            "use_multi_images_for_video": True,
        },
        "prompt": "",
        "train_task": "sft",
    },

    # ---- Multi-scale temporal clip plan (batch 2), theo nhóm video ----
    # (tên scale, độ dài window giây, hop giây). Hop = window/2 ⇒ chồng lấn 50%.
    # Tên scale giữ event/sequence/scene để backend không phải sửa; độ dài thật của cửa sổ
    # nằm trong run_config và Arrow metadata `scales` của từng Parquet.
    "scales_by_group": {
        # M tin tức: giữ đúng scale của collection L.
        "M": [["event", 8.0, 4.0], ["sequence", 24.0, 12.0], ["scene", 72.0, 36.0]],
        # N camera giao thông:
        #   event    : chuyển động đơn (xe rẽ trái, đi trái→phải), 8 frame / 4 s ≈ 0,57 s/frame.
        #   sequence : maneuver/tương tác (xe máy tiếp cận → đi ngang → vượt ô tô), ≈ 1,14 s/frame.
        #   scene    : tương tác dài (tiếp cận → tương tác → rời khung), ≈ 2,29 s/frame.
        "N": [["event", 4.0, 2.0], ["sequence", 8.0, 4.0], ["scene", 16.0, 8.0]],
        # S01 đua xe đạp: giữ đúng scale của collection L.
        "S": [["event", 8.0, 4.0], ["sequence", 24.0, 12.0], ["scene", 72.0, 36.0]],
    },
    "clip_time_decimals": 3,

    # ---- Throughput / execution policy (không nằm trong semantic fingerprint) ----
    # GPU được phép và cấu hình mặc định theo loại GPU. Profile đầu tiên có `match` nằm trong tên
    # GPU (vd "NVIDIA A100-SXM4-40GB") và VRAM ≥ `min_vram_gib` được dùng.
    #   A100 80GB (đo 2026-09-23): GPU là nút thắt — batch 1→8 chỉ 0,52→0,55 clip/s, 2 worker
    #     = 0,99× — nên đặt sẵn 1 worker × batch 8 để mỗi lần resume chỉ validate một batch.
    #   A100 40GB (chưa đo): batch 8 không chắc vừa VRAM, nên autotune batch.
    #   H100 (chưa đo): autotune batch và đo 1 vs 2 worker ở lần chạy đầu. GPU nhanh hơn nên
    #     preprocess 1 thread/worker trên CPU có thể thành nút thắt, khi đó 2 worker mới có lợi.
    # batch_size None = autotune theo clip/s trong ngưỡng VRAM. Nếu batch của profile OOM trên
    # runtime thực tế, notebook tự autotune lại với các batch nhỏ hơn.
    "gpu_profiles": {
        "A100-80GB": {"match": "A100", "min_vram_gib": 70, "batch_size": 8, "gpu_workers": 1},
        "A100-40GB": {"match": "A100", "min_vram_gib": 0, "batch_size": None, "gpu_workers": 1},
        "H100": {"match": "H100", "min_vram_gib": 0, "batch_size": None, "gpu_workers": 2},
    },
    # None = lấy theo gpu_profiles; đặt số cụ thể để ép giá trị cho mọi GPU.
    "batch_size": None,
    "batch_candidates": [1, 2, 3, 4, 6, 8],
    "benchmark_rounds": 3,
    "max_vram_fraction": 0.92,     # tổng cho mọi worker; mỗi worker ≤ max_vram_fraction / gpu_workers
    # Số worker TARA tối đa trên cùng GPU (None = theo gpu_profiles). Khi > 1, notebook đo 1 vs
    # nhiều worker trên clip thật và chỉ giữ nhiều worker nếu nhanh hơn ≥ gpu_worker_min_speedup.
    "gpu_workers": None,
    "gpu_worker_min_speedup": 1.10,
    "worker_scaling_clips": 24,    # clip mỗi worker trong phép đo scaling
    # Chunk encode: giao điểm của một video với một shard, tối đa encode_chunk_rows clip.
    "encode_chunk_rows": 1_024,
    # Số video tải sẵn trên disk = số worker + video_prefetch_extra.
    "video_prefetch_extra": 1,
    # PHẢI là 1. transformers 4.45 memoize compiled Jinja template bằng
    # @lru_cache, nên `AssistantTracker` (state của block {% generation %}) là
    # singleton dùng chung; hai lần `apply_chat_template` chạy song song sẽ ném
    # "AssistantTracker should not be reused before closed". Template của TARA
    # có {% generation %} và TarsierProcessor luôn bật return_assistant_tokens_mask,
    # nên mọi clip đều đi qua đúng đường đó. Pool 1 worker vẫn overlap CPU với
    # GPU qua hàng đợi prefetch — đó mới là thứ ta cần, không phải song song.
    # Muốn nhiều luồng preprocess thì tăng gpu_workers (mỗi worker là một process riêng).
    "prep_threads": 1,
    "prefetch_batches": 2,
    "decord_threads": 4,
    # Một số mp4 làm decord hết budget retry khi đọc vài frame cuối ("Unable to handle
    # EOF ... DECORD_EOF_RETRY_MAX=10240"). Nâng hạn mức, và worker còn probe độ dài
    # thực sự decode được lúc mở file.
    "decord_eof_retry_max": 65_536,
    "decode_substitution_window": 8,   # frame lỗi giữa file -> lùi tối đa 8 frame
    "decode_probe_max_steps": 16,
    "video_encode_attempts": 2,        # thử lại một chunk trước khi bỏ cuộc
    "bypass_lm_head": True,        # gated bởi audit; False = dùng nguyên generate()

    # ---- S01: chỉ nhận bản H.264 do notebook S01 tạo ----
    "s01_pipeline": "aic2026-s01-av1-to-h264-v1",
    "s01_transform": "h264-high-yuv420p-crf16-veryfast-aac-copy-faststart-v1",

    # ---- Temporal self test (mục 11) ----
    # Số liệu tham chiếu lấy từ output demo chính chủ trong README của TARA:
    #   video assets/folding_paper.mp4
    #   'someone is folding a paper'   -> 0.6488
    #   'cutting a paper'              -> 0.3952
    #   'someone is unfolding a paper' -> 0.3009
    "self_test_demo_asset": "assets/folding_paper.mp4",
    "self_test_demo_expected": {
        "someone is folding a paper": 0.6488,
        "cutting a paper": 0.3952,
        "someone is unfolding a paper": 0.3009,
    },
    "self_test_demo_tolerance": 0.05,
    # Arrow-of-time chỉ là smoke test của pipeline: camera giao thông có xe đi cả hai chiều
    # và cảnh đèn đỏ đứng yên, nên đảo frame có thể gần như không đổi mà TARA vẫn đúng.
    "self_test_aot_clips": 24,
    "self_test_aot_min_median": 1e-4,

    # ---- Quality gate ----
    "audit_clip_samples": 6,
    "audit_min_cosine_vs_official": 0.999,
    "audit_mean_cosine_vs_official": 0.9995,
    "audit_min_cosine_batch": 0.999,
    "audit_full_video_min_cosine": 0.999,
    "require_frame_index_exact_match": True,

    # ---- Shard / commit ----
    "shard_rows": 1_024,           # 1.024 × 3584 × 4 B ≈ 14 MiB vector/shard
    "upload": True,
    "cleanup_video_after_encode": True,
    "cleanup_local_shards_after_upload": True,
    "verify_video_etag": True,

    # ---- Runtime venv (uv + CPython 3.10 standalone) ----
    "venv_python_version": "3.10",
    "venv_root": "/content/tara-venv",
    "torch_version": "2.5.1",
    "torchvision_version": "0.20.1",
    "torch_index_url": "https://download.pytorch.org/whl/cu121",
    "transformers_version": "4.45.0",
    "flash_attn_version": "2.8.3",
    "flash_attn_wheel_url": (
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    ),
    # Wheel prebuilt chính thức của Dao-AILab, pin theo nội dung để bản cache
    # trên Drive không bao giờ được tin nếu byte không khớp.
    "flash_attn_wheel_bytes": 255_955_166,
    "flash_attn_wheel_sha256": (
        "c59be18fa934e132e5a405bcca6673af1d9d09a0036a9e081133bc7b7fc2992e"
    ),
    # Cache wheel trên Google Drive để các Colab session sau không phải tải lại
    # 244 MiB. Wheel chỉ được restore sau khi verify manifest + size + SHA-256.
    "flash_attn_drive_cache": True,
    "flash_attn_drive_mount_point": "/content/drive",
    "flash_attn_drive_cache_relative": "MyDrive/AIC2026/flash_attn_wheels",
    "numpy_version": "1.26.4",
    "decord_version": "0.6.0",
    # Ba pin dưới đây bám đúng requirements.txt chính chủ của TARA. Pillow nằm
    # thẳng trên image preprocessing path (resize2pixels + Qwen2VLImageProcessor)
    # nên lệch version là lệch pixel — mà audit nội bộ sẽ KHÔNG bắt được, vì nó
    # so production path với official path trong cùng một environment.
    "pillow_version": "10.0.0",
    "sentencepiece_version": "0.2.1",
    "opencv_version": "4.11.0.86",
    "uv_install_url": "https://astral.sh/uv/install.sh",
    "subprocess_heartbeat_seconds": 15,
    "subprocess_silence_warning_seconds": 180,

    # Colab scratch. Không đặt vào Google Drive.
    "scratch_root": "/content/aic_tara_batch2",
    "hf_token_file_candidates": [
        "/content/HF_TOKEN.txt",
        "/content/drive/MyDrive/HF_TOKEN.txt",
    ],
    # Dạng `Account ID: ...`, `Access Key ID: ...`, `Secret Access Key: ...`.
    "r2_credential_file_candidates": [
        "/content/cloudflareR2_api.txt",
        "/content/drive/MyDrive/cloudflareR2_api.txt",
    ],
    "min_free_disk_gib": 60,
}

ACTIVE_SESSION = CONFIG["active_session"]
if ACTIVE_SESSION not in CONFIG["session_assignments"]:
    raise ValueError(f"active_session không tồn tại: {ACTIVE_SESSION}")
CONFIG["categories"] = list(CONFIG["session_assignments"][ACTIVE_SESSION])
CONFIG["session_name"] = f"colab_{ACTIVE_SESSION}"

# Checkpoint bpiyush/TARA @ 3e7cb730 (4 shard safetensors).
EXPECTED_MODEL_TENSOR_PAYLOAD_BYTES = 16_582_751_232

# Số clip sinh ra bởi đúng CONFIG["scales_by_group"] mặc định trên manifest đã pin (section 4).
# Nếu đổi scales thì con số này khác và assert sẽ tự tắt.
REFERENCE_SCALES_BY_GROUP = {
    "M": [["event", 8.0, 4.0], ["sequence", 24.0, 12.0], ["scene", 72.0, 36.0]],
    "N": [["event", 4.0, 2.0], ["sequence", 8.0, 4.0], ["scene", 16.0, 8.0]],
    "S": [["event", 8.0, 4.0], ["sequence", 24.0, 12.0], ["scene", 72.0, 36.0]],
}
REFERENCE_CLIP_TOTALS_BY_GROUP = {"M": 128_945, "N": 154_509, "S": 60_562}


def scales_for(category: str):
    # M01..M10 → "M", N001..N100 → "N", S01 → "S".
    return CONFIG["scales_by_group"][category[0]]

print("Active session:", CONFIG["session_name"], "categories:", CONFIG["categories"])

# %% [markdown]
# ## 1. Dựng venv Python 3.10 cho TARA và kiểm tra runtime Colab
#
# TARA yêu cầu `python==3.10`, `torch==2.5.1+cu121`, `transformers==4.45.0`,
# `numpy==1.26.4`, `flash-attn==2.8.3`. Colab Latest chạy Python 3.13 nên
# `numpy==1.26.4` không có wheel và `transformers==4.45` không tương thích Torch 2.11.
#
# Cell dưới cài `uv`, tải CPython 3.10 standalone, dựng venv riêng và cài đúng
# stack đó. `flash-attn` lấy wheel prebuilt chính thức của Dao-AILab (không build
# source ~40 phút). Kernel Colab **không bị đụng đến** — nó giữ nguyên
# `huggingface_hub` 1.x để dùng Bucket API.

# %%
from collections import deque
from pathlib import Path
import hashlib
import json
import os
import platform
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time


def run_checked(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=True, **kwargs)


def _human_bytes(value):
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def _elapsed_text(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_observable(
    cmd,
    *,
    phase,
    env=None,
    heartbeat_seconds=15,
    silence_warning_seconds=180,
    log_path=None,
    check=True,
):
    """Stream stdout/stderr, lưu log và chứng minh child vẫn sống khi im lặng."""
    command = list(map(str, cmd))
    log_file = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8", buffering=1)

    def emit(message):
        print(message, flush=True)
        if log_file is not None:
            print(message, file=log_file, flush=True)

    started = time.monotonic()
    last_output = started
    output_tail = deque(maxlen=80)
    emit(f"[{phase} START] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"[{phase} CMD] {shlex.join(command)}")
    if log_path is not None:
        emit(f"[{phase} LOG] {log_path}")

    process = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1,
    )
    emit(f"[{phase} PID] {process.pid}")

    output_queue = queue.Queue()
    reader_done = object()

    def read_output():
        try:
            for raw_line in iter(process.stdout.readline, ""):
                output_queue.put(raw_line.rstrip("\r\n"))
        finally:
            output_queue.put(reader_done)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    reader_finished = False
    next_heartbeat = started + heartbeat_seconds

    try:
        while True:
            now = time.monotonic()
            try:
                item = output_queue.get(timeout=max(0.05, min(1.0, next_heartbeat - now)))
            except queue.Empty:
                item = None
            if item is reader_done:
                reader_finished = True
            elif item is not None:
                last_output = time.monotonic()
                output_tail.append(item)
                emit(f"[{phase} OUT {_elapsed_text(last_output - started)}] {item}")
            now = time.monotonic()
            if now >= next_heartbeat and process.poll() is None:
                silent_for = now - last_output
                state = "alive"
                if silent_for >= silence_warning_seconds:
                    state += "; log đã im lặng lâu"
                emit(
                    f"[{phase} HEARTBEAT] pid={process.pid} | state={state}"
                    f" | elapsed={_elapsed_text(now - started)}"
                    f" | last_output={_elapsed_text(silent_for)} ago"
                )
                next_heartbeat = now + heartbeat_seconds
            if process.poll() is not None and reader_finished and output_queue.empty():
                break
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        emit(f"[{phase} ABORTED] pid={process.pid}")
        if log_file is not None:
            log_file.close()
        raise
    finally:
        reader.join(timeout=2)

    return_code = process.wait()
    emit(
        f"[{phase} {'DONE' if return_code == 0 else 'FAILED'}] return_code={return_code} | "
        f"elapsed={_elapsed_text(time.monotonic() - started)}"
    )
    if log_file is not None:
        log_file.close()
    if check and return_code != 0:
        raise RuntimeError(
            f"{phase} thất bại (return_code={return_code}). Full log: {log_path}\n"
            "--- last output ---\n" + "\n".join(output_tail)
        )
    return return_code


SCRATCH_ROOT = Path(CONFIG["scratch_root"]).resolve()
VIDEO_ROOT = SCRATCH_ROOT / "videos"
LOCAL_OUTPUT_ROOT = SCRATCH_ROOT / "output"
STATE_ROOT = SCRATCH_ROOT / "state"
MODEL_ROOT = SCRATCH_ROOT / "model"
NPZ_ROOT = SCRATCH_ROOT / "npz"
LOG_ROOT = SCRATCH_ROOT / "logs"
for directory in (VIDEO_ROOT, LOCAL_OUTPUT_ROOT, STATE_ROOT, MODEL_ROOT, NPZ_ROOT, LOG_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

VENV_ROOT = Path(CONFIG["venv_root"]).resolve()
VENV_PYTHON = VENV_ROOT / "bin" / "python"
VENV_MARKER = VENV_ROOT / ".aic_tara_ready.json"

VENV_CONTRACT = {
    "schema_version": 2,
    "python": CONFIG["venv_python_version"],
    "torch": CONFIG["torch_version"],
    "torchvision": CONFIG["torchvision_version"],
    "transformers": CONFIG["transformers_version"],
    "flash_attn": CONFIG["flash_attn_version"],
    "numpy": CONFIG["numpy_version"],
    "decord": CONFIG["decord_version"],
    "pillow": CONFIG["pillow_version"],
    "sentencepiece": CONFIG["sentencepiece_version"],
    "opencv": CONFIG["opencv_version"],
    "flash_attn_wheel_url": CONFIG["flash_attn_wheel_url"],
    "flash_attn_wheel_sha256": CONFIG["flash_attn_wheel_sha256"],
}

# Bộ dep tối thiểu để `import shared.utils` + `import modeling_tara` chạy được.
# shared/utils/__init__.py import cả visualize (matplotlib, cv2, sklearn,
# seaborn, IPython) nên phải có mặt; worker vẫn có fallback stub nếu thiếu.
TARA_VENV_PACKAGES = [
    f"transformers=={CONFIG['transformers_version']}",
    "tokenizers==0.20.3",
    "accelerate==0.34.2",
    "huggingface-hub==0.35.3",
    "safetensors==0.4.5",
    f"sentencepiece=={CONFIG['sentencepiece_version']}",
    f"numpy=={CONFIG['numpy_version']}",
    "pandas==2.2.3",
    "pyarrow==17.0.0",
    f"Pillow=={CONFIG['pillow_version']}",
    f"decord=={CONFIG['decord_version']}",
    "einops==0.8.1",
    f"opencv-python-headless=={CONFIG['opencv_version']}",
    "matplotlib==3.9.2",
    "seaborn==0.13.2",
    "scikit-learn==1.5.2",
    "ipython==8.27.0",
    "ipywidgets==8.1.5",
    "ffmpeg-python==0.2.0",
    "func-timeout==4.3.5",
    "termcolor==2.4.0",
    "PyYAML==6.0.2",
    "tqdm==4.67.1",
    "psutil==6.1.0",
]


def _canonical_sha256(payload):
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


FLASH_ATTN_WHEEL_CONTRACT = {
    "schema_version": 1,
    "artifact": "flash-attn-official-prebuilt-wheel",
    "flash_attn_version": CONFIG["flash_attn_version"],
    "url": CONFIG["flash_attn_wheel_url"],
    "wheel_bytes": CONFIG["flash_attn_wheel_bytes"],
    "wheel_sha256": CONFIG["flash_attn_wheel_sha256"],
    "python_tag": "cp310",
    "torch": CONFIG["torch_version"],
    "cuda": "12.1",
    "cxx11_abi": False,
}
FLASH_ATTN_WHEEL_NAME = CONFIG["flash_attn_wheel_url"].rsplit("/", 1)[-1]
FLASH_ATTN_LOCAL_DIR = SCRATCH_ROOT / "flash_attn_wheels"
FLASH_ATTN_LOCAL_DIR.mkdir(parents=True, exist_ok=True)


def mount_flash_attn_drive_cache():
    """Mount Drive nếu cần; lỗi cache không bao giờ được chặn đường tải lại."""
    if not CONFIG["flash_attn_drive_cache"]:
        print("[FA2 DRIVE CACHE] disabled by config.", flush=True)
        return None
    mount_point = Path(CONFIG["flash_attn_drive_mount_point"]).resolve()
    relative = Path(CONFIG["flash_attn_drive_cache_relative"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("flash_attn_drive_cache_relative phải là relative path an toàn.")
    try:
        my_drive = mount_point / "MyDrive"
        if not my_drive.is_dir():
            print(f"[FA2 DRIVE CACHE] Mount Google Drive tại {mount_point}...", flush=True)
            from google.colab import drive

            drive.mount(str(mount_point), force_remount=False)
        if not my_drive.is_dir():
            raise RuntimeError(f"Drive mount thiếu directory: {my_drive}")
        cache_root = mount_point / relative
        cache_root.mkdir(parents=True, exist_ok=True)
        contract_key = _canonical_sha256(FLASH_ATTN_WHEEL_CONTRACT)[:12]
        cache_dir = cache_root / (
            f"flash-attn-{CONFIG['flash_attn_version']}_cp310_torch-"
            f"{CONFIG['torch_version']}_cu121_abiFALSE_{contract_key}"
        )
        print(f"[FA2 DRIVE CACHE] dir={cache_dir}", flush=True)
        return cache_dir
    except Exception as exc:
        print(
            f"[FA2 DRIVE CACHE WARNING] Không dùng được Drive cache: "
            f"{type(exc).__name__}: {exc}. Sẽ tải wheel trực tiếp từ GitHub.",
            flush=True,
        )
        return None


def restore_flash_attn_wheel_from_drive(cache_dir):
    """Copy cache về local rồi verify manifest, size và SHA-256 trước khi tin."""
    if cache_dir is None:
        return None
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.is_file():
        print(f"[FA2 DRIVE CACHE MISS] Chưa có {manifest_path}", flush=True)
        return None
    partial_path = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != FLASH_ATTN_WHEEL_CONTRACT:
            raise RuntimeError("wheel contract mismatch")
        wheel_name = manifest.get("wheel_filename")
        if not isinstance(wheel_name, str) or Path(wheel_name).name != wheel_name:
            raise RuntimeError("wheel_filename không an toàn")
        drive_wheel = Path(cache_dir) / wheel_name
        if not drive_wheel.is_file():
            raise RuntimeError(f"thiếu cached wheel {drive_wheel}")
        FLASH_ATTN_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        local_wheel = FLASH_ATTN_LOCAL_DIR / wheel_name
        partial_path = FLASH_ATTN_LOCAL_DIR / f".{wheel_name}.{os.getpid()}.partial"
        print(f"[FA2 DRIVE CACHE COPY] {drive_wheel} -> {local_wheel}", flush=True)
        shutil.copyfile(drive_wheel, partial_path)
        actual_bytes = partial_path.stat().st_size
        actual_sha256 = _file_sha256(partial_path)
        if actual_bytes != CONFIG["flash_attn_wheel_bytes"]:
            raise RuntimeError(f"wheel size mismatch: {actual_bytes}")
        if actual_sha256 != CONFIG["flash_attn_wheel_sha256"]:
            raise RuntimeError("wheel SHA-256 mismatch")
        os.replace(partial_path, local_wheel)
        partial_path = None
        print({"fa2_drive_cache": "HIT_VERIFIED", "wheel": str(local_wheel),
               "bytes": actual_bytes}, flush=True)
        return local_wheel
    except Exception as exc:
        print(
            f"[FA2 DRIVE CACHE INVALID] {type(exc).__name__}: {exc}. "
            "Bỏ cache và tải lại từ GitHub.",
            flush=True,
        )
        return None
    finally:
        if partial_path is not None:
            Path(partial_path).unlink(missing_ok=True)


def publish_flash_attn_wheel_to_drive(wheel_path, cache_dir):
    """Publish atomic; manifest ghi cuối cùng và đóng vai trò commit marker."""
    if cache_dir is None:
        return None
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = Path(wheel_path)
    destination = cache_dir / wheel_path.name
    manifest_path = cache_dir / "manifest.json"
    try:
        if manifest_path.is_file() and destination.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (existing.get("contract") == FLASH_ATTN_WHEEL_CONTRACT
                    and destination.stat().st_size == CONFIG["flash_attn_wheel_bytes"]):
                print(f"[FA2 DRIVE CACHE] Artifact đã commit: {destination}", flush=True)
                return manifest_path
    except Exception as exc:
        print(f"[FA2 DRIVE CACHE] Manifest cũ không dùng được: {exc}; ghi lại.", flush=True)

    unique = f"{os.getpid()}-{time.time_ns()}"
    wheel_partial = cache_dir / f".{wheel_path.name}.{unique}.partial"
    manifest_partial = cache_dir / f".manifest.{unique}.partial"
    try:
        print(f"[FA2 DRIVE CACHE SAVE] {wheel_path} -> {destination}", flush=True)
        shutil.copyfile(wheel_path, wheel_partial)
        if (wheel_partial.stat().st_size != CONFIG["flash_attn_wheel_bytes"]
                or _file_sha256(wheel_partial) != CONFIG["flash_attn_wheel_sha256"]):
            raise RuntimeError("Drive copy verification thất bại")
        os.replace(wheel_partial, destination)
        manifest_partial.write_text(json.dumps({
            "schema_version": 1,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_name": CONFIG["session_name"],
            "contract": FLASH_ATTN_WHEEL_CONTRACT,
            "contract_sha256": _canonical_sha256(FLASH_ATTN_WHEEL_CONTRACT),
            "wheel_filename": wheel_path.name,
            "wheel_bytes": CONFIG["flash_attn_wheel_bytes"],
            "wheel_sha256": CONFIG["flash_attn_wheel_sha256"],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(manifest_partial, manifest_path)
        print({"fa2_drive_cache": "SAVED_VERIFIED", "cache_dir": str(cache_dir)}, flush=True)
        return manifest_path
    except Exception as exc:
        print(f"[FA2 DRIVE CACHE WARNING] Không save được wheel: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None
    finally:
        wheel_partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)


def ensure_flash_attn_wheel(stamp):
    """local cache -> Drive cache -> GitHub. Mọi đường đều phải qua SHA-256 pin."""
    local_wheel = FLASH_ATTN_LOCAL_DIR / FLASH_ATTN_WHEEL_NAME
    if local_wheel.is_file():
        if (local_wheel.stat().st_size == CONFIG["flash_attn_wheel_bytes"]
                and _file_sha256(local_wheel) == CONFIG["flash_attn_wheel_sha256"]):
            print(f"[FA2 LOCAL CACHE HIT] {local_wheel}", flush=True)
            return local_wheel, "local_cache"
        print("[FA2 LOCAL CACHE] wheel hỏng; xóa và lấy lại.", flush=True)
        local_wheel.unlink()

    drive_cache_dir = mount_flash_attn_drive_cache()
    restored = restore_flash_attn_wheel_from_drive(drive_cache_dir)
    if restored is not None:
        return restored, "drive_cache"

    partial = FLASH_ATTN_LOCAL_DIR / f".{FLASH_ATTN_WHEEL_NAME}.download"
    partial.unlink(missing_ok=True)
    run_observable(
        ["curl", "-fL", "--retry", "3", "--retry-delay", "5", "--progress-bar",
         "-o", str(partial), CONFIG["flash_attn_wheel_url"]],
        phase="FA2 DOWNLOAD",
        heartbeat_seconds=CONFIG["subprocess_heartbeat_seconds"],
        silence_warning_seconds=CONFIG["subprocess_silence_warning_seconds"],
        log_path=LOG_ROOT / f"fa2_download_{stamp}.log",
    )
    actual_bytes = partial.stat().st_size
    actual_sha256 = _file_sha256(partial)
    if actual_bytes != CONFIG["flash_attn_wheel_bytes"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"FA2 wheel size mismatch: {actual_bytes} != {CONFIG['flash_attn_wheel_bytes']}"
        )
    if actual_sha256 != CONFIG["flash_attn_wheel_sha256"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"FA2 wheel SHA-256 mismatch: {actual_sha256}")
    os.replace(partial, local_wheel)
    publish_flash_attn_wheel_to_drive(local_wheel, drive_cache_dir)
    return local_wheel, "github_download"


def _venv_marker_ok():
    if not VENV_MARKER.is_file() or not VENV_PYTHON.is_file():
        return False
    try:
        marker = json.loads(VENV_MARKER.read_text(encoding="utf-8"))
    except Exception:
        return False
    return marker.get("contract") == VENV_CONTRACT


def find_uv():
    for candidate in (shutil.which("uv"), str(Path.home() / ".local/bin/uv"), "/usr/local/bin/uv"):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


UV_BIN = find_uv()
if UV_BIN is None:
    run_observable(
        ["bash", "-lc", f"curl -LsSf {CONFIG['uv_install_url']} | sh"],
        phase="UV INSTALL",
        heartbeat_seconds=CONFIG["subprocess_heartbeat_seconds"],
        log_path=LOG_ROOT / "uv_install.log",
    )
    UV_BIN = find_uv()
if UV_BIN is None:
    raise RuntimeError("Không cài được uv; không dựng được venv Python 3.10 cho TARA.")
print("uv:", UV_BIN)

if _venv_marker_ok():
    print("Venv TARA đã sẵn sàng và khớp contract; bỏ qua cài đặt.")
else:
    venv_env = os.environ.copy()
    venv_env.update({
        "PYTHONUNBUFFERED": "1",
        "UV_HTTP_TIMEOUT": "600",
        "UV_NO_PROGRESS": "0",
    })
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_observable(
        [UV_BIN, "python", "install", CONFIG["venv_python_version"]],
        phase="UV PYTHON", env=venv_env,
        heartbeat_seconds=CONFIG["subprocess_heartbeat_seconds"],
        log_path=LOG_ROOT / f"uv_python_{stamp}.log",
    )
    if VENV_ROOT.exists():
        shutil.rmtree(VENV_ROOT)
    run_observable(
        [UV_BIN, "venv", "--python", CONFIG["venv_python_version"], str(VENV_ROOT)],
        phase="UV VENV", env=venv_env,
        heartbeat_seconds=CONFIG["subprocess_heartbeat_seconds"],
        log_path=LOG_ROOT / f"uv_venv_{stamp}.log",
    )
    pip_prefix = [UV_BIN, "pip", "install", "--python", str(VENV_PYTHON)]
    run_observable(
        pip_prefix + [
            "--index-url", CONFIG["torch_index_url"],
            f"torch=={CONFIG['torch_version']}",
            f"torchvision=={CONFIG['torchvision_version']}",
        ],
        phase="VENV TORCH", env=venv_env,
        heartbeat_seconds=CONFIG["subprocess_heartbeat_seconds"],
        silence_warning_seconds=CONFIG["subprocess_silence_warning_seconds"],
        log_path=LOG_ROOT / f"venv_torch_{stamp}.log",
    )
    run_observable(
        pip_prefix + TARA_VENV_PACKAGES,
        phase="VENV DEPS", env=venv_env,
        heartbeat_seconds=CONFIG["subprocess_heartbeat_seconds"],
        silence_warning_seconds=CONFIG["subprocess_silence_warning_seconds"],
        log_path=LOG_ROOT / f"venv_deps_{stamp}.log",
    )
    flash_attn_wheel, flash_attn_origin = ensure_flash_attn_wheel(stamp)
    print("FA2 wheel origin:", flash_attn_origin, "->", flash_attn_wheel, flush=True)
    run_observable(
        pip_prefix + ["--no-deps", str(flash_attn_wheel)],
        phase="VENV FLASH-ATTN", env=venv_env,
        heartbeat_seconds=CONFIG["subprocess_heartbeat_seconds"],
        silence_warning_seconds=CONFIG["subprocess_silence_warning_seconds"],
        log_path=LOG_ROOT / f"venv_flash_attn_{stamp}.log",
    )

VENV_PROBE = r"""
import json, sys, importlib.metadata as md
import torch, torchvision, transformers, numpy, decord, flash_attn
import cv2, sentencepiece, PIL
from PIL import Image, ImageDraw  # noqa: F401  - bắt Pillow bị trộn file giữa 2 version
from flash_attn import flash_attn_func  # noqa: F401
print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "decord": md.version("decord"),
    "flash_attn": md.version("flash-attn"),
    "pillow": md.version("Pillow"),
    "sentencepiece": md.version("sentencepiece"),
    "opencv": md.version("opencv-python-headless"),
    "cv2_runtime": cv2.__version__,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}))
"""
probe = subprocess.run(
    [str(VENV_PYTHON), "-c", VENV_PROBE],
    check=True, capture_output=True, text=True,
)
VENV_INFO = json.loads(probe.stdout.strip().splitlines()[-1])
print("Venv TARA:", json.dumps(VENV_INFO, indent=2))

# Mọi package trong requirements.txt chính chủ đều được verify, không chỉ một
# vài cái. Đặc biệt Pillow: audit embedding của notebook so production path với
# official path trong CÙNG environment, nên nó không thể phát hiện việc cả hai
# cùng chạy sai version Pillow — chỉ preflight này bắt được.
expected_probe = {
    "torch": CONFIG["torch_version"],
    "torchvision": CONFIG["torchvision_version"],
    "transformers": CONFIG["transformers_version"],
    "numpy": CONFIG["numpy_version"],
    "decord": CONFIG["decord_version"],
    "flash_attn": CONFIG["flash_attn_version"],
    "pillow": CONFIG["pillow_version"],
    "sentencepiece": CONFIG["sentencepiece_version"],
    "opencv": CONFIG["opencv_version"],
}
if not VENV_INFO["python"].startswith(CONFIG["venv_python_version"]):
    raise RuntimeError(f"Venv Python mismatch: {VENV_INFO['python']}")
for key in ("torch", "torchvision"):
    if not VENV_INFO[key].startswith(expected_probe[key]):
        raise RuntimeError(f"Venv {key} mismatch: {VENV_INFO[key]} != {expected_probe[key]}")
for key in ("transformers", "numpy", "decord", "flash_attn", "pillow",
            "sentencepiece", "opencv"):
    if VENV_INFO[key] != expected_probe[key]:
        raise RuntimeError(
            f"Venv {key} mismatch: {VENV_INFO[key]} != {expected_probe[key]}. "
            "Xóa /content/tara-venv rồi chạy lại cell này."
        )
if not VENV_INFO["cuda_available"]:
    raise RuntimeError("Venv TARA không thấy CUDA GPU.")

VENV_MARKER.write_text(
    json.dumps({"contract": VENV_CONTRACT, "probe": VENV_INFO}, indent=2), encoding="utf-8"
)
print("Venv TARA verified.")

# %% [markdown]
# ## 2. Xác thực HF/R2 và preflight phần cứng
#
# Token lấy theo thứ tự: Colab Secret `HF_TOKEN` → environment → file cấu hình →
# prompt ẩn. R2 credential (`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`,
# chỉ cần quyền đọc bucket `aic26-media`) đi theo cùng thứ tự. Token/credential không
# được in và không được ghi vào notebook/manifest.

# %%
import getpass
import importlib.metadata
import re
import socket
from datetime import datetime, timezone
from typing import Dict, List, Sequence

import psutil
from huggingface_hub import bucket_info, whoami


def get_hf_token():
    token = None
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception:
        pass
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        for candidate in CONFIG["hf_token_file_candidates"]:
            path = Path(candidate)
            if path.is_file():
                token = path.read_text(encoding="utf-8").strip()
                break
    if not token:
        token = getpass.getpass("HF_TOKEN (input hidden): ").strip()
    if not token:
        raise RuntimeError("Không tìm thấy HF_TOKEN.")
    return token


R2_CREDENTIAL_NAMES = {
    "account_id": "R2_ACCOUNT_ID",
    "access_key_id": "R2_ACCESS_KEY_ID",
    "secret_access_key": "R2_SECRET_ACCESS_KEY",
}


def read_r2_credential_file(path: Path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip().lower()] = value.strip()
    return {
        "account_id": values.get("account id"),
        "access_key_id": values.get("access key id"),
        "secret_access_key": values.get("secret access key"),
    }


def get_r2_credentials():
    credentials = {key: None for key in R2_CREDENTIAL_NAMES}
    try:
        from google.colab import userdata
        for key, secret_name in R2_CREDENTIAL_NAMES.items():
            try:
                credentials[key] = userdata.get(secret_name)
            except Exception:
                pass
    except Exception:
        pass
    for key, env_name in R2_CREDENTIAL_NAMES.items():
        credentials[key] = credentials[key] or os.environ.get(env_name)
    if not all(credentials.values()):
        for candidate in CONFIG["r2_credential_file_candidates"]:
            path = Path(candidate)
            if path.is_file():
                from_file = read_r2_credential_file(path)
                for key in credentials:
                    credentials[key] = credentials[key] or from_file.get(key)
                break
    for key, secret_name in R2_CREDENTIAL_NAMES.items():
        if not credentials[key]:
            credentials[key] = getpass.getpass(f"{secret_name} (input hidden): ").strip()
    if not all(credentials.values()):
        raise RuntimeError("Thiếu R2 credential (cần quyền đọc bucket aic26-media).")
    return credentials


HF_TOKEN = get_hf_token()
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
identity = whoami(token=HF_TOKEN)
print("Hugging Face account:", identity.get("name", "authenticated"))
R2_CREDENTIALS = get_r2_credentials()
print("Source R2 bucket:", CONFIG["source_r2_bucket"])
print("Output bucket:", bucket_info(CONFIG["output_bucket_id"], token=HF_TOKEN).id)

GPU_NAME = VENV_INFO["gpu"]
if GPU_NAME is None:
    raise RuntimeError("Không thấy GPU.")
nvidia = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
     "--format=csv,noheader,nounits"],
    check=True, capture_output=True, text=True,
).stdout.strip().splitlines()[0]
gpu_fields = [field.strip() for field in nvidia.split(",")]
GPU_VRAM_GIB = float(gpu_fields[1]) / 1024
SYSTEM_RAM_GIB = psutil.virtual_memory().total / 2 ** 30

# Chọn profile theo tên GPU + VRAM: A100 40 GB và 80 GB cùng tên "A100" nhưng batch vừa VRAM khác nhau.
GPU_PROFILE_NAME = next(
    (tag for tag, profile in CONFIG["gpu_profiles"].items()
     if profile["match"] in GPU_NAME.upper() and GPU_VRAM_GIB >= float(profile["min_vram_gib"])),
    None,
)
if GPU_PROFILE_NAME is None:
    raise RuntimeError(
        f"GPU {GPU_NAME} ({GPU_VRAM_GIB:.0f} GiB) không khớp gpu_profiles {list(CONFIG['gpu_profiles'])}."
    )
GPU_PROFILE = CONFIG["gpu_profiles"][GPU_PROFILE_NAME]
# manual = người dùng ép batch (phải vừa VRAM, không tự hạ); profile = mặc định của GPU (OOM thì
# autotune lại batch nhỏ hơn); autotune = benchmark mọi batch_candidates.
if CONFIG["batch_size"] is not None:
    BATCH_SIZE_SOURCE = "manual"
else:
    CONFIG["batch_size"] = GPU_PROFILE["batch_size"]
    BATCH_SIZE_SOURCE = "profile" if CONFIG["batch_size"] is not None else "autotune"
if CONFIG["gpu_workers"] is None:
    CONFIG["gpu_workers"] = GPU_PROFILE["gpu_workers"]
print(f"GPU profile {GPU_PROFILE_NAME} ({GPU_VRAM_GIB:.0f} GiB): batch_size={CONFIG['batch_size']} "
      f"({BATCH_SIZE_SOURCE}), gpu_workers={CONFIG['gpu_workers']}")
disk = shutil.disk_usage(SCRATCH_ROOT)
if disk.free / 2 ** 30 < CONFIG["min_free_disk_gib"]:
    raise RuntimeError(
        f"Cần tối thiểu {CONFIG['min_free_disk_gib']} GiB free; hiện có "
        f"{disk.free / 2 ** 30:.1f} GiB."
    )

print(json.dumps({
    "gpu": gpu_fields[0],
    "driver": gpu_fields[2],
    "vram_gib": round(GPU_VRAM_GIB, 2),
    "system_ram_gib": round(SYSTEM_RAM_GIB, 2),
    "cpu_count": os.cpu_count(),
    "scratch_free_gib": round(disk.free / 2 ** 30, 2),
    "kernel_python": platform.python_version(),
    "kernel_hf_hub": importlib.metadata.version("huggingface_hub"),
    "kernel_pyarrow": importlib.metadata.version("pyarrow"),
}, indent=2))

# %% [markdown]
# ## 3. I/O Bucket, đọc video R2, checksum và commit protocol

# %%
import math

from huggingface_hub import (
    batch_bucket_files,
    download_bucket_files,
    get_bucket_paths_info,
    list_bucket_tree,
)
from tqdm.auto import tqdm


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bucket_uri(bucket_id: str, path: str):
    return f"hf://buckets/{bucket_id}/{path.lstrip('/')}"


def remote_info(bucket_id: str, path: str):
    # Exact-path lookup phải dùng metadata API; list_bucket_tree(prefix=...) nhận
    # directory prefix nên có thể báo thiếu dù file path thực sự tồn tại.
    for item in get_bucket_paths_info(bucket_id, [path], token=HF_TOKEN):
        if item.path == path and getattr(item, "type", "file") == "file":
            return item
    return None


def read_remote_bytes(bucket_id: str, path: str):
    result = subprocess.run(
        ["hf", "buckets", "cp", bucket_uri(bucket_id, path), "-"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy(),
    )
    return result.stdout


def read_remote_json_optional(bucket_id: str, path: str):
    if remote_info(bucket_id, path) is None:
        return None
    return json.loads(read_remote_bytes(bucket_id, path))


def upload_bytes_verified(bucket_id: str, remote_path: str, data: bytes):
    batch_bucket_files(bucket_id, add=[(data, remote_path)], token=HF_TOKEN)
    info = remote_info(bucket_id, remote_path)
    if info is None or int(info.size) != len(data):
        raise RuntimeError(f"Remote verify thất bại: {remote_path}")


def upload_file_verified(bucket_id: str, remote_path: str, local_path: Path):
    expected_size = local_path.stat().st_size
    batch_bucket_files(bucket_id, add=[(str(local_path), remote_path)], token=HF_TOKEN)
    info = remote_info(bucket_id, remote_path)
    if info is None or int(info.size) != expected_size:
        raise RuntimeError(f"Remote size mismatch: {remote_path}")


def natural_video_key(video_id: str):
    # M dùng `M01_V001`; N và S01 dùng `N001-V001`, `S01-V001`.
    match = re.search(r"[_-]V(\d+)$", video_id)
    return (int(match.group(1)) if match else 10 ** 12, video_id)


# ---------------------------------------------------------------- R2 (chỉ đọc)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "boto3>=1.36,<2"], check=True
)
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig

R2_PART_SIZE = int(CONFIG["r2_part_size_mib"]) * 2 ** 20
S3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_CREDENTIALS['account_id']}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_CREDENTIALS["access_key_id"],
    aws_secret_access_key=R2_CREDENTIALS["secret_access_key"],
    region_name="auto",
    config=BotoConfig(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        max_pool_connections=int(CONFIG["r2_download_concurrency"]) * 3 + 8,
        # adaptive có token bucket phía client, dễ tự bóp băng thông cả pool.
        retries={"max_attempts": 8, "mode": "standard"},
        connect_timeout=15,
        read_timeout=60,
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    ),
)
R2_TRANSFER = TransferConfig(
    multipart_threshold=R2_PART_SIZE,
    multipart_chunksize=R2_PART_SIZE,
    max_concurrency=int(CONFIG["r2_download_concurrency"]),
    use_threads=True,
)


def multipart_etag(path: Path, part_size: int = R2_PART_SIZE):
    """ETag R2/S3 của file upload theo part cố định: md5(md5(part1)…md5(partN))-N."""
    digests = []
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(part_size), b""):
            digests.append(hashlib.md5(chunk).digest())
    if len(digests) <= 1:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    return f"{hashlib.md5(b''.join(digests)).hexdigest()}-{len(digests)}"


def r2_list(prefix: str):
    objects = {}
    for page in S3.get_paginator("list_objects_v2").paginate(
        Bucket=CONFIG["source_r2_bucket"], Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            objects[obj["Key"]] = {"size": int(obj["Size"]), "etag": str(obj["ETag"]).strip('"')}
    return objects


def r2_head(key: str):
    return S3.head_object(Bucket=CONFIG["source_r2_bucket"], Key=key)


FFPROBE = shutil.which("ffprobe")
if FFPROBE is None:
    subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=True)
    FFPROBE = shutil.which("ffprobe")
if FFPROBE is None:
    raise RuntimeError("Không có ffprobe để kiểm codec video tải về.")


def probe_video_codec(path: Path):
    return subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def is_batch2_category(category: str):
    if category == "S01" or re.fullmatch(r"M(?:0[1-9]|10)", category):
        return True
    return bool(re.fullmatch(r"N\d{3}", category)) and 1 <= int(category[1:]) <= 100


def validate_config():
    assignments = CONFIG["session_assignments"]
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("session_assignments phải là dict khác rỗng.")
    owners = {}
    for session_id, assigned in assignments.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id):
            raise ValueError(f"Tên session không hợp lệ: {session_id}")
        if not assigned or len(set(assigned)) != len(assigned):
            raise ValueError(f"Assignment rỗng/trùng category: {session_id}")
        for category in assigned:
            if category in owners:
                raise ValueError(
                    f"Category {category} bị gán cho cả {owners[category]} và {session_id}. "
                    "Không được chạy overlap trên cùng output_prefix."
                )
            owners[category] = session_id
    if set(owners) != set(ALL_CATEGORIES):
        raise ValueError(
            "session_assignments phải phủ đúng M01..M10 + N001..N100 + S01; "
            f"missing={sorted(set(ALL_CATEGORIES) - set(owners))}, "
            f"extra={sorted(set(owners) - set(ALL_CATEGORIES))}"
        )
    if CONFIG["categories"] != list(assignments[CONFIG["active_session"]]):
        raise ValueError("categories phải được suy ra từ active_session.")
    for category in CONFIG["categories"]:
        if not is_batch2_category(category):
            raise ValueError(f"Category không hợp lệ: {category}")

    if CONFIG["embedding_dim"] != 3584:
        raise ValueError("TARA (Tarsier2-7B) khóa embedding_dim = 3584.")
    if CONFIG["model_weight_dtype"] != "bfloat16":
        raise ValueError("Checkpoint TARA là BF16; không đổi weight dtype.")
    if CONFIG["storage_dtype"] != "float32":
        raise ValueError("Master embedding contract khóa storage_dtype='float32'.")
    if CONFIG["attention_implementation"] != "flash_attention_2":
        raise ValueError("Pipeline này khóa FlashAttention-2 trên A100.")
    if CONFIG["n_frames"] != CONFIG["expected_default_config"]["n_frames"]:
        raise ValueError("n_frames phải khớp default_config chính chủ.")
    if CONFIG["max_pixels"] != CONFIG["expected_default_config"]["max_pixels"]:
        raise ValueError("max_pixels phải khớp default_config chính chủ.")
    if CONFIG["video_eol_prompt"] != "<video>\nSummary above video in one word:":
        raise ValueError("Video EOL prompt phải khớp EOL_PROMPTS chính chủ.")

    if set(CONFIG["scales_by_group"]) != {"M", "N", "S"}:
        raise ValueError("scales_by_group phải có đúng ba nhóm M, N, S.")
    for group, scales in CONFIG["scales_by_group"].items():
        if not scales:
            raise ValueError(f"Nhóm {group} cần ít nhất một scale.")
        names = [s[0] for s in scales]
        if len(set(names)) != len(names):
            raise ValueError(f"Tên scale bị trùng trong nhóm {group}.")
        for name, window, hop in scales:
            if not re.fullmatch(r"[a-z0-9_]+", name):
                raise ValueError(f"Tên scale không hợp lệ: {name}")
            if not (0 < float(hop) <= float(window)):
                raise ValueError(f"Scale {group}/{name}: cần 0 < hop <= window.")
    if int(CONFIG["prep_threads"]) != 1:
        raise ValueError(
            "prep_threads phải bằng 1.\n"
            "  - transformers 4.45 cache compiled Jinja template bằng @lru_cache nên "
            "AssistantTracker là singleton; gọi apply_chat_template song song sẽ ném "
            "'AssistantTracker should not be reused before closed'.\n"
            "  - Muốn thêm luồng preprocess thì tăng gpu_workers: mỗi worker là một "
            "process riêng với Jinja environment riêng."
        )
    if int(CONFIG["prefetch_batches"]) < 1:
        raise ValueError("prefetch_batches phải >= 1 để CPU prep overlap được với GPU.")
    if int(CONFIG["gpu_workers"]) < 1:
        raise ValueError("gpu_workers phải >= 1.")
    for tag, profile in CONFIG["gpu_profiles"].items():
        if (
            not str(profile.get("match", "")).strip()
            or float(profile.get("min_vram_gib", -1)) < 0
            or int(profile["gpu_workers"]) < 1
            or (profile["batch_size"] is not None and int(profile["batch_size"]) < 1)
        ):
            raise ValueError(f"gpu_profiles[{tag}] không hợp lệ: {profile}")
    if int(CONFIG["encode_chunk_rows"]) < 1 or int(CONFIG["video_prefetch_extra"]) < 0:
        raise ValueError("encode_chunk_rows phải >= 1 và video_prefetch_extra >= 0.")
    if CONFIG["output_prefix"].strip("/").startswith(("Videos", "Keyframes", "manifest")):
        raise ValueError("output_prefix không được ghi đè lên media/manifest nguồn.")
    if CONFIG["output_prefix"].strip("/") == CONFIG["reference_output_prefix"].strip("/"):
        raise ValueError("output_prefix không được trùng collection TARA của L.")
    if CONFIG["output_prefix"].strip("/") in {p.strip("/") for p in CONFIG["abandoned_output_prefixes"]}:
        raise ValueError("output_prefix trùng prefix đã bỏ (lần chạy thử 4/8/16 cho mọi nhóm).")
    if int(CONFIG["shard_rows"]) <= 0:
        raise ValueError("shard_rows phải > 0.")
    if not (0 < float(CONFIG["max_vram_fraction"]) < 1):
        raise ValueError("max_vram_fraction phải nằm trong (0,1).")
    if not set(CONFIG["delegated_categories"]) <= set(ALL_CATEGORIES):
        raise ValueError("delegated_categories chứa category lạ.")


validate_config()
print("Config hợp lệ.")

# %% [markdown]
# ## 4. Manifest 614 video batch 2 và đối chiếu với R2
#
# Manifest được pin ngày 2026-09-23 từ list trực tiếp trên R2 + `ffprobe` luồng video của
# từng file: `size`/`etag` là identity của bytes, `duration_sec` là duration **luồng video**
# (quyết định clip plan), `fps` = `avg_frame_rate`. Cell dưới là manifest nén gzip + base64;
# SHA-256 của CSV đã giải nén nằm trong semantic fingerprint.
#
# Lúc chạy, notebook list lại `Videos/` trên R2: video M/N phải khớp tuyệt đối size + ETag.
# Với 11 video S01 AV1, object trên R2 phải là bản H.264 do notebook S01 tạo (metadata
# `pipeline`, `transform`, `source-etag` = ETag AV1 đã pin, `video-frames` = số frame gốc).

# %%
BATCH2_VIDEO_MANIFEST_SHA256 = "ea094415db6184987ddd2070a5980663f5716b527dd42e42eb1f316a27733e45"
BATCH2_VIDEO_MANIFEST_B64 = (
    "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"
)

# %%
import base64
import csv
import gzip
import io
from dataclasses import dataclass

EXPECTED_GROUP_VIDEOS = {"M": 304, "N": 298, "S": 12}


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    category: str
    remote_path: str
    fps: float
    duration_sec: float
    size_bytes: int
    etag: str
    codec: str
    nb_frames: int


MANIFEST_TEXT = gzip.decompress(base64.b64decode(BATCH2_VIDEO_MANIFEST_B64)).decode("utf-8")
if sha256_bytes(MANIFEST_TEXT.encode("utf-8")) != BATCH2_VIDEO_MANIFEST_SHA256:
    raise RuntimeError("Manifest video batch 2 nhúng trong notebook bị hỏng (SHA-256 lệch).")
(STATE_ROOT / "batch2_video_manifest.csv").write_text(MANIFEST_TEXT, encoding="utf-8")


def load_video_registry():
    by_category: Dict[str, List[VideoRecord]] = {}
    for row in csv.DictReader(io.StringIO(MANIFEST_TEXT)):
        record = VideoRecord(
            video_id=row["video_id"],
            category=row["category"],
            remote_path=row["r2_key"],
            fps=float(row["fps"]),
            duration_sec=float(row["duration_sec"]),
            size_bytes=int(row["size_bytes"]),
            etag=row["etag"],
            codec=row["codec"],
            nb_frames=int(row["nb_frames"]),
        )
        separator = "_" if record.category.startswith("M") else "-"
        if not re.fullmatch(rf"{record.category}{separator}V\d{{3}}", record.video_id):
            raise RuntimeError(f"video_id lạ trong manifest: {record.video_id}")
        if record.remote_path != f"Videos/Videos_{record.category}/{record.video_id}.mp4":
            raise RuntimeError(f"r2_key sai layout: {record.remote_path}")
        if record.fps <= 0 or record.duration_sec <= 0 or record.size_bytes <= 0 or record.nb_frames <= 0:
            raise RuntimeError(f"Manifest row hỏng: {record.video_id}")
        if record.codec not in {"h264", "av1"} or (record.codec == "av1" and record.category != "S01"):
            raise RuntimeError(f"Codec ngoài dự kiến: {record.video_id} {record.codec}")
        by_category.setdefault(record.category, []).append(record)
    for records in by_category.values():
        records.sort(key=lambda r: natural_video_key(r.video_id))
        if len({r.video_id for r in records}) != len(records):
            raise RuntimeError("video_id trùng trong manifest.")
    if set(by_category) != set(ALL_CATEGORIES):
        raise RuntimeError("Manifest không phủ đúng M01..M10 + N001..N100 + S01.")
    group_counts = {g: sum(len(v) for c, v in by_category.items() if c[0] == g) for g in "MNS"}
    if group_counts != EXPECTED_GROUP_VIDEOS:
        raise RuntimeError(f"Số video theo nhóm lệch: {group_counts}")
    return {category: by_category[category] for category in ALL_CATEGORIES}


VIDEO_REGISTRY = load_video_registry()
EXPECTED_VIDEO_COUNTS = {category: len(records) for category, records in VIDEO_REGISTRY.items()}


def resolve_video_objects():
    """Đối chiếu từng video với R2; trả về object sẽ tải (key/size/etag) cho mỗi video_id."""
    live = r2_list("Videos/Videos_")
    resolved, problems = {}, []
    for records in VIDEO_REGISTRY.values():
        for record in records:
            obj = live.get(record.remote_path)
            if obj is None:
                problems.append(f"{record.video_id}: thiếu trên R2")
            elif obj["etag"] == record.etag and obj["size"] == record.size_bytes:
                kind = "av1_pending" if record.codec == "av1" else "pinned"
                resolved[record.video_id] = {"kind": kind, **obj}
            elif record.codec == "av1":
                metadata = r2_head(record.remote_path).get("Metadata", {})
                if (
                    metadata.get("pipeline") == CONFIG["s01_pipeline"]
                    and metadata.get("transform") == CONFIG["s01_transform"]
                    and metadata.get("source-etag") == record.etag
                    and int(metadata.get("source-size", -1)) == record.size_bytes
                    and int(metadata.get("video-frames", -1)) == record.nb_frames
                    and metadata.get("expected-etag") == obj["etag"]
                ):
                    resolved[record.video_id] = {"kind": "s01_h264", **obj}
                else:
                    problems.append(f"{record.video_id}: object S01 không phải AV1 gốc cũng không "
                                    "phải bản H.264 của notebook S01")
            else:
                problems.append(f"{record.video_id}: size/ETag trên R2 khác manifest đã pin")
    if problems:
        raise RuntimeError("Video nguồn trên R2 khác manifest đã pin:\n  " + "\n  ".join(problems[:20]))
    return resolved


VIDEO_OBJECTS = resolve_video_objects()
kind_counts = {}
for obj in VIDEO_OBJECTS.values():
    kind_counts[obj["kind"]] = kind_counts.get(obj["kind"], 0) + 1
pending_s01 = sorted(v for v, obj in VIDEO_OBJECTS.items() if obj["kind"] == "av1_pending")
if pending_s01 and "S01" in CONFIG["categories"] and "S01" not in CONFIG["delegated_categories"]:
    if CONFIG["categories"][0] == "S01":
        raise RuntimeError(
            f"{len(pending_s01)} video S01 trên R2 vẫn là AV1 ({pending_s01[:3]}...). Chạy "
            "AIC2026_S01_AV1_to_H264_R2_Colab.ipynb trước."
        )
    print(f"CẢNH BÁO: {len(pending_s01)} video S01 vẫn là AV1. Các category trước S01 vẫn chạy; "
          "notebook sẽ kiểm lại R2 khi tới S01, hãy chạy notebook S01 trước thời điểm đó.")
print(json.dumps({
    "videos_total": sum(EXPECTED_VIDEO_COUNTS.values()),
    "per_group": EXPECTED_GROUP_VIDEOS,
    "r2_objects": kind_counts,
    "hours_total": round(sum(r.duration_sec for v in VIDEO_REGISTRY.values() for r in v) / 3600, 2),
    "session_video_gib": round(sum(
        VIDEO_OBJECTS[r.video_id]["size"] for c in CONFIG["categories"] for r in VIDEO_REGISTRY[c]
    ) / 2 ** 30, 2),
}, indent=2))

# %% [markdown]
# ## 5. Multi-scale temporal clip plan
#
# Plan phải **deterministic**: row index của mỗi clip là identity của nó trong
# shard/commit, nên nó chỉ được suy ra từ `duration_sec` trong manifest (không
# phải từ decoder), và không đổi giữa các session.
#
# Với mỗi scale `(window W, hop H)` và video dài `D`:
#
# ```text
# D <= W          →  1 clip [0, D]
# D >  W          →  start_i = min(i*H, D-W) với i = 0..ceil((D-W)/H)
#                    end_i   = start_i + W        (clip cuối luôn kết thúc đúng tại D)
# ```
#
# Thứ tự row trong một category: `video (natural order) → scale (event, sequence,
# scene) → start_time tăng dần`. Mọi clip của cùng một video nằm liền nhau.

# %%
@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    video_id: str
    category: str
    scale: str
    scale_index: int
    start_time: float
    end_time: float
    video_index: int


def plan_video_clips(record: VideoRecord, video_index: int):
    decimals = int(CONFIG["clip_time_decimals"])
    duration = float(record.duration_sec)
    clips: List[ClipRecord] = []
    for scale_index, (name, window, hop) in enumerate(scales_for(record.category)):
        window = float(window)
        hop = float(hop)
        if duration <= window:
            spans = [(0.0, round(duration, decimals))]
        else:
            steps = int(math.ceil((duration - window) / hop)) + 1
            starts = sorted({round(min(i * hop, duration - window), decimals)
                             for i in range(steps)})
            spans = [(start, round(min(start + window, duration), decimals)) for start in starts]
        for start, end in spans:
            if end <= start:
                raise RuntimeError(f"Clip rỗng cho {record.video_id} @ {name} {start}-{end}")
            clips.append(ClipRecord(
                clip_id=f"{record.video_id}@{name}@t{int(round(start * 1000)):09d}",
                video_id=record.video_id,
                category=record.category,
                scale=name,
                scale_index=scale_index,
                start_time=start,
                end_time=end,
                video_index=video_index,
            ))
    return clips


def build_category_plan(category: str):
    clips: List[ClipRecord] = []
    for video_index, record in enumerate(VIDEO_REGISTRY[category]):
        clips.extend(plan_video_clips(record, video_index))
    if len({c.clip_id for c in clips}) != len(clips):
        raise RuntimeError(f"{category}: clip_id bị trùng trong plan.")
    return clips


CLIP_PLAN: Dict[str, List[ClipRecord]] = {
    category: build_category_plan(category) for category in ALL_CATEGORIES
}
PLAN_COUNTS = {category: len(clips) for category, clips in CLIP_PLAN.items()}
PLAN_TOTAL = sum(PLAN_COUNTS.values())
PLAN_TOTALS_BY_GROUP = {g: sum(n for c, n in PLAN_COUNTS.items() if c[0] == g) for g in "MNS"}

scales_match_reference = {
    group: [list(s) for s in scales] for group, scales in CONFIG["scales_by_group"].items()
} == REFERENCE_SCALES_BY_GROUP
if scales_match_reference and PLAN_TOTALS_BY_GROUP != REFERENCE_CLIP_TOTALS_BY_GROUP:
    raise RuntimeError(
        "Plan không tái tạo được số clip tham chiếu của scales mặc định: "
        f"{PLAN_TOTALS_BY_GROUP} != {REFERENCE_CLIP_TOTALS_BY_GROUP}."
    )
if not scales_match_reference:
    print(
        "CẢNH BÁO: CONFIG['scales_by_group'] khác REFERENCE_SCALES_BY_GROUP nên bỏ qua assert số clip. "
        "Đây là embedding collection MỚI — hãy đổi luôn output_prefix."
    )

per_scale = {}
for clips in CLIP_PLAN.values():
    for clip in clips:
        key = f"{clip.category[0]}/{clip.scale}"
        per_scale[key] = per_scale.get(key, 0) + 1
session_clips = sum(PLAN_COUNTS[c] for c in CONFIG["categories"])

print(json.dumps({
    "clips_total": PLAN_TOTAL,
    "per_group": PLAN_TOTALS_BY_GROUP,
    "per_scale": per_scale,
    "session": CONFIG["session_name"],
    "session_categories": len(CONFIG["categories"]),
    "session_clips": session_clips,
    "session_shards": sum(
        math.ceil(PLAN_COUNTS[c] / CONFIG["shard_rows"]) for c in CONFIG["categories"]
    ),
    "vector_bytes_total_gib": round(PLAN_TOTAL * CONFIG["embedding_dim"] * 4 / 2 ** 30, 2),
}, indent=2))

# %% [markdown]
# ## 6. Tải checkpoint TARA và khóa semantic fingerprint
#
# `bpiyush/TARA` là một repo model HF chứa **cả weights lẫn code**
# (`modeling_tara.py`, `tarsier2/`, `shared/`, `tarsier2/default_config.yaml`),
# nên `snapshot_download` là đủ — không cần `git lfs` và không cần clone GitHub.
# Revision được pin theo commit để mọi session nằm trong cùng một embedding space.
#
# Semantic (đổi là hỏng collection): model id/revision, checkpoint payload bytes,
# EOL prompt, `n_frames`, `max_pixels`, sampler v1, `use_multi_images_for_video`,
# pooling last-token, dtype, clip plan (scales). Execution policy (đổi thoải mái):
# batch size, thread preprocess, decord threads, cache size, session name.

# %%
from huggingface_hub import snapshot_download

MODEL_SNAPSHOT_DIR = Path(snapshot_download(
    repo_id=CONFIG["model_id"],
    revision=CONFIG["model_revision"],
    cache_dir=MODEL_ROOT,
    token=HF_TOKEN,
    # Giữ lại assets/*.mp4: demo video của tác giả là ground truth cho self test
    # ở mục 11 (README chính chủ có sẵn số similarity mong đợi).
    ignore_patterns=["*.png", "*.md"],
))
MODEL_INDEX_PATH = MODEL_SNAPSHOT_DIR / "model.safetensors.index.json"
if not MODEL_INDEX_PATH.is_file():
    raise FileNotFoundError(f"Thiếu model index: {MODEL_INDEX_PATH}")
MODEL_INDEX_SHA256 = sha256_file(MODEL_INDEX_PATH)
MODEL_INDEX = json.loads(MODEL_INDEX_PATH.read_text(encoding="utf-8"))
MODEL_TENSOR_PAYLOAD_BYTES = int(MODEL_INDEX.get("metadata", {}).get("total_size", -1))
if MODEL_TENSOR_PAYLOAD_BYTES != EXPECTED_MODEL_TENSOR_PAYLOAD_BYTES:
    raise RuntimeError(
        "Checkpoint tensor payload bytes lạ: "
        f"{MODEL_TENSOR_PAYLOAD_BYTES} != {EXPECTED_MODEL_TENSOR_PAYLOAD_BYTES}"
    )
weight_files = sorted(set(MODEL_INDEX["weight_map"].values()))
missing_weights = [name for name in weight_files if not (MODEL_SNAPSHOT_DIR / name).is_file()]
if missing_weights:
    raise FileNotFoundError(f"Thiếu checkpoint shards: {missing_weights}")

for required in ("modeling_tara.py", "tarsier2/default_config.yaml",
                 "tarsier2/modeling_tarsier2.py", "tarsier2/dataset/tarsier_datamodule.py",
                 "shared/utils/log.py", "config.json", "chat_template.json"):
    if not (MODEL_SNAPSHOT_DIR / required).is_file():
        raise FileNotFoundError(f"Snapshot thiếu {required}; checkpoint không đầy đủ code.")

MODEL_CONFIG = json.loads((MODEL_SNAPSHOT_DIR / "config.json").read_text(encoding="utf-8"))
if MODEL_CONFIG["architectures"] != ["Tarsier2ForConditionalGeneration"]:
    raise RuntimeError(f"Architecture lạ: {MODEL_CONFIG['architectures']}")
if int(MODEL_CONFIG["text_config"]["hidden_size"]) != CONFIG["embedding_dim"]:
    raise RuntimeError("text_config.hidden_size != 3584.")
if int(MODEL_CONFIG["text_config"]["num_hidden_layers"]) != 28:
    raise RuntimeError("Tarsier2-7B phải có 28 text layer.")
if int(MODEL_CONFIG["vision_config"]["patch_size"]) != 14:
    raise RuntimeError("Vision patch_size phải là 14.")

MODELING_TARA_SHA256 = sha256_file(MODEL_SNAPSHOT_DIR / "modeling_tara.py")
DEFAULT_CONFIG_SHA256 = sha256_file(MODEL_SNAPSHOT_DIR / "tarsier2" / "default_config.yaml")
print(json.dumps({
    "model_snapshot": str(MODEL_SNAPSHOT_DIR),
    "checkpoint_tensor_payload_gib": round(MODEL_TENSOR_PAYLOAD_BYTES / 2 ** 30, 3),
    "weight_shards": weight_files,
    "model_index_sha256": MODEL_INDEX_SHA256,
    "modeling_tara_sha256": MODELING_TARA_SHA256,
    "default_config_sha256": DEFAULT_CONFIG_SHA256,
}, indent=2))


SEMANTIC_CONFIG = {
    "schema_version": 1,
    "contract": "official_TARA_encode_vision_multiscale_clip",
    "source": {
        "storage": "cloudflare_r2",
        "r2_bucket": CONFIG["source_r2_bucket"],
        "video_manifest_sha256": BATCH2_VIDEO_MANIFEST_SHA256,
        "duration_source": "ffprobe video stream duration (manifest pin 2026-09-23)",
        "expected_video_counts": EXPECTED_VIDEO_COUNTS,
        "expected_total_videos": sum(EXPECTED_VIDEO_COUNTS.values()),
        "s01_av1_handling": {
            "pipeline": CONFIG["s01_pipeline"],
            "transform": CONFIG["s01_transform"],
            "rule": "S01 AV1 phải được thay bằng bản H.264 đúng từng frame trước khi encode",
        },
    },
    "model": {
        "id": CONFIG["model_id"],
        "revision": CONFIG["model_revision"],
        "base_model": CONFIG["base_model"],
        "architecture": "Tarsier2ForConditionalGeneration",
        "checkpoint_index_sha256": MODEL_INDEX_SHA256,
        "checkpoint_tensor_payload_bytes": MODEL_TENSOR_PAYLOAD_BYTES,
        "modeling_tara_sha256": MODELING_TARA_SHA256,
        "default_config_sha256": DEFAULT_CONFIG_SHA256,
        "text_layers": 28,
        "embedding_dim": CONFIG["embedding_dim"],
    },
    "preprocess": {
        "input": "video_clip",
        "eol_prompt": CONFIG["video_eol_prompt"],
        "sample_builder": "tarsier2.dataset.utils.format_one_sample + explicit frame_indices",
        "frame_selection": "sample_video(start_time,end_time) v1 uniform, endpoints included",
        "n_frames": CONFIG["n_frames"],
        "use_multi_images_for_video": True,
        "effective_image_slots": CONFIG["n_frames"] * 2,
        "max_pixels": CONFIG["max_pixels"],
        "min_pixels": CONFIG["min_pixels"],
        "do_image_resize": False,
        "do_image_padding": False,
        "do_image_crop": False,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "max_seq_len": 16_384,
        "padding_side": "left",
    },
    "clip_plan": {
        "scales_by_group": {
            group: [list(scale) for scale in scales]
            for group, scales in CONFIG["scales_by_group"].items()
        },
        "overlap": "hop = window / 2",
        "clip_time_decimals": CONFIG["clip_time_decimals"],
        "row_order": "video natural order -> scale order -> start_time",
        "clip_id": "{video_id}@{scale}@t{start_ms:09d}",
        "expected_clip_counts": PLAN_COUNTS,
        "expected_total_clips": PLAN_TOTAL,
    },
    "inference": {
        "compute_precision": "bf16",
        "model_weight_dtype": CONFIG["model_weight_dtype"],
        "autocast": False,
        "attention_implementation": CONFIG["attention_implementation"],
        "pooling": "last_token_final_hidden_state",
        "eval": True,
        "inference_mode": True,
        "quantization": None,
        "l2_normalize": True,
        "normalization_dtype": "float32",
    },
    "storage": {
        "format": "parquet_fixed_size_list",
        "dtype": CONFIG["storage_dtype"],
        "shard_rows": CONFIG["shard_rows"],
        "compression": None,
    },
}
SEMANTIC_FINGERPRINT = sha256_bytes(canonical_json_bytes(SEMANTIC_CONFIG))
print("Semantic fingerprint:", SEMANTIC_FINGERPRINT)

QUERY_CONTRACT = {
    "model_id": CONFIG["model_id"],
    "model_revision": CONFIG["model_revision"],
    "text_eol_prompt": CONFIG["text_eol_prompt"],
    "input": "text only, prompt = EOL_PROMPTS['text'] với <sent> thay bằng query",
    "api": "TARA.encode_text(query) rồi F.normalize(z.float(), p=2, dim=-1)",
    "language": "English (TARA fine-tune trên NLI-Nuance tiếng Anh; dịch query VI→EN trước)",
    "pooling": "last_token_final_hidden_state",
    "dimension": CONFIG["embedding_dim"],
    "l2_normalize_fp32": True,
    "metric": "COSINE",
    "note": (
        "Không cộng thẳng cosine của TARA với PE/Qwen. Fuse bằng rank/RRF vì "
        "score distribution của từng embedding space khác nhau."
    ),
}

RUN_CONFIG_REMOTE_PATH = f"{CONFIG['output_prefix'].strip('/')}/run_config.json"
RUN_CONFIG_PAYLOAD = {
    "semantic_fingerprint": SEMANTIC_FINGERPRINT,
    "semantic_config": SEMANTIC_CONFIG,
    "query_contract": QUERY_CONTRACT,
    "parallel_plan": CONFIG["session_assignments"],
}


def assert_run_config_compatible(remote_config):
    if remote_config.get("semantic_fingerprint") != SEMANTIC_FINGERPRINT:
        raise RuntimeError(
            "Output prefix đã tồn tại với semantic fingerprint khác. Để bảo vệ vector cũ, "
            "hãy khôi phục đúng CONFIG cũ hoặc dùng output_prefix mới.\n"
            f"local={SEMANTIC_FINGERPRINT}\nremote={remote_config.get('semantic_fingerprint')}"
        )
    remote_plan = remote_config.get("parallel_plan")
    if not isinstance(remote_plan, dict) or not remote_plan:
        return
    remote_owners = {}
    for session_id, categories in remote_plan.items():
        if not isinstance(categories, list) or not categories:
            raise RuntimeError(f"Remote parallel_plan rỗng/sai kiểu ở {session_id}.")
        for category in categories:
            if category in remote_owners:
                raise RuntimeError(
                    f"Remote parallel_plan overlap {category}: "
                    f"{remote_owners[category]} và {session_id}."
                )
            remote_owners[category] = session_id
    if set(remote_owners) != set(ALL_CATEGORIES):
        raise RuntimeError("Remote parallel_plan không phủ đúng M01..M10 + N001..N100 + S01.")
    if remote_plan != CONFIG["session_assignments"]:
        active = CONFIG["active_session"]
        if active not in remote_plan:
            raise RuntimeError(f"active_session={active} không có trong remote parallel_plan.")
        local_plan = CONFIG["session_assignments"]
        CONFIG["session_assignments"] = remote_plan
        CONFIG["categories"] = list(remote_plan[active])
        RUN_CONFIG_PAYLOAD["parallel_plan"] = remote_plan
        print("Remote run_config là source of truth cho parallel_plan.",
              {"local": local_plan, "remote": remote_plan, "active": CONFIG["categories"]})


existing_run_config = (
    read_remote_json_optional(CONFIG["output_bucket_id"], RUN_CONFIG_REMOTE_PATH)
    if CONFIG["upload"] else None
)
if existing_run_config is not None:
    assert_run_config_compatible(existing_run_config)
    print("Remote run_config tương thích: resume được phép.")
elif CONFIG["upload"]:
    preexisting = list(list_bucket_tree(
        CONFIG["output_bucket_id"], prefix=CONFIG["output_prefix"].strip("/"),
        recursive=True, token=HF_TOKEN,
    ))
    if preexisting:
        raced = read_remote_json_optional(CONFIG["output_bucket_id"], RUN_CONFIG_REMOTE_PATH)
        if raced is None:
            raise RuntimeError(
                "Output prefix đã có object nhưng thiếu run_config.json; từ chối ghi đè. "
                f"Object đầu tiên: {[i.path for i in preexisting[:20]]}"
            )
        assert_run_config_compatible(raced)
        print("Session khác vừa khởi tạo run_config; cấu hình tương thích.")
    else:
        upload_bytes_verified(
            CONFIG["output_bucket_id"], RUN_CONFIG_REMOTE_PATH,
            json.dumps(RUN_CONFIG_PAYLOAD, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        persisted = read_remote_json_optional(CONFIG["output_bucket_id"], RUN_CONFIG_REMOTE_PATH)
        if persisted is None:
            raise RuntimeError("Không đọc lại được run_config sau upload.")
        assert_run_config_compatible(persisted)
        print("Đã tạo remote run_config.")

# %% [markdown]
# ## 7. Worker TARA: reader có cache, forward bỏ `lm_head`, RPC JSON-lines
#
# Ba điểm kỹ thuật trong worker và lý do:
#
# 1. **`CachedVideoReaderShim`.** `tarsier2 ... utils.VideoReader` mở lại file mp4
#    và seek cho mỗi lần `sample_video`. Với ~193 clip/video × 8 frame, đó là hàng
#    nghìn seek ngẫu nhiên trên một file 300 MB. Shim giữ reader mở + LRU frame
#    cache và prefetch theo batch. Nội dung frame giữ nguyên tuyệt đối
#    (`get_batch → asnumpy → Image.fromarray().convert("RGB")`).
# 2. **`frame_indices` tính sẵn.** `VisionParser` chấp nhận cả
#    `start_time`/`end_time` lẫn `frame_indices`. Worker tự tính bằng bản sao chính
#    xác của `sample_video` rồi truyền `frame_indices` — deterministic, ghi được
#    vào Parquet, và audit sẽ so trực tiếp với `sample_video` thật.
#
#    Preprocess chạy trên **một** thread (`prep_threads = 1`), xếp hàng trước GPU
#    qua `prefetch_batches`. Không phải để tiết kiệm CPU mà vì bắt buộc:
#    transformers 4.45 memoize compiled Jinja template bằng `@lru_cache`, nên
#    `AssistantTracker` của block `{% generation %}` là singleton có state — gọi
#    `apply_chat_template` song song sẽ ném *"AssistantTracker should not be reused
#    before closed"*. Worker còn giữ thêm một lock quanh `super_processor` để ràng
#    buộc này không phụ thuộc vào việc ai đó chỉnh config.
# 3. **Bỏ `lm_head`.** `generate(max_new_tokens=1)` vẫn chạy `lm_head` trên toàn
#    bộ ~4.700 token (152.064-way, ≈1,4 GiB BF16) rồi chỉ dùng hidden state. Worker
#    thay `lm_head` bằng slice last-token nên `outputs.logits` **chính là** hidden
#    state sau final norm — cùng tensor mà `hidden_states[0][-1][:, -1, :]` trả về.
#
# Cả ba đều được gate bằng audit ở mục 9 trước khi ghi bất kỳ vector nào.

# %%
WORKER_SOURCE = r'''#!/usr/bin/env python3
# TARA clip-embedding worker.
#
# Chay trong venv Python 3.10 rieng (torch 2.5.1+cu121 / transformers 4.45.0 /
# flash-attn 2.8.3) vi TARA pin dung stack do; Colab kernel Python 3.13 khong
# cai duoc numpy==1.26.4 va khong chay duoc transformers 4.45 API.
#
# Giao thuc: JSON-lines tren stdin; response tren stdout co prefix @@TARA-RPC@@.
# Moi dong stdout khac deu la log cua library va duoc notebook echo nguyen van.
import contextlib
import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor

PROTO = "@@TARA-RPC@@"
_EMIT_LOCK = threading.Lock()


def emit(obj):
    with _EMIT_LOCK:
        sys.stdout.write(PROTO + json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def note(message):
    print("[worker] " + str(message), flush=True)


S = {
    "model": None,
    "torch": None,
    "np": None,
    "F": None,
    "tds_utils": None,
    "orig_video_reader": None,
    "format_one_sample": None,
    "video_sampling_strategy": None,
    "n_frames": None,
    "device": None,
    "last_token_head": None,
    "bypass_lm_head": True,
    "prep_threads": 3,
    "prefetch_batches": 2,
    "decord_threads": 4,
    "substitution_window": 8,
    "probe_max_steps": 16,
}


# ---------------------------------------------------------------- shared stub
def ensure_shared_utils(model_dir):
    # BaseModelForTARA.__init__ chi dung su.io.load_yml va su.log.repo_path,
    # nhung package `shared.utils` keo theo matplotlib/cv2/sklearn/seaborn.
    # Neu thieu bat ky dep visualize nao thi fallback sang stub tuong duong.
    try:
        import shared.utils  # noqa: F401
        return "real"
    except Exception as exc:
        note("import shared.utils that bai (" + type(exc).__name__ + ": " + str(exc) + "); dung stub toi thieu.")

    import types
    import yaml

    shared_mod = sys.modules.get("shared") or types.ModuleType("shared")
    utils_mod = types.ModuleType("shared.utils")
    io_mod = types.ModuleType("shared.utils.io")
    log_mod = types.ModuleType("shared.utils.log")

    def load_yml(path):
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    io_mod.load_yml = load_yml
    log_mod.repo_path = model_dir
    utils_mod.io = io_mod
    utils_mod.log = log_mod
    shared_mod.utils = utils_mod
    sys.modules["shared"] = shared_mod
    sys.modules["shared.utils"] = utils_mod
    sys.modules["shared.utils.io"] = io_mod
    sys.modules["shared.utils.log"] = log_mod
    return "stub"


# ------------------------------------------------------------- frame source
# Frame cua moi batch duoc decode dung mot lan roi giu trong buffer THREAD-LOCAL
# cua chinh thread do.
#
# Ban truoc dung mot LRU cache dung chung giua cac prep thread va no sai: get()
# tinh `missing` (bo qua cac frame dang hit), decode phan missing, roi vong
# `while len(cache) > max_frames: popitem(last=False)` co the evict chinh nhung
# frame dang hit do truoc khi chung kip duoc doc ra -> KeyError. Mo phong lai
# tren plan that cho thay loi xay ra ngay ca khi chi co mot thread.
#
# LRU 512 frame chi giam so frame phai decode tu 7,5 xuong 6,6 moi clip (~12%)
# trong khi decode khong phai bottleneck (GPU ~0,85 s/clip), va no ton ~1,4 GiB
# RAM. Buffer theo batch giu nguyen phan quan trong - mot get_batch cho ca batch,
# tuc la decode tuan tu thay vi seek ngau nhien - ma khong the evict nham.
_TL = threading.local()
_SOURCE_LOCK = threading.RLock()
_ACTIVE_SOURCE = {"source": None}


class VideoFrameSource:
    def __init__(self, path, num_threads, substitution_window, probe_max_steps):
        import decord
        from PIL import Image

        self._decord = decord
        self._Image = Image
        self.path = path
        self.num_threads = int(num_threads)
        self.substitution_window = int(substitution_window)
        self.probe_max_steps = int(probe_max_steps)
        self.lock = threading.RLock()
        self.decoded_frames = 0
        self.decode_seconds = 0.0
        self.substitutions = 0
        self.reopens = 0
        self.vr = None
        self._open()
        self.reported_length = self._raw_length
        self._length = self._probe_decodable_length()

    # -- lifecycle ----------------------------------------------------------
    def _open(self):
        decord = self._decord
        self.vr = decord.VideoReader(
            self.path, num_threads=self.num_threads, ctx=decord.cpu(0), fault_tol=1
        )
        self.vr.seek(0)
        self._raw_length = len(self.vr)
        self._fps = self.vr.get_avg_fps()

    def _reopen(self):
        # Sau mot DECORDError, state ben trong reader co the hong; mo lai la cach
        # duy nhat chac chan de lan decode sau khong keo theo loi cu.
        self.reopens += 1
        self.vr = None
        self._open()

    def close(self):
        with self.lock:
            self.vr = None

    @property
    def length(self):
        return self._length

    @property
    def fps(self):
        return self._fps

    # -- decoding -----------------------------------------------------------
    def _raw_batch(self, indices):
        self._decord.bridge.set_bridge("native")
        started = time.monotonic()
        batch = self.vr.get_batch(list(indices)).asnumpy()
        self.decode_seconds += time.monotonic() - started
        self.decoded_frames += len(indices)
        return batch

    def _to_pil(self, array):
        return self._Image.fromarray(array).convert("RGB")

    def _probe(self, index):
        try:
            with self.lock:
                self._raw_batch([index])
            return True
        except Exception as exc:
            note(
                "Frame " + str(index) + " cua " + os.path.basename(self.path)
                + " khong doc duoc (" + type(exc).__name__ + ")."
            )
            self._reopen()
            return False

    def _probe_decodable_length(self):
        # decord co the bao len(vr) lon hon so frame no thuc su doc duoc o cuoi
        # file: "Unable to handle EOF ... DECORD_EOF_RETRY_MAX". Tim frame cuoi
        # doc duoc that roi tra ve do dai do, de plan_clip_indices truot window
        # lui lai bang dung co che clamp da co san.
        #
        # Moi lan probe TRUOT ton nguyen budget retry cua decord nen rat cham;
        # vi vay lui theo buoc nhan doi de cham day nhanh, roi binary search
        # giua diem tot va diem xau gan nhat de lay con so chinh xac (phan lon
        # cac probe trong pha nay roi vao frame doc duoc, tuc la nhanh).
        length = self._raw_length
        if length <= 0:
            raise RuntimeError("decord bao 0 frame cho " + self.path)
        if self._probe(length - 1):
            return length

        first_bad = length - 1
        good = -1
        probe = length - 2
        step = 1
        budget = self.probe_max_steps
        while probe >= 0 and budget > 0:
            budget -= 1
            if self._probe(probe):
                good = probe
                break
            first_bad = probe
            step = min(step * 2, 64)
            probe -= step
        if good < 0:
            raise RuntimeError(
                "Khong tim duoc frame doc duoc nao gan cuoi " + self.path
                + " sau " + str(self.probe_max_steps) + " lan thu."
            )

        refine_budget = self.probe_max_steps
        while first_bad - good > 1 and refine_budget > 0:
            refine_budget -= 1
            middle = (good + first_bad) // 2
            if self._probe(middle):
                good = middle
            else:
                first_bad = middle
        note(
            "EOF trim " + os.path.basename(self.path) + ": "
            + str(length) + " -> " + str(good + 1) + " frame doc duoc."
        )
        return good + 1

    def _decode_one(self, index):
        # Fallback cuoi cung: neu mot frame khong decode duoc, lay frame gan nhat
        # truoc no. Lech toi da substitution_window frame (~0,3 s) va duoc dem
        # vao `substitutions` de commit metadata phan anh dung su that.
        last_error = None
        for offset in range(0, self.substitution_window + 1):
            candidate = index - offset
            if candidate < 0:
                break
            try:
                batch = self._raw_batch([candidate])
            except Exception as exc:
                last_error = exc
                self._reopen()
                continue
            if offset:
                self.substitutions += 1
                note(
                    "Frame " + str(index) + " loi; thay bang frame " + str(candidate)
                    + " (" + os.path.basename(self.path) + ")."
                )
            return self._to_pil(batch[0])
        raise RuntimeError(
            "Khong decode duoc frame " + str(index) + " cua " + self.path
        ) from last_error

    def decode(self, indices):
        """Tra ve dict {frame_index: PIL.Image} cho cac index yeu cau."""
        unique = sorted({int(i) for i in indices})
        if not unique:
            return {}
        with self.lock:
            try:
                batch = self._raw_batch(unique)
                return {index: self._to_pil(array) for index, array in zip(unique, batch)}
            except Exception as exc:
                note(
                    "get_batch loi tren " + os.path.basename(self.path) + " ("
                    + type(exc).__name__ + ": " + str(exc).split("\n")[0]
                    + "); mo lai reader va decode tung frame."
                )
                self._reopen()
            return {index: self._decode_one(index) for index in unique}

    def stats(self):
        return {
            "reported_frames": self.reported_length,
            "decodable_frames": self._length,
            "eof_trimmed_frames": self.reported_length - self._length,
            "decoded_frames": self.decoded_frames,
            "decode_seconds": self.decode_seconds,
            "frame_substitutions": self.substitutions,
            "reader_reopens": self.reopens,
        }


def reset_reader():
    with _SOURCE_LOCK:
        source = _ACTIVE_SOURCE["source"]
        if source is not None:
            source.close()
        _ACTIVE_SOURCE["source"] = None


def active_source(path):
    # Goi tu nhieu prep thread nen viec tao source phai nam trong lock, neu
    # khong hai thread co the cung mo mot decord reader cho cung mot file.
    with _SOURCE_LOCK:
        source = _ACTIVE_SOURCE["source"]
        if source is None or source.path != path or source.vr is None:
            reset_reader()
            _ACTIVE_SOURCE["source"] = VideoFrameSource(
                path, S["decord_threads"], S["substitution_window"], S["probe_max_steps"]
            )
        return _ACTIVE_SOURCE["source"]


class BatchFrameReaderShim:
    # Drop-in cho tarsier2.dataset.custom_data_parsers.utils.VideoReader.
    # Noi dung frame giong het ban goc (get_batch -> asnumpy ->
    # Image.fromarray().convert("RGB")); chi khac o cho reader duoc giu mo va
    # frame duoc decode theo batch.
    def __init__(self, path):
        self.source = active_source(path)

    @property
    def length(self):
        return self.source.length

    @property
    def fps(self):
        return self.source.fps

    def sample(self, frame_indices):
        wanted = [int(i) for i in frame_indices]
        buffer = getattr(_TL, "frames", None)
        if buffer is None:
            # Duong audit/self-test goi thang preprocess_samples, khong qua
            # prepare(), nen khong co buffer: decode truc tiep.
            buffer = self.source.decode(wanted)
        else:
            missing = [i for i in wanted if i not in buffer]
            if missing:
                buffer.update(self.source.decode(missing))
        return [buffer[i] for i in wanted]

    def preprocess(self):
        return self.source.path

    def postprocess(self):
        return None


@contextlib.contextmanager
def batch_frame_buffer(source, indices):
    _TL.frames = source.decode(indices)
    try:
        yield
    finally:
        _TL.frames = None


@contextlib.contextmanager
def official_reader():
    # Tam thoi tra lai VideoReader goc de audit doi chieu voi duong chinh chu.
    tds = S["tds_utils"]
    patched = tds.VideoReader
    tds.VideoReader = S["orig_video_reader"]
    try:
        yield
    finally:
        tds.VideoReader = patched


# ------------------------------------------------------------ frame indexing
def official_frame_indices(total_frames, fps, start_time, end_time, n_frames):
    # Ban sao chinh xac nhanh start_time/end_time trong tarsier2 sample_video,
    # gom _sample_frame_indices_v1 va check_frame_indices.
    start_frame = int(round(float(start_time) * fps))
    end_frame = int(round(float(end_time) * fps))
    if end_frame == total_frames:
        end_frame -= 1
    span = end_frame - start_frame + 1
    if n_frames == 1:
        indices = [0]
    elif span <= n_frames:
        indices = list(range(span))
    else:
        indices = [int(round(i * (span - 1) / (n_frames - 1))) for i in range(n_frames)]
    indices = [i + start_frame for i in indices]
    if indices and indices[-1] == total_frames:
        indices[-1] = total_frames - 1
    return [i for i in indices if 0 <= i < total_frames]


def plan_clip_indices(total_frames, fps, start_time, end_time, n_frames):
    # Tra ve (frame_indices, clamped). Neu duration_sec trong manifest dai hon so
    # frame decoder thuc su doc duoc (clip duoi cung cua video), truot window lui
    # lai thay vi bop ngan no: giu nguyen do dai window nen semantics cua scale
    # khong doi. start_time/end_time trong Parquet van la gia tri plan (identity);
    # frame_indices moi la ground truth cua nhung frame da di vao model.
    indices = official_frame_indices(total_frames, fps, start_time, end_time, n_frames)
    if len(indices) == n_frames:
        return indices, False
    safe_end = (total_frames - 1) / float(fps)
    window = float(end_time) - float(start_time)
    safe_start = max(0.0, min(float(start_time), safe_end - window))
    indices = official_frame_indices(total_frames, fps, safe_start, safe_end, n_frames)
    if len(indices) != n_frames:
        raise RuntimeError(
            "Khong lay du " + str(n_frames) + " frame cho window ["
            + str(start_time) + ", " + str(end_time) + "] tren video "
            + str(total_frames) + " frame @ " + str(fps) + " fps."
        )
    return indices, True


# ------------------------------------------------------------ sample builder
def build_sample(video_path, frame_indices=None, start_time=None, end_time=None):
    # Dung format_one_sample chinh chu roi chi bo sung key chon frame, nen
    # message/task/prompt con lai giong het duong chinh chu.
    sample = S["format_one_sample"](media_file=video_path, prompt=S["model"].video_eol_prompt)
    video_item = sample["messages"][0]["content"][0]["video"]
    if frame_indices is not None:
        video_item["frame_indices"] = [int(i) for i in frame_indices]
    if start_time is not None:
        video_item["start_time"] = float(start_time)
    if end_time is not None:
        video_item["end_time"] = float(end_time)
    return sample


# transformers 4.45 memoize compiled Jinja template bang @lru_cache, nen
# AssistantTracker cua block {% generation %} la singleton co state:
# `if self._rendered_blocks or self._generation_indices: raise ValueError(
#  "AssistantTracker should not be reused before closed")`.
# Template cua TARA co {% generation %} va TarsierProcessor luon bat
# return_assistant_tokens_mask, nen MOI clip deu di qua doan do. Hai lan render
# chong nhau la hong. Lock nay bao dam dieu do khong the xay ra du prep_threads
# co bi chinh len bao nhieu (notebook cung da validate prep_threads == 1).
_PROCESSOR_LOCK = threading.RLock()


def preprocess_samples(samples):
    # TarsierDataProcessor.transform nuot exception va tra ve [], nen loi that
    # chi hien ra duoi dang IndexError vo nghia. In kem video + frame indices.
    samples = list(samples)
    try:
        with _PROCESSOR_LOCK:
            return S["model"].super_processor(samples)
    except Exception as exc:
        detail = []
        for sample in samples[:4]:
            try:
                item = sample["messages"][0]["content"][0]["video"]
                detail.append({
                    "video_file": item.get("video_file"),
                    "frame_indices": item.get("frame_indices"),
                })
            except Exception:
                detail.append("unparsed")
        raise RuntimeError(
            "Tarsier preprocess that bai cho " + str(len(samples))
            + " sample; " + json.dumps(detail, ensure_ascii=False)
        ) from exc


# ------------------------------------------------------------------ forwards
def to_device(model_inputs, drop_labels):
    torch = S["torch"]
    out = {}
    for key, value in model_inputs.items():
        if not torch.is_tensor(value):
            continue
        if drop_labels and key == "labels":
            continue
        out[key] = value.to(S["device"], non_blocking=True)
    return out


@contextlib.contextmanager
def last_token_head():
    # Thay lm_head bang slice last-token: hidden state cuoi cung khong doi,
    # nhung tiet kiem ~1.4 GiB logits BF16 moi clip va bo luon lm_head matmul.
    language_model = S["model"].model.language_model
    real_head = language_model.lm_head
    language_model.lm_head = S["last_token_head"]
    try:
        yield
    finally:
        language_model.lm_head = real_head


def production_forward(model_inputs):
    torch = S["torch"]
    inputs = to_device(model_inputs, drop_labels=True)
    with torch.inference_mode():
        if S["bypass_lm_head"]:
            with last_token_head():
                outputs = S["model"].model(
                    **inputs, use_cache=False, output_hidden_states=False, return_dict=True
                )
            raw = outputs.logits[:, -1, :]
        else:
            outputs = S["model"].model(
                **inputs, use_cache=False, output_hidden_states=True, return_dict=True
            )
            raw = outputs.hidden_states[-1][:, -1, :]
    return S["F"].normalize(raw.float(), p=2, dim=-1)


def official_forward(model_inputs):
    # Giong het TARA.encode_vision: generate(max_new_tokens=1) roi lay
    # hidden_states[0][-1][:, -1, :].
    torch = S["torch"]
    inputs = to_device(model_inputs, drop_labels=False)
    with torch.inference_mode():
        outputs = S["model"].model.generate(
            **inputs,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=S["model"].processor.tokenizer.eos_token_id,
        )
        raw = outputs.hidden_states[0][-1][:, -1, :]
    return S["F"].normalize(raw.float(), p=2, dim=-1)


# ------------------------------------------------------------------ commands
def cmd_init(req):
    # Phai set TRUOC khi decord duoc import lan dau. Mot so mp4 cua corpus lam
    # decord het budget retry khi doc vai frame cuoi:
    #   "Unable to handle EOF ... DECORD_EOF_RETRY_MAX=10240".
    os.environ["DECORD_EOF_RETRY_MAX"] = str(int(req.get("decord_eof_retry_max", 65536)))
    model_dir = os.path.abspath(req["model_dir"])
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    shared_mode = ensure_shared_utils(model_dir)

    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml
    import decord
    import transformers

    S["torch"] = torch
    S["np"] = np
    S["F"] = F
    S["bypass_lm_head"] = bool(req.get("bypass_lm_head", True))
    S["prep_threads"] = int(req.get("prep_threads", 3))
    S["prefetch_batches"] = int(req.get("prefetch_batches", 2))
    S["decord_threads"] = int(req.get("decord_threads", 4))
    S["substitution_window"] = int(req.get("substitution_window", 8))
    S["probe_max_steps"] = int(req.get("probe_max_steps", 16))

    if not torch.cuda.is_available():
        raise RuntimeError("Worker yeu cau CUDA GPU.")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config_path = os.path.join(model_dir, "tarsier2", "default_config.yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    for key, value in req["expected_default_config"].items():
        if base_config.get(key) != value:
            raise RuntimeError(
                "default_config.yaml khac contract o key=" + key
                + ": " + repr(base_config.get(key)) + " != " + repr(value)
            )
    S["n_frames"] = int(base_config["n_frames"])
    S["video_sampling_strategy"] = dict(base_config["video_sampling_strategy"])

    from modeling_tara import TARA, EOL_PROMPTS
    from tarsier2.dataset.utils import format_one_sample
    import tarsier2.dataset.custom_data_parsers.utils as tds_utils

    S["format_one_sample"] = format_one_sample
    S["tds_utils"] = tds_utils
    S["orig_video_reader"] = tds_utils.VideoReader

    note("Nap TARA tu " + model_dir)
    model = TARA.from_pretrained(
        model_dir,
        device_map={"": 0},
        attn_implementation=req.get("attn_implementation", "flash_attention_2"),
        low_cpu_mem_usage=True,
    )
    model.model.requires_grad_(False)
    model.model.eval()
    S["model"] = model

    first_parameter = next(model.model.parameters())
    S["device"] = first_parameter.device

    class LastTokenHead(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states[:, -1:, :]

    S["last_token_head"] = LastTokenHead()

    # Chi cai shim reader sau khi model da nap xong.
    tds_utils.VideoReader = BatchFrameReaderShim

    processor = model.processor
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer
    text_config = model.model.config.text_config

    if str(first_parameter.dtype) != "torch.bfloat16":
        raise RuntimeError("Model dtype khong phai bfloat16: " + str(first_parameter.dtype))
    if tokenizer.padding_side != "left":
        raise RuntimeError("Pooling last-token yeu cau left padding, dang la " + str(tokenizer.padding_side))
    if int(text_config.hidden_size) != int(req["embedding_dim"]):
        raise RuntimeError("hidden_size khac embedding_dim contract.")
    if int(model.super_processor.max_pixels) != int(req["max_pixels"]):
        raise RuntimeError("max_pixels khac contract.")
    if int(model.super_processor.n_frames) != int(req["n_frames"]):
        raise RuntimeError("n_frames khac contract.")

    checks = {
        "dtype": str(first_parameter.dtype),
        "device": str(first_parameter.device),
        "padding_side": tokenizer.padding_side,
        "attn_implementation": getattr(model.model.config, "_attn_implementation", None),
        "hidden_size": int(text_config.hidden_size),
        "num_hidden_layers": int(text_config.num_hidden_layers),
        "vocab_size": int(text_config.vocab_size),
        "patch_size": int(image_processor.patch_size),
        "merge_size": int(image_processor.merge_size),
        "temporal_patch_size": int(image_processor.temporal_patch_size),
        "image_mean": [float(x) for x in image_processor.image_mean],
        "image_std": [float(x) for x in image_processor.image_std],
        "processor_n_frames": int(model.super_processor.n_frames),
        "processor_max_n_frames": int(model.super_processor.max_n_frames),
        "processor_max_pixels": int(model.super_processor.max_pixels),
        "processor_min_pixels": int(model.super_processor.min_pixels),
        "processor_max_seq_len": int(model.super_processor.processor.max_seq_len),
        "processor_is_training": bool(model.super_processor.is_training),
        "processor_do_image_resize": bool(model.super_processor.do_image_resize),
        "processor_do_image_padding": bool(model.super_processor.do_image_padding),
        "processor_do_image_crop": bool(model.super_processor.do_image_crop),
        "video_eol_prompt": model.video_eol_prompt,
        "text_eol_prompt": model.text_eol_prompt,
        "eol_prompts": dict(EOL_PROMPTS),
        "video_sampling_strategy": S["video_sampling_strategy"],
    }
    return {
        "shared_utils": shared_mode,
        "n_params": int(sum(p.numel() for p in model.model.parameters())),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "decord": str(getattr(decord, "__version__", "unknown")),
        "decord_eof_retry_max": os.environ.get("DECORD_EOF_RETRY_MAX"),
        "python": sys.version.split()[0],
        "gpu": torch.cuda.get_device_name(0),
        "gpu_vram_gib": torch.cuda.get_device_properties(0).total_memory / 2 ** 30,
        "base_config": base_config,
        "checks": checks,
    }


def _video_geometry(video_path):
    source = active_source(video_path)
    return int(source.length), float(source.fps)


def _batched(items, batch_size):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _build_plans(video_path, clips):
    total_frames, fps = _video_geometry(video_path)
    plans = []
    clamped = 0
    for clip in clips:
        indices, was_clamped = plan_clip_indices(
            total_frames, fps, float(clip["start_time"]), float(clip["end_time"]), S["n_frames"]
        )
        clamped += int(was_clamped)
        # `reverse` chi dung cho arrow-of-time self test: cung y het frame nhung
        # dua vao model theo thu tu nguoc. Pipeline production khong bao gio set.
        if clip.get("reverse"):
            indices = list(reversed(indices))
        plans.append({"frame_indices": indices})
    return plans, clamped, total_frames, fps


def _run_batches(video_path, plans, batch_size, forward_fn, progress_label=None, progress_total=None):
    # Pool CO DUNG MOT worker (S["prep_threads"] == 1). Muc dich khong phai chay
    # preprocess song song - apply_chat_template cua transformers 4.45 khong
    # thread-safe - ma la OVERLAP: trong khi main thread chay GPU forward cho
    # batch k thi worker chuan bi batch k+1. `prefetch_batches` quyet dinh do sau
    # hang doi.
    torch = S["torch"]
    if not plans:
        return None
    batches = _batched(plans, max(1, int(batch_size)))
    results = []
    done = 0
    started = time.monotonic()
    last_report = started

    def prepare(batch):
        source = active_source(video_path)
        indices = sorted({i for p in batch for i in p["frame_indices"]})
        with batch_frame_buffer(source, indices):
            samples = [
                build_sample(video_path, frame_indices=p["frame_indices"]) for p in batch
            ]
            return preprocess_samples(samples)

    with ThreadPoolExecutor(max_workers=max(1, S["prep_threads"])) as pool:
        pending = deque()
        cursor = 0
        while cursor < len(batches) and len(pending) < max(1, S["prefetch_batches"]):
            pending.append((batches[cursor], pool.submit(prepare, batches[cursor])))
            cursor += 1
        while pending:
            batch, future = pending.popleft()
            model_inputs = future.result()
            embeddings = forward_fn(model_inputs)
            results.append(embeddings.detach().to("cpu"))
            done += len(batch)
            del model_inputs, embeddings
            if cursor < len(batches):
                pending.append((batches[cursor], pool.submit(prepare, batches[cursor])))
                cursor += 1
            now = time.monotonic()
            if progress_label is not None and now - last_report >= 3.0:
                emit({
                    "event": "progress",
                    "label": progress_label,
                    "done": done,
                    "total": progress_total if progress_total is not None else len(plans),
                    "clips_per_s": done / max(1e-6, now - started),
                })
                last_report = now
    return torch.cat(results, dim=0)


def cmd_encode_video(req):
    torch = S["torch"]
    np = S["np"]
    video_path = req["video_path"]
    clips = req["clips"]
    if not clips:
        raise RuntimeError("encode_video nhan danh sach clip rong.")
    plans, clamped, total_frames, fps = _build_plans(video_path, clips)

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    embeddings = _run_batches(
        video_path, plans, int(req["batch_size"]), production_forward,
        progress_label=req.get("video_id", os.path.basename(video_path)),
        progress_total=len(plans),
    )
    elapsed = time.monotonic() - started

    array = embeddings.numpy().astype(np.float32, copy=False)
    if array.shape != (len(clips), int(req["embedding_dim"])):
        raise RuntimeError("Embedding shape sai: " + str(array.shape))
    if not np.isfinite(array).all():
        raise RuntimeError("Embedding chua NaN/Inf.")
    norms = np.linalg.norm(array, axis=1)
    np.savez(
        req["out_npz"],
        embeddings=array,
        frame_indices=np.asarray([p["frame_indices"] for p in plans], dtype=np.int32),
        rows=np.asarray([int(c["row"]) for c in clips], dtype=np.int64),
    )

    response = {
        "rows": len(clips),
        "out_npz": req["out_npz"],
        "total_frames": total_frames,
        "fps": fps,
        "clamped_clips": clamped,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "elapsed_s": elapsed,
        "clips_per_s": len(clips) / max(1e-6, elapsed),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
    }
    response.update(active_source(video_path).stats())
    return response


def cmd_benchmark(req):
    torch = S["torch"]
    video_path = req["video_path"]
    plans, _, _, _ = _build_plans(video_path, req["clips"])
    total_vram = torch.cuda.get_device_properties(0).total_memory
    oom_errors = (torch.cuda.OutOfMemoryError, RuntimeError)
    results = []
    for candidate in req["batch_candidates"]:
        batch_size = int(candidate)
        if batch_size <= 0 or batch_size > len(plans):
            continue
        subset = plans[:max(batch_size * int(req.get("rounds", 3)), batch_size)]
        try:
            _run_batches(video_path, plans[:batch_size], batch_size, production_forward)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            _run_batches(video_path, subset, batch_size, production_forward)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            peak_reserved = torch.cuda.max_memory_reserved()
            results.append({
                "batch_size": batch_size,
                "clips": len(subset),
                "clips_per_s": len(subset) / max(1e-6, elapsed),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                "peak_reserved_gib": peak_reserved / 2 ** 30,
                "peak_fraction": peak_reserved / total_vram,
            })
            note("batch=" + str(batch_size) + " -> " + json.dumps(results[-1]))
        except oom_errors as exc:
            if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            # free thap ma reserved cua process nay cung thap => process khac dang giu VRAM.
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            note(
                "OOM tai batch=" + str(batch_size) + ": VRAM trong "
                + str(round(free_bytes / 2 ** 30, 1)) + "/" + str(round(total_bytes / 2 ** 30, 1))
                + " GiB, process nay reserved " + str(round(torch.cuda.memory_reserved() / 2 ** 30, 1))
                + " GiB; " + str(exc).split("\n")[0][:200] + "; dung tang batch."
            )
            break
        finally:
            torch.cuda.empty_cache()
    return {"results": results}


def cmd_audit(req):
    torch = S["torch"]
    F = S["F"]
    video_path = req["video_path"]
    clips = req["clips"]
    batch_size = int(req["batch_size"])
    n_frames = S["n_frames"]
    total_frames, fps = _video_geometry(video_path)

    # (1) Frame-index parity: cong thuc tu tinh vs sample_video chinh chu.
    index_mismatch = []
    with official_reader():
        for clip in clips:
            mine, _ = plan_clip_indices(
                total_frames, fps, float(clip["start_time"]), float(clip["end_time"]), n_frames
            )
            _, theirs = S["tds_utils"].sample_video(
                video_path=video_path,
                start_time=float(clip["start_time"]),
                end_time=float(clip["end_time"]),
                n_frames=n_frames,
                is_training=False,
                video_sampling_strategy=S["video_sampling_strategy"],
                return_frame_ids=True,
            )
            if [int(i) for i in mine] != [int(i) for i in theirs]:
                index_mismatch.append({
                    "start_time": clip["start_time"],
                    "end_time": clip["end_time"],
                    "mine": [int(i) for i in mine],
                    "official": [int(i) for i in theirs],
                })

    # (2) Full-video parity: TARA.encode_vision goc vs production forward.
    reset_reader()
    with official_reader():
        with torch.inference_mode():
            reference_full = S["model"].encode_vision(video_path).float()
    reference_full = F.normalize(reference_full, p=2, dim=-1).to("cpu")
    reset_reader()
    production_full = production_forward(preprocess_samples([build_sample(video_path)])).to("cpu")
    full_cosine = F.cosine_similarity(production_full, reference_full, dim=-1)

    # (3) Clip parity: duong chinh chu (start/end + reader goc + generate) vs
    #     duong production (frame_indices + shim reader + forward khong lm_head).
    official_chunks = []
    reset_reader()
    with official_reader():
        for clip in clips:
            sample = build_sample(
                video_path, start_time=float(clip["start_time"]), end_time=float(clip["end_time"])
            )
            official_chunks.append(official_forward(preprocess_samples([sample])).to("cpu"))
    official_clip = torch.cat(official_chunks, dim=0)

    reset_reader()
    plans, _, _, _ = _build_plans(video_path, clips)
    production_single = _run_batches(video_path, plans, 1, production_forward)
    production_batched = _run_batches(video_path, plans, batch_size, production_forward)

    clip_cosine = F.cosine_similarity(production_single, official_clip, dim=-1)
    batch_cosine = F.cosine_similarity(production_batched, production_single, dim=-1)
    return {
        "sample_count": len(clips),
        "batch_size": batch_size,
        "frame_index_mismatches": index_mismatch,
        "full_video_min_cosine": float(full_cosine.min()),
        "clip_vs_official_min_cosine": float(clip_cosine.min()),
        "clip_vs_official_mean_cosine": float(clip_cosine.mean()),
        "batch_vs_single_min_cosine": float(batch_cosine.min()),
        "batch_vs_single_mean_cosine": float(batch_cosine.mean()),
        "norm_min": float(production_batched.norm(dim=-1).min()),
        "norm_max": float(production_batched.norm(dim=-1).max()),
        "bypass_lm_head": S["bypass_lm_head"],
    }


def cmd_encode_text(req):
    np = S["np"]
    with _PROCESSOR_LOCK:
        with S["torch"].inference_mode():
            raw = S["model"].encode_text(list(req["texts"])).float()
    normalized = S["F"].normalize(raw, p=2, dim=-1).to("cpu").numpy().astype(np.float32)
    np.savez(req["out_npz"], embeddings=normalized)
    return {"rows": int(normalized.shape[0]), "dim": int(normalized.shape[1]), "out_npz": req["out_npz"]}


def cmd_probe_video(req):
    total_frames, fps = _video_geometry(req["video_path"])
    response = {
        "total_frames": total_frames,
        "fps": fps,
        "duration_sec": (total_frames - 1) / float(fps) if fps > 0 else 0.0,
    }
    response.update(active_source(req["video_path"]).stats())
    return response


def cmd_release_video(req):
    reset_reader()
    if S["torch"] is not None:
        S["torch"].cuda.empty_cache()
    return {"released": True}


def cmd_ping(req):
    return {"pong": True, "pid": os.getpid()}


def cmd_shutdown(req):
    return {"bye": True}


HANDLERS = {
    "ping": cmd_ping,
    "init": cmd_init,
    "benchmark": cmd_benchmark,
    "audit": cmd_audit,
    "encode_video": cmd_encode_video,
    "encode_text": cmd_encode_text,
    "probe_video": cmd_probe_video,
    "release_video": cmd_release_video,
    "shutdown": cmd_shutdown,
}


def main():
    emit({"event": "ready", "pid": os.getpid()})
    # readline() thay vi `for line in sys.stdin`: iterator cua file object doc
    # truoc (read-ahead) nen co the giu lai command da nhan va gay deadlock.
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        request_id = request.get("id")
        command = request.get("cmd")
        handler = HANDLERS.get(command)
        if handler is None:
            emit({"id": request_id, "ok": False, "error": "Unknown command: " + str(command)})
            continue
        try:
            response = handler(request)
            response["ok"] = True
        except BaseException as exc:
            response = {
                "ok": False,
                "error": type(exc).__name__ + ": " + str(exc),
                "traceback": traceback.format_exc(),
            }
        response["id"] = request_id
        emit(response)
        if command == "shutdown":
            break


if __name__ == "__main__":
    main()
'''


from collections import Counter

WORKER_PATH = SCRATCH_ROOT / "tara_encode_worker.py"
WORKER_PATH.write_text(WORKER_SOURCE, encoding="utf-8")
WORKER_SOURCE_SHA256 = sha256_bytes(WORKER_SOURCE.encode("utf-8"))
print("Worker:", WORKER_PATH, "sha256:", WORKER_SOURCE_SHA256)

WORKER_PROTO = "@@TARA-RPC@@"
# Log FFmpeg bên trong decord có dạng "[h264 @ 0x55…] …" (SEI truncated, mmco…). Chúng là thông
# báo của decoder, không phải lỗi frame: frame hỏng thật đi qua đường substitution/reopen của
# worker và được in bằng "[worker] …". Camera giao thông sinh hàng chục nghìn dòng như vậy nên
# notebook chỉ ghi chúng vào log file của worker và in tóm tắt định kỳ.
FFMPEG_LOG_LINE = re.compile(r"^\[[A-Za-z0-9_]+ @ 0x[0-9A-Fa-f]+\] ")
FFMPEG_SUMMARY_SECONDS = 600


class WorkerDied(RuntimeError):
    pass


def gpu_memory_used_mib():
    """VRAM đang dùng trên GPU 0 theo nvidia-smi (gồm mọi process, kể cả process vừa chết)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(out.strip().splitlines()[0])


def host_memory_text():
    vm = psutil.virtual_memory()
    return f"RAM host còn {vm.available / 2 ** 30:.1f}/{vm.total / 2 ** 30:.1f} GiB"


def describe_worker_death(worker):
    rc = worker.proc.poll()
    text = (f"rc={rc}; {host_memory_text()}; VRAM đang dùng {gpu_memory_used_mib() / 1024:.1f}/"
            f"{GPU_VRAM_GIB:.0f} GiB")
    if rc == -9:
        text += (". SIGKILL: notebook không tự kill worker ở đây, nên thường là Linux OOM killer do "
                 "hết RAM host (xem RAM trong các dòng 'chờ cmd' phía trên)")
    return text


def wait_gpu_release(dead_slot, timeout_s=180):
    """Chờ driver thu hồi VRAM của worker vừa chết trước khi nạp worker mới.

    Process bị SIGKILL có thể mất vài giây tới vài phút mới trả VRAM. Nạp model mới ngay lúc đó
    khiến batch đã validate bị OOM giả.
    """
    others_alive = any(w.proc.poll() is None for s, w in WORKERS.items() if s != dead_slot)
    readings = [gpu_memory_used_mib()]
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not others_alive and readings[-1] <= GPU_BASELINE_USED_MIB + 1024:
            break
        if others_alive and len(readings) >= 4 and max(readings[-4:]) - min(readings[-4:]) < 256:
            break
        time.sleep(3)
        readings.append(gpu_memory_used_mib())
    print(f"[tara{dead_slot}] VRAM đang dùng {readings[0] / 1024:.1f} → {readings[-1] / 1024:.1f} GiB "
          f"trước khi nạp lại worker (baseline {GPU_BASELINE_USED_MIB / 1024:.1f} GiB).", flush=True)


class TaraWorker:
    """Client RPC cho worker chạy trong venv Python 3.10."""

    def __init__(self, python_bin, script_path, log_path, env=None, label="tara"):
        self.label = label
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8", buffering=1)
        worker_env = os.environ.copy()
        worker_env.update({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
        if env:
            worker_env.update(env)
        self.proc = subprocess.Popen(
            [str(python_bin), "-u", str(script_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1, env=worker_env,
        )
        self._responses = {}
        self._condition = threading.Condition()
        self._closed = False
        self._counter = 0
        self._ready = threading.Event()
        self._tail = deque(maxlen=120)
        self._progress_state = {"last_print": 0.0}
        self._ffmpeg_counts = Counter()
        self._ffmpeg_last_report = time.monotonic()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=120):
            raise WorkerDied("Worker không gửi tín hiệu ready trong 120 s.")
        print(f"[{self.label}] worker pid={self.proc.pid} log={self.log_path}")

    def _read_loop(self):
        try:
            for line in self.proc.stdout:
                line = line.rstrip("\r\n")
                self._log.write(line + "\n")
                if line.startswith(WORKER_PROTO):
                    try:
                        payload = json.loads(line[len(WORKER_PROTO):])
                    except json.JSONDecodeError:
                        self._tail.append(line)
                        continue
                    event = payload.get("event")
                    if event == "ready":
                        self._ready.set()
                    elif event == "progress":
                        self._on_progress(payload)
                    else:
                        with self._condition:
                            self._responses[payload.get("id")] = payload
                            self._condition.notify_all()
                else:
                    self._tail.append(line)
                    if FFMPEG_LOG_LINE.match(line):
                        self._note_ffmpeg(line)
                    else:
                        print(f"[{self.label}] {line}", flush=True)
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()

    def _note_ffmpeg(self, line):
        # Gộp theo mẫu (bỏ địa chỉ/số) để tóm tắt gọn, vd "[h264 @ #] SEI type # size # truncated at #".
        self._ffmpeg_counts[re.sub(r"0x[0-9A-Fa-f]+|\b\d+\b", "#", line)] += 1
        now = time.monotonic()
        if now - self._ffmpeg_last_report >= FFMPEG_SUMMARY_SECONDS:
            self._ffmpeg_last_report = now
            total = sum(self._ffmpeg_counts.values())
            top = "; ".join(f"{pattern} ×{count:,}" for pattern, count in self._ffmpeg_counts.most_common(3))
            print(f"[{self.label}] ffmpeg: đã ẩn {total:,} dòng log decoder trong "
                  f"{FFMPEG_SUMMARY_SECONDS // 60} phút (không phải lỗi; đầy đủ trong {self.log_path}). "
                  f"Nhiều nhất: {top}", flush=True)
            self._ffmpeg_counts.clear()

    def _on_progress(self, payload):
        now = time.monotonic()
        if now - self._progress_state["last_print"] < 10.0:
            return
        self._progress_state["last_print"] = now
        print(
            f"[{self.label}] {payload.get('label')}: {payload.get('done')}/{payload.get('total')} clip"
            f" | {payload.get('clips_per_s', 0.0):.2f} clip/s",
            flush=True,
        )

    def call(self, cmd, timeout=None, heartbeat=60.0, **kwargs):
        with self._condition:
            if self._closed or self.proc.poll() is not None:
                raise WorkerDied(
                    f"Worker đã chết trước khi nhận cmd={cmd} (rc={self.proc.poll()}).\n"
                    "--- last output ---\n" + "\n".join(list(self._tail)[-20:])
                )
            self._counter += 1
            request_id = self._counter
        payload = {"id": request_id, "cmd": cmd}
        payload.update(kwargs)
        try:
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise WorkerDied("stdin của worker đã đóng.") from exc

        started = time.monotonic()
        next_beat = started + heartbeat
        with self._condition:
            while request_id not in self._responses:
                if self._closed and request_id not in self._responses:
                    raise WorkerDied(
                        f"Worker chết khi đang chạy cmd={cmd} (rc={self.proc.poll()}).\n"
                        "--- last output ---\n" + "\n".join(self._tail)
                    )
                self._condition.wait(timeout=2.0)
                now = time.monotonic()
                if timeout is not None and now - started > timeout:
                    raise TimeoutError(f"cmd={cmd} quá {timeout}s.")
                if now >= next_beat:
                    print(f"[{self.label}] chờ cmd={cmd} … {_elapsed_text(now - started)} | "
                          f"{host_memory_text()}", flush=True)
                    next_beat = now + heartbeat
            response = self._responses.pop(request_id)
        if not response.get("ok"):
            raise RuntimeError(
                f"Worker lỗi ở cmd={cmd}: {response.get('error')}\n"
                + str(response.get("traceback", ""))
            )
        return response

    def close(self):
        if self.proc.poll() is None:
            try:
                self.call("shutdown", timeout=60)
            except Exception:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()


def start_worker(slot: int = 0):
    log_path = LOG_ROOT / f"worker{slot}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.log"
    # expandable_segments giảm phân mảnh VRAM khi nhiều worker chia một GPU; không đổi kết quả tính.
    worker = TaraWorker(VENV_PYTHON, WORKER_PATH, log_path, label=f"tara{slot}",
                        env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    try:
        info = worker.call(
            "init", timeout=1800, heartbeat=30,
            model_dir=str(MODEL_SNAPSHOT_DIR),
            embedding_dim=CONFIG["embedding_dim"],
            max_pixels=CONFIG["max_pixels"],
            n_frames=CONFIG["n_frames"],
            attn_implementation=CONFIG["attention_implementation"],
            bypass_lm_head=CONFIG["bypass_lm_head"],
            prep_threads=CONFIG["prep_threads"],
            prefetch_batches=CONFIG["prefetch_batches"],
            decord_threads=CONFIG["decord_threads"],
            decord_eof_retry_max=CONFIG["decord_eof_retry_max"],
            substitution_window=CONFIG["decode_substitution_window"],
            probe_max_steps=CONFIG["decode_probe_max_steps"],
            expected_default_config=CONFIG["expected_default_config"],
        )
        checks = info["checks"]
        if checks["video_eol_prompt"] != CONFIG["video_eol_prompt"]:
            raise RuntimeError(f"Video EOL prompt mismatch: {checks['video_eol_prompt']!r}")
        if checks["text_eol_prompt"] != CONFIG["text_eol_prompt"]:
            raise RuntimeError(f"Text EOL prompt mismatch: {checks['text_eol_prompt']!r}")
        if checks["attn_implementation"] != CONFIG["attention_implementation"]:
            raise RuntimeError(f"Attention backend mismatch: {checks['attn_implementation']}")
        if checks["image_mean"] != SEMANTIC_CONFIG["preprocess"]["image_mean"]:
            raise RuntimeError(f"image_mean mismatch: {checks['image_mean']}")
        if checks["image_std"] != SEMANTIC_CONFIG["preprocess"]["image_std"]:
            raise RuntimeError(f"image_std mismatch: {checks['image_std']}")
        if not checks["video_sampling_strategy"].get("use_multi_images_for_video"):
            raise RuntimeError("use_multi_images_for_video phải bật (F=16 image slot).")
        if checks["video_sampling_strategy"].get("video_sampler_version") != "v1":
            raise RuntimeError("Sampler phải là v1.")
        if round(info["n_params"] / 1e9, 3) != 8.291:
            raise RuntimeError(f"Số tham số lạ: {info['n_params']}")
    except BaseException:
        # Worker init hỏng (vd. OOM) vẫn là process sống giữ VRAM: phải đóng trước khi raise.
        worker.close()
        raise
    return worker, info


# Chạy lại cell này (vd Run All sau lỗi) không được để worker của lần trước sống giữ VRAM.
for _old_worker in list(globals().get("WORKERS", {}).values()):
    try:
        _old_worker.close()
    except Exception:
        pass

# Mỗi worker là một process riêng (model riêng, Jinja riêng) trên cùng GPU.
WORKERS: Dict[int, TaraWorker] = {}
WORKER_ABORT = threading.Event()
# VRAM đã dùng trước khi nạp worker nào; dùng để biết VRAM của worker chết đã được trả hết chưa.
GPU_BASELINE_USED_MIB = gpu_memory_used_mib()
if GPU_BASELINE_USED_MIB > 2048:
    print(f"CẢNH BÁO: GPU đã dùng {GPU_BASELINE_USED_MIB / 1024:.1f} GiB trước khi nạp TARA; "
          "có process khác giữ VRAM (chạy `!nvidia-smi` để xem).", flush=True)


def call_worker(slot, cmd, _restarts=1, **kwargs):
    """Gọi worker `slot`, tự restart một lần nếu process chết giữa chừng.

    Một job nhiều chục giờ không nên mất sạch vì một lần CUDA error. Restart chỉ nạp lại
    model — batch size và semantic contract giữ nguyên, và mọi shard đã commit vẫn hợp lệ
    nên phần việc mất đi nhiều nhất là một chunk.
    """
    try:
        return WORKERS[slot].call(cmd, **kwargs)
    except WorkerDied as exc:
        if _restarts <= 0 or WORKER_ABORT.is_set():
            raise
        print(f"[tara{slot}] worker chết ({exc}); restart và thử lại cmd={cmd}.", flush=True)
        print(f"[tara{slot}] {describe_worker_death(WORKERS[slot])}", flush=True)
        try:
            WORKERS[slot].close()
        except Exception:
            pass
        wait_gpu_release(slot)
        WORKERS[slot], restarted_info = start_worker(slot)
        if restarted_info["checks"]["video_eol_prompt"] != CONFIG["video_eol_prompt"]:
            raise RuntimeError("Worker restart nhưng contract đã đổi.") from exc
        return call_worker(slot, cmd, _restarts=_restarts - 1, **kwargs)


def close_workers(slots=None):
    for slot in sorted(slots if slots is not None else list(WORKERS)):
        worker = WORKERS.pop(slot, None)
        if worker is not None:
            worker.close()


def abort_workers():
    """Dừng/lỗi giữa chunk: kill process để thread chờ RPC thoát ngay.

    Chunk đang dở chưa được commit nên không mất dữ liệu; lần chạy sau call_worker tự
    nạp lại worker.
    """
    WORKER_ABORT.set()
    for worker in list(WORKERS.values()):
        if worker.proc.poll() is None:
            worker.proc.kill()


WORKERS[0], WORKER_INFO = start_worker(0)
print(json.dumps({k: v for k, v in WORKER_INFO.items() if k != "checks"}, indent=2))
print(json.dumps(WORKER_INFO["checks"], indent=2, ensure_ascii=False))

# %% [markdown]
# ## 8. Resume state, autotune batch, số worker và quality gate
#
# Thứ tự bắt buộc: đọc commit hợp lệ → chọn category chưa xong → tải một video
# thật → benchmark batch trên clip thật → đo 1 vs 2 worker → audit đối chiếu với đường
# chính chủ → mới cho phép ghi vector.

# %%
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def np_linspace(start, stop, count):
    if count <= 1:
        return [float(start)]
    step = (float(stop) - float(start)) / (count - 1)
    return [float(start) + step * i for i in range(count)]


def shard_bounds(total_rows: int, shard_id: int):
    start = shard_id * CONFIG["shard_rows"]
    return start, min(total_rows, start + CONFIG["shard_rows"])


def total_shards(total_rows: int):
    return math.ceil(total_rows / CONFIG["shard_rows"])


def remote_commit_prefix(category: str):
    return f"{CONFIG['output_prefix'].strip('/')}/commits/{category}"


def remote_parquet_path(category: str, shard_id: int):
    return f"{CONFIG['output_prefix'].strip('/')}/embeddings/{category}/part-{shard_id:05d}.parquet"


def remote_commit_path(category: str, shard_id: int):
    return f"{remote_commit_prefix(category)}/part-{shard_id:05d}.json"


def load_valid_remote_commits(category: str, clips: Sequence[ClipRecord]):
    if not CONFIG["upload"]:
        return {}
    prefix = remote_commit_prefix(category)
    commit_items = [
        item for item in list_bucket_tree(
            CONFIG["output_bucket_id"], prefix=prefix, recursive=True, token=HF_TOKEN
        )
        if getattr(item, "type", "file") == "file" and item.path.endswith(".json")
    ]
    if not commit_items:
        return {}

    local_dir = STATE_ROOT / "remote_commits" / category
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    download_bucket_files(
        CONFIG["output_bucket_id"],
        files=[(item, str(local_dir / Path(item.path).name)) for item in commit_items],
        token=HF_TOKEN,
    )
    parquet_items = {
        item.path: item for item in list_bucket_tree(
            CONFIG["output_bucket_id"],
            prefix=f"{CONFIG['output_prefix'].strip('/')}/embeddings/{category}",
            recursive=True, token=HF_TOKEN,
        )
        if getattr(item, "type", "file") == "file"
    }
    valid = {}
    expected_total = total_shards(len(clips))
    for local_commit in local_dir.glob("*.json"):
        commit = json.loads(local_commit.read_text(encoding="utf-8"))
        if commit.get("semantic_fingerprint") != SEMANTIC_FINGERPRINT:
            raise RuntimeError(f"Commit fingerprint mismatch: {local_commit.name}")
        shard_id = int(commit["shard_id"])
        if not 0 <= shard_id < expected_total:
            raise RuntimeError(f"Commit shard_id ngoài range: {local_commit.name}")
        start, stop = shard_bounds(len(clips), shard_id)
        expected = {
            "category": category,
            "row_start": start,
            "row_stop": stop,
            "rows": stop - start,
            "first_clip_id": clips[start].clip_id,
            "last_clip_id": clips[stop - 1].clip_id,
            "parquet_path": remote_parquet_path(category, shard_id),
        }
        for key, value in expected.items():
            if commit.get(key) != value:
                raise RuntimeError(f"Commit metadata mismatch {local_commit.name}: {key}")
        # Vector phải đến từ đúng bytes video hiện có trên R2 (quan trọng với S01 H.264).
        for video_id, etag in commit.get("video_etags", {}).items():
            if VIDEO_OBJECTS.get(video_id, {}).get("etag") != etag:
                raise RuntimeError(
                    f"{local_commit.name}: {video_id} được encode từ ETag {etag} nhưng R2 hiện là "
                    f"{VIDEO_OBJECTS.get(video_id, {}).get('etag')}; video nguồn đã đổi."
                )
        info = parquet_items.get(commit["parquet_path"])
        if info is None or int(info.size) != int(commit["parquet_bytes"]):
            raise RuntimeError(f"Commit trỏ tới Parquet thiếu/sai size: {commit['parquet_path']}")
        if shard_id in valid:
            raise RuntimeError(f"Duplicate commit shard_id={shard_id}")
        valid[shard_id] = commit
    return valid


def local_video_path(record: VideoRecord):
    return VIDEO_ROOT / record.category / f"{record.video_id}.mp4"


def fetch_video(record: VideoRecord):
    """Tải video từ R2 (ranged GET song song), kiểm size + ETag multipart + codec."""
    obj = VIDEO_OBJECTS[record.video_id]
    if obj["kind"] == "av1_pending":
        raise RuntimeError(
            f"{record.video_id} trên R2 vẫn là AV1 (decord không đọc được). Chạy "
            "AIC2026_S01_AV1_to_H264_R2_Colab.ipynb rồi Run All lại."
        )
    destination = local_video_path(record)
    if destination.is_file() and destination.stat().st_size == obj["size"]:
        # Size khớp là chưa đủ: một runtime trước có thể để lại file hỏng mà vẫn đúng size.
        if not CONFIG["verify_video_etag"] or multipart_etag(destination) == obj["etag"]:
            return destination
        print(f"[cache] {record.video_id}: ETag không khớp; xóa và tải lại.", flush=True)
        destination.unlink()
    free = shutil.disk_usage(SCRATCH_ROOT).free
    if free < obj["size"] + (4 << 30):
        raise RuntimeError(
            f"Không đủ disk cho {record.video_id}: cần {(obj['size'] + (4 << 30)) / 2 ** 30:.1f} GiB, "
            f"còn {free / 2 ** 30:.1f} GiB."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    for attempt in range(1, int(CONFIG["r2_download_attempts"]) + 1):
        partial.unlink(missing_ok=True)
        try:
            S3.download_file(CONFIG["source_r2_bucket"], record.remote_path, str(partial),
                             Config=R2_TRANSFER)
        except Exception as exc:
            print(f"[r2] {record.video_id}: tải lần {attempt} lỗi {type(exc).__name__}: {exc}", flush=True)
            continue
        size_ok = partial.stat().st_size == obj["size"]
        if size_ok and (not CONFIG["verify_video_etag"] or multipart_etag(partial) == obj["etag"]):
            os.replace(partial, destination)
            break
        print(f"[r2] {record.video_id}: tải lần {attempt} lệch size/ETag; tải lại.", flush=True)
    else:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{record.video_id}: không tải được bản khớp ETag {obj['etag']}.")
    codec = probe_video_codec(destination)
    if codec not in {"h264", "hevc"}:
        raise RuntimeError(f"{record.video_id}: codec {codec} không đọc được bằng decord 0.6.0.")
    return destination


def drop_video(record: VideoRecord):
    if not CONFIG["cleanup_video_after_encode"]:
        return
    path = local_video_path(record)
    expected_root = VIDEO_ROOT.resolve()
    if path.resolve().parent.parent != expected_root:
        raise RuntimeError(f"Từ chối xóa ngoài scratch video root: {path}")
    path.unlink(missing_ok=True)


def clip_payload(clip: ClipRecord, row: int = 0):
    return {"row": row, "start_time": clip.start_time, "end_time": clip.end_time}


def timed_parallel_encode(slots, clip_lists, batch_size, video_path):
    """Mỗi slot encode một danh sách clip đồng thời; trả về (clip/s tổng, embeddings theo slot)."""
    def run(slot, clips):
        out_path = NPZ_ROOT / f"scaling-w{slot}.npz"
        call_worker(slot, "encode_video", timeout=None, heartbeat=120,
                    video_path=str(video_path), video_id=f"scaling-w{slot}", clips=clips,
                    batch_size=batch_size, embedding_dim=CONFIG["embedding_dim"],
                    out_npz=str(out_path))
        with np.load(out_path) as bundle:
            vectors = bundle["embeddings"].astype(np.float32)
        out_path.unlink(missing_ok=True)
        return vectors

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = {slot: pool.submit(run, slot, clips) for slot, clips in zip(slots, clip_lists)}
        vectors = {slot: future.result() for slot, future in futures.items()}
    elapsed = time.monotonic() - started
    return sum(len(c) for c in clip_lists) / elapsed, vectors


# --- chọn category đầu tiên chưa hoàn tất để calibrate trên dữ liệu thật ---
FIRST_CATEGORY = None
FIRST_CLIPS = None
FIRST_COMMITS = None
for candidate in CONFIG["categories"]:
    if candidate in CONFIG["delegated_categories"]:
        continue
    candidate_clips = CLIP_PLAN[candidate]
    commits = load_valid_remote_commits(candidate, candidate_clips)
    if len(commits) < total_shards(len(candidate_clips)):
        FIRST_CATEGORY, FIRST_CLIPS, FIRST_COMMITS = candidate, candidate_clips, commits
        break

BATCH_SIZE = int(CONFIG["batch_size"] or 1)
BATCH_BENCHMARK = []
AUDIT = {}
WORKER_SCALING = {}
GPU_WORKERS = 1

if FIRST_CLIPS is None:
    print("Mọi category của session này đã hoàn tất; bỏ qua autotune và audit.")
else:
    calib_record = VIDEO_REGISTRY[FIRST_CATEGORY][0]
    calib_path = str(fetch_video(calib_record))
    calib_clips = [c for c in FIRST_CLIPS if c.video_id == calib_record.video_id]
    print(f"Calibrate trên {calib_record.video_id} "
          f"({calib_record.duration_sec:.1f}s, {len(calib_clips)} clip)")

    # --- autotune batch trên worker 0 ---
    candidates = ([int(CONFIG["batch_size"])] if CONFIG["batch_size"] is not None
                  else [int(x) for x in CONFIG["batch_candidates"]])
    candidates = sorted({c for c in candidates if c > 0})

    def run_benchmark(batch_candidates):
        bench_clips = [clip_payload(c) for c in calib_clips[:max(batch_candidates) * CONFIG["benchmark_rounds"]]]
        return call_worker(
            0, "benchmark", timeout=3600, heartbeat=30,
            video_path=calib_path, clips=bench_clips,
            batch_candidates=batch_candidates, rounds=CONFIG["benchmark_rounds"],
        )["results"]

    BATCH_BENCHMARK = run_benchmark(candidates)
    if not BATCH_BENCHMARK and BATCH_SIZE_SOURCE == "profile":
        # Batch mặc định của profile không vừa VRAM của runtime này (vd A100 40 GB, hoặc VRAM
        # chưa trả hết sau khi worker cũ chết): autotune lại với các batch nhỏ hơn.
        smaller = sorted({int(x) for x in CONFIG["batch_candidates"] if 0 < int(x) < min(candidates)})
        print(f"Batch {candidates} của profile {GPU_PROFILE_NAME} OOM trên {GPU_NAME} "
              f"({GPU_VRAM_GIB:.0f} GiB, đang dùng {gpu_memory_used_mib() / 1024:.1f} GiB); "
              f"autotune lại với {smaller}.", flush=True)
        if smaller:
            BATCH_BENCHMARK = run_benchmark(smaller)
    if not BATCH_BENCHMARK:
        raise RuntimeError(
            f"Benchmark không trả về kết quả nào: OOM ngay batch nhỏ nhất đã thử trên {GPU_NAME} "
            f"({GPU_VRAM_GIB:.0f} GiB, đang dùng {gpu_memory_used_mib() / 1024:.1f} GiB). "
            "Kiểm tra còn process nào khác giữ VRAM không (nvidia-smi)."
        )
    for row in BATCH_BENCHMARK:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    safe = [r for r in BATCH_BENCHMARK if r["peak_fraction"] <= CONFIG["max_vram_fraction"]]
    best = max(safe or BATCH_BENCHMARK, key=lambda r: r["clips_per_s"])
    if BATCH_SIZE_SOURCE == "manual" and best["batch_size"] != int(CONFIG["batch_size"]):
        raise RuntimeError("Manual batch không vượt qua validation VRAM.")
    BATCH_SIZE = int(best["batch_size"])
    print("Batch 1 worker:", BATCH_SIZE, f"({best['clips_per_s']:.2f} clip/s)")

    # --- đo 1 vs nhiều worker trên cùng GPU ---
    requested_workers = int(CONFIG["gpu_workers"])
    per_worker_budget = CONFIG["max_vram_fraction"] / max(1, requested_workers)
    multi_safe = [r for r in BATCH_BENCHMARK if r["peak_fraction"] <= per_worker_budget]
    scaling_clips = min(int(CONFIG["worker_scaling_clips"]), len(calib_clips) // max(1, requested_workers))
    if requested_workers > 1 and multi_safe and scaling_clips >= 4:
        multi_batch = int(max(multi_safe, key=lambda r: r["clips_per_s"])["batch_size"])
        pool_clips = [clip_payload(c, row=i) for i, c in enumerate(calib_clips[:scaling_clips * requested_workers])]
        single_rate, single_vectors = timed_parallel_encode([0], [pool_clips], BATCH_SIZE, calib_path)
        # Batch 1-worker có thể giữ ~90% VRAM trong cache của worker 0: trả lại trước khi nạp
        # worker phụ (release_video = đóng reader + torch.cuda.empty_cache()).
        call_worker(0, "release_video", timeout=300)
        try:
            for slot in range(1, requested_workers):
                if slot not in WORKERS:
                    WORKERS[slot], _ = start_worker(slot)
            extra_workers_ready = True
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            close_workers([slot for slot in list(WORKERS) if slot != 0])
            extra_workers_ready = False
            print("Không đủ VRAM để nạp worker phụ; dùng 1 worker.", flush=True)
        if extra_workers_ready:
            slots = list(range(requested_workers))
            split = [pool_clips[i * scaling_clips:(i + 1) * scaling_clips] for i in slots]
            multi_rate, multi_vectors = timed_parallel_encode(slots, split, multi_batch, calib_path)
            # Worker phụ phải cho cùng vector với worker 0 trên cùng clip.
            cross_cosine = [
                float(np.min(np.sum(
                    multi_vectors[slot] * single_vectors[0][slot * scaling_clips:(slot + 1) * scaling_clips],
                    axis=1,
                )))
                for slot in slots
            ]
            WORKER_SCALING = {
                "requested_workers": requested_workers,
                "per_worker_vram_budget": per_worker_budget,
                "single_batch": BATCH_SIZE,
                "single_clips_per_s": single_rate,
                "multi_batch": multi_batch,
                "multi_clips_per_s": multi_rate,
                "speedup": multi_rate / single_rate,
                "cross_worker_min_cosine": min(cross_cosine),
            }
            print(json.dumps(WORKER_SCALING, indent=2))
            if min(cross_cosine) < CONFIG["audit_min_cosine_batch"]:
                raise RuntimeError(f"Worker phụ cho vector lệch worker 0: {cross_cosine}")
            if multi_rate >= single_rate * float(CONFIG["gpu_worker_min_speedup"]):
                GPU_WORKERS = requested_workers
                BATCH_SIZE = multi_batch
            else:
                close_workers([slot for slot in list(WORKERS) if slot != 0])
    elif requested_workers > 1:
        print(f"Không batch nào vừa {per_worker_budget:.0%} VRAM mỗi worker; dùng 1 worker.")
    print(f"Chọn {GPU_WORKERS} worker × batch {BATCH_SIZE}.")

    # --- quality gate (worker 0, batch đã chọn) ---
    sample_count = min(int(CONFIG["audit_clip_samples"]), len(calib_clips))
    audit_indices = sorted({int(round(i)) for i in
                            np_linspace(0, len(calib_clips) - 1, sample_count)})
    audit_clips = [clip_payload(calib_clips[i]) for i in audit_indices]
    AUDIT = call_worker(
        0, "audit", timeout=3600, heartbeat=30,
        video_path=calib_path, clips=audit_clips, batch_size=BATCH_SIZE,
    )
    print(json.dumps({k: v for k, v in AUDIT.items()
                      if k not in ("frame_index_mismatches", "id", "ok")}, indent=2))
    if AUDIT["frame_index_mismatches"]:
        print(json.dumps(AUDIT["frame_index_mismatches"][:3], indent=2))
        if CONFIG["require_frame_index_exact_match"]:
            raise RuntimeError(
                "frame_indices tự tính KHÔNG trùng sample_video chính chủ; "
                "từ chối encode vì clip sẽ lệch khỏi contract."
            )
    if AUDIT["full_video_min_cosine"] < CONFIG["audit_full_video_min_cosine"]:
        raise RuntimeError(f"Full-video parity fail: {AUDIT['full_video_min_cosine']}")
    if AUDIT["clip_vs_official_min_cosine"] < CONFIG["audit_min_cosine_vs_official"]:
        raise RuntimeError(f"Clip parity fail (min): {AUDIT['clip_vs_official_min_cosine']}")
    if AUDIT["clip_vs_official_mean_cosine"] < CONFIG["audit_mean_cosine_vs_official"]:
        raise RuntimeError(f"Clip parity fail (mean): {AUDIT['clip_vs_official_mean_cosine']}")
    if AUDIT["batch_vs_single_min_cosine"] < CONFIG["audit_min_cosine_batch"]:
        raise RuntimeError(f"Batch parity fail: {AUDIT['batch_vs_single_min_cosine']}")
    if abs(AUDIT["norm_min"] - 1.0) > 2e-5 or abs(AUDIT["norm_max"] - 1.0) > 2e-5:
        raise RuntimeError(f"Vector không unit-norm: {AUDIT['norm_min']}, {AUDIT['norm_max']}")
    print("Quality gate PASS: frame index khớp tuyệt đối, embedding khớp đường chính chủ.")

    for slot in list(WORKERS):
        call_worker(slot, "release_video", timeout=300)
    measured = (WORKER_SCALING.get("multi_clips_per_s") if GPU_WORKERS > 1
                else WORKER_SCALING.get("single_clips_per_s", best["clips_per_s"]))
    remaining = 0
    for category in CONFIG["categories"]:
        if category in CONFIG["delegated_categories"]:
            continue
        clips_here = CLIP_PLAN[category]
        done = FIRST_COMMITS if category == FIRST_CATEGORY else load_valid_remote_commits(
            category, clips_here
        )
        remaining += len(clips_here) - sum(int(c["rows"]) for c in done.values())
    print(json.dumps({
        "gpu_workers": GPU_WORKERS,
        "batch_size": BATCH_SIZE,
        "measured_clips_per_s": round(measured, 3),
        "session_clips_total": session_clips,
        "session_clips_remaining": remaining,
        "session_hours_remaining_estimate": round(remaining / measured / 3600, 2),
        "note": "Ước tính từ clip của một video; throughput thật đổi theo độ phân giải video.",
    }, indent=2))

EXECUTION_INFO = {
    "started_at": utc_now(),
    "session_name": CONFIG["session_name"],
    "host": socket.gethostname(),
    "gpu": GPU_NAME,
    "gpu_vram_gib": GPU_VRAM_GIB,
    "system_ram_gib": SYSTEM_RAM_GIB,
    "cpu_count": os.cpu_count(),
    "gpu_profile": GPU_PROFILE_NAME,
    "batch_size_source": BATCH_SIZE_SOURCE,
    "batch_size": BATCH_SIZE,
    "gpu_workers": GPU_WORKERS,
    "batch_benchmark": BATCH_BENCHMARK,
    "worker_scaling": WORKER_SCALING,
    "audit": AUDIT,
    "bypass_lm_head": CONFIG["bypass_lm_head"],
    "prep_threads": CONFIG["prep_threads"],
    "prefetch_batches": CONFIG["prefetch_batches"],
    "decord_threads": CONFIG["decord_threads"],
    "decord_eof_retry_max": CONFIG["decord_eof_retry_max"],
    "decode_substitution_window": CONFIG["decode_substitution_window"],
    "encode_chunk_rows": CONFIG["encode_chunk_rows"],
    "worker_source_sha256": WORKER_SOURCE_SHA256,
    "venv": VENV_INFO,
    "model_runtime": {k: v for k, v in WORKER_INFO.items() if k != "checks"},
    "kernel_python": platform.python_version(),
}
if CONFIG["upload"]:
    upload_bytes_verified(
        CONFIG["output_bucket_id"],
        f"{CONFIG['output_prefix'].strip('/')}/executions/"
        f"{CONFIG['session_name']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
        json.dumps(EXECUTION_INFO, ensure_ascii=False, indent=2).encode("utf-8"),
    )

# %% [markdown]
# ## 9. Ghi Parquet shard và encode E2E
#
# Đơn vị việc là **chunk** = giao của một video với một shard (≤ `encode_chunk_rows` clip).
# Các worker lấy chunk theo đúng thứ tự plan; video được tải trước trên thread nền (giữ tối
# đa `số worker + video_prefetch_extra` video trên disk) và xóa ngay khi mọi chunk của nó
# xong. Main thread gom kết quả vào buffer theo shard; shard nào đủ row thì ghi Parquet →
# upload → verify size → **rồi mới** ghi commit JSON. Commit là transaction boundary duy nhất.

# %%
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import field


PARQUET_SCHEMA_METADATA = {
    b"schema_version": b"1",
    b"semantic_fingerprint": SEMANTIC_FINGERPRINT.encode(),
    b"model_id": CONFIG["model_id"].encode(),
    b"model_revision": CONFIG["model_revision"].encode(),
    b"checkpoint_index_sha256": MODEL_INDEX_SHA256.encode(),
    b"video_eol_prompt": CONFIG["video_eol_prompt"].encode(),
    b"n_frames": str(CONFIG["n_frames"]).encode(),
    b"max_pixels": str(CONFIG["max_pixels"]).encode(),
    b"pooling": b"last_token_final_hidden_state",
    b"embedding_dim": str(CONFIG["embedding_dim"]).encode(),
    b"embedding_dtype": CONFIG["storage_dtype"].encode(),
    b"l2_normalized": b"true",
    b"scales_by_group": json.dumps(CONFIG["scales_by_group"]).encode(),
    b"source_r2_bucket": CONFIG["source_r2_bucket"].encode(),
    b"video_manifest_sha256": BATCH2_VIDEO_MANIFEST_SHA256.encode(),
}


def write_parquet_shard(category, shard_id, clips, video_by_id, embeddings, frame_indices, row_indices):
    start, stop = shard_bounds(len(clips), shard_id)
    expected_rows = np.arange(start, stop, dtype=np.int64)
    order = np.argsort(row_indices, kind="stable")
    row_indices = row_indices[order]
    embeddings = embeddings[order]
    frame_indices = frame_indices[order]
    if not np.array_equal(row_indices, expected_rows):
        raise RuntimeError(f"{category} shard {shard_id}: row set không khớp shard bounds.")
    if embeddings.shape != (stop - start, CONFIG["embedding_dim"]):
        raise RuntimeError(f"Embedding shape sai: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"{category} shard {shard_id}: NaN/Inf embedding.")
    norms = np.linalg.norm(embeddings, axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if max_norm_error > 2e-5:
        raise RuntimeError(f"{category} shard {shard_id}: norm error {max_norm_error}")

    selected = clips[start:stop]
    stored = np.ascontiguousarray(embeddings.astype(np.float32, copy=False))
    embedding_array = pa.FixedSizeListArray.from_arrays(
        pa.array(stored.reshape(-1), type=pa.float32()), CONFIG["embedding_dim"]
    )
    frames_stored = np.ascontiguousarray(frame_indices.astype(np.int32, copy=False))
    frame_array = pa.FixedSizeListArray.from_arrays(
        pa.array(frames_stored.reshape(-1), type=pa.int32()), CONFIG["n_frames"]
    )

    table = pa.Table.from_arrays(
        [
            pa.array([c.clip_id for c in selected], type=pa.string()),
            pa.array([c.video_id for c in selected], type=pa.string()),
            pa.array([c.category for c in selected], type=pa.string()),
            pa.array([c.scale for c in selected], type=pa.string()),
            pa.array([c.scale_index for c in selected], type=pa.int8()),
            pa.array([c.start_time for c in selected], type=pa.float64()),
            pa.array([c.end_time for c in selected], type=pa.float64()),
            pa.array([video_by_id[c.video_id].fps for c in selected], type=pa.float32()),
            pa.array([video_by_id[c.video_id].duration_sec for c in selected], type=pa.float64()),
            pa.array([video_by_id[c.video_id].remote_path for c in selected], type=pa.string()),
            frame_array,
            embedding_array,
        ],
        names=[
            "clip_id", "video_id", "category", "scale", "scale_index",
            "start_time", "end_time", "fps", "duration_sec", "video_relpath",
            "frame_indices", "embedding",
        ],
    ).replace_schema_metadata({
        **PARQUET_SCHEMA_METADATA,
        # Scale thật của shard này (N: 4/8/16 s; M và S01: 8/24/72 s).
        b"scales": json.dumps(scales_for(category)).encode(),
    })

    local_dir = LOCAL_OUTPUT_ROOT / "embeddings" / category
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"part-{shard_id:05d}.parquet"
    partial_path = local_path.with_suffix(".parquet.partial")
    pq.write_table(
        table, partial_path, compression=None,
        use_dictionary=["video_id", "category", "scale", "video_relpath"],
        row_group_size=1_024,
        write_statistics=["clip_id", "video_id", "scale", "start_time"],
    )
    os.replace(partial_path, local_path)

    shard_videos = sorted({c.video_id for c in selected}, key=natural_video_key)
    commit = {
        "schema_version": 1,
        "created_at": utc_now(),
        "semantic_fingerprint": SEMANTIC_FINGERPRINT,
        "category": category,
        "shard_id": shard_id,
        "row_start": start,
        "row_stop": stop,
        "rows": stop - start,
        "first_clip_id": selected[0].clip_id,
        "last_clip_id": selected[-1].clip_id,
        "parquet_path": remote_parquet_path(category, shard_id),
        "parquet_bytes": local_path.stat().st_size,
        "parquet_sha256": sha256_file(local_path),
        "embedding_dim": CONFIG["embedding_dim"],
        "embedding_dtype": CONFIG["storage_dtype"],
        "l2_normalized": True,
        "norm_min_fp32": float(norms.min()),
        "norm_max_fp32": float(norms.max()),
        "norm_mean_fp32": float(norms.mean()),
        # Bytes video đã thực sự đưa vào model (S01 là bản H.264 của notebook S01).
        "video_etags": {video_id: VIDEO_OBJECTS[video_id]["etag"] for video_id in shard_videos},
        "execution": {
            "session_name": CONFIG["session_name"],
            "gpu": GPU_NAME,
            "batch_size": BATCH_SIZE,
            "gpu_workers": GPU_WORKERS,
            "bypass_lm_head": CONFIG["bypass_lm_head"],
            "attention_implementation": CONFIG["attention_implementation"],
            "torch": VENV_INFO["torch"],
            "transformers": VENV_INFO["transformers"],
        },
    }
    if CONFIG["upload"]:
        # Data first, commit last: commit là transaction boundary.
        upload_file_verified(CONFIG["output_bucket_id"], commit["parquet_path"], local_path)
        upload_bytes_verified(
            CONFIG["output_bucket_id"], remote_commit_path(category, shard_id),
            json.dumps(commit, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        if CONFIG["cleanup_local_shards_after_upload"]:
            local_path.unlink()
    else:
        commit_dir = LOCAL_OUTPUT_ROOT / "commits" / category
        commit_dir.mkdir(parents=True, exist_ok=True)
        (commit_dir / f"part-{shard_id:05d}.json").write_text(
            json.dumps(commit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return commit


def pending_rows(total_rows: int, completed_shards):
    ranges = []
    for shard_id in range(total_shards(total_rows)):
        if shard_id not in completed_shards:
            start, stop = shard_bounds(total_rows, shard_id)
            ranges.append(np.arange(start, stop, dtype=np.int64))
    return np.concatenate(ranges) if ranges else np.empty(0, dtype=np.int64)


@dataclass
class WorkItem:
    record: VideoRecord
    rows: List[int] = field(default_factory=list)


def build_work_items(clips, rows_todo):
    """Chunk = rows của một video trong cùng một shard, tối đa encode_chunk_rows, đúng thứ tự plan."""
    video_by_id = {r.video_id: r for records in VIDEO_REGISTRY.values() for r in records}
    items: List[WorkItem] = []
    current_key = None
    for row in (int(r) for r in rows_todo):
        key = (clips[row].video_id, row // CONFIG["shard_rows"])
        if key != current_key or len(items[-1].rows) >= int(CONFIG["encode_chunk_rows"]):
            items.append(WorkItem(record=video_by_id[clips[row].video_id]))
            current_key = key
        items[-1].rows.append(row)
    return items


class VideoPrefetcher:
    """Tải video theo thứ tự plan trên thread nền; giữ tối đa `depth` video trên disk.

    `depth` ≥ số worker + 1 nên video của chunk kế tiếp luôn có chỗ: mọi video trước nó
    hoặc đã xóa, hoặc đang được một worker encode.
    """

    def __init__(self, records, chunks_per_video, depth):
        self.records = list(records)
        self.remaining = dict(chunks_per_video)
        self.depth = max(1, int(depth))
        self.paths = {}
        self.errors = {}
        self.on_disk = set()
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        for record in self.records:
            with self.condition:
                while len(self.on_disk) >= self.depth and not self.stop_event.is_set():
                    self.condition.wait(timeout=1.0)
                if self.stop_event.is_set():
                    return
            try:
                path = fetch_video(record)
            except BaseException as exc:  # noqa: BLE001 - chuyển cho consumer
                with self.condition:
                    self.errors[record.video_id] = exc
                    self.condition.notify_all()
                return
            with self.condition:
                self.paths[record.video_id] = path
                self.on_disk.add(record.video_id)
                self.condition.notify_all()

    def wait(self, video_id):
        with self.condition:
            while video_id not in self.paths:
                if self.errors:
                    failed_id, exc = next(iter(self.errors.items()))
                    raise RuntimeError(f"Tải {failed_id} thất bại") from exc
                if self.stop_event.is_set():
                    raise RuntimeError("Prefetcher đã dừng.")
                self.condition.wait(timeout=1.0)
            return self.paths[video_id]

    def chunk_done(self, record):
        with self.condition:
            self.remaining[record.video_id] -= 1
            finished = self.remaining[record.video_id] == 0
        if finished:
            drop_video(record)
            with self.condition:
                self.on_disk.discard(record.video_id)
                self.paths.pop(record.video_id, None)
                self.condition.notify_all()

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()


def encode_item(slot, category, clips, item, video_path):
    """Encode một chunk trên worker `slot`; thử lại chunk nếu decode lỗi giữa chừng."""
    payload = [
        {"row": int(r), "start_time": clips[r].start_time, "end_time": clips[r].end_time}
        for r in item.rows
    ]
    npz_path = NPZ_ROOT / f"{category}-w{slot}.npz"
    attempts = max(1, int(CONFIG["video_encode_attempts"]))
    for attempt in range(1, attempts + 1):
        try:
            response = call_worker(
                slot, "encode_video", timeout=None, heartbeat=600,
                video_path=str(video_path), video_id=f"{item.record.video_id}#w{slot}",
                clips=payload, batch_size=BATCH_SIZE,
                embedding_dim=CONFIG["embedding_dim"], out_npz=str(npz_path),
            )
            break
        except RuntimeError as exc:
            if attempt >= attempts or WORKER_ABORT.is_set():
                raise RuntimeError(
                    f"{item.record.video_id}: encode chunk thất bại sau {attempt} lần thử."
                ) from exc
            print(f"[retry] {item.record.video_id} (w{slot}) lỗi ở lần {attempt}/{attempts}: "
                  f"{str(exc).splitlines()[0]}", flush=True)
            npz_path.unlink(missing_ok=True)
            call_worker(slot, "release_video", timeout=300)
    if abs(response["norm_min"] - 1.0) > 2e-5 or abs(response["norm_max"] - 1.0) > 2e-5:
        raise RuntimeError(f"{item.record.video_id}: vector không unit-norm.")
    with np.load(npz_path) as bundle:
        arrays = (bundle["embeddings"].copy(), bundle["frame_indices"].copy(), bundle["rows"].copy())
    npz_path.unlink(missing_ok=True)
    if not np.array_equal(arrays[2], np.asarray(item.rows, dtype=np.int64)):
        raise RuntimeError(f"{item.record.video_id}: worker trả sai row index.")
    return response, arrays


def process_category(category: str):
    global VIDEO_OBJECTS
    if any(VIDEO_OBJECTS[r.video_id]["kind"] == "av1_pending" for r in VIDEO_REGISTRY[category]):
        # Notebook S01 có thể đã chạy xong sau lúc khởi động: đọc lại R2.
        VIDEO_OBJECTS = resolve_video_objects()
        still_pending = [r.video_id for r in VIDEO_REGISTRY[category]
                         if VIDEO_OBJECTS[r.video_id]["kind"] == "av1_pending"]
        if still_pending:
            raise RuntimeError(
                f"{category}: {len(still_pending)} video vẫn là AV1 trên R2 ({still_pending[:3]}...). "
                "Chạy AIC2026_S01_AV1_to_H264_R2_Colab.ipynb rồi Run All lại; shard đã commit được giữ."
            )

    clips = CLIP_PLAN[category]
    completed = load_valid_remote_commits(category, clips)
    expected_shards = total_shards(len(clips))
    if len(completed) == expected_shards:
        print(f"{category}: đã hoàn tất {len(clips):,} clip / {expected_shards} shard; skip.")
        return {"category": category, "rows": len(clips), "shards": expected_shards, "skipped": True}

    rows_todo = pending_rows(len(clips), set(completed))
    items = build_work_items(clips, rows_todo)
    chunks_per_video = Counter(item.record.video_id for item in items)
    records = [r for r in VIDEO_REGISTRY[category] if r.video_id in chunks_per_video]
    video_by_id = {r.video_id: r for r in VIDEO_REGISTRY[category]}
    slots = sorted(WORKERS)
    print(
        f"{category}: resume {len(completed)}/{expected_shards} shard; còn {len(rows_todo):,}/"
        f"{len(clips):,} clip, {len(items)} chunk trên {len(records)} video, {len(slots)} worker."
    )

    buffers: Dict[int, Dict[str, list]] = {}
    buffered_rows: Dict[int, int] = {}
    new_commits = []
    stats = {"clamped": 0, "encode_seconds": 0.0, "clips": 0, "substitutions": 0,
             "eof_trimmed": set(), "reopens": 0}
    progress = tqdm(total=len(rows_todo), desc=f"Encode {category}", unit="clip", smoothing=0.05)
    prefetcher = VideoPrefetcher(records, chunks_per_video,
                                 depth=len(slots) + int(CONFIG["video_prefetch_extra"]))
    work = deque(items)
    work_lock = threading.Lock()
    results = queue.Queue()
    stop_event = threading.Event()

    def worker_loop(slot):
        while not stop_event.is_set():
            with work_lock:
                if not work:
                    return
                item = work.popleft()
            try:
                path = prefetcher.wait(item.record.video_id)
                response, arrays = encode_item(slot, category, clips, item, path)
                results.put(("ok", item, response, arrays))
            except BaseException as exc:  # noqa: BLE001 - chuyển cho main thread
                results.put(("error", item, exc, None))
                return

    threads = [threading.Thread(target=worker_loop, args=(slot,), daemon=True) for slot in slots]
    WORKER_ABORT.clear()
    for thread in threads:
        thread.start()
    remaining = len(items)
    finished_cleanly = False
    try:
        while remaining:
            status, item, payload, arrays = results.get()
            if status == "error":
                raise RuntimeError(f"{category}: chunk của {item.record.video_id} thất bại") from payload
            remaining -= 1
            response = payload
            embeddings, frame_indices, returned_rows = arrays
            stats["clamped"] += int(response["clamped_clips"])
            stats["encode_seconds"] += float(response["elapsed_s"])
            stats["clips"] += int(response["rows"])
            stats["substitutions"] += int(response.get("frame_substitutions", 0))
            stats["reopens"] += int(response.get("reader_reopens", 0))
            if response.get("eof_trimmed_frames") and item.record.video_id not in stats["eof_trimmed"]:
                stats["eof_trimmed"].add(item.record.video_id)
                print(f"[decode] {item.record.video_id}: decord báo {response['reported_frames']} "
                      f"frame nhưng chỉ đọc được {response['decodable_frames']}; window cuối trượt lùi.",
                      flush=True)

            shard_ids = returned_rows // CONFIG["shard_rows"]
            for shard_id in np.unique(shard_ids):
                shard_id = int(shard_id)
                mask = shard_ids == shard_id
                bucket = buffers.setdefault(shard_id, {"emb": [], "fi": [], "rows": []})
                bucket["emb"].append(embeddings[mask])
                bucket["fi"].append(frame_indices[mask])
                bucket["rows"].append(returned_rows[mask])
                buffered_rows[shard_id] = buffered_rows.get(shard_id, 0) + int(mask.sum())
                start, stop = shard_bounds(len(clips), shard_id)
                if buffered_rows[shard_id] == stop - start:
                    bucket = buffers.pop(shard_id)
                    new_commits.append(write_parquet_shard(
                        category, shard_id, clips, video_by_id,
                        np.concatenate(bucket["emb"], axis=0),
                        np.concatenate(bucket["fi"], axis=0),
                        np.concatenate(bucket["rows"], axis=0),
                    ))
                    buffered_rows.pop(shard_id)
                elif buffered_rows[shard_id] > stop - start:
                    raise RuntimeError(f"Buffer overflow ở {category} shard {shard_id}")
            progress.update(len(item.rows))
            prefetcher.chunk_done(item.record)
        finished_cleanly = True
    finally:
        stop_event.set()
        prefetcher.close()
        progress.close()
        if not finished_cleanly:
            # Chunk đang dở chưa commit nên kill worker không mất dữ liệu; Run All lại sẽ
            # tự nạp lại worker và resume từ commit.
            abort_workers()
        for thread in threads:
            thread.join(timeout=60)

    if buffers or buffered_rows:
        raise RuntimeError(f"{category}: còn shard buffer chưa flush: {sorted(buffered_rows)}")
    for slot in slots:
        call_worker(slot, "release_video", timeout=300)

    verified = (load_valid_remote_commits(category, clips) if CONFIG["upload"]
                else {c["shard_id"]: c for c in new_commits})
    if len(verified) != expected_shards:
        raise RuntimeError(f"{category}: final commits {len(verified)} != {expected_shards}")

    success = {
        "schema_version": 1,
        "completed_at": utc_now(),
        "semantic_fingerprint": SEMANTIC_FINGERPRINT,
        "category": category,
        "rows": len(clips),
        "shards": expected_shards,
        "videos": len(VIDEO_REGISTRY[category]),
        "session_name": CONFIG["session_name"],
        "gpu_workers": len(slots),
        "batch_size": BATCH_SIZE,
        "clamped_clips_this_run": stats["clamped"],
        "clips_this_run": stats["clips"],
        # Tổng clip / tổng giây encode của từng worker (mỗi worker chạy song song).
        "clips_per_s_per_worker_this_run": stats["clips"] / max(1e-6, stats["encode_seconds"]),
        "frame_substitutions_this_run": stats["substitutions"],
        "videos_eof_trimmed_this_run": len(stats["eof_trimmed"]),
        "reader_reopens_this_run": stats["reopens"],
        "video_etags": {r.video_id: VIDEO_OBJECTS[r.video_id]["etag"] for r in VIDEO_REGISTRY[category]},
    }
    if CONFIG["upload"]:
        upload_bytes_verified(
            CONFIG["output_bucket_id"],
            f"{CONFIG['output_prefix'].strip('/')}/success/{category}.json",
            json.dumps(success, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    print(json.dumps({k: v for k, v in success.items() if k != "video_etags"}, indent=2))
    return success


CATEGORY_RESULTS = []
for category in CONFIG["categories"]:
    if category in CONFIG["delegated_categories"]:
        print(f"{category}: bỏ qua — encode bằng notebook riêng vào "
              f"{CONFIG['delegated_categories'][category]}.")
        continue
    CATEGORY_RESULTS.append(process_category(category))

# %% [markdown]
# ## 10. Global audit và success manifest
#
# Mỗi runtime ghi audit riêng vào `audits/`. Global manifest và `_SUCCESS.json`
# chỉ được ghi khi **cả M01..M10 và N001..N100** đủ commit; `global_complete=false`
# ở session về sớm là bình thường, không phải lỗi. S01 nằm trong `delegated_categories`:
# collection riêng `derived/tara-tarsier2-7b-3584-batch2-s01-clip-v1` (notebook S01).

# %%
def audit_category_commits(category):
    clips = CLIP_PLAN[category]
    commits = load_valid_remote_commits(category, clips) if CONFIG["upload"] else {}
    expected = total_shards(len(clips))
    committed_rows = sum(int(c["rows"]) for c in commits.values())
    return {
        "videos": EXPECTED_VIDEO_COUNTS[category],
        "expected_rows": len(clips),
        "committed_rows": committed_rows,
        "expected_shards": expected,
        "committed_shards": len(commits),
        "complete": len(commits) == expected and committed_rows == len(clips),
    }


def final_global_audit():
    global VIDEO_OBJECTS
    # Session khác có thể đã encode S01 bằng bản H.264 sau lúc session này khởi động.
    VIDEO_OBJECTS = resolve_video_objects()
    # Category độc lập (state dir riêng) nên audit song song để khỏi chờ API tuần tự.
    audited = [c for c in ALL_CATEGORIES if c not in CONFIG["delegated_categories"]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        per_category = dict(zip(audited, pool.map(audit_category_commits, audited)))
    total_rows = sum(item["committed_rows"] for item in per_category.values())
    all_complete = all(item["complete"] for item in per_category.values())

    manifest = {
        "schema_version": 1,
        "audited_at": utc_now(),
        "semantic_fingerprint": SEMANTIC_FINGERPRINT,
        "semantic_config": SEMANTIC_CONFIG,
        "per_category": per_category,
        "total_committed_rows": total_rows,
        "expected_total_rows": sum(PLAN_COUNTS[c] for c in audited),
        "expected_total_videos": sum(EXPECTED_VIDEO_COUNTS[c] for c in audited),
        # Clip plan trong semantic_config gồm cả category giao đi; phần đó nằm ở collection khác.
        "plan_total_rows_including_delegated": PLAN_TOTAL,
        "delegated_categories": {
            c: {"output_prefix": prefix, "expected_rows": PLAN_COUNTS[c]}
            for c, prefix in CONFIG["delegated_categories"].items()
        },
        "video_etags": {video_id: obj["etag"] for video_id, obj in sorted(VIDEO_OBJECTS.items())},
        "complete": bool(all_complete),
        "retrieval_contract": QUERY_CONTRACT,
    }
    if CONFIG["upload"]:
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        upload_bytes_verified(
            CONFIG["output_bucket_id"],
            f"{CONFIG['output_prefix'].strip('/')}/audits/{CONFIG['session_name']}-{stamp}.json",
            payload,
        )
        if all_complete:
            for name in ("embedding_dataset_manifest.json", "_SUCCESS.json"):
                upload_bytes_verified(
                    CONFIG["output_bucket_id"],
                    f"{CONFIG['output_prefix'].strip('/')}/{name}", payload,
                )
    return manifest


FINAL_MANIFEST = final_global_audit()
print(json.dumps({
    "active_session": CONFIG["active_session"],
    "session_categories": CONFIG["categories"],
    "global_complete": FINAL_MANIFEST["complete"],
    "total_committed_rows": FINAL_MANIFEST["total_committed_rows"],
    "expected_total_rows": FINAL_MANIFEST["expected_total_rows"],
    "output": bucket_uri(CONFIG["output_bucket_id"], CONFIG["output_prefix"] + "/"),
}, ensure_ascii=False, indent=2))

# %% [markdown]
# ## 11. Temporal self-test và đóng worker
#
# Cosine giữa hai câu text khác nhau thì gần như luôn `< 1` với **bất kỳ** embedding
# model nào, nên so text-với-text không chứng minh được gì về temporal ability.
# Ba test dưới đây đo đúng thứ cần đo — video có được xếp gần caption đúng chiều
# hơn caption đảo chiều không:
#
# **A. Known-answer (ground truth của tác giả).** Encode `assets/folding_paper.mp4`
# rồi so với ba caption trong demo chính chủ. README của TARA công bố sẵn
# `folding 0.6488 / cutting 0.3952 / unfolding 0.3009`, nên đây là regression test
# end-to-end cho **toàn bộ** stack — venv, Pillow, decord, reader shim, forward bỏ
# `lm_head` — đối chiếu với số do chính authors đo.
#
# ```text
#               folding_paper.mp4
#                      │
#       ┌──────────────┼──────────────┐
#       ▼              ▼              ▼
#   "folding"      "cutting"     "unfolding"
#     S₊=0.649       0.395          S₋=0.301
#                      Δ = S₊ − S₋ = 0.348
# ```
#
# **B. Arrow-of-time trên chính corpus AIC (không cần caption).** Với `N` clip thật,
# đưa **đúng cùng những frame đó** vào model theo thứ tự ngược:
#
# ```text
# frames  f₁ f₂ f₃ f₄ f₅ f₆ f₇ f₈   →  z_fwd
# frames  f₈ f₇ f₆ f₅ f₄ f₃ f₂ f₁   →  z_rev
#
# d_time    = 1 − cos(z_fwd, z_rev)     ← cùng nội dung, khác chiều thời gian
# d_content = 1 − cos(z_fwd_i, z_fwd_j) ← khác nội dung  (mốc so sánh)
# ```
#
# Với batch 2 đây chỉ là **smoke test** của pipeline, không phải thước đo chất lượng:
# camera giao thông có xe đi cả hai chiều và cảnh đèn đỏ gần như đứng yên, nên đảo thứ tự
# frame có thể trông vẫn tự nhiên mà TARA không hề sai. Vì vậy test chỉ cảnh báo.
#
# **C. Temporal pairs giao thông.** Bảng `TEMPORAL_PAIRS` để trống sẵn — điền
# `video_id / start_time / end_time` cùng caption đúng và hard negative đảo hướng/vai
# (trái↔phải, rẽ trái↔rẽ phải, A vượt B↔B vượt A), cell sẽ đo `Acc = mean(S₊ > S₋)` và
# `Δ = mean(S₊ − S₋)`. **Đây mới là con số quyết định chất lượng cho camera giao thông.**

# %%
# ---------------------------------------------------------------------------
# Điền bảng này bằng ground truth của bạn rồi chạy lại riêng cell này.
# forward  = caption mô tả ĐÚNG thứ tự hành động trong clip
# reversed = ĐÚNG những hành động đó nhưng đảo thứ tự (hard negative)
# Dịch sang tiếng Anh: TARA fine-tune trên NLI-Nuance tiếng Anh.
# ---------------------------------------------------------------------------
TEMPORAL_PAIRS = [
    # Hướng:
    # {"video_id": "N001-V001", "start_time": 120.0, "end_time": 124.0,
    #  "forward":  "a car moves from left to right across the intersection",
    #  "reversed": "a car moves from right to left across the intersection"},
    # Rẽ:
    # {"video_id": "N001-V002", "start_time": 300.0, "end_time": 308.0,
    #  "forward":  "a white car turns left at the intersection",
    #  "reversed": "a white car turns right at the intersection"},
    # Vượt:
    # {"video_id": "N002-V001", "start_time": 64.0, "end_time": 72.0,
    #  "forward":  "a motorcycle overtakes a car",
    #  "reversed": "a car overtakes a motorcycle"},
]

SELF_TEST = {}


def encode_texts(texts):
    out_path = NPZ_ROOT / "self_test_text.npz"
    call_worker(0, "encode_text", timeout=1800, heartbeat=60,
                texts=list(texts), out_npz=str(out_path))
    with np.load(out_path) as bundle:
        vectors = bundle["embeddings"].astype(np.float32)
    out_path.unlink(missing_ok=True)
    return vectors


def encode_clips(video_path, spans, reverse=False):
    # spans = [(start, end), ...] -> ndarray [len(spans), 3584] đã L2-normalize.
    out_path = NPZ_ROOT / "self_test_video.npz"
    payload = [
        {"row": i, "start_time": float(a), "end_time": float(b), "reverse": bool(reverse)}
        for i, (a, b) in enumerate(spans)
    ]
    call_worker(0, "encode_video", timeout=None, heartbeat=120,
                video_path=str(video_path), video_id=Path(video_path).stem,
                clips=payload, batch_size=BATCH_SIZE,
                embedding_dim=CONFIG["embedding_dim"], out_npz=str(out_path))
    with np.load(out_path) as bundle:
        vectors = bundle["embeddings"].astype(np.float32)
    out_path.unlink(missing_ok=True)
    return vectors


try:
    # ---------------- A. Known-answer regression vs demo chính chủ ----------------
    demo_path = MODEL_SNAPSHOT_DIR / CONFIG["self_test_demo_asset"]
    if not demo_path.is_file():
        print(f"BỎ QUA test A: không có {demo_path} "
              "(snapshot_download đang ignore assets/*.mp4?).")
    else:
        probe = call_worker(0, "probe_video", timeout=600, video_path=str(demo_path))
        demo_vector = encode_clips(demo_path, [(0.0, probe["duration_sec"])])[0]
        call_worker(0, "release_video", timeout=300)
        demo_texts = list(CONFIG["self_test_demo_expected"])
        demo_scores = encode_texts(demo_texts) @ demo_vector
        observed = {t: float(v) for t, v in zip(demo_texts, demo_scores)}
        expected = CONFIG["self_test_demo_expected"]
        s_plus = observed["someone is folding a paper"]
        s_minus = observed["someone is unfolding a paper"]
        deviation = max(abs(observed[t] - expected[t]) for t in demo_texts)
        SELF_TEST["A_known_answer"] = {
            "observed": {t: round(v, 4) for t, v in observed.items()},
            "expected_from_official_readme": expected,
            "max_abs_deviation": round(deviation, 4),
            "delta_forward_minus_reversed": round(s_plus - s_minus, 4),
        }
        print("A. Known-answer (assets/folding_paper.mp4)")
        for text in demo_texts:
            print(f"   {observed[text]:.4f}  (chính chủ {expected[text]:.4f})  '{text}'")
        if s_plus <= s_minus:
            raise RuntimeError(
                f"Test A FAIL: 'folding' ({s_plus:.4f}) không thắng 'unfolding' "
                f"({s_minus:.4f}). Stack đang sai ở đâu đó — dừng lại trước khi tin index."
            )
        if deviation > CONFIG["self_test_demo_tolerance"]:
            print(f"   CẢNH BÁO: lệch tối đa {deviation:.4f} so với số chính chủ "
                  f"(ngưỡng {CONFIG['self_test_demo_tolerance']}). Thứ tự vẫn đúng nên "
                  "không fail, nhưng nên soi lại version Pillow/decord/GPU kernel.")
        else:
            print(f"   PASS — Δ = {s_plus - s_minus:.4f}, lệch tối đa {deviation:.4f}.")

    # ---------------- B. Arrow-of-time trên corpus AIC ----------------
    aot_record = min(
        (r for c in CONFIG["categories"] if c not in CONFIG["delegated_categories"]
         for r in VIDEO_REGISTRY[c]),
        key=lambda r: r.size_bytes,
    )
    event_clips = [c for c in CLIP_PLAN[aot_record.category]
                   if c.video_id == aot_record.video_id and c.scale == "event"]
    count = min(int(CONFIG["self_test_aot_clips"]), len(event_clips))
    picked = [event_clips[int(round(i))] for i in np_linspace(0, len(event_clips) - 1, count)]
    spans = [(c.start_time, c.end_time) for c in picked]
    print(f"\nB. Arrow-of-time trên {aot_record.video_id} "
          f"({aot_record.size_bytes / 2 ** 20:.0f} MiB, {count} clip)")
    aot_path = fetch_video(aot_record)
    forward = encode_clips(aot_path, spans, reverse=False)
    backward = encode_clips(aot_path, spans, reverse=True)
    call_worker(0, "release_video", timeout=300)
    drop_video(aot_record)

    d_time = 1.0 - np.sum(forward * backward, axis=1)
    content_matrix = 1.0 - (forward @ forward.T)
    d_content = content_matrix[np.triu_indices(len(forward), k=1)]
    median_time = float(np.median(d_time))
    median_content = float(np.median(d_content))
    SELF_TEST["B_arrow_of_time"] = {
        "video_id": aot_record.video_id,
        "clips": count,
        "d_time_median": round(median_time, 5),
        "d_time_min": round(float(d_time.min()), 5),
        "d_time_max": round(float(d_time.max()), 5),
        "d_content_median": round(median_content, 5),
        "ratio_time_over_content": round(median_time / max(median_content, 1e-9), 4),
        "clips_with_d_time_above_0p01": int((d_time > 0.01).sum()),
    }
    print(json.dumps(SELF_TEST["B_arrow_of_time"], indent=2))
    if median_time < CONFIG["self_test_aot_min_median"]:
        print(
            f"   CẢNH BÁO: đảo thứ tự frame gần như không đổi embedding "
            f"(median d_time = {median_time:.2e}). Với camera cố định có thể do cảnh gần như "
            "đứng yên; hãy kiểm lại bằng TEMPORAL_PAIRS trước khi kết luận."
        )
    print(f"   d_time/d_content = {median_time / max(median_content, 1e-9):.3f} — "
          "càng cao thì thứ tự thời gian càng chiếm nhiều tín hiệu so với nội dung.")

    # ---------------- C. Temporal pairs do bạn cung cấp ----------------
    if not TEMPORAL_PAIRS:
        print("\nC. BỎ QUA: TEMPORAL_PAIRS đang rỗng.\n"
              "   Điền video_id/start_time/end_time + caption xuôi & caption đảo thứ tự\n"
              "   (20–50 cặp từ query đợt 3) rồi chạy lại riêng cell này. Acc và Δ ở đây\n"
              "   mới là căn cứ để quyết định có index toàn corpus hay không.")
    else:
        by_video = {}
        for index, pair in enumerate(TEMPORAL_PAIRS):
            by_video.setdefault(pair["video_id"], []).append((index, pair))
        registry = {r.video_id: r for records in VIDEO_REGISTRY.values() for r in records}
        video_vectors = np.zeros((len(TEMPORAL_PAIRS), CONFIG["embedding_dim"]), np.float32)
        for video_id, entries in by_video.items():
            if video_id not in registry:
                raise RuntimeError(f"TEMPORAL_PAIRS tham chiếu video lạ: {video_id}")
            record = registry[video_id]
            path = fetch_video(record)
            vectors = encode_clips(
                path, [(p["start_time"], p["end_time"]) for _, p in entries]
            )
            call_worker(0, "release_video", timeout=300)
            drop_video(record)
            for (index, _), vector in zip(entries, vectors):
                video_vectors[index] = vector
        forward_text = encode_texts([p["forward"] for p in TEMPORAL_PAIRS])
        reversed_text = encode_texts([p["reversed"] for p in TEMPORAL_PAIRS])
        s_plus = np.sum(video_vectors * forward_text, axis=1)
        s_minus = np.sum(video_vectors * reversed_text, axis=1)
        wins = s_plus > s_minus
        SELF_TEST["C_temporal_pairs"] = {
            "pairs": len(TEMPORAL_PAIRS),
            "accuracy_order": round(float(wins.mean()), 4),
            "delta_mean": round(float((s_plus - s_minus).mean()), 4),
            "delta_median": round(float(np.median(s_plus - s_minus)), 4),
            "s_plus_mean": round(float(s_plus.mean()), 4),
            "s_minus_mean": round(float(s_minus.mean()), 4),
        }
        print("\nC. Temporal pairs")
        print(json.dumps(SELF_TEST["C_temporal_pairs"], indent=2))
        for index, pair in enumerate(TEMPORAL_PAIRS):
            mark = "OK  " if wins[index] else "MISS"
            print(f"   {mark} Δ={s_plus[index] - s_minus[index]:+.4f}  "
                  f"{pair['video_id']} [{pair['start_time']}, {pair['end_time']}]")

    if CONFIG["upload"] and SELF_TEST:
        upload_bytes_verified(
            CONFIG["output_bucket_id"],
            f"{CONFIG['output_prefix'].strip('/')}/self_tests/"
            f"{CONFIG['session_name']}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
            json.dumps({"semantic_fingerprint": SEMANTIC_FINGERPRINT,
                        "results": SELF_TEST}, ensure_ascii=False, indent=2).encode("utf-8"),
        )
finally:
    close_workers()
    print("\nWorker đã shutdown.")

# %% [markdown]
# ## Retrieval contract sau khi encode
#
# ### Vector side
#
# | | |
# |---|---|
# | Model | `bpiyush/TARA` @ `3e7cb730d86ae10da7eb1c85f17e45ece5a5e353` (Tarsier2-7B) |
# | Video prompt | `<video>\nSummary above video in one word:` |
# | Frame | 8 frame uniform (kèm cả frame đầu/cuối window), nhân đôi → 16 image slot |
# | Pixel budget | `max_pixels = 460.800`, aspect ratio giữ nguyên, không crop |
# | Clip scale | N: event 4 s / sequence 8 s / scene 16 s; M và S01: 8 s / 24 s / 72 s; hop 50% |
# | Pooling | hidden state cuối cùng của token cuối |
# | Vector | 3584-d FP32 unit-norm |
# | Metric | COSINE — Milvus `FLOAT_VECTOR(3584)` |
#
# Query side giống hệt collection L (`TARA.encode_text` + FP32 L2 normalize, dịch query
# VI→EN trước khi encode). M và S01 dùng đúng scale của L; N dùng cửa sổ ngắn hơn nên cùng
# tên `scale` nhưng độ dài khác — khi cần, lọc theo `category` (N…) hoặc dùng
# `end_time - start_time`.
#
# ### Layout output
#
# ```text
# derived/tara-tarsier2-7b-3584-batch2-clip-v1/
# ├── run_config.json
# ├── executions/*.json
# ├── audits/*.json
# ├── self_tests/*.json
# ├── embeddings/{M01..M10,N001..N100}/part-*.parquet
# ├── commits/{M01..M10,N001..N100}/part-*.json
# ├── success/{category}.json
# ├── embedding_dataset_manifest.json
# └── _SUCCESS.json                 # chỉ có khi đủ 283.454 clip M/N; S01 (60.562 clip) nằm ở
#                                   # derived/tara-tarsier2-7b-3584-batch2-s01-clip-v1
# ```
#
# Schema mỗi row giữ nguyên collection L:
#
# ```text
# clip_id        string            N001-V001@event@t000012000
# video_id       string            N001-V001
# category       string            N001
# scale          string            event | sequence | scene
# scale_index    int8              0 | 1 | 2
# start_time     float64           giây
# end_time       float64           giây
# fps            float32
# duration_sec   float64           duration luồng video trong manifest pin
# video_relpath  string            Videos/Videos_N001/N001-V001.mp4  (key trên R2 aic26-media)
# frame_indices  int32[8]          frame index thật đã đưa vào model
# embedding      float32[3584]     unit-norm
# ```
#
# `start_time`/`end_time` là **identity của plan** (quyết định row index và shard).
# `frame_indices` mới là **ground truth** của những frame model thực sự nhìn thấy; cửa sổ
# thật là `frame_indices[0]/fps .. frame_indices[-1]/fps`. Clip cuối của một video có thể bị
# trượt lùi nếu decoder đọc được ít frame hơn manifest (`clamped_clips_this_run` trong
# `success/{category}.json`). Khi cần định vị thời điểm (TRAKE), hãy dùng `frame_indices`.
#
# Mỗi commit và `success/*.json` ghi `video_etags`: ETag của đúng bytes video đã encode. Với
# S01 đó là bản H.264 do notebook S01 tạo (bản AV1 gốc nằm ở `Videos_AV1_Original/`).
#
# ### Dùng cho camera giao thông
#
# - Query một chuyển động đơn ("a car turns left", "a motorcycle moves from left to right")
#   → ưu tiên `scale='event'`.
# - Query tương tác/maneuver ("a motorcycle overtakes a car") → `scale='sequence'`, rồi
#   `scene` làm ngữ cảnh.
# - Trước khi tin index, điền `TEMPORAL_PAIRS` ở section 11 bằng 30–100 cặp giao thông thật
#   (hướng, rẽ, vượt, cắt ngang) và đo `Acc_order`. Nếu trái/phải tốt mà vượt xe yếu, thử
#   `n_frames=16`; nếu xe máy xa bị lẫn, thử `max_pixels≈921.600` — mỗi thử nghiệm là một
#   collection mới, đừng đổi hai tham số cùng lúc.
#
# ### Fusion
#
# Không cộng thẳng cosine của TARA với PE-Core hay Qwen3-VL-Embedding: ba embedding space
# có score distribution khác nhau. Fuse bằng rank/RRF.
