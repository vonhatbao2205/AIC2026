# %% [markdown]
# # AIC 2026 — TARA (Tarsier2-7B) video-clip embedding (Colab A100 40 GB)
#
# Notebook E2E này encode **873 video L21–L30** của bucket `Baonenha1/aic26-media`
# thành **168.536 clip embedding 3584-d** bằng
# [`bpiyush/TARA`](https://huggingface.co/bpiyush/TARA) (ECCV 2026,
# *Adapting MLLMs for Nuanced Video Retrieval*), ghi Parquet shard và upload
# checkpoint-safe lên Hugging Face Storage Bucket.
#
# ## Vì sao TARA và vì sao index theo clip
#
# Kênh retrieval hiện tại của repo (PE-Core, Qwen3-VL-Embedding) đều là
# **frame-level**: một vector cho một keyframe. Query đợt 3 lại có rất nhiều mô tả
# *chuỗi hành động* (`đổ trứng → cắt đậu hũ → cho vào nồi → khuấy`), và benchmark
# T-KIS nội bộ đã cho thấy đúng failure mode đó: R@1 rơi từ 60,00% (moment) xuống
# 18,18% (sequence) trong khi R@100 gần như không đổi — tức là **tìm đúng video
# nhưng không xếp đúng lên đầu**.
#
# TARA fine-tune Tarsier2-7B bằng contrastive loss trên text triplet có
# hard-negative *chiral* (`opening a door` vs `closing a door`), nên embedding
# phân biệt được chiều/thứ tự thời gian. Trên CiA-EPIC (EPIC-KITCHENS, đúng domain
# bếp núc của đợt 3): Chiral 81,1 / Static 45,6 / All 38,9 so với Qwen3-VL-Embedding
# 62,1 / 28,6 / 20,6.
#
# Nhưng một vector cho cả video 45 phút thì vô dụng, còn `TARA(keyframe.jpg)` thì
# biến video model thành image model. Notebook này vì vậy build
# **multi-scale temporal clip index**: mỗi video sinh ra clip ở ba scale
# chồng lấn 50%, mỗi clip là một record riêng có `start_time`/`end_time`/`frame_indices`.
#
# ## Thiết kế chính
#
# - **Environment tách đôi.** TARA pin `torch==2.5.1+cu121` / `transformers==4.45.0`
#   / `numpy==1.26.4` / `flash-attn==2.8.3` và code của nó import sâu vào internals
#   của transformers 4.45 (`ProcessingKwargs`, `AttentionMaskConverter`,
#   `SlidingWindowCache`, …). Colab Latest là Python 3.13 + Torch 2.11 và
#   **không cài được `numpy==1.26.4`**. Nên notebook dựng một venv Python 3.10
#   riêng bằng `uv` và chạy model trong **worker subprocess** giao tiếp JSON-lines;
#   kernel Colab chỉ lo bucket I/O, planning, Parquet, audit và commit.
# - **Checkpoint chính chủ, pin theo commit.** `snapshot_download("bpiyush/TARA",
#   revision=…)` — repo HF chứa cả weights lẫn code (`modeling_tara.py`,
#   `tarsier2/`, `shared/`, `tarsier2/default_config.yaml`), không cần git-lfs
#   và không cần clone GitHub.
# - **Semantic contract khóa đúng default chính chủ**: EOL prompt
#   `<video>\nSummary above video in one word:`, `n_frames=8` +
#   `use_multi_images_for_video=true` (⇒ 16 image slot, đúng `F=16` của paper),
#   `max_pixels=460.800`, `video_sampler_version=v1`, pooling last-token,
#   BF16 + FlashAttention-2.
# - **Clip window đi qua đúng code path chính chủ.** `VisionParser` hỗ trợ sẵn
#   `{"video_file": …, "start_time": …, "end_time": …}` và `{"frame_indices": […]}`,
#   nên không cần cắt file mp4. Notebook tự tính `frame_indices` bằng bản sao chính
#   xác của `sample_video`, rồi **audit đối chiếu bằng chính `sample_video`** để
#   chứng minh index trùng khớp tuyệt đối.
# - **Decode một lần cho mỗi video.** `tarsier2 VideoReader` gốc mở lại file và seek
#   cho *từng* clip. Worker thay bằng reader có LRU frame cache (nội dung frame giữ
#   nguyên: `get_batch → asnumpy → Image.fromarray().convert("RGB")`), prefetch theo
#   batch nên decode chạy tiến về phía trước thay vì seek ngẫu nhiên ~1,3M lần.
# - **Bỏ `lm_head` trong forward production.** `generate(max_new_tokens=1)` vẫn tính
#   đủ `logits` 152.064-way cho toàn bộ ~4.700 token (≈1,4 GiB BF16/clip) rồi vứt đi.
#   Worker thay `lm_head` bằng slice last-token nên hidden state cuối **không đổi**,
#   và điều đó được gate bằng audit cosine với đúng `TARA.encode_vision`.
# - **Resume theo shard 1.024 clip.** Parquet upload → verify size → mới ghi commit
#   JSON; chỉ commit hợp lệ mới được coi là hoàn tất.
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

CONFIG = {
    # Media nguồn (notebook không bao giờ sửa/xóa).
    "source_bucket_id": "Baonenha1/aic26-media",
    "video_manifest_path": "manifest/video_manifest.csv",
    "dataset_manifest_path": "manifest/dataset_manifest.json",

    # Derived output. Cùng bucket nhưng prefix tách biệt hoàn toàn.
    "output_bucket_id": "Baonenha1/aic26-media",
    "output_prefix": "derived/tara-tarsier2-7b-3584-clip-v1",

    # Chạy song song: tạo ba bản Colab của notebook, mỗi bản chỉ đổi đúng dòng
    # active_session. Assignment phải rời nhau để không có hai runtime ghi cùng shard.
    "active_session": "session_1",  # session_1 | session_2 | session_3
    "session_assignments": {
        "session_1": ["L26"],
        "session_2": ["L25", "L23", "L27"],
        "session_3": ["L21", "L22", "L24", "L28", "L29", "L30"],
    },

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

    # ---- Multi-scale temporal clip plan ----
    # (tên scale, độ dài window giây, hop giây). Hop = window/2 ⇒ chồng lấn 50%.
    #   event    : một thao tác đơn ("cắt đầu tôm"), 8 frame / 8 s = 1 fps.
    #   sequence : chuỗi 3–4 bước ("nhúng thịt → cho vào tô → chan nước dùng").
    #   scene    : ngữ cảnh cảnh quay, dùng cho KIS mô tả bối cảnh.
    "scales": [
        ["event", 8.0, 4.0],
        ["sequence", 24.0, 12.0],
        ["scene", 72.0, 36.0],
    ],
    "clip_time_decimals": 3,

    # ---- Throughput / execution policy (không nằm trong semantic fingerprint) ----
    "batch_size": None,            # None = autotune theo clip/s trong ngưỡng VRAM
    "batch_candidates": [1, 2, 3, 4, 6, 8],
    "benchmark_rounds": 3,
    "max_vram_fraction": 0.92,
    # PHẢI là 1. transformers 4.45 memoize compiled Jinja template bằng
    # @lru_cache, nên `AssistantTracker` (state của block {% generation %}) là
    # singleton dùng chung; hai lần `apply_chat_template` chạy song song sẽ ném
    # "AssistantTracker should not be reused before closed". Template của TARA
    # có {% generation %} và TarsierProcessor luôn bật return_assistant_tokens_mask,
    # nên mọi clip đều đi qua đúng đường đó. Pool 1 worker vẫn overlap CPU với
    # GPU qua hàng đợi prefetch — đó mới là thứ ta cần, không phải song song.
    "prep_threads": 1,
    "prefetch_batches": 2,
    "decord_threads": 4,
    # Một số mp4 trong corpus làm decord hết budget retry khi đọc vài frame
    # cuối ("Unable to handle EOF ... DECORD_EOF_RETRY_MAX=10240"). Nâng hạn
    # mức, và worker còn probe độ dài thực sự decode được lúc mở file.
    "decord_eof_retry_max": 65_536,
    "decode_substitution_window": 8,   # frame lỗi giữa file -> lùi tối đa 8 frame
    "decode_probe_max_steps": 16,
    "video_encode_attempts": 2,        # thử lại một video trước khi bỏ cuộc
    "bypass_lm_head": True,        # gated bởi audit; False = dùng nguyên generate()

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
    "self_test_aot_clips": 24,      # số clip corpus dùng cho arrow-of-time test
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
    "verify_video_sha256": True,

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
    "scratch_root": "/content/aic_tara_clip",
    "hf_token_file_candidates": [
        "/content/HF_TOKEN.txt",
        "/content/drive/MyDrive/HF_TOKEN.txt",
    ],
    "fail_on_non_a100": True,
    "min_free_disk_gib": 40,
}

ACTIVE_SESSION = CONFIG["active_session"]
if ACTIVE_SESSION not in CONFIG["session_assignments"]:
    raise ValueError(f"active_session không tồn tại: {ACTIVE_SESSION}")
CONFIG["categories"] = list(CONFIG["session_assignments"][ACTIVE_SESSION])
CONFIG["session_name"] = f"colab_{ACTIVE_SESSION}"


# Source-of-truth checksums lấy từ manifest phân phối `aic26-media`.
SOURCE_MANIFEST = {
    "video_manifest": {
        "bytes": 716_552,
        "sha256": "db8d3cd72b58095cb354a0c6650fe9d1fddb3f773275e0ca172837020d047464",
    },
    "dataset_manifest": {
        "bytes": 8_070,
        "sha256": "fff4774571c30efbb1252b69c9a77d40a0d98fd6fb848716192d654f65f9ba16",
    },
}

# Checkpoint bpiyush/TARA @ 3e7cb730 (4 shard safetensors).
EXPECTED_MODEL_TENSOR_PAYLOAD_BYTES = 16_582_751_232

# Số video L21..L30 trong `aic26-media` (component L = 873 video).
EXPECTED_VIDEO_COUNTS = {
    "L21": 29, "L22": 31, "L23": 25, "L24": 43, "L25": 88,
    "L26": 498, "L27": 16, "L28": 24, "L29": 23, "L30": 96,
}

# Số clip sinh ra bởi đúng CONFIG["scales"] mặc định ở trên. Đây là invariant
# của plan: nếu đổi scales thì con số này sẽ khác và assert sẽ tự tắt.
REFERENCE_SCALES = [["event", 8.0, 4.0], ["sequence", 24.0, 12.0], ["scene", 72.0, 36.0]]
REFERENCE_CLIP_COUNTS = {
    "L21": 11_487, "L22": 13_207, "L23": 3_471, "L24": 7_933, "L25": 46_966,
    "L26": 56_154, "L27": 3_383, "L28": 9_820, "L29": 8_863, "L30": 7_252,
}

print("Active session:", CONFIG["session_name"], "categories:", CONFIG["categories"])
print("Clip dự kiến cho session này:",
      f'{sum(REFERENCE_CLIP_COUNTS[c] for c in CONFIG["categories"]):,}',
      f'/ {sum(REFERENCE_CLIP_COUNTS.values()):,} toàn bộ L21..L30')

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
# ## 2. Xác thực HF và preflight phần cứng
#
# Token lấy theo thứ tự: Colab Secret `HF_TOKEN` → environment → file cấu hình →
# prompt ẩn. Token không được in và không được ghi vào notebook/manifest.

# %%
import getpass
import importlib.metadata
import re
import socket
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

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


HF_TOKEN = get_hf_token()
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
identity = whoami(token=HF_TOKEN)
print("Hugging Face account:", identity.get("name", "authenticated"))
print("Source bucket:", bucket_info(CONFIG["source_bucket_id"], token=HF_TOKEN).id)
print("Output bucket:", bucket_info(CONFIG["output_bucket_id"], token=HF_TOKEN).id)

GPU_NAME = VENV_INFO["gpu"]
if GPU_NAME is None:
    raise RuntimeError("Không thấy GPU.")
if CONFIG["fail_on_non_a100"] and "A100" not in GPU_NAME.upper():
    raise RuntimeError(f"Yêu cầu A100 nhưng runtime hiện tại là {GPU_NAME}.")

nvidia = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
     "--format=csv,noheader,nounits"],
    check=True, capture_output=True, text=True,
).stdout.strip().splitlines()[0]
gpu_fields = [field.strip() for field in nvidia.split(",")]
GPU_VRAM_GIB = float(gpu_fields[1]) / 1024
SYSTEM_RAM_GIB = psutil.virtual_memory().total / 2 ** 30
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
# ## 3. I/O Bucket, checksum và commit protocol

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


def download_bucket_file_verified(
    bucket_id: str, remote_path: str, local_path: Path,
    expected_bytes: Optional[int] = None, expected_sha256: Optional[str] = None,
):
    item = remote_info(bucket_id, remote_path)
    if item is None:
        raise FileNotFoundError(f"Không thấy object: {bucket_uri(bucket_id, remote_path)}")
    if expected_bytes is not None and int(item.size) != int(expected_bytes):
        raise RuntimeError(
            f"Remote size mismatch cho {remote_path}: {item.size} != {expected_bytes}"
        )
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.unlink(missing_ok=True)
    download_bucket_files(bucket_id, files=[(item, str(local_path))], token=HF_TOKEN)
    actual_bytes = local_path.stat().st_size
    if int(item.size) != actual_bytes:
        raise RuntimeError(f"Download size mismatch cho {remote_path}: {actual_bytes}")
    if expected_sha256 is not None:
        actual_sha = sha256_file(local_path)
        if actual_sha != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch cho {remote_path}: {actual_sha}")
    return local_path


def natural_video_key(video_id: str):
    match = re.search(r"_V(\d+)$", video_id)
    return (int(match.group(1)) if match else 10 ** 12, video_id)


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
    if set(owners) != set(EXPECTED_VIDEO_COUNTS):
        raise ValueError(
            "session_assignments phải phủ đúng L21..L30; "
            f"missing={sorted(set(EXPECTED_VIDEO_COUNTS) - set(owners))}, "
            f"extra={sorted(set(owners) - set(EXPECTED_VIDEO_COUNTS))}"
        )
    if CONFIG["categories"] != list(assignments[CONFIG["active_session"]]):
        raise ValueError("categories phải được suy ra từ active_session.")
    for category in CONFIG["categories"]:
        if not re.fullmatch(r"L(?:2[1-9]|30)", category):
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

    scales = CONFIG["scales"]
    if not scales:
        raise ValueError("Cần ít nhất một scale.")
    names = [s[0] for s in scales]
    if len(set(names)) != len(names):
        raise ValueError("Tên scale bị trùng.")
    for name, window, hop in scales:
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError(f"Tên scale không hợp lệ: {name}")
        if not (0 < float(hop) <= float(window)):
            raise ValueError(f"Scale {name}: cần 0 < hop <= window.")
    if int(CONFIG["prep_threads"]) != 1:
        raise ValueError(
            "prep_threads phải bằng 1.\n"
            "  - transformers 4.45 cache compiled Jinja template bằng @lru_cache nên "
            "AssistantTracker là singleton; gọi apply_chat_template song song sẽ ném "
            "'AssistantTracker should not be reused before closed'.\n"
            "  - Và tăng lên cũng không được gì: cả frame source lẫn processor đều đã "
            "bị serialize bằng lock. Muốn GPU bận hơn thì tăng batch_size/prefetch_batches."
        )
    if int(CONFIG["prefetch_batches"]) < 1:
        raise ValueError("prefetch_batches phải >= 1 để CPU prep overlap được với GPU.")
    if CONFIG["output_prefix"].strip("/").startswith(("Videos", "Keyframes", "manifest")):
        raise ValueError("output_prefix không được ghi đè lên media/manifest nguồn.")
    if int(CONFIG["shard_rows"]) <= 0:
        raise ValueError("shard_rows phải > 0.")
    if not (0 < float(CONFIG["max_vram_fraction"]) < 1):
        raise ValueError("max_vram_fraction phải nằm trong (0,1).")


validate_config()
print("Config hợp lệ.")

# %% [markdown]
# ## 4. Tải/verify video manifest và dựng registry L21..L30

# %%
import csv
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    category: str
    remote_path: str
    fps: float
    duration_sec: float
    size_bytes: int
    sha256: str


VIDEO_MANIFEST_LOCAL = STATE_ROOT / "video_manifest.csv"
DATASET_MANIFEST_LOCAL = STATE_ROOT / "dataset_manifest.json"

download_bucket_file_verified(
    CONFIG["source_bucket_id"], CONFIG["video_manifest_path"], VIDEO_MANIFEST_LOCAL,
    SOURCE_MANIFEST["video_manifest"]["bytes"], SOURCE_MANIFEST["video_manifest"]["sha256"],
)
download_bucket_file_verified(
    CONFIG["source_bucket_id"], CONFIG["dataset_manifest_path"], DATASET_MANIFEST_LOCAL,
    SOURCE_MANIFEST["dataset_manifest"]["bytes"], SOURCE_MANIFEST["dataset_manifest"]["sha256"],
)
DATASET_MANIFEST = json.loads(DATASET_MANIFEST_LOCAL.read_text(encoding="utf-8"))
L_COMPONENT = DATASET_MANIFEST["components"]["L"]["counts"]
if int(L_COMPONENT["videos"]) != sum(EXPECTED_VIDEO_COUNTS.values()):
    raise RuntimeError(
        f"dataset_manifest báo {L_COMPONENT['videos']} video L, "
        f"contract mong đợi {sum(EXPECTED_VIDEO_COUNTS.values())}."
    )
print("L component:", json.dumps(L_COMPONENT))


def load_video_registry():
    by_category: Dict[str, List[VideoRecord]] = {}
    with VIDEO_MANIFEST_LOCAL.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] != "video":
                continue
            category = row["category"].strip()
            if category not in EXPECTED_VIDEO_COUNTS:
                continue
            record = VideoRecord(
                video_id=row["video_id"].strip(),
                category=category,
                remote_path=row["hf_path"].strip(),
                fps=float(row["fps"]),
                duration_sec=float(row["duration_sec"]),
                size_bytes=int(row["size_bytes"]),
                sha256=row["sha256"].strip(),
            )
            if record.fps <= 0 or record.duration_sec <= 0 or record.size_bytes <= 0:
                raise RuntimeError(f"Manifest row hỏng: {record.video_id}")
            if len(record.sha256) != 64:
                raise RuntimeError(f"Thiếu SHA-256 cho {record.video_id}")
            by_category.setdefault(category, []).append(record)
    for category, records in by_category.items():
        records.sort(key=lambda r: natural_video_key(r.video_id))
        if len(records) != EXPECTED_VIDEO_COUNTS[category]:
            raise RuntimeError(
                f"{category}: {len(records)} video != {EXPECTED_VIDEO_COUNTS[category]}"
            )
        if len({r.video_id for r in records}) != len(records):
            raise RuntimeError(f"{category}: video_id trùng trong manifest.")
    missing = sorted(set(EXPECTED_VIDEO_COUNTS) - set(by_category))
    if missing:
        raise RuntimeError(f"Thiếu category trong manifest: {missing}")
    return by_category


VIDEO_REGISTRY = load_video_registry()
print(json.dumps({
    "videos_total": sum(len(v) for v in VIDEO_REGISTRY.values()),
    "per_category": {c: len(v) for c, v in sorted(VIDEO_REGISTRY.items())},
    "hours_total": round(
        sum(r.duration_sec for v in VIDEO_REGISTRY.values() for r in v) / 3600, 2
    ),
    "bytes_total_gib": round(
        sum(r.size_bytes for v in VIDEO_REGISTRY.values() for r in v) / 2 ** 30, 2
    ),
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
# scene) → start_time tăng dần`. Nhờ vậy mọi clip của cùng một video nằm liền
# nhau, và vòng encode chỉ cần giữ **một** video trên disk tại một thời điểm.

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
    for scale_index, (name, window, hop) in enumerate(CONFIG["scales"]):
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
    category: build_category_plan(category) for category in EXPECTED_VIDEO_COUNTS
}
PLAN_COUNTS = {category: len(clips) for category, clips in CLIP_PLAN.items()}
PLAN_TOTAL = sum(PLAN_COUNTS.values())

scales_match_reference = [list(s) for s in CONFIG["scales"]] == REFERENCE_SCALES
if scales_match_reference and PLAN_COUNTS != REFERENCE_CLIP_COUNTS:
    raise RuntimeError(
        "Plan không tái tạo được số clip tham chiếu của scales mặc định: "
        f"{PLAN_COUNTS} != {REFERENCE_CLIP_COUNTS}. Manifest hoặc plan đã đổi."
    )
if not scales_match_reference:
    print(
        "CẢNH BÁO: CONFIG['scales'] khác REFERENCE_SCALES nên bỏ qua assert số clip. "
        "Đây là embedding collection MỚI — hãy đổi luôn output_prefix."
    )

per_scale = {}
for clips in CLIP_PLAN.values():
    for clip in clips:
        per_scale[clip.scale] = per_scale.get(clip.scale, 0) + 1
session_clips = sum(PLAN_COUNTS[c] for c in CONFIG["categories"])
session_bytes = sum(r.size_bytes for c in CONFIG["categories"] for r in VIDEO_REGISTRY[c])

print(json.dumps({
    "clips_total": PLAN_TOTAL,
    "per_category": PLAN_COUNTS,
    "per_scale": per_scale,
    "session": CONFIG["session_name"],
    "session_categories": CONFIG["categories"],
    "session_clips": session_clips,
    "session_video_bytes_gib": round(session_bytes / 2 ** 30, 2),
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
        "bucket_id": CONFIG["source_bucket_id"],
        "video_manifest_sha256": SOURCE_MANIFEST["video_manifest"]["sha256"],
        "dataset_manifest_sha256": SOURCE_MANIFEST["dataset_manifest"]["sha256"],
        "expected_video_counts": EXPECTED_VIDEO_COUNTS,
        "expected_total_videos": sum(EXPECTED_VIDEO_COUNTS.values()),
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
        "scales": [list(scale) for scale in CONFIG["scales"]],
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
    if set(remote_owners) != set(EXPECTED_VIDEO_COUNTS):
        raise RuntimeError("Remote parallel_plan không phủ đúng L21..L30.")
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
            note("OOM tai batch=" + str(batch_size) + "; dung tang batch.")
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


WORKER_PATH = SCRATCH_ROOT / "tara_encode_worker.py"
WORKER_PATH.write_text(WORKER_SOURCE, encoding="utf-8")
WORKER_SOURCE_SHA256 = sha256_bytes(WORKER_SOURCE.encode("utf-8"))
print("Worker:", WORKER_PATH, "sha256:", WORKER_SOURCE_SHA256)

WORKER_PROTO = "@@TARA-RPC@@"


class WorkerDied(RuntimeError):
    pass


class TaraWorker:
    """Client RPC cho worker chạy trong venv Python 3.10."""

    def __init__(self, python_bin, script_path, log_path, env=None):
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
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=120):
            raise WorkerDied("Worker không gửi tín hiệu ready trong 120 s.")
        print(f"[tara] worker pid={self.proc.pid} log={self.log_path}")

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
                    print(f"[tara] {line}", flush=True)
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()

    def _on_progress(self, payload):
        now = time.monotonic()
        if now - self._progress_state["last_print"] < 10.0:
            return
        self._progress_state["last_print"] = now
        print(
            f"[tara] {payload.get('label')}: {payload.get('done')}/{payload.get('total')} clip"
            f" | {payload.get('clips_per_s', 0.0):.2f} clip/s",
            flush=True,
        )

    def call(self, cmd, timeout=None, heartbeat=60.0, **kwargs):
        with self._condition:
            if self._closed or self.proc.poll() is not None:
                raise WorkerDied(f"Worker đã chết (rc={self.proc.poll()}).")
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
                    print(f"[tara] chờ cmd={cmd} … {_elapsed_text(now - started)}", flush=True)
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


def start_worker():
    log_path = LOG_ROOT / f"worker_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.log"
    worker = TaraWorker(VENV_PYTHON, WORKER_PATH, log_path)
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
    return worker, info


WORKER, WORKER_INFO = start_worker()
print(json.dumps({k: v for k, v in WORKER_INFO.items() if k != "checks"}, indent=2))
print(json.dumps(WORKER_INFO["checks"], indent=2, ensure_ascii=False))

# %% [markdown]
# ## 8. Resume state, autotune batch và quality gate
#
# Thứ tự bắt buộc: đọc commit hợp lệ → chọn category chưa xong → tải một video
# thật → benchmark batch trên clip thật → audit đối chiếu với đường chính chủ →
# mới cho phép ghi vector.

# %%
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
    destination = local_video_path(record)
    if destination.is_file() and destination.stat().st_size == record.size_bytes:
        # Size khớp là chưa đủ. Một runtime trước có thể để lại file hỏng do bị
        # kill giữa chừng hoặc disk lỗi mà kích thước vẫn đúng; nếu tin nó thì
        # `verify_video_sha256` chỉ bảo vệ lần tải mới chứ không bảo vệ resume.
        if (not CONFIG["verify_video_sha256"]
                or sha256_file(destination) == record.sha256):
            return destination
        print(f"[cache] {record.video_id}: SHA-256 không khớp; xóa và tải lại.", flush=True)
        destination.unlink()
    free = shutil.disk_usage(SCRATCH_ROOT).free
    if free < record.size_bytes + (4 << 30):
        raise RuntimeError(
            f"Không đủ disk cho {record.video_id}: cần "
            f"{(record.size_bytes + (4 << 30)) / 2 ** 30:.1f} GiB, còn {free / 2 ** 30:.1f} GiB."
        )
    download_bucket_file_verified(
        CONFIG["source_bucket_id"], record.remote_path, destination,
        record.size_bytes, record.sha256 if CONFIG["verify_video_sha256"] else None,
    )
    return destination


def drop_video(record: VideoRecord):
    if not CONFIG["cleanup_video_after_encode"]:
        return
    path = local_video_path(record)
    expected_root = VIDEO_ROOT.resolve()
    if path.resolve().parent.parent != expected_root:
        raise RuntimeError(f"Từ chối xóa ngoài scratch video root: {path}")
    path.unlink(missing_ok=True)


# --- chọn category đầu tiên chưa hoàn tất để calibrate trên dữ liệu thật ---
FIRST_CATEGORY = None
FIRST_CLIPS = None
FIRST_COMMITS = None
for candidate in CONFIG["categories"]:
    candidate_clips = CLIP_PLAN[candidate]
    commits = load_valid_remote_commits(candidate, candidate_clips)
    if len(commits) < total_shards(len(candidate_clips)):
        FIRST_CATEGORY, FIRST_CLIPS, FIRST_COMMITS = candidate, candidate_clips, commits
        break

BATCH_SIZE = int(CONFIG["batch_size"] or 1)
BATCH_BENCHMARK = []
AUDIT = {}

if FIRST_CLIPS is None:
    print("Mọi category của session này đã hoàn tất; bỏ qua autotune và audit.")
else:
    calib_record = VIDEO_REGISTRY[FIRST_CATEGORY][0]
    calib_path = str(fetch_video(calib_record))
    calib_clips = [c for c in FIRST_CLIPS if c.video_id == calib_record.video_id]
    print(f"Calibrate trên {calib_record.video_id} "
          f"({calib_record.duration_sec:.1f}s, {len(calib_clips)} clip)")

    def clip_payload(clip: ClipRecord, row: int = 0):
        return {"row": row, "start_time": clip.start_time, "end_time": clip.end_time}

    # --- autotune batch ---
    candidates = ([int(CONFIG["batch_size"])] if CONFIG["batch_size"] is not None
                  else [int(x) for x in CONFIG["batch_candidates"]])
    candidates = sorted({c for c in candidates if c > 0})
    bench_clips = [clip_payload(c) for c in calib_clips[:max(candidates) * CONFIG["benchmark_rounds"]]]
    bench = WORKER.call(
        "benchmark", timeout=3600, heartbeat=30,
        video_path=calib_path, clips=bench_clips,
        batch_candidates=candidates, rounds=CONFIG["benchmark_rounds"],
    )
    BATCH_BENCHMARK = bench["results"]
    if not BATCH_BENCHMARK:
        raise RuntimeError("Benchmark không trả về kết quả nào (OOM ngay batch nhỏ nhất?).")
    safe = [r for r in BATCH_BENCHMARK if r["peak_fraction"] <= CONFIG["max_vram_fraction"]]
    best = max(safe or BATCH_BENCHMARK, key=lambda r: r["clips_per_s"])
    if CONFIG["batch_size"] is not None and best["batch_size"] != int(CONFIG["batch_size"]):
        raise RuntimeError("Manual batch không vượt qua validation VRAM.")
    BATCH_SIZE = int(best["batch_size"])
    for row in BATCH_BENCHMARK:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    print("Selected batch size:", BATCH_SIZE, f"({best['clips_per_s']:.2f} clip/s)")

    # --- quality gate ---
    sample_count = min(int(CONFIG["audit_clip_samples"]), len(calib_clips))
    audit_indices = sorted({int(round(i)) for i in
                            np_linspace(0, len(calib_clips) - 1, sample_count)})
    audit_clips = [clip_payload(calib_clips[i]) for i in audit_indices]
    AUDIT = WORKER.call(
        "audit", timeout=3600, heartbeat=30,
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

    WORKER.call("release_video", timeout=300)
    measured = best["clips_per_s"]
    remaining = 0
    for category in CONFIG["categories"]:
        clips_here = CLIP_PLAN[category]
        done = FIRST_COMMITS if category == FIRST_CATEGORY else load_valid_remote_commits(
            category, clips_here
        )
        remaining += len(clips_here) - sum(int(c["rows"]) for c in done.values())
    print(json.dumps({
        "measured_clips_per_s": round(measured, 3),
        "session_clips_total": session_clips,
        "session_clips_remaining": remaining,
        "session_hours_remaining_estimate": round(remaining / measured / 3600, 2),
        "note": "Ước tính từ benchmark trên một video; throughput thật đổi theo độ phân giải video.",
    }, indent=2))

EXECUTION_INFO = {
    "started_at": utc_now(),
    "session_name": CONFIG["session_name"],
    "host": socket.gethostname(),
    "gpu": GPU_NAME,
    "gpu_vram_gib": GPU_VRAM_GIB,
    "system_ram_gib": SYSTEM_RAM_GIB,
    "batch_size": BATCH_SIZE,
    "batch_benchmark": BATCH_BENCHMARK,
    "audit": AUDIT,
    "bypass_lm_head": CONFIG["bypass_lm_head"],
    "prep_threads": CONFIG["prep_threads"],
    "prefetch_batches": CONFIG["prefetch_batches"],
    "decord_threads": CONFIG["decord_threads"],
    "decord_eof_retry_max": CONFIG["decord_eof_retry_max"],
    "decode_substitution_window": CONFIG["decode_substitution_window"],
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
# Vòng chính: **prefetch tải video kế tiếp trong khi GPU đang encode video hiện
# tại**. Mỗi video đi qua worker một lần, trả về `.npz` (embedding + frame indices
# + row indices); notebook gom vào buffer theo shard, shard nào đủ row thì ghi
# Parquet → upload → verify size → **rồi mới** ghi commit JSON. Commit là
# transaction boundary duy nhất.

# %%
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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
    b"scales": json.dumps(CONFIG["scales"]).encode(),
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
    ).replace_schema_metadata(PARQUET_SCHEMA_METADATA)

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
        "execution": {
            "session_name": CONFIG["session_name"],
            "gpu": GPU_NAME,
            "batch_size": BATCH_SIZE,
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


def call_worker(cmd, _restarts=1, **kwargs):
    """Gọi worker, tự restart một lần nếu process chết giữa chừng.

    Một job 14 giờ không nên mất sạch vì một lần CUDA error. Restart chỉ nạp lại
    model — batch size và semantic contract giữ nguyên, và mọi shard đã commit
    vẫn hợp lệ nên phần việc mất đi nhiều nhất là một video.
    """
    global WORKER
    try:
        return WORKER.call(cmd, **kwargs)
    except WorkerDied as exc:
        if _restarts <= 0:
            raise
        print(f"[tara] worker chết ({exc}); restart và thử lại cmd={cmd}.", flush=True)
        try:
            WORKER.close()
        except Exception:
            pass
        WORKER, restarted_info = start_worker()
        if restarted_info["checks"]["video_eol_prompt"] != CONFIG["video_eol_prompt"]:
            raise RuntimeError("Worker restart nhưng contract đã đổi.") from exc
        return call_worker(cmd, _restarts=_restarts - 1, **kwargs)


class VideoPrefetcher:
    """Tải video kế tiếp trên thread nền trong khi GPU encode video hiện tại."""

    def __init__(self, records, depth=1):
        self.records = list(records)
        self.queue = queue.Queue(maxsize=max(1, depth))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        for record in self.records:
            if self.stop_event.is_set():
                break
            try:
                path = fetch_video(record)
            except BaseException as exc:  # noqa: BLE001 - propagate to consumer
                self.queue.put(("error", record, exc))
                return
            self.queue.put(("ok", record, path))
        self.queue.put(("done", None, None))

    def __iter__(self):
        while True:
            status, record, payload = self.queue.get()
            if status == "done":
                return
            if status == "error":
                raise RuntimeError(f"Tải {record.video_id} thất bại") from payload
            yield record, payload

    def close(self):
        self.stop_event.set()


def process_category(category: str):
    clips = CLIP_PLAN[category]
    completed = load_valid_remote_commits(category, clips)
    expected_shards = total_shards(len(clips))
    if len(completed) == expected_shards:
        print(f"{category}: đã hoàn tất {len(clips):,} clip / {expected_shards} shard; skip.")
        return {"category": category, "rows": len(clips), "shards": expected_shards, "skipped": True}

    rows_todo = pending_rows(len(clips), set(completed))
    todo = set(int(r) for r in rows_todo)
    video_by_id = {r.video_id: r for r in VIDEO_REGISTRY[category]}
    rows_by_video: Dict[str, List[int]] = {}
    for row in sorted(todo):
        rows_by_video.setdefault(clips[row].video_id, []).append(row)
    records = [r for r in VIDEO_REGISTRY[category] if r.video_id in rows_by_video]
    print(
        f"{category}: resume {len(completed)}/{expected_shards} shard; còn "
        f"{len(todo):,}/{len(clips):,} clip trên {len(records)} video."
    )

    buffers: Dict[int, Dict[str, list]] = {}
    buffered_rows: Dict[int, int] = {}
    new_commits = []
    progress = tqdm(total=len(todo), desc=f"Encode {category}", unit="clip", smoothing=0.05)
    prefetcher = VideoPrefetcher(records, depth=1)
    npz_path = NPZ_ROOT / f"{category}.npz"
    stats = {"clamped": 0, "encode_seconds": 0.0, "clips": 0,
             "substitutions": 0, "eof_trimmed": 0, "reopens": 0, "retried_videos": []}

    def encode_one_video(record, local_path, payload):
        """Gọi worker, thử lại video nếu decode lỗi giữa chừng.

        Lỗi decord thường bám theo state của reader hiện tại, nên `release_video`
        rồi encode lại từ đầu với reader mới là cách phục hồi rẻ nhất. File trên
        disk đã được verify SHA-256 nên không cần tải lại.
        """
        attempts = max(1, int(CONFIG["video_encode_attempts"]))
        for attempt in range(1, attempts + 1):
            try:
                return call_worker(
                    "encode_video", timeout=None, heartbeat=120,
                    video_path=str(local_path), video_id=record.video_id,
                    clips=payload, batch_size=BATCH_SIZE,
                    embedding_dim=CONFIG["embedding_dim"], out_npz=str(npz_path),
                )
            except RuntimeError as exc:
                if attempt >= attempts:
                    raise RuntimeError(
                        f"{record.video_id}: encode thất bại sau {attempts} lần thử."
                    ) from exc
                print(f"[retry] {record.video_id} lỗi ở lần {attempt}/{attempts}: "
                      f"{str(exc).splitlines()[0]}", flush=True)
                stats["retried_videos"].append(record.video_id)
                npz_path.unlink(missing_ok=True)
                call_worker("release_video", timeout=300)

    try:
        for record, local_path in prefetcher:
            rows = rows_by_video[record.video_id]
            payload = [
                {"row": int(r), "start_time": clips[r].start_time, "end_time": clips[r].end_time}
                for r in rows
            ]
            response = encode_one_video(record, local_path, payload)
            stats["clamped"] += int(response["clamped_clips"])
            stats["encode_seconds"] += float(response["elapsed_s"])
            stats["clips"] += int(response["rows"])
            stats["substitutions"] += int(response.get("frame_substitutions", 0))
            stats["eof_trimmed"] += int(bool(response.get("eof_trimmed_frames", 0)))
            stats["reopens"] += int(response.get("reader_reopens", 0))
            if response.get("eof_trimmed_frames"):
                print(f"[decode] {record.video_id}: decord báo "
                      f"{response['reported_frames']} frame nhưng chỉ đọc được "
                      f"{response['decodable_frames']}; window cuối đã trượt lùi.",
                      flush=True)
            if response.get("frame_substitutions"):
                print(f"[decode] {record.video_id}: "
                      f"{response['frame_substitutions']} frame phải thay bằng frame "
                      f"liền trước (lệch < {CONFIG['decode_substitution_window']} frame).",
                      flush=True)
            if abs(response["norm_min"] - 1.0) > 2e-5 or abs(response["norm_max"] - 1.0) > 2e-5:
                raise RuntimeError(f"{record.video_id}: vector không unit-norm.")

            with np.load(npz_path) as bundle:
                embeddings = bundle["embeddings"]
                frame_indices = bundle["frame_indices"]
                returned_rows = bundle["rows"]
            npz_path.unlink(missing_ok=True)
            if not np.array_equal(returned_rows, np.asarray(rows, dtype=np.int64)):
                raise RuntimeError(f"{record.video_id}: worker trả sai row index.")

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
                    commit = write_parquet_shard(
                        category, shard_id, clips, video_by_id,
                        np.concatenate(bucket["emb"], axis=0),
                        np.concatenate(bucket["fi"], axis=0),
                        np.concatenate(bucket["rows"], axis=0),
                    )
                    new_commits.append(commit)
                    buffered_rows.pop(shard_id)
                elif buffered_rows[shard_id] > stop - start:
                    raise RuntimeError(f"Buffer overflow ở {category} shard {shard_id}")

            progress.update(len(rows))
            call_worker("release_video", timeout=300)
            drop_video(record)
            del embeddings, frame_indices, returned_rows
    finally:
        progress.close()
        prefetcher.close()
        npz_path.unlink(missing_ok=True)

    if buffers or buffered_rows:
        raise RuntimeError(f"{category}: còn shard buffer chưa flush: {sorted(buffered_rows)}")

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
        "clamped_clips_this_run": stats["clamped"],
        "clips_this_run": stats["clips"],
        "clips_per_s_this_run": stats["clips"] / max(1e-6, stats["encode_seconds"]),
        "frame_substitutions_this_run": stats["substitutions"],
        "videos_eof_trimmed_this_run": stats["eof_trimmed"],
        "reader_reopens_this_run": stats["reopens"],
        "retried_videos_this_run": sorted(set(stats["retried_videos"])),
    }
    if CONFIG["upload"]:
        upload_bytes_verified(
            CONFIG["output_bucket_id"],
            f"{CONFIG['output_prefix'].strip('/')}/success/{category}.json",
            json.dumps(success, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    print(json.dumps(success, indent=2))
    return success


CATEGORY_RESULTS = []
for category in CONFIG["categories"]:
    CATEGORY_RESULTS.append(process_category(category))

# %% [markdown]
# ## 10. Global audit và success manifest
#
# Mỗi runtime ghi audit riêng vào `audits/`. Global manifest và `_SUCCESS.json`
# chỉ được ghi khi **cả L21..L30** đủ commit; `global_complete=false` ở session về
# sớm là bình thường, không phải lỗi.

# %%
def final_global_audit():
    per_category = {}
    total_rows = 0
    all_complete = True
    for category in sorted(EXPECTED_VIDEO_COUNTS):
        clips = CLIP_PLAN[category]
        commits = load_valid_remote_commits(category, clips) if CONFIG["upload"] else {}
        expected = total_shards(len(clips))
        committed_rows = sum(int(c["rows"]) for c in commits.values())
        complete = len(commits) == expected and committed_rows == len(clips)
        per_category[category] = {
            "videos": EXPECTED_VIDEO_COUNTS[category],
            "expected_rows": len(clips),
            "committed_rows": committed_rows,
            "expected_shards": expected,
            "committed_shards": len(commits),
            "complete": complete,
        }
        total_rows += committed_rows
        all_complete &= complete

    manifest = {
        "schema_version": 1,
        "audited_at": utc_now(),
        "semantic_fingerprint": SEMANTIC_FINGERPRINT,
        "semantic_config": SEMANTIC_CONFIG,
        "per_category": per_category,
        "total_committed_rows": total_rows,
        "expected_total_rows": PLAN_TOTAL,
        "expected_total_videos": sum(EXPECTED_VIDEO_COUNTS.values()),
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
# Nếu `d_time ≈ 0` thì embedding chỉ là bag-of-frames và toàn bộ luận điểm dùng
# TARA cho đợt 3 sụp đổ — test này fail cứng ở trường hợp đó. Tỷ lệ
# `d_time / d_content` cho biết thông tin thứ tự chiếm bao nhiêu so với nội dung.
#
# **C. Temporal pairs của bạn.** Bảng `TEMPORAL_PAIRS` để trống sẵn — điền
# `video_id / start_time / end_time` cùng caption xuôi và caption đảo thứ tự (dịch
# từ query đợt 3), cell sẽ đo `Acc = mean(S₊ > S₋)` và `Δ = mean(S₊ − S₋)`. **Đây
# mới là con số quyết định có nên index toàn corpus hay không** — chạy trên 20–50
# cặp trước khi commit 40 GPU-giờ.

# %%
# ---------------------------------------------------------------------------
# Điền bảng này bằng ground truth của bạn rồi chạy lại riêng cell này.
# forward  = caption mô tả ĐÚNG thứ tự hành động trong clip
# reversed = ĐÚNG những hành động đó nhưng đảo thứ tự (hard negative)
# Dịch sang tiếng Anh: TARA fine-tune trên NLI-Nuance tiếng Anh.
# ---------------------------------------------------------------------------
TEMPORAL_PAIRS = [
    # {
    #     "video_id": "L21_V001", "start_time": 120.0, "end_time": 144.0,
    #     "forward":  "a chef dips slices of meat into boiling broth, puts them in a bowl, "
    #                 "then ladles soup over them",
    #     "reversed": "a chef ladles soup into a bowl, then dips slices of meat "
    #                 "into boiling broth",
    # },
]

SELF_TEST = {}


def encode_texts(texts):
    out_path = NPZ_ROOT / "self_test_text.npz"
    call_worker("encode_text", timeout=1800, heartbeat=60,
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
    call_worker("encode_video", timeout=None, heartbeat=120,
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
        probe = call_worker("probe_video", timeout=600, video_path=str(demo_path))
        demo_vector = encode_clips(demo_path, [(0.0, probe["duration_sec"])])[0]
        call_worker("release_video", timeout=300)
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
        (r for c in CONFIG["categories"] for r in VIDEO_REGISTRY[c]),
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
    call_worker("release_video", timeout=300)
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
        raise RuntimeError(
            f"Test B FAIL: đảo ngược thứ tự frame gần như không đổi embedding "
            f"(median d_time = {median_time:.2e}). Embedding đang là bag-of-frames, "
            "index theo clip sẽ không giải quyết được failure mode đợt 3."
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
            call_worker("release_video", timeout=300)
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
    WORKER.close()
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
# | Pooling | hidden state cuối cùng của token cuối |
# | Vector | 3584-d FP32 unit-norm |
# | Metric | COSINE — Milvus `FLOAT_VECTOR(3584)` |
#
# ### Query side
#
# ```python
# # Trong venv TARA (Python 3.10)
# import torch, torch.nn.functional as F
# from modeling_tara import TARA
#
# model = TARA.from_pretrained(SNAPSHOT_DIR, device_map={"": 0},
#                              attn_implementation="flash_attention_2")
# z = model.encode_text("a chef dips meat into broth, then ladles soup over it")
# z = F.normalize(z.float(), p=2, dim=-1)   # [1, 3584]
# ```
#
# TARA fine-tune trên NLI-Nuance **tiếng Anh**, nên dịch query VI→EN trước khi
# encode. Query instruction không nằm trong vector side nên có thể A/B test tự do
# mà **không** phải encode lại clip.
#
# ### Layout output
#
# ```text
# derived/tara-tarsier2-7b-3584-clip-v1/
# ├── run_config.json
# ├── executions/*.json
# ├── audits/*.json
# ├── embeddings/Lxx/part-*.parquet
# ├── commits/Lxx/part-*.json
# ├── success/Lxx.json
# ├── embedding_dataset_manifest.json
# └── _SUCCESS.json                 # chỉ có khi đủ 168.536 clip
# ```
#
# Schema mỗi row:
#
# ```text
# clip_id        string            L21_V001@event@t000012000
# video_id       string            L21_V001
# category       string            L21
# scale          string            event | sequence | scene
# scale_index    int8              0 | 1 | 2
# start_time     float64           giây
# end_time       float64           giây
# fps            float32
# duration_sec   float64
# video_relpath  string            Videos/Videos_L21/L21_V001.mp4
# frame_indices  int32[8]          frame index thật đã đưa vào model
# embedding      float32[3584]     unit-norm
# ```
#
# `start_time`/`end_time` là **identity của plan** (quyết định row index và shard).
# `frame_indices` mới là **ground truth** của những frame model thực sự nhìn thấy;
# cửa sổ thật là `frame_indices[0]/fps .. frame_indices[-1]/fps`. Hai giá trị này
# trùng nhau ở mọi clip trừ clip cuối của một video khi `duration_sec` trong
# manifest dài hơn số frame decoder đọc được — khi đó worker **trượt window lùi
# lại** để giữ nguyên độ dài window (~124/168.536 clip, lệch dưới một giây), và
# `clamped_clips_this_run` trong `success/Lxx.json` đếm đúng số đó. Khi cần định vị
# thời điểm (TRAKE), hãy dùng `frame_indices`.
#
# ### Dùng thế nào cho KIS và TRAKE
#
# **KIS sequence.** Chấm điểm video bằng cả hai đường rồi fuse:
#
# ```text
#                       Q
#                       │
#         ┌─────────────┴──────────────┐
#         │                            │
#  full sequence query          event decomposition
#         │                            │
#  TARA vs clip scale            E1 E2 E3 E4
#  sequence/scene                vs clip scale event
#         │                            │
#         └────────────┬───────────────┘
#                      ↓
#       S(v) = α·S_whole-sequence + (1-α)·S_event-chain
#                      ↓
#              RRF với PE / Qwen / OCR / ASR
# ```
#
# `scale` là cột lọc: query mô tả một thao tác đơn thì ưu tiên `scale='event'`,
# query mô tả chuỗi 3–4 bước thì `scale='sequence'`.
#
# **TRAKE.** Clip embedding **không** thay được thuật toán DP hiện có — nó chỉ
# thay phần *candidate generation*:
#
# ```text
# query → event decomposition E1..Ek
#       → với mỗi Ei: top-N clip ở scale 'event'  (đã có start_time/end_time)
#       → temporal DP hiện tại trên t1 < t2 < ... < tk
#       → best ordered sequence
# ```
#
# Vì mỗi row đã có `start_time`, `end_time` và `frame_indices`, việc map một clip
# về keyframe gần nhất chỉ là join `video_id` + `frame_idx` với registry keyframe.
#
# ### Fusion
#
# Không cộng thẳng cosine của TARA với PE-Core hay Qwen3-VL-Embedding: ba
# embedding space có score distribution khác nhau. Fuse bằng rank/RRF. Paper TARA
# cũng report ensemble TARA ⊕ Qwen3-VL-Embedding cải thiện Qwen standalone
# (MSR-VTT 53,8 → 54,5; MSVD 87,2 → 88,4; VATEX 64,8 → 66,2), tức là hai model bổ
# sung nhau chứ không trùng nhau.
#
# ### Trước khi index cả corpus
#
# Nếu muốn kiểm chứng rẻ trước, đặt `session_assignments` cho một session chỉ chứa
# `L23` (25 video, 3.471 clip, ~1,9 GiB) hoặc `L27` (16 video, 3.383 clip), chạy
# xong rồi đo `Acc_order` trên positive vs reversed-order hard negative bằng đúng
# query đợt 3. Nếu TARA không thắng ở benchmark đó thì đừng index phần còn lại.
