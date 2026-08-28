# %% [markdown]
# # AIC 2026 — Qwen3-VL-Embedding-8B image embedding (Colab Latest A100)
#
# Notebook E2E này encode **1.339.055 keyframe** của `infoshootpp-v1` bằng
# `Qwen/Qwen3-VL-Embedding-8B`, ghi vector native 4096-d đã L2-normalize thành
# các Parquet shard và upload checkpoint-safe lên Hugging Face Storage Bucket.
#
# Thiết kế chính:
#
# - Image-side instruction được khóa đúng official default:
#   `Represent the user's input.` + image-only user message.
# - Official `qwen-vl-utils` smart-resize giữ aspect ratio, `min_pixels=4.096`,
#   `max_pixels=1.843.200`; không resize cứng 448/512 và không center-crop.
# - Official chat template + generation prompt, forward toàn bộ multimodal 8B,
#   pooling tại last attended token. Không lấy riêng vision tower/CLS/mean pool.
# - BF16 weights + FlashAttention-2 trên A100. Colab Latest dùng Python 3.13,
#   Torch 2.11/CUDA 12 và build FA2 2.8.3 từ source chính thức có log quan sát được.
# - Collection v3 dùng legacy semantic reference 2.8.3.post1. Upstream chỉ đổi
#   packaging/CUDA-13 wheel selection giữa hai tag; runtime version thật luôn
#   được ghi trong execution log.
# - Pooling, MRL selection và L2
#   normalization trong FP32; lưu master vector native 4096-d FP32.
# - DataLoader worker làm official preprocessing, pixel tensor được pre-cast BF16
#   sau preprocessing để giảm IPC/H2D; startup parity kiểm lại với wrapper official.
# - Tự benchmark batch trên các frame có visual-token budget lớn cho A100 40/80 GB.
# - TAR được stream thẳng từ Bucket vào `tar`, đồng thời kiểm SHA-256; không lưu
#   thêm bản TAR 80 GB trên local disk.
# - Resume theo shard 2.048 ảnh. Parquet được upload và verify size trước khi commit
#   JSON được tạo; chỉ commit hợp lệ mới được xem là hoàn tất.
#
# > Mỗi Colab runtime chỉ đổi `active_session` rồi chạy từ trên xuống. Các session
# > dùng chung `output_prefix` nhưng encode category rời nhau; shard đã commit được skip.
#
# Official references:
#
# - https://github.com/QwenLM/Qwen3-VL-Embedding
# - https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B

# %%
# =========================
# CONFIG — chỉ sửa cell này
# =========================

CONFIG = {
    # Corpus nguồn (không bị notebook sửa/xóa).
    "source_bucket_id": "Baonenha1/DATA-AIC-Keyframe",
    "source_prefix": "infoshootpp-v1",

    # Derived output. Mặc định dùng bucket hiện tại nhưng prefix tách biệt hoàn toàn.
    "output_bucket_id": "Baonenha1/DATA-AIC-Keyframe",
    "output_prefix": "derived/qwen3-vl-embedding-8b-4096-v3",

    # Chạy song song: tạo ba bản Colab của notebook và mỗi bản chỉ đổi đúng dòng
    # active_session. Assignment phải rời nhau để không có hai runtime ghi cùng shard.
    "active_session": "session_1",  # session_1 | session_2 | session_3
    "session_assignments": {
        "session_1": ["L21", "L22", "L23", "L24"],
        "session_2": ["L26"],
        "session_3": ["L25", "L27", "L28", "L29", "L30"],
    },
    "expected_counts": {
        "L21": 111_031,
        "L22": 130_134,
        "L23": 40_808,
        "L24": 69_826,
        "L25": 154_211,
        "L26": 559_060,
        "L27": 33_201,
        "L28": 90_042,
        "L29": 83_697,
        "L30": 67_045,
    },

    # Model + official implementation được pin để mọi session cùng embedding space.
    "model_id": "Qwen/Qwen3-VL-Embedding-8B",
    "model_revision": "2c4565515e0f265c6511776e7193b22c0968ddc7",
    "qwen_git_ref": "393e2978d27852b0d0230d6994f37f9c15bed73c",
    "transformers_version": "5.14.1",
    "qwen_vl_utils_version": "0.0.14",
    # Colab Latest quan sát ngày 2026-08-27: Python 3.13, Torch 2.11, CUDA 12.8.
    # Upstream chưa có prebuilt FA2 wheel cho Torch 2.11 nên build source.
    "target_colab_runtime": "latest",
    "target_python_major_minor": "3.13",
    "target_torch_major_minor": "2.11",
    "target_torch_cuda_major": 12,
    "flash_attn_version": "2.8.3",
    "flash_attn_build_max_jobs": 4,
    "flash_attn_build_nvcc_threads": 1,
    "flash_attn_heartbeat_seconds": 15,
    "flash_attn_silence_warning_seconds": 120,
    # Cache wheel đã tự build trên Google Drive để các Colab session cùng contract
    # dùng lại. Wheel chỉ được restore sau khi verify manifest + SHA-256.
    "flash_attn_drive_cache": True,
    "flash_attn_drive_mount_point": "/content/drive",
    "flash_attn_drive_cache_relative": "MyDrive/AIC2026/flash_attn_wheels",
    # Chỉ dùng để tái tạo đúng fingerprint v3 đã được khởi tạo bằng post1.
    # Không dùng giá trị này để cài/import package.
    "legacy_semantic_flash_attn_version": "2.8.3.post1",
    "expected_legacy_semantic_fingerprint": (
        "8047495b9ffdd325610b7ea8ecefb15389564632a7e2e20c23ae6995e1454ac3"
    ),
    "require_existing_legacy_run_config": True,
    "pillow_version": "11.3.0",  # Pin để tránh Pillow 12 gây lỗi _Ink.
    "attention_implementation": "flash_attention_2",
    "compute_precision": "bf16",
    "model_weight_dtype": "bfloat16",
    "storage_dtype": "float32",
    "embedding_dim": 4096,
    "mrl_dimension": None,  # None = native 4096; không giảm thời gian forward nếu cắt MRL.
    "l2_normalize_fp32": True,

    # Official semantic image-side contract. Không đổi giữa các lần resume.
    "image_instruction": "Represent the user's input.",
    "max_length": 8192,
    "image_patch_size": 16,
    "min_pixels": 4_096,
    "max_pixels": 1_843_200,
    "precast_pixel_values_bf16": True,

    # Throughput. Batch candidates được test trên các frame nặng; chọn theo img/s
    # trong ngưỡng VRAM. Worker chạy official processor trên CPU.
    "batch_size": 32,
    "batch_candidates": None,
    "num_workers": 4,
    "prefetch_factor": 1,
    "persistent_workers": False,
    "max_vram_fraction": 0.94,

    # Check pipeline production so với official wrapper và batch=1 BF16 reference.
    # Với Qwen3-VL + FA2, batch=1 và batch>1 có thể đi qua kernel shape khác nhau;
    # batch=1 chỉ là mốc ổn định số học, không phải embedding "chất lượng hơn".
    # Checkpoint gốc vốn là BF16; ép full FP32 cho 8B không phục hồi precision weight.
    "preprocess_parity_samples": 8,
    "numeric_audit_samples": 16,
    "numeric_audit_min_cosine": 0.999,
    "numeric_audit_mean_cosine": 0.9995,
    # So trực tiếp vector Torch 2.11/FA2 2.8.3 với một shard v3 cũ trước khi ghi.
    "legacy_cross_runtime_audit_samples": 16,
    "legacy_cross_runtime_min_cosine": 0.999,
    "legacy_cross_runtime_mean_cosine": 0.9995,

    # 2.048 × 4096 × 4 bytes ≈ 32 MiB vector/shard. Shard nhỏ giảm lượng encode
    # phải chạy lại nếu Colab ngắt giữa chừng.
    "shard_rows": 2_048,
    "upload": True,
    "cleanup_images_after_category": True,
    "cleanup_local_shards_after_upload": True,
    "cleanup_model_cache_after_load": True,
    "strict_file_audit": True,

    # Colab scratch. Không đặt vào Google Drive: hàng triệu JPEG nhỏ sẽ rất chậm.
    "scratch_root": "/content/aic_qwen3_vl_embedding_8b",
    "hf_token_file_candidates": [
        "/content/HF_TOKEN.txt",
        "/content/drive/MyDrive/HF_TOKEN.txt",
    ],
    "fail_on_non_a100": True,
}

ACTIVE_SESSION = CONFIG["active_session"]
if ACTIVE_SESSION not in CONFIG["session_assignments"]:
    raise ValueError(f"active_session không tồn tại: {ACTIVE_SESSION}")
CONFIG["categories"] = list(CONFIG["session_assignments"][ACTIVE_SESSION])
CONFIG["session_name"] = f"colab_{ACTIVE_SESSION}"
print("Active session:", CONFIG["session_name"], "categories:", CONFIG["categories"])


# Source-of-truth checksums lấy từ manifest phân phối `infoshootpp-v1`.
SOURCE_SHARDS = {
    "L21": {"bytes": 18_262_026_240, "sha256": "9e641def2d2da0e75cd45f0d6e686c07a91b1f0d483832960cce0f79bc734146"},
    "L22": {"bytes": 21_711_011_840, "sha256": "4fd4a5bdbfe927ebfa7e3d964fb2cf3c10f34fef4a4262de69aa8c6b20a0c390"},
    "L23": {"bytes": 7_851_427_840,  "sha256": "0316081ce228d9e4c8c8bc84e6dcd47a4106fc38f670a13c0deee2d2a0717084"},
    "L24": {"bytes": 15_279_052_800, "sha256": "c71e663d346b062803a34981e909f367d67f0d51496820ceb518afa903a9d28e"},
    "L25": {"bytes": 23_876_280_320, "sha256": "ce8afc5a0dbf9700d0545c64c10371e6366c60ecf327adae9fafbabdba421a4b"},
    "L26": {"bytes": 79_997_450_240, "sha256": "82ed494c26248472856c134fcd37b982f1ea9fc500f37534f77297fa492d67af"},
    "L27": {"bytes": 6_732_646_400,  "sha256": "4c4b40b82e0c9e6a9beddc511445514d039a4f24ffa407b14bc981ea563db6fd"},
    "L28": {"bytes": 17_562_224_640, "sha256": "a2c27d67e139482ecbe4b9df0ef7aec8ed515a74cccfa0528eeffcfed6aaaef9"},
    "L29": {"bytes": 17_598_791_680, "sha256": "3c7d9bdeb8e5878b8aacaea009e3391a1de252cc56b076ea3e99a2f60d050808"},
    "L30": {"bytes": 10_603_458_560, "sha256": "597d0db9a1da4ec321a0c1534c7fa75a161d81057c6a777516f8cad05ea84992"},
}

SOURCE_METADATA = {
    "bytes": 888_432_640,
    "sha256": "65f99d79320d1e2d5d7222379e5117d96b1d5d1163b58a2462433fdd1c11c857",
    "run_manifest_sha256": "11b5a0bfaf805f4a615ed5c77d796b30c2214783ccd4f59b9fa6123366981d9e",
    "run_summary_sha256": "8ccd83d2113ad6f9d5e4a1b29000ed83ec7476edbdbc0fa327691cbe224d9a93",
}


# %% [markdown]
# ## 1. Cài dependency và checkout implementation Qwen chính thức
#
# Giữ PyTorch CUDA có sẵn của Colab. Pin Transformers/qwen-vl-utils theo semantic
# runtime và cài FlashAttention-2 chính thức để giảm memory/latency trên A100.

# %%
from pathlib import Path
from collections import deque
import hashlib
import importlib.metadata
import json
import os
import platform
import queue
import shlex
import shutil
import subprocess
import sys
import sysconfig
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
    silence_warning_seconds=120,
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
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
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
    tracked_processes = {}

    def process_tree_stats():
        try:
            import psutil

            root = psutil.Process(process.pid)
            processes = [root, *root.children(recursive=True)]
            cpu_percent = 0.0
            rss_bytes = 0
            live_children = 0
            for child in processes:
                try:
                    cached = tracked_processes.get(child.pid)
                    if cached is None:
                        tracked_processes[child.pid] = child
                        child.cpu_percent(interval=None)
                    else:
                        cpu_percent += cached.cpu_percent(interval=None)
                    rss_bytes += child.memory_info().rss
                    live_children += child.pid != process.pid
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return {
                "cpu_percent": cpu_percent,
                "system_cpu_percent": psutil.cpu_percent(interval=None),
                "rss_bytes": rss_bytes,
                "children": live_children,
                "available_ram_bytes": psutil.virtual_memory().available,
            }
        except Exception:
            return None

    try:
        while True:
            now = time.monotonic()
            wait_seconds = max(0.05, min(1.0, next_heartbeat - now))
            try:
                item = output_queue.get(timeout=wait_seconds)
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
                elapsed = now - started
                silent_for = now - last_output
                stats = process_tree_stats()
                detail = ""
                if stats is not None:
                    detail = (
                        f" | cpu_tree={stats['cpu_percent']:.1f}%"
                        f" | cpu_system={stats['system_cpu_percent']:.1f}%"
                        f" | rss_tree={_human_bytes(stats['rss_bytes'])}"
                        f" | children={stats['children']}"
                        f" | ram_available={_human_bytes(stats['available_ram_bytes'])}"
                    )
                    if stats["cpu_percent"] >= 1.0:
                        state = "alive/CPU active"
                    elif stats["children"] > 0:
                        state = "alive/có child process; có thể đang compile hoặc chờ I/O"
                    else:
                        state = "alive/đang chờ; chưa thấy CPU activity"
                else:
                    state = "alive/không đọc được process metrics"
                if silent_for >= silence_warning_seconds:
                    state += "; log đã im lặng lâu"
                emit(
                    f"[{phase} HEARTBEAT] pid={process.pid} | state={state}"
                    f" | elapsed={_elapsed_text(elapsed)}"
                    f" | last_output={_elapsed_text(silent_for)} ago{detail}"
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
        emit(
            f"[{phase} ABORTED] pid={process.pid} | "
            f"elapsed={_elapsed_text(time.monotonic() - started)}"
        )
        if log_file is not None:
            log_file.close()
        raise
    finally:
        reader.join(timeout=2)

    return_code = process.wait()
    elapsed = time.monotonic() - started
    status = "DONE" if return_code == 0 else "FAILED"
    emit(
        f"[{phase} {status}] return_code={return_code} | "
        f"elapsed={_elapsed_text(elapsed)}"
    )
    if log_file is not None:
        log_file.close()
    if check and return_code != 0:
        tail_text = "\n".join(output_tail)
        raise RuntimeError(
            f"{phase} thất bại (return_code={return_code}). "
            f"Full log: {log_path}\n--- last output ---\n{tail_text}"
        )
    return return_code


def _wheel_sha256(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload):
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mount_flash_attn_drive_cache():
    """Mount Drive nếu cần; lỗi cache không được chặn fallback source build."""
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
            print(
                f"[FA2 DRIVE CACHE] Mount Google Drive tại {mount_point}...",
                flush=True,
            )
            from google.colab import drive

            drive.mount(str(mount_point), force_remount=False)
        if not my_drive.is_dir():
            raise RuntimeError(f"Drive mount thiếu directory: {my_drive}")
        cache_root = mount_point / relative
        cache_root.mkdir(parents=True, exist_ok=True)
        print(f"[FA2 DRIVE CACHE] root={cache_root}", flush=True)
        return cache_root
    except Exception as exc:
        print(
            f"[FA2 DRIVE CACHE WARNING] Không dùng được Drive cache: "
            f"{type(exc).__name__}: {exc}. Sẽ fallback local/source build.",
            flush=True,
        )
        return None


def restore_flash_attn_wheel_from_drive(cache_dir, contract, local_wheel_dir):
    """Copy cache về local và verify contract, size, SHA trước khi trả wheel."""
    if cache_dir is None:
        return None
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"[FA2 DRIVE CACHE MISS] Chưa có {manifest_path}", flush=True)
        return None

    partial_path = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise RuntimeError("manifest schema_version không hỗ trợ")
        if manifest.get("contract") != contract:
            raise RuntimeError("runtime/build contract mismatch")
        if manifest.get("contract_sha256") != _canonical_sha256(contract):
            raise RuntimeError("contract SHA-256 mismatch")

        wheel_name = manifest.get("wheel_filename")
        if not isinstance(wheel_name, str) or Path(wheel_name).name != wheel_name:
            raise RuntimeError("wheel_filename không an toàn")
        drive_wheel = cache_dir / wheel_name
        if not drive_wheel.is_file():
            raise RuntimeError(f"thiếu cached wheel {drive_wheel}")

        local_wheel_dir = Path(local_wheel_dir)
        local_wheel_dir.mkdir(parents=True, exist_ok=True)
        local_wheel = local_wheel_dir / wheel_name
        partial_path = local_wheel_dir / (
            f".{wheel_name}.drive-{os.getpid()}-{time.time_ns()}.partial"
        )
        print(
            f"[FA2 DRIVE CACHE COPY] {drive_wheel} -> {local_wheel}",
            flush=True,
        )
        shutil.copyfile(drive_wheel, partial_path)
        actual_bytes = partial_path.stat().st_size
        actual_sha256 = _wheel_sha256(partial_path)
        if actual_bytes != int(manifest.get("wheel_bytes", -1)):
            raise RuntimeError(
                f"wheel size mismatch: {actual_bytes} != {manifest.get('wheel_bytes')}"
            )
        if actual_sha256 != manifest.get("wheel_sha256"):
            raise RuntimeError("wheel SHA-256 mismatch")
        os.replace(partial_path, local_wheel)
        partial_path = None
        print({
            "fa2_drive_cache": "HIT_VERIFIED",
            "wheel": str(local_wheel),
            "bytes": actual_bytes,
            "sha256": actual_sha256,
        }, flush=True)
        return local_wheel
    except Exception as exc:
        print(
            f"[FA2 DRIVE CACHE INVALID] {type(exc).__name__}: {exc}. "
            "Bỏ cache và fallback source build.",
            flush=True,
        )
        return None
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)


def publish_flash_attn_wheel_to_drive(wheel_path, cache_dir, contract):
    """Publish wheel atomically; manifest được ghi cuối như commit marker."""
    if cache_dir is None:
        return None
    wheel_path = Path(wheel_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wheel_bytes = wheel_path.stat().st_size
    wheel_sha256 = _wheel_sha256(wheel_path)
    contract_sha256 = _canonical_sha256(contract)
    destination = cache_dir / wheel_path.name
    manifest_path = cache_dir / "manifest.json"

    # Manifest hợp lệ đã commit thì không copy lại một artifact lớn lên Drive.
    try:
        if manifest_path.is_file() and destination.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                existing.get("schema_version") == 1
                and existing.get("contract") == contract
                and existing.get("contract_sha256") == contract_sha256
                and existing.get("wheel_filename") == wheel_path.name
                and int(existing.get("wheel_bytes", -1)) == wheel_bytes
                and existing.get("wheel_sha256") == wheel_sha256
                and destination.stat().st_size == wheel_bytes
                and _wheel_sha256(destination) == wheel_sha256
            ):
                print(
                    f"[FA2 DRIVE CACHE] Artifact đã được commit: {destination}",
                    flush=True,
                )
                return manifest_path
    except Exception as exc:
        print(
            f"[FA2 DRIVE CACHE] Existing manifest không dùng được: {exc}; ghi lại.",
            flush=True,
        )

    unique = f"{os.getpid()}-{time.time_ns()}"
    wheel_partial = cache_dir / f".{wheel_path.name}.{unique}.partial"
    manifest_partial = cache_dir / f".manifest.{unique}.partial"
    try:
        print(
            f"[FA2 DRIVE CACHE SAVE] {wheel_path} -> {destination}",
            flush=True,
        )
        shutil.copyfile(wheel_path, wheel_partial)
        copied_bytes = wheel_partial.stat().st_size
        copied_sha256 = _wheel_sha256(wheel_partial)
        if copied_bytes != wheel_bytes or copied_sha256 != wheel_sha256:
            raise RuntimeError("Drive copy verification thất bại")
        os.replace(wheel_partial, destination)

        manifest = {
            "schema_version": 1,
            "artifact": "flash-attn-source-built-wheel",
            "source": f"PyPI sdist flash-attn=={CONFIG['flash_attn_version']}",
            "created_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "session_name": CONFIG["session_name"],
            "contract": contract,
            "contract_sha256": contract_sha256,
            "wheel_filename": wheel_path.name,
            "wheel_bytes": wheel_bytes,
            "wheel_sha256": wheel_sha256,
        }
        manifest_partial.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_partial, manifest_path)
        print({
            "fa2_drive_cache": "SAVED_VERIFIED",
            "cache_dir": str(cache_dir),
            "wheel": wheel_path.name,
            "bytes": wheel_bytes,
            "sha256": wheel_sha256,
        }, flush=True)
        return manifest_path
    except Exception as exc:
        print(
            f"[FA2 DRIVE CACHE WARNING] Không save được wheel: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None
    finally:
        wheel_partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)


MINIMAL_PACKAGES = [
    f"transformers=={CONFIG['transformers_version']}",
    "accelerate==1.14.0",
    "huggingface_hub==1.24.0",
    f"qwen-vl-utils=={CONFIG['qwen_vl_utils_version']}",
    "pyarrow>=18,<25",
    "pandas>=2.2,<3",
    f"pillow=={CONFIG['pillow_version']}",
    "safetensors>=0.5,<1",
    "ninja>=1.11,<2",
    "setuptools>=80.9,<82",
    "wheel>=0.45,<1",
    "packaging>=24,<27",
    "psutil>=6,<8",
    "tqdm>=4.67,<5",
    "einops>=0.8,<1",
]
dependency_stamp = time.strftime("%Y%m%d_%H%M%S")
run_observable([
    sys.executable, "-u", "-m", "pip", "install", "--upgrade",
    "--progress-bar", "on", *MINIMAL_PACKAGES,
],
    phase="DEPENDENCY INSTALL",
    heartbeat_seconds=CONFIG["flash_attn_heartbeat_seconds"],
    silence_warning_seconds=CONFIG["flash_attn_silence_warning_seconds"],
    log_path=(
        f"/content/flash_attn_build_logs/dependencies_{dependency_stamp}.log"
    ),
)

# Fail sớm nếu Pillow bị trộn file giữa version cũ/mới trong một kernel đã chạy.
try:
    from PIL import Image as _PIL_IMAGE_CHECK
    from PIL import ImageDraw as _PIL_DRAW_CHECK
except Exception as exc:
    raise RuntimeError(
        "Pillow runtime không nhất quán. Hãy Terminate runtime (không chỉ Restart), "
        "chọn Latest (recommended) rồi Run All lại bằng notebook mới."
    ) from exc
if importlib.metadata.version("Pillow") != CONFIG["pillow_version"]:
    raise RuntimeError("Pillow version không khớp config.")
print("Pillow import/version OK:", importlib.metadata.version("Pillow"))

QWEN_REPO_DIR = Path("/content/Qwen3-VL-Embedding")
if not (QWEN_REPO_DIR / ".git").exists():
    run_checked([
        "git", "clone", "--filter=blob:none", "--no-tags",
        "https://github.com/QwenLM/Qwen3-VL-Embedding.git",
        str(QWEN_REPO_DIR),
    ])

run_checked([
    "git", "-C", QWEN_REPO_DIR, "remote", "set-url", "origin",
    "https://github.com/QwenLM/Qwen3-VL-Embedding.git",
])
run_checked([
    "git", "-C", QWEN_REPO_DIR, "fetch", "--depth", "1", "origin",
    CONFIG["qwen_git_ref"],
])
run_checked(["git", "-C", QWEN_REPO_DIR, "checkout", "--detach", "FETCH_HEAD"])

# Colab Latest/Torch 2.11 chưa có exact upstream wheel. Build source upstream
# v2.8.3, stream từng dòng stdout/stderr và heartbeat để không nhầm build với idle.
import torch
from packaging.version import Version


def flash_attn_status():
    try:
        installed = importlib.metadata.version("flash-attn")
        if installed != CONFIG["flash_attn_version"]:
            return False, f"version hiện tại là {installed}"
        import flash_attn
        from flash_attn import flash_attn_func  # noqa: F401
        return True, f"{installed} tại {flash_attn.__file__}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


flash_ready, flash_status = flash_attn_status()
print("FA2 initial status:", flash_status)
flash_version = CONFIG["flash_attn_version"]
torch_version = Version(torch.__version__.split("+")[0])
python_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
torch_major_minor = f"{torch_version.major}.{torch_version.minor}"
if not torch.version.cuda:
    raise RuntimeError("PyTorch runtime không có CUDA version.")
torch_cuda_major = int(torch.version.cuda.split(".")[0])
runtime_contract = {
    "target_colab_runtime": CONFIG["target_colab_runtime"],
    "expected_python": CONFIG["target_python_major_minor"],
    "python": platform.python_version(),
    "expected_torch": CONFIG["target_torch_major_minor"],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
}
print("FA2 runtime contract:", runtime_contract)
if python_major_minor != CONFIG["target_python_major_minor"]:
    raise RuntimeError(f"Sai Python cho Colab Latest: {runtime_contract}")
if torch_major_minor != CONFIG["target_torch_major_minor"]:
    raise RuntimeError(f"Sai PyTorch cho Colab Latest: {runtime_contract}")
if torch_cuda_major != int(CONFIG["target_torch_cuda_major"]):
    raise RuntimeError(f"Sai CUDA major cho Colab Latest: {runtime_contract}")
if not torch.cuda.is_available():
    raise RuntimeError("FA2 yêu cầu CUDA GPU đang hoạt động.")

gpu_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
if capability[0] == 8:
    flash_arch = "80"
elif capability[0] == 9:
    flash_arch = "90"
else:
    raise RuntimeError(
        f"FA2 2.8.3 notebook này chỉ hỗ trợ Ampere/Ada/Hopper; "
        f"GPU={gpu_name}, capability={capability}."
    )

import psutil

available_ram_gib = psutil.virtual_memory().available / 2**30
cpu_limit = max(1, (os.cpu_count() or 1) // 2)
nvcc_threads = max(1, int(CONFIG["flash_attn_build_nvcc_threads"]))
# Upstream ước lượng khoảng 5 GiB cho mỗi MAX_JOBS × NVCC_THREADS.
ram_limit = max(1, int(max(5.0, available_ram_gib - 4.0) / (5 * nvcc_threads)))
max_jobs = max(
    1,
    min(int(CONFIG["flash_attn_build_max_jobs"]), cpu_limit, ram_limit),
)
python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
cuda_tag = torch.version.cuda.replace(".", "")
cxx11_abi = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
abi_tag = str(cxx11_abi).upper()
wheel_contract = {
    "schema_version": 1,
    "flash_attn_version": flash_version,
    "python_version": platform.python_version(),
    "python_tag": python_tag,
    "python_soabi": sysconfig.get_config_var("SOABI"),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cxx11_abi": cxx11_abi,
    "platform_system": platform.system(),
    "platform_machine": platform.machine(),
    "compute_capability": list(capability),
    "flash_arch": flash_arch,
}
wheel_contract_sha256 = _canonical_sha256(wheel_contract)
wheel_dir = Path(
    f"/content/flash_attn_wheels/torch{torch_major_minor}-"
    f"{python_tag}-cu{cuda_tag}-sm{flash_arch}"
)
wheel_dir.mkdir(parents=True, exist_ok=True)

drive_cache_root = mount_flash_attn_drive_cache()
drive_cache_dir = None
if drive_cache_root is not None:
    drive_cache_name = (
        f"flash-attn-{flash_version}_{python_tag}_torch-{torch_major_minor}_"
        f"cu{cuda_tag}_abi{abi_tag}_sm{flash_arch}_"
        f"{wheel_contract_sha256[:12]}"
    )
    drive_cache_dir = drive_cache_root / drive_cache_name
print({
    "fa2_wheel_contract": wheel_contract,
    "contract_sha256": wheel_contract_sha256,
    "drive_cache_dir": str(drive_cache_dir) if drive_cache_dir else None,
}, flush=True)

cached_wheels = sorted(
    wheel_dir.glob(f"flash_attn-{flash_version}-*.whl"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
wheel_path = cached_wheels[0] if cached_wheels else None
wheel_origin = "local_cache" if wheel_path is not None else None
if wheel_path is not None:
    print(f"[FA2 LOCAL CACHE HIT] {wheel_path}", flush=True)

# Chỉ restore artifact lớn khi package hiện tại chưa import được.
if not flash_ready and wheel_path is None:
    wheel_path = restore_flash_attn_wheel_from_drive(
        drive_cache_dir, wheel_contract, wheel_dir
    )
    if wheel_path is not None:
        wheel_origin = "drive_cache"

build_stamp = time.strftime("%Y%m%d_%H%M%S")
log_dir = Path("/content/flash_attn_build_logs")
build_log = log_dir / f"fa2_build_{build_stamp}.log"
install_log = log_dir / f"fa2_install_{build_stamp}.log"
build_env = os.environ.copy()
build_env.update({
    "PYTHONUNBUFFERED": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_INPUT": "1",
    "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
    "FLASH_ATTN_CUDA_ARCHS": flash_arch,
    "MAX_JOBS": str(max_jobs),
    "NVCC_THREADS": str(nvcc_threads),
})

if not flash_ready and wheel_path is None:
    nvcc_path = shutil.which("nvcc")
    ninja_path = shutil.which("ninja")
    if not nvcc_path:
        raise RuntimeError(
            "Drive cache miss và không tìm thấy nvcc để build FA2 cho Torch 2.11."
        )
    if not ninja_path:
        raise RuntimeError("Drive cache miss và không tìm thấy ninja.")
    run_checked([nvcc_path, "--version"])
    run_checked([ninja_path, "--version"])
    build_plan = {
        "source": f"PyPI sdist flash-attn=={flash_version} (Dao-AILab upstream)",
        "install_mode": "source_build_after_local_and_drive_cache_miss",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "gpu": gpu_name,
        "compute_capability": capability,
        "flash_arch": flash_arch,
        "cxx11_abi": cxx11_abi,
        "available_ram_gib": round(available_ram_gib, 2),
        "max_jobs": max_jobs,
        "nvcc_threads": nvcc_threads,
        "wheel_dir": str(wheel_dir),
        "drive_cache_dir": str(drive_cache_dir) if drive_cache_dir else None,
        "build_log": str(build_log),
        "install_log": str(install_log),
    }
    print("FA2 build plan:", build_plan)
    run_observable([
        sys.executable, "-u", "-m", "pip", "wheel", "--verbose",
        "--no-build-isolation", "--no-deps", "--no-cache-dir",
        "--wheel-dir", wheel_dir, f"flash-attn=={flash_version}",
    ],
        phase="FA2 BUILD",
        env=build_env,
        heartbeat_seconds=CONFIG["flash_attn_heartbeat_seconds"],
        silence_warning_seconds=CONFIG["flash_attn_silence_warning_seconds"],
        log_path=build_log,
    )
    built_wheels = sorted(
        wheel_dir.glob(f"flash_attn-{flash_version}-*.whl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not built_wheels:
        raise RuntimeError(
            f"pip báo thành công nhưng không tìm thấy wheel trong {wheel_dir}. "
            f"Xem log: {build_log}"
        )
    wheel_path = built_wheels[0]
    wheel_origin = "source_build"
    print(
        f"FA2 wheel build xong: {wheel_path} "
        f"({_human_bytes(wheel_path.stat().st_size)})"
    )

if flash_ready:
    print("flash-attn đã đúng version và import được; bỏ qua build/install.")
    # Hỗ trợ publish wheel của một build vừa hoàn tất trong cùng runtime cũ.
    if wheel_path is not None and wheel_origin == "local_cache":
        publish_flash_attn_wheel_to_drive(
            wheel_path, drive_cache_dir, wheel_contract
        )
else:
    if wheel_path is None:
        raise RuntimeError("Không có FA2 wheel sau cache lookup/source build.")
    print(f"FA2 wheel origin: {wheel_origin}", flush=True)
    run_observable([
        sys.executable, "-u", "-m", "pip", "install", "--verbose",
        "--no-deps", "--force-reinstall", wheel_path,
    ],
        phase="FA2 INSTALL",
        env=build_env,
        heartbeat_seconds=CONFIG["flash_attn_heartbeat_seconds"],
        silence_warning_seconds=CONFIG["flash_attn_silence_warning_seconds"],
        log_path=install_log,
    )

    # Tránh module cache từ một binary FA2 cũ trong kernel đã từng import lỗi.
    for module_name in tuple(sys.modules):
        if module_name == "flash_attn" or module_name.startswith("flash_attn."):
            sys.modules.pop(module_name, None)
    sys.modules.pop("flash_attn_2_cuda", None)
    flash_ready, flash_status = flash_attn_status()
    print("FA2 final status:", flash_status)
    if not flash_ready:
        raise RuntimeError(
            "flash-attn build/install xong nhưng import validation thất bại. "
            f"Chi tiết: {flash_status}. Install log: {install_log}"
        )
    if wheel_origin in {"local_cache", "source_build"}:
        publish_flash_attn_wheel_to_drive(
            wheel_path, drive_cache_dir, wheel_contract
        )

QWEN_REPO_STR = str(QWEN_REPO_DIR.resolve())
if QWEN_REPO_STR not in sys.path:
    sys.path.insert(0, QWEN_REPO_STR)
QWEN_GIT_SHA = subprocess.check_output(
    ["git", "-C", str(QWEN_REPO_DIR), "rev-parse", "HEAD"], text=True
).strip()
if QWEN_GIT_SHA != CONFIG["qwen_git_ref"]:
    raise RuntimeError(f"Qwen Git SHA mismatch: {QWEN_GIT_SHA}")
print("Qwen3-VL-Embedding git SHA:", QWEN_GIT_SHA)
print("Qwen import root:", QWEN_REPO_STR)


# %% [markdown]
# ## 2. Xác thực HF và preflight phần cứng
#
# Token được lấy theo thứ tự: Colab Secret `HF_TOKEN` → environment → file cấu hình
# → prompt ẩn. Token không được in và không được ghi vào notebook/manifest.

# %%
import getpass
import importlib.metadata
import json
import platform
import shutil
import socket
from datetime import datetime, timezone

import psutil
import torch
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

if not torch.cuda.is_available():
    raise RuntimeError("Notebook này yêu cầu CUDA GPU.")

GPU_NAME = torch.cuda.get_device_name(0)
GPU_VRAM_GIB = torch.cuda.get_device_properties(0).total_memory / 2**30
SYSTEM_RAM_GIB = psutil.virtual_memory().total / 2**30
if CONFIG["fail_on_non_a100"] and "A100" not in GPU_NAME.upper():
    raise RuntimeError(f"Yêu cầu A100 nhưng runtime hiện tại là {GPU_NAME}.")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError(f"{GPU_NAME} hoặc runtime hiện tại không hỗ trợ BF16.")

runtime_python_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
runtime_torch_version = Version(torch.__version__.split("+")[0])
runtime_torch_major_minor = (
    f"{runtime_torch_version.major}.{runtime_torch_version.minor}"
)
runtime_torch_cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else None
runtime_contract = {
    "target_colab_runtime": CONFIG["target_colab_runtime"],
    "expected_python": CONFIG["target_python_major_minor"],
    "python": platform.python_version(),
    "expected_torch": CONFIG["target_torch_major_minor"],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
}
if runtime_python_major_minor != CONFIG["target_python_major_minor"]:
    raise RuntimeError(f"Notebook yêu cầu Colab Latest/Python 3.13: {runtime_contract}")
if runtime_torch_major_minor != CONFIG["target_torch_major_minor"]:
    raise RuntimeError(f"Notebook yêu cầu Colab Latest/Torch 2.11: {runtime_contract}")
if runtime_torch_cuda_major != int(CONFIG["target_torch_cuda_major"]):
    raise RuntimeError(f"Notebook yêu cầu Torch CUDA major 12: {runtime_contract}")

import flash_attn
import transformers

runtime_versions = {
    "transformers": transformers.__version__,
    "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
    "flash_attn": importlib.metadata.version("flash-attn"),
    "pillow": importlib.metadata.version("Pillow"),
}
expected_versions = {
    "transformers": CONFIG["transformers_version"],
    "qwen_vl_utils": CONFIG["qwen_vl_utils_version"],
    "flash_attn": CONFIG["flash_attn_version"],
    "pillow": CONFIG["pillow_version"],
}
if runtime_versions != expected_versions:
    raise RuntimeError(
        f"Dependency version mismatch: {runtime_versions} != {expected_versions}. "
        "Hãy restart runtime rồi Run All nếu package cũ còn trong kernel."
    )
print("flash-attn extension:", flash_attn.__file__)

SCRATCH_ROOT = Path(CONFIG["scratch_root"]).resolve()
DATA_ROOT = SCRATCH_ROOT / "data"
LOCAL_OUTPUT_ROOT = SCRATCH_ROOT / "output"
STATE_ROOT = SCRATCH_ROOT / "state"
MODEL_CACHE_ROOT = SCRATCH_ROOT / "model_cache"
for directory in (DATA_ROOT, LOCAL_OUTPUT_ROOT, STATE_ROOT, MODEL_CACHE_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

disk = shutil.disk_usage(SCRATCH_ROOT)
print({
    "gpu": GPU_NAME,
    "vram_gib": round(GPU_VRAM_GIB, 2),
    "system_ram_gib": round(SYSTEM_RAM_GIB, 2),
    "scratch_free_gib": round(disk.free / 2**30, 2),
    "torch": torch.__version__,
    "torchvision": importlib.metadata.version("torchvision"),
    **runtime_versions,
    "python": platform.python_version(),
})

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


# %% [markdown]
# ## 3. I/O Bucket, checksum và commit protocol

# %%
import hashlib
import math
import re
from typing import Dict, List, Optional, Sequence

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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bucket_uri(bucket_id: str, path: str):
    return f"hf://buckets/{bucket_id}/{path.lstrip('/')}"


def remote_info(bucket_id: str, path: str):
    # Exact-path lookup phải dùng metadata API. `list_bucket_tree(prefix=...)`
    # nhận directory prefix nên có thể báo thiếu dù file path thực sự tồn tại.
    # get_bucket_paths_info bỏ qua path không tồn tại nhưng vẫn propagate lỗi auth/network.
    items = list(get_bucket_paths_info(bucket_id, [path], token=HF_TOKEN))
    for item in items:
        if item.path == path and getattr(item, "type", "file") == "file":
            return item
    return None


def read_remote_bytes(bucket_id: str, path: str):
    result = subprocess.run(
        ["hf", "buckets", "cp", bucket_uri(bucket_id, path), "-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
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


def stream_bucket_tar_and_extract(
    remote_uri: str,
    destination: Path,
    expected_sha256: str,
    expected_bytes: int,
):
    """Stream HF → SHA-256 → tar, đồng thời hiện byte progress; không tạo TAR local."""
    if expected_bytes <= 0:
        raise ValueError(f"expected_bytes phải > 0 cho {remote_uri}")
    destination.mkdir(parents=True, exist_ok=True)
    hf_proc = subprocess.Popen(
        ["hf", "buckets", "cp", remote_uri, "-"],
        stdout=subprocess.PIPE,
        env=os.environ.copy(),
    )
    tar_proc = subprocess.Popen(
        ["tar", "-xf", "-", "--no-same-owner", "-C", str(destination)],
        stdin=subprocess.PIPE,
    )
    digest = hashlib.sha256()
    streamed = 0
    filename = remote_uri.rstrip("/").rsplit("/", 1)[-1]
    progress = tqdm(
        total=expected_bytes,
        desc=f"Download + extract {filename}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=0.5,
        smoothing=0.05,
        leave=True,
    )
    try:
        assert hf_proc.stdout is not None and tar_proc.stdin is not None
        while True:
            chunk = hf_proc.stdout.read(16 << 20)
            if not chunk:
                break
            digest.update(chunk)
            tar_proc.stdin.write(chunk)
            streamed += len(chunk)
            progress.update(len(chunk))
        tar_proc.stdin.close()
    except BaseException:
        hf_proc.kill()
        tar_proc.kill()
        raise
    finally:
        progress.close()
        if hf_proc.stdout is not None:
            hf_proc.stdout.close()

    hf_rc = hf_proc.wait()
    tar_rc = tar_proc.wait()
    actual_sha256 = digest.hexdigest()
    if hf_rc != 0 or tar_rc != 0:
        raise RuntimeError(f"Stream/extract lỗi: hf={hf_rc}, tar={tar_rc}, uri={remote_uri}")
    if streamed != expected_bytes:
        raise RuntimeError(
            f"Size mismatch cho {remote_uri}: {streamed:,} != {expected_bytes:,} bytes"
        )
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch cho {remote_uri}: {actual_sha256} != {expected_sha256}"
        )
    print(f"Extracted {streamed / 2**30:.2f} GiB; SHA-256 OK")
    return streamed


def validate_categories():
    assignments = CONFIG["session_assignments"]
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("session_assignments phải là dict khác rỗng.")
    owners = {}
    for session_id, assigned_categories in assignments.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id):
            raise ValueError(f"Tên session không hợp lệ: {session_id}")
        if not assigned_categories or len(set(assigned_categories)) != len(assigned_categories):
            raise ValueError(f"Assignment rỗng/trùng category: {session_id}")
        for category in assigned_categories:
            if category in owners:
                raise ValueError(
                    f"Category {category} bị gán cho cả {owners[category]} và {session_id}. "
                    "Không được chạy overlap trên cùng output_prefix."
                )
            owners[category] = session_id
    expected_categories = set(CONFIG["expected_counts"])
    assigned_categories = set(owners)
    if assigned_categories != expected_categories:
        raise ValueError(
            "session_assignments phải phủ đúng L21..L30; "
            f"missing={sorted(expected_categories - assigned_categories)}, "
            f"extra={sorted(assigned_categories - expected_categories)}"
        )
    if CONFIG["active_session"] not in assignments:
        raise ValueError(f"active_session không tồn tại: {CONFIG['active_session']}")
    if CONFIG["categories"] != list(assignments[CONFIG["active_session"]]):
        raise ValueError("categories phải được suy ra từ active_session.")

    categories = CONFIG["categories"]
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("CONFIG['categories'] phải khác rỗng và không trùng.")
    for category in categories:
        if not re.fullmatch(r"L(?:2[1-9]|30)", category):
            raise ValueError(f"Category không hợp lệ: {category}")
        if category not in CONFIG["expected_counts"] or category not in SOURCE_SHARDS:
            raise ValueError(f"Thiếu manifest cho {category}")
    if CONFIG["compute_precision"] != "bf16":
        raise ValueError("A100 Qwen 8B pipeline khóa compute_precision='bf16'.")
    if CONFIG["model_weight_dtype"] != "bfloat16":
        raise ValueError("Checkpoint production phải dùng model_weight_dtype='bfloat16'.")
    if CONFIG["storage_dtype"] != "float32":
        raise ValueError("Master embedding contract khóa storage_dtype='float32'.")
    if CONFIG["attention_implementation"] != "flash_attention_2":
        raise ValueError("Notebook compatibility v3 khóa FlashAttention-2.")
    if (
        CONFIG["flash_attn_version"],
        CONFIG["legacy_semantic_flash_attn_version"],
    ) != ("2.8.3", "2.8.3.post1"):
        raise ValueError("Compatibility pair phải là 2.8.3 -> 2.8.3.post1.")
    if CONFIG["embedding_dim"] != 4096 or CONFIG["mrl_dimension"] is not None:
        raise ValueError("Notebook final này khóa native 4096-d, không truncate MRL.")
    if CONFIG["image_instruction"] != "Represent the user's input.":
        raise ValueError("Image instruction phải khớp official default.")
    if (CONFIG["min_pixels"], CONFIG["max_pixels"]) != (4_096, 1_843_200):
        raise ValueError("Pixel budget phải khớp official wrapper 4096..1843200.")
    if not (0 < float(CONFIG["max_vram_fraction"]) < 1):
        raise ValueError("max_vram_fraction phải nằm trong (0,1).")
    if int(CONFIG["prefetch_factor"]) <= 0:
        raise ValueError("prefetch_factor phải là số nguyên dương.")
    if not isinstance(CONFIG["persistent_workers"], bool):
        raise ValueError("persistent_workers phải là bool.")
    if CONFIG["output_prefix"].strip("/") == CONFIG["source_prefix"].strip("/"):
        raise ValueError("output_prefix không được trùng source_prefix.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", CONFIG["session_name"]):
        raise ValueError("session_name chỉ dùng chữ, số, _, -, dấu chấm.")


validate_categories()


# %% [markdown]
# ## 4. Tải/verify metadata trước

# %%
METADATA_MARKER = STATE_ROOT / "metadata_extracted.json"
metadata_uri = bucket_uri(
    CONFIG["source_bucket_id"],
    f"{CONFIG['source_prefix']}/metadata/infoshootpp_metadata.tar",
)
marker_ok = False
if METADATA_MARKER.is_file():
    marker = json.loads(METADATA_MARKER.read_text(encoding="utf-8"))
    marker_ok = (
        marker.get("sha256") == SOURCE_METADATA["sha256"]
        and (DATA_ROOT / "infoshootpp" / "run_manifest.json").is_file()
        and (DATA_ROOT / "infoshootpp" / "run_summary.csv").is_file()
    )

if not marker_ok:
    stream_bucket_tar_and_extract(
        metadata_uri,
        DATA_ROOT,
        SOURCE_METADATA["sha256"],
        SOURCE_METADATA["bytes"],
    )
    run_manifest_path = DATA_ROOT / "infoshootpp" / "run_manifest.json"
    run_summary_path = DATA_ROOT / "infoshootpp" / "run_summary.csv"
    if sha256_file(run_manifest_path) != SOURCE_METADATA["run_manifest_sha256"]:
        raise RuntimeError("run_manifest.json không khớp source corpus.")
    if sha256_file(run_summary_path) != SOURCE_METADATA["run_summary_sha256"]:
        raise RuntimeError("run_summary.csv không khớp source corpus.")
    METADATA_MARKER.write_text(
        json.dumps({"sha256": SOURCE_METADATA["sha256"], "completed_at": utc_now()}, indent=2),
        encoding="utf-8",
    )
print("Metadata verified:", DATA_ROOT / "infoshootpp")

source_dataset_manifest_path = f"{CONFIG['source_prefix']}/manifests/dataset_manifest.json"
SOURCE_DATASET_MANIFEST_BYTES = read_remote_bytes(CONFIG["source_bucket_id"], source_dataset_manifest_path)
SOURCE_DATASET_MANIFEST_SHA256 = sha256_bytes(SOURCE_DATASET_MANIFEST_BYTES)
print("Remote dataset_manifest SHA-256:", SOURCE_DATASET_MANIFEST_SHA256)


# %% [markdown]
# ## 5. Đọc registry và chuẩn bị category

# %%
import csv
from dataclasses import dataclass

@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    video_id: str
    category: str
    frame_idx: int
    pts_time: float
    fps: float
    image_relpath: str
    image_path: str


def natural_video_key(video_id: str):
    match = re.search(r"_V(\d+)$", video_id)
    return (int(match.group(1)) if match else 10**12, video_id)


def safe_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_category_records(category: str):
    registry_dir = DATA_ROOT / "infoshootpp" / "registry" / category
    registry_files = sorted(registry_dir.glob("*.csv"), key=lambda p: natural_video_key(p.stem))
    if not registry_files:
        raise FileNotFoundError(f"Không có registry CSV cho {category}: {registry_dir}")

    records: List[FrameRecord] = []
    for csv_path in registry_files:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "video_id" not in reader.fieldnames or "frame_idx" not in reader.fieldnames:
                raise RuntimeError(f"Registry schema thiếu cột bắt buộc: {csv_path}")
            for row in reader:
                video_id = row["video_id"].strip()
                frame_idx = int(float(row["frame_idx"]))
                frame_id = (row.get("frame_id") or f"{video_id}@f{frame_idx:08d}").strip()
                image_relpath = f"infoshootpp/keyframes/{category}/{video_id}/f{frame_idx:08d}.jpg"
                records.append(FrameRecord(
                    frame_id=frame_id,
                    video_id=video_id,
                    category=category,
                    frame_idx=frame_idx,
                    pts_time=safe_float(row.get("pts_time")),
                    fps=safe_float(row.get("fps")),
                    image_relpath=image_relpath,
                    image_path=str(DATA_ROOT / image_relpath),
                ))

    records.sort(key=lambda r: (natural_video_key(r.video_id), r.frame_idx))
    expected = CONFIG["expected_counts"][category]
    if len(records) != expected:
        raise RuntimeError(f"{category}: registry rows {len(records):,} != expected {expected:,}")
    frame_ids = {record.frame_id for record in records}
    if len(frame_ids) != len(records):
        raise RuntimeError(f"{category}: duplicate canonical frame_id trong registry.")
    return records


def count_jpegs(category_dir: Path):
    total = 0
    for video_dir in category_dir.iterdir():
        if video_dir.is_dir():
            total += sum(1 for entry in os.scandir(video_dir) if entry.is_file() and entry.name.endswith(".jpg"))
    return total


def assert_safe_category_path(path: Path, category: str):
    expected = (DATA_ROOT / "infoshootpp" / "keyframes" / category).resolve()
    if path.resolve() != expected or not re.fullmatch(r"L(?:2[1-9]|30)", category):
        raise RuntimeError(f"Refuse destructive cleanup ngoài category scratch: {path}")


PREPARED_CATEGORIES = set()


def prepare_category_images(category: str, records: Sequence[FrameRecord]):
    category_dir = DATA_ROOT / "infoshootpp" / "keyframes" / category
    if category in PREPARED_CATEGORIES and category_dir.is_dir():
        return category_dir
    marker_path = STATE_ROOT / f"{category}_extracted.json"
    expected_count = CONFIG["expected_counts"][category]
    expected_sha = SOURCE_SHARDS[category]["sha256"]
    marker_ok = False
    if marker_path.is_file() and category_dir.is_dir():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_ok = marker.get("sha256") == expected_sha and marker.get("count") == expected_count

    if not marker_ok:
        if category_dir.exists():
            assert_safe_category_path(category_dir, category)
            shutil.rmtree(category_dir)
        # Model cache 15.2 GiB đã được xóa sau khi weights nằm trên GPU. TAR được
        # stream thẳng vào extractor; giữ 2 GiB cho metadata/state/Parquet tạm.
        needed = SOURCE_SHARDS[category]["bytes"] + (2 << 30)
        free = shutil.disk_usage(SCRATCH_ROOT).free
        if free < needed:
            raise RuntimeError(
                f"{category} cần tối thiểu khoảng {needed / 2**30:.1f} GiB free; "
                f"hiện có {free / 2**30:.1f} GiB."
            )
        uri = bucket_uri(
            CONFIG["source_bucket_id"],
            f"{CONFIG['source_prefix']}/keyframes/{category}.tar",
        )
        stream_bucket_tar_and_extract(
            uri,
            DATA_ROOT,
            expected_sha,
            SOURCE_SHARDS[category]["bytes"],
        )
        actual_count = count_jpegs(category_dir)
        if actual_count != expected_count:
            raise RuntimeError(f"{category}: JPEG count {actual_count:,} != {expected_count:,}")
        marker_path.write_text(
            json.dumps({"sha256": expected_sha, "count": actual_count, "completed_at": utc_now()}, indent=2),
            encoding="utf-8",
        )

    if CONFIG["strict_file_audit"]:
        missing = [record.image_path for record in tqdm(records, desc=f"Audit paths {category}") if not Path(record.image_path).is_file()]
        if missing:
            raise RuntimeError(f"{category}: thiếu {len(missing):,} ảnh; ví dụ {missing[:3]}")
    PREPARED_CATEGORIES.add(category)
    return category_dir


# %% [markdown]
# ## 6. Pin/load Qwen3-VL-Embedding-8B và khóa semantic fingerprint
#
# Wrapper chính thức được pin bằng Git SHA. Checkpoint được pin bằng exact HF commit.
# Image instruction/pixel budget/chat template/pooling/native dimension là semantic;
# batch, worker và session name chỉ là execution policy.

# %%
import gc
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder


QWEN_IMPLEMENTATION_PATH = QWEN_REPO_DIR / "src/models/qwen3_vl_embedding.py"
QWEN_IMPLEMENTATION_SHA256 = sha256_file(QWEN_IMPLEMENTATION_PATH)
MODEL_SNAPSHOT_DIR = Path(snapshot_download(
    repo_id=CONFIG["model_id"],
    revision=CONFIG["model_revision"],
    cache_dir=MODEL_CACHE_ROOT,
    token=HF_TOKEN,
    allow_patterns=[
        "*.json", "*.jinja", "*.model", "*.txt", "*.safetensors",
        "merges.txt", "vocab.json", "scripts/*",
    ],
))
MODEL_INDEX_PATH = MODEL_SNAPSHOT_DIR / "model.safetensors.index.json"
if not MODEL_INDEX_PATH.is_file():
    raise FileNotFoundError(f"Thiếu model index: {MODEL_INDEX_PATH}")
MODEL_INDEX_SHA256 = sha256_file(MODEL_INDEX_PATH)
MODEL_INDEX = json.loads(MODEL_INDEX_PATH.read_text(encoding="utf-8"))
MODEL_TENSOR_PAYLOAD_BYTES = int(MODEL_INDEX.get("metadata", {}).get("total_size", -1))
EXPECTED_MODEL_TENSOR_PAYLOAD_BYTES = 16_289_587_680
EXPECTED_MODEL_WEIGHT_FILE_BYTES = 16_289_679_624
if MODEL_TENSOR_PAYLOAD_BYTES != EXPECTED_MODEL_TENSOR_PAYLOAD_BYTES:
    raise RuntimeError(
        "Unexpected checkpoint tensor payload bytes: "
        f"{MODEL_TENSOR_PAYLOAD_BYTES} != {EXPECTED_MODEL_TENSOR_PAYLOAD_BYTES}"
    )
weight_files = sorted(set(MODEL_INDEX["weight_map"].values()))
missing_weights = [name for name in weight_files if not (MODEL_SNAPSHOT_DIR / name).is_file()]
if missing_weights:
    raise FileNotFoundError(f"Thiếu checkpoint shards: {missing_weights}")
MODEL_WEIGHT_FILE_BYTES = sum(
    (MODEL_SNAPSHOT_DIR / name).stat().st_size for name in weight_files
)
if MODEL_WEIGHT_FILE_BYTES != EXPECTED_MODEL_WEIGHT_FILE_BYTES:
    raise RuntimeError(
        "Checkpoint physical shard bytes mismatch: "
        f"{MODEL_WEIGHT_FILE_BYTES} != {EXPECTED_MODEL_WEIGHT_FILE_BYTES}"
    )
SAFETENSORS_CONTAINER_OVERHEAD_BYTES = (
    MODEL_WEIGHT_FILE_BYTES - MODEL_TENSOR_PAYLOAD_BYTES
)

MODEL_CONFIG = AutoConfig.from_pretrained(MODEL_SNAPSHOT_DIR, local_files_only=True)
if MODEL_CONFIG.model_type != "qwen3_vl":
    raise RuntimeError(f"Unexpected model_type: {MODEL_CONFIG.model_type}")
if int(MODEL_CONFIG.text_config.hidden_size) != CONFIG["embedding_dim"]:
    raise RuntimeError("Qwen text hidden_size không phải 4096.")
if int(MODEL_CONFIG.text_config.num_hidden_layers) != 36:
    raise RuntimeError("Qwen3-VL-Embedding-8B phải có 36 text layers.")
if int(MODEL_CONFIG.vision_config.patch_size) != CONFIG["image_patch_size"]:
    raise RuntimeError("Vision patch_size không khớp official wrapper.")

print({
    "model_snapshot": str(MODEL_SNAPSHOT_DIR),
    "checkpoint_tensor_payload_gib": round(MODEL_TENSOR_PAYLOAD_BYTES / 2**30, 3),
    "checkpoint_weight_files_gib": round(MODEL_WEIGHT_FILE_BYTES / 2**30, 3),
    "safetensors_container_overhead_bytes": SAFETENSORS_CONTAINER_OVERHEAD_BYTES,
    "weight_shards": weight_files,
    "model_index_sha256": MODEL_INDEX_SHA256,
    "official_wrapper_sha256": QWEN_IMPLEMENTATION_SHA256,
})

QWEN_EMBEDDER = Qwen3VLEmbedder(
    model_name_or_path=str(MODEL_SNAPSHOT_DIR),
    max_length=CONFIG["max_length"],
    min_pixels=CONFIG["min_pixels"],
    max_pixels=CONFIG["max_pixels"],
    default_instruction=CONFIG["image_instruction"],
    torch_dtype=torch.bfloat16,
    attn_implementation=CONFIG["attention_implementation"],
    low_cpu_mem_usage=True,
)
embedding_model = QWEN_EMBEDDER.model
processor = QWEN_EMBEDDER.processor
embedding_model.requires_grad_(False).eval()
embedding_model.config.use_cache = False
first_parameter = next(embedding_model.parameters())
if first_parameter.dtype != torch.bfloat16 or first_parameter.device.type != "cuda":
    raise RuntimeError(
        f"Model residency/dtype sai: {first_parameter.device}, {first_parameter.dtype}"
    )
loaded_attention = getattr(embedding_model.config, "_attn_implementation", None)
if loaded_attention != CONFIG["attention_implementation"]:
    raise RuntimeError(
        f"Attention backend mismatch: {loaded_attention} != "
        f"{CONFIG['attention_implementation']}"
    )
if QWEN_EMBEDDER.default_instruction != CONFIG["image_instruction"]:
    raise RuntimeError("Official wrapper default instruction mismatch.")
if processor.tokenizer.padding_side != "right":
    raise RuntimeError("Official wrapper phải dùng right padding.")
image_processor = processor.image_processor
if int(image_processor.patch_size) != CONFIG["image_patch_size"]:
    raise RuntimeError("Processor image patch_size mismatch.")
if int(image_processor.merge_size) != 2 or int(image_processor.temporal_patch_size) != 2:
    raise RuntimeError("Processor spatial/temporal merge contract mismatch.")
if list(image_processor.image_mean) != [0.5, 0.5, 0.5]:
    raise RuntimeError(f"Processor image_mean mismatch: {image_processor.image_mean}")
if list(image_processor.image_std) != [0.5, 0.5, 0.5]:
    raise RuntimeError(f"Processor image_std mismatch: {image_processor.image_std}")
gc.collect()
torch.cuda.empty_cache()

# Cho phép chạy lại riêng cell này sau khi cập nhật notebook trong một runtime cũ.
# Một Run All sạch đã tạo key này ở preflight; runtime cũ có thể chưa có nó.
PILLOW_RUNTIME_VERSION = importlib.metadata.version("Pillow")
if PILLOW_RUNTIME_VERSION != CONFIG["pillow_version"]:
    raise RuntimeError(
        f"Pillow runtime mismatch: {PILLOW_RUNTIME_VERSION} != "
        f"{CONFIG['pillow_version']}"
    )
runtime_versions["pillow"] = PILLOW_RUNTIME_VERSION


SEMANTIC_CONFIG = {
    "schema_version": 1,
    "contract": "official_Qwen3VLEmbedder_image_only",
    "source": {
        "bucket_id": CONFIG["source_bucket_id"],
        "prefix": CONFIG["source_prefix"],
        "dataset_manifest_sha256": SOURCE_DATASET_MANIFEST_SHA256,
        "run_manifest_sha256": SOURCE_METADATA["run_manifest_sha256"],
        "run_summary_sha256": SOURCE_METADATA["run_summary_sha256"],
        "expected_total_frames": sum(CONFIG["expected_counts"].values()),
        "expected_counts": CONFIG["expected_counts"],
    },
    "model": {
        "id": CONFIG["model_id"],
        "revision": CONFIG["model_revision"],
        "checkpoint_index_sha256": MODEL_INDEX_SHA256,
        "checkpoint_tensor_payload_bytes": MODEL_TENSOR_PAYLOAD_BYTES,
        "checkpoint_weight_file_bytes": MODEL_WEIGHT_FILE_BYTES,
        "safetensors_container_overhead_bytes": SAFETENSORS_CONTAINER_OVERHEAD_BYTES,
        "official_implementation_git_sha": QWEN_GIT_SHA,
        "official_implementation_sha256": QWEN_IMPLEMENTATION_SHA256,
        "transformers_version": runtime_versions["transformers"],
        "qwen_vl_utils_version": runtime_versions["qwen_vl_utils"],
        # Legacy v3 compatibility reference. Runtime version thật (2.8.3) nằm
        # trong execution log; post1 chỉ thay packaging/CUDA-13 wheel selection.
        "flash_attn_version": CONFIG["legacy_semantic_flash_attn_version"],
        "pillow_version": runtime_versions["pillow"],
        "architecture": "Qwen3VLForEmbedding_full_multimodal_model",
        "text_layers": 36,
        "embedding_dim": CONFIG["embedding_dim"],
        "mrl_dimension": CONFIG["mrl_dimension"],
    },
    "preprocess": {
        "input": "image_only",
        "system_instruction": CONFIG["image_instruction"],
        "chat_template": "processor.apply_chat_template(add_generation_prompt=True)",
        "vision_preprocess": "qwen_vl_utils.process_vision_info",
        "image_patch_size": CONFIG["image_patch_size"],
        "min_pixels": CONFIG["min_pixels"],
        "max_pixels": CONFIG["max_pixels"],
        "preserve_aspect_ratio": True,
        "smart_resize_factor": 32,
        "color": "RGB via qwen_vl_utils.to_rgb",
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "center_crop": False,
        "max_length": CONFIG["max_length"],
        "truncation": True,
        "padding": True,
        "processor_do_resize": False,
        "precast_pixel_values_bf16": CONFIG["precast_pixel_values_bf16"],
    },
    "inference": {
        "compute_precision": CONFIG["compute_precision"],
        "model_weight_dtype": CONFIG["model_weight_dtype"],
        "autocast": False,
        "attention_implementation": CONFIG["attention_implementation"],
        "pooling": "last_attended_token",
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
if SEMANTIC_FINGERPRINT != CONFIG["expected_legacy_semantic_fingerprint"]:
    raise RuntimeError(
        "Không tái tạo đúng legacy v3 semantic fingerprint; từ chối resume. "
        f"{SEMANTIC_FINGERPRINT} != "
        f"{CONFIG['expected_legacy_semantic_fingerprint']}"
    )
print({
    "flash_attn_runtime_version": runtime_versions["flash_attn"],
    "legacy_semantic_flash_attn_version": CONFIG[
        "legacy_semantic_flash_attn_version"
    ],
    "compatibility_reason": (
        "upstream v2.8.3.post1 only changes CUDA-13 wheel selection/versioning"
    ),
})

QUERY_CONTRACT = {
    "model_id": CONFIG["model_id"],
    "model_revision": CONFIG["model_revision"],
    "recommended_instruction": "Retrieve images or text relevant to the user's query.",
    "instruction_language": "English",
    "note": "Query instruction may be A/B tested without re-encoding image vectors.",
    "input": "text_only user message with task-specific system instruction",
    "max_length": CONFIG["max_length"],
    "chat_template": "official Qwen3VLProcessor with add_generation_prompt=True",
    "pooling": "last_attended_token",
    "dimension": CONFIG["embedding_dim"],
    "l2_normalize_fp32": True,
    "metric": "COSINE",
}

RUN_CONFIG_REMOTE_PATH = f"{CONFIG['output_prefix'].strip('/')}/run_config.json"
RUN_CONFIG_PAYLOAD = {
    # Payload cố ý deterministic để nhiều session khởi tạo đồng thời vẫn ghi cùng bytes.
    "semantic_fingerprint": SEMANTIC_FINGERPRINT,
    "semantic_config": SEMANTIC_CONFIG,
    "query_contract": QUERY_CONTRACT,
    "parallel_plan": CONFIG["session_assignments"],
}


def assert_run_config_compatible(remote_config):
    if remote_config.get("semantic_fingerprint") != SEMANTIC_FINGERPRINT:
        raise RuntimeError(
            "Output prefix đã tồn tại với semantic fingerprint khác. "
            "Để bảo vệ vector cũ, hãy checkout git SHA/checkpoint cũ hoặc dùng output_prefix mới."
        )
    remote_plan = remote_config.get("parallel_plan")
    if remote_plan is None:
        return
    if not isinstance(remote_plan, dict) or not remote_plan:
        raise RuntimeError("Remote parallel_plan không hợp lệ.")

    remote_owners = {}
    for session_id, categories in remote_plan.items():
        if not isinstance(categories, list) or not categories:
            raise RuntimeError(f"Remote parallel_plan rỗng/sai kiểu ở {session_id}.")
        if len(categories) != len(set(categories)):
            raise RuntimeError(f"Remote parallel_plan trùng category trong {session_id}.")
        for category in categories:
            if category in remote_owners:
                raise RuntimeError(
                    f"Remote parallel_plan overlap {category}: "
                    f"{remote_owners[category]} và {session_id}."
                )
            remote_owners[category] = session_id
    if set(remote_owners) != set(CONFIG["expected_counts"]):
        raise RuntimeError("Remote parallel_plan không phủ đúng toàn bộ expected categories.")

    if remote_plan != CONFIG["session_assignments"]:
        active_session = CONFIG["active_session"]
        if active_session not in remote_plan:
            raise RuntimeError(
                f"active_session={active_session} không tồn tại trong remote parallel_plan."
            )
        local_plan = CONFIG["session_assignments"]
        CONFIG["session_assignments"] = remote_plan
        CONFIG["categories"] = list(remote_plan[active_session])
        CONFIG["session_name"] = f"colab_{active_session}"
        RUN_CONFIG_PAYLOAD["parallel_plan"] = remote_plan
        print(
            "Local parallel_plan khác remote; đã dùng remote run_config làm source of truth.",
            {"local": local_plan, "remote": remote_plan,
             "active_categories": CONFIG["categories"]},
        )


existing_run_config = read_remote_json_optional(
    CONFIG["output_bucket_id"], RUN_CONFIG_REMOTE_PATH
) if CONFIG["upload"] else None
if existing_run_config is not None:
    assert_run_config_compatible(existing_run_config)
    print("Remote run_config tương thích: resume được phép.")
elif CONFIG["upload"]:
    preexisting_objects = list(list_bucket_tree(
        CONFIG["output_bucket_id"],
        prefix=CONFIG["output_prefix"].strip("/"),
        recursive=True,
        token=HF_TOKEN,
    ))
    if preexisting_objects:
        # Session khác có thể vừa tạo run_config sau lần read đầu tiên.
        raced_config = read_remote_json_optional(
            CONFIG["output_bucket_id"], RUN_CONFIG_REMOTE_PATH
        )
        if raced_config is None:
            object_preview = [
                {
                    "path": item.path,
                    "type": getattr(item, "type", None),
                    "size": getattr(item, "size", None),
                }
                for item in preexisting_objects[:20]
            ]
            raise RuntimeError(
                "Output prefix đã có object nhưng thiếu run_config.json; từ chối ghi đè. "
                f"Các object đầu tiên: {object_preview}. "
                "Hãy audit prefix cũ hoặc chọn output_prefix mới."
            )
        assert_run_config_compatible(raced_config)
        print("Session khác vừa khởi tạo run_config; cấu hình tương thích.")
    else:
        if CONFIG["require_existing_legacy_run_config"]:
            raise RuntimeError(
                "Notebook compatibility này chỉ được resume collection v3 đã có "
                "run_config.json; từ chối khởi tạo collection mới bằng legacy fingerprint."
            )
        upload_bytes_verified(
            CONFIG["output_bucket_id"], RUN_CONFIG_REMOTE_PATH,
            json.dumps(RUN_CONFIG_PAYLOAD, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        persisted_config = read_remote_json_optional(
            CONFIG["output_bucket_id"], RUN_CONFIG_REMOTE_PATH
        )
        if persisted_config is None:
            raise RuntimeError("Không đọc lại được run_config sau upload.")
        assert_run_config_compatible(persisted_config)
        print("Đã tạo remote run_config.")


def cleanup_local_model_cache():
    if not CONFIG["cleanup_model_cache_after_load"]:
        return
    expected = (SCRATCH_ROOT / "model_cache").resolve()
    if MODEL_CACHE_ROOT.resolve() != expected:
        raise RuntimeError(f"Refuse cleanup model cache ngoài scratch: {MODEL_CACHE_ROOT}")
    if MODEL_CACHE_ROOT.exists():
        shutil.rmtree(MODEL_CACHE_ROOT)
    print("Đã giải phóng local model cache; model BF16 vẫn nằm trên GPU.")


cleanup_local_model_cache()


# %% [markdown]
# ## 7. DataLoader: official Qwen chat + smart-resize trong CPU workers

# %%
import numpy as np
from PIL import Image, ImageFile
from qwen_vl_utils.vision_process import process_vision_info, smart_resize
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = False


def auto_num_workers():
    cpu_count = os.cpu_count() or 4
    # High-RAM Colab có thể feed A100 bằng tối đa 8 process; vẫn chừa CPU cho
    # notebook, upload và Parquet. Runtime RAM thấp giữ cap bảo thủ.
    if SYSTEM_RAM_GIB >= 50:
        return max(1, min(8, cpu_count - 2))
    if SYSTEM_RAM_GIB >= 25:
        return max(1, min(6, cpu_count - 2))
    return max(1, min(4, cpu_count // 2))


NUM_WORKERS = auto_num_workers() if CONFIG["num_workers"] is None else int(CONFIG["num_workers"])
if NUM_WORKERS < 0:
    raise ValueError("num_workers phải >= 0.")
print("DataLoader workers:", NUM_WORKERS)


def format_image_conversation(image_path: str):
    # Khớp Qwen3VLEmbedder.format_model_input(image=..., instruction=None).
    absolute_path = str(Path(image_path).resolve())
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": CONFIG["image_instruction"]}],
        },
        {
            "role": "user",
            "content": [{
                "type": "image",
                "image": "file://" + absolute_path,
                "min_pixels": CONFIG["min_pixels"],
                "max_pixels": CONFIG["max_pixels"],
            }],
        },
    ]


def preprocess_image_paths(image_paths: Sequence[str], precast_pixel_values: bool):
    conversations = [format_image_conversation(path) for path in image_paths]
    texts = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    images, video_inputs, video_kwargs = process_vision_info(
        conversations,
        image_patch_size=CONFIG["image_patch_size"],
        return_video_metadata=True,
        return_video_kwargs=True,
    )
    if video_inputs is not None:
        raise RuntimeError("Image-only pipeline nhận video_inputs ngoài dự kiến.")
    inputs = processor(
        text=texts,
        images=images,
        videos=None,
        video_metadata=None,
        truncation=True,
        max_length=CONFIG["max_length"],
        padding=True,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )
    result = {key: value for key, value in inputs.items() if torch.is_tensor(value)}
    if precast_pixel_values:
        # Qwen visual tower cast pixel_values sang weight dtype BF16 trước patch embed.
        # Cast ở CPU giữ cùng BF16 values nhưng giảm 2× worker IPC và H2D.
        result["pixel_values"] = result["pixel_values"].to(torch.bfloat16)
    return result


class QwenFrameDataset(Dataset):
    def __init__(self, records: Sequence[FrameRecord], indices: Sequence[int]):
        self.records = records
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row_idx = int(self.indices[item])
        return self.records[row_idx].image_path, row_idx


class QwenBatchCollator:
    def __call__(self, samples):
        image_paths, row_indices = zip(*samples)
        try:
            inputs = preprocess_image_paths(
                image_paths,
                precast_pixel_values=CONFIG["precast_pixel_values_bf16"],
            )
        except Exception as exc:
            raise RuntimeError(f"Qwen preprocess lỗi, paths={list(image_paths)[:3]}") from exc
        return inputs, torch.tensor(row_indices, dtype=torch.int64)


QWEN_COLLATOR = QwenBatchCollator()


def qwen_worker_init_fn(_worker_id):
    torch.set_num_threads(1)


def build_loader(records, indices, batch_size, workers=NUM_WORKERS):
    kwargs = {
        "dataset": QwenFrameDataset(records, indices),
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": QWEN_COLLATOR,
        "worker_init_fn": qwen_worker_init_fn,
    }
    if workers > 0:
        kwargs.update(
            persistent_workers=CONFIG["persistent_workers"],
            prefetch_factor=CONFIG["prefetch_factor"],
        )
    return DataLoader(**kwargs)


def move_inputs_to_cuda(cpu_inputs):
    return {
        key: value.cuda(non_blocking=True)
        for key, value in cpu_inputs.items()
    }


class CUDAPrefetcher:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        iterator = iter(self.loader)
        stream = torch.cuda.Stream()

        def preload():
            try:
                cpu_inputs, row_indices = next(iterator)
            except StopIteration:
                return None, None
            with torch.cuda.stream(stream):
                gpu_inputs = move_inputs_to_cuda(cpu_inputs)
            return gpu_inputs, row_indices

        next_inputs, next_indices = preload()
        while next_inputs is not None:
            torch.cuda.current_stream().wait_stream(stream)
            gpu_inputs, row_indices = next_inputs, next_indices
            for tensor in gpu_inputs.values():
                tensor.record_stream(torch.cuda.current_stream())
            next_inputs, next_indices = preload()
            yield gpu_inputs, row_indices


def assert_preprocess_parity(records, sample_count=None):
    sample_count = int(sample_count or CONFIG["preprocess_parity_samples"])
    chosen = np.linspace(0, len(records) - 1, min(sample_count, len(records)), dtype=np.int64)
    paths = [records[int(idx)].image_path for idx in chosen]
    conversations = [QWEN_EMBEDDER.format_model_input(image=path) for path in paths]
    official = dict(QWEN_EMBEDDER._preprocess_inputs(conversations))
    reference = preprocess_image_paths(paths, precast_pixel_values=False)
    if set(official) != set(reference):
        raise RuntimeError(f"Preprocess keys mismatch: {set(official)} != {set(reference)}")
    for key, official_value in official.items():
        if not torch.equal(official_value, reference[key]):
            max_abs = (
                (official_value.float() - reference[key].float()).abs().max().item()
            )
            raise RuntimeError(f"Official preprocess parity fail key={key}, max_abs={max_abs}")

    production = preprocess_image_paths(
        paths, precast_pixel_values=CONFIG["precast_pixel_values_bf16"]
    )
    expected_pixels = official["pixel_values"].to(torch.bfloat16)
    if not torch.equal(production["pixel_values"], expected_pixels):
        raise RuntimeError("BF16 pixel pre-cast không khớp official float→BF16.")
    token_counts = official["attention_mask"].sum(dim=1).tolist()
    grids = official["image_grid_thw"].tolist()
    print(f"Official preprocess parity OK trên {len(chosen)} ảnh.", {
        "token_count_min": int(min(token_counts)),
        "token_count_max": int(max(token_counts)),
        "image_grid_thw": grids,
    })


# %% [markdown]
# ## 8. Resume state, batch autotune và official-wrapper quality gate

# %%
def shard_bounds(total_rows: int, shard_id: int):
    start = shard_id * CONFIG["shard_rows"]
    stop = min(total_rows, start + CONFIG["shard_rows"])
    return start, stop


def total_shards(total_rows: int):
    return math.ceil(total_rows / CONFIG["shard_rows"])


def remote_commit_prefix(category: str):
    return f"{CONFIG['output_prefix'].strip('/')}/commits/{category}"


def remote_parquet_path(category: str, shard_id: int):
    return f"{CONFIG['output_prefix'].strip('/')}/embeddings/{category}/part-{shard_id:05d}.parquet"


def remote_commit_path(category: str, shard_id: int):
    return f"{remote_commit_prefix(category)}/part-{shard_id:05d}.json"


def load_valid_remote_commits(category: str, records: Sequence[FrameRecord]):
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
    downloads = []
    for item in commit_items:
        downloads.append((item, str(local_dir / Path(item.path).name)))
    download_bucket_files(CONFIG["output_bucket_id"], files=downloads, token=HF_TOKEN)

    parquet_items = {
        item.path: item for item in list_bucket_tree(
            CONFIG["output_bucket_id"],
            prefix=f"{CONFIG['output_prefix'].strip('/')}/embeddings/{category}",
            recursive=True,
            token=HF_TOKEN,
        )
        if getattr(item, "type", "file") == "file"
    }
    valid = {}
    expected_total_shards = total_shards(len(records))
    for local_commit in local_dir.glob("*.json"):
        commit = json.loads(local_commit.read_text(encoding="utf-8"))
        if commit.get("semantic_fingerprint") != SEMANTIC_FINGERPRINT:
            raise RuntimeError(f"Commit fingerprint mismatch: {local_commit.name}")
        shard_id = int(commit["shard_id"])
        if not 0 <= shard_id < expected_total_shards:
            raise RuntimeError(f"Commit shard_id ngoài range: {local_commit.name}")
        start, stop = shard_bounds(len(records), shard_id)
        expected = {
            "category": category,
            "row_start": start,
            "row_stop": stop,
            "rows": stop - start,
            "first_frame_id": records[start].frame_id,
            "last_frame_id": records[stop - 1].frame_id,
            "parquet_path": remote_parquet_path(category, shard_id),
        }
        for key, value in expected.items():
            if commit.get(key) != value:
                raise RuntimeError(f"Commit metadata mismatch {local_commit.name}: {key}")
        remote_path = commit["parquet_path"]
        info = parquet_items.get(remote_path)
        if info is None or int(info.size) != int(commit["parquet_bytes"]):
            raise RuntimeError(f"Commit trỏ tới Parquet thiếu/sai size: {remote_path}")
        if shard_id in valid:
            raise RuntimeError(f"Duplicate commit shard_id={shard_id}")
        valid[shard_id] = commit
    return valid


def default_batch_candidates():
    if GPU_VRAM_GIB >= 70:
        return [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
    return [1, 2, 3, 4, 6, 8]


def forward_normalized(gpu_inputs):
    # Khớp official Qwen3VLEmbedder.process(): model BF16 forward trực tiếp,
    # không thêm autocast execution policy. Pooling xong mới cast FP32 để normalize.
    with torch.inference_mode():
        outputs = embedding_model(**gpu_inputs)
        raw = QWEN_EMBEDDER._pooling_last(
            outputs.last_hidden_state, gpu_inputs["attention_mask"]
        )
    if CONFIG["mrl_dimension"] is not None:
        raw = raw[:, : int(CONFIG["mrl_dimension"])]
    return F.normalize(raw.float(), p=2, dim=-1)


def benchmark_batch_sizes(records):
    candidates = (
        [int(CONFIG["batch_size"])]
        if CONFIG["batch_size"] is not None
        else list(map(int, CONFIG["batch_candidates"] or default_batch_candidates()))
    )
    candidates = sorted(set(candidates))
    if not candidates or candidates[0] <= 0:
        raise ValueError("Batch candidates phải là số nguyên dương.")

    # Synthetic 1920×960 = đúng 1.843.200 pixels và chia hết factor 32. Benchmark
    # trên worst-case visual-token budget để batch đã chọn không OOM ở category sau.
    stress_path = STATE_ROOT / "autotune_max_pixels_1920x960.jpg"
    if not stress_path.is_file():
        Image.new("RGB", (1_920, 960), color=(127, 127, 127)).save(
            stress_path, format="JPEG", quality=90
        )
    target_height, target_width = smart_resize(
        960,
        1_920,
        factor=CONFIG["image_patch_size"] * 2,
        min_pixels=CONFIG["min_pixels"],
        max_pixels=CONFIG["max_pixels"],
    )
    if target_height * target_width != CONFIG["max_pixels"]:
        raise RuntimeError("Autotune stress image không đạt exact max_pixels.")

    results = []
    total_vram = torch.cuda.get_device_properties(0).total_memory
    for batch_size in candidates:
        cpu_inputs = None
        gpu_inputs = None
        try:
            paths = [str(stress_path)] * batch_size
            cpu_inputs = preprocess_image_paths(
                paths, precast_pixel_values=CONFIG["precast_pixel_values_bf16"]
            )
            token_counts = cpu_inputs["attention_mask"].sum(dim=1)
            gpu_inputs = move_inputs_to_cuda(cpu_inputs)
            torch.cuda.reset_peak_memory_stats()
            _ = forward_normalized(gpu_inputs)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            timed_steps = 2
            for _ in range(timed_steps):
                _ = forward_normalized(gpu_inputs)
            end.record()
            torch.cuda.synchronize()
            elapsed_s = start.elapsed_time(end) / 1000.0
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            result = {
                "batch_size": batch_size,
                "images_per_second": timed_steps * batch_size / elapsed_s,
                "peak_allocated_gib": peak_allocated / 2**30,
                "peak_reserved_gib": peak_reserved / 2**30,
                "peak_fraction": peak_reserved / total_vram,
                "max_sequence_tokens": int(token_counts.max().item()),
                "max_visual_grid_tokens": int(
                    cpu_inputs["image_grid_thw"][:, 0:3].prod(dim=1).max().item()
                ),
            }
            results.append(result)
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()})
        except torch.OutOfMemoryError:
            print(f"OOM ở batch={batch_size}; dừng tăng batch.")
            break
        finally:
            if gpu_inputs is not None:
                del gpu_inputs
            if cpu_inputs is not None:
                del cpu_inputs
            gc.collect()
            torch.cuda.empty_cache()  # Chỉ autotune/OOM, tuyệt đối không nằm trong hot loop.

    safe = [
        result for result in results
        if result["peak_fraction"] <= CONFIG["max_vram_fraction"]
    ]
    if not safe:
        safe = results
    if not safe:
        raise RuntimeError("Không tìm được batch size chạy được.")
    best = max(safe, key=lambda r: r["images_per_second"])
    if CONFIG["batch_size"] is not None and best["batch_size"] != int(CONFIG["batch_size"]):
        raise RuntimeError("Manual batch không vượt qua validation.")
    print("Selected batch size:", best["batch_size"])
    return int(best["batch_size"]), results


def official_wrapper_embedding_audit(records, sample_count=4):
    chosen = np.linspace(0, len(records) - 1, min(sample_count, len(records)), dtype=np.int64)
    paths = [records[int(idx)].image_path for idx in chosen]
    with torch.inference_mode():
        official = QWEN_EMBEDDER.process(
            [{"image": path} for path in paths], normalize=True
        ).float()
    cpu_inputs = preprocess_image_paths(
        paths, precast_pixel_values=CONFIG["precast_pixel_values_bf16"]
    )
    gpu_inputs = move_inputs_to_cuda(cpu_inputs)
    production = forward_normalized(gpu_inputs)
    cosine = F.cosine_similarity(
        F.normalize(official, dim=-1), production, dim=-1
    ).float()
    audit = {
        "sample_count": len(paths),
        "production_vs_official_wrapper_min_cosine": cosine.min().item(),
        "production_vs_official_wrapper_mean_cosine": cosine.mean().item(),
    }
    if audit["production_vs_official_wrapper_min_cosine"] < 0.999:
        raise RuntimeError(f"Official wrapper embedding parity fail: {audit}")
    print("Official wrapper embedding parity:", audit)
    del cpu_inputs, gpu_inputs, official, production
    return audit


def numeric_quality_audit(records, production_batch_size):
    sample_count = min(int(CONFIG["numeric_audit_samples"]), len(records))
    audit_batch_size = min(int(production_batch_size), sample_count)
    if sample_count <= 0 or audit_batch_size <= 0:
        raise ValueError("Numeric audit không có sample/batch hợp lệ.")
    sample_indices = np.linspace(
        0, len(records) - 1, sample_count, dtype=np.int64
    )

    # Batch=1 BF16/FlashAttention-2 là reference có memory an toàn trên A100 40 GB.
    # Checkpoint gốc chứa BF16 weights, nên full FP32 không phục hồi precision weight.
    reference_chunks = []
    for idx in tqdm(sample_indices, desc="Audit reference batch=1", unit="img"):
        cpu_inputs = preprocess_image_paths(
            [records[int(idx)].image_path],
            precast_pixel_values=CONFIG["precast_pixel_values_bf16"],
        )
        gpu_inputs = move_inputs_to_cuda(cpu_inputs)
        reference_chunks.append(forward_normalized(gpu_inputs).cpu())
        del cpu_inputs, gpu_inputs
    reference = torch.cat(reference_chunks, dim=0)

    loader = build_loader(
        records, sample_indices, audit_batch_size, workers=min(NUM_WORKERS, 2)
    )
    selected_chunks = []
    for gpu_inputs, _ in CUDAPrefetcher(loader):
        selected_chunks.append(forward_normalized(gpu_inputs).cpu())
        del gpu_inputs
    selected = torch.cat(selected_chunks, dim=0)
    cosine = F.cosine_similarity(selected, reference, dim=-1).float()
    norms = selected.norm(dim=-1).float()
    all_finite = bool(torch.isfinite(selected).all().item())
    audit = {
        "sample_count": sample_count,
        "reference": "official_pipeline_bf16_flash_attention_2_batch1",
        "production_batch_size": audit_batch_size,
        "production_vs_batch1_min_cosine": cosine.min().item(),
        "production_vs_batch1_mean_cosine": cosine.mean().item(),
        "normalized_norm_min": norms.min().item(),
        "normalized_norm_max": norms.max().item(),
        "required_min_cosine": CONFIG["numeric_audit_min_cosine"],
        "required_mean_cosine": CONFIG["numeric_audit_mean_cosine"],
    }
    if (
        not all_finite
        or audit["production_vs_batch1_min_cosine"] < CONFIG["numeric_audit_min_cosine"]
        or audit["production_vs_batch1_mean_cosine"] < CONFIG["numeric_audit_mean_cosine"]
    ):
        raise RuntimeError(f"Numeric quality audit fail: {audit}")
    print("Numeric quality audit:", audit)
    del loader, reference, selected
    gc.collect()
    return audit


# Runtime-specific extension: giữ nguyên ba gate canonical phía trên, sau đó mới
# so vector Torch 2.11 với shard v3 đã commit từ runtime cũ trước khi cho phép ghi.
def legacy_v3_cross_runtime_audit(records, valid_commits, production_batch_size):
    """Gate Torch 2.11/FA2 2.8.3 against vectors already committed by v3."""
    if not valid_commits:
        raise RuntimeError(
            "Category đầu tiên chưa hoàn tất không có shard v3 cũ để cross-runtime "
            "audit; từ chối ghi bằng compatibility mode."
        )
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_id = sorted(valid_commits)[0]
    commit = valid_commits[shard_id]
    remote_path = commit["parquet_path"]
    remote_item = remote_info(CONFIG["output_bucket_id"], remote_path)
    if remote_item is None:
        raise RuntimeError(f"Thiếu legacy audit parquet: {remote_path}")

    audit_dir = STATE_ROOT / "legacy_cross_runtime_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    local_path = audit_dir / f"{commit['category']}-part-{shard_id:05d}.parquet"
    local_path.unlink(missing_ok=True)
    download_bucket_files(
        CONFIG["output_bucket_id"],
        files=[(remote_item, str(local_path))],
        token=HF_TOKEN,
    )
    if local_path.stat().st_size != int(commit["parquet_bytes"]):
        raise RuntimeError("Legacy audit parquet size mismatch.")
    if sha256_file(local_path) != commit["parquet_sha256"]:
        raise RuntimeError("Legacy audit parquet SHA-256 mismatch.")

    table = pq.read_table(local_path, columns=["frame_id", "embedding"])
    if table.num_rows != int(commit["rows"]):
        raise RuntimeError("Legacy audit parquet row count mismatch.")
    sample_count = min(
        int(CONFIG["legacy_cross_runtime_audit_samples"]), table.num_rows
    )
    positions = np.linspace(0, table.num_rows - 1, sample_count, dtype=np.int64)
    sampled = table.take(pa.array(positions))
    old_frame_ids = sampled.column("frame_id").to_pylist()
    old_embeddings = torch.from_numpy(np.asarray(
        sampled.column("embedding").to_pylist(), dtype=np.float32
    ))

    row_start = int(commit["row_start"])
    record_indices = row_start + positions
    expected_frame_ids = [records[int(idx)].frame_id for idx in record_indices]
    if old_frame_ids != expected_frame_ids:
        raise RuntimeError("Legacy audit frame_id/order mismatch.")

    new_chunks = []
    audit_batch_size = min(int(production_batch_size), sample_count)
    for begin in range(0, sample_count, audit_batch_size):
        batch_indices = record_indices[begin:begin + audit_batch_size]
        paths = [records[int(idx)].image_path for idx in batch_indices]
        cpu_inputs = preprocess_image_paths(
            paths, precast_pixel_values=CONFIG["precast_pixel_values_bf16"]
        )
        gpu_inputs = move_inputs_to_cuda(cpu_inputs)
        new_chunks.append(forward_normalized(gpu_inputs).cpu())
        del cpu_inputs, gpu_inputs
    new_embeddings = torch.cat(new_chunks, dim=0).float()
    cosine = F.cosine_similarity(new_embeddings, old_embeddings, dim=-1).float()
    audit = {
        "sample_count": sample_count,
        "category": commit["category"],
        "legacy_shard_id": shard_id,
        "legacy_flash_attn_version": CONFIG["legacy_semantic_flash_attn_version"],
        "runtime_flash_attn_version": runtime_versions["flash_attn"],
        "runtime_torch": torch.__version__,
        "min_cosine": cosine.min().item(),
        "mean_cosine": cosine.mean().item(),
        "required_min_cosine": CONFIG["legacy_cross_runtime_min_cosine"],
        "required_mean_cosine": CONFIG["legacy_cross_runtime_mean_cosine"],
    }
    if (
        not bool(torch.isfinite(new_embeddings).all().item())
        or audit["min_cosine"] < CONFIG["legacy_cross_runtime_min_cosine"]
        or audit["mean_cosine"] < CONFIG["legacy_cross_runtime_mean_cosine"]
    ):
        raise RuntimeError(f"Legacy v3 cross-runtime audit fail: {audit}")
    print("Legacy v3 cross-runtime audit:", audit)
    local_path.unlink(missing_ok=True)
    del table, sampled, old_embeddings, new_embeddings, new_chunks
    gc.collect()
    return audit


# Chọn category đầu tiên chưa hoàn tất để benchmark trên ảnh thật.
FIRST_RECORDS = None
FIRST_CATEGORY = None
FIRST_VALID_COMMITS = None
for candidate_category in CONFIG["categories"]:
    candidate_records = load_category_records(candidate_category)
    valid = load_valid_remote_commits(candidate_category, candidate_records)
    if len(valid) < total_shards(len(candidate_records)):
        FIRST_CATEGORY = candidate_category
        FIRST_RECORDS = candidate_records
        FIRST_VALID_COMMITS = valid
        break

if FIRST_RECORDS is not None:
    prepare_category_images(FIRST_CATEGORY, FIRST_RECORDS)
    assert_preprocess_parity(FIRST_RECORDS)
    BATCH_SIZE, BATCH_BENCHMARK = benchmark_batch_sizes(FIRST_RECORDS)
    OFFICIAL_WRAPPER_AUDIT = official_wrapper_embedding_audit(FIRST_RECORDS)
    NUMERIC_AUDIT = numeric_quality_audit(FIRST_RECORDS, BATCH_SIZE)
    LEGACY_CROSS_RUNTIME_AUDIT = legacy_v3_cross_runtime_audit(
        FIRST_RECORDS, FIRST_VALID_COMMITS, BATCH_SIZE
    )
else:
    BATCH_SIZE = int(CONFIG["batch_size"] or 1)
    BATCH_BENCHMARK = []
    OFFICIAL_WRAPPER_AUDIT = {}
    NUMERIC_AUDIT = {}
    LEGACY_CROSS_RUNTIME_AUDIT = {}
    print("Mọi category được chọn đã hoàn tất; bỏ qua autotune.")

EXECUTION_INFO = {
    "started_at": utc_now(),
    "session_name": CONFIG["session_name"],
    "host": socket.gethostname(),
    "gpu": GPU_NAME,
    "gpu_vram_gib": GPU_VRAM_GIB,
    "system_ram_gib": SYSTEM_RAM_GIB,
    "batch_size": BATCH_SIZE,
    "batch_benchmark": BATCH_BENCHMARK,
    "num_workers": NUM_WORKERS,
    "prefetch_factor": CONFIG["prefetch_factor"],
    "persistent_workers": CONFIG["persistent_workers"],
    "official_wrapper_audit": OFFICIAL_WRAPPER_AUDIT,
    "numeric_audit": NUMERIC_AUDIT,
    "legacy_cross_runtime_audit": LEGACY_CROSS_RUNTIME_AUDIT,
    "attention_implementation": CONFIG["attention_implementation"],
    "flash_attn_runtime_version": runtime_versions["flash_attn"],
    "legacy_semantic_flash_attn_version": CONFIG[
        "legacy_semantic_flash_attn_version"
    ],
    "flash_attn_compatibility_mode": "v2.8.3_kernel_equivalent_to_v2.8.3.post1",
    "autocast": False,
    "image_instruction": CONFIG["image_instruction"],
    "min_pixels": CONFIG["min_pixels"],
    "max_pixels": CONFIG["max_pixels"],
    "torch": torch.__version__,
    "torchvision": importlib.metadata.version("torchvision"),
    **runtime_versions,
}
if CONFIG["upload"]:
    execution_name = (
        f"{CONFIG['session_name']}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    upload_bytes_verified(
        CONFIG["output_bucket_id"],
        f"{CONFIG['output_prefix'].strip('/')}/executions/{execution_name}",
        json.dumps(EXECUTION_INFO, ensure_ascii=False, indent=2).encode("utf-8"),
    )


# %% [markdown]
# ## 9. Ghi Parquet shard và encode E2E

# %%
import pyarrow as pa
import pyarrow.parquet as pq


def storage_numpy_dtype():
    return np.float32 if CONFIG["storage_dtype"] == "float32" else np.float16


def write_parquet_shard(
    category: str,
    shard_id: int,
    records: Sequence[FrameRecord],
    embeddings_fp32: np.ndarray,
    row_indices: np.ndarray,
):
    start, stop = shard_bounds(len(records), shard_id)
    expected_indices = np.arange(start, stop, dtype=np.int64)
    if not np.array_equal(row_indices, expected_indices):
        raise RuntimeError(f"{category} shard {shard_id}: row order không deterministic.")
    if embeddings_fp32.shape != (stop - start, CONFIG["embedding_dim"]):
        raise RuntimeError(f"Embedding shape sai: {embeddings_fp32.shape}")
    if not np.isfinite(embeddings_fp32).all():
        raise RuntimeError(f"{category} shard {shard_id}: NaN/Inf embedding.")

    norms = np.linalg.norm(embeddings_fp32, axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if max_norm_error > 2e-5:
        raise RuntimeError(f"{category} shard {shard_id}: norm error {max_norm_error}")

    selected_records = records[start:stop]
    stored = np.ascontiguousarray(embeddings_fp32.astype(storage_numpy_dtype(), copy=False))
    value_type = pa.float32() if stored.dtype == np.float32 else pa.float16()
    flat = pa.array(stored.reshape(-1), type=value_type)
    embedding_array = pa.FixedSizeListArray.from_arrays(flat, CONFIG["embedding_dim"])

    table = pa.Table.from_arrays(
        [
            pa.array([r.frame_id for r in selected_records], type=pa.string()),
            pa.array([r.video_id for r in selected_records], type=pa.string()),
            pa.array([r.category for r in selected_records], type=pa.string()),
            pa.array([r.frame_idx for r in selected_records], type=pa.int64()),
            pa.array([r.pts_time for r in selected_records], type=pa.float64()),
            pa.array([r.fps for r in selected_records], type=pa.float32()),
            pa.array([r.image_relpath for r in selected_records], type=pa.string()),
            embedding_array,
        ],
        names=[
            "frame_id", "video_id", "category", "frame_idx", "pts_time", "fps",
            "image_relpath", "embedding",
        ],
    )
    arrow_metadata = {
        b"schema_version": b"1",
        b"semantic_fingerprint": SEMANTIC_FINGERPRINT.encode(),
        b"model_id": CONFIG["model_id"].encode(),
        b"model_revision": CONFIG["model_revision"].encode(),
        b"checkpoint_index_sha256": MODEL_INDEX_SHA256.encode(),
        b"image_instruction": CONFIG["image_instruction"].encode(),
        b"max_pixels": str(CONFIG["max_pixels"]).encode(),
        b"pooling": b"last_attended_token",
        b"embedding_dim": str(CONFIG["embedding_dim"]).encode(),
        b"embedding_dtype": CONFIG["storage_dtype"].encode(),
        b"l2_normalized": b"true",
    }
    table = table.replace_schema_metadata(arrow_metadata)

    local_dir = LOCAL_OUTPUT_ROOT / "embeddings" / category
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"part-{shard_id:05d}.parquet"
    partial_path = local_path.with_suffix(".parquet.partial")
    pq.write_table(
        table,
        partial_path,
        compression=None,
        use_dictionary=["video_id", "category"],
        row_group_size=2_048,
        write_statistics=["frame_id", "video_id", "frame_idx", "pts_time"],
    )
    os.replace(partial_path, local_path)
    parquet_sha = sha256_file(local_path)
    parquet_bytes = local_path.stat().st_size
    parquet_remote = remote_parquet_path(category, shard_id)

    commit = {
        "schema_version": 1,
        "created_at": utc_now(),
        "semantic_fingerprint": SEMANTIC_FINGERPRINT,
        "category": category,
        "shard_id": shard_id,
        "row_start": start,
        "row_stop": stop,
        "rows": stop - start,
        "first_frame_id": records[start].frame_id,
        "last_frame_id": records[stop - 1].frame_id,
        "parquet_path": parquet_remote,
        "parquet_bytes": parquet_bytes,
        "parquet_sha256": parquet_sha,
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
            "compute_precision": CONFIG["compute_precision"],
            "autocast": False,
            "attention_implementation": CONFIG["attention_implementation"],
            "flash_attn_runtime_version": runtime_versions["flash_attn"],
            "legacy_semantic_flash_attn_version": CONFIG[
                "legacy_semantic_flash_attn_version"
            ],
            "num_workers": NUM_WORKERS,
            "prefetch_factor": CONFIG["prefetch_factor"],
            "persistent_workers": CONFIG["persistent_workers"],
            "torch": torch.__version__,
        },
    }

    if CONFIG["upload"]:
        # Data first, commit last: commit là transaction boundary.
        upload_file_verified(CONFIG["output_bucket_id"], parquet_remote, local_path)
        upload_bytes_verified(
            CONFIG["output_bucket_id"],
            remote_commit_path(category, shard_id),
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


def active_indices_for_incomplete_shards(total_rows: int, completed_shards: set):
    ranges = []
    for shard_id in range(total_shards(total_rows)):
        if shard_id not in completed_shards:
            start, stop = shard_bounds(total_rows, shard_id)
            ranges.append(np.arange(start, stop, dtype=np.int64))
    return np.concatenate(ranges) if ranges else np.empty(0, dtype=np.int64)


def process_category(category: str, records: Optional[Sequence[FrameRecord]] = None):
    records = records if records is not None else load_category_records(category)
    completed = load_valid_remote_commits(category, records)
    expected_shards = total_shards(len(records))
    if len(completed) == expected_shards:
        print(f"{category}: đã hoàn tất {len(records):,} rows / {expected_shards} shards; skip.")
        return {"category": category, "rows": len(records), "shards": expected_shards, "skipped": True}

    category_dir = prepare_category_images(category, records)
    active_indices = active_indices_for_incomplete_shards(len(records), set(completed))
    print(
        f"{category}: resume {len(completed)}/{expected_shards} shards; "
        f"còn encode {len(active_indices):,}/{len(records):,} ảnh."
    )
    loader = build_loader(records, active_indices, BATCH_SIZE)
    progress = tqdm(total=len(active_indices), desc=f"Encode {category}", unit="img", smoothing=0.05)

    embedding_chunks: Dict[int, List[np.ndarray]] = {}
    index_chunks: Dict[int, List[np.ndarray]] = {}
    buffered_rows: Dict[int, int] = {}
    new_commits = []
    torch.cuda.reset_peak_memory_stats()

    try:
        for gpu_inputs, row_indices_t in CUDAPrefetcher(loader):
            normalized = forward_normalized(gpu_inputs)
            embeddings = normalized.cpu().numpy().astype(np.float32, copy=False)
            row_indices = row_indices_t.numpy().astype(np.int64, copy=False)

            shard_ids = row_indices // CONFIG["shard_rows"]
            boundaries = np.flatnonzero(np.diff(shard_ids)) + 1
            splits = np.split(np.arange(len(row_indices)), boundaries)
            for positions in splits:
                shard_id = int(shard_ids[positions[0]])
                embedding_chunks.setdefault(shard_id, []).append(embeddings[positions].copy())
                index_chunks.setdefault(shard_id, []).append(row_indices[positions].copy())
                buffered_rows[shard_id] = buffered_rows.get(shard_id, 0) + len(positions)
                start, stop = shard_bounds(len(records), shard_id)
                if buffered_rows[shard_id] == stop - start:
                    shard_embeddings = np.concatenate(embedding_chunks.pop(shard_id), axis=0)
                    shard_indices = np.concatenate(index_chunks.pop(shard_id), axis=0)
                    commit = write_parquet_shard(
                        category, shard_id, records, shard_embeddings, shard_indices
                    )
                    new_commits.append(commit)
                    buffered_rows.pop(shard_id)
                    del shard_embeddings, shard_indices
                elif buffered_rows[shard_id] > stop - start:
                    raise RuntimeError(f"Buffer overflow ở {category} shard {shard_id}")
            progress.update(len(row_indices))
            del gpu_inputs, normalized, embeddings, row_indices
    finally:
        progress.close()
        del loader

    if embedding_chunks or index_chunks or buffered_rows:
        raise RuntimeError(f"{category}: còn shard buffer chưa flush: {list(buffered_rows)}")

    verified = load_valid_remote_commits(category, records) if CONFIG["upload"] else {
        commit["shard_id"]: commit for commit in new_commits
    }
    if len(verified) != expected_shards:
        raise RuntimeError(f"{category}: final commits {len(verified)} != {expected_shards}")

    category_success = {
        "schema_version": 1,
        "completed_at": utc_now(),
        "semantic_fingerprint": SEMANTIC_FINGERPRINT,
        "category": category,
        "rows": len(records),
        "shards": expected_shards,
        "session_name": CONFIG["session_name"],
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    if CONFIG["upload"]:
        upload_bytes_verified(
            CONFIG["output_bucket_id"],
            f"{CONFIG['output_prefix'].strip('/')}/success/{category}.json",
            json.dumps(category_success, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    if CONFIG["cleanup_images_after_category"] and CONFIG["upload"]:
        assert_safe_category_path(category_dir, category)
        shutil.rmtree(category_dir)
        marker_path = STATE_ROOT / f"{category}_extracted.json"
        if marker_path.exists():
            marker_path.unlink()
        print(f"Đã xóa scratch JPEG {category}; dữ liệu nguồn vẫn an toàn trên Bucket.")
    return category_success


CATEGORY_RESULTS = []
for category in CONFIG["categories"]:
    cached_records = FIRST_RECORDS if category == FIRST_CATEGORY else None
    CATEGORY_RESULTS.append(process_category(category, cached_records))
    if category == FIRST_CATEGORY:
        FIRST_RECORDS = None
    gc.collect()


# %% [markdown]
# ## 10. Global audit và success manifest
#
# Mỗi runtime ghi audit riêng vào `audits/`. Global manifest và `_SUCCESS.json` chỉ
# được ghi khi **cả L21..L30** đủ commit; `global_complete=false` ở session về sớm
# là bình thường, không phải lỗi.

# %%
def final_global_audit():
    per_category = {}
    total_rows = 0
    all_complete = True
    for category in [f"L{i:02d}" for i in range(21, 31)]:
        records = load_category_records(category)
        commits = load_valid_remote_commits(category, records) if CONFIG["upload"] else {}
        expected = total_shards(len(records))
        committed_rows = sum(int(c["rows"]) for c in commits.values())
        complete = len(commits) == expected and committed_rows == len(records)
        per_category[category] = {
            "expected_rows": len(records),
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
        "expected_total_rows": sum(CONFIG["expected_counts"].values()),
        "complete": bool(all_complete),
        "retrieval_contract": QUERY_CONTRACT,
    }
    if CONFIG["upload"]:
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        audit_name = (
            f"{CONFIG['session_name']}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        # Partial audit là per-session; không cho runtime chưa hoàn tất ghi đè global manifest.
        upload_bytes_verified(
            CONFIG["output_bucket_id"],
            f"{CONFIG['output_prefix'].strip('/')}/audits/{audit_name}",
            manifest_bytes,
        )
        if all_complete:
            upload_bytes_verified(
                CONFIG["output_bucket_id"],
                f"{CONFIG['output_prefix'].strip('/')}/embedding_dataset_manifest.json",
                manifest_bytes,
            )
            upload_bytes_verified(
                CONFIG["output_bucket_id"],
                f"{CONFIG['output_prefix'].strip('/')}/_SUCCESS.json",
                manifest_bytes,
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
# ## Retrieval contract sau khi encode
#
# - Image/document side đã cố định: system `Represent the user's input.` + image,
#   official pixel budget 4.096..1.843.200, last-token pooling, 4096-d FP32 unit norm.
# - Query baseline: system `Retrieve images or text relevant to the user's query.`
#   + user text, cùng model/revision, last-token pooling rồi FP32 L2-normalize.
# - Có thể A/B query instruction sau này mà không encode lại ảnh; tuyệt đối không
#   thay image-side instruction rồi trộn vector vào collection hiện tại.
# - Milvus: `FLOAT_VECTOR(4096)` + `COSINE`.
# - Không cộng cosine của PE với model khác trực tiếp. Khi ensemble PE/Qwen/SigLIP,
#   fuse bằng rank/RRF vì distribution score của từng embedding space khác nhau.
#
# Query reference:
#
# ```python
# query = "một người đang đua xe đạp trên đường"
# query_vec = QWEN_EMBEDDER.process([{
#     "text": query,
#     "instruction": "Retrieve images or text relevant to the user's query.",
# }], normalize=False)
# query_vec = F.normalize(query_vec.float(), p=2, dim=-1)
# ```
#
# Output layout:
#
# ```text
# derived/qwen3-vl-embedding-8b-4096-v3/
# ├── run_config.json
# ├── executions/*.json
# ├── audits/*.json
# ├── embeddings/Lxx/part-*.parquet
# ├── commits/Lxx/part-*.json
# ├── success/Lxx.json
# ├── embedding_dataset_manifest.json
# └── _SUCCESS.json                 # chỉ có khi đủ 1.339.055 rows
# ```
