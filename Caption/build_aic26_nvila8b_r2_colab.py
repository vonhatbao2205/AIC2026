"""Build the checked-in Colab notebook for AIC26 NVILA-8B captioning.

The notebook itself is generated so that its JSON stays deterministic and all
code cells remain easy to syntax-check in CI/local review.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent, indent


def _source(text: str) -> list[str]:
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


cells = [
    markdown(
        r"""
        # AIC26 — NVILA-8B Captioning E2E trên Colab + Cloudflare R2

        Notebook production-oriented để caption toàn bộ keyframe AIC26 bằng
        **`Efficient-Large-Model/NVILA-8B-hf`**:

        ```text
        R2 keyframes ─→ Pass A: target-only Dynamic-S2 ─→ keyframe visual index
                                                             ↕ segment IDs only
        R2 videos ────→ Pass B: overlapping shot windows ─→ temporal segment index
                               ↓
        schema validation + independent resumable/versioned R2 JSONL parts
                               ↓
                  QA + optional two-index Elasticsearch bulk
        ```

        ## Runtime khuyến nghị

        - **Colab A100 80 GB**: cấu hình mặc định và đáng tin cậy nhất.
        - **A100 40 GB**: có thể thử inference tuần tự, nhưng Dynamic-S2/một số ảnh rộng có
          thể OOM. Notebook không CPU-offload vì rất chậm và dễ lỗi với multimodal generation.
        - Host RAM nên từ 40 GB; output được chia part, không giữ ảnh toàn dataset trên RAM.

        Checkpoint được pin:

        - Model: `Efficient-Large-Model/NVILA-8B-hf`
        - Revision: `e722ed6889396b54ff37a10fcfeabf8c2df3c935`
        - Transformers: `4.55.4`
        - Weights: BF16, khoảng 16.2 GB

        > NVILA weights dành cho research/non-commercial theo model card. Kiểm tra license
        > trước khi dùng ngoài phạm vi cuộc thi/nghiên cứu.

        ## Colab Secrets cần tạo

        Vào **Colab → Secrets** và thêm:

        - `R2_ACCOUNT_ID`
        - `R2_ACCESS_KEY_ID`
        - `R2_SECRET_ACCESS_KEY`
        - `R2_BUCKET` (có thể bỏ, mặc định `aic26-media`)
        - `HF_TOKEN` (optional; model hiện public)
        - `R2_MEDIA_BASE_URL` (optional; để ghi `keyframe_url`)

        Pass B mặc định dùng các secret sau để đọc true timing từ keyframe-map Elastic.
        Chúng cũng được tái sử dụng nếu bật bulk index ở cuối notebook:

        - `ELASTIC_ENDPOINT` (mặc định cần cho Pass B để lấy true `pts_time`)
        - `ELASTIC_API_KEY` (mặc định cần cho Pass B)

        Notebook không đọc file secret local và không in credential.
        """
    ),
    code(
        r"""
        # Cài dependency trước khi import transformers.
        # Không reinstall torch/torchvision: dùng CUDA build sẵn của Colab.
        %pip -q install --upgrade \
          "transformers==4.55.4" \
          "accelerate>=1.2,<2" \
          "huggingface_hub>=0.34,<1" \
          "boto3>=1.35,<2" \
          "botocore>=1.35,<2" \
          "einops>=0.8,<1" \
          "jinja2>=3.1,<4" \
          "pillow>=10,<12" \
          "pandas>=2.1,<3" \
          "requests>=2.31,<3" \
          "ffmpeg-python==0.2.0" \
          "transnetv2-pytorch @ https://files.pythonhosted.org/packages/85/be/bb14f9015d357c9dff395a065bd658d1f7a199eb1f8d3abb9f4792416f11/transnetv2_pytorch-1.0.5-py3-none-any.whl#sha256=9f8e72085526aaa95383d219b6750b1fa45b865fd10d840cafa12ef78ab3bf27" \
          "tqdm>=4.66,<5"

        print("Dependencies installed. Nếu Colab đã import transformers trước cell này, restart runtime một lần.")
        """
    ),
    code(
        r"""
        import os

        # Giảm fragmentation trên các lượt generate có số visual token khác nhau.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        import csv
        import copy
        import gc
        import hashlib
        import importlib.metadata
        import io
        import json
        import math
        import random
        import re
        import time
        import traceback
        import zlib
        from collections import OrderedDict, defaultdict
        from dataclasses import asdict, dataclass
        from datetime import datetime, timezone
        from pathlib import Path
        from typing import Any, Iterable

        import boto3
        import cv2
        import numpy as np
        import pandas as pd
        import torch
        from botocore.config import Config as BotoConfig
        from botocore.exceptions import ClientError
        from PIL import Image
        from tqdm.auto import tqdm

        assert torch.cuda.is_available(), "Chọn Runtime → Change runtime type → A100 GPU."
        gpu = torch.cuda.get_device_properties(0)
        VRAM_GIB = gpu.total_memory / 2**30
        print("GPU:", gpu.name)
        print(f"VRAM: {VRAM_GIB:.1f} GiB")
        print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)

        if VRAM_GIB < 38:
            raise RuntimeError("Notebook NVILA-8B BF16 này yêu cầu A100 40 GB trở lên; A100 80 GB được khuyến nghị cho temporal batch 8.")
        if VRAM_GIB < 55:
            print("WARNING: đang ở chế độ 40 GB thử nghiệm. Nếu OOM, chuyển sang A100 80 GB.")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        """
    ),
    code(
        r"""
        # =========================== CONFIG — EDIT CELL NÀY ===========================
        @dataclass(frozen=True)
        class CaptionConfig:
            # Model được pin để remote custom code/weights không đổi giữa các session.
            model_id: str = "Efficient-Large-Model/NVILA-8B-hf"
            model_revision: str = "e722ed6889396b54ff37a10fcfeabf8c2df3c935"
            transformers_version: str = "4.55.4"
            schema_version: str = "4.1.0"
            prompt_version: str = "aic26-nvila8b-caption-v4.1.0"
            # Tăng version này mỗi khi normalize/sanitize visual output thay đổi.
            # Nó thuộc visual contract để checkpoint tạo bằng normalizer cũ
            # không bị resume nhầm sau khi logic hậu xử lý được cập nhật.
            visual_normalizer_version: str = "aic26-visual-normalizer-v3"

            # R2 input. Nếu có map thật chứa frame_idx/pts_time/fps, đổi key tại đây.
            # media_manifest.csv hiện tại chỉ chắc chắn có identity + r2_key.
            source_manifest_key: str = "manifest/media_manifest.csv"
            keyframe_prefix: str = "Keyframes/"
            # Pass B cần true pts_time. Tắt nếu source manifest đã có timing đầy đủ.
            enrich_timing_from_elastic: bool = True
            elastic_keyframe_map_index: str = "aic26_keyframe_map_v1"
            # Optional explicit immutable release tag. Nếu None, notebook fingerprint
            # index UUID/creation_date/docs_count/max_seq_no từ Elastic.
            timing_dataset_version: str | None = None

            # R2 output. Mỗi shard phải chỉ có MỘT writer tại một thời điểm.
            output_prefix: str = "caption/nvila8b/v4"
            shard_count: int = 16
            shard_index: int = 0
            part_size: int = 100
            checkpoint_every: int = 10

            # Smoke mặc định lấy 64 target deterministic, ưu tiên nhiều group/video.
            run_mode: str = "smoke"
            smoke_limit: int = 64
            smoke_seed: int = 26
            smoke_sampling_strategy: str = "group_video_diverse"
            include_groups: tuple[str, ...] = ()
            exclude_groups: tuple[str, ...] = ()
            video_id_min: str | None = None
            video_id_max: str | None = None

            # Production default: target-only để giữ Dynamic-S2 high-resolution.
            # Nếu radius>0, processor chuyển sang multi-image 448 và target mất Dynamic-S2.
            # Sparse neighbors cũng KHÔNG phải consecutive video frames.
            context_radius: int = 0
            # 8B dễ kéo dài hoặc bỏ phần cuối JSON hơn 15B. Đây chỉ là trần
            # generation; model vẫn dừng sớm khi sinh EOS nên không ép mọi output
            # chạy đủ 640 token.
            visual_max_new_tokens: int = 640
            temporal_max_new_tokens: int = 384
            parse_retries: int = 1
            inference_retries: int = 2
            image_cache_size: int = 24
            # manifest_or_head: dùng ETag/size trong manifest; nếu thiếu thì HEAD
            # các target đã checkpoint khi resume. manifest_required tránh HEAD hàng loạt.
            visual_resume_source_check: str = "manifest_or_head"

            # Chỉ dùng dedup khi context_radius=0; record duplicate vẫn được xuất đầy đủ.
            enable_dedup: bool = False
            dedup_hamming_threshold: int = 5
            dedup_lookback: int = 8

            # Pass B: original video -> overlapping shot windows -> temporal segment index.
            run_temporal_pass: bool = True
            temporal_prompt_version: str = "aic26-nvila8b-temporal-v2.1.0"
            temporal_segmenter: str = "transnetv2_gpu"
            transnetv2_package_version: str = "1.0.5"
            transnetv2_wheel_sha256: str = "9f8e72085526aaa95383d219b6750b1fa45b865fd10d840cafa12ef78ab3bf27"
            transnetv2_weights_sha256: str = "a313d0b3bebfa9a71914b375bfdf918a30b5c3b1e6be51972d35dd8078b442de"
            transnetv2_threshold: float = 0.5
            transnetv2_input_height: int = 27
            transnetv2_input_width: int = 48
            transnetv2_max_duration_mismatch_ratio: float = 0.01
            temporal_min_scene_len_frames: int = 15
            temporal_window_seconds: float = 8.0
            temporal_stride_seconds: float = 4.0
            temporal_sample_frames: int = 8
            # Số VIDEO SEGMENT đưa qua NVILA trong một generate call. Đây không
            # phải số frame/segment. Batch 8 yêu cầu A100 80 GB; khi OOM notebook
            # tự chia đôi batch và vẫn checkpoint riêng từng segment.
            temporal_batch_size: int = 8
            temporal_oom_split: bool = True
            temporal_frame_sampler_version: str = (
                "opencv-seek+ffmpeg-fps-pipe-fallback-v1"
            )
            temporal_part_size: int = 50
            temporal_checkpoint_every: int = 5
            temporal_keep_video_cache: bool = False
            temporal_fail_on_missing_timing: bool = True
            allow_scene_fallback: bool = False
            temporal_stop_on_consecutive_video_errors: int = 5
            temporal_stop_on_total_segment_errors: int = 50

            # An toàn vận hành.
            stop_on_error_count: int = 100
            write_combined_shard_at_end: bool = False

        CFG = CaptionConfig(
            shard_count=16,
            shard_index=0,       # chạy lần lượt 0..15; có thể chạy song song trên các Colab khác nhau
            run_mode="smoke",   # đổi thành "full" sau khi xem smoke output
            smoke_limit=64,
            smoke_seed=26,
            smoke_sampling_strategy="group_video_diverse",
            context_radius=0,
            run_temporal_pass=True,
            timing_dataset_version="aic26-keyframe-map-v1-20260729-v1",
        )

        assert 0 <= CFG.shard_index < CFG.shard_count
        assert CFG.run_mode in {"smoke", "full"}
        assert CFG.smoke_limit > 0
        assert CFG.smoke_sampling_strategy in {"group_video_diverse", "random_frames"}
        assert CFG.visual_resume_source_check in {
            "manifest_required",
            "manifest_or_head",
            "none",
        }
        assert CFG.part_size > 0 and 0 < CFG.checkpoint_every <= CFG.part_size
        assert CFG.temporal_segmenter in {"transnetv2_gpu", "fixed_windows"}
        assert 0.0 < CFG.transnetv2_threshold < 1.0
        assert CFG.transnetv2_input_height == 27
        assert CFG.transnetv2_input_width == 48
        assert 0.0 < CFG.transnetv2_max_duration_mismatch_ratio <= 0.1
        assert 2 <= CFG.temporal_sample_frames <= 32
        assert 1 <= CFG.temporal_batch_size <= 32
        assert CFG.temporal_window_seconds > 0
        assert 0 < CFG.temporal_stride_seconds <= CFG.temporal_window_seconds
        assert 0 < CFG.temporal_checkpoint_every <= CFG.temporal_part_size
        assert CFG.temporal_stop_on_consecutive_video_errors > 0
        assert CFG.temporal_stop_on_total_segment_errors > 0
        if CFG.enable_dedup and CFG.context_radius:
            raise ValueError("Dedup copy chỉ an toàn khi context_radius=0.")
        if CFG.run_temporal_pass and CFG.temporal_batch_size > 1 and VRAM_GIB < 70:
            raise RuntimeError(
                "temporal_batch_size > 1 chỉ được bật trên A100 80 GB. "
                "Với A100 40 GB hãy đặt temporal_batch_size=1."
            )

        LOCAL_ROOT = Path("/content/aic26_nvila8b_caption")
        LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
        print(json.dumps(asdict(CFG), indent=2, ensure_ascii=False))
        """
    ),
    code(
        r"""
        # =========================== COLAB SECRETS + R2 ===========================
        def get_secret(name: str, default: str | None = None, required: bool = False) -> str | None:
            value = os.environ.get(name)
            if not value:
                try:
                    from google.colab import userdata
                    value = userdata.get(name)
                except Exception:
                    value = None
            value = value or default
            if required and not value:
                raise RuntimeError(f"Thiếu secret {name} trong Colab Secrets.")
            return value


        R2_ACCOUNT_ID = get_secret("R2_ACCOUNT_ID", required=True)
        R2_ACCESS_KEY_ID = get_secret("R2_ACCESS_KEY_ID", required=True)
        R2_SECRET_ACCESS_KEY = get_secret("R2_SECRET_ACCESS_KEY", required=True)
        R2_BUCKET = get_secret("R2_BUCKET", "aic26-media")
        R2_MEDIA_BASE_URL = (get_secret("R2_MEDIA_BASE_URL", "") or "").rstrip("/")
        HF_TOKEN = get_secret("HF_TOKEN")
        R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

        r2 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto",
            config=BotoConfig(
                signature_version="s3v4",
                retries={"max_attempts": 8, "mode": "adaptive"},
                connect_timeout=15,
                read_timeout=90,
                max_pool_connections=24,
            ),
        )

        # Read-only connection check.
        r2.head_bucket(Bucket=R2_BUCKET)
        print("R2 connected:", R2_BUCKET, "| endpoint account:", str(R2_ACCOUNT_ID)[:4] + "…")


        def r2_get_bytes(key: str) -> tuple[bytes, str | None]:
            obj = r2.get_object(Bucket=R2_BUCKET, Key=key)
            body = obj["Body"].read()
            etag = str(obj.get("ETag", "")).strip('"') or None
            return body, etag


        def r2_put_bytes(
            key: str,
            body: bytes,
            *,
            content_type: str,
            metadata: dict[str, str] | None = None,
        ) -> None:
            r2.put_object(
                Bucket=R2_BUCKET,
                Key=key,
                Body=body,
                ContentType=content_type,
                Metadata=metadata or {},
            )


        def r2_object_exists(key: str) -> bool:
            try:
                r2.head_object(Bucket=R2_BUCKET, Key=key)
                return True
            except ClientError as exc:
                status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if status == 404:
                    return False
                raise


        def r2_list_keys(prefix: str) -> list[str]:
            keys: list[str] = []
            paginator = r2.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
                keys.extend(x["Key"] for x in page.get("Contents", []))
            return keys
        """
    ),
    code(
        r"""
        # =========================== IDENTITY + CAPTION SCHEMA ===========================
        ENVIRONMENTS = ["indoor", "outdoor", "mixed", "unknown"]
        SETTINGS = [
            "studio", "street", "stage", "vehicle_interior", "room", "office", "home",
            "hall", "shop", "factory", "school", "hospital", "field", "coastal",
            "highway", "road", "kitchen", "laboratory", "construction_site",
            "workshop", "market", "farm", "water", "sky", "graphic", "other", "unknown",
        ]
        LOCATION_TYPES = [
            "urban", "rural", "coastal", "highway", "field", "office", "home", "hall",
            "shop", "factory", "school", "hospital", "water", "sky", "unknown",
        ]
        TIME_OF_DAY = ["day", "night", "dawn_dusk", "indeterminate"]
        WEATHER = ["sunny", "cloudy", "rainy", "stormy", "foggy", "snowy", "not_applicable", "unknown"]
        LIGHTING = ["bright", "normal", "dim", "artificial", "backlit", "mixed", "unknown"]
        CAMERA_SHOTS = ["extreme_close_up", "close_up", "medium", "wide", "aerial", "graphic", "unknown"]
        CAMERA_ANGLES = ["eye_level", "high_angle", "low_angle", "top_down", "tilted", "unknown"]
        PEOPLE_BUCKETS = ["0", "1", "2", "few", "crowd", "unknown"]
        EVENT_TYPES = [
            "anchor_desk", "interview", "press_conference", "speech_podium", "outdoor_report",
            "accident", "fire", "flood", "storm", "traffic", "protest", "ceremony", "sports",
            "weather_map", "chart_graphic", "building_exterior", "agriculture", "military",
            "crowd", "meeting", "disaster", "market", "transport", "medical", "education",
            "cooking", "construction", "scientific_experiment", "fishing", "crafting",
            "transaction", "wildlife", "animal_activity", "performance",
        ]

        ENUM_ALIASES = {
            "setting": {
                "lab": "laboratory",
                "vehicle": "vehicle_interior",
                "inside_vehicle": "vehicle_interior",
                "construction": "construction_site",
                "construction_area": "construction_site",
            },
        }
        EVENT_TYPE_ALIASES = {
            "lab_experiment": "scientific_experiment",
            "laboratory_experiment": "scientific_experiment",
            "science_experiment": "scientific_experiment",
            "animal": "animal_activity",
            "animals": "animal_activity",
            "wildlife_activity": "wildlife",
            "commerce": "transaction",
            "shopping": "transaction",
            "performing": "performance",
        }

        ENUMS = {
            "environment": ENVIRONMENTS,
            "setting": SETTINGS,
            "location_type": LOCATION_TYPES,
            "time_of_day": TIME_OF_DAY,
            "weather": WEATHER,
            "lighting": LIGHTING,
            "camera_shot": CAMERA_SHOTS,
            "camera_angle": CAMERA_ANGLES,
            "people_count_bucket": PEOPLE_BUCKETS,
        }
        LIST_FIELDS = [
            "event_types", "people_roles", "key_objects", "dominant_colors",
            "visible_actions", "temporal_changes", "distinctive_details",
        ]
        CAPTION_SCHEMA_FIELDS = {
            "caption_free", "environment", "setting", "location_type", "time_of_day",
            "weather", "lighting", "camera_shot", "camera_angle", "num_people",
            "people_count_bucket", "event_types", "people_roles", "people_description",
            "key_objects", "dominant_colors", "visible_actions", "temporal_changes",
            "distinctive_details", "text_on_screen",
        }
        REQUIRED_CAPTION_KEYS = CAPTION_SCHEMA_FIELDS | {"uncertain_fields"}


        def group_from_video_id(video_id: str) -> str:
            match = re.match(r"^([KL]\d+)", video_id)
            if not match:
                raise ValueError(f"video_id không đúng convention AIC: {video_id!r}")
            return match.group(1)


        def canonical_identity(video_id: str, keyframe_n: int, r2_key: str | None = None) -> dict[str, Any]:
            group = group_from_video_id(video_id)
            frame_token = f"{int(keyframe_n):03d}"
            submit_id = f"{group}/{video_id}/{frame_token}"
            resolved_r2_key = r2_key or f"Keyframes/Keyframes_{group}/{video_id}/{frame_token}.jpg"
            return {
                "group": group,
                "category": group,
                "video_id": video_id,
                "keyframe_n": int(keyframe_n),
                "frame_name": f"{frame_token}.jpg",
                "submit_keyframe_id": submit_id,
                "image_id": submit_id,
                "r2_key": resolved_r2_key,
                "keyframe_url": (
                    f"{R2_MEDIA_BASE_URL}/{resolved_r2_key}" if R2_MEDIA_BASE_URL else None
                ),
            }


        CAPTION_PROMPT_TEMPLATE = '''
        You are producing retrieval metadata for ONE TARGET keyframe from a Vietnamese news-video dataset.
        You receive {image_count} image(s) in chronological order. The TARGET is IMAGE {target_position}.
        Other images, when present, are nearby sparse keyframes and not necessarily adjacent video frames.
        Use them only for coarse temporal context. Describe the TARGET, never merge different scenes.
        {temporal_instruction}

        Output exactly one valid JSON object and no markdown. Use English except text_on_screen.
        Do not identify real people by name. Do not infer an exact location, organization, relationship,
        cause, intent, or event when it is not visually supported. For unreadable text use an empty string.
        Treat OCR as a separate retrieval channel. caption_free, people_description, key_objects,
        visible_actions, temporal_changes, and distinctive_details MUST NOT transcribe or quote any
        text, number, logo, watermark, ticker, timestamp, subtitle, label, or sign. Those fields may
        describe a sign/screen as a visual object without stating what it says. Put every readable
        character sequence only in text_on_screen.

        Required schema:
        {{
          "caption_free": "Two concrete visual-only sentences; never transcribe or mention the content of on-screen text.",
          "environment": "one of {environments}",
          "setting": "one of {settings}",
          "location_type": "one of {location_types}",
          "time_of_day": "one of {time_of_day}",
          "weather": "one of {weather}",
          "lighting": "one of {lighting}",
          "camera_shot": "one of {camera_shots}",
          "camera_angle": "one of {camera_angles}",
          "num_people": 0,
          "people_count_bucket": "one of {people_buckets}",
          "event_types": ["zero or more of {event_types}"],
          "people_roles": ["generic visible roles such as anchor, reporter, official, athlete, civilian"],
          "people_description": "brief appearance/clothing description; empty if no people",
          "key_objects": ["salient visible objects, concrete lowercase nouns"],
          "dominant_colors": ["2-4 common color words"],
          "visible_actions": ["actions visibly supported in the target; present participles"],
          "temporal_changes": ["coarse changes supported by context; empty if uncertain"],
          "distinctive_details": ["1-3 visual details useful for retrieval; do not quote text"],
          "text_on_screen": "verbatim clearly readable text only; empty if none/unclear",
          "uncertain_fields": ["field names whose values are uncertain"]
        }}

        num_people must be a non-negative integer for clearly visible people, or null if a reliable count
        is impossible. For enums choose unknown/not_applicable instead of guessing.
        setting=vehicle_interior is valid only when the camera/scene is physically inside a vehicle;
        use street/road/highway for an exterior view of vehicles. Indoor facilities such as studio,
        kitchen, laboratory, office, room, hall, shop, factory, school, and hospital must not be paired
        with environment=outdoor; use unknown instead of returning a contradictory pair.
        '''.strip()


        def build_caption_prompt(image_count: int, target_position: int, repair_note: str = "") -> str:
            temporal_instruction = (
                "No temporal context is available. temporal_changes MUST be an empty JSON array; "
                "do not infer motion or change from a single image."
                if image_count == 1
                else
                "Temporal context is available, but report temporal_changes only when visibly supported "
                "across the supplied sparse keyframes."
            )
            prompt = CAPTION_PROMPT_TEMPLATE.format(
                image_count=image_count,
                target_position=target_position,
                temporal_instruction=temporal_instruction,
                environments=ENVIRONMENTS,
                settings=SETTINGS,
                location_types=LOCATION_TYPES,
                time_of_day=TIME_OF_DAY,
                weather=WEATHER,
                lighting=LIGHTING,
                camera_shots=CAMERA_SHOTS,
                camera_angles=CAMERA_ANGLES,
                people_buckets=PEOPLE_BUCKETS,
                event_types=EVENT_TYPES,
            )
            if repair_note:
                prompt += (
                    "\n\nYour previous response was invalid. Correct the JSON format and required values. "
                    "Return the full object again. Problem: " + repair_note[:500]
                )
            return prompt


        TEMPORAL_ACTIONS = [
            "speaking", "walking", "running", "standing_up", "sitting_down", "gesturing",
            "handing_object", "carrying", "opening", "closing", "entering", "leaving",
            "vehicle_moving", "vehicle_stopping", "camera_subject_approaching",
            "camera_subject_receding", "crowd_gathering", "fire_spreading", "water_flowing",
            "sports_play", "rowing", "dancing", "writing", "cooking", "stirring",
            "weaving", "loading", "clearing", "working", "tending", "fishing",
            "playing", "playing_instrument", "performing", "holding", "standing", "sitting",
            "kneeling", "camera_only_change", "static", "other",
        ]
        TEMPORAL_ACTION_ALIASES = {
            "row": "rowing",
            "rowing_boat": "rowing",
            "dance": "dancing",
            "write": "writing",
            "cook": "cooking",
            "mixing": "stirring",
            "load": "loading",
            "work": "working",
            "play_instrument": "playing_instrument",
            "playing_music": "playing_instrument",
            "holding_object": "holding",
            "driving": "vehicle_moving",
            "driving_forward": "vehicle_moving",
            "vehicle_driving": "vehicle_moving",
            "standing_still": "standing",
            "sitting_still": "sitting",
            "zoom_in": "camera_only_change",
            "zoom_out": "camera_only_change",
            "pan": "camera_only_change",
            "tilt": "camera_only_change",
        }
        CAMERA_MOTIONS = [
            "static", "pan", "tilt", "zoom", "zoom_in", "zoom_out",
            "tracking", "handheld", "cut", "unknown",
        ]
        CAMERA_MOTION_ALIASES = {
            "panning": "pan",
            "tilting": "tilt",
            "zooming": "zoom",
            "camera_pan": "pan",
            "camera_tilt": "tilt",
            "camera_zoom": "zoom",
            "none": "static",
        }
        EVENT_PHASES = ["starting", "ongoing", "ending", "transition", "static", "unclear"]
        TEMPORAL_CONFIDENCE = ["high", "medium", "low"]
        TEMPORAL_SCHEMA_FIELDS = {
            "temporal_caption", "action_motion", "camera_motion", "state_changes",
            "interaction_summary", "event_phase", "confidence",
        }
        REQUIRED_TEMPORAL_KEYS = TEMPORAL_SCHEMA_FIELDS | {"uncertain_fields"}

        TEMPORAL_PROMPT_TEMPLATE = '''
        You are analyzing one short chronological VIDEO SEGMENT from a Vietnamese news video.
        The frames are chronologically ordered samples from the same detected shot/window.
        They are not necessarily adjacent video frames. Describe only motion and state change visibly
        supported across frames. Ignore all on-screen text and broadcast overlays completely:
        never transcribe, quote, summarize, or mention text, numbers, logos, watermarks, tickers,
        timestamps, subtitles, labels, signs, banners, screens, or whiteboard writing.
        Do not identify real people by name and do not infer intent, cause, or an exact location.
        This output describes the WHOLE SEGMENT as retrieval context. It is not a target-keyframe caption.

        Output exactly one valid JSON object and no markdown:
        {{
          "temporal_caption": "One concrete visual-only English sentence describing the visible progression.",
          "action_motion": ["zero or more of {temporal_actions}"],
          "camera_motion": "one of {camera_motions}",
          "state_changes": ["concrete before-to-after changes; empty if none"],
          "interaction_summary": "brief interaction between visible people/objects; empty if none",
          "event_phase": "one of {event_phases}",
          "confidence": "one of {confidence}",
          "uncertain_fields": ["field names that are uncertain"]
        }}

        Use "static" only when no meaningful subject motion or state change is visible. For a static
        segment, temporal_caption must explicitly say that no meaningful motion/state change is visible
        and briefly name the unchanged visual subject. A camera pan/zoom alone is camera_only_change,
        not subject movement.

        state_changes is mandatory whenever the visible progression includes entering/leaving,
        appearing/disappearing, opening/closing, standing up/sitting down, starting/stopping, a
        before-to-after transition, or another concrete state change. Otherwise use [] and never guess.
        interaction_summary may describe only visible contact/exchange; do not infer purpose or intent.

        confidence="high" is allowed only for unambiguous motion visible across multiple samples.
        Use "medium" for partly observed or occluded motion and "low" for weak evidence. If any wording
        such as possibly, likely, appears, seems, may, or might is necessary, confidence cannot be high
        and uncertain_fields must list every affected field.
        '''.strip()


        def build_temporal_prompt(
            repair_note: str = "",
            recovery_round: int = 0,
        ) -> str:
            prompt = TEMPORAL_PROMPT_TEMPLATE.format(
                temporal_actions=TEMPORAL_ACTIONS,
                camera_motions=CAMERA_MOTIONS,
                event_phases=EVENT_PHASES,
                confidence=TEMPORAL_CONFIDENCE,
            )
            if recovery_round:
                prompt += (
                    "\n\nRECOVERY PASS "
                    f"{recovery_round}: Start the response with {{ and end it with }}. "
                    "Emit one complete JSON object containing every required key. "
                    "Do not continue a partial object and do not output prose."
                )
            if repair_note:
                prompt += (
                    "\n\nYour previous response was invalid. Return the full corrected JSON object. "
                    "Problem: " + repair_note[:500]
                )
            return prompt


        # Contract hash phải bao phủ cả template lẫn vocabulary/schema thực tế. Nếu chỉ
        # hash placeholder template, thay vocabulary mà quên bump version sẽ tái sử dụng
        # checkpoint không tương thích.
        VISUAL_PROMPT_SPEC = {
            "template": CAPTION_PROMPT_TEMPLATE,
            "environments": ENVIRONMENTS,
            "settings": SETTINGS,
            "location_types": LOCATION_TYPES,
            "time_of_day": TIME_OF_DAY,
            "weather": WEATHER,
            "lighting": LIGHTING,
            "camera_shots": CAMERA_SHOTS,
            "camera_angles": CAMERA_ANGLES,
            "people_buckets": PEOPLE_BUCKETS,
            "event_types": EVENT_TYPES,
            "enum_aliases": ENUM_ALIASES,
            "event_type_aliases": EVENT_TYPE_ALIASES,
            "required_keys": sorted(REQUIRED_CAPTION_KEYS),
        }
        TEMPORAL_PROMPT_SPEC = {
            "template": TEMPORAL_PROMPT_TEMPLATE,
            "temporal_actions": TEMPORAL_ACTIONS,
            "camera_motions": CAMERA_MOTIONS,
            "event_phases": EVENT_PHASES,
            "confidence": TEMPORAL_CONFIDENCE,
            "action_aliases": TEMPORAL_ACTION_ALIASES,
            "camera_motion_aliases": CAMERA_MOTION_ALIASES,
            "required_keys": sorted(REQUIRED_TEMPORAL_KEYS),
        }


        def prompt_spec_sha256(spec: dict[str, Any]) -> str:
            return hashlib.sha256(
                json.dumps(
                    spec,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()


        PROMPT_SHA256 = prompt_spec_sha256(VISUAL_PROMPT_SPEC)
        TEMPORAL_PROMPT_SHA256 = prompt_spec_sha256(TEMPORAL_PROMPT_SPEC)
        print("Schema:", CFG.schema_version)
        print("Prompt:", CFG.prompt_version, PROMPT_SHA256[:12])
        print("Temporal prompt:", CFG.temporal_prompt_version, TEMPORAL_PROMPT_SHA256[:12])
        """
    ),
    code(
        r"""
        # =========================== LOAD/STANDARDIZE R2 MANIFEST ===========================
        def _noneish(value: Any) -> bool:
            return value is None or str(value).strip().lower() in {"", "none", "null", "nan"}


        def _safe_int(value: Any) -> int | None:
            if _noneish(value):
                return None
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None


        def _safe_float(value: Any) -> float | None:
            if _noneish(value):
                return None
            try:
                value = float(value)
                return value if math.isfinite(value) else None
            except (TypeError, ValueError):
                return None


        def _rows_from_manifest_bytes(raw: bytes, key: str) -> list[dict[str, Any]]:
            suffix = Path(key).suffix.lower()
            if suffix == ".csv":
                return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
            if suffix in {".jsonl", ".ndjson"}:
                return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
            if suffix == ".json":
                data = json.loads(raw)
                return data if isinstance(data, list) else data.get("rows", [])
            raise ValueError("Manifest hỗ trợ .csv, .jsonl/.ndjson hoặc .json")


        KEY_RE = re.compile(
            r"^Keyframes/Keyframes_(?P<group>[KL]\d+[^/]*)/"
            r"(?P<video_id>[KL]\d+_V[^/]+)/(?P<frame>\d+)\.(?:jpg|jpeg|png)$",
            re.IGNORECASE,
        )


        def _standardize_manifest_row(row: dict[str, Any]) -> dict[str, Any] | None:
            if str(row.get("kind", "keyframe")).lower() not in {"keyframe", "image", ""}:
                return None

            r2_key = str(row.get("r2_key") or row.get("key") or "").strip()
            match = KEY_RE.match(r2_key) if r2_key else None
            video_id = str(row.get("video_id") or (match.group("video_id") if match else "")).strip()
            if not video_id:
                return None

            n = _safe_int(row.get("keyframe_n"))
            if n is None:
                n = _safe_int(row.get("n"))
            if n is None:
                frame_name = str(row.get("frame_name") or "").strip()
                n = _safe_int(Path(frame_name).stem)
            if n is None and match:
                n = int(match.group("frame"))
            if n is None:
                return None

            identity = canonical_identity(video_id, n, r2_key or None)
            pts_time = _safe_float(row.get("pts_time"))
            if pts_time is None:
                pts_time = _safe_float(row.get("timestamp_s"))
            fps = _safe_float(row.get("fps"))
            raw_frame_idx = _safe_int(row.get("frame_idx"))
            raw_manifest_etag = (
                row.get("etag")
                or row.get("source_etag")
                or row.get("r2_etag")
                or ""
            )
            manifest_etag = (
                None
                if _noneish(raw_manifest_etag)
                else str(raw_manifest_etag).strip().strip('"') or None
            )
            manifest_size_bytes = _safe_int(
                row.get("size_bytes")
                or row.get("content_length")
                or row.get("source_size_bytes")
            )

            # media_manifest.csv cũ đặt frame_idx = stem (001 -> 1), không phải frame video thật.
            # Chỉ công nhận timing khi có đủ frame_idx + pts_time + fps.
            timing_complete = raw_frame_idx is not None and pts_time is not None and fps is not None and fps > 0
            identity.update(
                {
                    "frame_idx": raw_frame_idx if timing_complete else None,
                    "pts_time": pts_time if timing_complete else None,
                    "fps": fps if timing_complete else None,
                    "timing_status": "complete" if timing_complete else "missing",
                    "manifest_etag": manifest_etag,
                    "manifest_size_bytes": manifest_size_bytes,
                }
            )
            return identity


        def load_keyframe_refs() -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
            refs: list[dict[str, Any]] = []
            source = CFG.source_manifest_key
            if source and r2_object_exists(source):
                print("Loading manifest:", source)
                raw, manifest_etag = r2_get_bytes(source)
                refs = [
                    ref
                    for row in _rows_from_manifest_bytes(raw, source)
                    if (ref := _standardize_manifest_row(row)) is not None
                ]
                source_fingerprint = {
                    "mode": "r2_manifest",
                    "key": source,
                    "etag": manifest_etag,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                }
            else:
                source = f"R2 listing:{CFG.keyframe_prefix}"
                print("Manifest không tồn tại; fallback list object prefix:", CFG.keyframe_prefix)
                listed_keys = sorted(r2_list_keys(CFG.keyframe_prefix))
                for key in tqdm(listed_keys, desc="R2 keys"):
                    match = KEY_RE.match(key)
                    if not match:
                        continue
                    refs.append(
                        {
                            **canonical_identity(
                                match.group("video_id"),
                                int(match.group("frame")),
                                key,
                            ),
                            "frame_idx": None,
                            "pts_time": None,
                            "fps": None,
                            "timing_status": "missing",
                            "manifest_etag": None,
                            "manifest_size_bytes": None,
                        }
                    )
                listing_bytes = "\n".join(listed_keys).encode("utf-8")
                source_fingerprint = {
                    "mode": "r2_key_listing",
                    "prefix": CFG.keyframe_prefix,
                    "etag": None,
                    "sha256": hashlib.sha256(listing_bytes).hexdigest(),
                    "object_count": len(listed_keys),
                }

            by_id: dict[str, dict[str, Any]] = {}
            for ref in refs:
                kid = ref["submit_keyframe_id"]
                if kid in by_id and by_id[kid]["r2_key"] != ref["r2_key"]:
                    raise ValueError(f"Identity collision {kid}: {by_id[kid]['r2_key']} vs {ref['r2_key']}")
                by_id[kid] = ref
            refs = sorted(by_id.values(), key=lambda r: (r["video_id"], r["keyframe_n"]))
            if not refs:
                raise RuntimeError("Không tìm thấy keyframe nào trên R2.")
            return refs, source, source_fingerprint


        ALL_REFS, SOURCE_DESCRIPTION, SOURCE_MANIFEST_FINGERPRINT = load_keyframe_refs()
        image_manifest_bytes = json.dumps(
            [
                {
                    "submit_keyframe_id": ref["submit_keyframe_id"],
                    "r2_key": ref["r2_key"],
                    "etag": ref["manifest_etag"],
                    "size_bytes": ref["manifest_size_bytes"],
                }
                for ref in ALL_REFS
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        IMAGE_MANIFEST_FINGERPRINT = {
            "source_key": CFG.source_manifest_key,
            "sha256": hashlib.sha256(image_manifest_bytes).hexdigest(),
            "rows": len(ALL_REFS),
            "etag_coverage": sum(
                ref["manifest_etag"] is not None for ref in ALL_REFS
            ),
        }
        identity_bytes = json.dumps(
            [
                {
                    "submit_keyframe_id": ref["submit_keyframe_id"],
                    "video_id": ref["video_id"],
                    "keyframe_n": ref["keyframe_n"],
                    "r2_key": ref["r2_key"],
                }
                for ref in ALL_REFS
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        KEYFRAME_IDENTITY_FINGERPRINT = {
            "sha256": hashlib.sha256(identity_bytes).hexdigest(),
            "rows": len(ALL_REFS),
        }
        print(f"Discovered {len(ALL_REFS):,} unique keyframes from {SOURCE_DESCRIPTION}")
        print("Source fingerprint:", json.dumps(SOURCE_MANIFEST_FINGERPRINT, indent=2))
        print("Image manifest fingerprint:", json.dumps(IMAGE_MANIFEST_FINGERPRINT, indent=2))
        """
    ),
    code(
        r"""
        # =========================== FILTER + VIDEO-BASED SHARD ===========================
        def video_shard(video_id: str, shard_count: int) -> int:
            # Stable across Python sessions; all frames of a video stay together.
            return zlib.crc32(video_id.encode("utf-8")) % shard_count


        def in_scope(ref: dict[str, Any]) -> bool:
            group = ref["group"]
            video_id = ref["video_id"]
            if CFG.include_groups and group not in CFG.include_groups:
                return False
            if CFG.exclude_groups and group in CFG.exclude_groups:
                return False
            if CFG.video_id_min and video_id < CFG.video_id_min:
                return False
            if CFG.video_id_max and video_id > CFG.video_id_max:
                return False
            return video_shard(video_id, CFG.shard_count) == CFG.shard_index


        SHARD_REFS = [r for r in ALL_REFS if in_scope(r)]
        if not SHARD_REFS:
            raise RuntimeError("Shard/filter hiện tại không có keyframe.")

        VIDEO_REFS: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ref in SHARD_REFS:
            VIDEO_REFS[ref["video_id"]].append(ref)
        for refs in VIDEO_REFS.values():
            refs.sort(key=lambda r: r["keyframe_n"])


        def _smoke_score(label: str) -> int:
            # Không dùng Python hash(): SHA-256 giữ selection ổn định giữa session/runtime.
            digest = hashlib.sha256(f"{CFG.smoke_seed}|{label}".encode("utf-8")).digest()
            return int.from_bytes(digest[:8], "big")


        def select_smoke_refs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            limit = min(CFG.smoke_limit, len(SHARD_REFS))
            if CFG.smoke_sampling_strategy == "random_frames":
                selected = sorted(
                    SHARD_REFS,
                    key=lambda r: _smoke_score("frame|" + r["submit_keyframe_id"]),
                )[:limit]
            else:
                # Round-robin qua group, mỗi lượt lấy một video chưa dùng. Điều này tránh
                # video dài hoặc video đứng đầu sort chiếm phần lớn smoke set.
                videos_by_group: dict[str, list[str]] = defaultdict(list)
                for video_id, refs in VIDEO_REFS.items():
                    videos_by_group[refs[0]["group"]].append(video_id)
                for group, video_ids in videos_by_group.items():
                    video_ids.sort(key=lambda v: _smoke_score(f"video|{group}|{v}"))

                group_order = sorted(
                    videos_by_group,
                    key=lambda g: _smoke_score("group|" + g),
                )
                chosen_videos: list[str] = []
                round_index = 0
                while len(chosen_videos) < limit:
                    added = False
                    for group in group_order:
                        group_videos = videos_by_group[group]
                        if round_index < len(group_videos):
                            chosen_videos.append(group_videos[round_index])
                            added = True
                            if len(chosen_videos) == limit:
                                break
                    if not added:
                        break
                    round_index += 1

                selected = []
                for video_id in chosen_videos:
                    refs = VIDEO_REFS[video_id]
                    # Seeded representative position covers early/middle/late across videos,
                    # thay vì luôn lấy frame đầu hoặc frame giữa.
                    ref_index = _smoke_score("position|" + video_id) % len(refs)
                    selected.append(refs[ref_index])

                # Chỉ xảy ra khi smoke_limit lớn hơn số video trong shard.
                if len(selected) < limit:
                    selected_ids = {r["submit_keyframe_id"] for r in selected}
                    remaining = [
                        r for r in SHARD_REFS
                        if r["submit_keyframe_id"] not in selected_ids
                    ]
                    remaining.sort(
                        key=lambda r: _smoke_score("fill|" + r["submit_keyframe_id"])
                    )
                    selected.extend(remaining[: limit - len(selected)])

            selected.sort(key=lambda r: (r["video_id"], r["keyframe_n"]))
            per_group: dict[str, int] = defaultdict(int)
            position_buckets = {"early": 0, "middle": 0, "late": 0}
            ref_index_by_id = {
                ref["submit_keyframe_id"]: index
                for refs in VIDEO_REFS.values()
                for index, ref in enumerate(refs)
            }
            for ref in selected:
                per_group[ref["group"]] += 1
                refs = VIDEO_REFS[ref["video_id"]]
                denominator = max(1, len(refs) - 1)
                relative_position = ref_index_by_id[ref["submit_keyframe_id"]] / denominator
                bucket = "early" if relative_position < 1 / 3 else "middle" if relative_position < 2 / 3 else "late"
                position_buckets[bucket] += 1

            report = {
                "strategy": CFG.smoke_sampling_strategy,
                "seed": CFG.smoke_seed,
                "requested": CFG.smoke_limit,
                "selected": len(selected),
                "unique_videos": len({r["video_id"] for r in selected}),
                "unique_groups": len({r["group"] for r in selected}),
                "per_group": dict(sorted(per_group.items())),
                "keyframe_position_buckets": position_buckets,
            }
            return selected, report


        if CFG.run_mode == "smoke":
            SMOKE_REFS, SMOKE_SELECTION_REPORT = select_smoke_refs()
            TARGET_IDS = {r["submit_keyframe_id"] for r in SMOKE_REFS}
            print("Smoke selection:")
            print(json.dumps(SMOKE_SELECTION_REPORT, indent=2, ensure_ascii=False))
        else:
            SMOKE_REFS = []
            SMOKE_SELECTION_REPORT = {
                "strategy": "full",
                "selected": len(SHARD_REFS),
                "unique_videos": len(VIDEO_REFS),
                "unique_groups": len({r["group"] for r in SHARD_REFS}),
            }
            TARGET_IDS = {r["submit_keyframe_id"] for r in SHARD_REFS}

        if CFG.visual_resume_source_check == "manifest_required":
            missing_source_meta = [
                ref["submit_keyframe_id"]
                for ref in SHARD_REFS
                if (
                    ref["submit_keyframe_id"] in TARGET_IDS
                    and (
                        ref.get("manifest_etag") is None
                        or ref.get("manifest_size_bytes") is None
                    )
                )
            ]
            if missing_source_meta:
                raise RuntimeError(
                    "Immutable manifest thiếu etag/size cho "
                    f"{len(missing_source_meta)} targets; "
                    f"sample={missing_source_meta[:5]}"
                )


        def fetch_elastic_timing(
            video_ids: list[str],
        ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
            # Query only videos needed by this shard/smoke scope, but fingerprint the
            # complete index generation so any timing-map mutation changes the contract.
            import requests

            endpoint = (
                get_secret("ELASTIC_ENDPOINT")
                or get_secret("ELASTIC_URL")
                or ""
            ).rstrip("/")
            api_key = get_secret("ELASTIC_API_KEY", required=True)
            if not endpoint:
                raise RuntimeError("Thiếu ELASTIC_ENDPOINT trong Colab Secrets.")
            headers = {
                "Authorization": f"ApiKey {api_key}",
                "Content-Type": "application/json",
            }

            if CFG.timing_dataset_version:
                fingerprint = {
                    "index_or_alias": CFG.elastic_keyframe_map_index,
                    "explicit_version": CFG.timing_dataset_version,
                }
            else:
                metadata_response = requests.get(
                    f"{endpoint}/{CFG.elastic_keyframe_map_index}",
                    headers=headers,
                    timeout=60,
                )
                metadata_response.raise_for_status()
                metadata = metadata_response.json()
                stats_response = requests.get(
                    f"{endpoint}/{CFG.elastic_keyframe_map_index}/"
                    "_stats/docs,seq_no?level=shards",
                    headers=headers,
                    timeout=60,
                )
                if stats_response.status_code == 410:
                    raise RuntimeError(
                        "Elastic Serverless không hỗ trợ _stats cho timing "
                        "fingerprint. Đặt CFG.timing_dataset_version thành "
                        "release tag immutable và dùng cùng tag cho mọi shard."
                    )
                stats_response.raise_for_status()
                stats = stats_response.json().get("indices", {})

                generation_rows = []
                for concrete_index, index_meta in sorted(metadata.items()):
                    index_settings = index_meta.get("settings", {}).get("index", {})
                    primaries = stats.get(concrete_index, {}).get("primaries", {})
                    primary_shards = []
                    for shard_id, shard_copies in sorted(
                        stats.get(concrete_index, {}).get("shards", {}).items()
                    ):
                        for shard_copy in shard_copies:
                            if shard_copy.get("routing", {}).get("primary"):
                                primary_shards.append(
                                    {
                                        "shard": shard_id,
                                        "docs_count": shard_copy.get("docs", {}).get("count"),
                                        "max_seq_no": shard_copy.get("seq_no", {}).get("max_seq_no"),
                                        "global_checkpoint": shard_copy.get("seq_no", {}).get(
                                            "global_checkpoint"
                                        ),
                                    }
                                )
                    generation_rows.append(
                        {
                            "index": concrete_index,
                            "uuid": index_settings.get("uuid"),
                            "creation_date": index_settings.get("creation_date"),
                            "docs_count": primaries.get("docs", {}).get("count"),
                            "max_seq_no": primaries.get("seq_no", {}).get("max_seq_no"),
                            "primary_shards": primary_shards,
                        }
                    )
                if not generation_rows:
                    raise RuntimeError("Không fingerprint được Elastic timing index.")
                generation_sha256 = hashlib.sha256(
                    json.dumps(
                        generation_rows,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                fingerprint = {
                    "index_or_alias": CFG.elastic_keyframe_map_index,
                    "generation": generation_rows,
                    "generation_sha256": generation_sha256,
                }

            url = f"{endpoint}/{CFG.elastic_keyframe_map_index}/_search?scroll=3m"
            body = {
                "size": 5000,
                "sort": ["_doc"],
                "query": {"terms": {"video_id": video_ids}},
                "_source": [
                    "submit_keyframe_id", "image_id", "category", "group",
                    "video_id", "keyframe_n", "n", "frame_idx", "pts_time", "fps",
                ],
            }
            response = requests.post(url, headers=headers, json=body, timeout=120)
            response.raise_for_status()
            page = response.json()
            scroll_id = page.get("_scroll_id")
            timing: dict[str, dict[str, Any]] = {}
            try:
                while True:
                    hits = page.get("hits", {}).get("hits", [])
                    if not hits:
                        break
                    for hit in hits:
                        src = hit.get("_source", {})
                        kid = src.get("submit_keyframe_id") or src.get("image_id")
                        if not kid:
                            video_id = str(src.get("video_id") or "")
                            n = _safe_int(src.get("keyframe_n"))
                            if n is None:
                                n = _safe_int(src.get("n"))
                            if video_id and n is not None:
                                kid = canonical_identity(video_id, n)["submit_keyframe_id"]
                        frame_idx = _safe_int(src.get("frame_idx"))
                        pts_time = _safe_float(src.get("pts_time"))
                        fps = _safe_float(src.get("fps"))
                        if kid and frame_idx is not None and pts_time is not None and fps and fps > 0:
                            timing[str(kid)] = {
                                "frame_idx": frame_idx,
                                "pts_time": pts_time,
                                "fps": fps,
                                "timing_status": "complete",
                            }
                    response = requests.post(
                        f"{endpoint}/_search/scroll",
                        headers=headers,
                        json={"scroll": "3m", "scroll_id": scroll_id},
                        timeout=120,
                    )
                    response.raise_for_status()
                    page = response.json()
                    scroll_id = page.get("_scroll_id", scroll_id)
            finally:
                if scroll_id:
                    try:
                        requests.delete(
                            f"{endpoint}/_search/scroll",
                            headers=headers,
                            json={"scroll_id": [scroll_id]},
                            timeout=30,
                        )
                    except Exception:
                        pass
            return timing, fingerprint


        TIMING_SCOPE_VIDEO_IDS = sorted({
            ref["video_id"]
            for ref in SHARD_REFS
            if CFG.run_mode == "full" or ref["submit_keyframe_id"] in TARGET_IDS
        })
        if CFG.enrich_timing_from_elastic:
            print(
                f"Reading Elastic timing for {len(TIMING_SCOPE_VIDEO_IDS):,} scoped videos:",
                CFG.elastic_keyframe_map_index,
            )
            timing_by_id, TIMING_DATASET_FINGERPRINT = fetch_elastic_timing(
                TIMING_SCOPE_VIDEO_IDS
            )
            timing_scope_set = set(TIMING_SCOPE_VIDEO_IDS)
            for ref in SHARD_REFS:
                if ref["video_id"] not in timing_scope_set:
                    continue
                overlay = timing_by_id.get(ref["submit_keyframe_id"])
                if overlay:
                    ref.update(overlay)
            SOURCE_DESCRIPTION += f" + elastic:{CFG.elastic_keyframe_map_index}"
        else:
            TIMING_DATASET_FINGERPRINT = {
                "mode": "source_manifest",
                "source_manifest_sha256": SOURCE_MANIFEST_FINGERPRINT["sha256"],
            }

        timing_scope_set = set(TIMING_SCOPE_VIDEO_IDS)
        timing_scope_refs = [
            ref for ref in SHARD_REFS
            if ref["video_id"] in timing_scope_set
        ]
        print(
            f"Timing scope: {len(TIMING_SCOPE_VIDEO_IDS):,} videos, "
            f"{sum(r['timing_status'] == 'complete' for r in timing_scope_refs):,}/"
            f"{len(timing_scope_refs):,} keyframes complete"
        )
        print(
            f"Shard {CFG.shard_index}/{CFG.shard_count}: "
            f"{len(SHARD_REFS):,} refs, {len(VIDEO_REFS):,} videos, "
            f"{len(TARGET_IDS):,} targets ({CFG.run_mode})"
        )
        """
    ),
    code(
        r"""
        # =========================== R2 IMAGE CACHE ===========================
        class R2ImageCache:
            def __init__(self, max_items: int = 24):
                self.max_items = max_items
                self.cache: OrderedDict[
                    str,
                    tuple[Image.Image, str | None, int],
                ] = OrderedDict()

            def get(
                self,
                ref: dict[str, Any],
            ) -> tuple[Image.Image, str | None, int]:
                key = ref["r2_key"]
                if key in self.cache:
                    image, etag, size_bytes = self.cache.pop(key)
                    self.cache[key] = (image, etag, size_bytes)
                    return image.copy(), etag, size_bytes

                raw, etag = r2_get_bytes(key)
                size_bytes = len(raw)
                expected_etag = ref.get("manifest_etag")
                expected_size = ref.get("manifest_size_bytes")
                if expected_etag and etag != expected_etag:
                    raise RuntimeError(
                        f"Image ETag differs from immutable manifest: {key}"
                    )
                if expected_size is not None and size_bytes != expected_size:
                    raise RuntimeError(
                        f"Image size differs from immutable manifest: {key}"
                    )
                with Image.open(io.BytesIO(raw)) as opened:
                    image = opened.convert("RGB")
                    image.load()
                self.cache[key] = (image, etag, size_bytes)
                while len(self.cache) > self.max_items:
                    _, (old_image, _, _) = self.cache.popitem(last=False)
                    old_image.close()
                return image.copy(), etag, size_bytes

            def clear(self) -> None:
                for image, _, _ in self.cache.values():
                    image.close()
                self.cache.clear()


        IMAGE_CACHE = R2ImageCache(CFG.image_cache_size)


        def dhash(image: Image.Image, size: int = 8) -> int:
            gray = image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
            px = np.asarray(gray, dtype=np.int16)
            bits = px[:, 1:] > px[:, :-1]
            value = 0
            for bit in bits.ravel():
                value = (value << 1) | int(bit)
            return value


        def hamming(a: int, b: int) -> int:
            return (a ^ b).bit_count()
        """
    ),
    code(
        r"""
        # =========================== RESUMABLE R2 JSONL PART WRITER ===========================
        VISUAL_CONTRACT = {
            "model_id": CFG.model_id,
            "model_revision": CFG.model_revision,
            "transformers_version": CFG.transformers_version,
            "visual_schema_version": CFG.schema_version,
            "visual_prompt_version": CFG.prompt_version,
            "visual_prompt_sha256": PROMPT_SHA256,
            "visual_normalizer_version": CFG.visual_normalizer_version,
            "image_manifest_fingerprint": IMAGE_MANIFEST_FINGERPRINT,
            "context_radius": CFG.context_radius,
            "max_new_tokens": CFG.visual_max_new_tokens,
            "visual_resume_source_check": CFG.visual_resume_source_check,
            "dedup": {
                "enabled": CFG.enable_dedup,
                "hamming_threshold": CFG.dedup_hamming_threshold,
                "lookback": CFG.dedup_lookback,
            },
        }
        if CFG.temporal_segmenter == "transnetv2_gpu":
            SEGMENTER_CONTRACT = {
                "name": "transnetv2_gpu",
                "package": "transnetv2-pytorch",
                "package_version": CFG.transnetv2_package_version,
                "wheel_sha256": CFG.transnetv2_wheel_sha256,
                "weights_sha256": CFG.transnetv2_weights_sha256,
                "device": "cuda",
                "threshold": CFG.transnetv2_threshold,
                "input_size": [
                    CFG.transnetv2_input_height,
                    CFG.transnetv2_input_width,
                    3,
                ],
                "min_scene_len_frames": CFG.temporal_min_scene_len_frames,
                "max_duration_mismatch_ratio": (
                    CFG.transnetv2_max_duration_mismatch_ratio
                ),
                "allow_fixed_window_fallback": CFG.allow_scene_fallback,
            }
        else:
            SEGMENTER_CONTRACT = {
                "name": "fixed_windows",
                "window_seconds": CFG.temporal_window_seconds,
                "stride_seconds": CFG.temporal_stride_seconds,
                "allow_fixed_window_fallback": False,
            }
        SEGMENTER_CONTRACT_SHA256 = hashlib.sha256(
            json.dumps(
                SEGMENTER_CONTRACT,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        TEMPORAL_CONTRACT = {
            "model_id": CFG.model_id,
            "model_revision": CFG.model_revision,
            "transformers_version": CFG.transformers_version,
            "temporal_schema_version": CFG.schema_version,
            "temporal_prompt_version": CFG.temporal_prompt_version,
            "temporal_prompt_sha256": TEMPORAL_PROMPT_SHA256,
            "keyframe_identity_fingerprint": KEYFRAME_IDENTITY_FINGERPRINT,
            "timing_dataset_fingerprint": TIMING_DATASET_FINGERPRINT,
            "segmenter": SEGMENTER_CONTRACT,
            "segmenter_contract_sha256": SEGMENTER_CONTRACT_SHA256,
            "window_seconds": CFG.temporal_window_seconds,
            "stride_seconds": CFG.temporal_stride_seconds,
            "sample_frames": CFG.temporal_sample_frames,
            "inference_batch": {
                "configured_batch_size": CFG.temporal_batch_size,
                "oom_split": CFG.temporal_oom_split,
                "padding": True,
                "tokenizer_padding_side": "left",
            },
            "frame_sampler": {
                "version": CFG.temporal_frame_sampler_version,
                "primary": "opencv_cap_prop_pos_msec",
                "fallback": "ffmpeg_fps_pipe",
                "fallback_processes_per_segment": 1,
            },
            "max_new_tokens": CFG.temporal_max_new_tokens,
        }
        VISUAL_CONTRACT_SHA256 = hashlib.sha256(
            json.dumps(VISUAL_CONTRACT, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        TEMPORAL_CONTRACT_SHA256 = hashlib.sha256(
            json.dumps(TEMPORAL_CONTRACT, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        # Generation dùng chung cho mọi shard. Video source-set là dữ liệu theo shard
        # và được hash riêng trong LINK_CONTRACT sau khi Pass B HEAD video.
        LINK_GENERATION_CONTRACT = {
            "schema_version": CFG.schema_version,
            "visual_contract_sha256": VISUAL_CONTRACT_SHA256,
            "temporal_contract_sha256": TEMPORAL_CONTRACT_SHA256,
        }
        LINK_GENERATION_CONTRACT_SHA256 = hashlib.sha256(
            json.dumps(
                LINK_GENERATION_CONTRACT,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        VISUAL_GENERATION_PREFIX = (
            f"{CFG.output_prefix}/visual/schema={CFG.schema_version}/"
            f"contract={VISUAL_CONTRACT_SHA256[:16]}"
        )
        TEMPORAL_GENERATION_PREFIX = (
            f"{CFG.output_prefix}/temporal/schema={CFG.schema_version}/"
            f"contract={TEMPORAL_CONTRACT_SHA256[:16]}"
        )
        VISUAL_RUN_PREFIX = (
            f"{VISUAL_GENERATION_PREFIX}/"
            f"shard-{CFG.shard_index:03d}-of-{CFG.shard_count:03d}"
        )
        TEMPORAL_RUN_PREFIX = (
            f"{TEMPORAL_GENERATION_PREFIX}/"
            f"shard-{CFG.shard_index:03d}-of-{CFG.shard_count:03d}"
        )
        LINK_GENERATION_PREFIX = (
            f"{CFG.output_prefix}/linked/schema={CFG.schema_version}/"
            f"visual={VISUAL_CONTRACT_SHA256[:16]}/"
            f"temporal={TEMPORAL_CONTRACT_SHA256[:16]}"
        )
        PARTS_PREFIX = f"{VISUAL_RUN_PREFIX}/parts/"
        VISUAL_LOCAL_DIR = (
            LOCAL_ROOT
            / "visual"
            / VISUAL_CONTRACT_SHA256[:16]
            / f"shard-{CFG.shard_index:03d}-of-{CFG.shard_count:03d}"
        )
        LOCAL_RUN_DIR = (
            LOCAL_ROOT
            / "session"
            / f"visual-{VISUAL_CONTRACT_SHA256[:12]}"
            / f"temporal-{TEMPORAL_CONTRACT_SHA256[:12]}"
            / f"shard-{CFG.shard_index:03d}-of-{CFG.shard_count:03d}"
        )
        VISUAL_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
        print("Visual contract:", VISUAL_CONTRACT_SHA256[:16], VISUAL_RUN_PREFIX)
        print("Temporal contract:", TEMPORAL_CONTRACT_SHA256[:16], TEMPORAL_RUN_PREFIX)


        def parse_jsonl_bytes(raw: bytes, key: str) -> list[dict[str, Any]]:
            rows = []
            for line_no, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    raise ValueError(f"JSONL hỏng tại {key}:{line_no}") from exc
            return rows


        class R2PartWriter:
            PART_RE = re.compile(r"part-(\d{6})\.jsonl$")

            def __init__(
                self,
                part_size: int,
                checkpoint_every: int,
                *,
                parts_prefix: str = PARTS_PREFIX,
                local_dir: Path = VISUAL_LOCAL_DIR,
                id_field: str = "submit_keyframe_id",
                label: str = "caption",
                allow_upsert: bool = False,
                contract_sha256: str = VISUAL_CONTRACT_SHA256,
                prompt_version: str = CFG.prompt_version,
            ):
                self.part_size = part_size
                self.checkpoint_every = checkpoint_every
                self.parts_prefix = parts_prefix
                self.local_dir = local_dir
                self.local_dir.mkdir(parents=True, exist_ok=True)
                self.id_field = id_field
                self.label = label
                self.allow_upsert = allow_upsert
                self.contract_sha256 = contract_sha256
                self.prompt_version = prompt_version
                self.records: dict[str, dict[str, Any]] = {}
                self.buffer: list[dict[str, Any]] = []
                self.part_seq = 0
                self.since_upload = 0
                self._load_resume()

            def _part_key(self, seq: int) -> str:
                return f"{self.parts_prefix}part-{seq:06d}.jsonl"

            @staticmethod
            def _encode(rows: list[dict[str, Any]]) -> bytes:
                text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
                return text.encode("utf-8")

            def _load_resume(self) -> None:
                part_keys = []
                for key in r2_list_keys(self.parts_prefix):
                    match = self.PART_RE.search(key)
                    if match:
                        part_keys.append((int(match.group(1)), key))
                part_keys.sort()

                last_rows: list[dict[str, Any]] = []
                for seq, key in tqdm(
                    part_keys,
                    desc=f"Resume {self.label} parts",
                    disable=not part_keys,
                ):
                    raw, _ = r2_get_bytes(key)
                    rows = parse_jsonl_bytes(raw, key)
                    if len(rows) > self.part_size:
                        raise ValueError(f"{key} có {len(rows)} rows > part_size={self.part_size}")
                    for row in rows:
                        kid = row[self.id_field]
                        if kid in self.records and not self.allow_upsert:
                            raise ValueError(f"Duplicate ID giữa checkpoint parts: {kid}")
                        self.records[kid] = row
                    self.part_seq = seq
                    last_rows = rows

                if part_keys and len(last_rows) < self.part_size:
                    self.buffer = last_rows
                elif part_keys:
                    self.part_seq += 1
                    self.buffer = []
                print(
                    f"Resume {self.label}: {len(self.records):,} records | "
                    f"active part={self.part_seq:06d} rows={len(self.buffer)}"
                )

            def append(self, row: dict[str, Any]) -> None:
                kid = row[self.id_field]
                if kid in self.records:
                    if not self.allow_upsert:
                        raise ValueError(f"Refuse duplicate append: {kid}")
                    self.upsert(row)
                    return
                self.records[kid] = row
                self.buffer.append(row)
                self.since_upload += 1
                if len(self.buffer) >= self.part_size:
                    self.flush(finalize_part=True)
                elif self.since_upload >= self.checkpoint_every:
                    self.flush(finalize_part=False)

            def upsert(self, row: dict[str, Any]) -> None:
                if not self.allow_upsert:
                    raise ValueError(f"{self.label} writer does not allow upsert")
                kid = row[self.id_field]
                self.records[kid] = row
                self.buffer.append(row)
                self.since_upload += 1
                if len(self.buffer) >= self.part_size:
                    self.flush(finalize_part=True)
                elif self.since_upload >= self.checkpoint_every:
                    self.flush(finalize_part=False)

            def flush(self, finalize_part: bool = False) -> None:
                if not self.buffer:
                    return
                raw = self._encode(self.buffer)
                sha = hashlib.sha256(raw).hexdigest()
                key = self._part_key(self.part_seq)
                r2_put_bytes(
                    key,
                    raw,
                    content_type="application/x-ndjson",
                    metadata={
                        "sha256": sha,
                        "schema": CFG.schema_version,
                        "prompt": self.prompt_version,
                        "contract": self.contract_sha256,
                        "rows": str(len(self.buffer)),
                    },
                )
                (self.local_dir / Path(key).name).write_bytes(raw)
                self.since_upload = 0
                if finalize_part or len(self.buffer) >= self.part_size:
                    self.part_seq += 1
                    self.buffer = []


        WRITER = R2PartWriter(
            CFG.part_size,
            CFG.checkpoint_every,
            label="caption",
            allow_upsert=True,
        )


        def head_keyframe_source(ref: dict[str, Any]) -> dict[str, Any]:
            head = r2.head_object(Bucket=R2_BUCKET, Key=ref["r2_key"])
            return {
                "etag": str(head.get("ETag", "")).strip('"') or None,
                "size_bytes": int(head.get("ContentLength", 0)),
            }


        def visual_record_is_fresh(
            ref: dict[str, Any],
            existing: dict[str, Any],
        ) -> bool:
            if CFG.visual_resume_source_check == "none":
                return True
            expected_etag = ref.get("manifest_etag")
            expected_size = ref.get("manifest_size_bytes")
            if expected_etag:
                return (
                    existing.get("source_etag") == expected_etag
                    and (
                        expected_size is None
                        or existing.get("source_size_bytes") == expected_size
                    )
                )
            if CFG.visual_resume_source_check == "manifest_required":
                raise RuntimeError(
                    f"Manifest thiếu etag cho visual resume: {ref['submit_keyframe_id']}. "
                    "Tạo immutable image manifest có etag,size_bytes hoặc dùng "
                    "visual_resume_source_check='manifest_or_head'."
                )
            current = head_keyframe_source(ref)
            return (
                existing.get("source_etag") == current["etag"]
                and existing.get("source_size_bytes") == current["size_bytes"]
            )


        DONE_IDS: set[str] = set()
        VISUAL_STALE_IDS: set[str] = set()
        refs_by_id = {ref["submit_keyframe_id"]: ref for ref in SHARD_REFS}
        resume_candidates = TARGET_IDS & set(WRITER.records)
        for kid in tqdm(
            sorted(resume_candidates),
            desc="Validate visual resume source",
            disable=not resume_candidates,
        ):
            if visual_record_is_fresh(refs_by_id[kid], WRITER.records[kid]):
                DONE_IDS.add(kid)
            else:
                VISUAL_STALE_IDS.add(kid)
        if VISUAL_STALE_IDS:
            print(
                f"Visual source changed: regenerate {len(VISUAL_STALE_IDS):,} targets "
                "using upsert checkpoints."
            )
        print(
            f"Visual resume fresh={len(DONE_IDS):,} stale={len(VISUAL_STALE_IDS):,}"
        )

        SESSION_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ERROR_ROWS: list[dict[str, Any]] = []


        def log_error(ref: dict[str, Any], exc: BaseException, attempt: int) -> None:
            ERROR_ROWS.append(
                {
                    "submit_keyframe_id": ref["submit_keyframe_id"],
                    "r2_key": ref["r2_key"],
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "traceback": traceback.format_exc(limit=4)[-4000:],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )


        def sync_errors() -> None:
            if not ERROR_ROWS:
                return
            raw = R2PartWriter._encode(ERROR_ROWS)
            key = f"{VISUAL_RUN_PREFIX}/errors/errors-{SESSION_ID}.jsonl"
            r2_put_bytes(key, raw, content_type="application/x-ndjson")
            (LOCAL_RUN_DIR / f"errors-{SESSION_ID}.jsonl").write_bytes(raw)
        """
    ),
    code(
        r"""
        # =========================== LOAD NVILA-8B ===========================
        from transformers import AutoModelForVision2Seq, AutoProcessor
        import transformers

        assert transformers.__version__ == CFG.transformers_version, (
            transformers.__version__,
            CFG.transformers_version,
        )

        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            CFG.model_id,
            revision=CFG.model_revision,
            trust_remote_code=True,
            token=HF_TOKEN,
            use_fast=False,
        )

        print("Loading ~16.2 GB BF16 weights directly to cuda:0...")
        model = AutoModelForVision2Seq.from_pretrained(
            CFG.model_id,
            revision=CFG.model_revision,
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": 0},
            attn_implementation="sdpa",
        ).eval()

        model.generation_config.do_sample = False
        model.generation_config.max_new_tokens = CFG.visual_max_new_tokens
        if processor.tokenizer.pad_token_id is None:
            if processor.tokenizer.eos_token is None:
                raise RuntimeError(
                    "Tokenizer thiếu cả pad_token lẫn eos_token; không thể batch an toàn."
                )
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

        allocated = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        print(f"Model ready | allocated={allocated:.1f} GiB reserved={reserved:.1f} GiB")
        """
    ),
    code(
        r"""
        # =========================== GENERATION + ROBUST JSON NORMALIZATION ===========================
        def extract_first_json_object(raw: str) -> dict[str, Any]:
            text = (raw or "").strip()
            decoder = json.JSONDecoder()
            for match in re.finditer(r"\{", text):
                try:
                    value, _ = decoder.raw_decode(text[match.start():])
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    continue
            raise ValueError("Không tìm thấy JSON object hợp lệ trong model output: " + repr(text[:200]))


        def _clean_string(value: Any, max_chars: int = 1000) -> str:
            if value is None:
                return ""
            value = re.sub(r"\s+", " ", str(value)).strip()
            return value[:max_chars]


        def _clean_list(value: Any, max_items: int = 16, max_chars: int = 120) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                value = [value]
            out = []
            for item in value:
                item = _clean_string(item, max_chars=max_chars).lower()
                if item and item not in out:
                    out.append(item)
            return out[:max_items]


        def _enum_token(value: Any, max_chars: int = 120) -> str:
            return (
                _clean_string(value, max_chars)
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )


        def _clean_action_phrases(value: Any, max_items: int = 16) -> list[str]:
            # Tách lỗi model gộp nhiều action vào một list item bằng comma/slash.
            raw = value if isinstance(value, list) else [value]
            flattened: list[str] = []
            for item in raw:
                for phrase in re.split(r"\s*(?:,|/|;|\band\b)\s*", _clean_string(item, 300)):
                    phrase = phrase.strip().lower()
                    if phrase and phrase not in flattened:
                        flattened.append(phrase)
            return flattened[:max_items]


        OCR_LEAK_PHRASE_RE = re.compile(
            r"\b(?:on[- ]screen text|screen text|text on (?:the )?screen|"
            r"timestamp|time stamp|news ticker|ticker text|watermark|subtitle|"
            r"caption text|logo (?:reads?|says?|showing)|"
            r"(?:sign|banner|label|screen) (?:reads?|says?|is labeled|"
            r"shows? (?:text|words?|writing|letters?|numbers?))|"
            r"(?:visible|vietnamese|english|printed|written) "
            r"(?:text|writing|words?|letters?|lettering)|"
            r"(?:text|writing|words?|letters?) in "
            r"(?:vietnamese|english|another language)|"
            r"words? (?:reads?|say|says|shown|displayed))\b",
            flags=re.IGNORECASE,
        )
        # Chỉ bắt token code có chữ + số (HTV7, VTV1) hoặc giờ. Không coi mọi
        # acronym in hoa là OCR vì SUV, TV, PPE... có thể là object/category thật.
        OCR_LIKE_TOKEN_RE = re.compile(
            r"\b(?:[A-ZÀ-Ỹ]{2,}\d+[A-ZÀ-Ỹ\d]*|\d{1,2}:\d{2})\b"
        )
        TEMPORAL_OCR_LEAK_RE = re.compile(
            r"\b(?:on[- ]screen text|screen text|text on (?:the )?screen|"
            r"written text|whiteboard writing|writing on (?:a|the) (?:board|screen)|"
            r"timestamp|time stamp|news ticker|ticker|watermark|subtitle|"
            r"logo (?:reads?|says?|showing|with (?:text|words?))|"
            r"banner (?:reads?|says?|showing|with (?:text|words?))|"
            r"sign (?:reads?|says?|showing)|"
            r"(?:screen|board) (?:shows?|displays?) "
            r"(?:text|words?|writing|letters?|numbers?)|"
            r"(?:text|words?) (?:appears?|changes?|scrolls?|reads?|is shown))\b",
            flags=re.IGNORECASE,
        )


        def _visual_ocr_spans(text_on_screen: Any) -> set[str]:
            raw = _clean_string(text_on_screen, 1000)
            spans = {
                part.strip(" '\"“”")
                for part in re.split(r"[\n|;]+", raw)
                if len(part.strip(" '\"“”")) >= 3
            }
            # Token in hoa rõ ràng như STOP/BÌ/CAO được phép scrub riêng. Không
            # dùng mọi token OCR để tránh xóa nhầm object/color như house/white.
            spans.update(
                token
                for token in re.findall(
                    r"(?u)\b[A-ZÀ-Ỹ]{3,}\b",
                    raw,
                )
            )
            return {span.lower() for span in spans}


        def _scrub_visual_ocr_text(
            value: Any,
            *,
            text_on_screen: str,
        ) -> str:
            text = _clean_string(value, 1400)
            if not text:
                return ""

            # Bỏ riêng attribution chứa transcription, giữ object/cảnh:
            # "a sign that reads 'BÌ CAO'" -> "a sign".
            clause_patterns = [
                (
                    r"\s+(?:that|which)\s+(?:reads?|says?)\s+"
                    r"(?:['\"“][^'\"”]*['\"”]|[^,.!?;]+)",
                    "",
                ),
                (
                    r"\s+with\s+(?:the\s+)?"
                    r"(?:text|words?|letters?|lettering|label)\s+"
                    r"(?:['\"“][^'\"”]*['\"”]|[^,.!?;]+)",
                    "",
                ),
                (
                    r"\s+(?:reading|labeled|captioned)\s+"
                    r"(?:['\"“][^'\"”]*['\"”]|[^,.!?;]+)",
                    "",
                ),
            ]
            for pattern, replacement in clause_patterns:
                text = re.sub(
                    pattern,
                    replacement,
                    text,
                    flags=re.IGNORECASE,
                )

            # Nếu model đã điền text_on_screen, loại exact OCR span hoặc token
            # uppercase rõ ràng bị lặp lại. Không xóa mọi từ OCR vì có thể trùng
            # object/color hợp lệ như house/white.
            for token in sorted(
                _visual_ocr_spans(text_on_screen),
                key=len,
                reverse=True,
            ):
                text = re.sub(
                    rf"(?iu)\b{re.escape(token)}\b",
                    "",
                    text,
                )
            text = OCR_LIKE_TOKEN_RE.sub("", text)

            # Câu chỉ mô tả broadcast overlay không mang visual retrieval value.
            sentences = re.split(r"(?<=[.!?])\s+", text)
            kept = []
            for sentence in sentences:
                sentence = re.sub(
                    r"\s+",
                    " ",
                    sentence,
                ).strip()
                if not sentence:
                    continue
                if OCR_LEAK_PHRASE_RE.search(sentence):
                    continue
                kept.append(sentence)
            text = " ".join(kept)
            text = re.sub(r"\s+([,.!?;:])", r"\1", text)
            text = re.sub(r"\(\s*\)|\[\s*\]|['\"“”]\s*['\"“”]", "", text)
            return re.sub(r"\s+", " ", text).strip(" ,;:-")


        def _scrub_visual_ocr_fields(
            caption: dict[str, Any],
            warnings: list[str],
        ) -> None:
            text_on_screen = _clean_string(
                caption.get("text_on_screen"),
                1000,
            )
            for field in (
                "caption_free",
                "people_description",
                "key_objects",
                "visible_actions",
                "temporal_changes",
                "distinctive_details",
            ):
                value = caption.get(field)
                if isinstance(value, list):
                    scrubbed = []
                    for item in value:
                        cleaned = _scrub_visual_ocr_text(
                            item,
                            text_on_screen=text_on_screen,
                        )
                        if cleaned and cleaned not in scrubbed:
                            scrubbed.append(cleaned)
                    if scrubbed != value:
                        warnings.append(f"ocr_isolation:scrubbed:{field}")
                    caption[field] = scrubbed
                else:
                    cleaned = _scrub_visual_ocr_text(
                        value,
                        text_on_screen=text_on_screen,
                    )
                    if cleaned != _clean_string(value, 1400):
                        warnings.append(f"ocr_isolation:scrubbed:{field}")
                    caption[field] = cleaned


        def _assert_visual_ocr_isolated(caption: dict[str, Any]) -> None:
            protected_fields = [
                "caption_free",
                "people_description",
                "key_objects",
                "visible_actions",
                "temporal_changes",
                "distinctive_details",
            ]
            protected_values = []
            for field in protected_fields:
                value = caption.get(field)
                protected_values.extend(value if isinstance(value, list) else [value])
            protected_text = " | ".join(
                _clean_string(value, 1000) for value in protected_values if value
            )
            if OCR_LEAK_PHRASE_RE.search(protected_text):
                raise ValueError(
                    "OCR leakage outside text_on_screen; rewrite visual fields "
                    "without transcribing/mentioning text content"
                )
            if OCR_LIKE_TOKEN_RE.search(protected_text):
                raise ValueError(
                    "OCR-like uppercase/code/time token leaked into visual-only fields"
                )

        def _assert_temporal_ocr_isolated(temporal: dict[str, Any]) -> None:
            values = [
                temporal.get("temporal_caption"),
                temporal.get("interaction_summary"),
                *(temporal.get("state_changes") or []),
            ]
            text = " | ".join(_clean_string(value, 1000) for value in values if value)
            if TEMPORAL_OCR_LEAK_RE.search(text):
                raise ValueError(
                    "Temporal output mentions OCR/overlay content; rewrite using "
                    "visible motion/state only"
                )


        def normalize_caption(
            data: dict[str, Any],
            *,
            temporal_context_available: bool,
        ) -> tuple[dict[str, Any], list[str]]:
            warnings: list[str] = []
            out: dict[str, Any] = {}

            missing = REQUIRED_CAPTION_KEYS - set(data)
            if missing:
                raise ValueError(
                    "Missing required caption fields: "
                    + ", ".join(sorted(missing))
                )
            out["caption_free"] = _clean_string(data.get("caption_free"), 1400)
            if not out["caption_free"]:
                raise ValueError("caption_free rỗng")

            for field, allowed in ENUMS.items():
                value = _enum_token(data.get(field), 80)
                alias = ENUM_ALIASES.get(field, {}).get(value)
                if alias is not None:
                    warnings.append(f"{field}:alias:{value}->{alias}")
                    value = alias
                if value not in allowed:
                    fallback = "indeterminate" if field == "time_of_day" else "unknown"
                    warnings.append(f"{field}:coerced_from:{value or '<empty>'}")
                    value = fallback
                out[field] = value

            strict_indoor_settings = {
                "studio", "vehicle_interior", "room", "office", "home",
                "hall", "shop", "factory", "school", "hospital", "kitchen",
                "laboratory", "workshop",
            }
            if (
                out["environment"] == "outdoor"
                and out["setting"] in strict_indoor_settings
            ):
                warnings.append(
                    "environment:derived:outdoor->indoor_from_setting:"
                    + out["setting"]
                )
                out["environment"] = "indoor"

            raw_num_people = data.get("num_people")
            if raw_num_people is None or isinstance(raw_num_people, bool):
                out["num_people"] = None
            else:
                parsed_count = _safe_int(raw_num_people)
                if parsed_count is None:
                    out["num_people"] = None
                    warnings.append("num_people:invalid")
                else:
                    out["num_people"] = max(0, min(10000, parsed_count))

            for field in LIST_FIELDS:
                out[field] = _clean_list(data.get(field))
            raw_event_types = [_enum_token(value) for value in out["event_types"]]
            canonical_event_types = []
            for value in raw_event_types:
                canonical = EVENT_TYPE_ALIASES.get(value, value)
                if canonical != value:
                    warnings.append(f"event_types:alias:{value}->{canonical}")
                if canonical not in canonical_event_types:
                    canonical_event_types.append(canonical)
            out["event_types"] = [x for x in canonical_event_types if x in EVENT_TYPES]
            dropped_events = sorted(set(canonical_event_types) - set(out["event_types"]))
            if dropped_events:
                warnings.append("event_types:dropped:" + ",".join(dropped_events))

            raw_uncertain_fields = [
                value.replace("-", "_").replace(" ", "_")
                for value in _clean_list(data.get("uncertain_fields"))
            ]
            out["uncertain_fields"] = [
                field for field in raw_uncertain_fields
                if field in CAPTION_SCHEMA_FIELDS
            ]
            dropped_uncertain = sorted(
                set(raw_uncertain_fields) - set(out["uncertain_fields"])
            )
            if dropped_uncertain:
                warnings.append(
                    "uncertain_fields:dropped:" + ",".join(dropped_uncertain)
                )
            if not temporal_context_available:
                if out["temporal_changes"]:
                    warnings.append("temporal_changes:cleared_without_context")
                out["temporal_changes"] = []
            out["people_description"] = _clean_string(data.get("people_description"), 500)
            out["text_on_screen"] = _clean_string(data.get("text_on_screen"), 1000)
            out["visible_actions"] = _clean_action_phrases(
                data.get("visible_actions"),
                max_items=16,
            )

            # Khi model không thể đếm chính xác, giữ coarse bucket hợp lệ (đặc biệt
            # "crowd"). Chỉ derive để sửa mâu thuẫn khi num_people có số cụ thể.
            reported_bucket = out["people_count_bucket"]
            count = out["num_people"]
            if count is None:
                out["people_count_bucket"] = reported_bucket
            else:
                derived_bucket = (
                    "0"
                    if count == 0
                    else "1"
                    if count == 1
                    else "2"
                    if count == 2
                    else "few"
                    if count < 10
                    else "crowd"
                )
                if reported_bucket != derived_bucket:
                    warnings.append(
                        f"people_count_bucket:derived:{reported_bucket}->{derived_bucket}"
                    )
                out["people_count_bucket"] = derived_bucket
            if out["num_people"] == 0:
                out["people_roles"] = []
                out["people_description"] = ""

            _scrub_visual_ocr_fields(out, warnings)
            if not out["caption_free"]:
                raise ValueError(
                    "caption_free rỗng sau khi tách OCR; cần rewrite visual-only"
                )
            _assert_visual_ocr_isolated(out)
            return out, warnings


        def visual_search_text(
            caption: dict[str, Any],
            *,
            include_on_screen_text: bool,
        ) -> str:
            named_fields = [
                ("description", "caption_free"),
                ("environment", "environment"),
                ("setting", "setting"),
                ("location", "location_type"),
                ("time", "time_of_day"),
                ("weather", "weather"),
                ("lighting", "lighting"),
                ("shot", "camera_shot"),
                ("angle", "camera_angle"),
                ("people count", "people_count_bucket"),
                ("events", "event_types"),
                ("people roles", "people_roles"),
                ("people", "people_description"),
                ("objects", "key_objects"),
                ("colors", "dominant_colors"),
                ("actions", "visible_actions"),
                ("changes", "temporal_changes"),
                ("details", "distinctive_details"),
            ]
            if include_on_screen_text:
                named_fields.append(("on-screen text", "text_on_screen"))
            parts = []
            for label, field in named_fields:
                value = caption.get(field)
                if isinstance(value, list):
                    value = " ".join(value)
                if value not in {None, "", "unknown", "not_applicable", "indeterminate"}:
                    parts.append(f"{label}: {value}")
            return " | ".join(parts)


        def prepare_nvila_inputs(images: list[Image.Image], prompt: str):
            content = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch = processor(text=chat, images=images, return_tensors="pt")
            return batch.to("cuda")


        def generate_raw(images: list[Image.Image], prompt: str) -> tuple[str, float]:
            inputs = prepare_nvila_inputs(images, prompt)
            input_len = inputs["input_ids"].shape[1]
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=CFG.visual_max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                torch.cuda.synchronize()
                latency = time.perf_counter() - started
                generated = output_ids[:, input_len:]
                text = processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                return text, latency
            finally:
                del inputs


        def infer_caption(
            images: list[Image.Image],
            target_position: int,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            repair_note = ""
            raw = ""
            total_latency = 0.0
            last_parsed: dict[str, Any] | None = None
            for parse_attempt in range(CFG.parse_retries + 1):
                prompt = build_caption_prompt(len(images), target_position, repair_note)
                raw, latency = generate_raw(images, prompt)
                total_latency += latency
                try:
                    parsed = extract_first_json_object(raw)
                    last_parsed = parsed
                    caption, warnings = normalize_caption(
                        parsed,
                        temporal_context_available=len(images) > 1,
                    )
                    return caption, {
                        "parse_status": "ok" if parse_attempt == 0 else "repaired",
                        "parse_attempts": parse_attempt + 1,
                        "parse_warnings": warnings,
                        "latency_seconds": round(total_latency, 3),
                        "raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    }
                except Exception as exc:
                    repair_note = str(exc) + " | previous output: " + raw[:300]

            # Strict validation/repair vẫn chạy trước. Sau khi model 8B đã thất bại
            # toàn bộ repair attempts, chỉ điền default cho các field mà giá trị
            # rỗng không tạo false positive. Field được điền phải được đánh dấu
            # uncertain để QA/retrieval biết đây không phải dự đoán của model.
            if last_parsed is not None:
                missing = REQUIRED_CAPTION_KEYS - set(last_parsed)
                safe_missing_defaults = {
                    "people_description": "",
                    "event_types": [],
                }
                if missing and missing <= set(safe_missing_defaults):
                    fallback_data = dict(last_parsed)
                    for field in sorted(missing):
                        default_value = safe_missing_defaults[field]
                        fallback_data[field] = (
                            list(default_value)
                            if isinstance(default_value, list)
                            else default_value
                        )
                    fallback_uncertain = fallback_data.get(
                        "uncertain_fields",
                        [],
                    )
                    if not isinstance(fallback_uncertain, list):
                        fallback_uncertain = [fallback_uncertain]
                    for field in sorted(missing):
                        if field not in fallback_uncertain:
                            fallback_uncertain.append(field)
                    fallback_data["uncertain_fields"] = fallback_uncertain
                    try:
                        caption, warnings = normalize_caption(
                            fallback_data,
                            temporal_context_available=len(images) > 1,
                        )
                        for field in reversed(sorted(missing)):
                            warnings.insert(
                                0,
                                f"{field}:filled_empty_after_repair_exhausted",
                            )
                        return caption, {
                            "parse_status": "fallback_repaired",
                            "parse_attempts": CFG.parse_retries + 1,
                            "parse_warnings": warnings,
                            "latency_seconds": round(total_latency, 3),
                            "raw_output_sha256": hashlib.sha256(
                                raw.encode("utf-8")
                            ).hexdigest(),
                        }
                    except Exception as fallback_exc:
                        repair_note = (
                            repair_note
                            + " | safe fallback failed: "
                            + str(fallback_exc)
                        )
            raise ValueError("NVILA output không qua được parser sau retry: " + repair_note[:700])


        def build_output_record(
            ref: dict[str, Any],
            caption: dict[str, Any],
            generation_meta: dict[str, Any],
            context_refs: list[dict[str, Any]],
            target_etag: str | None,
            target_size_bytes: int,
        ) -> dict[str, Any]:
            caption_struct = {k: v for k, v in caption.items() if k != "caption_free"}
            return {
                "schema_version": CFG.schema_version,
                "submit_keyframe_id": ref["submit_keyframe_id"],
                "image_id": ref["image_id"],
                "group": ref["group"],
                "category": ref["category"],
                "video_id": ref["video_id"],
                "keyframe_n": ref["keyframe_n"],
                "frame_name": ref["frame_name"],
                "frame_idx": ref["frame_idx"],
                "pts_time": ref["pts_time"],
                "fps": ref["fps"],
                "timing_status": ref["timing_status"],
                "r2_key": ref["r2_key"],
                "keyframe_url": ref["keyframe_url"],
                "source_etag": target_etag,
                "source_size_bytes": target_size_bytes,
                "caption_free": caption["caption_free"],
                "caption_text_visual": visual_search_text(
                    caption,
                    include_on_screen_text=False,
                ),
                "caption_text_with_text": visual_search_text(
                    caption,
                    include_on_screen_text=True,
                ),
                "caption_struct": caption_struct,
                "generation": {
                    "model_id": CFG.model_id,
                    "model_revision": CFG.model_revision,
                    "dtype": "bfloat16",
                    "prompt_version": CFG.prompt_version,
                    "prompt_sha256": PROMPT_SHA256,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    **generation_meta,
                },
                "context": {
                    "kind": "sparse_neighbor_keyframes" if len(context_refs) > 1 else "target_only",
                    "target_position": next(
                        i + 1 for i, x in enumerate(context_refs)
                        if x["submit_keyframe_id"] == ref["submit_keyframe_id"]
                    ),
                    "submit_keyframe_ids": [x["submit_keyframe_id"] for x in context_refs],
                },
                "dedup": {"is_copy": False, "source_submit_keyframe_id": None},
            }


        def maybe_build_dedup_record(
            video_refs: list[dict[str, Any]],
            idx: int,
        ) -> dict[str, Any] | None:
            if not CFG.enable_dedup:
                return None
            ref = video_refs[idx]
            target_image, target_etag, target_size_bytes = IMAGE_CACHE.get(ref)
            try:
                target_hash = dhash(target_image)
            finally:
                target_image.close()

            start = max(0, idx - CFG.dedup_lookback)
            for prev_idx in range(idx - 1, start - 1, -1):
                prev_ref = video_refs[prev_idx]
                if prev_ref["submit_keyframe_id"] not in DONE_IDS:
                    continue
                source = WRITER.records.get(prev_ref["submit_keyframe_id"])
                if source is None:
                    continue
                prev_image, _, _ = IMAGE_CACHE.get(prev_ref)
                try:
                    distance = hamming(target_hash, dhash(prev_image))
                finally:
                    prev_image.close()
                if distance > CFG.dedup_hamming_threshold:
                    continue

                caption = {
                    "caption_free": source["caption_free"],
                    **source["caption_struct"],
                }
                generation_meta = {
                    "parse_status": "dedup_copy",
                    "parse_attempts": 0,
                    "parse_warnings": [f"dhash_distance:{distance}"],
                    "latency_seconds": 0.0,
                    "raw_output_sha256": source["generation"].get("raw_output_sha256"),
                }
                record = build_output_record(
                    ref,
                    caption,
                    generation_meta,
                    [ref],
                    target_etag,
                    target_size_bytes,
                )
                record["dedup"] = {
                    "is_copy": True,
                    "source_submit_keyframe_id": prev_ref["submit_keyframe_id"],
                    "dhash_distance": distance,
                }
                return record
            return None


        def process_ref(video_refs: list[dict[str, Any]], idx: int) -> dict[str, Any]:
            ref = video_refs[idx]
            duplicate = maybe_build_dedup_record(video_refs, idx)
            if duplicate is not None:
                return duplicate
            lo = max(0, idx - CFG.context_radius)
            hi = min(len(video_refs), idx + CFG.context_radius + 1)
            context_refs = video_refs[lo:hi]
            images: list[Image.Image] = []
            target_etag = None
            target_size_bytes = 0
            try:
                for context_ref in context_refs:
                    image, etag, size_bytes = IMAGE_CACHE.get(context_ref)
                    images.append(image)
                    if context_ref["submit_keyframe_id"] == ref["submit_keyframe_id"]:
                        target_etag = etag
                        target_size_bytes = size_bytes
                target_position = idx - lo + 1
                caption, generation_meta = infer_caption(images, target_position)
                return build_output_record(
                    ref,
                    caption,
                    generation_meta,
                    context_refs,
                    target_etag,
                    target_size_bytes,
                )
            finally:
                for image in images:
                    image.close()
        """
    ),
    code(
        r"""
        # =========================== SMOKE TEST (được checkpoint, không chạy lặp) ===========================
        pending_smoke = [
            (video_id, idx, ref)
            for video_id, refs in VIDEO_REFS.items()
            for idx, ref in enumerate(refs)
            if ref["submit_keyframe_id"] in TARGET_IDS
            and ref["submit_keyframe_id"] not in DONE_IDS
        ]

        if not pending_smoke:
            print("Không còn smoke target pending; checkpoint đã có đủ target hiện tại.")
        else:
            video_id, idx, ref = pending_smoke[0]
            print("Smoke target:", ref["submit_keyframe_id"], ref["r2_key"])
            smoke_record = process_ref(VIDEO_REFS[video_id], idx)
            WRITER.append(smoke_record)
            WRITER.flush()
            DONE_IDS.add(ref["submit_keyframe_id"])
            display(
                pd.json_normalize(smoke_record).T.rename(columns={0: "value"}).head(60)
            )
            print("Smoke record đã checkpoint lên:", PARTS_PREFIX)
        """
    ),
    code(
        r"""
        # =========================== MAIN CAPTION LOOP ===========================
        targets = [
            (video_id, idx, ref)
            for video_id in sorted(VIDEO_REFS)
            for idx, ref in enumerate(VIDEO_REFS[video_id])
            if ref["submit_keyframe_id"] in TARGET_IDS
        ]
        pending = [(v, i, r) for v, i, r in targets if r["submit_keyframe_id"] not in DONE_IDS]

        print(f"Targets={len(targets):,} | already done={len(targets)-len(pending):,} | pending={len(pending):,}")
        if CFG.run_mode == "smoke":
            print("SMOKE MODE. Đổi CFG.run_mode='full' rồi chạy lại từ CONFIG để chạy hết shard.")

        started = time.time()
        session_success = 0
        consecutive_errors = 0

        try:
            progress = tqdm(pending, desc="NVILA-8B caption", unit="kf")
            for video_id, idx, ref in progress:
                success = False
                for attempt in range(1, CFG.inference_retries + 1):
                    try:
                        record = process_ref(VIDEO_REFS[video_id], idx)
                        WRITER.append(record)
                        DONE_IDS.add(ref["submit_keyframe_id"])
                        session_success += 1
                        consecutive_errors = 0
                        success = True
                        break
                    except torch.cuda.OutOfMemoryError as exc:
                        log_error(ref, exc, attempt)
                        IMAGE_CACHE.clear()
                        gc.collect()
                        torch.cuda.empty_cache()
                        print(
                            f"\nOOM {ref['submit_keyframe_id']} attempt={attempt}. "
                            "Nếu lặp lại trên A100 40 GB, chuyển A100 80 GB."
                        )
                    except Exception as exc:
                        log_error(ref, exc, attempt)
                        print(f"\nERR {ref['submit_keyframe_id']} attempt={attempt}: {type(exc).__name__}: {exc}")
                        time.sleep(min(2**attempt, 8))

                if not success:
                    consecutive_errors += 1
                    sync_errors()
                    if consecutive_errors >= CFG.stop_on_error_count:
                        raise RuntimeError(
                            f"Dừng vì {consecutive_errors} target lỗi liên tiếp; kiểm tra error log."
                        )

                elapsed = max(time.time() - started, 1e-6)
                rate = session_success / elapsed
                remaining = len(pending) - progress.n
                progress.set_postfix(
                    ok=session_success,
                    err=len(ERROR_ROWS),
                    kf_min=f"{rate * 60:.2f}",
                    eta_h=f"{remaining / max(rate, 1e-9) / 3600:.1f}",
                )
        except KeyboardInterrupt:
            print("Interrupted: đang flush checkpoint trước khi dừng.")
            raise
        finally:
            WRITER.flush()
            sync_errors()
            IMAGE_CACHE.clear()

        elapsed = time.time() - started
        print(
            f"Session done | new={session_success:,} errors={len(ERROR_ROWS):,} "
            f"elapsed={elapsed/3600:.2f}h | remote={PARTS_PREFIX}"
        )
        """
    ),
    code(
        r"""
        # =========================== QA / VALIDATION REPORT ===========================
        current_rows = [
            row for kid, row in WRITER.records.items()
            if kid in TARGET_IDS and kid in DONE_IDS
        ]
        print(f"QA rows in current target scope: {len(current_rows):,}/{len(TARGET_IDS):,}")

        if current_rows:
            qa = pd.json_normalize(current_rows)
            visual_ocr_isolation_violations = []
            for row in current_rows:
                try:
                    _assert_visual_ocr_isolated(
                        {
                            "caption_free": row.get("caption_free", ""),
                            **row.get("caption_struct", {}),
                        }
                    )
                except ValueError:
                    visual_ocr_isolation_violations.append(
                        row["submit_keyframe_id"]
                    )
            summary = {
                "records": len(qa),
                "unique_submit_keyframe_id": int(qa["submit_keyframe_id"].nunique()),
                "timing_complete_pct": round(float((qa["timing_status"] == "complete").mean() * 100), 2),
                "parse_repaired_pct": round(
                    float(
                        qa["generation.parse_status"]
                        .isin(["repaired", "fallback_repaired"])
                        .mean()
                        * 100
                    ),
                    2,
                ),
                "mean_latency_seconds": round(float(qa["generation.latency_seconds"].mean()), 3),
                "empty_caption_count": int((qa["caption_free"].str.len() == 0).sum()),
                "visual_ocr_isolation_violation_count": len(
                    visual_ocr_isolation_violations
                ),
            }
            assert summary["records"] == summary["unique_submit_keyframe_id"], "Duplicate output IDs"
            assert summary["empty_caption_count"] == 0, "Có caption rỗng"
            assert not visual_ocr_isolation_violations, (
                "OCR leak vào visual retrieval fields: "
                + ",".join(visual_ocr_isolation_violations[:5])
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False))

            enum_quality = {}
            for field in ENUMS:
                col = f"caption_struct.{field}"
                enum_quality[field] = {
                    "unknown_pct": round(float(qa[col].isin(["unknown", "indeterminate"]).mean() * 100), 2),
                    "top": qa[col].value_counts(dropna=False).head(5).to_dict(),
                }
            print("\nEnum quality:")
            print(json.dumps(enum_quality, indent=2, ensure_ascii=False))

            normalization_warning_counts: dict[str, int] = defaultdict(int)
            for row in current_rows:
                for warning in row.get("generation", {}).get("parse_warnings", []):
                    warning_family = str(warning).split(":", 1)[0]
                    normalization_warning_counts[warning_family] += 1
            print("\nNormalization warning counts:")
            print(
                json.dumps(
                    dict(sorted(normalization_warning_counts.items())),
                    indent=2,
                    ensure_ascii=False,
                )
            )

            event_types_seen = sorted({
                event
                for events in qa["caption_struct.event_types"]
                for event in (events if isinstance(events, list) else [])
            })
            coverage_probes = {
                "studio": bool((qa["caption_struct.setting"] == "studio").any()),
                "outdoor": bool((qa["caption_struct.environment"] == "outdoor").any()),
                "night": bool((qa["caption_struct.time_of_day"] == "night").any()),
                "chart_graphic": "chart_graphic" in event_types_seen,
                "crowd": (
                    "crowd" in event_types_seen
                    or bool((qa["caption_struct.people_count_bucket"] == "crowd").any())
                ),
                "disaster_fire_flood_accident": bool(
                    {"disaster", "fire", "flood", "accident"} & set(event_types_seen)
                ),
                "wide_shot": bool((qa["caption_struct.camera_shot"] == "wide").any()),
                "close_up": bool((qa["caption_struct.camera_shot"] == "close_up").any()),
                "text_on_screen_nonempty": bool(
                    qa["caption_struct.text_on_screen"].fillna("").str.strip().ne("").any()
                ),
            }
            semantic_coverage = {
                "environment": qa["caption_struct.environment"].value_counts().to_dict(),
                "setting": qa["caption_struct.setting"].value_counts().to_dict(),
                "time_of_day": qa["caption_struct.time_of_day"].value_counts().to_dict(),
                "camera_shot": qa["caption_struct.camera_shot"].value_counts().to_dict(),
                "people_count_bucket": qa[
                    "caption_struct.people_count_bucket"
                ].value_counts().to_dict(),
                "event_types_seen": event_types_seen,
                "text_on_screen_nonempty_pct": round(
                    float(
                        qa["caption_struct.text_on_screen"]
                        .fillna("")
                        .str.strip()
                        .ne("")
                        .mean()
                        * 100
                    ),
                    2,
                ),
                "coverage_probes": coverage_probes,
                "probe_gaps": [
                    name for name, covered in coverage_probes.items()
                    if not covered
                ],
            }
            print("\nSmoke semantic coverage (model-predicted, diagnostic only):")
            print(json.dumps(semantic_coverage, indent=2, ensure_ascii=False))

            display(
                qa[
                    [
                        "submit_keyframe_id",
                        "caption_free",
                        "caption_struct.environment",
                        "caption_struct.setting",
                        "caption_struct.event_types",
                        "caption_struct.visible_actions",
                        "generation.parse_status",
                        "generation.latency_seconds",
                    ]
                ].sample(min(8, len(qa)), random_state=26)
            )
        """
    ),
    markdown(
        r"""
        ## Pass B — Temporal caption từ video gốc

        Pass này không thay đổi caption Dynamic-S2 của Pass A:

        ```text
        R2 original video
          → TransNetV2 1.0.5 trên CUDA (weights đã verify SHA-256)
          → boundary cache khóa theo segmenter contract + video ETag/version
          → overlapping window 8 giây / stride 4 giây trong từng shot
          → chỉ giữ window chứa keyframe thuộc scope
          → sample frame thật theo thời gian
          → NVILA input `videos`
          → một document trong temporal segment index/window
          → keyframe chỉ giữ segment IDs, không copy temporal text
        ```

        `pts_time` thật là bắt buộc để map keyframe vào segment. Mặc định notebook đọc
        `aic26_keyframe_map_v1` từ Elastic. TransNetV2 và NVILA đều chạy GPU nhưng tuần tự;
        tensor detect được giải phóng trước temporal inference. Temporal caption có scope
        `segment_context` và không phải mô tả target-specific.
        """
    ),
    code(
        r"""
        # =========================== PASS B — VIDEO / SHOT / WINDOW UTILITIES ===========================
        TEMPORAL_PARTS_PREFIX = f"{TEMPORAL_RUN_PREFIX}/segments/parts/"
        TEMPORAL_LOCAL_DIR = (
            LOCAL_ROOT
            / "temporal"
            / TEMPORAL_CONTRACT_SHA256[:16]
            / f"shard-{CFG.shard_index:03d}-of-{CFG.shard_count:03d}"
        )
        KEYFRAME_INDEX_LOCAL_DIR = LOCAL_RUN_DIR / "keyframes-index"
        VIDEO_CACHE_DIR = LOCAL_RUN_DIR / "video-cache"
        VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        TEMPORAL_WRITER = (
            R2PartWriter(
                CFG.temporal_part_size,
                CFG.temporal_checkpoint_every,
                parts_prefix=TEMPORAL_PARTS_PREFIX,
                local_dir=TEMPORAL_LOCAL_DIR,
                id_field="temporal_segment_id",
                label="temporal",
                contract_sha256=TEMPORAL_CONTRACT_SHA256,
                prompt_version=CFG.temporal_prompt_version,
            )
            if CFG.run_temporal_pass
            else None
        )
        KEYFRAME_INDEX_PARTS_PREFIX: str | None = None
        KEYFRAME_INDEX_WRITER: R2PartWriter | None = None
        TEMPORAL_ERROR_ROWS: list[dict[str, Any]] = []


        def video_r2_key(video_id: str) -> str:
            group = group_from_video_id(video_id)
            return f"Videos/Videos_{group}/{video_id}.mp4"


        def head_video_source(video_id: str) -> dict[str, Any]:
            key = video_r2_key(video_id)
            head = r2.head_object(Bucket=R2_BUCKET, Key=key)
            etag = str(head.get("ETag", "")).strip('"') or None
            size_bytes = int(head.get("ContentLength", 0))
            version_token = hashlib.sha256(
                f"{key}|{etag}|{size_bytes}".encode("utf-8")
            ).hexdigest()[:16]
            return {
                "r2_key": key,
                "etag": etag,
                "size_bytes": size_bytes,
                "version_token": version_token,
            }


        def ensure_local_video(
            video_id: str,
            source_meta: dict[str, Any],
        ) -> Path:
            key = source_meta["r2_key"]
            expected_size = int(source_meta["size_bytes"])
            local_path = VIDEO_CACHE_DIR / (
                f"{video_id}-{source_meta['version_token']}.mp4"
            )
            if not local_path.exists() or local_path.stat().st_size != expected_size:
                temp_path = local_path.with_suffix(".mp4.part")
                if temp_path.exists():
                    temp_path.unlink()
                r2.download_file(R2_BUCKET, key, str(temp_path))
                if expected_size and temp_path.stat().st_size != expected_size:
                    raise IOError(f"Video download size mismatch: {key}")
                temp_path.replace(local_path)
            verified_source = head_video_source(video_id)
            if verified_source["version_token"] != source_meta["version_token"]:
                raise RuntimeError(
                    f"Video changed while downloading: {key}; retry the video."
                )
            return local_path


        def probe_video(video_path: Path) -> dict[str, Any]:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise IOError(f"Không mở được video: {video_path}")
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            finally:
                capture.release()
            if fps <= 0 or duration <= 0:
                raise ValueError(f"Video metadata không hợp lệ: fps={fps}, duration={duration}")
            return {
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration,
                "width": width,
                "height": height,
            }


        TRANSNET_MODEL = None


        def sha256_file(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()


        def get_transnet_model():
            global TRANSNET_MODEL
            if TRANSNET_MODEL is not None:
                return TRANSNET_MODEL

            installed_version = importlib.metadata.version("transnetv2-pytorch")
            if installed_version != CFG.transnetv2_package_version:
                raise RuntimeError(
                    "transnetv2-pytorch version mismatch: "
                    f"expected={CFG.transnetv2_package_version}, "
                    f"installed={installed_version}"
                )

            import transnetv2_pytorch

            weights_path = (
                Path(transnetv2_pytorch.__file__).resolve().parent
                / "transnetv2-pytorch-weights.pth"
            )
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"TransNetV2 weights missing: {weights_path}"
                )
            actual_weights_sha256 = sha256_file(weights_path)
            if actual_weights_sha256 != CFG.transnetv2_weights_sha256:
                raise RuntimeError(
                    "Refuse to torch.load unverified TransNetV2 weights: "
                    f"expected={CFG.transnetv2_weights_sha256}, "
                    f"actual={actual_weights_sha256}"
                )

            # Package initialization sets global RNG/determinism flags. Preserve
            # NVILA's runtime state and restore it immediately after construction.
            torch_rng_state = torch.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all()
            numpy_rng_state = np.random.get_state()
            python_rng_state = random.getstate()
            deterministic_enabled = (
                torch.are_deterministic_algorithms_enabled()
            )
            deterministic_warn_only = (
                torch.is_deterministic_algorithms_warn_only_enabled()
            )
            cudnn_deterministic = torch.backends.cudnn.deterministic
            cudnn_benchmark = torch.backends.cudnn.benchmark
            try:
                from transnetv2_pytorch import TransNetV2

                candidate = TransNetV2(device="cuda").eval()
            finally:
                torch.set_rng_state(torch_rng_state)
                torch.cuda.set_rng_state_all(cuda_rng_states)
                np.random.set_state(numpy_rng_state)
                random.setstate(python_rng_state)
                torch.use_deterministic_algorithms(
                    deterministic_enabled,
                    warn_only=deterministic_warn_only,
                )
                torch.backends.cudnn.deterministic = cudnn_deterministic
                torch.backends.cudnn.benchmark = cudnn_benchmark

            if not next(candidate.parameters()).is_cuda:
                raise RuntimeError("TransNetV2 was not loaded on CUDA.")
            TRANSNET_MODEL = candidate
            print(
                "TransNetV2 ready:",
                f"package={installed_version}",
                "device=cuda",
                f"weights_sha256={actual_weights_sha256[:16]}...",
            )
            return TRANSNET_MODEL


        def boundary_cache_key(
            video_id: str,
            source_meta: dict[str, Any],
        ) -> str:
            return (
                f"{CFG.output_prefix}/boundaries/"
                f"segmenter={CFG.temporal_segmenter}/"
                f"contract={SEGMENTER_CONTRACT_SHA256[:16]}/"
                f"{video_id}/version={source_meta['version_token']}.json"
            )


        def validate_scene_coverage(
            scenes: list[dict[str, Any]],
            duration: float,
        ) -> None:
            if not scenes:
                raise ValueError("Segmenter returned no scenes.")
            tolerance_s = max(1e-3, 1.0 / 120.0)
            if abs(float(scenes[0]["start_s"])) > tolerance_s:
                raise ValueError("First scene does not start at video time 0.")
            if abs(float(scenes[-1]["end_s"]) - duration) > tolerance_s:
                raise ValueError("Last scene does not end at video duration.")
            previous_end = 0.0
            for idx, scene in enumerate(scenes):
                start_s = float(scene["start_s"])
                end_s = float(scene["end_s"])
                if end_s <= start_s:
                    raise ValueError(f"Invalid scene interval at index {idx}.")
                if abs(start_s - previous_end) > tolerance_s:
                    raise ValueError(
                        f"Scene coverage gap/overlap at index {idx}: "
                        f"previous_end={previous_end}, start={start_s}"
                    )
                previous_end = end_s


        def load_boundary_cache(
            video_id: str,
            source_meta: dict[str, Any],
            duration: float,
        ) -> list[dict[str, Any]] | None:
            key = boundary_cache_key(video_id, source_meta)
            if not r2_object_exists(key):
                return None
            try:
                raw, _ = r2_get_bytes(key)
                cached = json.loads(raw)
                scenes = cached["scenes"]
                valid = (
                    cached.get("status") == "complete"
                    and cached.get("segmenter_contract_sha256")
                    == SEGMENTER_CONTRACT_SHA256
                    and cached.get("source_video", {}).get("etag")
                    == source_meta.get("etag")
                    and int(
                        cached.get("source_video", {}).get("size_bytes") or 0
                    )
                    == int(source_meta.get("size_bytes") or 0)
                    and cached.get("source_video", {}).get("version_token")
                    == source_meta.get("version_token")
                    and abs(
                        float(cached.get("video_duration_seconds", -1))
                        - duration
                    )
                    <= max(1e-3, duration * 1e-6)
                )
                if not valid:
                    return None
                validate_scene_coverage(scenes, duration)
                return scenes
            except Exception as exc:
                print(
                    f"Ignore invalid boundary cache for {video_id}: "
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )
                return None


        def save_boundary_cache(
            video_id: str,
            source_meta: dict[str, Any],
            video_meta: dict[str, Any],
            scenes: list[dict[str, Any]],
            diagnostics: dict[str, Any],
        ) -> None:
            payload = {
                "status": "complete",
                "video_id": video_id,
                "segmenter_contract_sha256": SEGMENTER_CONTRACT_SHA256,
                "segmenter_contract": SEGMENTER_CONTRACT,
                "source_video": source_meta,
                "video_duration_seconds": video_meta["duration_seconds"],
                "scenes": scenes,
                "diagnostics": diagnostics,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            r2_put_bytes(
                boundary_cache_key(video_id, source_meta),
                json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                content_type="application/json",
            )


        def transnet_predictions_to_scenes(
            predictions: np.ndarray,
            decoded_frame_count: int,
            fps: float,
            duration: float,
        ) -> tuple[list[dict[str, Any]], int]:
            model = get_transnet_model()
            raw_scenes = model.predictions_to_scenes(
                predictions,
                threshold=CFG.transnetv2_threshold,
            )
            raw_cut_frames = sorted({
                int(row[0])
                for row in raw_scenes[1:]
                if 0 < int(row[0]) < decoded_frame_count
            })

            # Turn TransNet's transition-excluding ranges into contiguous coverage.
            # Merge boundaries that would create scenes shorter than the configured
            # minimum, including a too-short tail.
            kept_starts = [0]
            for cut_frame in raw_cut_frames:
                if (
                    cut_frame - kept_starts[-1]
                    >= CFG.temporal_min_scene_len_frames
                ):
                    kept_starts.append(cut_frame)
            if (
                len(kept_starts) > 1
                and decoded_frame_count - kept_starts[-1]
                < CFG.temporal_min_scene_len_frames
            ):
                kept_starts.pop()

            boundaries = kept_starts + [decoded_frame_count]
            scenes = []
            for idx, (start_frame, end_frame) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            ):
                start_s = 0.0 if idx == 0 else min(
                    duration,
                    start_frame / fps,
                )
                end_s = (
                    duration
                    if idx == len(boundaries) - 2
                    else min(duration, end_frame / fps)
                )
                if end_s > start_s:
                    scenes.append(
                        {
                            "shot_id": len(scenes),
                            "start_s": float(start_s),
                            "end_s": float(end_s),
                        }
                    )
            validate_scene_coverage(scenes, duration)
            return scenes, len(raw_cut_frames)


        def run_transnet_scene_detection(
            video_path: Path,
            video_meta: dict[str, Any],
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            model = get_transnet_model()
            video_frames = None
            single_predictions = None
            all_predictions = None
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    (
                        video_frames,
                        single_predictions,
                        all_predictions,
                    ) = model.predict_video(str(video_path), quiet=True)
                decoded_frame_count = int(single_predictions.shape[0])
                if decoded_frame_count <= 0:
                    raise RuntimeError(
                        "TransNetV2/ffmpeg decoded zero video frames."
                    )
                fps = float(video_meta["fps"])
                duration = float(video_meta["duration_seconds"])
                decoded_duration = decoded_frame_count / fps
                allowed_mismatch_s = max(
                    2.0,
                    duration
                    * CFG.transnetv2_max_duration_mismatch_ratio,
                )
                duration_mismatch_s = abs(decoded_duration - duration)
                if duration_mismatch_s > allowed_mismatch_s:
                    raise RuntimeError(
                        "TransNetV2 decoded duration mismatch: "
                        f"decoded_frames={decoded_frame_count}, fps={fps:.6f}, "
                        f"decoded_duration={decoded_duration:.3f}s, "
                        f"metadata_duration={duration:.3f}s, "
                        f"allowed={allowed_mismatch_s:.3f}s"
                    )
                predictions = (
                    single_predictions.detach().float().cpu().numpy()
                )
                scenes, raw_cut_count = transnet_predictions_to_scenes(
                    predictions,
                    decoded_frame_count,
                    fps,
                    duration,
                )
                diagnostics = {
                    "device": "cuda",
                    "decoded_frame_count": decoded_frame_count,
                    "decoded_duration_seconds": decoded_duration,
                    "duration_mismatch_seconds": duration_mismatch_s,
                    "raw_cut_count": raw_cut_count,
                    "postprocessed_scene_count": len(scenes),
                    "latency_seconds": round(
                        time.perf_counter() - started,
                        3,
                    ),
                }
                return scenes, diagnostics
            finally:
                del video_frames, single_predictions, all_predictions
                gc.collect()
                torch.cuda.empty_cache()


        def detect_video_scenes(
            video_id: str,
            video_path: Path,
            video_meta: dict[str, Any],
            source_meta: dict[str, Any],
        ) -> tuple[list[dict[str, Any]], str, str | None, bool]:
            duration = float(video_meta["duration_seconds"])
            if CFG.temporal_segmenter == "fixed_windows":
                return (
                    [{"shot_id": 0, "start_s": 0.0, "end_s": duration}],
                    "fixed_windows",
                    None,
                    False,
                )

            cached_scenes = load_boundary_cache(
                video_id,
                source_meta,
                duration,
            )
            if cached_scenes is not None:
                return cached_scenes, "transnetv2_gpu", None, True

            try:
                scenes, diagnostics = run_transnet_scene_detection(
                    video_path,
                    video_meta,
                )
                save_boundary_cache(
                    video_id,
                    source_meta,
                    video_meta,
                    scenes,
                    diagnostics,
                )
                return scenes, "transnetv2_gpu", None, False
            except Exception as exc:
                fallback_reason = f"{type(exc).__name__}: {str(exc)[:500]}"
                if not CFG.allow_scene_fallback:
                    raise RuntimeError(
                        "TransNetV2 failed; refuse fixed-window fallback in "
                        "production: "
                        + fallback_reason
                    ) from exc
                print(
                    f"TransNetV2 fallback for {video_path.name}: "
                    f"{fallback_reason}"
                )
                return (
                    [{"shot_id": 0, "start_s": 0.0, "end_s": duration}],
                    "fixed_windows_fallback",
                    fallback_reason,
                    False,
                )


        def make_temporal_segment_id(
            video_id: str,
            start_s: float,
            end_s: float,
            video_version: str,
        ) -> str:
            return (
                f"{video_id}@{round(start_s * 1000):010d}-"
                f"{round(end_s * 1000):010d}@v={video_version}"
            )


        def build_temporal_segments(
            video_id: str,
            refs: list[dict[str, Any]],
            scenes: list[dict[str, Any]],
            video_meta: dict[str, Any],
            video_version: str,
        ) -> list[dict[str, Any]]:
            timed_refs = sorted(
                [r for r in refs if r.get("pts_time") is not None],
                key=lambda r: (r["pts_time"], r["keyframe_n"]),
            )
            relevant_ids = {
                r["submit_keyframe_id"]
                for r in refs
                if r["submit_keyframe_id"] in TARGET_IDS
            }
            missing_relevant = relevant_ids - {r["submit_keyframe_id"] for r in timed_refs}
            if missing_relevant and CFG.temporal_fail_on_missing_timing:
                sample = sorted(missing_relevant)[:5]
                raise ValueError(
                    f"{video_id}: {len(missing_relevant)} target thiếu pts_time; sample={sample}"
                )

            segments: list[dict[str, Any]] = []
            duration = float(video_meta["duration_seconds"])
            window_seconds = CFG.temporal_window_seconds
            stride_seconds = CFG.temporal_stride_seconds

            for scene_idx, scene in enumerate(scenes):
                shot_start = max(0.0, float(scene["start_s"]))
                shot_end = min(duration, float(scene["end_s"]))
                if shot_end <= shot_start:
                    continue
                is_last_scene = scene_idx == len(scenes) - 1

                def in_scene(ref: dict[str, Any]) -> bool:
                    pts = float(ref["pts_time"])
                    return shot_start <= pts < shot_end or (is_last_scene and math.isclose(pts, shot_end))

                shot_refs = [r for r in timed_refs if in_scene(r)]
                if not shot_refs:
                    continue

                shot_duration = shot_end - shot_start
                if shot_duration <= window_seconds:
                    window_starts = [shot_start]
                else:
                    last_start = shot_end - window_seconds
                    window_starts = []
                    cursor = shot_start
                    while cursor <= last_start + 1e-9:
                        window_starts.append(cursor)
                        cursor += stride_seconds
                    if last_start - window_starts[-1] > 1e-6:
                        # Anchor một window vào cuối shot, không tạo trailing clip quá ngắn.
                        window_starts.append(last_start)

                for window_index, start_s in enumerate(window_starts):
                    end_s = min(shot_end, start_s + window_seconds)
                    is_last_window = math.isclose(end_s, shot_end)
                    window_refs = [
                        r for r in shot_refs
                        if (
                            start_s <= float(r["pts_time"]) < end_s
                            or (is_last_window and math.isclose(float(r["pts_time"]), end_s))
                        )
                    ]
                    if not any(r["submit_keyframe_id"] in relevant_ids for r in window_refs):
                        continue
                    segment_id = make_temporal_segment_id(
                        video_id,
                        start_s,
                        end_s,
                        video_version,
                    )
                    segments.append(
                        {
                            "temporal_segment_id": segment_id,
                            "video_id": video_id,
                            "group": group_from_video_id(video_id),
                            "shot_id": int(scene["shot_id"]),
                            "window_index": int(window_index),
                            "window_seconds": CFG.temporal_window_seconds,
                            "stride_seconds": CFG.temporal_stride_seconds,
                            "start_s": round(start_s, 6),
                            "end_s": round(end_s, 6),
                            "start_frame": round(start_s * video_meta["fps"]),
                            "end_frame": round(end_s * video_meta["fps"]),
                            "keyframe_ids": [r["submit_keyframe_id"] for r in window_refs],
                            "keyframe_pts": {
                                r["submit_keyframe_id"]: r["pts_time"]
                                for r in window_refs
                            },
                        }
                    )
            return segments


        FFMPEG_SAMPLE_PATHS: set[str] = set()


        def temporal_sample_times(
            start_s: float,
            end_s: float,
            sample_count: int,
        ) -> list[float]:
            duration = end_s - start_s
            margin = min(0.10, duration * 0.05)
            sample_start = start_s + margin
            sample_end = max(sample_start, end_s - margin)
            return np.linspace(
                sample_start,
                sample_end,
                sample_count,
            ).tolist()


        def sample_video_frames_opencv(
            video_path: Path,
            times: list[float],
            video_meta: dict[str, Any],
        ) -> tuple[list[Image.Image], list[float]]:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise IOError(f"Không mở được video để sample: {video_path}")
            frames: list[Image.Image] = []
            actual_times: list[float] = []
            fps = float(video_meta["fps"])
            seek_tolerance_s = max(0.50, 3.0 / fps)
            try:
                for sample_time in times:
                    capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
                    ok, bgr = capture.read()
                    if not ok or bgr is None:
                        raise IOError(f"Decode frame lỗi tại {sample_time:.3f}s: {video_path.name}")
                    actual_time = (
                        float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                    )
                    if (
                        not math.isfinite(actual_time)
                        or abs(actual_time - sample_time) > seek_tolerance_s
                    ):
                        raise IOError(
                            "OpenCV seek sai timestamp: "
                            f"requested={sample_time:.3f}s, "
                            f"decoded={actual_time:.3f}s, "
                            f"tolerance={seek_tolerance_s:.3f}s"
                        )
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
                    actual_times.append(round(actual_time, 6))
            except Exception:
                for frame in frames:
                    frame.close()
                raise
            finally:
                capture.release()
            return frames, actual_times


        def sample_video_frames_ffmpeg(
            video_path: Path,
            times: list[float],
            video_meta: dict[str, Any],
        ) -> tuple[list[Image.Image], list[float]]:
            import ffmpeg

            sample_count = len(times)
            if sample_count < 2:
                raise ValueError("ffmpeg sampler cần ít nhất hai sample times.")
            width = int(video_meta["width"])
            height = int(video_meta["height"])
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Video dimensions không hợp lệ: {width}x{height}"
                )
            sample_start = float(times[0])
            sample_span = max(
                float(times[-1]) - sample_start,
                1.0 / float(video_meta["fps"]),
            )
            output_fps = (sample_count - 1) / sample_span

            # One ffmpeg process per segment. Input-side seek is much more robust
            # than OpenCV random CAP_PROP_POS_MSEC on long-GOP/VFR media. tpad
            # guarantees enough frames for a segment ending at the video EOF.
            stream = ffmpeg.input(str(video_path), ss=sample_start)
            stream = stream.filter(
                "tpad",
                stop_mode="clone",
                stop_duration=sample_span + max(0.1, 1.0 / output_fps),
            )
            stream = stream.filter(
                "fps",
                fps=output_fps,
                start_time=0,
                round="near",
                eof_action="pass",
            )
            try:
                raw, stderr = (
                    ffmpeg.output(
                        stream,
                        "pipe:",
                        format="rawvideo",
                        pix_fmt="rgb24",
                        s=f"{width}x{height}",
                        vframes=sample_count,
                        loglevel="error",
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as exc:
                stderr_text = (
                    (exc.stderr or b"").decode("utf-8", errors="replace")
                )
                raise IOError(
                    "ffmpeg frame sampling failed: "
                    + stderr_text[-1500:]
                ) from exc

            frame_bytes = width * height * 3
            decoded_count, remainder = divmod(len(raw), frame_bytes)
            if remainder or decoded_count != sample_count:
                stderr_text = stderr.decode("utf-8", errors="replace")
                raise IOError(
                    "ffmpeg returned unexpected raw frame count: "
                    f"expected={sample_count}, decoded={decoded_count}, "
                    f"remainder_bytes={remainder}; stderr={stderr_text[-800:]}"
                )
            frames = [
                Image.frombytes(
                    "RGB",
                    (width, height),
                    raw[index * frame_bytes : (index + 1) * frame_bytes],
                )
                for index in range(sample_count)
            ]
            return frames, [round(float(value), 6) for value in times]


        def sample_video_frames(
            video_path: Path,
            start_s: float,
            end_s: float,
            sample_count: int,
            video_meta: dict[str, Any],
        ) -> tuple[list[Image.Image], list[float], str]:
            if end_s <= start_s:
                raise ValueError(f"Invalid segment: {start_s}..{end_s}")
            times = temporal_sample_times(start_s, end_s, sample_count)
            path_key = str(video_path)
            if path_key not in FFMPEG_SAMPLE_PATHS:
                try:
                    frames, actual_times = sample_video_frames_opencv(
                        video_path,
                        times,
                        video_meta,
                    )
                    return frames, actual_times, "opencv_seek"
                except Exception as primary_exc:
                    FFMPEG_SAMPLE_PATHS.add(path_key)
                    print(
                        f"Sampler INFO: using ffmpeg for {video_path.name}; "
                        f"OpenCV random seek unavailable "
                        f"({type(primary_exc).__name__}: "
                        f"{str(primary_exc)[:400]}"
                        ")"
                    )
                    try:
                        frames, actual_times = sample_video_frames_ffmpeg(
                            video_path,
                            times,
                            video_meta,
                        )
                    except Exception as fallback_exc:
                        raise IOError(
                            "Cả OpenCV và ffmpeg sampler đều lỗi. "
                            f"OpenCV={type(primary_exc).__name__}: "
                            f"{str(primary_exc)[:500]} | "
                            f"ffmpeg={type(fallback_exc).__name__}: "
                            f"{str(fallback_exc)[:1000]}"
                        ) from fallback_exc
                    return frames, actual_times, "ffmpeg_fps_pipe"

            frames, actual_times = sample_video_frames_ffmpeg(
                video_path,
                times,
                video_meta,
            )
            return frames, actual_times, "ffmpeg_fps_pipe"
        """
    ),
    code(
        r"""
        # =========================== PASS B — NVILA VIDEO INFERENCE ===========================
        HEDGE_RE = re.compile(
            r"\b(?:possibly|probably|likely|apparently|appears?|seems?|"
            r"may|might|perhaps|presumably)\b",
            flags=re.IGNORECASE,
        )
        STATE_CHANGE_ACTIONS = {
            "standing_up", "sitting_down", "opening", "closing", "entering",
            "leaving", "vehicle_stopping", "crowd_gathering", "fire_spreading",
        }
        STATE_CHANGE_CUE_RE = re.compile(
            r"\b(?:changes? from|transitions? (?:from|to)|before[- ]to[- ]after|"
            r"stands? up|sits? down|comes? into view|disappears?|"
            r"(?:person|man|woman|child|vehicle|animal|subject) "
            r"(?:enters?|leaves?)|"
            r"(?:door|gate|lid|window|container) "
            r"(?:opens?|closes?)|"
            r"(?:starts?|begins?) to|stops? (?:moving|walking|running))\b",
            flags=re.IGNORECASE,
        )
        CAMERA_ONLY_STATE_RE = re.compile(
            r"\b(?:camera|view|framing|shot|angle|"
            r"pan|pans|panning|zoom|zooms|zooming|tracking)\b",
            flags=re.IGNORECASE,
        )
        STATIC_CAPTION_RE = re.compile(
            r"\b(?:no (?:meaningful|significant) (?:subject )?"
            r"(?:motion|movement|state change)|remains? (?:still|stationary|unchanged)|"
            r"stay(?:s)? (?:still|stationary|unchanged))\b",
            flags=re.IGNORECASE,
        )
        INFERRED_INTENT_RE = re.compile(
            r"\b(?:trying to|in order to|preparing to|waiting to|about to|"
            r"because|intends? to|plans? to)\b",
            flags=re.IGNORECASE,
        )


        def normalize_temporal(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            warnings: list[str] = []
            missing = REQUIRED_TEMPORAL_KEYS - set(data)
            if missing:
                raise ValueError(
                    "Missing required temporal fields: "
                    + ", ".join(sorted(missing))
                )
            temporal_caption = _clean_string(data.get("temporal_caption"), 1000)
            if not temporal_caption:
                raise ValueError("temporal_caption rỗng")

            raw_actions = [
                _enum_token(x)
                for x in _clean_action_phrases(
                    data.get("action_motion"),
                    max_items=12,
                )
            ]
            canonical_actions = []
            for value in raw_actions:
                canonical = TEMPORAL_ACTION_ALIASES.get(value, value)
                if canonical != value:
                    warnings.append(f"action_motion:alias:{value}->{canonical}")
                if canonical not in canonical_actions:
                    canonical_actions.append(canonical)
            actions = [x for x in canonical_actions if x in TEMPORAL_ACTIONS]
            dropped = sorted(set(canonical_actions) - set(actions))
            if dropped:
                warnings.append("action_motion:dropped:" + ",".join(dropped))
            if raw_actions and not actions:
                actions = ["other"]
                warnings.append("action_motion:all_values_dropped")
            elif not raw_actions:
                actions = []

            camera_motion = _enum_token(data.get("camera_motion"), 60)
            camera_alias = CAMERA_MOTION_ALIASES.get(camera_motion)
            if camera_alias is not None:
                warnings.append(
                    f"camera_motion:alias:{camera_motion}->{camera_alias}"
                )
                camera_motion = camera_alias
            if camera_motion not in CAMERA_MOTIONS:
                warnings.append(f"camera_motion:coerced_from:{camera_motion or '<empty>'}")
                camera_motion = "unknown"

            event_phase = _enum_token(data.get("event_phase"), 60)
            if event_phase not in EVENT_PHASES:
                warnings.append(f"event_phase:coerced_from:{event_phase or '<empty>'}")
                event_phase = "unclear"

            confidence = _clean_string(data.get("confidence"), 30).lower()
            if confidence not in TEMPORAL_CONFIDENCE:
                warnings.append(f"confidence:coerced_from:{confidence or '<empty>'}")
                confidence = "low"

            raw_uncertain_fields = [
                value.replace("-", "_").replace(" ", "_")
                for value in _clean_list(
                    data.get("uncertain_fields"),
                    max_items=12,
                )
            ]
            uncertain_fields = [
                field for field in raw_uncertain_fields
                if field in TEMPORAL_SCHEMA_FIELDS
            ]
            dropped_uncertain = sorted(
                set(raw_uncertain_fields) - set(uncertain_fields)
            )
            if dropped_uncertain:
                warnings.append(
                    "uncertain_fields:dropped:" + ",".join(dropped_uncertain)
                )

            state_changes = _clean_list(
                data.get("state_changes"),
                max_items=12,
                max_chars=180,
            )
            camera_only_state_changes = [
                value
                for value in state_changes
                if CAMERA_ONLY_STATE_RE.search(value)
            ]
            if camera_only_state_changes:
                state_changes = [
                    value
                    for value in state_changes
                    if value not in camera_only_state_changes
                ]
                warnings.append(
                    "state_changes:removed_camera_only:"
                    + str(len(camera_only_state_changes))
                )
            interaction_summary = _clean_string(
                data.get("interaction_summary"),
                600,
            )
            normalized = {
                "temporal_caption": temporal_caption,
                "action_motion": actions,
                "camera_motion": camera_motion,
                "state_changes": state_changes,
                "interaction_summary": interaction_summary,
                "event_phase": event_phase,
                "confidence": confidence,
                "uncertain_fields": uncertain_fields,
            }

            _assert_temporal_ocr_isolated(normalized)
            if INFERRED_INTENT_RE.search(
                temporal_caption + " " + interaction_summary
            ):
                raise ValueError(
                    "Temporal output infers intent/cause; describe only visible interaction"
                )

            # Một action/caption mang nghĩa chuyển trạng thái mà lại bỏ trống
            # state_changes là schema-incomplete và phải đi qua repair retry.
            change_claimed = (
                bool(set(actions) & STATE_CHANGE_ACTIONS)
                or bool(STATE_CHANGE_CUE_RE.search(temporal_caption))
            )
            if change_claimed and not state_changes:
                raise ValueError(
                    "Visible state change is claimed but state_changes is empty"
                )
            if actions == ["static"] and state_changes:
                raise ValueError("action_motion=static conflicts with state_changes")
            if actions == ["static"] and not STATIC_CAPTION_RE.search(temporal_caption):
                static_context = re.sub(
                    r"^the (?:video segment|video|segment) shows?\s+",
                    "",
                    temporal_caption,
                    flags=re.IGNORECASE,
                )
                if static_context:
                    static_context = (
                        static_context[:1].lower()
                        + static_context[1:]
                    )
                    temporal_caption = (
                        "No meaningful subject motion or state change is visible; "
                        + static_context
                    )
                else:
                    temporal_caption = (
                        "No meaningful subject motion or state change is visible; "
                        "the visible scene remains unchanged."
                    )
                normalized["temporal_caption"] = temporal_caption
                warnings.append(
                    "temporal_caption:canonicalized_static_prefix"
                )

            # Calibrate hedged language deterministically. A hedged record can never
            # retain high confidence or an empty uncertain_fields array.
            hedged_caption = bool(HEDGE_RE.search(temporal_caption))
            hedged_interaction = bool(HEDGE_RE.search(interaction_summary))
            if hedged_caption:
                for field in ("temporal_caption", "action_motion"):
                    if field not in uncertain_fields:
                        uncertain_fields.append(field)
            if hedged_interaction and "interaction_summary" not in uncertain_fields:
                uncertain_fields.append("interaction_summary")
            if (hedged_caption or hedged_interaction) and confidence == "high":
                confidence = "medium"
                warnings.append("confidence:calibrated:high->medium_due_to_hedge")
            if (hedged_caption or hedged_interaction) and not uncertain_fields:
                raise ValueError(
                    "Hedged temporal language requires uncertain_fields"
                )

            normalized["confidence"] = confidence
            normalized["uncertain_fields"] = uncertain_fields
            return normalized, warnings


        def temporal_search_text(temporal: dict[str, Any]) -> str:
            named_fields = [
                ("progression", "temporal_caption"),
                ("actions", "action_motion"),
                ("camera motion", "camera_motion"),
                ("state changes", "state_changes"),
                ("interaction", "interaction_summary"),
                ("event phase", "event_phase"),
            ]
            parts = []
            for label, field in named_fields:
                value = temporal.get(field)
                if isinstance(value, list):
                    value = " ".join(value)
                if value not in {None, "", "unknown", "static", "unclear"}:
                    parts.append(f"{label}: {value}")
            return " | ".join(parts)


        def prepare_nvila_video_batch_inputs(
            frame_batches: list[list[Image.Image]],
            prompts: list[str],
        ):
            if not frame_batches or len(frame_batches) != len(prompts):
                raise ValueError("frame_batches/prompts phải khác rỗng và cùng length")
            chats = []
            for prompt in prompts:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                chats.append(
                    processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            # Mỗi phần tử ngoài là một video segment; mỗi phần tử trong là các
            # chronological samples của segment đó. Đây mới là temporal batching thật.
            batch = processor(
                text=chats,
                videos=frame_batches,
                padding=True,
                return_tensors="pt",
            )
            return batch.to("cuda")


        def generate_temporal_raw_batch(
            frame_batches: list[list[Image.Image]],
            prompts: list[str],
        ) -> tuple[list[str], float]:
            inputs = prepare_nvila_video_batch_inputs(frame_batches, prompts)
            input_len = inputs["input_ids"].shape[1]
            batch_size = len(frame_batches)
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=CFG.temporal_max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                torch.cuda.synchronize()
                latency = time.perf_counter() - started
                generated = output_ids[:, input_len:]
                texts = processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if len(texts) != batch_size:
                    raise RuntimeError(
                        f"NVILA trả {len(texts)} outputs cho batch_size={batch_size}"
                    )
                TEMPORAL_PASS_SUMMARY["model_generate_calls"] += 1
                TEMPORAL_PASS_SUMMARY["model_generate_segments"] += batch_size
                TEMPORAL_PASS_SUMMARY["model_generate_wall_seconds"] += latency
                TEMPORAL_PASS_SUMMARY["max_batch_size_used"] = max(
                    TEMPORAL_PASS_SUMMARY["max_batch_size_used"],
                    batch_size,
                )
                key = str(batch_size)
                histogram = TEMPORAL_PASS_SUMMARY["batch_size_histogram"]
                histogram[key] = histogram.get(key, 0) + 1
                return [text.strip() for text in texts], latency
            finally:
                del inputs


        def infer_temporal_batch(
            frame_batches: list[list[Image.Image]],
            *,
            recovery_round: int = 0,
        ) -> list[
            tuple[dict[str, Any], dict[str, Any]] | Exception
        ]:
            count = len(frame_batches)
            repair_notes = [""] * count
            last_raw = [""] * count
            latency_shares = [0.0] * count
            outcomes: list[
                tuple[dict[str, Any], dict[str, Any]] | Exception | None
            ] = [None] * count
            pending = list(range(count))
            for parse_attempt in range(CFG.parse_retries + 1):
                if not pending:
                    break
                prompts = [
                    build_temporal_prompt(
                        repair_notes[index],
                        recovery_round=recovery_round,
                    )
                    for index in pending
                ]
                raws, batch_latency = generate_temporal_raw_batch(
                    [frame_batches[index] for index in pending],
                    prompts,
                )
                latency_share = batch_latency / len(pending)
                next_pending = []
                for index, raw in zip(pending, raws):
                    last_raw[index] = raw
                    latency_shares[index] += latency_share
                    try:
                        parsed = extract_first_json_object(raw)
                        temporal, warnings = normalize_temporal(parsed)
                        outcomes[index] = (
                            temporal,
                            {
                                "parse_status": (
                                    "ok" if parse_attempt == 0 else "repaired"
                                ),
                                "parse_attempts": parse_attempt + 1,
                                "parse_warnings": warnings,
                                # Amortized latency makes per-segment throughput
                                # comparable with the old batch=1 smoke report.
                                "latency_seconds": round(
                                    latency_shares[index],
                                    3,
                                ),
                                "batch_wall_seconds": round(batch_latency, 3),
                                "batch_size": len(pending),
                                "configured_batch_size": (
                                    CFG.temporal_batch_size
                                ),
                                "raw_output_sha256": hashlib.sha256(
                                    raw.encode("utf-8")
                                ).hexdigest(),
                            },
                        )
                    except Exception as exc:
                        repair_notes[index] = (
                            str(exc) + " | previous output: " + raw[:300]
                        )
                        next_pending.append(index)
                pending = next_pending

            for index in pending:
                outcomes[index] = ValueError(
                    "Temporal output không qua parser: "
                    + repair_notes[index][:700]
                )
            return [
                outcome
                if outcome is not None
                else RuntimeError("Missing temporal batch outcome")
                for outcome in outcomes
            ]


        def infer_temporal_batch_resilient(
            frame_batches: list[list[Image.Image]],
            *,
            retry_budget: int | None = None,
            recovery_round_base: int = 0,
        ) -> list[
            tuple[dict[str, Any], dict[str, Any]] | Exception
        ]:
            # Batch inference, isolate bad items, split recursively on OOM/errors.
            retry_budget = (
                CFG.inference_retries
                if retry_budget is None
                else max(1, retry_budget)
            )
            last_exception: Exception | None = None
            for attempt in range(1, retry_budget + 1):
                try:
                    outcomes = infer_temporal_batch(
                        frame_batches,
                        recovery_round=(
                            recovery_round_base + attempt - 1
                        ),
                    )
                    failed_indices = [
                        index
                        for index, outcome in enumerate(outcomes)
                        if isinstance(outcome, Exception)
                    ]
                    if (
                        failed_indices
                        and attempt < retry_budget
                    ):
                        retry_frames = [
                            frame_batches[index]
                            for index in failed_indices
                        ]
                        retry_outcomes = infer_temporal_batch_resilient(
                            retry_frames,
                            retry_budget=retry_budget - attempt,
                            recovery_round_base=(
                                recovery_round_base + attempt
                            ),
                        )
                        for index, outcome in zip(
                            failed_indices,
                            retry_outcomes,
                        ):
                            outcomes[index] = outcome
                    return outcomes
                except torch.cuda.OutOfMemoryError as exc:
                    last_exception = exc
                    gc.collect()
                    torch.cuda.empty_cache()
                    if len(frame_batches) > 1 and CFG.temporal_oom_split:
                        TEMPORAL_PASS_SUMMARY["oom_batch_splits"] += 1
                        midpoint = max(1, len(frame_batches) // 2)
                        return (
                            infer_temporal_batch_resilient(
                                frame_batches[:midpoint],
                                retry_budget=retry_budget,
                                recovery_round_base=recovery_round_base,
                            )
                            + infer_temporal_batch_resilient(
                                frame_batches[midpoint:],
                                retry_budget=retry_budget,
                                recovery_round_base=recovery_round_base,
                            )
                        )
                except Exception as exc:
                    last_exception = exc
                    # Không để một input/processor exception làm hỏng cả batch.
                    # Chia đôi cũng giúp phân biệt lỗi batching với lỗi một segment.
                    if len(frame_batches) > 1:
                        TEMPORAL_PASS_SUMMARY["non_oom_batch_splits"] += 1
                        midpoint = max(1, len(frame_batches) // 2)
                        return (
                            infer_temporal_batch_resilient(
                                frame_batches[:midpoint],
                                retry_budget=retry_budget,
                                recovery_round_base=recovery_round_base,
                            )
                            + infer_temporal_batch_resilient(
                                frame_batches[midpoint:],
                                retry_budget=retry_budget,
                                recovery_round_base=recovery_round_base,
                            )
                        )
                    time.sleep(min(2**attempt, 8))
            return [
                last_exception
                or RuntimeError("Temporal inference failed without exception")
                for _ in frame_batches
            ]


        def build_temporal_record(
            segment: dict[str, Any],
            temporal: dict[str, Any],
            generation_meta: dict[str, Any],
            sample_times: list[float],
            sample_decode_backend: str,
            video_meta: dict[str, Any],
            source_meta: dict[str, Any],
            segmenter_used: str,
            fallback_reason: str | None,
        ) -> dict[str, Any]:
            return {
                "schema_version": CFG.schema_version,
                **segment,
                "sample_times_s": sample_times,
                "sample_frame_count": len(sample_times),
                "sample_decode_backend": sample_decode_backend,
                "temporal_scope": "segment_context",
                "temporal_is_target_specific": False,
                "temporal_caption": temporal["temporal_caption"],
                "temporal_search_text": temporal_search_text(temporal),
                "temporal_struct": {
                    k: v for k, v in temporal.items()
                    if k != "temporal_caption"
                },
                "segmenter": {
                    "name": segmenter_used,
                    "configured_name": CFG.temporal_segmenter,
                    "contract_sha256": SEGMENTER_CONTRACT_SHA256,
                    "package": (
                        "transnetv2-pytorch"
                        if CFG.temporal_segmenter == "transnetv2_gpu"
                        else None
                    ),
                    "package_version": (
                        CFG.transnetv2_package_version
                        if CFG.temporal_segmenter == "transnetv2_gpu"
                        else None
                    ),
                    "weights_sha256": (
                        CFG.transnetv2_weights_sha256
                        if CFG.temporal_segmenter == "transnetv2_gpu"
                        else None
                    ),
                    "device": (
                        "cuda"
                        if CFG.temporal_segmenter == "transnetv2_gpu"
                        else None
                    ),
                    "threshold": (
                        CFG.transnetv2_threshold
                        if CFG.temporal_segmenter == "transnetv2_gpu"
                        else None
                    ),
                    "input_size": (
                        [
                            CFG.transnetv2_input_height,
                            CFG.transnetv2_input_width,
                            3,
                        ]
                        if CFG.temporal_segmenter == "transnetv2_gpu"
                        else None
                    ),
                    "min_scene_len_frames": CFG.temporal_min_scene_len_frames,
                    "window_seconds": CFG.temporal_window_seconds,
                    "stride_seconds": CFG.temporal_stride_seconds,
                    "fallback_reason": fallback_reason,
                },
                "source_video": {
                    **source_meta,
                    **video_meta,
                },
                "generation": {
                    "model_id": CFG.model_id,
                    "model_revision": CFG.model_revision,
                    "dtype": "bfloat16",
                    "prompt_version": CFG.temporal_prompt_version,
                    "prompt_sha256": TEMPORAL_PROMPT_SHA256,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    **generation_meta,
                },
            }
        """
    ),
    code(
        r"""
        # =========================== PASS B — MAIN LOOP ===========================
        # DONE_IDS chỉ chứa record đã được xác nhận fresh với source hiện tại. Dùng
        # WRITER.records ở đây có thể coi nhầm một checkpoint stale là hoàn tất.
        VISUAL_SCOPE_COMPLETE = TARGET_IDS <= DONE_IDS
        if CFG.run_temporal_pass and not VISUAL_SCOPE_COMPLETE:
            missing_visual = TARGET_IDS - DONE_IDS
            raise RuntimeError(
                f"Pass A chưa complete: thiếu {len(missing_visual)} visual captions. "
                "Resume Pass A trước khi chạy Pass B."
            )

        TEMPORAL_PASS_SUMMARY = {
            "enabled": CFG.run_temporal_pass,
            "segments_new": 0,
            "segments_expected": 0,
            "segments_completed_current": 0,
            "videos_processed": 0,
            "videos_resumed": 0,
            "boundary_cache_hits": 0,
            "boundary_cache_misses": 0,
            "scene_videos_inspected": 0,
            "scene_count_total": 0,
            "single_scene_video_count": 0,
            "single_scene_video_sample": [],
            "fallback_count": 0,
            "fallback_videos": [],
            "failed_segments": 0,
            "failed_videos": [],
            "errors": 0,
            "configured_batch_size": CFG.temporal_batch_size,
            "model_generate_calls": 0,
            "model_generate_segments": 0,
            "model_generate_wall_seconds": 0.0,
            "max_batch_size_used": 0,
            "batch_size_histogram": {},
            "oom_batch_splits": 0,
            "non_oom_batch_splits": 0,
        }
        CURRENT_VIDEO_SOURCES: dict[str, dict[str, Any]] = {}
        CURRENT_TEMPORAL_RECORDS: dict[str, dict[str, Any]] = {}
        EXPECTED_TEMPORAL_SEGMENT_IDS: set[str] = set()
        TEMPORAL_FAILED_VIDEO_IDS: set[str] = set()


        def sync_temporal_errors() -> None:
            if not TEMPORAL_ERROR_ROWS:
                return
            raw = R2PartWriter._encode(TEMPORAL_ERROR_ROWS)
            key = f"{TEMPORAL_RUN_PREFIX}/errors/errors-{SESSION_ID}.jsonl"
            r2_put_bytes(key, raw, content_type="application/x-ndjson")
            (TEMPORAL_LOCAL_DIR / f"errors-{SESSION_ID}.jsonl").write_bytes(raw)


        def temporal_circuit_reason(
            consecutive_video_errors: int,
            failed_segment_count: int,
        ) -> str | None:
            if (
                consecutive_video_errors
                >= CFG.temporal_stop_on_consecutive_video_errors
            ):
                return (
                    "consecutive_video_errors="
                    f"{consecutive_video_errors}"
                )
            if (
                failed_segment_count
                >= CFG.temporal_stop_on_total_segment_errors
            ):
                return f"failed_segments={failed_segment_count}"
            return None


        def record_matches_video_source(
            row: dict[str, Any],
            source_meta: dict[str, Any],
        ) -> bool:
            existing = row.get("source_video", {})
            return (
                existing.get("etag") == source_meta.get("etag")
                and int(existing.get("size_bytes") or 0)
                == int(source_meta.get("size_bytes") or 0)
                and existing.get("version_token") == source_meta.get("version_token")
            )


        def video_complete_marker_key(
            video_id: str,
            source_meta: dict[str, Any],
            target_ids: set[str],
        ) -> str:
            scope_sha = hashlib.sha256(
                "\n".join(sorted(target_ids)).encode("utf-8")
            ).hexdigest()[:16]
            return (
                f"{TEMPORAL_RUN_PREFIX}/video-complete/{video_id}/"
                f"version={source_meta['version_token']}/scope={scope_sha}.json"
            )


        def load_video_complete_marker(
            video_id: str,
            source_meta: dict[str, Any],
            target_ids: set[str],
        ) -> dict[str, Any] | None:
            key = video_complete_marker_key(video_id, source_meta, target_ids)
            if not r2_object_exists(key):
                return None
            raw, _ = r2_get_bytes(key)
            marker = json.loads(raw)
            segment_ids = set(marker.get("temporal_segment_ids", []))
            valid = (
                marker.get("temporal_contract_sha256") == TEMPORAL_CONTRACT_SHA256
                and marker.get("segmenter_contract_sha256")
                == SEGMENTER_CONTRACT_SHA256
                and marker.get("source_video", {}).get("etag") == source_meta.get("etag")
                and marker.get("source_video", {}).get("version_token")
                == source_meta.get("version_token")
                and set(marker.get("target_keyframe_ids", [])) == target_ids
                and segment_ids
                and segment_ids <= set(TEMPORAL_WRITER.records)
                and all(
                    record_matches_video_source(
                        TEMPORAL_WRITER.records[segment_id],
                        source_meta,
                    )
                    for segment_id in segment_ids
                )
            )
            return marker if valid else None


        def save_video_complete_marker(
            video_id: str,
            source_meta: dict[str, Any],
            target_ids: set[str],
            segment_ids: set[str],
            segmenter_used: str,
            fallback_reason: str | None,
        ) -> None:
            marker = {
                "status": "complete",
                "video_id": video_id,
                "temporal_contract_sha256": TEMPORAL_CONTRACT_SHA256,
                "segmenter_contract_sha256": SEGMENTER_CONTRACT_SHA256,
                "source_video": source_meta,
                "target_keyframe_ids": sorted(target_ids),
                "temporal_segment_ids": sorted(segment_ids),
                "segmenter_used": segmenter_used,
                "fallback_reason": fallback_reason,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            r2_put_bytes(
                video_complete_marker_key(video_id, source_meta, target_ids),
                json.dumps(marker, ensure_ascii=False, indent=2).encode("utf-8"),
                content_type="application/json",
            )


        if not CFG.run_temporal_pass:
            print("Pass B disabled: CFG.run_temporal_pass=False")
        else:
            assert TEMPORAL_WRITER is not None
            target_video_ids = sorted({
                ref["video_id"]
                for ref in SHARD_REFS
                if ref["submit_keyframe_id"] in TARGET_IDS
            })
            consecutive_video_errors = 0
            failed_segment_count = 0

        """
        + "\n            try:\n"
        + indent(
            dedent(
                r"""
            for video_id in tqdm(target_video_ids, desc="Pass B videos", unit="video"):
                video_target_ids = {
                    ref["submit_keyframe_id"]
                    for ref in VIDEO_REFS[video_id]
                    if ref["submit_keyframe_id"] in TARGET_IDS
                }

                video_path = None
                video_completed = False
                try:
                    source_meta = head_video_source(video_id)
                    CURRENT_VIDEO_SOURCES[video_id] = source_meta
                    marker = load_video_complete_marker(
                        video_id,
                        source_meta,
                        video_target_ids,
                    )
                    if marker is not None:
                        marker_segment_ids = set(marker["temporal_segment_ids"])
                        EXPECTED_TEMPORAL_SEGMENT_IDS.update(marker_segment_ids)
                        CURRENT_TEMPORAL_RECORDS.update({
                            segment_id: TEMPORAL_WRITER.records[segment_id]
                            for segment_id in marker_segment_ids
                        })
                        TEMPORAL_PASS_SUMMARY["segments_expected"] += len(
                            marker_segment_ids
                        )
                        TEMPORAL_PASS_SUMMARY["videos_resumed"] += 1
                        if marker.get("segmenter_used") == "fixed_windows_fallback":
                            TEMPORAL_PASS_SUMMARY["fallback_count"] += 1
                            TEMPORAL_PASS_SUMMARY["fallback_videos"].append(
                                {
                                    "video_id": video_id,
                                    "reason": marker.get("fallback_reason"),
                                }
                            )
                        consecutive_video_errors = 0
                        continue

                    video_path = ensure_local_video(video_id, source_meta)
                    video_meta = probe_video(video_path)
                    (
                        scenes,
                        segmenter_used,
                        fallback_reason,
                        boundary_cache_hit,
                    ) = detect_video_scenes(
                        video_id,
                        video_path,
                        video_meta,
                        source_meta,
                    )
                    cache_metric = (
                        "boundary_cache_hits"
                        if boundary_cache_hit
                        else "boundary_cache_misses"
                    )
                    TEMPORAL_PASS_SUMMARY[cache_metric] += 1
                    TEMPORAL_PASS_SUMMARY["scene_videos_inspected"] += 1
                    TEMPORAL_PASS_SUMMARY["scene_count_total"] += len(scenes)
                    if len(scenes) == 1:
                        TEMPORAL_PASS_SUMMARY[
                            "single_scene_video_count"
                        ] += 1
                        if (
                            len(
                                TEMPORAL_PASS_SUMMARY[
                                    "single_scene_video_sample"
                                ]
                            )
                            < 25
                        ):
                            TEMPORAL_PASS_SUMMARY[
                                "single_scene_video_sample"
                            ].append(video_id)
                    if segmenter_used == "fixed_windows_fallback":
                        TEMPORAL_PASS_SUMMARY["fallback_count"] += 1
                        TEMPORAL_PASS_SUMMARY["fallback_videos"].append(
                            {
                                "video_id": video_id,
                                "reason": fallback_reason,
                            }
                        )
                    segments = build_temporal_segments(
                        video_id,
                        VIDEO_REFS[video_id],
                        scenes,
                        video_meta,
                        source_meta["version_token"],
                    )
                    planned_segment_ids = {
                        segment["temporal_segment_id"] for segment in segments
                    }
                    EXPECTED_TEMPORAL_SEGMENT_IDS.update(planned_segment_ids)
                    planned_coverage = {
                        kid
                        for segment in segments
                        for kid in segment["keyframe_ids"]
                    }
                    unmapped_targets = video_target_ids - planned_coverage
                    if unmapped_targets:
                        sample = sorted(unmapped_targets)[:5]
                        raise ValueError(
                            f"{video_id}: {len(unmapped_targets)} target không map được vào video segment; "
                            f"kiểm tra pts_time/duration, sample={sample}"
                        )
                    TEMPORAL_PASS_SUMMARY["segments_expected"] += len(segments)
                    TEMPORAL_PASS_SUMMARY["videos_processed"] += 1

                    pending_segments = []
                    for segment in segments:
                        segment_id = segment["temporal_segment_id"]
                        if segment_id in TEMPORAL_WRITER.records:
                            existing = TEMPORAL_WRITER.records[segment_id]
                            if not record_matches_video_source(existing, source_meta):
                                raise RuntimeError(
                                    f"Temporal checkpoint source mismatch: {segment_id}"
                                )
                            CURRENT_TEMPORAL_RECORDS[segment_id] = existing
                            continue
                        pending_segments.append(segment)

                    for chunk_start in range(
                        0,
                        len(pending_segments),
                        CFG.temporal_batch_size,
                    ):
                        segment_chunk = pending_segments[
                            chunk_start:chunk_start + CFG.temporal_batch_size
                        ]
                        sampled_jobs: list[dict[str, Any]] = []

                        # Decode/sampling vẫn isolate từng segment. Chỉ các segment
                        # sample thành công mới được ghép thành một NVILA batch.
                        for segment in segment_chunk:
                            segment_id = segment["temporal_segment_id"]
                            sample_success = False
                            last_segment_error: dict[str, Any] | None = None
                            for attempt in range(
                                1,
                                CFG.inference_retries + 1,
                            ):
                                frames: list[Image.Image] = []
                                try:
                                    (
                                        frames,
                                        sample_times,
                                        sample_decode_backend,
                                    ) = sample_video_frames(
                                        video_path,
                                        segment["start_s"],
                                        segment["end_s"],
                                        CFG.temporal_sample_frames,
                                        video_meta,
                                    )
                                    sampled_jobs.append(
                                        {
                                            "segment": segment,
                                            "frames": frames,
                                            "sample_times": sample_times,
                                            "sample_decode_backend": (
                                                sample_decode_backend
                                            ),
                                        }
                                    )
                                    sample_success = True
                                    break
                                except Exception as exc:
                                    for frame in frames:
                                        frame.close()
                                    last_segment_error = {
                                        "temporal_segment_id": segment_id,
                                        "video_id": video_id,
                                        "segment_start_s": segment["start_s"],
                                        "segment_end_s": segment["end_s"],
                                        "stage": "sample_video_frames",
                                        "attempt": attempt,
                                        "error_type": type(exc).__name__,
                                        "error": str(exc)[:1000],
                                        "traceback": traceback.format_exc(
                                            limit=4
                                        )[-4000:],
                                        "created_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                    }
                                    TEMPORAL_ERROR_ROWS.append(
                                        last_segment_error
                                    )
                                    time.sleep(min(2**attempt, 8))
                            if not sample_success:
                                failed_segment_count += 1
                                TEMPORAL_PASS_SUMMARY["failed_segments"] = (
                                    failed_segment_count
                                )
                                print(
                                    f"Temporal ERR: {segment_id} | "
                                    "stage=sample_video_frames "
                                    f"{last_segment_error['error_type']}: "
                                    f"{last_segment_error['error']}"
                                )

                        if sampled_jobs:
                            try:
                                outcomes = infer_temporal_batch_resilient(
                                    [
                                        job["frames"]
                                        for job in sampled_jobs
                                    ]
                                )
                            except BaseException:
                                for job in sampled_jobs:
                                    for frame in job["frames"]:
                                        frame.close()
                                    job["frames"].clear()
                                raise
                            for job, outcome in zip(
                                sampled_jobs,
                                outcomes,
                            ):
                                segment = job["segment"]
                                segment_id = segment[
                                    "temporal_segment_id"
                                ]
                                try:
                                    if isinstance(outcome, Exception):
                                        raise outcome
                                    temporal, generation_meta = outcome
                                    record = build_temporal_record(
                                        segment,
                                        temporal,
                                        generation_meta,
                                        job["sample_times"],
                                        job["sample_decode_backend"],
                                        video_meta,
                                        source_meta,
                                        segmenter_used,
                                        fallback_reason,
                                    )
                                    TEMPORAL_WRITER.append(record)
                                    CURRENT_TEMPORAL_RECORDS[
                                        segment_id
                                    ] = record
                                    TEMPORAL_PASS_SUMMARY[
                                        "segments_new"
                                    ] += 1
                                except Exception as exc:
                                    failed_segment_count += 1
                                    TEMPORAL_PASS_SUMMARY[
                                        "failed_segments"
                                    ] = failed_segment_count
                                    error_row = {
                                        "temporal_segment_id": segment_id,
                                        "video_id": video_id,
                                        "segment_start_s": segment["start_s"],
                                        "segment_end_s": segment["end_s"],
                                        "stage": (
                                            "nvila_temporal_inference"
                                            if isinstance(outcome, Exception)
                                            else "build_and_checkpoint_record"
                                        ),
                                        "attempt": CFG.inference_retries,
                                        "error_type": type(exc).__name__,
                                        "error": str(exc)[:1000],
                                        "cuda_max_allocated_gib": round(
                                            torch.cuda.max_memory_allocated()
                                            / 2**30,
                                            3,
                                        ),
                                        "created_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                    }
                                    TEMPORAL_ERROR_ROWS.append(error_row)
                                    print(
                                        f"Temporal ERR: {segment_id} | "
                                        f"stage={error_row['stage']} "
                                        f"{error_row['error_type']}: "
                                        f"{error_row['error']}"
                                    )
                                finally:
                                    for frame in job["frames"]:
                                        frame.close()
                                    job["frames"].clear()

                        if (
                            failed_segment_count
                            >= CFG.temporal_stop_on_total_segment_errors
                        ):
                            sync_temporal_errors()
                            raise RuntimeError(
                                "Pass B segment-error circuit breaker threshold reached"
                            )
                        if TEMPORAL_ERROR_ROWS:
                            sync_temporal_errors()

                    TEMPORAL_WRITER.flush()
                    completed_ids = planned_segment_ids & set(CURRENT_TEMPORAL_RECORDS)
                    if completed_ids == planned_segment_ids:
                        save_video_complete_marker(
                            video_id,
                            source_meta,
                            video_target_ids,
                            planned_segment_ids,
                            segmenter_used,
                            fallback_reason,
                        )
                        video_completed = True
                    else:
                        TEMPORAL_FAILED_VIDEO_IDS.add(video_id)
                except Exception as exc:
                    TEMPORAL_FAILED_VIDEO_IDS.add(video_id)
                    TEMPORAL_ERROR_ROWS.append(
                        {
                            "temporal_segment_id": None,
                            "video_id": video_id,
                            "attempt": 0,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:1000],
                            "traceback": traceback.format_exc(limit=4)[-4000:],
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    sync_temporal_errors()
                    if CFG.temporal_fail_on_missing_timing and "pts_time" in str(exc):
                        raise
                    print(f"Pass B video error {video_id}: {type(exc).__name__}: {exc}")
                finally:
                    if (
                        video_path is not None
                        and video_path.exists()
                        and not CFG.temporal_keep_video_cache
                    ):
                        video_path.unlink()
                if video_completed:
                    consecutive_video_errors = 0
                else:
                    consecutive_video_errors += 1
                circuit_reason = temporal_circuit_reason(
                    consecutive_video_errors,
                    failed_segment_count,
                )
                if circuit_reason:
                    TEMPORAL_WRITER.flush()
                    sync_temporal_errors()
                    raise RuntimeError(
                        "Pass B circuit breaker opened: "
                        + circuit_reason
                    )

                """
            ).strip("\n"),
            "                ",
        )
        + r"""
            except KeyboardInterrupt:
                print("Interrupted Pass B: flushing temporal checkpoint.")
                raise
            finally:
                if TEMPORAL_WRITER is not None:
                    TEMPORAL_WRITER.flush()
                sync_temporal_errors()

            TEMPORAL_PASS_SUMMARY["errors"] = len(TEMPORAL_ERROR_ROWS)
            TEMPORAL_PASS_SUMMARY["segments_completed_current"] = len(
                EXPECTED_TEMPORAL_SEGMENT_IDS & set(CURRENT_TEMPORAL_RECORDS)
            )
            TEMPORAL_PASS_SUMMARY["failed_videos"] = sorted(
                TEMPORAL_FAILED_VIDEO_IDS
            )
            temporal_qa_rows = [
                row
                for segment_id, row in CURRENT_TEMPORAL_RECORDS.items()
                if segment_id in EXPECTED_TEMPORAL_SEGMENT_IDS
            ]
            confidence_counts: dict[str, int] = defaultdict(int)
            uncertain_nonempty = 0
            state_change_nonempty = 0
            hedged_high_count = 0
            temporal_ocr_leak_count = 0
            for row in temporal_qa_rows:
                temporal_struct = row.get("temporal_struct", {})
                confidence = temporal_struct.get("confidence", "unknown")
                confidence_counts[str(confidence)] += 1
                uncertain_nonempty += bool(
                    temporal_struct.get("uncertain_fields")
                )
                state_change_nonempty += bool(
                    temporal_struct.get("state_changes")
                )
                combined_text = " ".join(
                    [
                        str(row.get("temporal_caption") or ""),
                        str(
                            temporal_struct.get(
                                "interaction_summary"
                            )
                            or ""
                        ),
                        " ".join(
                            temporal_struct.get("state_changes") or []
                        ),
                    ]
                )
                hedged_high_count += bool(
                    confidence == "high"
                    and HEDGE_RE.search(combined_text)
                )
                temporal_ocr_leak_count += bool(
                    TEMPORAL_OCR_LEAK_RE.search(combined_text)
                )
            if hedged_high_count:
                raise AssertionError(
                    f"{hedged_high_count} temporal rows hedge nhưng confidence=high"
                )
            if temporal_ocr_leak_count:
                raise AssertionError(
                    f"{temporal_ocr_leak_count} temporal rows làm rò OCR"
                )
            TEMPORAL_PASS_SUMMARY["qa"] = {
                "confidence_counts": dict(
                    sorted(confidence_counts.items())
                ),
                "uncertain_fields_nonempty": uncertain_nonempty,
                "state_changes_nonempty": state_change_nonempty,
                "hedged_high_count": hedged_high_count,
                "ocr_leak_count": temporal_ocr_leak_count,
            }
            wall = TEMPORAL_PASS_SUMMARY[
                "model_generate_wall_seconds"
            ]
            TEMPORAL_PASS_SUMMARY["effective_segments_per_second"] = (
                round(
                    TEMPORAL_PASS_SUMMARY["model_generate_segments"]
                    / wall,
                    4,
                )
                if wall > 0
                else None
            )
            TEMPORAL_PASS_SUMMARY["model_generate_wall_seconds"] = round(
                wall,
                3,
            )
            print(json.dumps(TEMPORAL_PASS_SUMMARY, indent=2, ensure_ascii=False))
        """
    ),
    code(
        r"""
        # =========================== LINK KEYFRAME ↔ TEMPORAL SEGMENTS ===========================
        def link_caption_temporal_segments(
            caption_row: dict[str, Any],
            temporal_rows: list[dict[str, Any]],
        ) -> dict[str, Any]:
            linked = copy.deepcopy(caption_row)
            ordered = sorted(
                temporal_rows,
                key=lambda row: (
                    row["start_s"],
                    row["end_s"],
                    row["temporal_segment_id"],
                ),
            )
            linked["temporal_context"] = {
                "status": "complete",
                "scope": "segment_context",
                "temporal_is_target_specific": False,
                "temporal_segment_ids": [
                    row["temporal_segment_id"] for row in ordered
                ],
                "segment_ranges": [
                    {
                        "temporal_segment_id": row["temporal_segment_id"],
                        "start_s": row["start_s"],
                        "end_s": row["end_s"],
                    }
                    for row in ordered
                ],
            }
            # Không copy temporal text vào keyframe: tránh gắn action của cả clip
            # thành mô tả target-specific và gây false positive.
            return linked


        TEMPORAL_BY_KEYFRAME: dict[str, list[dict[str, Any]]] = defaultdict(list)
        LINK_CONTRACT: dict[str, Any] | None = None
        LINK_CONTRACT_SHA256: str | None = None
        LINK_RUN_PREFIX: str | None = None
        temporal_source_set_sha256: str | None = None
        TEMPORAL_SCOPE_COMPLETE = (
            not CFG.run_temporal_pass
            or (
                not TEMPORAL_FAILED_VIDEO_IDS
                and bool(EXPECTED_TEMPORAL_SEGMENT_IDS)
                and EXPECTED_TEMPORAL_SEGMENT_IDS
                <= set(CURRENT_TEMPORAL_RECORDS)
            )
        )
        if CFG.run_temporal_pass and not TEMPORAL_SCOPE_COMPLETE:
            print(
                "Skip keyframe temporal links: temporal scope chưa complete; "
                "visual và temporal checkpoints vẫn được giữ để resume."
            )
        elif CFG.run_temporal_pass:
            assert TEMPORAL_WRITER is not None
            temporal_source_set = [
                {
                    "video_id": video_id,
                    "etag": source.get("etag"),
                    "size_bytes": source.get("size_bytes"),
                    "version_token": source.get("version_token"),
                }
                for video_id, source in sorted(CURRENT_VIDEO_SOURCES.items())
            ]
            temporal_source_set_sha256 = hashlib.sha256(
                json.dumps(
                    temporal_source_set,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            LINK_CONTRACT = {
                "link_generation_contract_sha256": (
                    LINK_GENERATION_CONTRACT_SHA256
                ),
                "video_source_set_sha256": temporal_source_set_sha256,
            }
            LINK_CONTRACT_SHA256 = hashlib.sha256(
                json.dumps(
                    LINK_CONTRACT,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            LINK_RUN_PREFIX = (
                f"{LINK_GENERATION_PREFIX}/"
                f"video-sources={temporal_source_set_sha256[:16]}/"
                f"shard-{CFG.shard_index:03d}-of-{CFG.shard_count:03d}"
            )
            KEYFRAME_INDEX_PARTS_PREFIX = (
                f"{LINK_RUN_PREFIX}/parts/"
            )
            KEYFRAME_INDEX_WRITER = R2PartWriter(
                CFG.part_size,
                CFG.checkpoint_every,
                parts_prefix=KEYFRAME_INDEX_PARTS_PREFIX,
                local_dir=KEYFRAME_INDEX_LOCAL_DIR / temporal_source_set_sha256[:16],
                id_field="submit_keyframe_id",
                label="keyframe-index",
                allow_upsert=True,
                contract_sha256=LINK_CONTRACT_SHA256,
                prompt_version="link-v1",
            )
            for temporal_row in CURRENT_TEMPORAL_RECORDS.values():
                for kid in temporal_row.get("keyframe_ids", []):
                    TEMPORAL_BY_KEYFRAME[kid].append(temporal_row)

            for kid in sorted(TARGET_IDS):
                caption_row = WRITER.records.get(kid)
                temporal_rows = TEMPORAL_BY_KEYFRAME.get(kid, [])
                if caption_row is None or not temporal_rows:
                    continue
                linked_candidate = link_caption_temporal_segments(
                    caption_row,
                    temporal_rows,
                )
                existing_link = KEYFRAME_INDEX_WRITER.records.get(kid)
                if (
                    existing_link
                    and existing_link.get("source_etag")
                    == linked_candidate.get("source_etag")
                    and existing_link.get("source_size_bytes")
                    == linked_candidate.get("source_size_bytes")
                    and existing_link.get("temporal_context", {}).get(
                        "temporal_segment_ids"
                    )
                    == linked_candidate.get("temporal_context", {}).get(
                        "temporal_segment_ids"
                    )
                ):
                    continue
                KEYFRAME_INDEX_WRITER.append(linked_candidate)
            KEYFRAME_INDEX_WRITER.flush()

            linked_target_count = len(
                TARGET_IDS & set(KEYFRAME_INDEX_WRITER.records)
            )
            print(
                f"Linked keyframes: {linked_target_count:,}/{len(TARGET_IDS):,} "
                f"| current temporal segments={len(CURRENT_TEMPORAL_RECORDS):,}"
            )
            if KEYFRAME_INDEX_WRITER.records:
                linked_qa = pd.json_normalize([
                    row for kid, row in KEYFRAME_INDEX_WRITER.records.items()
                    if kid in TARGET_IDS
                ])
                display(
                    linked_qa[
                        [
                            "submit_keyframe_id",
                            "caption_free",
                            "temporal_context.scope",
                            "temporal_context.temporal_is_target_specific",
                            "temporal_context.temporal_segment_ids",
                        ]
                    ].head(8)
                )

        INDEX_RECORDS = (
            [
                row for kid, row in KEYFRAME_INDEX_WRITER.records.items()
                if kid in TARGET_IDS
            ]
            if CFG.run_temporal_pass and KEYFRAME_INDEX_WRITER is not None
            else []
            if CFG.run_temporal_pass
            else [
                row for kid, row in WRITER.records.items()
                if kid in TARGET_IDS and kid in DONE_IDS
            ]
        )
        TEMPORAL_INDEX_RECORDS = (
            [
                row
                for segment_id, row in CURRENT_TEMPORAL_RECORDS.items()
                if segment_id in EXPECTED_TEMPORAL_SEGMENT_IDS
            ]
            if CFG.run_temporal_pass
            else []
        )
        """
    ),
    code(
        r"""
        # =========================== FINALIZE RUN MANIFEST ===========================
        target_done = len(TARGET_IDS & DONE_IDS)
        linked_target_done = (
            len(TARGET_IDS & set(KEYFRAME_INDEX_WRITER.records))
            if CFG.run_temporal_pass and KEYFRAME_INDEX_WRITER is not None
            else target_done
        )
        temporal_segment_done = len(
            EXPECTED_TEMPORAL_SEGMENT_IDS & set(CURRENT_TEMPORAL_RECORDS)
        )
        visual_status = (
            "complete" if target_done == len(TARGET_IDS) else "partial"
        )
        temporal_status = (
            "disabled"
            if not CFG.run_temporal_pass
            else "complete"
            if TEMPORAL_SCOPE_COMPLETE
            else "partial"
        )
        status = (
            "complete"
            if (
                target_done == len(TARGET_IDS)
                and linked_target_done == len(TARGET_IDS)
                and TEMPORAL_SCOPE_COMPLETE
            )
            else "partial"
        )
        run_manifest = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "model_id": CFG.model_id,
            "model_revision": CFG.model_revision,
            "transformers_version": CFG.transformers_version,
            "schema_version": CFG.schema_version,
            "prompt_version": CFG.prompt_version,
            "prompt_sha256": PROMPT_SHA256,
            "visual_contract": VISUAL_CONTRACT,
            "visual_contract_sha256": VISUAL_CONTRACT_SHA256,
            "temporal_contract": TEMPORAL_CONTRACT,
            "temporal_contract_sha256": TEMPORAL_CONTRACT_SHA256,
            "link_generation_contract": LINK_GENERATION_CONTRACT,
            "link_generation_contract_sha256": (
                LINK_GENERATION_CONTRACT_SHA256
            ),
            "link_contract": LINK_CONTRACT,
            "link_contract_sha256": LINK_CONTRACT_SHA256,
            "source": SOURCE_DESCRIPTION,
            "visual_run_prefix": VISUAL_RUN_PREFIX,
            "temporal_run_prefix": TEMPORAL_RUN_PREFIX,
            "link_run_prefix": LINK_RUN_PREFIX,
            "shard_count": CFG.shard_count,
            "shard_index": CFG.shard_index,
            "run_mode": CFG.run_mode,
            "smoke_selection": SMOKE_SELECTION_REPORT,
            "target_count": len(TARGET_IDS),
            "target_done": target_done,
            "visual_status": visual_status,
            "visual_stale_regenerated": len(VISUAL_STALE_IDS),
            "linked_target_done": linked_target_done,
            "all_records_in_shard_checkpoint": len(WRITER.records),
            "session_errors": len(ERROR_ROWS),
            "temporal": TEMPORAL_PASS_SUMMARY,
            "temporal_status": temporal_status,
            "temporal_scope_complete": TEMPORAL_SCOPE_COMPLETE,
            "temporal_segments_expected": len(EXPECTED_TEMPORAL_SEGMENT_IDS),
            "temporal_segments_done": temporal_segment_done,
            "temporal_segments_checkpoint": (
                len(TEMPORAL_WRITER.records)
                if TEMPORAL_WRITER is not None
                else 0
            ),
            "keyframe_index_parts_prefix": KEYFRAME_INDEX_PARTS_PREFIX,
            "keyframe_index_records_checkpoint": (
                len(KEYFRAME_INDEX_WRITER.records)
                if KEYFRAME_INDEX_WRITER is not None
                else 0
            ),
            "config": asdict(CFG),
        }
        manifest_raw = json.dumps(run_manifest, ensure_ascii=False, indent=2).encode("utf-8")
        visual_manifest = {
            "status": visual_status,
            "updated_at": run_manifest["updated_at"],
            "contract": VISUAL_CONTRACT,
            "contract_sha256": VISUAL_CONTRACT_SHA256,
            "run_prefix": VISUAL_RUN_PREFIX,
            "run_mode": CFG.run_mode,
            "shard_count": CFG.shard_count,
            "shard_index": CFG.shard_index,
            "target_count": len(TARGET_IDS),
            "target_done": target_done,
            "stale_regenerated": len(VISUAL_STALE_IDS),
        }
        r2_put_bytes(
            f"{VISUAL_RUN_PREFIX}/run_manifest.json",
            json.dumps(
                visual_manifest,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            content_type="application/json",
            metadata={"status": visual_status, "schema": CFG.schema_version},
        )
        if CFG.run_temporal_pass:
            temporal_manifest = {
                "status": temporal_status,
                "updated_at": run_manifest["updated_at"],
                "contract": TEMPORAL_CONTRACT,
                "contract_sha256": TEMPORAL_CONTRACT_SHA256,
                "run_prefix": TEMPORAL_RUN_PREFIX,
                "run_mode": CFG.run_mode,
                "shard_count": CFG.shard_count,
                "shard_index": CFG.shard_index,
                "segments_expected": len(EXPECTED_TEMPORAL_SEGMENT_IDS),
                "segments_done": temporal_segment_done,
                "summary": TEMPORAL_PASS_SUMMARY,
            }
            r2_put_bytes(
                f"{TEMPORAL_RUN_PREFIX}/run_manifest.json",
                json.dumps(
                    temporal_manifest,
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8"),
                content_type="application/json",
                metadata={
                    "status": temporal_status,
                    "schema": CFG.schema_version,
                },
            )
        if LINK_RUN_PREFIX:
            final_manifest_key = f"{LINK_RUN_PREFIX}/run_manifest.json"
        elif CFG.run_temporal_pass:
            final_manifest_key = (
                f"{TEMPORAL_RUN_PREFIX}/run-manifests/"
                f"visual={VISUAL_CONTRACT_SHA256[:16]}/run_manifest.json"
            )
        else:
            final_manifest_key = f"{VISUAL_RUN_PREFIX}/run_manifest.json"
        r2_put_bytes(
            final_manifest_key,
            manifest_raw,
            content_type="application/json",
            metadata={"status": status, "schema": CFG.schema_version},
        )
        (LOCAL_RUN_DIR / "run_manifest.json").write_bytes(manifest_raw)
        print(json.dumps(run_manifest, indent=2, ensure_ascii=False))

        if CFG.write_combined_shard_at_end and CFG.run_mode == "full" and status == "complete":
            rows = sorted(
                INDEX_RECORDS,
                key=lambda r: (r["video_id"], r["keyframe_n"]),
            )
            combined = R2PartWriter._encode(rows)
            combined_name = "keyframes-index-shard.jsonl"
            combined_prefix = LINK_RUN_PREFIX or VISUAL_RUN_PREFIX
            combined_key = f"{combined_prefix}/{combined_name}"
            r2_put_bytes(
                combined_key,
                combined,
                content_type="application/x-ndjson",
                metadata={"sha256": hashlib.sha256(combined).hexdigest(), "rows": str(len(rows))},
            )
            print("Combined shard:", combined_key)
            if CFG.run_temporal_pass:
                temporal_rows = sorted(
                    TEMPORAL_INDEX_RECORDS,
                    key=lambda r: (
                        r["video_id"],
                        r["start_s"],
                        r["end_s"],
                    ),
                )
                temporal_combined = R2PartWriter._encode(temporal_rows)
                temporal_combined_key = (
                    f"{TEMPORAL_RUN_PREFIX}/temporal-segments-index-shard.jsonl"
                )
                r2_put_bytes(
                    temporal_combined_key,
                    temporal_combined,
                    content_type="application/x-ndjson",
                    metadata={
                        "sha256": hashlib.sha256(temporal_combined).hexdigest(),
                        "rows": str(len(temporal_rows)),
                    },
                )
                print("Temporal combined shard:", temporal_combined_key)
        """
    ),
    markdown(
        r"""
        ## Optional — index caption vào Elasticsearch

        Mặc định cell sau **không ghi gì**. Sau khi QA đạt yêu cầu:

        1. Tạo Colab Secrets `ELASTIC_ENDPOINT`, `ELASTIC_API_KEY`.
        2. Đặt `PUSH_TO_ELASTIC = True`.
        3. Giữ `PROMOTE_ELASTIC_ALIASES=False` và chạy cell một lần cho mỗi
           completed shard với cùng `ELASTIC_BUILD_REVISION`.
        4. Sau khi đủ 16 shard và QA đạt, chạy shard cuối với
           `PROMOTE_ELASTIC_ALIASES=True`.

        Cell ghi vào hai **build index theo generation**, không ghi trực tiếp live name:

        - `aic26_caption_keyframes_v2_<generation>_<revision>`
        - `aic26_caption_temporal_segments_v1_<generation>_<revision>`

        Khi promote, notebook kiểm tra completed manifest và đúng document count/link contract
        cho mọi shard rồi chuyển nguyên tử hai alias live:

        - `aic26_caption_keyframes_v2`
        - `aic26_caption_temporal_segments_v1`

        Rerun một shard sẽ xóa–ghi lại đúng shard trong build chưa-live, nên segment ID/boundary
        cũ không tồn tại sót. Build đang phục vụ live alias là immutable; muốn chạy generation
        mới phải bump `ELASTIC_BUILD_REVISION`. Notebook không tự xóa build cũ để giữ rollback.

        Keyframe channel search:

        - `caption_free^3`
        - `caption_text_visual^2`
        - `caption_text_with_text` chỉ là VLM OCR phụ cho T-KIS
        - `caption_struct.text_on_screen`
        - filter keyword trên `caption_struct`

        `caption_text_visual` cố ý không chứa `text_on_screen`, để V-KIS không bị nhiễu bởi
        hallucinated VLM text. OCR index chuyên dụng vẫn là nguồn text chính.

        Temporal channel search `temporal_caption^2`, `temporal_search_text` và
        `temporal_struct.action_motion`. Mỗi segment hit trả `keyframe_ids`; backend map hit về
        các keyframe trong interval rồi mới fuse bằng RRF. Nên dùng temporal channel weight thấp
        hơn visual (điểm khởi đầu `0.5`) vì đây là context-level evidence. Temporal text không
        được coi là target-specific caption. Cell mặc định từ chối cả smoke run và partial run.
        """
    ),
    code(
        r"""
        # =========================== OPTIONAL ELASTIC BULK INDEX ===========================
        PUSH_TO_ELASTIC = False
        PROMOTE_ELASTIC_ALIASES = False
        ALLOW_PARTIAL_INDEX = False
        ALLOW_MUTATE_LIVE_BUILD = False
        # Bump revision khi muốn rebuild cùng contract/source generation từ đầu.
        # Không reuse một build revision đã được promote làm live.
        ELASTIC_BUILD_REVISION = "build01"
        ELASTIC_KEYFRAME_ALIAS = "aic26_caption_keyframes_v2"
        ELASTIC_TEMPORAL_ALIAS = "aic26_caption_temporal_segments_v1"
        KEYFRAME_GENERATION_SHA256 = (
            LINK_GENERATION_CONTRACT_SHA256
            if CFG.run_temporal_pass
            else VISUAL_CONTRACT_SHA256
        )
        TEMPORAL_GENERATION_SHA256 = TEMPORAL_CONTRACT_SHA256
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", ELASTIC_BUILD_REVISION):
            raise ValueError(
                "ELASTIC_BUILD_REVISION chỉ được chứa a-z, 0-9, _ và -."
            )
        ELASTIC_KEYFRAME_BUILD_INDEX = (
            f"{ELASTIC_KEYFRAME_ALIAS}_"
            f"{KEYFRAME_GENERATION_SHA256[:12]}_"
            f"{ELASTIC_BUILD_REVISION}"
        )
        ELASTIC_TEMPORAL_BUILD_INDEX = (
            f"{ELASTIC_TEMPORAL_ALIAS}_"
            f"{TEMPORAL_GENERATION_SHA256[:12]}_"
            f"{ELASTIC_BUILD_REVISION}"
        )
        ELASTIC_BATCH_SIZE = 500

        if PROMOTE_ELASTIC_ALIASES and not PUSH_TO_ELASTIC:
            raise RuntimeError(
                "PROMOTE_ELASTIC_ALIASES=True yêu cầu PUSH_TO_ELASTIC=True "
                "để preflight và refresh build trước cutover."
            )

        if PUSH_TO_ELASTIC:
            import requests

            unsafe_reasons = []
            if status != "complete":
                unsafe_reasons.append(f"run status={status!r}")
            if CFG.run_mode != "full":
                unsafe_reasons.append(f"run_mode={CFG.run_mode!r}")
            if len(INDEX_RECORDS) != len(TARGET_IDS):
                unsafe_reasons.append(
                    f"keyframe rows={len(INDEX_RECORDS)}/{len(TARGET_IDS)}"
                )
            if CFG.run_temporal_pass and (
                len(TEMPORAL_INDEX_RECORDS)
                != len(EXPECTED_TEMPORAL_SEGMENT_IDS)
            ):
                unsafe_reasons.append(
                    "temporal rows="
                    f"{len(TEMPORAL_INDEX_RECORDS)}/"
                    f"{len(EXPECTED_TEMPORAL_SEGMENT_IDS)}"
                )
            if unsafe_reasons and not ALLOW_PARTIAL_INDEX:
                raise RuntimeError(
                    "Refuse Elastic indexing: "
                    + "; ".join(unsafe_reasons)
                    + ". Set ALLOW_PARTIAL_INDEX=True only for an intentional non-production load."
                )

            ELASTIC_URL = (
                get_secret("ELASTIC_ENDPOINT")
                or get_secret("ELASTIC_URL")
                or ""
            ).rstrip("/")
            if not ELASTIC_URL:
                raise RuntimeError("Thiếu ELASTIC_ENDPOINT trong Colab Secrets.")
            ELASTIC_API_KEY = get_secret("ELASTIC_API_KEY", required=True)
            headers = {
                "Authorization": f"ApiKey {ELASTIC_API_KEY}",
                "Content-Type": "application/json",
            }


            def elastic_alias_targets(alias: str) -> set[str]:
                response = requests.get(
                    f"{ELASTIC_URL}/_alias/{alias}",
                    headers=headers,
                    timeout=30,
                )
                if response.status_code == 404:
                    return set()
                response.raise_for_status()
                return set(response.json())


            keyframe_live_targets = elastic_alias_targets(
                ELASTIC_KEYFRAME_ALIAS
            )
            temporal_live_targets = (
                elastic_alias_targets(ELASTIC_TEMPORAL_ALIAS)
                if CFG.run_temporal_pass
                else set()
            )
            live_builds = []
            if ELASTIC_KEYFRAME_BUILD_INDEX in keyframe_live_targets:
                live_builds.append(ELASTIC_KEYFRAME_BUILD_INDEX)
            if ELASTIC_TEMPORAL_BUILD_INDEX in temporal_live_targets:
                live_builds.append(ELASTIC_TEMPORAL_BUILD_INDEX)
            if live_builds and not ALLOW_MUTATE_LIVE_BUILD:
                raise RuntimeError(
                    "Refuse mutating a build currently serving a live alias: "
                    + ", ".join(live_builds)
                    + ". Bump ELASTIC_BUILD_REVISION for a new immutable build."
                )

            keyframe_mapping = {
                "settings": {
                    "number_of_shards": 1,
                    "analysis": {
                        "analyzer": {
                            "caption_en": {"type": "standard", "stopwords": "_english_"}
                        }
                    },
                },
                "mappings": {
                    "dynamic": False,
                    "properties": {
                        "schema_version": {"type": "keyword"},
                        "build_generation": {"type": "keyword"},
                        "build_revision": {"type": "keyword"},
                        "pipeline_shard_index": {"type": "integer"},
                        "pipeline_shard_count": {"type": "integer"},
                        "link_contract_sha256": {"type": "keyword"},
                        "submit_keyframe_id": {"type": "keyword"},
                        "image_id": {"type": "keyword"},
                        "group": {"type": "keyword"},
                        "category": {"type": "keyword"},
                        "video_id": {"type": "keyword"},
                        "keyframe_n": {"type": "integer"},
                        "frame_idx": {"type": "integer"},
                        "pts_time": {"type": "double"},
                        "fps": {"type": "float"},
                        "timing_status": {"type": "keyword"},
                        "r2_key": {"type": "keyword", "index": False},
                        "keyframe_url": {"type": "keyword", "index": False},
                        "source_etag": {"type": "keyword"},
                        "source_size_bytes": {"type": "long"},
                        "caption_free": {"type": "text", "analyzer": "caption_en"},
                        "caption_text_visual": {"type": "text", "analyzer": "caption_en"},
                        "caption_text_with_text": {
                            "type": "text",
                            "analyzer": "caption_en",
                        },
                        "caption_struct": {
                            "properties": {
                                "environment": {"type": "keyword"},
                                "setting": {"type": "keyword"},
                                "location_type": {"type": "keyword"},
                                "time_of_day": {"type": "keyword"},
                                "weather": {"type": "keyword"},
                                "lighting": {"type": "keyword"},
                                "camera_shot": {"type": "keyword"},
                                "camera_angle": {"type": "keyword"},
                                "num_people": {"type": "integer"},
                                "people_count_bucket": {"type": "keyword"},
                                "event_types": {"type": "keyword"},
                                "people_roles": {"type": "keyword"},
                                "people_description": {"type": "text", "analyzer": "caption_en"},
                                "key_objects": {"type": "keyword"},
                                "dominant_colors": {"type": "keyword"},
                                "visible_actions": {"type": "keyword"},
                                "temporal_changes": {"type": "text", "analyzer": "caption_en"},
                                "distinctive_details": {"type": "text", "analyzer": "caption_en"},
                                "text_on_screen": {"type": "text"},
                                "uncertain_fields": {"type": "keyword"},
                            }
                        },
                        "generation": {
                            "properties": {
                                "model_id": {"type": "keyword"},
                                "model_revision": {"type": "keyword"},
                                "prompt_version": {"type": "keyword"},
                                "prompt_sha256": {"type": "keyword"},
                                "parse_status": {"type": "keyword"},
                                "latency_seconds": {"type": "float"},
                                "generated_at": {"type": "date"},
                            }
                        },
                        "temporal_context": {
                            "properties": {
                                "status": {"type": "keyword"},
                                "scope": {"type": "keyword"},
                                "temporal_is_target_specific": {"type": "boolean"},
                                "temporal_segment_ids": {"type": "keyword"},
                                "segment_ranges": {
                                    "properties": {
                                        "temporal_segment_id": {"type": "keyword"},
                                        "start_s": {"type": "double"},
                                        "end_s": {"type": "double"},
                                    }
                                },
                            }
                        },
                    },
                },
            }

            exists = requests.head(
                f"{ELASTIC_URL}/{ELASTIC_KEYFRAME_BUILD_INDEX}",
                headers=headers,
                timeout=30,
            )
            if exists.status_code == 404:
                response = requests.put(
                    f"{ELASTIC_URL}/{ELASTIC_KEYFRAME_BUILD_INDEX}",
                    headers=headers,
                    json=keyframe_mapping,
                    timeout=60,
                )
                response.raise_for_status()
                print("Created build index:", ELASTIC_KEYFRAME_BUILD_INDEX)
            else:
                exists.raise_for_status()
                response = requests.put(
                    f"{ELASTIC_URL}/{ELASTIC_KEYFRAME_BUILD_INDEX}/_mapping",
                    headers=headers,
                    json=keyframe_mapping["mappings"],
                    timeout=60,
                )
                response.raise_for_status()
                print("Updated build mapping:", ELASTIC_KEYFRAME_BUILD_INDEX)

            # Mỗi rerun thay toàn bộ document của đúng shard trong build chưa-live.
            # Vì vậy boundary/ID cũ của cùng shard không thể tồn tại sót lại.
            response = requests.post(
                f"{ELASTIC_URL}/{ELASTIC_KEYFRAME_BUILD_INDEX}/"
                "_delete_by_query?conflicts=proceed&refresh=true",
                headers=headers,
                json={
                    "query": {
                        "term": {
                            "pipeline_shard_index": CFG.shard_index,
                        }
                    }
                },
                timeout=300,
            )
            response.raise_for_status()
            print(
                "Cleared keyframe build shard:",
                CFG.shard_index,
                "| deleted=",
                response.json().get("deleted", 0),
            )

            current_link_contract_sha256 = (
                LINK_CONTRACT_SHA256 or VISUAL_CONTRACT_SHA256
            )
            rows_to_index = []
            for source_row in INDEX_RECORDS:
                row = copy.deepcopy(source_row)
                row.update(
                    {
                        "build_generation": KEYFRAME_GENERATION_SHA256,
                        "build_revision": ELASTIC_BUILD_REVISION,
                        "pipeline_shard_index": CFG.shard_index,
                        "pipeline_shard_count": CFG.shard_count,
                        "link_contract_sha256": (
                            current_link_contract_sha256
                        ),
                    }
                )
                rows_to_index.append(row)
            failures = []
            for start in tqdm(range(0, len(rows_to_index), ELASTIC_BATCH_SIZE), desc="Elastic bulk"):
                batch = rows_to_index[start:start + ELASTIC_BATCH_SIZE]
                lines = []
                for row in batch:
                    lines.append(json.dumps(
                        {
                            "index": {
                                "_index": ELASTIC_KEYFRAME_BUILD_INDEX,
                                "_id": row["submit_keyframe_id"],
                            }
                        },
                        separators=(",", ":"),
                    ))
                    lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                payload = ("\n".join(lines) + "\n").encode("utf-8")
                bulk_headers = dict(headers)
                bulk_headers["Content-Type"] = "application/x-ndjson"
                response = requests.post(
                    f"{ELASTIC_URL}/_bulk?refresh=false",
                    headers=bulk_headers,
                    data=payload,
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                if result.get("errors"):
                    failures.extend(
                        item for item in result["items"]
                        if next(iter(item.values())).get("error")
                    )
            if failures:
                raise RuntimeError(f"Elastic bulk có {len(failures)} failures; first={failures[0]}")
            print(
                f"Indexed {len(rows_to_index):,} documents -> "
                f"{ELASTIC_KEYFRAME_BUILD_INDEX}"
            )

            if CFG.run_temporal_pass:
                temporal_mapping = {
                    "settings": {
                        "number_of_shards": 1,
                        "analysis": {
                            "analyzer": {
                                "caption_en": {
                                    "type": "standard",
                                    "stopwords": "_english_",
                                }
                            }
                        },
                    },
                    "mappings": {
                        "dynamic": False,
                        "properties": {
                            "schema_version": {"type": "keyword"},
                            "build_generation": {"type": "keyword"},
                            "build_revision": {"type": "keyword"},
                            "pipeline_shard_index": {"type": "integer"},
                            "pipeline_shard_count": {"type": "integer"},
                            "link_contract_sha256": {"type": "keyword"},
                            "temporal_segment_id": {"type": "keyword"},
                            "group": {"type": "keyword"},
                            "video_id": {"type": "keyword"},
                            "shot_id": {"type": "integer"},
                            "window_index": {"type": "integer"},
                            "window_seconds": {"type": "float"},
                            "stride_seconds": {"type": "float"},
                            "start_s": {"type": "double"},
                            "end_s": {"type": "double"},
                            "start_frame": {"type": "integer"},
                            "end_frame": {"type": "integer"},
                            "keyframe_ids": {"type": "keyword"},
                            "sample_times_s": {"type": "double"},
                            "sample_frame_count": {"type": "integer"},
                            "sample_decode_backend": {"type": "keyword"},
                            "temporal_scope": {"type": "keyword"},
                            "temporal_is_target_specific": {"type": "boolean"},
                            "temporal_caption": {
                                "type": "text",
                                "analyzer": "caption_en",
                            },
                            "temporal_search_text": {
                                "type": "text",
                                "analyzer": "caption_en",
                            },
                            "segmenter": {
                                "properties": {
                                    "name": {"type": "keyword"},
                                    "configured_name": {"type": "keyword"},
                                    "contract_sha256": {"type": "keyword"},
                                    "package": {"type": "keyword"},
                                    "package_version": {"type": "keyword"},
                                    "weights_sha256": {
                                        "type": "keyword",
                                        "index": False,
                                    },
                                    "device": {"type": "keyword"},
                                    "threshold": {"type": "float"},
                                    "input_size": {"type": "integer"},
                                    "min_scene_len_frames": {
                                        "type": "integer",
                                    },
                                    "fallback_reason": {
                                        "type": "keyword",
                                        "index": False,
                                    },
                                    "window_seconds": {"type": "float"},
                                    "stride_seconds": {"type": "float"},
                                }
                            },
                            "temporal_struct": {
                                "properties": {
                                    "action_motion": {"type": "keyword"},
                                    "camera_motion": {"type": "keyword"},
                                    "state_changes": {
                                        "type": "text",
                                        "analyzer": "caption_en",
                                    },
                                    "interaction_summary": {
                                        "type": "text",
                                        "analyzer": "caption_en",
                                    },
                                    "event_phase": {"type": "keyword"},
                                    "confidence": {"type": "keyword"},
                                    "uncertain_fields": {"type": "keyword"},
                                }
                            },
                            "source_video": {
                                "properties": {
                                    "r2_key": {
                                        "type": "keyword",
                                        "index": False,
                                    },
                                    "etag": {"type": "keyword"},
                                    "size_bytes": {"type": "long"},
                                    "version_token": {"type": "keyword"},
                                    "fps": {"type": "float"},
                                    "duration_seconds": {"type": "double"},
                                }
                            },
                            "generation": {
                                "properties": {
                                    "model_id": {"type": "keyword"},
                                    "model_revision": {"type": "keyword"},
                                    "prompt_version": {"type": "keyword"},
                                    "prompt_sha256": {"type": "keyword"},
                                    "parse_status": {"type": "keyword"},
                                    "latency_seconds": {"type": "float"},
                                    "batch_wall_seconds": {"type": "float"},
                                    "batch_size": {"type": "integer"},
                                    "configured_batch_size": {
                                        "type": "integer"
                                    },
                                    "generated_at": {"type": "date"},
                                }
                            },
                        },
                    },
                }
                temporal_exists = requests.head(
                    f"{ELASTIC_URL}/{ELASTIC_TEMPORAL_BUILD_INDEX}",
                    headers=headers,
                    timeout=30,
                )
                if temporal_exists.status_code == 404:
                    response = requests.put(
                        f"{ELASTIC_URL}/{ELASTIC_TEMPORAL_BUILD_INDEX}",
                        headers=headers,
                        json=temporal_mapping,
                        timeout=60,
                    )
                    response.raise_for_status()
                    print(
                        "Created build index:",
                        ELASTIC_TEMPORAL_BUILD_INDEX,
                    )
                else:
                    temporal_exists.raise_for_status()
                    response = requests.put(
                        f"{ELASTIC_URL}/{ELASTIC_TEMPORAL_BUILD_INDEX}/_mapping",
                        headers=headers,
                        json=temporal_mapping["mappings"],
                        timeout=60,
                    )
                    response.raise_for_status()
                    print(
                        "Updated build mapping:",
                        ELASTIC_TEMPORAL_BUILD_INDEX,
                    )

                response = requests.post(
                    f"{ELASTIC_URL}/{ELASTIC_TEMPORAL_BUILD_INDEX}/"
                    "_delete_by_query?conflicts=proceed&refresh=true",
                    headers=headers,
                    json={
                        "query": {
                            "term": {
                                "pipeline_shard_index": CFG.shard_index,
                            }
                        }
                    },
                    timeout=300,
                )
                response.raise_for_status()
                print(
                    "Cleared temporal build shard:",
                    CFG.shard_index,
                    "| deleted=",
                    response.json().get("deleted", 0),
                )

                temporal_rows_to_index = []
                for source_row in TEMPORAL_INDEX_RECORDS:
                    row = copy.deepcopy(source_row)
                    row.update(
                        {
                            "build_generation": (
                                TEMPORAL_GENERATION_SHA256
                            ),
                            "build_revision": ELASTIC_BUILD_REVISION,
                            "pipeline_shard_index": CFG.shard_index,
                            "pipeline_shard_count": CFG.shard_count,
                            "link_contract_sha256": (
                                current_link_contract_sha256
                            ),
                        }
                    )
                    temporal_rows_to_index.append(row)
                temporal_failures = []
                for start in tqdm(
                    range(
                        0,
                        len(temporal_rows_to_index),
                        ELASTIC_BATCH_SIZE,
                    ),
                    desc="Elastic temporal bulk",
                ):
                    batch = temporal_rows_to_index[
                        start:start + ELASTIC_BATCH_SIZE
                    ]
                    lines = []
                    for row in batch:
                        lines.append(
                            json.dumps(
                                {
                                    "index": {
                                        "_index": (
                                            ELASTIC_TEMPORAL_BUILD_INDEX
                                        ),
                                        "_id": row["temporal_segment_id"],
                                    }
                                },
                                separators=(",", ":"),
                            )
                        )
                        lines.append(
                            json.dumps(
                                row,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        )
                    payload = ("\n".join(lines) + "\n").encode("utf-8")
                    bulk_headers = dict(headers)
                    bulk_headers["Content-Type"] = "application/x-ndjson"
                    response = requests.post(
                        f"{ELASTIC_URL}/_bulk?refresh=false",
                        headers=bulk_headers,
                        data=payload,
                        timeout=120,
                    )
                    response.raise_for_status()
                    result = response.json()
                    if result.get("errors"):
                        temporal_failures.extend(
                            item
                            for item in result["items"]
                            if next(iter(item.values())).get("error")
                        )
                if temporal_failures:
                    raise RuntimeError(
                        f"Elastic temporal bulk có "
                        f"{len(temporal_failures)} failures; "
                        f"first={temporal_failures[0]}"
                    )
                print(
                    f"Indexed {len(temporal_rows_to_index):,} documents -> "
                    f"{ELASTIC_TEMPORAL_BUILD_INDEX}"
                )

            if PROMOTE_ELASTIC_ALIASES:
                # Promotion chỉ được phép khi toàn bộ shard của cùng generation đã
                # complete trên R2 và hiện diện đúng count/link-contract trong build.
                manifest_prefix = (
                    LINK_GENERATION_PREFIX
                    if CFG.run_temporal_pass
                    else VISUAL_GENERATION_PREFIX
                )
                manifest_keys = [
                    key
                    for key in r2_list_keys(f"{manifest_prefix}/")
                    if key.endswith("/run_manifest.json")
                ]
                latest_manifest_by_shard: dict[int, dict[str, Any]] = {}
                for manifest_key in manifest_keys:
                    try:
                        raw, _ = r2_get_bytes(manifest_key)
                        candidate = json.loads(raw)
                        shard_index = int(candidate["shard_index"])
                    except Exception as exc:
                        print(
                            "Skip invalid promotion manifest:",
                            manifest_key,
                            type(exc).__name__,
                        )
                        continue
                    if (
                        int(candidate.get("shard_count", -1))
                        != CFG.shard_count
                        or not 0 <= shard_index < CFG.shard_count
                    ):
                        continue
                    if CFG.run_temporal_pass:
                        if (
                            candidate.get(
                                "link_generation_contract_sha256"
                            )
                            != LINK_GENERATION_CONTRACT_SHA256
                        ):
                            continue
                    elif (
                        candidate.get("visual_contract_sha256")
                        != VISUAL_CONTRACT_SHA256
                    ):
                        continue
                    previous = latest_manifest_by_shard.get(shard_index)
                    if (
                        previous is None
                        or str(candidate.get("updated_at", ""))
                        > str(previous.get("updated_at", ""))
                    ):
                        latest_manifest_by_shard[shard_index] = candidate

                expected_shards = set(range(CFG.shard_count))
                ready_shards = set(latest_manifest_by_shard)
                if ready_shards != expected_shards:
                    missing = sorted(expected_shards - ready_shards)
                    raise RuntimeError(
                        "Refuse alias promotion: chưa có generation manifest "
                        f"cho shards={missing} tại {manifest_prefix}"
                    )
                not_ready = {
                    shard_index: {
                        "status": manifest.get("status"),
                        "run_mode": manifest.get("run_mode"),
                    }
                    for shard_index, manifest
                    in latest_manifest_by_shard.items()
                    if (
                        manifest.get("status") != "complete"
                        or manifest.get("run_mode") != "full"
                    )
                }
                if not_ready:
                    raise RuntimeError(
                        "Refuse alias promotion: latest generation manifest "
                        f"chưa full+complete: {not_ready}"
                    )

                for build_index in [
                    ELASTIC_KEYFRAME_BUILD_INDEX,
                    *(
                        [ELASTIC_TEMPORAL_BUILD_INDEX]
                        if CFG.run_temporal_pass
                        else []
                    ),
                ]:
                    response = requests.post(
                        f"{ELASTIC_URL}/{build_index}/_refresh",
                        headers=headers,
                        timeout=60,
                    )
                    response.raise_for_status()


                def count_build_docs(
                    build_index: str,
                    shard_index: int,
                    link_sha256: str,
                ) -> int:
                    response = requests.post(
                        f"{ELASTIC_URL}/{build_index}/_count",
                        headers=headers,
                        json={
                            "query": {
                                "bool": {
                                    "filter": [
                                        {
                                            "term": {
                                                "pipeline_shard_index": (
                                                    shard_index
                                                )
                                            }
                                        },
                                        {
                                            "term": {
                                                "link_contract_sha256": (
                                                    link_sha256
                                                )
                                            }
                                        },
                                        {
                                            "term": {
                                                "build_revision": (
                                                    ELASTIC_BUILD_REVISION
                                                )
                                            }
                                        },
                                    ]
                                }
                            }
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    return int(response.json()["count"])


                expected_keyframe_total = 0
                expected_temporal_total = 0
                for shard_index in sorted(expected_shards):
                    shard_manifest = latest_manifest_by_shard[shard_index]
                    link_sha256 = (
                        shard_manifest.get("link_contract_sha256")
                        or VISUAL_CONTRACT_SHA256
                    )
                    expected_keyframes = int(
                        shard_manifest["target_count"]
                    )
                    actual_keyframes = count_build_docs(
                        ELASTIC_KEYFRAME_BUILD_INDEX,
                        shard_index,
                        link_sha256,
                    )
                    if actual_keyframes != expected_keyframes:
                        raise RuntimeError(
                            "Refuse alias promotion: keyframe build "
                            f"shard={shard_index} count={actual_keyframes}, "
                            f"expected={expected_keyframes}, "
                            f"link={link_sha256[:12]}"
                        )
                    expected_keyframe_total += expected_keyframes

                    if CFG.run_temporal_pass:
                        expected_temporal = int(
                            shard_manifest[
                                "temporal_segments_expected"
                            ]
                        )
                        actual_temporal = count_build_docs(
                            ELASTIC_TEMPORAL_BUILD_INDEX,
                            shard_index,
                            link_sha256,
                        )
                        if actual_temporal != expected_temporal:
                            raise RuntimeError(
                                "Refuse alias promotion: temporal build "
                                f"shard={shard_index} count={actual_temporal}, "
                                f"expected={expected_temporal}, "
                                f"link={link_sha256[:12]}"
                            )
                        expected_temporal_total += expected_temporal


                def count_all_docs(build_index: str) -> int:
                    response = requests.get(
                        f"{ELASTIC_URL}/{build_index}/_count",
                        headers=headers,
                        timeout=60,
                    )
                    response.raise_for_status()
                    return int(response.json()["count"])


                actual_keyframe_total = count_all_docs(
                    ELASTIC_KEYFRAME_BUILD_INDEX
                )
                if actual_keyframe_total != expected_keyframe_total:
                    raise RuntimeError(
                        "Refuse alias promotion: keyframe build contains "
                        f"{actual_keyframe_total} docs, expected exactly "
                        f"{expected_keyframe_total}; possible stale/foreign shard."
                    )
                if CFG.run_temporal_pass:
                    actual_temporal_total = count_all_docs(
                        ELASTIC_TEMPORAL_BUILD_INDEX
                    )
                    if actual_temporal_total != expected_temporal_total:
                        raise RuntimeError(
                            "Refuse alias promotion: temporal build contains "
                            f"{actual_temporal_total} docs, expected exactly "
                            f"{expected_temporal_total}; possible stale generation."
                        )

                # Nếu tên live hiện là concrete index cũ thì Elasticsearch không thể
                # dùng cùng tên làm alias. Refuse để operator snapshot/migrate thủ công.
                for alias, targets in [
                    (ELASTIC_KEYFRAME_ALIAS, keyframe_live_targets),
                    *(
                        [
                            (
                                ELASTIC_TEMPORAL_ALIAS,
                                temporal_live_targets,
                            )
                        ]
                        if CFG.run_temporal_pass
                        else []
                    ),
                ]:
                    if targets:
                        continue
                    response = requests.head(
                        f"{ELASTIC_URL}/{alias}",
                        headers=headers,
                        timeout=30,
                    )
                    if response.status_code == 200:
                        raise RuntimeError(
                            f"Refuse alias promotion: {alias!r} đang là "
                            "concrete index, không phải alias. Snapshot/migrate "
                            "legacy index trước khi cutover."
                        )
                    if response.status_code != 404:
                        response.raise_for_status()

                alias_actions = []
                for old_index in sorted(keyframe_live_targets):
                    alias_actions.append(
                        {
                            "remove": {
                                "index": old_index,
                                "alias": ELASTIC_KEYFRAME_ALIAS,
                            }
                        }
                    )
                alias_actions.append(
                    {
                        "add": {
                            "index": ELASTIC_KEYFRAME_BUILD_INDEX,
                            "alias": ELASTIC_KEYFRAME_ALIAS,
                        }
                    }
                )
                if CFG.run_temporal_pass:
                    for old_index in sorted(temporal_live_targets):
                        alias_actions.append(
                            {
                                "remove": {
                                    "index": old_index,
                                    "alias": ELASTIC_TEMPORAL_ALIAS,
                                }
                            }
                        )
                    alias_actions.append(
                        {
                            "add": {
                                "index": ELASTIC_TEMPORAL_BUILD_INDEX,
                                "alias": ELASTIC_TEMPORAL_ALIAS,
                            }
                        }
                    )
                response = requests.post(
                    f"{ELASTIC_URL}/_aliases",
                    headers=headers,
                    json={"actions": alias_actions},
                    timeout=60,
                )
                response.raise_for_status()
                print(
                    "Atomic alias promotion complete:",
                    ELASTIC_KEYFRAME_ALIAS,
                    "->",
                    ELASTIC_KEYFRAME_BUILD_INDEX,
                )
                if CFG.run_temporal_pass:
                    print(
                        ELASTIC_TEMPORAL_ALIAS,
                        "->",
                        ELASTIC_TEMPORAL_BUILD_INDEX,
                    )
                print(
                    "Old build indices were NOT deleted; remove only after "
                    "rollback window and an explicit snapshot/retention review."
                )
        else:
            print("PUSH_TO_ELASTIC=False — không thay đổi Elasticsearch.")
        """
    ),
    markdown(
        r"""
        ## Runbook vận hành

        ### Chạy thử

        1. Chọn A100 80 GB, tạo Colab Secrets.
        2. Giữ `run_mode="smoke"`, `smoke_limit=64`,
           `smoke_sampling_strategy="group_video_diverse"` và chạy từ trên xuống.
        3. Kiểm tra:
           - `Smoke selection` có đủ số video/group và phân bố early/middle/late;
           - caption có đúng target không;
           - `unknown_pct`;
           - `Smoke semantic coverage` và `probe_gaps`;
           - lỗi parse;
           - latency;
           - temporal caption đúng chuyển động của video;
           - R2 có ba namespace `visual/`, `temporal/`, `linked/`;
           - keyframe index không chứa `caption_text_temporal`/`caption_text_combined`;
           - `temporal_context.temporal_is_target_specific=false`.

        ### Chạy toàn bộ

        1. Đổi `run_mode="full"`.
        2. Chọn một `shard_index` trong `0..15`.
        3. Không chạy hai notebook cùng một shard.
        4. Khi Colab ngắt, chạy lại notebook với đúng config. Writer tải các part và skip ID đã xong.
        5. Sau khi shard complete, chuyển sang shard kế tiếp.

        Pass A và Pass B có contract độc lập. Đổi timing, TransNetV2, window/stride,
        temporal prompt hoặc bật/tắt Pass B không làm mất visual checkpoint. Visual resume
        so ETag/size với immutable manifest; nếu manifest thiếu ETag, chế độ
        `manifest_or_head` sẽ HEAD riêng các target đã checkpoint. Với full production nên tạo
        manifest có `etag,size_bytes` và đặt `visual_resume_source_check="manifest_required"`.
        Chế độ `manifest_required` preflight toàn bộ target ngay sau selection và dừng trước khi
        load model nếu thiếu source metadata.

        `run_temporal_pass=True` mặc định. Mỗi video được tải vào cache, detect shot, caption
        các overlapping window 8 giây/stride 4 giây cần thiết rồi xóa; notebook không giữ
        đồng thời 1.438 video trên local disk. Temporal segment được search ở index riêng;
        backend map `segment hit → keyframe_ids` rồi mới fuse với visual retrieval.
        `temporal_batch_size=8` là batch NVILA thật (8 segment × 8 sampled frames) và yêu
        cầu A100 80 GB. Khi OOM, batch tự chia 8→4→2→1; QA ghi `max_batch_size_used`,
        `batch_size_histogram` và `oom_batch_splits`. Pass A vẫn batch 1 để giữ nhánh
        target-only Dynamic-S2 high-resolution của processor chính thức.
        TransNetV2 dùng CUDA, package/wheel/weights được pin và verify trước `torch.load`.
        Boundary được cache trên R2 theo segmenter contract + video ETag/version nên resume
        không detect lại. Fixed-window fallback bị từ chối mặc định
        (`allow_scene_fallback=False`); circuit breaker dừng sau 5 video lỗi liên tiếp
        hoặc 50 segment lỗi.
        Frame sampler thử OpenCV random seek trước và xác nhận timestamp thực tế. Với
        long-GOP/VFR hoặc backend OpenCV không seek được, nó tự chuyển cả video sang
        ffmpeg fps-pipe (một process cho mỗi segment); backend đã dùng được lưu trong
        `sample_decode_backend`.
        Dừng Pass A bằng interrupt sẽ flush checkpoint rồi raise lại, nên `Run all` không trôi
        sang Pass B. Pass B cũng từ chối chạy nếu còn bất kỳ target visual nào chưa complete/fresh.
        Nếu interrupt giữa Pass B, temporal writer và error log được flush trong outer `finally`
        trước khi `KeyboardInterrupt` tiếp tục propagate.

        Smoke sampler dùng seed cố định và round-robin theo group/video. Mặc định mỗi target
        đến từ một video khác nhau cho tới khi hết video trong shard; vì dataset không có semantic
        label trước caption, các probe studio/night/chart/crowd/disaster chỉ là báo cáo diagnostic
        sau inference, không phải ground-truth coverage guarantee.

        Giữ `context_radius=0` cho caption retrieval chính. Đây là target-only Dynamic-S2,
        bảo toàn chi tiết nhỏ. `context_radius>0` là chế độ benchmark/metadata phụ vì processor
        sẽ resize từng ảnh còn 448 thay vì dùng Dynamic-S2 cho target.

        ### Output R2

        ```text
        caption/nvila8b/v4/
          boundaries/segmenter=transnetv2_gpu/contract=<segmenter-hash>/
            <video_id>/version=<video-hash>.json
          visual/schema=4.1.0/contract=<visual-hash>/
            shard-000-of-016/
              parts/part-000000.jsonl
              errors/errors-<session>.jsonl
              run_manifest.json
          temporal/schema=4.1.0/contract=<temporal-hash>/
            shard-000-of-016/
              segments/parts/part-000000.jsonl
              video-complete/<video_id>/version=<hash>/scope=<hash>.json
              errors/errors-<session>.jsonl
              run_manifest.json
          linked/schema=4.1.0/
            visual=<visual-hash>/temporal=<temporal-hash>/
              video-sources=<hash>/
                shard-000-of-016/
                  parts/part-000000.jsonl
                  run_manifest.json
        ```

        Visual contract chỉ đổi theo image/model/visual prompt/context/dedup. Temporal contract
        chỉ đổi theo timing/model/temporal prompt/segmenter/window. Link contract ghép hai hash
        cùng video source set. Prompt hash bao phủ template, required schema keys và toàn bộ
        vocabulary thực tế. Mỗi temporal record còn khóa theo R2 video ETag/size/version token.

        ### Promote Elasticsearch an toàn

        1. Chọn một `ELASTIC_BUILD_REVISION` mới và giữ nguyên cho cả 16 shard.
        2. Với từng full shard complete, đặt `PUSH_TO_ELASTIC=True`,
           `PROMOTE_ELASTIC_ALIASES=False`.
        3. Sau khi đủ 16 shard, kiểm tra QA và chạy promotion. Notebook đối chiếu manifest R2,
           count từng shard theo `link_contract_sha256`, tổng count build, rồi mới gọi một
           request `_aliases` chứa cả keyframe và temporal cutover.
        4. Không sửa build đã live. Không xóa build cũ cho tới khi hết rollback window.
        5. Nếu live name cũ đang là concrete index thay vì alias, notebook sẽ từ chối cutover;
           snapshot/migrate legacy index thủ công trước.

        ### Timing metadata

        `manifest/media_manifest.csv` cũ không có true video `frame_idx/pts_time/fps`.
        Notebook cố ý ghi `timing_status="missing"` thay vì dùng stem `001` làm frame video.
        Để output sẵn sàng submit DRES, upload một enriched CSV/JSONL lên R2 với:

        ```text
        video_id, keyframe_n, frame_idx, pts_time, fps, r2_key, etag, size_bytes
        ```

        rồi đổi `source_manifest_key`; hoặc đặt `enrich_timing_from_elastic=True` để đọc overlay
        read-only từ `aic26_keyframe_map_v1`. Full chỉ query video trong shard; smoke chỉ query
        video được chọn, không scroll toàn bộ timing index cho mỗi Colab. Canonical ID vẫn là:

        Nếu Elastic Serverless trả HTTP 410 cho `_stats`, đặt
        `timing_dataset_version` thành release tag immutable do bạn quản lý (ví dụ
        `aic26-keyframe-map-2026-07-30-r1`). Giữ nguyên tag cho cả 16 shard và đổi tag
        mỗi khi timing map được rebuild; không dùng ngày ví dụ này nếu nó không đúng release.

        ```text
        <group>/<video_id>/<keyframe_3_digits>
        ```

        ### A100 40 GB

        Config hiện tại đặt `temporal_batch_size=8` nên notebook chủ động từ chối A100 40 GB.
        Chọn A100 80 GB cho production. Nếu chỉ debug trên A100 40 GB, phải hạ
        `temporal_batch_size=1`; không bật CPU offload vì throughput sẽ rất thấp và dễ lỗi
        device placement.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "aic26_nvila8b_r2_captioning_colab.ipynb",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).with_name("aic26_nvila8b_r2_captioning_colab.ipynb")
out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(out_path)
