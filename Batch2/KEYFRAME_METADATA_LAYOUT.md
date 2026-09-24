# AIC 2026 keyframe metadata layout

The local workspace separates the two AIC keyframe families while preserving
the original `infoshootpp` corpus name:

```text
keyframe_L/infoshootpp/   # 873 L21..L30 videos (existing corpus)
keyframe_K/infoshootpp/   # 605 K01..K20 videos, canonical Pass 2
keyframe_K/history/infoshootpp_pre_pass2_2453542/  # snapshot K gốc để rollback
keyframe_batch2/          # batch 2: 304 M (209.111 kf) + 298 N (44.572 kf) + 12 S01 (621.117 kf), ảnh nằm trên R2
```

Batch 2 không có ảnh local: JPEG nằm trên R2 `aic26-infoshot-keyframes/Keyframes/Keyframes_{Mxx,Nxxx,S01}/`.
`keyframe_batch2/infoshoot_{m,n,s}/final/` là bản sao metadata `final/*` trên R2, còn `keyframe_batch2/audit/`
là kết quả `audit_batch2_r2_keyframes.py`. Xem `AIC2026_Batch2_M_N_Keyframe_R2_HANDOFF.md`.

Canonical K hiện tại có fingerprint `66621afe60c9e726`, gồm **1.979.073** keyframe sau
Pass 1 + Pass 2. Metadata nằm local; ảnh nằm trong 605 ZIP Pass 2 tại:

```text
hf://buckets/Baonenha1/DATA-AIC-Keyframe/
postprocessed_K_pass2_templates_v1/66621afe60c9e726/archives/
```

`keyframe_K/infoshootpp/frame_registry.parquet` là source of truth cho embedding/OCR. Registry
đã dùng locator `hf_archive_path + image_archive_member` và logical `hfzip://` URI, nên không cần
tải toàn bộ 314,925 GB ảnh khi chuẩn bị metadata.

## Reproduce local K Pass 2 promotion

```bash
.venv-l25-post/bin/python analyze_k_pass2_complete.py
.venv-l25-post/bin/python promote_k_pass2_metadata.py
```

Promotion dùng staging, kiểm tra đủ 605 registry/map/archive locator rồi mới đổi canonical folder.
Snapshot K gốc 2.453.542 frame được rename sang history, không bị xóa.

Canonical folder giữ:

- `frame_registry.parquet`: 1.979.073 row final;
- `asset_index.csv`: 605 ZIP Pass 2 và SHA-256;
- `map-keyframes/`: 605 map CSV đã đánh lại `n` sau Pass 2;
- `run_summary.csv`, `source_sessions.csv`, `source_index.csv`;
- `metadata/`: decision/excluded audit, archive manifest, map ZIP và analysis;
- `metadata/send.zip`: gói metadata downstream;
- `HANDOFF_VECTOR_OCR.md`: data contract cho embedding/OCR.

Per-video registry CSV raw cũ không được copy sang canonical Pass 2 vì trùng với global Parquet
và tốn hơn 2 GB. Chúng vẫn còn nguyên trong snapshot history.

Format map của K giống L:

```text
n,pts_time,fps,frame_idx
```

Tên JPEG vẫn là `f{frame_idx:08d}.jpg`; không suy `n=1` thành `001.jpg`. Xem
`KEYFRAME_K_PASS2_VECTOR_OCR_HANDOFF.md` để biết cách iterate ZIP và checkpoint downstream.
