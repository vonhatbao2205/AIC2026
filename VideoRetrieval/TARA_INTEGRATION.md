# TARA clip retrieval trong AIC2026

## Artifact đã đối chiếu

Nguồn: HF bucket `Baonenha1/aic26-media`, prefix
`derived/tara-tarsier2-7b-3584-clip-v1`. Dữ liệu tải về nằm ở
`VideoRetrieval/tara_embeddings/` và được `.gitignore` bỏ qua.

`tara_artifact.py` kiểm tra `_SUCCESS.json` bằng manifest, semantic fingerprint,
169 commit, SHA-256 từng Parquet, schema, 3584 chiều FP32, norm L2, clip ID duy
nhất và số dòng/video. Kết quả trên artifact hiện tại: **168.536 clip, 873 video,
169 shard, L21–L30**, fingerprint
`ef197331649dfb93e7057304f294d8bd1cd98bf22b1902bca48daf6dc4874819`.

Chạy lại:

```bash
./backend/.venv/bin/python VideoRetrieval/tara_artifact.py --download
./backend/.venv/bin/python VideoRetrieval/tara_artifact.py
```

Script đọc token từ `HF_BUCKET/HF_TOKEN.txt`; không ghi token vào artifact hay log.

## Milvus Cloud

Collection InfoShot++: `aic26_tara_clips_infoshotpp_v1`, metric COSINE, vector
`FLOAT_VECTOR(3584)`. Mỗi row có `clip_id` làm primary key, `video_id`, `category`,
`scale`, `start_time`, `end_time`, `fps`, `center_frame_idx` và vector. Chỉ số
frame ở đây là chỉ số frame video thật, không phải ordinal keyframe InfoShot++.

```bash
./backend/.venv/bin/python VideoRetrieval/milvus_upload_tara.py --dry-run
./backend/.venv/bin/python VideoRetrieval/milvus_upload_tara.py --workers 6
```

Uploader lấy `MILVUS_ENDPOINT_2`/`MILVUS_TOKEN_2` từ `backend/.env`, audit toàn
artifact trước khi ghi, tạo collection version mới, upsert theo `clip_id` và
resume qua `VideoRetrieval/tara_embeddings/milvus_upload_state.json`. Script
không drop collection. Kết thúc chỉ thành công khi cloud báo đúng 168.536 row.

Collection hiện đã upload xong và được kiểm tra trực tiếp: **168.536/168.536
row indexed, 0 row pending, AUTOINDEX/COSINE ở trạng thái `Finished`**. Self
search bằng vector gốc trả đúng `clip_id` đầu bảng cho cả ba scale
`event`/`sequence`/`scene`.

### Batch 2 (M, N, S01)

Hai artifact batch 2 (`tara-tarsier2-7b-3584-batch2-clip-v1`,
`tara-tarsier2-7b-3584-batch2-s01-clip-v1`, xem
`TARA_Tarsier2_7B_Batch2_AIC2026_Retrieval_Handoff.md`) cùng embedding contract
`611ef1c9…` với L, nên được upsert vào **cùng collection**, cùng schema. Bản local
nằm trong `VideoRetrieval/` và được `.gitignore` bỏ qua.

```bash
./backend/.venv/bin/python VideoRetrieval/milvus_upload_tara_batch2.py --dry-run
./backend/.venv/bin/python VideoRetrieval/milvus_upload_tara_batch2.py --check-remote
./backend/.venv/bin/python VideoRetrieval/milvus_upload_tara_batch2.py --workers 6
```

M/N chưa có `_SUCCESS.json` và S01 mới là snapshot một phần, nên script audit theo
commit: fingerprint, contract, size + SHA-256 từng Parquet, rồi từng dòng. Script
bỏ 1.591 clip đuôi đen trùng byte của M10_V029 (`start_time ≥ 1112`), không tạo
hay drop collection, và từ chối chạy nếu collection không còn đúng 168.536 row L.
Checkpoint ở `.milvus_upload_state_tara_batch2.json`, kết quả kiểm chứng ở
`VideoRetrieval/milvus_upload_tara_batch2_verification.json`.

| | Row trong collection |
|---|---:|
| L21–L30 | 168.536 |
| M01–M10 (đã bỏ đuôi đen M10_V029) | 127.354 |
| N001–N100 | 154.509 |
| S01 (snapshot: thiếu V010, V011 và phần cuối V007) | 45.756 |
| **Tổng** | **496.155** |

Khi notebook S01 commit thêm shard: tải commit + Parquet mới (handoff §9), tăng
số S01 trong `ARTIFACTS`/`EXPECTED_BY_SET` của script và `TARA_EXPECTED_ROWS`
trong `backend/app/adapters/milvus_client.py`, rồi chạy lại. Shard đã upload được
bỏ qua.

## Text encoder Colab

Mở `VideoRetrieval/TARA_Text_Encoder_Server_Colab_A100.ipynb` trên Colab A100,
chạy từ đầu đến cuối. Notebook tự chứa worker FastAPI (bản gốc:
`VideoRetrieval/tara_text_server.py`), pin Python 3.10 / Torch 2.5.1+cu121 /
Transformers 4.45.0 / FlashAttention 2.8.3, xác minh SHA-256 checkpoint và trả
FP32 unit-norm. **Không có bearer token**: `/health` và `/encode-text` public tại
custom domain cố định, nên backend chỉ cần:

```dotenv
TARA_ENABLED=true
TARA_ENCODER_URL=https://tara.baoencoder.site
MILVUS_TARA_COLLECTION_2=aic26_tara_clips_infoshotpp_v1
```

`TARA_ENCODER_TOKEN` để trống; backend chỉ gửi header `Authorization` khi biến
này có giá trị.

Làm một lần trên Cloudflare: tạo một named tunnel **riêng** cho TARA, thêm Public
Hostname `tara.baoencoder.site` → HTTP `localhost:8001`, rồi đặt token của
tunnel vào Colab Secret `CF_TUNNEL_TOKEN_TARA`. Không dùng lại tunnel của PE,
Qwen hay reranker: hai connector trên cùng tunnel thành replica và Cloudflare
chia request sang worker khác. Notebook đọc route Cloudflare đẩy xuống và dừng
nếu token thuộc tunnel của server khác. Không có secret này thì notebook dùng
quick tunnel, URL đổi mỗi lần.

Giữ Colab runtime chạy khi truy hồi. Không có token nên ai biết hostname cũng
gọi được `/encode-text`; mỗi request bị giới hạn 16 câu, 4.096 ký tự/câu.

## Fusion trong backend

- Chỉ profile **InfoShot++** (L21–L30 và batch 2) dùng TARA. BTC và canvas
  V-KIS vẫn theo đường hiện tại. `TARA_ENABLED=false` giữ nguyên retrieval cũ.
- Bộ lọc camera/chặng đua (`FrameFilter`) áp cho TARA ở mức video vì clip không
  có `keyframe_n`: camera/chặng được chọn giữ nguyên cả video, camera khác bị
  loại. TRAKE lọc lại theo keyframe sau khi map midpoint về keyframe thật.
- Cửa sổ của N là 4/8/16 s (tên scale vẫn `event`/`sequence`/`scene`).
- T-KIS/QA: query đã dịch tiếng Anh được encode một lần; Milvus tìm riêng ba
  scale `event`/`sequence`/`scene`. Mỗi scale chỉ cho một phiếu mỗi video,
  rồi RRF giữa scale. RRF tiếp theo giữa PE, Qwen, OCR/ASR/audio và TARA ở
  **video level**, tránh cộng cosine giữa các embedding space.
- Result có `best_clip` (scale, start/end, score). UI hiện nút thời gian để
  nhảy video đến clip. Video chỉ do TARA tìm ra vẫn xuất hiện dù chưa có
  keyframe trong frame retrieval.
- TRAKE: mỗi event tìm TARA clip scale `event`; Elasticsearch map midpoint về
  keyframe thật, rồi đưa vào temporal DP hiện có cùng các candidate PE/Qwen.
  Pass 2 (bù event còn thiếu trong một video) cũng tìm clip TARA trong đúng video
  đó, dùng lại vector query của pass 1. TARA không thay luật thứ tự của DP. Frame
  do TARA tìm ra mang channel `tara` và có tag TARA trên thẻ event.
- TARA-only: bỏ tick mọi model keyframe khi TARA đang bật sẽ tắt kênh VISUAL. T-KIS/QA
  khi đó chỉ chạy TARA; TRAKE bỏ PE/Qwen ở cả pass 1 và pass 2. Nếu TARA không khả
  dụng, TRAKE vẫn giữ PE.
- Nếu text server lỗi, backend báo warning và trả kết quả từ các kênh cũ.

Xem `/api/health?retrieval_database=infoshotpp`: chỉ khi text worker khớp model,
revision, fingerprint và Milvus có đúng 496.155 row (L + batch 2) thì
`capabilities.tara_clip_search` mới thành `true`.
