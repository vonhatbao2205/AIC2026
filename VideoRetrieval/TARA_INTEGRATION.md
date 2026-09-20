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

## Text encoder Colab

Mở `VideoRetrieval/TARA_Text_Encoder_Server_Colab_A100.ipynb` trên Colab A100,
chạy từ đầu đến cuối. Notebook tự chứa worker FastAPI, pin Python 3.10 / Torch
2.5.1+cu121 / Transformers 4.45.0 / FlashAttention 2.8.3, xác minh SHA-256
checkpoint và trả FP32 unit-norm. `/encode-text` dùng Bearer token. Notebook
kiểm tra local và public endpoint, rồi in các biến cần đưa vào `backend/.env`:

```dotenv
TARA_ENABLED=true
TARA_ENCODER_URL=https://<tunnel-hostname>
TARA_ENCODER_TOKEN=<token từ notebook>
MILVUS_TARA_COLLECTION_2=aic26_tara_clips_infoshotpp_v1
```

Giữ Colab runtime và tunnel chạy khi truy hồi. Có thể dùng quick tunnel, URL
sẽ đổi mỗi lần; để giữ URL, cấu hình riêng `CF_TUNNEL_TOKEN_TARA` và
`TARA_ENCODER_HOSTNAME` trong Colab Secrets, route hostname về
`http://localhost:8001`. Không dùng chung tunnel token của Qwen vì tunnel sẽ
load balance request sang worker khác.

## Fusion trong backend

- Chỉ profile **InfoShot++ L21–L30** dùng TARA. BTC và canvas V-KIS vẫn theo
  đường hiện tại. `TARA_ENABLED=false` giữ nguyên retrieval cũ.
- T-KIS/QA: query đã dịch tiếng Anh được encode một lần; Milvus tìm riêng ba
  scale `event`/`sequence`/`scene`. Mỗi scale chỉ cho một phiếu mỗi video,
  rồi RRF giữa scale. RRF tiếp theo giữa PE, Qwen, OCR/ASR/audio và TARA ở
  **video level**, tránh cộng cosine giữa các embedding space.
- Result có `best_clip` (scale, start/end, score). UI hiện nút thời gian để
  nhảy video đến clip. Video chỉ do TARA tìm ra vẫn xuất hiện dù chưa có
  keyframe trong frame retrieval.
- TRAKE: mỗi event tìm TARA clip scale `event`; Elasticsearch map midpoint về
  keyframe thật, rồi đưa vào temporal DP hiện có cùng các candidate PE/Qwen.
  TARA không thay luật thứ tự của DP.
- Nếu text server lỗi, backend báo warning và trả kết quả từ các kênh cũ.

Xem `/api/health?retrieval_database=infoshotpp`: chỉ khi text worker khớp model,
revision, fingerprint và Milvus có đúng 168.536 row thì
`capabilities.tara_clip_search` mới thành `true`.
