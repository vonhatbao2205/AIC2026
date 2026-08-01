# AIC26 Retrieval Console — Danh mục tính năng

Toàn bộ tính năng hiện có của hệ thống truy hồi video AIC26 (backend FastAPI + frontend React/Vite). Tài liệu vận hành/contract chi tiết: [RETRIEVAL_APP_HANDOFF.md](RETRIEVAL_APP_HANDOFF.md).

---

## 1. Kiến trúc tổng quan

- **Backend** `backend/` — FastAPI, adapter tách module: Elastic, Milvus, PE encoder, GLAP encoder; service: search / trake / timeline / submit; logic thuần (identity, media, fusion, trake, scoring) test được độc lập.
- **Frontend** `frontend/` — React + Vite + TypeScript. Frontend **chỉ gọi backend**, không giữ secret.
- **Models/infra** (Kaggle/Colab + cloud): PE-Core-G14 (ảnh+text), GLAP (audio↔text), NVILA-8B (visual QA), Milvus/Zilliz (vector), Elastic Cloud (OCR/speech/audio/keyframe-map), Cloudflare R2 (media), NVIDIA NIM (LLM parse/expansion), faster-whisper (voice STT).
- **Mock mode** (`AIC26_MOCK_MODE=true`): mọi adapter trả fixture cố định → chạy UI/test không cần service thật.

## 2. Quy tắc định danh (bất biến)
- `submit_keyframe_id` = `"<category>/<video_id>/<frame_3_digits>"` (vd `K01/K01_V001/001`), cũng là `image_id`.
- **Submit dùng `video_id` + `frame_idx`** (đúng format DRES) — KHÔNG dùng `image_path`/`source_path`.
- Chuẩn hoá shard **L26_a/L26_b → L26**.
- Media URL build từ `submit_keyframe_id`/`video_id`+`keyframe_n`.

## 3. Hai chế độ giao diện
- **Console (đầy đủ)** ⇄ **Simple** — nút chuyển góc trên, nhớ lựa chọn trong `localStorage`.
- **Simple**: 1 ô query + thanh trượt số keyframe → lưới keyframe theo cosine (vector-only, không parser/fusion).
- **Console**: dashboard 3 cột keyboard-first (query/kênh/understanding · kết quả+video+timeline · detail/TRAKE/history).

## 4. Dark / Light mode
- Toggle sun/moon (SVG) ở top bar + Simple. Lưu localStorage, lần đầu theo `prefers-color-scheme`.
- **Soft dark** (không OLED gắt) + light dịu; ít border, dùng elevation; scrollbar mảnh; chữ rõ.

## 5. Hiểu & định tuyến query (Query understanding)
- **Heuristic (mặc định, ~0ms)**: phân loại query_type, bật kênh theo dấu hiệu (OCR-likely/speech/audio), tách negation, tách event TRAKE.
- **LLM tùy chọn** (toggle **🧠 LLM on**): NVIDIA NIM (mặc định `qwen/qwen3-next-80b-a3b-instruct`) trả JSON routing đầy đủ. Có cache theo query; JSON retry 1 lần; lỗi/chậm → fallback heuristic.
  - **Slim-schema** (opt-in `SLIM_PARSE=true`): nhanh ~2× nhưng A/B cho thấy đôi khi sai query_type → mặc định TẮT.
- **Dịch VI→EN** (mặc định bật, `TRANSLATE_TO_EN`): khi LLM off, dịch query → `image_pe` (PE cần tiếng Anh); OCR/speech giữ tiếng Việt. TRAKE dịch từng event.
- **Query expansion** (toggle **🔎 Expand**): model nhanh (`NVIDIA_FAST_MODEL`, vd `meta/llama-3.1-8b-instruct`) sinh 2-3 mô tả thị giác/ query → backend search từng biến thể, **fuse max-cosine** → tăng recall cho concept khó (vd "The Thing"). Lọc bỏ biến thể không phải tiếng Anh.
- **Panel Query Understanding**: type+confidence, EN visual (clamp "xem thêm"), kênh được route, negations.

## 6. Kênh truy hồi (Retrieval channels)
- **image_pe (VECTOR)**: PE-Core-G14 encode query → Milvus `aic26_image_peg14_v1` (COSINE). Hỗ trợ **multi-variant max-fusion** (query expansion).
- **OCR**: Elastic `text_clean`/`text_nfc`/`text_clean_fold` (fuzzy) + filter `hour`/`clock`. Guard: query rỗng → trả rỗng (không match-all).
- **speech**: Elastic ASR `text`; hạ điểm `confidence_bucket` low/mid, `segment_role` intro/preview.
- **audio**: Elastic `top1_label`/`tag_labels`/`caption` (drop stoplist, hạ generic/vietnamese_asr) **FUSE (rank-RRF) với GLAP audio-vector** (`aic26_audio_glap_v1`, COSINE) khi GLAP bật. Frame do vector tìm có **badge `GLAP`** trên kết quả + evidence.
- Toggle bật/tắt từng kênh thủ công (force/disable), hiển thị weight + "auto".

## 7. Fusion & hiển thị kết quả
- **RRF (k=60)** trên `submit_keyframe_id`, có trọng số kênh; gom evidence mọi kênh.
- **Group-by-video** với điểm chuẩn hoá **coverage-first**: `0.75·best + 0.15·mean_top3 + 0.10·cluster_support` (đều [0,1]); video có frame mạnh không bị video nhiều-frame-yếu chôn. Cờ **AMBIGUOUS** khi top frame tách cụm thời gian xa.
- **2 chế độ xem**: **Group by video** (mở sẵn tất cả group, badge kênh, feedback) ⇄ **Flat top-K** (lưới phẳng theo điểm). Bù `pts_time`/`frame_idx`/`fps` bằng 1 lần Elastic `_mget`.
- **Channel attribution**: mỗi frame có badge kênh + "explain match" (evidence snippet, điểm, thời gian).
- **Relevance feedback**: nút **More like this** (boost video) / **Exclude** (loại frame) → re-rank phiên search.

## 8. 4 dạng tác vụ
- **T-KIS**: hỗ trợ **Append hint** (gộp hint tích lũy).
- **QA**: retrieval lấy tối đa 12 frame; frame user đang xem luôn là **C01** và được tính vào quota của video đó. Từ C02, candidate được xếp theo block ưu tiên `video_score`: lấy đủ tối đa 3 frame đa dạng của video đứng đầu trước rồi mới sang video hạng tiếp theo. Sau đó NVILA-8B chạy **3–5 hotspot hypotheses → 3–5 visual answer options** → DeepSeek web search → **NVILA pass 3**. `contradicted` bị loại; `insufficient/unverified` bị hạ confidence.
- **V-KIS**: như KIS hình ảnh.
- **TRAKE** (chi tiết §9).

## 9. TRAKE (chuỗi sự kiện)
- **Tách event**: nhận `E1:/E2:`, `sự kiện 1:`, đánh số, từ nối ("sau đó/rồi/…").
- **Two-pass retrieval**:
  1. Truy hồi từng event (song song, pool rộng `top_k≈400`).
  2. **In-video fill (order-aware)**: video tiềm năng thiếu event → search lại đúng video đó (Milvus filter `video_id`) cho event thiếu, có ngưỡng tương đồng tương đối. Chọn hit điểm cao nhất **chèn vừa thứ tự thời gian** giữa các neighbor đã có (tránh fill frame lệch thứ tự bị DP loại → phí lượt lấp); frame fill được đánh dấu `via_fill`.
- **DP tối ưu**: max-weight strictly-increasing chain theo (event_index, pts_time) → 1 frame/event, thời gian tăng dần, **coverage-first** (đếm event phủ trước, rồi điểm). Cho phép partial.
- **Xếp hạng theo evidence thật**: cross-video rank theo `confident_coverage` (event do retrieval thật phủ, **loại** event fill) trước → video 4/4 "giả" (nhiều fill) không đè được video có nhiều bằng chứng thật; clean full-coverage luôn lên đầu.
- **Badge coverage** `4/4 / 3/4 events` + chip `+N in-video` khi có fill + `μ` điểm TB frame thật → operator thấy độ tin cậy, tránh wrong submit.
- **Submit nhanh**: video đủ event → nút **Submit sequence ↵** điền hết slot + mở guard.
- **Frame-pick lúc pause là tính năng global** (phím `v` mở video → pause): lấy **frame_idx = round(pts×fps) ngay tại điểm dừng** (KHÔNG snap về BTC keyframe) và thumbnail trực tiếp từ video (canvas). T-KIS/QA/V-KIS tự dùng raw frame này làm submit target (có thể chuyển lại result keyframe); TRAKE kéo thả/Enter để gán vào event slot.
- **Kéo-thả keyframe giữa các ô event** để sắp xếp lại thứ tự (swap); validate thứ tự tăng dần, cảnh báo `⚠ order`.
- **Auto-fill E1..En** từ video đang chọn; marker event màu trên timeline.

## 10. Submit (định dạng DRES) & lịch sử
- Payload: **TKIS/VKIS** `{video_id, frame_idx, timestamp}` · **QA** `+answer` · **TRAKE** `{video_id, events:[{event_index, frame_idx}]}` (frame_idx tăng dần). Map T-KIS→TKIS, V-KIS→VKIS.
- **Submit guard** trước mọi submit: thumbnail, video_id, frame_idx, nguồn raw paused frame hoặc btc id, thời gian, evidence OCR/ASR/audio, ô answer (QA), danh sách frame_idx có thứ tự (TRAKE).
- **Chống nộp trùng**: dedup theo `video_id:frame_idx` (KIS/QA) / `video_id|f1,f2,…` (TRAKE) per task → 409 + nút "Submit anyway". Chặn submit khi không tính được frame_idx.
- **Submit history sidebar**: loại task, frame, trạng thái (local / dres_ok / dres_error), cờ dup, answer.
- **DRES adapter** tùy chọn (`DRES_BASE_URL`/`DRES_TOKEN`); thiếu thì lưu local history (đúng format).

## 11. Video & timeline
- **Video inline**: phím **`v`** trên keyframe đang chọn → video hiện **ngay dưới group đó**, **tua đúng keyframe**; chọn keyframe khác cùng video → chỉ seek (không reload, load 1 lần/video); chọn keyframe khác video → tự ẩn. Video **fill khung** (object-fit cover, bỏ viền đen). `Space` = play/pause gốc.
- **Timeline keyframe filmstrip** + playhead + marker event TRAKE (đã bỏ track OCR/speech/audio/heatmap để nhanh); click seek.
- Endpoint `GET /api/videos/{id}/timeline` trả keyframes (sorted, frame_idx/pts_time/url); `POST .../snap` snap raw→keyframe (cho frame-pick).

## 12. Voice input
- Nút **🎙 voice** / phím **Ctrl+M** (mọi nơi).
- **Web Speech API** (Chrome/Edge): interim realtime (`…đang nói`), nói liên tục, lang vi-VN.
- **Whisper backend** (`POST /api/transcribe`, faster-whisper, `WHISPER_MODEL`): chạy **mọi trình duyệt kể cả Brave** (Brave tắt Web Speech → tự fallback Whisper). Trả transcript VI + bản dịch EN.
- **Tự dịch VI→EN** sau khi nói (Whisper trả sẵn; Web Speech gọi `/api/translate`).

## 13. Phím tắt
`/` focus query · `Enter` search / mở guard / xác nhận / gán paused frame cho TRAKE · `↑↓` chọn video · `←→` chọn frame · `v` hiện/ẩn video · `Space` play/pause và capture raw frame · `Tab` đổi vùng · `T` ẩn/hiện timeline · `Esc` đóng modal/xóa paused frame · `Ctrl+M` voice · `Ctrl+/` **bảng phím tắt**. Nút **?** ở top bar mở keymap.

## 14. Hiệu năng & độ bền
- **Latency badge**: parse/fusion/total ms.
- **Song song hoá**: kênh search, TRAKE per-event + pass-2, timeline (4 query Elastic), enrich 1 `_mget`.
- **Chịu lỗi từng kênh**: kênh chết (PE/GLAP down) → bỏ qua + cảnh báo, kênh khác vẫn chạy (không sập search).
- **Fallback**: dịch lỗi → giữ bản gốc; GLAP/Whisper/PE không có → fallback rõ ràng; cache parser + timeline (client).
- **Health** `GET /api/health`: trạng thái Elastic/Milvus/PE + `capabilities` (llm_query_parser, audio_vector_search, dres_submit, vlm_*) + warnings. UI hiện banner mock/lỗi.

## 15. API endpoints
| Method | Path | Mục đích |
|---|---|---|
| GET | `/api/health` | trạng thái service + capabilities |
| POST | `/api/query/parse` | routing JSON (heuristic/LLM) |
| POST | `/api/search` | search đầy đủ (group-by-video) |
| POST | `/api/search/simple` | vector-only flat top-K |
| POST | `/api/search/trake` | chuỗi event (two-pass + DP) |
| POST | `/api/qa/analyze` | NVILA visual QA + optional DeepSeek web-search grounding/citations |
| GET | `/api/keyframes/{submit_keyframe_id:path}` | chuẩn hoá id + URL + timing |
| GET | `/api/videos/{video_id}/timeline` | keyframes timeline |
| POST | `/api/videos/{video_id}/snap` | snap raw→keyframe |
| POST | `/api/submit` | submit (DRES format) + dedup guard |
| GET | `/api/submit/history` | lịch sử submit |
| POST | `/api/translate` | dịch VI→EN |
| POST | `/api/transcribe` | Whisper STT (audio→text+EN) |

## 16. Cấu hình (env, xem `backend/.env.example`)
`ELASTIC_ENDPOINT/API_KEY` · `MILVUS_ENDPOINT/TOKEN` · `PE_ENCODER_URL`(+`_TOKEN`) · `GLAP_ENCODER_URL` (mặc định = PE) · `NVILA_BASE_URL/TOKEN` · `NVILA_TIMEOUT_SECONDS/MAX_CANDIDATES` · `DEEPSEEK_API_KEY` · `DEEPSEEK_GROUNDING_*` · `MEDIA_BASE_URL` · `NVIDIA_API_KEY/BASE_URL` · `NVIDIA_MODEL` (parse) · `NVIDIA_FAST_MODEL` (expansion) · `SLIM_PARSE` · `TRANSLATE_TO_EN` · `WHISPER_MODEL` · `DRES_BASE_URL/TOKEN` · `IDX_*` · `MILVUS_IMAGE_COLLECTION` · `MILVUS_AUDIO_COLLECTION` · `AIC26_MOCK_MODE` · `CORS_ORIGINS`.

## 17. Models & dữ liệu
- **PE-Core-G14-448** (1280-d) — ảnh + text, Kaggle FastAPI + cloudflared (`model-setup-backend.ipynb`).
- **GLAP `mispeech/GLAP`** (1024-d) — audio↔text, text encoder chạy **CPU** (`/encode-audio-text`); cùng notebook PE hoặc `glap-encoder-kaggle.ipynb` riêng.
- **Milvus**: `aic26_image_peg14_v1` (382,299), `aic26_audio_glap_v1` (466,996).
- **Elastic**: `aic26_keyframe_map_v1`, `aic26_ocr_keyframes_v1`, `aic26_speech_segments_v1`, `aic26_audio_windows_v1`.
- **NVIDIA NIM**: qwen3-next (parse), llama-3.1-8b (expansion). **faster-whisper** (voice).
- **NVILA-8B**: Colab A100 BF16 worker (`aic26_nvila8b_qa_colab_server.ipynb`), multi-image QA hai pass cộng post-search visual verification, không cần caption keyframe tạo sẵn.
- **DeepSeek `deepseek-v4-flash` + built-in `web_search`**: backend-only knowledge grounding sau NVILA, gọi qua **Responses API** (`POST /responses`) — `/chat/completions` từ chối mọi tool không phải `function`, và chỉ `deepseek-v4-flash` hỗ trợ tool này. Query và các trang model thực sự mở được trả về UI để operator kiểm chứng.

## 18. Chưa làm (cố ý) / điểm mở rộng
- **Chưa có**: general Qwen3-VL reranker cho mọi task, structured VLM caption offline, object/scene/action tag filter, SigLIP/EVA ensemble, true clip-query V-KIS. QA đã có NVILA trực tiếp nhìn candidate, nhưng chưa có dense-caption Answer Span Prediction trên toàn corpus.
- **Điểm mở rộng**: thêm `RerankerClient` + bước rerank sau fusion (flip `capabilities.vlm_rerank`); object filter qua parser `filters` + Elastic; query-vector morphing cho feedback.

## 19. Chạy & test
```bash
# Backend (mock, không cần service)
cd backend && AIC26_MOCK_MODE=true PYTHONPATH=. ../.venv/bin/python -m uvicorn app.main:app --port 8000
# Backend (live: đọc .env)
cd backend && PYTHONPATH=. ../.venv/bin/python -m uvicorn app.main:app --port 8000
# Frontend
cd frontend && npm install && npm run dev      # proxy /api -> :8000

# Tests
cd backend && PYTHONPATH=. ../.venv/bin/python -m pytest -q     # 88
cd frontend && npm run test                                    # 18
```
