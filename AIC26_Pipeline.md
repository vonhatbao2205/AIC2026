# AIC 2026 – Pipeline thống nhất

> File này đặc tả pipeline duy nhất, từ ingestion đến UI, có thể triển khai trực tiếp. Phần phân tích nền và brainstorm chiến lược nằm tại `AIC26_Analysis_Strategy.md`. Các phần đánh dấu `[verify]` là version/benchmark cần kiểm tra trước khi commit.

---

## 0. Sơ đồ tổng thể

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            OFFLINE INDEXING                              │
│                                                                          │
│  Videos BTC ──┬──► Keyframe Self-Cut ─┐                                 │
│               │      (TransNetV2)      ├──► Dedup (pHash + cosine)      │
│   Keyframes ──┴──────────────────────┘         │                        │
│   BTC                                          ▼                        │
│                                          KeyframeRegistry               │
│                                          (id, video, t, source)         │
│                                                │                        │
│       ┌──────────┬──────────┬──────────┬──────┴────┬──────────┐         │
│       ▼          ▼          ▼          ▼           ▼          ▼         │
│   PE-G     SigLIP2/EVA   Qwen2.5-VL   Whisper    OCR        Action     │
│   embed    embed         caption      ASR        Paddle+    VideoMAE   │
│                          (free+struct)            VietOCR    +Places   │
│       │          │          │          │           │          │         │
│       ▼          ▼          ▼          ▼           ▼          ▼         │
│   FAISS-PE   FAISS-SIG   ES/PG-cap   ES/PG-asr  ES/PG-ocr  PG-tags     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              ONLINE QUERY                                │
│                                                                          │
│   User Voice/Text Query                                                  │
│          │                                                               │
│          ▼                                                               │
│   Query Understanding (LLM)                                              │
│   ├─ Translate VI→EN                                                     │
│   ├─ Extract entities/actions/attributes                                 │
│   ├─ Classify: T-KIS / QA / V-KIS / TRAKE                                │
│   ├─ Modal cues: OCR-likely? ASR-likely?                                 │
│   └─ Paraphrase × 3-5                                                    │
│          │                                                               │
│          ▼                                                               │
│   Multi-channel Retrieval (parallel)                                     │
│   ├─ Dense: PE, SigLIP2, EVA-CLIP                                        │
│   ├─ Text: BM25 caption + BM25 OCR (cond) + BM25 ASR (cond)              │
│   └─ Filter: scene/action/object tags                                    │
│          │                                                               │
│          ▼                                                               │
│   Fusion: RRF (k=60) → top-100                                           │
│          │                                                               │
│          ▼                                                               │
│   Rerank: VLM cross-encoder (top-20)                                     │
│          │                                                               │
│          ▼                                                               │
│   Type-specific Postprocessing                                           │
│   ├─ T-KIS: group-by-video + dispersion score                            │
│   ├─ QA: pre-compute CoT for top-20                                      │
│   ├─ V-KIS: structured field matching                                    │
│   └─ TRAKE: DP sequence localization                                     │
│          │                                                               │
│          ▼                                                               │
│   UI Display + Submit Guard + DRES Submit                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Stage 1 – Ingestion & Preprocessing

### 1.1. Keyframe registry

**Mục tiêu:** mọi keyframe có duy nhất một `keyframe_id`, traceable về frame BTC nếu cần submit.

**Schema (Postgres):**

```sql
CREATE TABLE keyframe_registry (
    keyframe_id   TEXT PRIMARY KEY,         -- vd "L01_V001__0042" hoặc "L01_V001__extra_0007"
    video_id      TEXT NOT NULL,            -- "L01_V001"
    frame_idx     INT,                      -- index trong BTC (NULL nếu self-cut)
    timestamp_s   FLOAT NOT NULL,           -- giây từ đầu video
    source        TEXT NOT NULL,            -- 'btc' | 'self_cut'
    phash         BIGINT,                   -- pHash 64-bit
    closest_btc   TEXT,                     -- với self_cut: id keyframe BTC gần nhất
    INDEX (video_id, timestamp_s)
);
```

### 1.2. Self-cut keyframe

- **Shot boundary detector:** TransNetV2 `[verify model name/version]` – chính xác cho video tin tức cắt cảnh nhanh.
- **Fallback dense sampling:** 1 frame/giây cho video không có shot rõ.
- **Dedup:**
  1. pHash hashing → các pair có Hamming distance ≤ 4 ⇒ trùng.
  2. Cho phần còn nghi ngờ, check cosine similarity PE embedding ≥ 0.97 ⇒ trùng.
  3. Ưu tiên giữ keyframe BTC khi trùng (vì submit phải dùng id BTC).

### 1.3. Mapping self-cut → BTC nearest

Khi submit kết quả là self-cut keyframe, map về BTC keyframe gần nhất theo `timestamp_s`. Lưu sẵn cột `closest_btc` để O(1) lookup.

---

## 2. Stage 2 – Feature Extraction Multi-modal

### 2.1. Schema lưu trữ

**Vector indexes (FAISS HNSW):**
- `index_pe.faiss` – PE embedding, dim `[verify]` (~1024)
- `index_siglip.faiss` – SigLIP 2 embedding
- `index_eva.faiss` – EVA-CLIP-L embedding
- Mỗi index map `int_id → keyframe_id` qua file `mapping.json`.

**Postgres (cho text-fields, structured fields, tags):**

```sql
CREATE TABLE keyframe_features (
    keyframe_id     TEXT PRIMARY KEY REFERENCES keyframe_registry,
    caption_free    TEXT,                   -- 2-3 câu mô tả VLM tự do
    caption_struct  JSONB,                  -- template structured (xem 2.3)
    ocr_text        TEXT,                   -- OCR concatenated
    ocr_lang        TEXT[],                 -- ['vi', 'en']
    object_tags     TEXT[],                 -- ['person', 'fire', 'building']
    object_details  JSONB,                  -- [{label, bbox, conf}, ...]
    scene_tag       TEXT[],                 -- ['street', 'outdoor']
    action_tag      TEXT[],                 -- ['running', 'shouting']
    -- full-text search indexes
    caption_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', caption_free)) STORED,
    ocr_tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', ocr_text)) STORED
);
CREATE INDEX idx_caption_tsv ON keyframe_features USING GIN(caption_tsv);
CREATE INDEX idx_ocr_tsv     ON keyframe_features USING GIN(ocr_tsv);
CREATE INDEX idx_object_tags ON keyframe_features USING GIN(object_tags);

CREATE TABLE video_asr (
    asr_id        SERIAL PRIMARY KEY,
    video_id      TEXT NOT NULL,
    start_s       FLOAT,
    end_s         FLOAT,
    transcript    TEXT,
    transcript_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', transcript)) STORED
);
CREATE INDEX idx_asr_tsv     ON video_asr USING GIN(transcript_tsv);
CREATE INDEX idx_asr_video   ON video_asr (video_id, start_s);
```

### 2.2. Captioning Free-form (VLM)

**Model:** Qwen2.5-VL-7B-Instruct `[verify version]`, quantization int8.

**Prompt:**
```
You are describing a frame from a Vietnamese news video for a retrieval system.
Write 2-3 sentences in English describing:
- the main subject and what they are doing
- the scene/location
- any salient visual details (colors, weather, signs)
Be concrete. Avoid speculation. No bullet points.
```

### 2.3. Captioning Structured

**Cùng VLM, prompt khác. Output JSON tuân thủ schema:**

```json
{
  "scene_type": "outdoor_street",
  "camera_angle": "eye_level",
  "weather": "sunny",
  "lighting": "bright_daylight",
  "location_hints": ["urban", "vietnam"],
  "num_people": 5,
  "people_desc": "men in dark suits, one woman in red ao dai",
  "key_objects": ["microphone", "podium", "flag"],
  "dominant_colors": ["#1a3d7c", "#d62828", "#f5f5f5"],
  "ocr_text": "(blank if none)",
  "action_verbs": ["speaking", "standing"],
  "notable_details": "official press conference setup"
}
```

Lưu vào `caption_struct JSONB`, query bằng `caption_struct -> 'scene_type' = 'outdoor_street'`.

### 2.4. OCR

- **PaddleOCR multilingual** primary (vie + en).
- **VietOCR** fallback nếu PaddleOCR confidence trung bình < 0.6 cho một frame.
- **Post-process:** LLM (small, vd Qwen 1.5B) fix lỗi dấu phổ biến cho tiếng Việt nếu sentence-level confidence thấp.
- Lưu raw OCR + cleaned OCR.

### 2.5. ASR

- **Whisper large-v3**, language `vi`.
- **VAD pre-process** với Silero VAD để cắt segment không có giọng nói.
- Output: list segments `(start_s, end_s, transcript)`. Lưu vào `video_asr`.

### 2.6. Object Detection (open-vocab)

- **YOLO-World** hoặc **OWLv2** `[verify]`.
- Prompt-able với danh sách concept dynamic, nhưng cho offline indexing, chạy với vocabulary mặc định ~1000 concepts (mở rộng từ OpenImages + concepts đặc thù VN: "áo dài", "xe ôm", "phở", "công an"...).
- Combine với Faster R-CNN BTC để có 2 nguồn.

### 2.7. Scene + Action Tags

- **Places365 CNN** cho scene tag (top-3 mỗi frame).
- **VideoMAE V2 Kinetics-700** `[verify]` cho action tag, chạy trên cửa sổ 16 frame quanh keyframe.

### 2.8. Dense embedding text-fields

- **Caption (free + struct fields concat)**: encode bằng multilingual-E5 `[verify]` để search dense trên text.
- **ASR transcript per segment**: encode tương tự, lưu vào FAISS riêng cho text-search.

---

## 3. Stage 3 – Query Pipeline (Online)

### 3.1. Query Understanding

**Input:** câu query tiếng Việt (gõ hoặc voice → Whisper).

**LLM call (Qwen2.5-7B-Instruct hoặc API):**

```
System: You parse Vietnamese video retrieval queries.
User query: {query}

Output JSON:
{
  "query_type_hint": "T-KIS|QA|V-KIS|TRAKE",   # if obvious
  "translated_en": "...",
  "entities": ["..."],
  "actions": ["..."],
  "attributes": {"color": "...", "location": "..."},
  "ocr_likely": true|false,
  "asr_likely": true|false,
  "paraphrases_vi": ["...", "...", "..."],
  "paraphrases_en": ["...", "...", "..."],
  "negations": ["..."]  # what should NOT be in result
}
```

### 3.2. Multi-channel retrieval

Chạy **song song** (5 channels):

1. **Dense PE:** encode `translated_en` + paraphrases_en, mean-pool → search FAISS-PE top-200.
2. **Dense SigLIP 2:** encode multilingual query (giữ tiếng Việt) → FAISS-SIG top-200.
3. **Dense EVA-CLIP:** tương tự PE → FAISS-EVA top-200.
4. **Dense text (E5)** trên caption + ASR.
5. **Sparse BM25**:
   - Always: `caption_tsv` match `entities + actions`.
   - Conditional (ocr_likely): `ocr_tsv` match.
   - Conditional (asr_likely): `transcript_tsv` match → map ASR segment về video_id + timestamp → tìm keyframe gần.

### 3.3. Fusion với RRF

```python
def rrf_fuse(ranked_lists, k=60):
    scores = defaultdict(float)
    for L in ranked_lists:
        for rank, doc_id in enumerate(L):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:100]
```

Trả về **top-100 keyframe_id**.

### 3.4. Rerank tầng 2 (VLM cross-encoder)

- Top-20 keyframe → VLM prompt:
  ```
  Image: <frame>
  Query: "{translated_en}"
  Question: Does this image match the query?
  Answer in JSON: {"match": "yes|partial|no", "confidence": 0.0-1.0, "reason": "..."}
  ```
- Batch 4-8 frame song song trên A100.
- Score: `2*P(yes) + 1*P(partial)` để phá đảo top.

### 3.5. Negation filter

Nếu `negations` không rỗng: với mỗi top-K kết quả, encode `negation` qua PE/SigLIP, nếu cosine ≥ threshold (0.3) ⇒ loại.

### 3.6. Type-specific postprocessing

#### 3.6.1. T-KIS

```python
def tkis_postprocess(top100):
    grouped = group_by_video(top100)
    for vid, frames in grouped.items():
        scores = [f.score for f in frames]
        ts     = [f.timestamp_s for f in frames]
        s_video = (
            0.5 * max(scores)
            + 0.3 * mean(top_n(scores, 5))
            + 0.2 * len(frames) * exp(-variance(ts) / 1000)
        )
        grouped[vid].video_score = s_video
        # cluster detection
        if has_two_distant_clusters(ts):
            grouped[vid].warning = "AMBIGUOUS"
    return sorted(grouped.values(), key=lambda v: -v.video_score)
```

UI hiển thị từng video (header + thumbnail cluster), cảnh báo `AMBIGUOUS` nếu có.

#### 3.6.2. QA – pre-compute CoT

Ngay khi top-20 chốt, async dispatch 20 VLM-CoT call:

```
Image: <frame>
Question: "{question}"
Think step by step. Use the image and the following ASR context: "{asr_window}"
Output JSON: {"reasoning": "...", "answer": "...", "confidence": 0.0-1.0}
```

UI hiển thị frame + answer + reasoning. Người vận hành chọn.

#### 3.6.3. V-KIS – structured field match

UI có form-based query với các field giống template structured caption. Người vận hành điền (có thể bỏ trống). Match:

```sql
SELECT keyframe_id, similarity_score
FROM keyframe_features
WHERE (caption_struct->>'scene_type') = $scene_type
  AND (caption_struct->>'weather')    = $weather   -- (or NULL match)
  AND (caption_struct->>'camera_angle') = $camera_angle
  AND ...
ORDER BY -- combined with dense score
```

Kết hợp với dense search trên `notable_details + people_desc` free-text.

#### 3.6.4. TRAKE – DP sequence localization

```python
def trake_localize(video_id, events):  # events = [E_1, ..., E_n]
    frames = get_frames_of_video(video_id)  # sorted by timestamp
    T = len(frames)
    n = len(events)
    # score_matrix[i][t] = match score of E_i with frame_t
    score = [[match(E_i, frames[t]) for t in range(T)] for E_i in events]
    # DP: dp[i][t] = best sum picking E_1..E_i with last frame at t
    dp = [[-inf]*T for _ in range(n)]
    bt = [[None]*T for _ in range(n)]
    for t in range(T):
        dp[0][t] = score[0][t]
    for i in range(1, n):
        running_max = -inf
        running_argmax = None
        for t in range(T):
            if dp[i-1][t-1] > running_max:
                running_max = dp[i-1][t-1]
                running_argmax = t-1
            dp[i][t] = score[i][t] + running_max
            bt[i][t] = running_argmax
    # trace back from argmax(dp[n-1])
    end_t = argmax(dp[n-1])
    path = [end_t]
    for i in range(n-1, 0, -1):
        path.append(bt[i][path[-1]])
    return list(reversed(path)), dp[n-1][end_t]
```

Chạy DP trên top-5 video candidates (xếp hạng theo tổng max score across events). Trả về video có DP score cao nhất kèm `n` frame.

---

## 4. Stage 4 – UI/UX

### 4.1. Layout

```
┌───────────────────────────────────────────────────────────────────────┐
│ [TYPE] T-KIS / QA / V-KIS / TRAKE   ⏱ 0:42 / 5:00   PENALTY: 0       │
├───────────────────────────────────────────────────────────────────────┤
│ QUERY:                                                                │
│ ┌─────────────────────────────────────────────────────────────────┐   │
│ │ [voice 🎙] một đám cháy ở khu dân cư, có 3 xe cứu hỏa...        │   │
│ │ [LLM rewrite suggestion: ...] [accept]                           │   │
│ └─────────────────────────────────────────────────────────────────┘   │
│ [SEARCH] [+ APPEND HINT]                                              │
├───────────────────────────────────────────────────────────────────────┤
│ RESULTS (grouped by video):                                           │
│ ┌─ ▼ L05_V012 ─────────────────────── score: 0.92  ⚠ no warning ─┐  │
│ │  [frame thumb1] [frame thumb2] [frame thumb3] [+ 4 more]        │  │
│ │  caption: "outdoor street fire with three fire trucks..."        │  │
│ │  OCR: "TIN NÓNG | HỎA HOẠN Q.7"                                 │  │
│ └──────────────────────────────────────────────────────────────────┘  │
│ ┌─ ▶ L02_V088 ─────────────────────── score: 0.78  ⚠ AMBIGUOUS  ─┐  │
│ │  ...                                                             │  │
│ └──────────────────────────────────────────────────────────────────┘  │
├───────────────────────────────────────────────────────────────────────┤
│ SELECTED:  L05_V012 / frame 0042                                      │
│ [PREVIEW ±2s]  [CoT view (QA only)]  [TIMELINE (TRAKE only)]          │
│                                                                       │
│ [SUBMIT] (Enter)   – guard modal sẽ confirm trước khi gửi DRES         │
├───────────────────────────────────────────────────────────────────────┤
│ SUBMIT HISTORY: ✓ L05_V012/0042  ✗ L05_V012/0089  ...                 │
└───────────────────────────────────────────────────────────────────────┘
```

### 4.2. Keyboard shortcuts

| Phím | Action |
|---|---|
| `/` | Focus query input |
| `Ctrl+M` | Bắt đầu voice input |
| `Enter` | Search (in query) hoặc Submit (in result) |
| `↑/↓` | Chọn video |
| `←/→` | Chọn frame trong video |
| `Space` | Preview ±2s |
| `Tab` | Toggle CoT view (QA) |
| `T` | Toggle timeline (TRAKE) |
| `Esc` | Hủy submit guard |
| `Ctrl+Z` | Undo (rollback selection) |

### 4.3. Submit guard modal

Khi `Enter` trên SUBMIT, hiện modal:

```
┌──────────────────────────────────────────────────┐
│  CONFIRM SUBMIT                                  │
│  ┌──────────────────────────────────────────┐    │
│  │ [Frame thumbnail full-size]              │    │
│  │ video: L05_V012  |  frame: 0042          │    │
│  │ caption: "outdoor fire with 3 trucks..." │    │
│  │ OCR: "TIN NÓNG | HỎA HOẠN Q.7"           │    │
│  │ ASR ±5s: "phóng viên có mặt tại..."      │    │
│  │ (QA only) answer: "3"                    │    │
│  └──────────────────────────────────────────┘    │
│  Type:  T-KIS                                    │
│  Penalty so far: 0                               │
│                                                  │
│  [CANCEL (Esc)]     [SUBMIT TO DRES (Enter)]      │
└──────────────────────────────────────────────────┘
```

### 4.4. Voice-input + LLM rewrite

- Whisper local (small/base model đủ) cho real-time.
- LLM rewrite: lấy raw transcript, output query chuẩn hóa, hiển thị inline cho người vận hành accept/reject.

### 4.5. Submit history sidebar

- Mỗi submit lưu (timestamp, type, keyframe_id, answer, result).
- Block resubmit cùng `keyframe_id` cho cùng câu.
- Hiển thị penalty đếm dần.

---

## 5. Stack công nghệ

| Lớp | Công nghệ |
|---|---|
| Vector index | FAISS HNSW (CPU-served) |
| Relational + Text index | Postgres 15+ với GIN/GIST (`pg_trgm`, `tsvector`) |
| Optional full-text | Elasticsearch hoặc Meilisearch nếu Postgres chậm |
| ML serving | vLLM hoặc TGI cho Qwen2.5-VL; sentence-transformers cho E5 |
| API layer | FastAPI (Python) |
| Front-end | Next.js (React) hoặc Streamlit (nếu cần build nhanh) |
| Voice | OpenAI Whisper (local, small) |
| Orchestration | Docker Compose cho dev, K8s không cần |
| Monitoring | Prometheus + Grafana (latency mỗi endpoint) |

---

## 6. Latency budget mỗi câu

| Step | Budget |
|---|---|
| Voice → text | 1-2 s |
| LLM query rewrite | 1-2 s |
| Multi-channel search + RRF | 0.5 s |
| VLM rerank top-20 | 5-8 s |
| (QA only) CoT top-20 | 10-15 s, async ẩn sau UI |
| UI render | 0.2 s |
| Human review + select | 15-40 s |
| Submit guard read | 3-5 s |
| **Total p50** | **~25-40 s/câu** với câu dễ |
| **Total p90** | **~60-90 s/câu** với câu khó |

---

## 7. Lưu trữ dự kiến

- 100K keyframe × 1024 dim × 4 bytes × 3 backbones ≈ **1.2 GB** vector index
- Captions free + struct: ~500 KB × 100K = **50 GB** plain text JSON (nén ~10 GB)
- ASR transcript: ~10 GB
- OCR: ~5 GB
- pHash + meta: < 1 GB

Total ~80 GB. Vừa 1 ổ SSD 256 GB thoải mái.

---

## 8. Checkpoint & resilience (Colab)

- **Mỗi 500 keyframe** save partial: vector batch → `.npy`, captions → JSONL append.
- **Khi resume:** đọc `last_keyframe_id`, skip những id đã xong.
- **Tách thành 5 shard** theo `video_id % 5`, mỗi A100 phụ trách 1 shard ⇒ độc lập, không lock.
- **Export FAISS sau mỗi shard** xong, merge cuối cùng bằng `index.merge_from(...)`.

---

## 9. Bộ test thử / mock

- Tự tạo 40 mock query (10 mỗi dạng) từ video đã biết.
- Mỗi mock có ground truth `<video, frame>` (TRAKE: nhiều frame).
- Metric:
  - Recall@1, Recall@10 cho T-KIS.
  - Exact-match answer cho QA.
  - Recall@10 + frame distance cho V-KIS.
  - Sequence accuracy (full vs ≥n/2) cho TRAKE.
- Đo cả thời gian end-to-end mỗi câu để mô phỏng điều kiện thi.

---

## 10. Checklist trước ngày thi

- [ ] Vector indexes tải xong, mở thử và search sanity check
- [ ] Postgres backup full snapshot, restore test
- [ ] Toàn bộ serving endpoint chạy trên local/VPS, không phụ thuộc Colab
- [ ] Voice input test với loa phòng thi (giả lập)
- [ ] Người vận hành luyện ngón ≥ 200 query mock
- [ ] DRES connection test với credentials thật
- [ ] Submit history persistent (không mất khi reload tab)
- [ ] Backup plan: nếu UI sập, fallback CLI tool gửi DRES qua curl
- [ ] In sẵn 1 cheatsheet phím tắt để dán cạnh máy

---

> Pipeline này được thiết kế để **không có single point of failure**. Mỗi component (vector, BM25, OCR, ASR, VLM) đều có thể tạm sập mà các kênh còn lại vẫn cho ra kết quả khả dụng. Đây là tính chất quan trọng nhất khi triển khai trong môi trường thi đấu.
