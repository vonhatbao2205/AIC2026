# AIC26 Retrieval App — Backend + Frontend Handoff

An operator-first video retrieval console for AIC26 over the existing assets
(PE-Core-G14 image vectors in Milvus; OCR / speech / audio / keyframe-map in
Elastic; media on Cloudflare R2; optional NVIDIA Nemotron query parser; optional
DRES submit). Design principle: **UI > Method > Model** — reduce wrong submits
and navigation time.

The data-extraction / upload scripts at the repo root (`keyframe_mapping.py`,
`elastic_upload.py`, `milvus_upload.py`, the notebooks) are **unchanged**. This
app only *reads* the indices/collections they produced.

```
backend/    FastAPI service (adapters + fusion + parser + submit)
frontend/   Vite + React + TS operator console (keyboard-first single screen)
```

---

## 1. Identity & routing rules (load-bearing)

- `submit_keyframe_id` = `"<category>/<video_id>/<frame_3_digits>"`, e.g.
  `K01/K01_V001/001`. It is the public id and **`image_id == submit_keyframe_id`**.
- `image_path` / `source_path` are **never** used for routing, joins, media
  lookup, cache keys, or submit. Robust joins use `video_id + keyframe_n`.
- L26 shards (`L26_a`, `L26_b`, …) normalize to category **`L26`** for
  submit/display. Implemented in `backend/app/identity.py` and mirrored in
  `frontend/src/lib/identity.ts`.
- Media URLs are built only from `submit_keyframe_id` / `video_id + keyframe_n`
  (`backend/app/media.py`, `frontend/src/lib/media.ts`).

---

## 2. Environment variables

Backend reads config from env (or, for dev convenience, the secret `.txt`
files already in the repo root). See `backend/.env.example`.

| Var | Required | Purpose |
|---|---|---|
| `ELASTIC_ENDPOINT`, `ELASTIC_API_KEY` | for live OCR/speech/audio/timeline | Elastic Cloud |
| `MILVUS_ENDPOINT`, `MILVUS_TOKEN` | for live image vector search | Zilliz/Milvus |
| `PE_ENCODER_URL` | for live image search | Kaggle PE-Core-G14 `/encode-text` server |
| `PE_ENCODER_TOKEN` | optional | sent as `Authorization: Bearer` if set |
| `MEDIA_BASE_URL` | yes | Cloudflare R2 public base (keyframes/videos) |
| `NVIDIA_API_KEY` | optional | enables Nemotron query parser (else heuristics) |
| `NVIDIA_BASE_URL`, `NVIDIA_MODEL` | optional | default NIM endpoint + `nvidia/nemotron-3-ultra-550b-a55b` |
| `NVILA_BASE_URL`, `NVILA_TOKEN` | optional | Colab A100 NVILA-8B QA worker; both are required to enable it |
| `NVILA_TIMEOUT_SECONDS`, `NVILA_MAX_CANDIDATES` | optional | visual QA timeout and trusted candidate cap (defaults 240s/12) |
| `DEEPSEEK_API_KEY` | optional | DeepSeek built-in `web_search` grounding for QA; backend-only secret |
| `DEEPSEEK_GROUNDING_*` | optional | model/base URL/enable flag/timeout/max output tokens/reasoning effort/auto-confidence threshold |
| `DRES_BASE_URL`, `DRES_TOKEN` | optional | DRES submit adapter (else local history only) |
| `IDX_*`, `MILVUS_IMAGE_COLLECTION` | optional | override index/collection names |
| `AIC26_MOCK_MODE` | optional | `true` ⇒ deterministic fixtures (no live services) |
| `CORS_ORIGINS` | optional | comma-separated; default `*` |

**Secrets never reach the frontend.** The browser talks only to the backend.

---

## 3. Running

### Backend

```bash
cd backend
# deps already in repo .venv; otherwise:  pip install -r requirements.txt
# Dev / UI work without live services:
AIC26_MOCK_MODE=true PYTHONPATH=. ../.venv/bin/python -m uvicorn app.main:app --reload --port 8000
# Live (reads ELASTIC_*/MILVUS_*/PE_ENCODER_URL/MEDIA_BASE_URL from env or repo .txt files):
PYTHONPATH=. ../.venv/bin/python -m uvicorn app.main:app --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api -> http://localhost:8000
# point elsewhere: VITE_API_TARGET=http://host:8000 npm run dev
```

### Tests

```bash
cd backend  && PYTHONPATH=. ../.venv/bin/python -m pytest -q     # 68 tests
cd frontend && npm run test                                      # 13 tests
```

---

## 4. API contract

All endpoints are under `/api`. Responses are JSON.

| Method | Path | Body / Notes |
|---|---|---|
| GET | `/api/health` | service reachability + `mode` (`mock`/`live`) + `capabilities` flags + `warnings[]` |
| POST | `/api/query/parse` | `{query, query_type_hint, previous_hints[], manual_overrides}` → routing JSON |
| POST | `/api/search` | `{query, query_type_hint, previous_hints[], manual_overrides, parsed?, feedback?, top_k, max_videos}` |
| POST | `/api/search/trake` | `{query, previous_hints[], manual_overrides}` → ordered sequences |
| GET | `/api/canvas/palette` | V-KIS canvas vocabulary: 16 OD colours + canonical labels (+ `colorable`) |
| POST | `/api/search/canvas` | `{canvas{objects[{label,bbox,color,required}], action_text, mode}}` → same group shape, plus `object_layout` evidence |
| POST | `/api/qa/analyze` | `{question, candidates[{submit_keyframe_id,...}], max_answers}` → grounded answers + hotspots |
| GET | `/api/keyframes/{submit_keyframe_id:path}` | normalizes shard/padding; returns ids + media URLs + timing |
| GET | `/api/videos/{video_id}/timeline` | keyframes + speech + ocr + audio + heatmap |
| POST | `/api/videos/{video_id}/snap` | `{raw_time, fps?}` → nearest BTC keyframe (for TRAKE frame-pick) |
| POST | `/api/submit` | DRES format: `{task_id, query_type, payload{video_id, frame_idx?, timestamp?, answer?, events?[{event_index,frame_idx}]}, allow_duplicate}` |
| GET | `/api/submit/history` | `?task_id=` optional |

### Search result shape

```jsonc
{
  "query": "...",
  "parsed": { /* routing JSON, see §5 */ },
  "groups": [{
    "video_id": "K01_V001",
    "video_score": 1.23, "max_score": 0.9, "mean_top_score": 0.5,
    "frame_count": 8, "timestamp_dispersion": 12.3, "ambiguous": false,
    "channels": ["image_pe","ocr"],
    "video_url": "https://.../Videos/Videos_K01/K01_V001.mp4",
    "frames": [{
      "image_id": "K01/K01_V001/001",        // == submit_keyframe_id
      "submit_keyframe_id": "K01/K01_V001/001",
      "video_id": "K01_V001", "keyframe_n": 1, "pts_time": 0.0,
      "score": 0.0312, "channels": ["image_pe","ocr"],
      "per_channel_score": {"image_pe": 0.9, "ocr": 5.0},
      "keyframe_url": "https://.../Keyframes/Keyframes_K01/K01_V001/001.jpg",
      "video_url": "https://.../Videos/Videos_K01/K01_V001.mp4",
      "evidence": [{"type":"ocr","score":5.0,"text":"...","clock":"18:29:57"}]
    }]
  }],
  "latency_ms": {"parse_ms":1,"fusion_ms":0.2,"channels":{...},"total_ms":5},
  "mode": "mock"
}
```

Submit guard returns HTTP **409** `{detail:{error:"duplicate_submit",submit_keyframe_id,task_id}}`
when the same frame was already submitted for the task (unless `allow_duplicate`).

---

## 5. Pipeline

1. **Parse** (`query_parser.py`): NVIDIA Nemotron (OpenAI-compatible
   `chat.completions`, `temperature=0.1`, `enable_thinking:false`, one JSON
   retry) when `NVIDIA_API_KEY` is set; otherwise deterministic heuristics.
   `manual_overrides` (operator channel toggles) are applied last and
   `image_pe` is force-enabled if everything else is off.
2. **Retrieve in parallel** over enabled channels:
   - `image_pe`: PE `/encode-text` → Milvus `aic26_image_peg14_v1` (COSINE); hit
     ids are `submit_keyframe_id`.
   - `ocr`: Elastic `text_clean` (+`text_nfc`) / `text_clean_fold` (fuzzy) +
     `hour`/`clock` filters.
   - `speech`: Elastic ASR `text`; `confidence_bucket` low/mid and
     `segment_role` intro/preview demoted (`scoring.py`).
   - `audio`: Elastic `top1_label`/`tag_labels`/`caption`; top1 trusted over
     secondary tags; stoplisted dropped; `generic`/`vietnamese_asr` captions demoted.
3. **Fuse** with RRF (`k=60`) over `submit_keyframe_id`, channel-weighted;
   relevance feedback (positive videos boost, negative frames drop / videos demote).
4. **Group by video** with `video_score = max + mean(top) + count_bonus +
   dispersion`, and an **ambiguous** flag when top frames split into distant
   time clusters.

**V-KIS canvas** (`/api/search/canvas`) is a separate entry point. The operator
draws on a real `<canvas>`: object silhouettes with a label/colour/bbox, plus
freehand strokes for what has no label at all (sky, a rice field). Three channels
come out of that one drawing:

- `object_layout`: OD candidates from Elastic (`aic26_od_frames_v1`, frame-level
  `canonical_labels` gate with a ratio `minimum_should_match`), then a per-frame
  **Hungarian one-to-one assignment** between drawn objects and real detections
  (`backend/app/canvas.py`). Elastic's nested clauses score independently, so two
  drawn people would otherwise both match one detected person; the assignment is
  what forbids that. Label mismatch scores 0; colour is scored (only when OD
  marked it reliable), never filtered; a missing *required* object and a present
  *excluded* object are penalties, because the detector's vocabulary is finite.
  `rough` weights label/centre/relative-relation, `precise` weights IoU/size/colour.
- `image_pe`: 2–3 short English sentences generated from the same JSON by rule
  (0 ms, no hallucination) through the existing PE → Milvus path.
- `canvas_image` (optional, weight 0.2): the rendered PNG through
  `{PE_ENCODER_URL}/encode-image` → the same Milvus keyframe collection. Editor
  chrome (thirds guide, handles, label chips) is excluded from that render. The
  weight stays low on purpose — a sketch is far outside PE's photo distribution,
  so this channel exists to reach objects with no OD label, not to rank.
  The route is hot-added by §7 of `model-setup-backend.ipynb`; on an older server
  the backend reports it as a warning and the other two channels still answer.
  Only inline `data:image/png|jpeg;base64` is accepted — never a URL, which would
  make the encoder fetch arbitrary hosts.

All three are fused with the same RRF/group-by-video as the main search, and the
matched detections travel back as `object_layout` evidence so the UI can draw
each detection over the keyframe in the colour of the box that matched it.

TRAKE runs the pipeline per event (each event's visual query translated VI→EN),
then assembles the best per-video sequence with an **exact DP** — a maximum-weight
strictly-increasing chain over (event index, `pts_time`) that picks ≤1 frame per
event, maximizing (events covered, then total relevance). It is globally optimal
(unlike greedy), can skip an event for a partial sequence, and ranks videos
coverage-first. See `assemble_trake_sequences` in `backend/app/trake.py`.

---

## 6. Frontend (operator console)

Single screen, three columns: **query/channels/understanding** · **grouped
results + video + multi-track timeline** · **detail/evidence + TRAKE + history**.

Keyboard: `/` focus query · `Enter` search (in query) / open guard (results) /
confirm (modal) / assign chip (TRAKE) · `↑/↓` video group · `←/→` frame ·
`Space` play-pause the candidate video · `Tab` switch zone · `T` toggle timeline ·
`Esc` cancel modal/chip · `Ctrl+M` voice (if browser supports it).

**TRAKE pause frame-pick**: `requestVideoFrameCallback` tracks the latest
`mediaTime`; on pause the raw frame is snapped to the nearest BTC keyframe by
`frame_idx` (fallback `pts_time`). A draggable chip shows raw time/frame, snapped
`submit_keyframe_id`, and Δframes/Δseconds with a far-warning; `Enter` or drag
assigns it to the active event slot. Slots validate increasing `pts_time`.

**QA copilot** packs a maximum of 12 canonical keyframes. The operator-inspected
frame is always C01 and counts toward that video's three-frame quota. From C02,
the remaining candidates are packed in video-score order: fill the diverse-frame
quota of the highest-ranked video before moving to the next video. NVILA pass 1
retains 3–5 answer-bearing hypotheses;
pass 2 asks for 3–5 alternatives when evidence supports them. Zero-confidence,
placeholder, and unknown-ID outputs are discarded. For entity/world-knowledge
questions, the backend can call DeepSeek's built-in web_search tool and merge
cited alternatives without granting the web stage authority to create frames.
Those alternatives are sent back to NVILA for pass-3 visual verification:
`contradicted` is discarded, while `insufficient/unverified` is confidence-capped
and visibly labelled. The UI displays all input candidates and waits for the
operator to click an answer before it fills the draft or opens evidence.

**Submit guard** shows thumbnail, ids, timestamp, OCR/ASR/audio snippets (and an
editable NVILA/manual answer for QA), blocks duplicates, and shows the ordered id list for TRAKE.

**Mock mode**: if services are unreachable, `/api/health` surfaces a warning
banner and the UI can be driven entirely from backend fixtures. Live mode never
fabricates retrieval results.

---

### HTTP connection pooling

Adapters share one pooled `httpx.AsyncClient` each (`adapters/http_pool.py`).
Creating a client per request re-ran the TLS handshake every time: measured at
765 ms vs 249 ms per Elastic call, paid by every channel of every search. After
pooling, a warm process answers a 2-object canvas search in ~1.7 s (layout 0.94 s,
PE text 0.73 s, fusion/enrich 0.75 s) and a two-channel T-KIS search in ~0.87 s.
The first request of a process still pays the handshake once.

## 7. Current limitations (by design)

- **Audio vector search** uses GLAP (`mispeech/GLAP`, 1024-d) over Milvus
  `aic26_audio_glap_v1`, FUSED (rank-RRF) with the Elastic tag/caption signal in
  the audio channel. It needs the `/encode-audio-text` endpoint on the Kaggle PE
  server (see `model-setup-backend.ipynb`); if that endpoint is down the audio
  channel falls back to Elastic-only automatically.
- **QA visual assistance is online/candidate-based** — NVILA directly inspects
  retrieved frames because dense captions are not yet indexed. It does not scan
  the whole video autonomously; web grounding resolves external facts but
  cannot repair a missing visual candidate. Timeline verification remains operator-guided.
- **No** general Qwen3-VL reranker, structured VLM captions, object/scene/action tag
  filtering, SigLIP/EVA ensemble, or true clip-query V-KIS.
- Elastic uses built-in `text` analyzers (no custom Vietnamese tokenizer yet).

---

## 8. Extension points (clean seams left in place)

- **General VLM reranker**: QA-specific NVILA hotspot/answer assistance is DONE.
  A post-fusion reranker for T-KIS/V-KIS/TRAKE still needs a separate adapter and
  search-service stage; `capabilities.vlm_rerank` deliberately remains false.
- **Audio vector search**: DONE — `GlapEncoderClient` (/encode-audio-text) +
  `milvus.search_audio` over `aic26_audio_glap_v1`, fused into the audio channel.
- **Object/scene/action filters**: extend the parser `filters` block + an
  Elastic adapter method; apply in `_apply_filters`.
- **Official DRES**: set `DRES_BASE_URL`/`DRES_TOKEN`; `submit_service._submit_dres`
  already speaks `POST /api/submissions` — adjust the body to the live schema.
- **Query-vector morphing** for relevance feedback: today it is deterministic
  (boost/drop sets); swap in centroid-shifted PE vectors in `reciprocal_rank_fusion`.

---

## 9. Module map

```
backend/app/
  config.py            env/secret loading, mock-mode flag, capability props
  identity.py          submit_keyframe_id parse/normalize, L26 shard rules
  media.py             R2 keyframe/video URL builders
  scoring.py           speech/audio demotion multipliers
  fusion.py            RRF + group-by-video + ambiguous detection
  trake.py             sequence assembly, order validation, keyframe snapping
  canvas.py            V-KIS canvas: palette, zones, PE text, Hungarian matching
  query_parser.py      Nemotron + heuristic routing, manual overrides
  types.py / models.py internal dataclasses / pydantic request models
  adapters/            elastic_client, milvus_client, pe_encoder, nvila_client,
                       object_elastic (OD frames), http_pool (+ mock paths)
  services/            search_service, trake_service, canvas_service,
                       timeline_service, submit_service
  main.py              FastAPI routes
frontend/src/
  api/                 client.ts, types.ts
  lib/                 media, identity, snap, qa, canvas, constants
  components/          TopBar, QueryPanel, ChannelControls, QueryUnderstanding,
                       Results, DetailPanel, VideoViewer, Timeline, TrakePanel,
                       CanvasPanel, SubmitGuard, HistorySidebar, Badges
  App.tsx              state, keyboard, orchestration
```
