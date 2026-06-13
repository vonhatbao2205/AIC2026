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

**Submit guard** shows thumbnail, ids, timestamp, OCR/ASR/audio snippets (and an
editable answer for QA), blocks duplicates, and shows the ordered id list for TRAKE.

**Mock mode**: if services are unreachable, `/api/health` surfaces a warning
banner and the UI can be driven entirely from backend fixtures. Live mode never
fabricates retrieval results.

---

## 7. Current limitations (by design)

- **Audio vector search** uses GLAP (`mispeech/GLAP`, 1024-d) over Milvus
  `aic26_audio_glap_v1`, FUSED (rank-RRF) with the Elastic tag/caption signal in
  the audio channel. It needs the `/encode-audio-text` endpoint on the Kaggle PE
  server (see `model-setup-backend.ipynb`); if that endpoint is down the audio
  channel falls back to Elastic-only automatically.
- **QA is locate + text-evidence only** — no image-based VLM CoT. An LLM draft,
  when enabled, is built strictly from OCR/speech/audio text and labeled as such.
- **No** Qwen3-VL reranker, structured VLM captions, object/scene/action tag
  filtering, SigLIP/EVA ensemble, or true clip-query V-KIS.
- Elastic uses built-in `text` analyzers (no custom Vietnamese tokenizer yet).

---

## 8. Extension points (clean seams left in place)

- **VLM reranker / CoT**: add a `RerankerClient` adapter and a post-fusion
  rerank step in `search_service.retrieve` (frames already carry evidence). Flip
  `capabilities.vlm_rerank` / `vlm_cot` in `/api/health`.
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
  query_parser.py      Nemotron + heuristic routing, manual overrides
  types.py / models.py internal dataclasses / pydantic request models
  adapters/            elastic_client, milvus_client, pe_encoder (+ mock paths)
  services/            search_service, trake_service, timeline_service, submit_service
  main.py              FastAPI routes
frontend/src/
  api/                 client.ts, types.ts
  lib/                 media, identity, snap, constants
  components/          TopBar, QueryPanel, ChannelControls, QueryUnderstanding,
                       Results, DetailPanel, VideoViewer, Timeline, TrakePanel,
                       SubmitGuard, HistorySidebar, Badges
  App.tsx              state, keyboard, orchestration
```
