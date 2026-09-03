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
| `QWEN3_VL_ENCODER_URL`, `QWEN3_VL_ENCODER_TOKEN` | optional, InfoShot++ only | Colab A100 Qwen3-VL-Embedding-8B `/encode-text` worker; both are required to enable it |
| `QWEN3_VL_ENCODER_TIMEOUT_SECONDS` | optional | text-encode timeout, default 120s (covers the worker's cold first inference) |
| `MEDIA_BASE_URL` | yes | Cloudflare R2 public base (keyframes/videos) |
| `NVIDIA_API_KEY` | optional | enables Nemotron query parser (else heuristics) |
| `NVIDIA_BASE_URL`, `NVIDIA_MODEL` | optional | default NIM endpoint + `nvidia/nemotron-3-ultra-550b-a55b` |
| `NVILA_BASE_URL`, `NVILA_TOKEN` | optional | Colab A100 NVILA-8B QA worker; both are required to enable it |
| `NVILA_TIMEOUT_SECONDS`, `NVILA_MAX_CANDIDATES` | optional | visual QA timeout and trusted candidate cap (defaults 240s/12) |
| `DEEPSEEK_API_KEY` | optional | DeepSeek built-in `web_search` grounding for QA; backend-only secret |
| `DEEPSEEK_GROUNDING_*` | optional | model/base URL/enable flag/timeout/max output tokens/reasoning effort/auto-confidence threshold |
| `DRES_BASE_URL` | yes for submit | official DRES host (default `http://if-wan4.selab.edu.vn:20740`; SELab: `http://10.0.1.21:20740`) |
| `DRES_USERNAME`, `DRES_PASSWORD` | yes for submit | participant account; backend logs in and keeps the `sessionId` |
| `DRES_SESSION`, `DRES_EVALUATION_ID` | optional | reuse an issued session / pin one run (else auto-routed by query type) |
| `DRES_SEGMENT_PAD_MS` | optional | ± ms around the picked instant for KIS/TRAKE temporal answers (default 500) |
| `VIDEO_MEDIA_BASE_URL_2` | optional | video origin for L21–L30 (default: the HF bucket) |
| `VIDEO_MEDIA_FALLBACK_BASE_URL_{1,2}` | optional | origin the console retries a failed video under; empty disables the retry |
| `IDX_*`, `MILVUS_IMAGE_COLLECTION` | optional | override index/collection names |
| `MILVUS_QWEN3_VL_IMAGE_COLLECTION_2` | optional | native 4096-d Qwen image collection (default `aic26_image_qwen3vl8b_infoshotpp_v3`); never point it at a PE collection |
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
| POST | `/api/query/parse` | `{query, query_type_hint, previous_hints[], manual_overrides, translate?}` → routing JSON |
| POST | `/api/search` | `{query, query_type_hint, previous_hints[], manual_overrides, parsed?, feedback?, image_models?, translate?, top_k, max_videos}` |
| POST | `/api/search/simple` | `{query, top_k, scope?, rerank?, image_models?, translate?}` → flat keyframe list, no parser and no group-by-video |
| POST | `/api/search/trake` | `{query, previous_hints[], manual_overrides, image_models?, translate?}` → `videos[]` (video-centric: per-event heat peaks + representatives + best chain) **and** `sequences[]` (the same assembly as flat chains) |
| POST | `/api/answers/generate` | `{query, query_type_hint, scope?, limit<=100, params?, answer_text?, event_count?, groups?/sequences?}` → the ordered answer list (§10). Pass `groups`/`sequences` to rank a result already on screen instead of searching again; `event_count` is the TRAKE row width taken from the statement |
| GET | `/api/canvas/palette` | V-KIS canvas vocabulary: 16 OD colours + canonical labels (+ `colorable`) |
| POST | `/api/search/canvas` | `{canvas{objects[{label,bbox,color,required}], action_text, mode}}` → same group shape, plus `object_layout` evidence |
| POST | `/api/qa/analyze` | `{question, candidates[{submit_keyframe_id,...}], max_answers}` → grounded answers + hotspots |
| GET | `/api/keyframes/{submit_keyframe_id:path}` | normalizes shard/padding; returns ids + media URLs + timing |
| GET | `/api/videos/{video_id}/timeline` | keyframes + speech + ocr + audio + heatmap |
| POST | `/api/videos/{video_id}/snap` | `{raw_time, fps?}` → nearest BTC keyframe (for TRAKE frame-pick) |
| GET | `/api/dres/status` | connection + logged-in user + configured pad (no credentials) |
| POST | `/api/dres/login` | force a fresh login after the session expired |
| GET | `/api/dres/evaluations` | every visible run + its open task + `taskStatus`/`timeLeft` |
| GET | `/api/dres/current-task` | `?evaluation_id=&query_type=` → open task of one run |
| GET | `/api/dres/task-hint` | the task statement: `{text, elements[{content_type,content,offset}], task_name, task_status}` |
| POST | `/api/submit/preview` | same body as submit → the exact DRES `answerSets` + duplicate key, sends nothing |
| POST | `/api/submit` | `{evaluation_id?, task_name?, query_type, payload{video_id, frame_idx?, timestamp?, fps?, start_ms?, end_ms?, answer?, events?[]}, answer_mode?, segment_pad_ms?, allow_duplicate}` |
| GET | `/api/submit/history` | `?task_id=` optional (task id = `{evaluationId}/{taskName}`) |
| DELETE | `/api/submit/history` | deletes log entries: `?ids=a,b`, `?task_id=`, or all; writes a `.bak.json` first |

### Image embedding models (`image_models`)

Two independent visual indices answer the **InfoShot++** keyframes:

| id | channel | encoder | collection | dim |
|---|---|---|---|---:|
| `pe` | `image_pe` | PE-Core-G14-448 `/encode-text` | `aic26_image_peg14_infoshotpp_v1` | 1280 |
| `qwen3_vl` | `image_qwen` | Qwen3-VL-Embedding-8B `/encode-text` | `aic26_image_qwen3vl8b_infoshotpp_v3` | 4096 |

- `image_models` defaults to `["pe"]`. `qwen3_vl` is **rejected with 422** on the
  BTC profile: those keyframes were never encoded with Qwen, so answering from PE
  while the UI shows a ticked Qwen box would be a lie about what was searched.
- Selecting both runs both as separate ranked channels that meet in RRF. Their
  cosine values are **never added** — a 1280-d PE cosine and a 4096-d Qwen cosine
  come from unrelated score distributions, so only ranks are comparable. The same
  rule holds inside `/api/search/simple`, the `similar` feedback channel and the
  TRAKE pass-2 fill.
- Query expansion variants are max-fused *within* one model, where cosine is
  comparable, before that model becomes a channel.
- **`channels.image_pe.enabled` is the master switch for BOTH indices** — the
  Qwen runner reads that same parser block, so turning the visual channel off
  disables Qwen too. The console labels that switch `VISUAL` (not `PE Core`) for
  exactly this reason; `CHANNEL_LABEL` still says "PE Core" for the *evidence*
  badge on a frame, which names the index that found it.
- One dead worker degrades to the other index with a warning; it never empties
  the result list. The Qwen client refuses to fabricate a vector in live mode.
- Serve the worker with
  `Qwen3VL-Embedding-8B/Qwen3_VL_Embedding_8B_Text_Encoder_Server_Colab_A100.ipynb`;
  the checkbox is hidden on BTC and the health flag is
  `capabilities.qwen3_vl_embedding_search`.

### Search result shape

```jsonc
{
  "query": "...",
  "image_models": ["pe"],                  // which visual indices answered
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
   - `image_qwen` (InfoShot++, opt-in via `image_models`): Qwen3-VL-Embedding-8B
     `/encode-text` → Milvus `aic26_image_qwen3vl8b_infoshotpp_v3` (COSINE, 4096-d).
     It reads the same `image_pe` routing block from the parser, so the operator
     picks the index without the parser having to know about it.
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

TRAKE runs the pipeline per event (each event's visual query translated VI→EN).
The unit of a TRAKE answer is the **video**, so the flat per-event results are
folded into `video_id -> per-event candidates` (`build_video_event_map`)
immediately after pass 1; that map, not a list of chains, is what pass-2
targeting, the DP, the heatmap and the ranking all read.

Per (video, event) the candidates go through a **temporal NMS**
(`temporal_diversify`): up to 8 peaks at least 2 s apart, then the best
near-duplicates while budget is left. Twelve frames of one two-second burst are
one hypothesis, and spending the cap on them dropped the genuinely different
moment later in the video that the chain needed. Each surviving video is then
assembled with the same **exact DP** — a maximum-weight strictly-increasing chain
over (event index, `pts_time`) that picks ≤1 frame per event, maximizing (events
REAL evidence, then coverage, then quality) — term for term the priority
`trake_video_score` encodes. It is globally optimal (unlike greedy) and can skip
an event for a partial sequence.

The ORDER is load-bearing, not just the terms. `(coverage, real, ...)` looks
equivalent and is not: with E1/E4/E5 found for real and E2/E3 only fillable after
E4, it prefers `E1 + fill + fill + E5` (coverage 4, real 2) over `E1 + E4 + E5`
(coverage 3, real 3), and the ranking then scores that video on
`confident_coverage = 2` — marking it down for the trade the DP made on its
behalf. Real evidence dominating fill coverage is the policy across videos, so it
has to be the policy inside one too. The known cost: when a fill would DISPLACE
real evidence, the shorter real chain wins and the video may contribute no
submittable row at all (a TRAKE row is only valid at full width). That is
deliberate — the longer chain would have been buried by the ranking anyway.

Real evidence is also its own term because the scales genuinely overlap: a pass-2
fill scores `0.02 x quality` while a frame found by one channel at rank 0 scores
`1/(60+1) = 0.0164`. The quality term is the per-event normalized `strength`, not
the raw score, so an event whose scores run high cannot decide the chain alone.

`trake_video_score` encodes the lexicographic ranking
`(confident_coverage, coverage, chain quality)` as one number in [0, 1]: with
`B = n+1`, `(C·B² + T·B + Q) / (n·B² + n·B + 1)` where
`Q = 0.70·mean + 0.30·min` of the per-event normalized strengths. One more
confidently covered event therefore always beats any cosine difference, and
`Q_min` carries its own weight so three superb events plus one hopeless one
cannot outrank four solid ones. `Q` covers every link of the chain, fills
included — unlike the sequence's `mean_score`/`min_score`, which are deliberately
about real evidence only. `confident_coverage` already dominates the ranking, so
excluding fills from `Q` too threw away the one thing separating two videos with
identical coverage: a chain closed by a convincing fill versus one closed by a
fill that barely cleared the floor. Heat `strength` is normalized **per event**
against `G_e` (that event's best score anywhere in the pool), because each event
is a different query with its own difficulty; a pass-2 fill reports its own
`fill_quality` instead, since the 0.02 fill scale means nothing against RRF.

Pass 2 is planned by `select_pass2_gaps`, and it reads the preliminary **DP**,
not the candidate lists. A gap is an event the chain could not place, not an
event with no candidate: a video whose E1 only fires at 90 s and whose E2 only
fires at 10 s has a candidate for every event and still answers nothing. Asking
"does this event have a candidate?" made pass 2 blind to that case — the commoner
one, since retrieval usually finds the right action in the wrong part of the
video. Videos are then tiered by how many events the chain is still missing
(one short is a single query from a full chain) and, inside a tier, by the
preliminary `trake_video_score`.

Pass 2 hands the assembly DP up to three hypotheses per gap rather than choosing
one. Retrieval had already paid for top-3, and the DP is the only thing that can
see the gaps together: E2's best at 80 s plus E3's best at 20 s is a dead chain,
while E2@30 + E3@70 completes it. The temporal window
`[min(earlier candidates), max(later candidates)]` is a NECESSARY condition only
— with E1@90 and E2@100, a candidate at 95 clears it and still cannot follow E2 —
so it orders the hypotheses and never picks among them. Across two image models, RRF decides the fill's *order*
while `fill_quality` stays the model-local `score / that model's best hit`;
normalising RRF by the gap's own top reported every winning candidate as 1.0,
including one that had just scraped past the 45% floor.

See `backend/app/trake.py` (`build_trake_videos`, `assemble_trake_sequences` for
the flat chain view the answer generator consumes).

---

## 6. Frontend (operator console)

Single screen, three columns: **query/channels/understanding** · **grouped
results + video + multi-track timeline** · **detail/evidence + TRAKE + history**.

Keyboard: `/` focus query · `Enter` search (in query) / open guard (results) /
confirm (modal) / assign chip (TRAKE) · `↑/↓` video group · `←/→` frame ·
`Space` play-pause the candidate video · `Tab` switch zone · `T` toggle timeline ·
`Esc` cancel modal/chip · `Ctrl+M` voice (if browser supports it).

Inside a video group the frames read **chronologically by default**; `↺` puts
one group back into relevance order. Reading a scene is what a group is for, and
a strip out of scene order is harder to judge than one out of rank order. What
that costs is the ranking, so the group's three strongest frames carry a red
`#1`/`#2`/`#3` mark in place (`topRelevanceRanks`), and the accent selection
border still wins over them — it says what the submit guard will act on.

**TRAKE results are video cards**, not the generic `VideoGroup` list
(`components/TrakeVideoResults.tsx` + `TrakeHeatmap.tsx`): the chain across the
top — one thumbnail per event, the DP's pick or the strongest candidate badged
`⚠ ngoài chuỗi` when no orderable position exists — over one heat row per event
built from the sparse peaks the backend sends. The heat rows draw **the frames
themselves**, parked at their own timestamps: judging an alternative used to cost
a video load, a seek and a pause each, which is why nobody looked at them.

Every frame there is draggable (`lib/trakeDrag.ts`, `text/x-trake-peak`) onto an
event slot in the sidebar — it is submitted as-is, no video needed — or onto an
event card, where it replaces the DP's pick (`↺` restores it; the backend result
is never mutated). A frame from another video is refused: every event of a TRAKE
row has to come from the one video being submitted. A hand-made chain that is no
longer chronological is called out on the card, because the organiser's parser
rejects such a row and one rejected row blocks the whole submission. The DP only knows a chain is
orderable, not whether it is plausible; the heat rows are where a human sees
that E1 sits minutes away from a tight E2-E3-E4 cluster. Clicking a
representative or a peak selects the video, seeks the player to that instant and
arms that event's slot, so checking an alternative costs a click instead of
another search. It deliberately does NOT open the video — the card already shows
the frame, and a fetch plus a layout jump per glance is not what a click should
cost; `v` opens the player, parked on the moment that was picked. The verified peak becomes the selected frame, so the detail
panel, the timeline marker and the neighbour anchor describe the moment on
screen rather than the chain frame the player left behind. Heat rows are drawn
against the video's real `duration_s` (one keyframe-map aggregation for the whole
result page), and the peaks carry a fixed, generous hit area that does not shrink
with their score — they are click targets on a laptop under a clock, not a chart.
Crowded candidates stack into a second lane (`lib/trakeHeatLayout.ts`) rather
than covering each other: three hits eleven seconds apart in a ten-minute video
land on the same pixels, and only the last one painted can be hit while the two
underneath — the ones worth comparing — are unreachable. Each candidate keeps a
tick at its true timestamp, so a frame nudged clear of a cluster never misreports
when the event fires.
TRAKE shows no prioritise/deprioritise controls and no feedback bar:
`/api/search/trake` takes no feedback, so those buttons re-ran the search and
changed nothing while telling the operator they had.

**Video origins.** Keyframes and videos are served from separate hosts because a
thumbnail grid and a video stream do not want the same thing. For L21–L30 the
video origin is a named Cloudflare tunnel: measured against the HF bucket over 10
videos, median time-to-first-byte was **63 ms vs 834 ms** on the opening range and
**59 ms vs 831 ms** on a mid-file seek, with 13.1 MB/s vs 1.2 MB/s throughput. The
bucket answers every range request with a 302 to its CDN and pays that round trip
again on each seek, which a scrubbing player does constantly.

Speed and availability are not the same property, though: a tunnel terminates on
a machine that can be switched off, while the bucket is always-on hosting. So the
bucket stays configured as `VIDEO_MEDIA_FALLBACK_BASE_URL_2` rather than being
replaced. `/api/health` returns both origins for the active profile; when a video
fails to load, `VideoViewer` retries the same path under the other origin, once,
carrying the playhead across — and the console then starts subsequent videos on
the fallback instead of spending a failed request per clip rediscovering that the
primary is still down. Configuring both to the same origin is refused
(`swapVideoOrigin`): that is a single point of failure wearing a fallback's
clothes.

Opening the player (`v`, or the neighbour strip with `k`) scrolls it into view:
it is rendered inside the result card it belongs to, so a group holding hundreds
of keyframes opened it well below the viewport.

A TRAKE row names ONE video and lists a frame index per event under it, so a
frame from another video would be submitted as though it came from the named one
— well-formed and wrong, invisible to everything downstream. That is refused when
the frame is assigned to a slot, and refused again (independently) by the submit
guard, which also blocks a chain that is not chronological rather than only
warning about it.

**Frame-pick**: `requestVideoFrameCallback` tracks the latest `mediaTime`. The
inline video opens **paused on the frame it was opened to show** and every seek
that lands while it is paused captures that frame — capture used to be tied to
the `pause` event alone, so picking a frame meant playing the clip and pausing
it again, and what came back was wherever playback had reached, not the frame
chosen. Seeking mid-playback captures nothing; that is navigation. `Space`
starts playback. The raw frame is snapped to the nearest BTC keyframe by
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
- **Official DRES**: DONE — `adapters/dres_client.py` speaks Client API v2
  (login → session, `client/evaluation/list`, `currentTask`, `POST /api/v2/submit/{id}`)
  and `submit_service` builds the `answerSets` body; see §11.
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
  answer_gen.py        the ordered 100-answer list: temporal NMS -> anchors,
                       eps ladder -> coverage probes, pi x E x N rank budget
  trake.py             sequence assembly, order validation, keyframe snapping
  canvas.py            V-KIS canvas: palette, zones, PE text, Hungarian matching
  query_parser.py      Nemotron + heuristic routing, manual overrides
  types.py / models.py internal dataclasses / pydantic request models
  adapters/            elastic_client, milvus_client, pe_encoder,
                       qwen3_vl_encoder (4096-d text queries), nvila_client,
                       object_elastic (OD frames), dres_client (DRES v2 session),
                       http_pool (+ mock paths)
  services/            search_service, trake_service, canvas_service,
                       timeline_service, submit_service, answer_service
  main.py              FastAPI routes
frontend/src/
  api/                 client.ts, types.ts
  lib/                 media, identity, snap, qa, canvas, dres, constants,
                       imageModels (PE/Qwen selection rules),
                       submission (CSV pack), answerGen (bulk generation)
  components/          TopBar, DresBar, QueryPanel, ChannelControls, ImageModelSelector,
                       QueryUnderstanding,
                       Results, DetailPanel, VideoViewer, Timeline, TrakePanel,
                       CanvasPanel, SubmitGuard, HistorySidebar, Badges
  App.tsx              state, keyboard, orchestration
```

---

## 10. Answer generation (the ordered 100-answer list)

The preliminary round scores `Final = (R@1 + R@5 + R@20 + R@50 + R@100) / 5` over
at most 100 answers per query, so the answer list is a **rank-budget** problem,
not a top-100 dump: position `r` goes to whichever candidate adds the most NEW
chance of a hit, `U_r(V,c) = pi_B(V) * E(c|V) * N(c|A_V)`.

- `pi_B(V)` — softmax over the group score at the current cutoff's temperature.
  `T` rises 1 -> 5 -> 20 -> 50 -> 100, which turns exploitation into hedging
  without any hard per-video quota.
- `E(c|V)` — temporal evidence WITHIN the video. Anchors (what survives temporal
  NMS over the fused frames) keep their retrieval score; offsets `a +/- eps`
  inherit it, decayed per ladder rung.
- `N(c|A_V)` — novelty, a Gaussian penalty per already-picked answer of the same
  video. A covered region loses value, so a strong video yields once it has
  nothing new to say.

Two details are load-bearing and were measured, not assumed:

1. **Offsets snap onto real keyframes** (`snap_offsets`). On the dev set the
   median `|nearest retrieved frame - ground truth|` is **0** — most ground-truth
   frames ARE extracted keyframes. Snapping is worth **+0.052 Final**, the largest
   single effect in the parameter grid.
2. **No position is ever left empty.** `R@k` is a max, so a wrong answer at rank
   90 costs nothing while an empty rank 90 forfeits its chance. When a band's
   frontier is exhausted the gate is dropped for that position, and when the whole
   candidate space is too thin the eps ladder densifies (halving gaps, never
   marching past the end of the video) until 100 can be filled.

For TRAKE the candidate is a whole chain, and a row is only a row when it is
**complete**: `<video>,<f1>..<fN>` with N fixed by the statement, strictly
increasing. The assembler emits partial chains deliberately and ranks by
confident coverage, so a 3-of-4 chain outranks a complete one — on the dev
queries 45 of 50 chains were partial and up to 55 of 100 generated rows had the
wrong width, plus 13 out of chronological order from shifting one event past its
neighbour. Every one of those is rejected by the organiser's parser, and one
rejected row blocks the whole submission. N therefore comes from the statement
(`event_count` on the request), never inferred from the chains; when nothing is
complete the answer list is empty and says why.

Answers already at the head of a question's list are passed in as `taken`. The
list is graded as a whole, so a position already spent on an instant is coverage:
the generator does not repeat it, does not spend more positions covering it, and
**continues the band schedule after it**. The first answer following six
hand-picked ones is rank 7 — already in the diversifying band — not rank 1, where
the policy is to bet everything on the single strongest hypothesis. Generating a
tail this way is lossless: it reproduces exactly what a full run would have put at
those positions (0.5905 at every head size, against 0.5848 at head=6 before).

`taken_video_penalty` discounts a video by how many positions it already holds.
It is shipped at **0**, and that is a measured result, not an oversight: on the
dev set with a simulated hand-picked head it is worth +0.002 at head=1 (below the
0.0095 one-query quantum) and is monotonically worse at head=3 and head=6. The
idea that a video a person already bet on is a poor bet for the next position did
not survive contact with the data. The knob stays for whoever measures it next.

`ambiguous` never demotes a video; it says the uncertainty is temporal, so that
video opens more anchors and fewer offsets. **The fitted default turns this off**
(bonus 0 / penalty 0): on 21 dev queries it has no measurable effect either way
(0.004-0.006, smaller than one query moving one band). The mechanism is kept and
stays settable per request.

Parameters live in `AnswerGenParams` and are fitted by
`benchmarks/run_answer_gen.py` against the organisers' own scoring formula, on
the L21-L30 ground-truth queries (InfoShot++ profile). The objective averages
Final over tolerances {0, 12, 25, 50, 100} frames because the width of the
accepted window `[s, e]` is not knowable in advance — fitting at one tolerance
picks a strategy for that assumption alone. Honest generalisation from 3-fold x 2
cross-validation: **0.515 tuned vs 0.484 untuned** on held-out queries.

---

## 11. DRES submission (official server, Client API v2)

Reference: `instruction.md` and `http://if-wan4.selab.edu.vn:20740/clientapi.json`
(spec 2.0.5). Inside SELab (I87) replace the host with `10.0.1.21`.

**Session.** `POST /api/v2/login {username,password}` → `sessionId`; every other
call carries it as `?session=…`. `dres_client.py` caches it, and a 401/403
triggers exactly one re-login + retry, so an expired session never costs a
submit. Credentials live only in `backend/.env`.

**Which run, which task.** The BTC opens one evaluation run per task type
(`tkis`, `qa`, …), each with its own open task. The backend picks the run from
the query type (`resolve_evaluation`, mirrored in `frontend/src/lib/dres.ts` for
display) and reads the exact task name from
`GET /api/v2/client/evaluation/currentTask/{evaluationId}` at submit time. The
operator can pin a run in the DRES bar or type a task name in the guard.

**Body.** `POST /api/v2/submit/{evaluationId}?session=…`

```jsonc
{"answerSets": [{"taskName": "tkis-00", "answers": [
  {"mediaItemName": "L30_V095", "start": 373900, "end": 374900}   // KIS/TRAKE, ms
]}]}
```

- KIS (T-KIS / V-KIS): one answer, `mediaItemName` + `start`/`end` **in ms**.
  `mediaItemName` never carries `.mp4` or a directory.
- TRAKE: one temporal answer **per event**, in event order, in the same answer set.
- QA: ONE answer carrying all four — `{mediaItemName, start, end, text}` — the
  form the BTC payload example documents (identify the segment *and* answer).
  `answer_mode` overrides it to `text` / `temporal` / `item` per submit.
- The shape resets to `auto` on every query-type and task change, and `prepare()`
  compares it with the DRES `taskType`: a QA task built as a bare segment (the
  answer text silently dropped) raises a red warning in the guard.
- The window is the picked instant ± `DRES_SEGMENT_PAD_MS` (default 500 ms): a
  zero-length range is rejected by strict overlap checks, and ±0.5 s stays well
  inside a KIS target segment. Override per submit in the guard, or send explicit
  `start_ms`/`end_ms`.

**Đề bài (task statement).** `GET /api/v2/evaluation/{evaluationId}/template/task/{taskTemplateId}/hint`
is **not** in `clientapi.json` — it lives in the full `/openapi.json` — but a
PARTICIPANT session may read it, and it returns exactly what the DRES viewer
shows the team: the TEXT query for T-KIS/QA, image/video hints for V-KIS. The
backend resolves `taskTemplateId` from the run state, normalises the sequence
(TEXT joined by `offset`, media over 12 MB dropped with a warning) and caches it
per task template. The console shows it above the query box and prefills the
query with it — never over words the operator already typed.

**Guard.** `POST /api/submit/preview` returns the exact body, the resolved
run/task and the duplicate key without sending anything; the submit guard renders
it, so the operator confirms the real JSON. A format error (QA with no text, a
frame with no resolvable time) blocks the submit locally instead of spending a
wrong submit on it.

**Dedup & history.** Scope is `{evaluationId}/{taskName}`. KIS dedups on
`video_id:frame_idx`, TRAKE on the ordered frame sequence, QA on
`video_id:frame_idx|text:<answer>` — a new answer for the same segment, or the
same answer on another segment, is a genuinely different guess. Every
attempt is stored in `backend/data/submit_history.json` with the DRES verdict
(`CORRECT` / `WRONG` / `INDETERMINATE` / `UNDECIDABLE`). The console can delete
individual entries (checkbox multi-select or a per-row ✕), one task's, or all of
it behind a two-step confirm; whatever is removed is backed up beside the log,
and deleting also erases that entry's dedup memory.

**Mock mode never talks to DRES** (`Settings.has_dres` is false when
`AIC26_MOCK_MODE=true`), so the test suite cannot submit to the live server.
