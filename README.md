# AIC26 Retrieval

An operator-first **video retrieval system** built for the AI Challenge 2026
(AIC) — interactive Known-Item Search (KIS), question answering (QA), and
temporal action search (TRAKE) over a large broadcast-video corpus.

The repository contains two parts:

- **A multi-channel retrieval app** (`backend/` + `frontend/`) — the live console
  an operator uses during the competition to search, inspect evidence, and submit
  answers to DRES.
- **An offline indexing pipeline** (root scripts + notebooks) — the tooling that
  turns raw videos/keyframes into the OCR / speech / audio / image-vector indices
  the app reads from.

> Design principle: **UI > Method > Model** — minimise wrong submits and
> navigation time for a human operator under a clock.

---

## Highlights

- **Multi-channel retrieval** fused with Reciprocal Rank Fusion (RRF, `k=60`):
  - `image_pe` — PE-Core-G14 text→image vectors in **Milvus** (cosine)
  - `image_qwen` — Qwen3-VL-Embedding-8B native 4096-d vectors for InfoShot++;
    the operator may search PE, Qwen, or both in parallel. A two-model search
    combines ranks with RRF and never adds their incomparable cosine scores.
  - `ocr` / `speech` / `audio` — full-text + filters in **Elastic**
  - `audio` vector — **GLAP** audio↔text embeddings in Milvus, fused with the Elastic audio signal
- **Query understanding** via DeepSeek V4.1 Flash (`deepseek-flash`, thinking off,
  JSON output, ~2 s) with a prompt written for the final-round collection and task
  types, validated field by field, and a deterministic heuristic fallback; VI→EN
  translation of the visual query. The same model writes the **Expand** paraphrases.
- **Qwen3-VL visual reranker** (opt-in, off by default) — a **Rerank** tick box in the
  query panel widens EVERY selected image index to `QWEN_RERANKER_CANDIDATES`, merges
  their candidates into one pool (round-robin over the two rank lists, so a frame only
  Qwen retrieved is judged too), and has Qwen3-VL-Reranker-8B rescore each
  `(query, keyframe)` pair at frame level. The result is a single `image_visual` ranking
  that meets OCR/speech/audio in the RRF, so a low visual score cannot overrule the
  channels whose evidence is not visual at all. Fail-open: if the worker is unreachable
  the union order (RRF over the indices, never a sum of their cosines) is kept and the
  panel says so. Serve it with `aic26_qwen3vl_reranker8b_colab_server.ipynb`; the box is
  hidden when no worker answers.
- **Search scope** — a checkbox filter over the dataset folders (BTC: L21–L30 +
  K01–K20; InfoShot++: L21–L30 + batch 2 M01–M10 news, N001–N100 traffic
  cameras, S01 cycling), pushed down into the Milvus/Elastic queries. Each folder is one
  programme, so a topic heuristic reads the folders off the query the same way
  the parser reads channels ("đầu bếp" → L26), while never excluding the
  programmes that carry every subject (the 60-second bulletins and L30).
- **PE text-window meter** — PE-Core reads 70 text tokens and silently drops the
  rest; about a third of the organisers' queries are longer than that even in
  English. The search box counts the query (plus hints) as the operator types —
  exactly with the PE tokenizer, or estimated when it will be translated — and
  says which part PE will ignore; each search reports the exact count of what
  it sent.
- **Traffic-camera / race filter** (InfoShot++) — every N camera prints its
  junction, date and clock in a banner that batch-2 OCR reads. Street names in the
  query ("Nguyễn Trãi – Cống Quỳnh", NTMK, CMT8), a date ("15/6") or a time
  ("19:05", "7 giờ tối") narrow the traffic cameras to the matching keyframes, and
  "chặng 6" narrows S01 to that stage's video. It is pushed into Milvus/Elastic like
  the scope, never excludes another folder, and drops a cue no recording matches
  with a warning; the console shows what it applied and can switch it off. Camera
  frames carry their banner reading, and OCR ranks the news ticker, banner and HUD
  below scene text.
- **TRAKE** is ranked by **video**, not by keyframe: after pass 1 the per-event
  results become one `video_id -> per-event candidates` map that drives pass-2
  targeting, the DP, the heatmap and the ranking. A temporal NMS keeps distinct
  moments (not twelve frames of one burst) before an exact dynamic-programming
  maximum-weight increasing chain assembles the sequence — globally optimal, not
  greedy. `trake_video_score` folds `(confident coverage, coverage, chain
  quality)` into one coverage-dominant number, and each result card shows one
  representative per event over a heat row of every moment that event fires;
  clicking one seeks the player there and arms that event's slot.
- **QA copilot** — top-video-first candidate blocks read by **DeepSeek V4.1 Flash**
  itself (image input, thinking on): answer-bearing frames and exact-format answer
  alternatives (as printed, bare names, the question's own format rules), then
  DeepSeek `deepseek-v4-pro` web search for names and facts the frames cannot show,
  then a Flash visual-consistency pass before explicit human verification.
  ~15–20 s for all three passes; the NVILA-8B Colab worker remains selectable
  (`QA_VISION_BACKEND=nvila`).
- **100-answer generator** — the preliminary round scores
  `Final = (R@1 + R@5 + R@20 + R@50 + R@100) / 5` over up to 100 answers per query,
  so the answer list is a rank-budget problem, not a top-100 dump. Each position
  goes to whichever candidate adds the most new chance of a hit
  (`U = π_B(video) × evidence × novelty`), with temporal NMS turning the fused
  frames into distinct hypotheses and a data-calibrated `±ε` ladder covering the
  case where the retrieved keyframe sits just outside the accepted window. One
  button in the Submission tab fills the whole question pack.
- **Agent sidecar search (AGENT button, on by default)** — each Search press also
  starts **Codex CLI** (`gpt-6-sol`, high) and **Claude Code CLI**
  (`claude-opus-5-5`, high) on the backend host, in the background. They search
  the corpus through a narrow MCP tool bridge (`backend/app/agent/`):
  - `search`: the same retrieval engine, in hybrid, visual, OCR or speech mode.
  - `list_videos`: the programme guide of a folder or series. Traffic cameras show
    junction, date and clock range; S01 shows the race stage; every other folder
    shows sample lines of what is said.
  - `folder_frames`: one frame per video in one grid (`N` gives one tile per
    camera).
  - `video_outline`: a video's table of contents, one line per minute of what is
    said and shown.

    These three browsing tools let an agent find a video without the search
    engine.
  - `video_frames`: labelled contact sheets of keyframes from R2.
  - `view_frames`: a large view of specific keyframes.
  - `video_text`: the speech and OCR text in a time window.
  - `report_candidate`: shows a candidate to the operator.

  Candidates appear in their own **Agents** panel as soon as they are reported,
  and the panel flags a moment both agents found on their own. The main
  `/api/search` never waits for the agents and its ranking is never changed by
  them. A new search cancels the tab's previous run, and every agent is killed at
  `AGENT_TIMEOUT_SECONDS`. The CLIs have no shell and receive none of the
  backend's credentials. Each run is a fresh process, so an account switched in
  the meantime is picked up by the next search.
- **Submit guard** — duplicate detection, evidence preview, and the exact DRES v2
  `answerSets` body previewed before it is sent (media item + ms window for KIS/TRAKE,
  text for QA); the open task name and evaluation run come live from DRES.
- **Keyboard-first single-screen UI** with a multi-track timeline (keyframes,
  speech, OCR, audio, heatmap) and BTC keyframe snapping for frame-accurate picks.
- **Mock mode** — deterministic fixtures so the UI and tests run with zero live services.

---

## Architecture

```
OFFLINE INDEXING (scripts + notebooks at repo root)
  Videos / BTC keyframes
        │  Whisper ASR · OCR · audio tagging · PE-G14 / GLAP embeddings
        ▼
  ┌───────────────┬───────────────┬───────────────┐
  │ Elastic Cloud │    Milvus     │ Cloudflare R2 │
  │ ocr / speech  │ image vectors │ keyframes +   │
  │ audio / kfmap │ audio vectors │ videos (media)│
  └───────┬───────┴───────┬───────┴───────┬───────┘
          │               │               │
ONLINE QUERY              │               │
          ▼               ▼               ▼
  backend/  FastAPI ── parse → retrieve (parallel channels) → RRF fuse
            → group-by-video / TRAKE DP → DeepSeek visual QA → DeepSeek web grounding
            → DeepSeek visual verification → human choice → submit guard → DRES
          │
          ▼
  frontend/ Vite + React + TS operator console (keyboard-first, timeline)
```

The detailed, load-bearing contract (identity rules, API shape, scoring,
fusion, TRAKE) lives in **[RETRIEVAL_APP_HANDOFF.md](RETRIEVAL_APP_HANDOFF.md)**.
The broader design/strategy is in **[AIC26_Pipeline.md](AIC26_Pipeline.md)**.

---

## Repository layout

| Path | What it is |
|---|---|
| `backend/` | FastAPI service — adapters (Elastic/Milvus/PE/GLAP), query parser, fusion, TRAKE, answer generation, submit. See `backend/app/`. |
| `backend/app/answer_gen.py` | The ordered 100-answer generator (rank-budget allocator). Parameters fitted by `benchmarks/run_answer_gen.py`. |
| `benchmarks/` | Ground-truth evaluation: retrieval benchmarks, answer-generator scoring and parameter fitting. See `benchmarks/README.md`. |
| `frontend/` | Vite + React + TypeScript operator console. See `frontend/src/`. |
| `Dockerfile`, `docker-compose.yml` | The packaged single-container app (console + API on one port). |
| `scripts/package.sh`, `scripts/dist/` | Builds the downloadable bundle and the files shipped inside it. |
| `keyframe_mapping.py` | Build the keyframe → (video, pts_time) map. |
| `elastic_upload.py` | Index OCR / speech / audio / keyframe-map records into Elastic. |
| `elastic_upload_batch2_ocr_speech.py` | Append batch-2 OCR (M/N/S01) and speech (M/S01) to the InfoShot++ `*_v2` indices, pinned to the audited sources. |
| `build_traffic_camera_catalog.py` | Build `backend/app/traffic_cameras.json` (junction, date, per-minute keyframe windows of the 298 N videos) from the batch-2 OCR banner fields. |
| `build_batch2_video_guide.py` | Build `backend/app/batch2_video_guide.json` (what each S01 race recording shows) from the batch-2 content sheet in `Batch2/batch2_content/`, for the agents' browsing tools. |
| `milvus_upload.py` | Upload PE-Core-G14 image vectors into Milvus. |
| `milvus_upload_pe_core.py` | Audit + upload the InfoShot++ PE-Core-G14-448 Parquet dataset (1280-d) into `aic26_image_peg14_infoshotpp_v1`. |
| `milvus_upload_qwen3_vl_embedding_8b.py` | Audit + upload the InfoShot++ Qwen3-VL-Embedding-8B dataset (native 4096-d, 1.339.055 keyframe, 658 shard) into `aic26_image_qwen3vl8b_infoshotpp_v3`. Shard-atomic resume; joins `frame_id` against the final map CSVs and never treats `frame_idx` as the keyframe ordinal. |
| `aic26_query_factory_qwen3vl8b_t4x2.ipynb` | Kaggle T4×2 workflow dùng Qwen3-VL-8B, PE-G14/Milvus visual hard negatives, critic, review và export benchmark T-KIS/QA/TRAKE/V-KIS. |
| `aic26_nvila8b_qa_colab_server.ipynb` | Colab A100 worker chạy NVILA-8B BF16: QA hotspot prediction → grounded answer suggestions qua FastAPI/Cloudflare tunnel. |
| `aic26_qwen3vl_reranker8b_colab_server.ipynb` | Colab A100 worker chạy Qwen3-VL-Reranker-8B BF16: rerank cặp (query, keyframe) ở frame level trước RRF, qua FastAPI/Cloudflare tunnel. |
| `Qwen3VL-Embedding-8B/Qwen3_VL_Embedding_8B_Text_Encoder_Server_Colab_A100.ipynb` | Colab A100 (40/80 GB) worker chạy Qwen3-VL-Embedding-8B BF16 + FA2: encode text query thành vector 4096-d FP32 unit-norm cho kênh `image_qwen`, qua FastAPI/Cloudflare tunnel. |
| `Qwen3VL-Embedding-8B/Qwen3_VL_Embedding_8B_AIC2026_HANDOFF.md` | Semantic contract, provenance và audit của collection embedding ảnh `qwen3-vl-embedding-8b-4096-v3`. |
| `*.ipynb` | Pipeline notebooks: `audio_pipeline`, `speech_pipeline`, `cloudflareR2`, `model-setup-backend`, `glap-encoder-kaggle`, `ui-streamlit`. |
| `*_PIPELINE.md`, `*_HANDOFF.md`, `FEATURES.md` | Per-stage documentation (audio, speech, OCR, Elastic, Milvus, R2). |
| `instruction.md` | Official DRES v2 submission protocol (BTC). |
| `DRES/HUONG_DAN_SU_DUNG_DRES.md`, `DRES/dres_readme.md` | Local DRES simulator notes (practice only). |

**Not tracked in git** (see `.gitignore`): secrets (`*.txt` keys, `.env`),
large data/model artifacts (`peG14.pkl`, `*.jsonl`, `speech_out/`,
`elastic_staging/`, `results_audio_event/`, keyframe maps), virtualenvs, and
build caches. These are regenerated by the pipeline scripts/notebooks above.

---

## Quickstart

### 0. Just run it (Docker — no Python, no Node, no build)

```bash
docker compose up -d          # http://localhost:8000
```

One container serves the console and the API on one port. It ships with **no
credentials**: open it, and the settings screen (⚙) asks you to import the team's
`.env`. The imported file lands in `./config/.env` and the submit history in
`./data/`, both outside the image, so a rebuild never loses them.

To hand the app to someone who has neither this repo nor a toolchain:

```bash
./scripts/package.sh          # → dist-app.zip (~470 MB, image included)
```

They unzip it and run `./run.sh` (Linux/macOS) or double-click `run.bat`
(Windows, Docker Desktop) — it loads the image, starts the container and opens
the browser. Setup guide for them: [scripts/dist/HUONG_DAN.md](scripts/dist/HUONG_DAN.md).

> The compose file binds to `127.0.0.1` on purpose: the running app can rewrite
> its own credentials and submit under your DRES account, so it is not exposed to
> the LAN unless you change that line deliberately.

### Prerequisites (development from source)
- Python 3.10+ and Node.js 18+
- (Optional for live mode) Elastic Cloud, Milvus/Zilliz, PE-Core-G14 and/or
  Qwen3-VL-Embedding-8B encoder endpoints, and a Cloudflare R2 media base URL.

### 1. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # or reuse an existing venv
pip install -r requirements.txt

# Dev / UI work without any live services (deterministic fixtures):
AIC26_MOCK_MODE=true PYTHONPATH=. python -m uvicorn app.main:app --reload --port 8000

# Live mode (reads ELASTIC_* / MILVUS_* / PE_ENCODER_URL / MEDIA_BASE_URL from .env):
PYTHONPATH=. python -m uvicorn app.main:app --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api -> http://localhost:8000
# point elsewhere:   VITE_API_TARGET=http://host:8000 npm run dev
```

Open http://localhost:5173. In mock mode the whole console is usable immediately.

---

## Configuration & secrets

Backend config is read from environment variables. **Copy the template and fill
in real values:**

```bash
cp backend/.env.example backend/.env
```

In the packaged app there is no shell to export variables in, so the same file is
imported through the UI instead: **⚙ → drag in a `.env` → Import**. The backend
writes it to `AIC26_CONFIG_DIR/.env` (mode `600`) and rebuilds every service in
place — no restart. Variables set on the container itself still win over the
file, and the settings screen tags those `env` so it is clear an import will not
change them.

Key variables (full list in [backend/.env.example](backend/.env.example)):

| Variable | Purpose |
|---|---|
| `ELASTIC_ENDPOINT`, `ELASTIC_API_KEY` | Elastic Cloud (OCR/speech/audio/timeline) |
| `MILVUS_ENDPOINT_1`, `MILVUS_TOKEN_1` | BTC Milvus/Zilliz (đầy đủ kênh hiện tại) |
| `MILVUS_ENDPOINT_2`, `MILVUS_TOKEN_2` | InfoShot++ Milvus/Zilliz (separate PE/Qwen collections) |
| `PE_ENCODER_URL` / `GLAP_ENCODER_URL` | text/audio encoder endpoints |
| `QWEN3_VL_ENCODER_URL`, `QWEN3_VL_ENCODER_TOKEN` | optional InfoShot++ Qwen3-VL text encoder tunnel/auth |
| `TARA_ENABLED`, `TARA_ENCODER_URL`, `TARA_ENCODER_TOKEN` | optional InfoShot++ TARA text encoder and clip retrieval |
| `MILVUS_QWEN3_VL_IMAGE_COLLECTION_2` | InfoShot++ native 4096-d Qwen image collection |
| `MILVUS_TARA_COLLECTION_2` | InfoShot++ TARA 3584-d clip collection |
| `MEDIA_BASE_URL` | Cloudflare R2: BTC keyframes và video của cả hai profile |
| `KEYFRAME_MEDIA_BASE_URL_2` | Cloudflare R2 public base cho keyframe InfoShot++ |
| `KEYFRAME_MEDIA_FALLBACK_BASE_URL_2` | Hugging Face fallback khi keyframe InfoShot++ trên R2 tải lỗi |
| `QUERY_LLM_API_KEY` | query LLM for the LLM switch and Expand; defaults to `DEEPSEEK_API_KEY` (else heuristics) |
| `QUERY_LLM_MODEL`, `QUERY_LLM_BASE_URL` | default `deepseek-flash` at `https://api.deepseek.com` |
| `NVIDIA_API_KEY` | optional LLM fallback of VI→EN translation when Google fails |
| `DRES_BASE_URL` | official DRES host (SELab: `http://10.0.1.21:20740`) |
| `DRES_USERNAME`, `DRES_PASSWORD` | participant account — the backend logs in (Client API v2) and keeps the session |
| `DRES_EVALUATION_ID`, `DRES_SEGMENT_PAD_MS` | optional: pin one run / ± ms around the picked instant (default 500) |
| `QA_VISION_BACKEND` | `deepseek` (default: V4.1 Flash reads the QA frames with `DEEPSEEK_API_KEY`) or `nvila` |
| `DEEPSEEK_GROUNDING_MODEL` | web grounding model; `deepseek-v4-pro` (the only DeepSeek model whose web search runs) |
| `NVILA_BASE_URL`, `NVILA_TOKEN` | optional NVILA-8B QA worker chạy từ Colab notebook (`QA_VISION_BACKEND=nvila`) |
| `DEEPSEEK_API_KEY` | optional DeepSeek built-in `web_search` grounding for QA; backend only |
| `AGENT_ENABLED`, `AGENT_CODEX_BIN`, `AGENT_CLAUDE_BIN` | agent sidecar search; the CLIs must be installed and logged in on the backend host |
| `AGENT_CODEX_MODEL`, `AGENT_CODEX_REASONING_EFFORT` | default `gpt-6-sol` / `high` |
| `AGENT_CLAUDE_MODEL`, `AGENT_CLAUDE_EFFORT` | default `claude-opus-5-5` / `high` (needs Claude Code ≥ 2.1.280) |
| `AGENT_CODEX_FAST`, `AGENT_CLAUDE_FAST` | fast mode, on by default: Codex `service_tier="fast"` (~1.5× on gpt-6-sol), Claude Code `fastMode`; both use quota faster |
| `AGENT_TIMEOUT_SECONDS`, `AGENT_MAX_CONCURRENT` | hard stop per agent (240 s); parallel runs per CLI across tabs (2) |
| `AIC26_MOCK_MODE` | `true` ⇒ run with fixtures, no live services |

TARA artifact verification, Milvus upload, Colab worker setup, and fusion details
are documented in [VideoRetrieval/TARA_INTEGRATION.md](VideoRetrieval/TARA_INTEGRATION.md).

> ⚠️ **Secrets are never committed and never reach the frontend.** The browser
> talks only to the backend. Do not paste real keys into tracked files.

---

## Tests

```bash
cd backend  && PYTHONPATH=. python -m pytest -q   # backend unit/integration tests
cd frontend && npm run test                       # vitest component tests
```

---

## Documentation map

| Doc | Topic |
|---|---|
| [RETRIEVAL_APP_HANDOFF.md](RETRIEVAL_APP_HANDOFF.md) | App contract: identity rules, API, fusion, TRAKE, module map |
| [AIC26_Pipeline.md](AIC26_Pipeline.md) | End-to-end pipeline design (ingestion → UI) |
| [FEATURES.md](FEATURES.md) | Feature inventory |
| [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md) / [SPEECH_PIPELINE.md](SPEECH_PIPELINE.md) / [OCR_PIPELINE.md](OCR_PIPELINE.md) | Per-modality extraction |
| [ELASTIC_RETRIEVAL_HANDOFF.md](ELASTIC_RETRIEVAL_HANDOFF.md) / [MILVUS_EMBEDDING_HANDOFF.md](MILVUS_EMBEDDING_HANDOFF.md) | Index/collection schemas |
| [CLOUDFLARE_R2_MEDIA_HANDOFF.md](CLOUDFLARE_R2_MEDIA_HANDOFF.md) | Media hosting on R2 |
| [HUONG_DAN_SU_DUNG_DRES.md](HUONG_DAN_SU_DUNG_DRES.md) | DRES usage (Vietnamese) |
| [aic26_query_factory_qwen3vl8b_t4x2.ipynb](aic26_query_factory_qwen3vl8b_t4x2.ipynb) | Source-first benchmark factory: Qwen3-VL generator/critic, PE-G14/Milvus + TF-IDF hard negatives, human review, private ground truth and evaluator |
| [NVILA_QA_INTEGRATION.md](NVILA_QA_INTEGRATION.md) | UIT/VBS method adaptation, Colab A100 worker, backend contract, UI workflow and operations |
| [Progressive Hint Memory](docs/PROGRESSIVE_HINT_MEMORY.md) | PHM session API, rescue/backfill, console workflow, event replay and implementation status |
| [PHM ground-truth guide](benchmarks/PHM_GROUND_TRUTH_GUIDE.md) | Instructions for an AI to verify start/end from video and derive three hints from `TKIS_queries.xlsx`; benchmark deferred |

---

## Tech stack

**Backend:** Python · FastAPI · Uvicorn · Elasticsearch · Milvus/pymilvus ·
DeepSeek (query LLM, web grounding) · Whisper (voice input).
**Frontend:** TypeScript · React 18 · Vite · Vitest.
**Infra:** Elastic Cloud · Zilliz/Milvus · Cloudflare R2 · Kaggle GPU encoders.

---

## Status

Implemented: multi-channel retrieval + RRF fusion, group-by-video, TRAKE DP,
query parser (LLM + heuristic), DeepSeek-assisted QA, the 100-answer generator
and its bulk button in the Submission tab, submit guard, mock mode,
audio vector search. Planned (clean seams left in place — see
RETRIEVAL_APP_HANDOFF.md §8): general-purpose VLM reranking,
object–scene–action tag filters, and a custom Vietnamese analyzer.
