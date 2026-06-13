# DRES Simulator

A local, DRES‑like **contest judge simulator** for an AIC / HCMC AI Challenge video‑retrieval
team. It is **not** the official BTC DRES server — it is a self‑hosted stand‑in used to:

- run training sessions and mock contests,
- test a separate video‑retrieval UI against a stable submit API,
- score TKIS / VKIS / QA / TRAKE answers against ground truth,
- log mistakes, penalties and submission history,
- and later **swap the submission adapter to the official DRES endpoint** without changing the
  retrieval UI.

Scoring mirrors the real contest:
`score = max(0, score_max_base − penalty_per_wrong · previous_wrong_count + time_bonus)`, with
`time_bonus = score_time_bonus · remaining / duration`. The penalty gradient dominates the time
gradient — so the simulator faithfully rewards **minimizing wrong submits**.

> 📖 **Hướng dẫn sử dụng web (tiếng Việt):** [docs/HUONG_DAN_SU_DUNG.md](docs/HUONG_DAN_SU_DUNG.md)

---

## 1. Overview

| Piece | Stack | Deploy target |
|------|-------|---------------|
| **backend/** | Python 3.11 · FastAPI · SQLAlchemy 2.0 · Alembic · JWT · pytest | Long‑running container (Docker on VPS / Render / Railway / Fly) |
| **frontend/** | React · Vite · TypeScript · Tailwind · Zustand · TanStack Query · React Hook Form + Zod · dnd‑kit | **Static SPA on Vercel** |
| **db** | PostgreSQL (prod) · SQLite (quick local/dev/tests) | docker‑compose / managed Postgres |

**Task types:** `TKIS` (textual KIS), `VKIS` (visual KIS), `QA` (KIS + answer), `TRAKE`
(ordered multi‑event). **Submission adapter** is pluggable: `local` (this simulator) or
`official` (stub for the real DRES server).

The frontend ships a reusable typed client at
[`frontend/src/lib/dresClient.ts`](frontend/src/lib/dresClient.ts) — the same file your external
retrieval UI imports.

---

## 2. Architecture

```
                          ┌─────────────────────────────┐
   Browser (Vercel SPA)   │   React + Vite frontend      │
   ─ Admin console        │   src/lib/dresClient.ts ─────┼──┐
   ─ Team console         └─────────────────────────────┘  │
                                                            │  HTTPS (JWT or X-API-Key)
   Your separate              import the same               │
   video‑retrieval UI  ─────  dresClient.ts  ───────────────┤
                                                            ▼
                          ┌──────────────────────────────────────────────┐
                          │              FastAPI backend                   │
                          │  routers → services → judging + scoring        │
                          │                     │                          │
                          │            SubmissionAdapter                   │
                          │            ├── LocalJudgeAdapter  (default)     │
                          │            └── OfficialDresAdapter (stub/TODO)  │
                          └───────────────┬───────────────────────────────┘
                                          │ SQLAlchemy
                                          ▼
                               PostgreSQL (or SQLite for dev)
```

Request flow for a submission: `POST /api/submissions` → `submission_service` (resolve task,
require running session + active task, dedup, get/create per‑team state) → `adapter.submit()`
(judge + score) → persist `Submission`, update `ScoreState` / `TaskTeamState`, append `EventLog`
→ structured `SubmissionResult`.

### Repo layout
```
DRES/
  backend/   app/{models,schemas,judging,scoring,adapters,services,routers,scripts}, alembic/, tests/
  frontend/  src/{lib,store,hooks,components,pages}
  data/sample_tasks.json
  docker-compose.yml   .env.example   README.md   docs/
```

---

## 3. Local development

### Prerequisites
Python 3.11+ (works on 3.14), Node 18+, and optionally Docker + Postgres.

### Backend (SQLite — zero config)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Uses sqlite:///./dres.db by default. Seed admin/team/sample tasks:
python -m app.scripts.seed

# Run the API (http://localhost:8000, docs at /docs)
uvicorn app.main:app --reload
```

### Backend (Postgres)
```bash
export DATABASE_URL=postgresql://dres:dres@localhost:5432/dres
alembic upgrade head          # create schema
python -m app.scripts.seed    # seed
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
cp .env.example .env          # set VITE_API_BASE_URL=http://localhost:8000
npm run dev                   # http://localhost:5173
```

### Tests
```bash
cd backend && pytest          # 53 tests: judging, scoring, dedup, penalties, full API flow
```

---

## 4. Docker Compose (full stack)

```bash
cp .env.example .env          # adjust secrets/origins as needed
docker compose up --build     # postgres + backend API on :8000 (auto-migrated + seeded)

# Also build & serve the frontend (nginx) on :5173:
docker compose --profile frontend up --build
```

The backend container runs `alembic upgrade head`, optionally seeds
(`RUN_SEED_ON_START=1`), then serves uvicorn. Postgres data persists in the `pgdata` volume.

---

## 5. Seed data

`python -m app.scripts.seed` (or `RUN_SEED_ON_START=1` in Docker) is **idempotent** and creates:

- **admin** user — `admin` / `admin123`
- **team** `team1` + user — `team1` / `team123` (with a demo API key `team1-demo-key`)
- a sample **contest** + **session** (`Training Session 1`)
- the four **sample tasks** from [`data/sample_tasks.json`](data/sample_tasks.json)
  (`q001` TKIS, `q002` QA, `q003` VKIS, `q004` TRAKE), added to the session as *ready*.

Credentials are configurable via `SEED_*` env vars (see `.env.example`).

**Bulk task import/export.** The Task Bank admin page imports & exports tasks as **JSON or
CSV**. Templates: [`data/sample_tasks.json`](data/sample_tasks.json) and
[`data/sample_tasks.csv`](data/sample_tasks.csv) (1 row per task — fill it in Excel/Sheets). CSV
column conventions are documented in
[`backend/app/services/csv_import.py`](backend/app/services/csv_import.py): `hints` as
`sec@text|sec@text`, `answer_aliases` as `a|b`, `trake_events` as a JSON array. Endpoints:
`POST /api/admin/tasks/import`, `POST /api/admin/tasks/import-csv`, `GET /api/admin/tasks/export`,
`GET /api/admin/tasks/export.csv`.

---

## 6. Frontend on Vercel

The frontend is a static SPA — **no backend code or secrets live in it**.

1. In Vercel, **New Project** → import this repo → set **Root Directory** to `frontend`.
   (Framework auto‑detects as Vite; `vercel.json` already sets build/output + SPA rewrites.)
2. Add an environment variable:
   `VITE_API_BASE_URL = https://your-backend-api.com`
3. **Deploy.**

`VITE_*` values are inlined at build time, so redeploy after changing the API URL. Ensure the
backend's `CORS_ORIGINS` includes your Vercel domain (step 7).

---

## 7. Backend deploy (Docker)

Build and run the long‑running API anywhere that runs containers:

```bash
docker build -t dres-backend ./backend
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:pass@host:5432/dres" \
  -e JWT_SECRET="$(python -c 'import secrets;print(secrets.token_hex(32))')" \
  -e CORS_ORIGINS="https://your-frontend.vercel.app" \
  -e SUBMISSION_MODE="local" \
  -e RUN_SEED_ON_START="1" \
  dres-backend
```

Key environment variables:

| Var | Purpose |
|-----|---------|
| `DATABASE_URL` | Postgres connection string, e.g. `postgresql://user:pass@host:5432/dres` |
| `JWT_SECRET` | HMAC secret for JWTs (use 32+ bytes) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token lifetime (default 720) |
| `CORS_ORIGINS` | Comma‑separated allowed origins; include your Vercel URL (`*` for dev only) |
| `SUBMISSION_MODE` | `local` (default) or `official` |
| `RUN_SEED_ON_START` | `1` to seed on container start |

The backend is a normal ASGI service — **not** serverless — so run it on a host that keeps it
alive (VPS / Render / Railway / Fly.io). On Render/Railway, point the start command at the image
or use `./entrypoint.sh`.

---

## 8. API examples (curl)

```bash
BASE=http://localhost:8000

# --- Login (admin) ---
TOKEN=$(curl -s -X POST $BASE/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

# Start the seeded session (admin). Grab its id first:
SID=$(curl -s $BASE/api/admin/sessions -H "Authorization: Bearer $TOKEN" | python -c "import sys,json;print(json.load(sys.stdin)[0]['id'])")
curl -s -X POST $BASE/api/admin/sessions/$SID/start -H "Authorization: Bearer $TOKEN"

# --- Team: current task ---
TEAM=$(curl -s -X POST $BASE/api/auth/login -H 'Content-Type: application/json' \
  -d '{"username":"team1","password":"team123"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl -s $BASE/api/tasks/current -H "Authorization: Bearer $TEAM"

# --- Submit TKIS ---
curl -s -X POST $BASE/api/submissions -H "Authorization: Bearer $TEAM" -H 'Content-Type: application/json' -d '{
  "task_id":"q001","task_type":"TKIS","video_id":"L01_V001","frame_id":"000123","frame_idx":123,"timestamp":4.10}'

# --- Submit QA ---
curl -s -X POST $BASE/api/submissions -H "Authorization: Bearer $TEAM" -H 'Content-Type: application/json' -d '{
  "task_id":"q002","task_type":"QA","video_id":"L02_V003","frame_idx":888,"answer":"15 người"}'

# --- Submit TRAKE ---
curl -s -X POST $BASE/api/submissions -H "Authorization: Bearer $TEAM" -H 'Content-Type: application/json' -d '{
  "task_id":"q004","task_type":"TRAKE","video_id":"L03_V005","events":[
    {"event_index":1,"frame_idx":100},{"event_index":2,"frame_idx":400},{"event_index":3,"frame_idx":700}]}'

# --- Scoreboard / history ---
curl -s $BASE/api/scoreboard -H "Authorization: Bearer $TEAM"
curl -s "$BASE/api/submissions/history?task_id=q001" -H "Authorization: Bearer $TEAM"
```

A submission responds with:
```json
{"submission_id":"…","status":"correct","accuracy":1.0,"score_delta":832.5,
 "penalty_count_after":0,"time_bonus":332.5,"message":"Correct within epsilon.","detail":{…}}
```

> External clients may authenticate with the team JWT **or** an `X-API-Key: <team api_key>` header
> instead of the bearer token.

---

## 9. External retrieval UI integration

Your separate React retrieval UI only needs `POST /api/submissions`. The easiest path is to reuse
the typed client:

```ts
import { createDresClient } from "./dresClient"; // copy frontend/src/lib/dresClient.ts

const dres = createDresClient({
  baseUrl: "https://your-backend-api.com",
  apiKey: "team1-demo-key",           // or getToken: () => myJwt
});

const task = await dres.getCurrentTask();
const res = await dres.submitTKIS({
  task_id: task.task_id,
  video_id: "L01_V001",
  frame_idx: 123,
});
console.log(res.status, res.score_delta);   // "correct", 832.5
```

Or call the endpoint directly:
```ts
await fetch(`${BASE}/api/submissions`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": "team1-demo-key" },
  body: JSON.stringify({ task_id: "q001", task_type: "TKIS", video_id: "L01_V001", frame_idx: 123 }),
});
```

Useful client methods: `getContestState`, `getCurrentTask`, `submitTKIS/VKIS/QA/TRAKE`,
`getSubmissionHistory`, `getScoreboard`.

---

## 10. Switching submission mode

Set `SUBMISSION_MODE` on the backend:

- `local` *(default)* — judge against local DB ground truth and score locally.
- `official` — forward to the real BTC DRES server. The adapter
  ([`backend/app/adapters/official.py`](backend/app/adapters/official.py)) is a **stub** marked
  `TODO`; it reads `OFFICIAL_DRES_BASE_URL` / `OFFICIAL_DRES_TOKEN`. Implement its `submit()` to
  map our payload to the DRES API and translate the verdict back.

Your retrieval UI never changes — it always calls `POST /api/submissions`; the backend decides how
to judge.

---

## Scoring & judging summary

- **TKIS/VKIS** — correct iff `video_id` matches and the frame is within epsilon (frame_idx →
  timestamp → exact frame_id, in that priority).
- **QA** — TKIS‑style localization **and** answer match. Vietnamese answers are normalized
  (lowercase, trim, de‑punctuate) and compared exactly, by alias, and accent‑insensitively
  (unidecode), e.g. `muoi lam nguoi` ↔ `mười lăm người`.
- **TRAKE** — `video_id` must match, submitted events must be strictly increasing (else *invalid*),
  per‑event epsilon match; `accuracy = correct/total` → correct (=1.0) / partial (≥0.5) / wrong.
- **Scoring** — configurable per session; wrong submits raise the team's `wrong_count` (penalizing
  future scores for that task), duplicates and already‑solved tasks are rejected without penalty,
  and TRAKE keeps the best partial score.

Session config defaults (editable per session):
```json
{"default_task_duration_seconds":180,"score_max_base":500,"score_time_bonus":500,
 "penalty_per_wrong":100,"invalid_counts_as_wrong":true,"partial_counts_as_finished":false,
 "allow_duplicate_submissions":false}
```
