# syntax=docker/dockerfile:1
#
# AIC26 retrieval console, packaged as one downloadable app.
#
# One container serves the operator console and its API on a single port. It
# ships with no retrieval credentials: the operator imports the team's `.env`
# through the settings screen, which writes to the mounted /config volume so the
# configuration outlives the image.
#
# Build from the REPOSITORY ROOT, pointing at this file:
#   docker build -f docker/Dockerfile -t aic26-console:dev .
#
# The Supabase values below are the only build-time configuration. They must be
# baked in because Vite inlines `import.meta.env.*` at build time — there is no
# runtime hook for them — and without them the Submission tab silently falls
# back to localStorage-only, which is exactly the team feature this bundle
# exists to deliver. The publishable key is low-privilege by design; access is
# decided by the RLS policies in supabase/migrations/001_shared_submission.sql.

# ---- 1. Build the console --------------------------------------------------
FROM node:20-alpine AS ui
WORKDIR /ui

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

ARG VITE_SUPABASE_URL=""
ARG VITE_SUPABASE_PUBLISHABLE_KEY=""
ARG VITE_SUBMISSION_ROOM="aic26"

# `.env.production.local` outranks every other env file Vite loads, so these
# values win over anything a stray checkout file might carry into the context.
# VITE_SUBMISSION_USER is deliberately absent: one image serves the whole team,
# so the operator name has to stay per-machine (localStorage) rather than baked.
RUN printf '%s\n' \
      "VITE_SUPABASE_URL=${VITE_SUPABASE_URL}" \
      "VITE_SUPABASE_PUBLISHABLE_KEY=${VITE_SUPABASE_PUBLISHABLE_KEY}" \
      "VITE_SUBMISSION_ROOM=${VITE_SUBMISSION_ROOM}" \
      > .env.production.local \
 && npm run build

# ---- 2. Runtime ------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    AIC26_CONFIG_DIR=/config \
    AIC26_DATA_DIR=/data \
    AIC26_STATIC_DIR=/app/static \
    HF_HOME=/opt/models

WORKDIR /app

# faster-whisper's ctranslate2 and av wheels carry their own ffmpeg, so voice
# input needs no apt packages of its own.
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the default Whisper model into the image. Downloading it on the first mic
# press would mean a stall mid-task — and no network at all in a venue that
# blocks Hugging Face. Owned by the runtime user so switching WHISPER_MODEL can
# still cache a different size at runtime.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

COPY backend/app ./app
COPY backend/.env.example ./.env.example
COPY --from=ui /ui/dist ./static

# uid 1000 matches the usual first login account on Linux, so files written to
# the bind-mounted config/ and data/ stay editable by the person who ran it.
# Docker Desktop on Windows ignores this and the launcher pins uid 0 instead.
RUN useradd --uid 1000 --create-home app \
 && mkdir -p /config /data \
 && chown -R app:app /config /data /opt/models

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).read()" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
