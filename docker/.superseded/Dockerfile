# syntax=docker/dockerfile:1
#
# AIC26 retrieval console, packaged as one downloadable app.
#
# The result is a single container that serves the operator console and its API
# on one port. It ships with no credentials: the person running it imports the
# team's `.env` through the settings screen, which writes to the mounted
# /config volume so the configuration outlives the image.
#
#   docker compose up -d      →  http://localhost:8000

# ---- 1. Build the console --------------------------------------------------
FROM node:20-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

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

# Bake the default Whisper model into the image. Downloading it on the first
# mic press would mean a stall mid-task — and no network at all in a venue that
# blocks Hugging Face. Owned by the runtime user so switching WHISPER_MODEL can
# still cache a different size at runtime.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

COPY backend/app ./app
COPY backend/.env.example ./.env.example
COPY --from=ui /ui/dist ./static

# uid 1000 matches the usual first login account on Linux, so files written to
# the bind-mounted config/ and data/ stay editable by the person who ran it.
RUN useradd --uid 1000 --create-home app \
 && mkdir -p /config /data \
 && chown -R app:app /config /data /opt/models

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).read()" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
