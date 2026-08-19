#!/usr/bin/env bash
# Build the AIC26 console image and assemble a folder a teammate can download,
# unzip and run — no repository, no Python, no Node, no build step on their side,
# and on Windows no manual Docker install either.
#
#   ./docker/build-bundle.sh                 # tag = today's date
#   ./docker/build-bundle.sh --tag 20260819b
#   ./docker/build-bundle.sh --no-config     # ship without the team's .env
#   ./docker/build-bundle.sh --skip-build    # re-zip using the image already built
#
# Output: docker/dist/aic26-console-<tag>.zip — the single file you send.
#
# Zip rather than tar.gz because Windows opens it with a double-click, and this
# file is very likely to pass through a Windows machine or Drive on its way to
# the operator.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/docker"
TEMPLATE_DIR="${DOCKER_DIR}/bundle"

TAG="$(date +%Y%m%d)"
EMBED_CONFIG=1
SKIP_BUILD=0

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)        TAG="$2"; shift 2 ;;
    --no-config)  EMBED_CONFIG=0; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help)    sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

IMAGE="aic26-console:${TAG}"
NAME="aic26-console-${TAG}"
DIST="${DOCKER_DIR}/dist"
BUNDLE="${DIST}/${NAME}"
ZIP="${DIST}/${NAME}.zip"

say()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[31m[X] %s\033[0m\n' "$*" >&2; exit 1; }

for tool in docker zip; do
  command -v "$tool" >/dev/null 2>&1 || die "Need '${tool}' (apt install ${tool} / pacman -S ${tool})."
done

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 1. Build args — the only values Vite can bake, so they must come from here.
# ---------------------------------------------------------------------------
VITE_SUPABASE_URL=""
VITE_SUPABASE_PUBLISHABLE_KEY=""
VITE_SUBMISSION_ROOM=""

if [ -f "${REPO_ROOT}/frontend/.env.local" ]; then
  # Read without sourcing: .env.local is data, not shell, and a stray backtick
  # in a key would otherwise execute here.
  while IFS='=' read -r key value; do
    key="${key#"${key%%[![:space:]]*}"}"
    case "$key" in
      VITE_SUPABASE_URL)             VITE_SUPABASE_URL="$value" ;;
      VITE_SUPABASE_PUBLISHABLE_KEY) VITE_SUPABASE_PUBLISHABLE_KEY="$value" ;;
      VITE_SUBMISSION_ROOM)          VITE_SUBMISSION_ROOM="$value" ;;
    esac
  done < <(grep -E '^[[:space:]]*VITE_' "${REPO_ROOT}/frontend/.env.local" || true)
fi

: "${VITE_SUBMISSION_ROOM:=aic26}"

if [ -z "${VITE_SUPABASE_URL}" ] || [ -z "${VITE_SUPABASE_PUBLISHABLE_KEY}" ]; then
  warn "frontend/.env.local has no Supabase values — the Submission tab will be"
  warn "local-only in this bundle (no shared answer table across the team)."
else
  say "Shared submission: room '${VITE_SUBMISSION_ROOM}' via ${VITE_SUPABASE_URL}"
fi

# ---------------------------------------------------------------------------
# 2. Build
# ---------------------------------------------------------------------------
if [ "${SKIP_BUILD}" -eq 0 ]; then
  say "Building ${IMAGE}"
  docker build \
    -f docker/Dockerfile \
    -t "${IMAGE}" \
    --build-arg "VITE_SUPABASE_URL=${VITE_SUPABASE_URL}" \
    --build-arg "VITE_SUPABASE_PUBLISHABLE_KEY=${VITE_SUPABASE_PUBLISHABLE_KEY}" \
    --build-arg "VITE_SUBMISSION_ROOM=${VITE_SUBMISSION_ROOM}" \
    .
else
  docker image inspect "${IMAGE}" >/dev/null 2>&1 \
    || die "--skip-build given but ${IMAGE} does not exist yet."
  say "Reusing existing ${IMAGE}"
fi

# ---------------------------------------------------------------------------
# 3. Assemble
# ---------------------------------------------------------------------------
say "Assembling ${BUNDLE}"
rm -rf "${BUNDLE}" "${ZIP}"
mkdir -p "${BUNDLE}/scripts"

cp "${TEMPLATE_DIR}/CHAY-APP.bat"       "${BUNDLE}/"
cp "${TEMPLATE_DIR}/DUNG-APP.bat"       "${BUNDLE}/"
cp "${TEMPLATE_DIR}/HUONG_DAN.md"       "${BUNDLE}/"
cp "${TEMPLATE_DIR}/run.sh"             "${BUNDLE}/"
cp "${TEMPLATE_DIR}/stop.sh"            "${BUNDLE}/"
cp "${TEMPLATE_DIR}/scripts/aic26.ps1"  "${BUNDLE}/scripts/"
cp "${TEMPLATE_DIR}/scripts/stop.ps1"   "${BUNDLE}/scripts/"
cp "${REPO_ROOT}/backend/.env.example"  "${BUNDLE}/.env.example"
chmod +x "${BUNDLE}/run.sh" "${BUNDLE}/stop.sh"

# Windows text files: CRLF, and UTF-8 *with BOM* for the PowerShell ones —
# Windows PowerShell 5.1 reads a BOM-less .ps1 as the legacy ANSI code page and
# turns every Vietnamese diacritic into mojibake.
crlf() { sed -i 's/\r\?$/\r/' "$1"; }
bom()  { printf '\xEF\xBB\xBF' | cat - "$1" > "$1.tmp" && mv "$1.tmp" "$1"; }

for f in "${BUNDLE}/CHAY-APP.bat" "${BUNDLE}/DUNG-APP.bat"; do crlf "$f"; done
for f in "${BUNDLE}/scripts/aic26.ps1" "${BUNDLE}/scripts/stop.ps1"; do crlf "$f"; bom "$f"; done

# ---------------------------------------------------------------------------
# 4. The team's configuration
# ---------------------------------------------------------------------------
if [ "${EMBED_CONFIG}" -eq 1 ]; then
  # backend/.env first: that is the file the app reads during development, so it
  # is the one that actually gets maintained. config/ is a Docker volume whose
  # copy is only as fresh as the last time someone imported through the UI —
  # it silently went a schema version stale once already.
  SOURCE_ENV=""
  for candidate in "${REPO_ROOT}/backend/.env" "${REPO_ROOT}/config/.env"; do
    if [ -f "${candidate}" ]; then SOURCE_ENV="${candidate}"; break; fi
  done

  if [ -n "${SOURCE_ENV}" ]; then
    mkdir -p "${BUNDLE}/config"
    # 600 in the archive is cosmetic (zip on Windows drops it), but it keeps the
    # staging copy on this machine from being world-readable.
    install -m 600 "${SOURCE_ENV}" "${BUNDLE}/config/.env"
    say "Embedded ${SOURCE_ENV#"${REPO_ROOT}"/} — the bundle runs with no setup"

    # Guard against shipping a config that predates a schema change. Only keys
    # the template documents count, and only those it ships with no default —
    # anything with a default in .env.example is genuinely optional.
    keys_of() { grep -oE '^[[:space:]]*[A-Z0-9_]+=' "$1" | tr -d ' =' | sort -u; }
    # `.env.example` marks these Optional in prose; absence is the normal case.
    optional='DRES_SESSION
DRES_EVALUATION_ID
PE_ENCODER_TOKEN
GLAP_ENCODER_URL'
    missing="$(comm -23 \
      <(grep -oE '^[A-Z0-9_]+=$' "${REPO_ROOT}/backend/.env.example" | tr -d '=' | sort -u) \
      <(printf '%s\n' "${optional}" | cat - <(keys_of "${SOURCE_ENV}") | sort -u))"
    if [ -n "${missing}" ]; then
      warn "These keys are in .env.example but absent from ${SOURCE_ENV#"${REPO_ROOT}"/}:"
      printf '%s\n' "${missing}" | sed 's/^/      /'
      warn "Features that depend on them will be dead in this bundle."
    fi

    warn "This zip now contains REAL API KEYS. Internal distribution only:"
    warn "do not put it on GitHub, a public Drive link, or anywhere outside the team."
  else
    warn "No backend/.env or config/.env found — shipping without a config."
    warn "Teammates will have to import a .env through the settings screen."
  fi
else
  say "Shipping without a config (--no-config)"
fi

# ---------------------------------------------------------------------------
# 5. Generated files
# ---------------------------------------------------------------------------
# The bundle has no source tree, so its compose file loads the shipped image
# rather than building one. Generated here, not copied, so it can never drift
# out of sync with the tag that was actually built.
cat > "${BUNDLE}/docker-compose.yml" <<YAML
name: aic26

services:
  console:
    image: ${IMAGE}
    container_name: aic26-console
    # Loopback only. The console can rewrite its own credentials through
    # /api/config, so exposing it to the LAN would hand that to anyone on the
    # network. To share it with a teammate, change to "8000:8000" deliberately.
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      # The team .env — shipped in the zip, and editable afterwards.
      - ./config:/config
      # Submit history — the local record of what was sent to DRES.
      - ./data:/data
    # The launcher exports these so files written here stay owned by you.
    # Windows pins 0:0; Docker Desktop's file sharing has no host uid.
    user: "\${AIC26_UID:-1000}:\${AIC26_GID:-1000}"
    restart: unless-stopped
YAML

# Stays LF even though Windows reads it too. bash `source`es this file, and a
# trailing CR would ride along into the image name and the archive filename —
# turning both into things that do not exist. PowerShell's Get-Content is
# newline-agnostic, so LF costs it nothing.
cat > "${BUNDLE}/bundle.env" <<ENV
# Written by docker/build-bundle.sh - read by run.sh and scripts/aic26.ps1.
AIC26_IMAGE=${IMAGE}
AIC26_IMAGE_ARCHIVE=${NAME}.tar.gz
ENV

# ---------------------------------------------------------------------------
# 6. The image itself (the slow part)
# ---------------------------------------------------------------------------
say "Saving the image — a few minutes"
docker save "${IMAGE}" | gzip -1 > "${BUNDLE}/${NAME}.tar.gz"

# ---------------------------------------------------------------------------
# 7. Zip
# ---------------------------------------------------------------------------
say "Zipping"
# -9 on the text files; the saved image is already gzipped, so storing it saves
# minutes of CPU for well under a percent of size.
( cd "${DIST}" && zip -q -r -9 -n ".tar.gz:.gz:.zip" "${ZIP}" "${NAME}" )

echo
say "Bundle ready"
du -h "${ZIP}" | sed 's/^/    /'
echo "    contents: $(cd "${BUNDLE}" && ls -A | tr '\n' ' ')"
echo
echo "  Send:  ${ZIP}"
echo "  They:  unzip, then double-click CHAY-APP.bat (Windows) or ./run.sh (Linux/macOS)"
