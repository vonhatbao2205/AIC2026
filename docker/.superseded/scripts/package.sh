#!/usr/bin/env bash
# Build the AIC26 console image and assemble a folder a teammate can download,
# unzip and run — no repository, no Python, no Node, no build step on their side.
#
#   ./scripts/package.sh            # tag = today's date
#   ./scripts/package.sh 20260806b  # explicit tag
#
# Output: dist-app/  (and dist-app.zip, the thing you actually send). Zip rather
# than tar.gz because Windows opens it with a double-click — the file is very
# likely to pass through a Windows machine or Drive on its way to the operator.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-$(date +%Y%m%d)}"
IMAGE="aic26-console:${TAG}"
NAME="aic26-console-${TAG}"
STAGE="${REPO_ROOT}/dist-app"
BUNDLE="${STAGE}/${NAME}"
ZIP="${REPO_ROOT}/dist-app.zip"

command -v zip >/dev/null 2>&1 || {
  echo "Need the 'zip' command (apt install zip / pacman -S zip)." >&2
  exit 1
}

cd "${REPO_ROOT}"

echo "==> Building ${IMAGE}"
docker build -t "${IMAGE}" .

echo "==> Assembling ${BUNDLE}"
rm -rf "${STAGE}" "${ZIP}"
mkdir -p "${BUNDLE}"

echo "==> Saving the image (this is the slow part)"
docker save "${IMAGE}" | gzip -1 > "${BUNDLE}/${NAME}.tar.gz"

cp "${REPO_ROOT}/backend/.env.example" "${BUNDLE}/.env.example"
cp "${REPO_ROOT}/scripts/dist/run.sh" "${BUNDLE}/run.sh"
cp "${REPO_ROOT}/scripts/dist/stop.sh" "${BUNDLE}/stop.sh"
cp "${REPO_ROOT}/scripts/dist/run.bat" "${BUNDLE}/run.bat"
cp "${REPO_ROOT}/scripts/dist/stop.bat" "${BUNDLE}/stop.bat"
cp "${REPO_ROOT}/scripts/dist/HUONG_DAN.md" "${BUNDLE}/HUONG_DAN.md"
chmod +x "${BUNDLE}/run.sh" "${BUNDLE}/stop.sh"

# The bundle has no source tree, so its compose file loads the shipped image
# rather than building one. Kept in sync by hand with ../docker-compose.yml.
cat > "${BUNDLE}/docker-compose.yml" <<EOF
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
      # The .env imported through the settings screen lives here.
      - ./config:/config
      # Submit history — the local record of what was sent to DRES.
      - ./data:/data
    # The launcher exports these so files written here stay owned by you.
    user: "\${AIC26_UID:-1000}:\${AIC26_GID:-1000}"
    restart: unless-stopped
EOF

# The launchers need to know which archive and tag they are looking at.
cat > "${BUNDLE}/bundle.env" <<EOF
# Written by scripts/package.sh - read by run.sh and run.bat.
AIC26_IMAGE=${IMAGE}
AIC26_IMAGE_ARCHIVE=${NAME}.tar.gz
EOF

echo "==> Zipping"
# -9 on the text files; the saved image is already gzipped, so storing it saves
# minutes of CPU for well under a percent of size.
( cd "${STAGE}" && zip -q -r -9 -n ".tar.gz:.gz:.zip" "${ZIP}" "${NAME}" )

echo
echo "Bundle ready:"
du -h "${ZIP}" | sed 's/^/  /'
echo "  contents: $(ls -A "${BUNDLE}" | tr '\n' ' ')"
echo
echo "Send dist-app.zip. The recipient unzips it, then runs:"
echo "  Linux/macOS:  ./run.sh"
echo "  Windows:      double-click run.bat"
