#!/usr/bin/env bash
# Start the AIC26 console. Loads the bundled image on first run, then opens it.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./bundle.env

URL="http://localhost:8000"

die() { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
say() { printf '\033[36m==>\033[0m %s\n' "$*"; }

command -v docker >/dev/null 2>&1 || die \
  "Chưa có Docker.
  Ubuntu/Debian: sudo apt install docker.io docker-compose-v2
  Arch:          sudo pacman -S docker docker-compose
  Sau đó:        sudo systemctl enable --now docker && sudo usermod -aG docker \$USER
                 (đăng xuất/đăng nhập lại để nhóm docker có hiệu lực)"

docker info >/dev/null 2>&1 || die \
  "Docker đã cài nhưng chưa chạy được.
  Thử:  sudo systemctl start docker
  Nếu báo permission denied:  sudo usermod -aG docker \$USER  rồi đăng nhập lại."

if ! docker image inspect "${AIC26_IMAGE}" >/dev/null 2>&1; then
  say "Nạp image lần đầu (${AIC26_IMAGE_ARCHIVE}) — mất 1–2 phút…"
  [ -f "${AIC26_IMAGE_ARCHIVE}" ] || die "Không tìm thấy ${AIC26_IMAGE_ARCHIVE} trong thư mục này."
  docker load -i "${AIC26_IMAGE_ARCHIVE}"
fi

# Files the container writes into config/ and data/ stay owned by you.
mkdir -p config data
export AIC26_UID="$(id -u)" AIC26_GID="$(id -g)"

say "Khởi động…"
docker compose up -d

say "Chờ backend sẵn sàng…"
for _ in $(seq 1 60); do
  if curl -fsS "${URL}/api/health" >/dev/null 2>&1; then
    say "Sẵn sàng: ${URL}"
    if [ ! -f config/.env ]; then
      printf '\n\033[33m%s\033[0m\n' \
        "Chưa có cấu hình — app sẽ tự mở màn hình Settings.
Kéo thả file .env của nhóm vào đó rồi bấm “Import & áp dụng”.
(Mẫu các biến cần điền: .env.example trong thư mục này.)"
    fi
    command -v xdg-open >/dev/null 2>&1 && xdg-open "${URL}" >/dev/null 2>&1 &
    echo
    echo "  Dừng app:   ./stop.sh"
    echo "  Xem log:    docker compose logs -f"
    exit 0
  fi
  sleep 1
done

die "Backend không phản hồi sau 60s. Xem log:  docker compose logs"
