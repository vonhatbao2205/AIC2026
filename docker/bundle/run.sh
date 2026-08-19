#!/usr/bin/env bash
# AIC26 Console — chạy hệ thống trên Linux / macOS.
#
# Bản Windows (CHAY-APP.bat) tự cài Docker Desktop; ở đây thì không, vì mỗi bản
# phân phối Linux có một trình quản lý gói khác nhau và việc cài gói hệ thống
# sau lưng người dùng là chuyện không nên làm. Thiếu Docker thì script in đúng
# câu lệnh cần chạy rồi dừng.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source ./bundle.env

URL="http://localhost:8000"

die() { printf '\n\033[31m[X] %s\033[0m\n' "$*" >&2; exit 1; }
say() { printf '\033[36m==>\033[0m %s\n' "$*"; }
ok()  { printf '\033[32m    OK  %s\033[0m\n' "$*"; }

command -v docker >/dev/null 2>&1 || die \
"Chưa có Docker.
    Ubuntu/Debian: sudo apt install docker.io docker-compose-v2
    Arch:          sudo pacman -S docker docker-compose
    Fedora:        sudo dnf install docker docker-compose
    Sau đó:        sudo systemctl enable --now docker
                   sudo usermod -aG docker \$USER
                   (đăng xuất rồi đăng nhập lại để nhóm docker có hiệu lực)"

docker info >/dev/null 2>&1 || die \
"Docker đã cài nhưng chưa chạy được.
    Thử:  sudo systemctl start docker
    Nếu báo permission denied:  sudo usermod -aG docker \$USER  rồi đăng nhập lại."

if ! docker image inspect "${AIC26_IMAGE}" >/dev/null 2>&1; then
  say "Nạp image lần đầu (${AIC26_IMAGE_ARCHIVE}) — mất 1–3 phút…"
  [ -f "${AIC26_IMAGE_ARCHIVE}" ] || die \
"Không tìm thấy ${AIC26_IMAGE_ARCHIVE} trong thư mục này.
    Giải nén lại file zip và giữ nguyên tất cả các file cạnh nhau."
  docker load -i "${AIC26_IMAGE_ARCHIVE}"
  ok "Đã nạp image"
fi

mkdir -p config data
# Files the container writes into config/ and data/ stay owned by you.
export AIC26_UID="$(id -u)" AIC26_GID="$(id -g)"

say "Khởi động…"
docker compose up -d

say "Chờ backend sẵn sàng…"
for _ in $(seq 1 90); do
  if curl -fsS "${URL}/api/health" >/dev/null 2>&1; then
    ok "Sẵn sàng: ${URL}"
    if [ ! -f config/.env ]; then
      printf '\n\033[33m%s\033[0m\n' \
"Chưa có cấu hình — app sẽ tự mở màn hình Cấu hình.
Kéo thả file .env của nhóm vào đó rồi bấm “Import & áp dụng”.
(Mẫu các biến cần điền: .env.example trong thư mục này.)"
    fi
    command -v xdg-open >/dev/null 2>&1 && xdg-open "${URL}" >/dev/null 2>&1 &
    command -v open >/dev/null 2>&1 && open "${URL}" >/dev/null 2>&1 &
    echo
    echo "  Dừng app:   ./stop.sh"
    echo "  Xem log:    docker compose logs -f"
    exit 0
  fi
  sleep 1
done

die "Backend không phản hồi sau 90s. Xem log:  docker compose logs"
