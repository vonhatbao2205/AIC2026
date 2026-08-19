#!/usr/bin/env bash
# Dừng AIC26 Console. config/ và data/ được giữ nguyên.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export AIC26_UID="$(id -u)" AIC26_GID="$(id -g)"
docker compose down
echo "Đã dừng. Cấu hình trong config/ và lịch sử submit trong data/ vẫn được giữ."
