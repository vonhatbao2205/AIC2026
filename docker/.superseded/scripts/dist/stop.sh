#!/usr/bin/env bash
# Stop the console. config/ and data/ are left untouched.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
docker compose down
echo "Đã dừng. Cấu hình trong config/ và lịch sử submit trong data/ vẫn được giữ."
