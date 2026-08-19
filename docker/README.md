# docker/ — đóng gói AIC26 Console cho team

Thư mục này chứa **toàn bộ** phần Docker của dự án: định nghĩa image, compose
cho máy phát triển, và bộ script mà team nhận được.

Mục tiêu: người nhận chỉ phải làm **một** việc — double-click `CHAY-APP.bat`.
Máy chưa có Docker thì script tự cài Docker Desktop và WSL2; đã có rồi thì tự
khởi động và chạy hệ thống.

```
docker/
├── Dockerfile                  # image: build UI (Vite) → runtime FastAPI + static
├── Dockerfile.dockerignore     # build context = repo root, nên phải chặn rất chặt
├── compose.dev.yml             # build từ source, cho máy của bạn
├── build-bundle.sh             # ← chạy cái này để tạo file zip gửi team
├── bundle/                     # khuôn của thư mục team nhận được
│   ├── CHAY-APP.bat            #   Windows: cài Docker (nếu cần) + chạy
│   ├── DUNG-APP.bat            #   Windows: dừng
│   ├── run.sh / stop.sh        #   Linux / macOS
│   ├── HUONG_DAN.md            #   hướng dẫn cho người nhận
│   └── scripts/
│       ├── aic26.ps1           #   toàn bộ logic cài đặt + khởi động
│       └── stop.ps1
└── dist/                       # kết quả build (đã .gitignore)
```

---

## Tạo bản gửi team

```bash
./docker/build-bundle.sh
```

Xong sẽ có `docker/dist/aic26-console-<ngày>.zip`. Gửi đúng file đó.

| Tuỳ chọn | Tác dụng |
|---|---|
| `--tag 20260819b` | Đặt tag khác (mặc định: ngày hôm nay) |
| `--no-config` | **Không** nhúng `config/.env` — người nhận phải tự import |
| `--skip-build` | Đóng gói lại bằng image đã build sẵn (nhanh, khi chỉ sửa script) |

> ⚠️ Mặc định script **nhúng `config/.env`** vào bundle để team chạy được ngay
> mà không phải import gì. Nghĩa là file zip **chứa API key thật**. Chỉ gửi nội
> bộ — không đưa lên GitHub hay link Drive công khai. Cần bản sạch thì dùng
> `--no-config`.

---

## Hai loại cấu hình, và vì sao chúng khác nhau

Đây là chỗ dễ nhầm nhất, nên nói rõ:

**1. Cấu hình backend (`config/.env`) — đọc lúc chạy.**
Elastic, Milvus, PE encoder, DRES, media URL… Nằm trên volume `/config`, đổi
được bất cứ lúc nào qua màn hình Cấu hình trong app, không cần build lại. Đây là
lý do image không bao giờ chứa key backend.

**2. Cấu hình frontend (`VITE_*`) — bake lúc build.**
Vite thay thẳng `import.meta.env.*` thành hằng số trong bundle JS, nên **không
có** đường nào sửa lúc chạy. `build-bundle.sh` đọc chúng từ
`frontend/.env.local` rồi truyền vào `docker build --build-arg`:

| Biến | Vì sao phải bake |
|---|---|
| `VITE_SUPABASE_URL` | Thiếu là tab Submission tụt về localStorage-only, mất tính năng chia sẻ đáp án của cả nhóm |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Như trên. Key này ít quyền theo thiết kế; quyền thật do RLS trong `supabase/migrations/001_shared_submission.sql` quyết định |
| `VITE_SUBMISSION_ROOM` | Cả nhóm phải cùng một room mới thấy đáp án của nhau |

`VITE_SUBMISSION_USER` **cố tình không** bake: một image dùng chung cho cả team,
bake vào thì ai cũng mang một tên. Mỗi người tự đặt tên trong màn hình Cấu hình
(ô **Tên của bạn**), lưu vào localStorage của máy đó — xem mục 3 của
`bundle/HUONG_DAN.md`.

Vì vậy `getDisplayName()` ưu tiên localStorage **trên** `VITE_SUBMISSION_USER`:
biến build-time chỉ còn là giá trị mặc định cho checkout dev. Một ô nhập mà thua
hằng số lúc build thì tệ hơn là không có ô nào.

> Đổi Supabase project hay đổi room ⇒ phải build lại image, không chỉ đổi `.env`.

---

## Chạy thử trên máy bạn

```bash
set -a; source frontend/.env.local; set +a
docker compose -f docker/compose.dev.yml up -d --build
# → http://localhost:8000     dừng: docker compose -f docker/compose.dev.yml down
```

Compose này mount `config/` và `data/` ở gốc repo, tức đúng những thư mục bạn
đang dùng — không tạo thêm bản sao cấu hình thứ hai.

---

## Kịch bản cài đặt trên máy Windows của team

`bundle/scripts/aic26.ps1` chạy tuần tự, mỗi bước tự bỏ qua nếu đã xong:

1. **Kiểm tra máy** — Windows build ≥ 19045, ảo hoá VT-x/AMD-V, đĩa còn ≥ 12 GB.
   Ảo hoá tắt trong BIOS là nguyên nhân lỗi hay gặp nhất và chỉ lộ ra rất muộn,
   nên nó được kiểm ngay từ đầu.
2. **Xin quyền Administrator** — chỉ khi thật sự cần cài. Tiến trình nâng quyền
   chỉ làm phần cài đặt rồi thoát; app luôn chạy dưới quyền người dùng thường.
3. **Bật WSL2** — `Enable-WindowsOptionalFeature` cho `Microsoft-Windows-Subsystem-Linux`
   và `VirtualMachinePlatform`, rồi `wsl --update` (lui về gói MSI nếu lỗi).
4. **Cài Docker Desktop** — tải trực tiếp từ desktop.docker.com,
   `install --quiet --accept-license --backend=wsl-2`. Lui về `winget` nếu mạng
   chặn (proxy công ty / mạng trường).
5. **Thêm user vào nhóm `docker-users`** — dùng tên truyền vào tường minh, vì khi
   UAC được nâng quyền bằng tài khoản khác thì `$env:USERNAME` không còn là người
   đang ngồi trước máy.
6. **Khởi động lại nếu cần** — đăng ký `RunOnce` để sau khi đăng nhập lại app tự
   chạy tiếp, người dùng không phải nhớ quay về thư mục này.
7. **Mở Docker Desktop**, chờ engine (tối đa 5 phút — lần đầu trên ổ HDD rất lâu).
8. **`docker load`** image từ `.tar.gz` đi kèm (chỉ lần đầu).
9. **`docker compose up -d`**, chờ `/api/health`, mở trình duyệt.

Vài chi tiết dễ mất công về sau:

- `.ps1` phải là **UTF-8 có BOM**: PowerShell 5.1 (mặc định của Windows) đọc file
  không BOM theo code page ANSI và làm hỏng toàn bộ dấu tiếng Việt.
  `build-bundle.sh` tự thêm BOM và đổi sang CRLF.
- `.bat` giữ **ASCII thuần**: cmd.exe làm vỡ dấu bất kể encoding. Mọi thứ có dấu
  đều nằm bên PowerShell.
- Trên Windows, container chạy `user: 0:0`. Docker Desktop chia sẻ file qua một
  lớp dịch không có uid, nên user không phải root sẽ không ghi được vào `config/`.
