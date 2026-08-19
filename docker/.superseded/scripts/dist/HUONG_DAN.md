# AIC26 Console — Hướng dẫn chạy

App tra cứu video của nhóm, đóng gói sẵn. Bạn **không cần** cài Python, Node hay
build gì cả — chỉ cần Docker.

Giải nén file zip ra một thư mục, **giữ nguyên tất cả các file cạnh nhau**.

---

## 1. Cài Docker (một lần duy nhất)

### Windows

Tải **Docker Desktop**: <https://www.docker.com/products/docker-desktop/>

Cài xong, mở Docker Desktop và đợi đến khi nó báo **Running** ở góc dưới bên
trái. (Lần đầu Windows có thể yêu cầu bật WSL2 và khởi động lại máy — làm theo
hướng dẫn hiện trên màn hình.)

### Linux

```bash
# Ubuntu / Debian / Linux Mint
sudo apt install docker.io docker-compose-v2

# Arch / Manjaro
sudo pacman -S docker docker-compose

# Fedora
sudo dnf install docker docker-compose

# Bật dịch vụ và cho phép tài khoản của bạn dùng docker không cần sudo
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

> ⚠️ Sau lệnh `usermod` phải **đăng xuất rồi đăng nhập lại** (hoặc khởi động lại
> máy) thì quyền mới có hiệu lực.

Kiểm tra: `docker info` chạy được mà không cần `sudo` là xong.

---

## 2. Chạy app

**Windows** — double-click **`run.bat`**

**Linux / macOS**

```bash
chmod +x run.sh stop.sh   # chỉ cần lần đầu, nếu giải nén bằng Windows
./run.sh
```

Lần đầu sẽ mất 1–2 phút để nạp image, các lần sau chỉ vài giây. Trình duyệt sẽ tự
mở **http://localhost:8000**.

Dừng app: double-click `stop.bat` (Windows) hoặc `./stop.sh` (Linux/macOS).

---

## 3. Nạp cấu hình (.env)

App **không kèm theo API key nào** — bạn phải tự nạp file `.env` mà nhóm gửi.

1. Lần đầu mở, app tự bật màn hình **Cấu hình · Configuration**.
   (Lần sau: bấm biểu tượng ⚙ ở góc phải thanh trên cùng.)
2. Kéo thả file `.env` vào ô, hoặc bấm để chọn file.
3. Bấm **Import & áp dụng**.

Xong. App tự kết nối lại, không cần khởi động lại container.

**Chưa có file `.env`?** Bấm **Tải .env mẫu** trong màn hình đó (hoặc mở
`.env.example` ngay trong thư mục này), điền các giá trị nhóm cung cấp, rồi import.

Các biến **bắt buộc** để tìm kiếm chạy được:

| Biến | Là gì |
|---|---|
| `ELASTIC_ENDPOINT`, `ELASTIC_API_KEY` | Elastic Cloud — OCR / speech / audio |
| `MILVUS_ENDPOINT`, `MILVUS_TOKEN` | Milvus/Zilliz — vector ảnh & âm thanh |
| `PE_ENCODER_URL` | Máy chủ encoder PE-Core-G14 (notebook Kaggle) |
| `MEDIA_BASE_URL` | Nơi chứa keyframe/video (Cloudflare R2) |

Muốn nộp bài lên DRES thì thêm `DRES_USERNAME` và `DRES_PASSWORD`.

Chỉ muốn xem thử giao diện mà chưa có key nào? Import một file `.env` chỉ chứa
`AIC26_MOCK_MODE=true` — app chạy bằng dữ liệu giả.

---

## 4. Cấu hình lưu ở đâu

| Thư mục | Nội dung |
|---|---|
| `config/.env` | File cấu hình bạn vừa import (quyền `600`, chỉ bạn đọc được) |
| `data/submit_history.json` | Lịch sử các lần nộp lên DRES |

Cả hai nằm ngoài image, nên **cập nhật app không làm mất cấu hình**: nhận bản
mới, chạy lại `./run.sh` là xong.

Bạn cũng có thể chép thẳng file `.env` vào `config/.env` bằng tay rồi bấm
**Đọc lại từ đĩa** trong màn hình Cấu hình.

---

## 5. Vướng mắc thường gặp

**Windows: `run.bat` mở ra rồi tắt ngay**
→ Docker Desktop chưa chạy. Mở Docker Desktop trước, đợi báo **Running**, rồi
double-click lại.

**Windows: SmartScreen chặn**
→ Bấm *More info* → *Run anyway*. File `.bat` chỉ là text, mở bằng Notepad xem
được toàn bộ nội dung.

**Linux: `permission denied` khi chạy `./run.sh`**
→ Giải nén bằng Windows làm mất quyền chạy: `chmod +x run.sh stop.sh`.

**`permission denied` khi chạy docker**
→ Chưa vào nhóm `docker`, hoặc chưa đăng nhập lại sau `usermod`.

**Cổng 8000 đã bị chiếm**
→ Sửa `docker-compose.yml`, đổi `"127.0.0.1:8000:8000"` thành
`"127.0.0.1:8080:8000"` rồi mở http://localhost:8080.

**Muốn người khác trong mạng LAN vào được**
→ Sửa thành `"8000:8000"`. Lưu ý: ai vào được cũng **đổi được cấu hình và nộp
bài dưới tài khoản DRES của bạn** — mặc định đóng là có lý do.

**Ô trạng thái báo `pe_encoder unreachable`**
→ Notebook Kaggle chưa chạy, hoặc URL cloudflared đã đổi. Chạy lại notebook, lấy
URL mới, cập nhật `PE_ENCODER_URL` rồi import lại. Tương tự với `NVILA_BASE_URL`
(notebook Colab).

**App vẫn báo mock mode**
→ Trong `.env` đang có `AIC26_MOCK_MODE=true`. Sửa thành `false` rồi import lại.

**Xem log lỗi**
```bash
docker compose logs -f
```

**Gỡ sạch để cài lại từ đầu** (Linux/macOS)
```bash
./stop.sh
docker rmi $(grep AIC26_IMAGE= bundle.env | cut -d= -f2)
rm -rf config data
```
Trên Windows: chạy `stop.bat`, rồi xoá image trong tab *Images* của Docker
Desktop và xoá hai thư mục `config` và `data`.
