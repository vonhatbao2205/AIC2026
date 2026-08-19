# AIC26 Console — Hướng dẫn cho team

App tra cứu video của nhóm, đóng gói sẵn thành **một thư mục chạy được ngay**.
Bạn **không cần** cài Python, Node, hay build gì cả — và cũng **không cần tự cài
Docker**: script sẽ lo luôn phần đó.

> **Bước 0 — quan trọng:** giải nén file `.zip` ra một thư mục thật (ví dụ
> `D:\AIC26`), **đừng chạy thẳng từ bên trong file zip**. Giữ nguyên tất cả các
> file cạnh nhau.
>
> Tránh để trong `Downloads` hay OneDrive/Google Drive đang đồng bộ — Docker ghi
> file liên tục vào đây, đồng bộ đám mây sẽ làm chậm và có khi hỏng dữ liệu.

---

## 1. Chạy app

### Windows

**Double-click `CHAY-APP.bat`.** Hết.

Lần đầu, script sẽ tự làm những việc sau — bạn chỉ cần bấm **Yes** khi Windows
hỏi quyền Administrator:

| | Việc | Thời gian |
|---|---|---|
| 1 | Kiểm tra Windows, ảo hoá, dung lượng đĩa | vài giây |
| 2 | Bật **WSL2** (tính năng Windows) | 1–2 phút |
| 3 | Tải và cài **Docker Desktop** (~600 MB) | 5–15 phút |
| 4 | **Khởi động lại máy** (nếu WSL2 vừa được bật lần đầu) | — |
| 5 | Mở Docker Desktop, chờ engine lên | 1–5 phút |
| 6 | Nạp image AIC26 từ file `.tar.gz` đi kèm | 1–3 phút |
| 7 | Chạy container, mở trình duyệt | vài giây |

> **Nếu máy cần khởi động lại ở bước 4:** script sẽ hỏi, bấm **Y**. Sau khi
> Windows đăng nhập lại, app **tự chạy tiếp** — không cần làm gì thêm. (Nếu vì
> lý do nào đó nó không tự chạy, cứ double-click lại `CHAY-APP.bat`.)

Những lần sau chỉ mất vài chục giây: script bỏ qua toàn bộ phần cài đặt.

Trình duyệt sẽ tự mở **<http://localhost:8000>**.

**Dừng app:** double-click `DUNG-APP.bat`.

### Linux / macOS

Docker phải tự cài (mỗi bản phân phối một kiểu, script không tự làm thay):

```bash
# Ubuntu / Debian / Mint
sudo apt install docker.io docker-compose-v2
# Arch / Manjaro
sudo pacman -S docker docker-compose
# Fedora
sudo dnf install docker docker-compose

sudo systemctl enable --now docker
sudo usermod -aG docker $USER      # rồi ĐĂNG XUẤT / ĐĂNG NHẬP LẠI
```

Sau đó:

```bash
chmod +x run.sh stop.sh    # chỉ cần lần đầu, nếu giải nén bằng Windows
./run.sh                   # dừng:  ./stop.sh
```

---

## 2. Cấu hình

Bản này **đã kèm sẵn file cấu hình của nhóm** (`config/.env`) — API key Elastic,
Milvus, PE encoder, media… App tự đọc lúc khởi động, bạn **không phải import gì
cả**.

> 🔒 File `config/.env` chứa API key thật của nhóm. Đây là bản **nội bộ**:
> đừng up lên GitHub, Drive công khai, hay gửi ra ngoài team.

Cần đổi cấu hình (ví dụ `PE_ENCODER_URL` mới sau khi chạy lại notebook Kaggle)?
Có hai cách, chọn cách nào cũng được:

- **Trong app:** bấm biểu tượng ⚙ góc phải thanh trên cùng → kéo thả file `.env`
  mới → **Import & áp dụng**. Không cần khởi động lại container.
- **Sửa file:** mở `config/.env` bằng Notepad, sửa, lưu, rồi bấm **Đọc lại từ
  đĩa** trong màn hình Cấu hình.

Các biến hay phải cập nhật giữa buổi:

| Biến | Là gì | Khi nào đổi |
|---|---|---|
| `PE_ENCODER_URL` | Encoder PE-Core-G14 (notebook Kaggle) | mỗi lần chạy lại notebook |
| `NVILA_BASE_URL` | Worker QA NVILA-8B (notebook Colab) | mỗi lần chạy lại notebook |
| `DRES_USERNAME` / `DRES_PASSWORD` | Tài khoản nộp bài | khi BTC cấp tài khoản |
| `DRES_ENABLED` | **Công tắc nộp bài** | xem ngay bên dưới |

> ⚠️ **Nộp bài lên DRES mặc định TẮT.** Có đủ user/password vẫn chưa đủ: phải
> thêm hẳn một dòng `DRES_ENABLED=true` vào `config/.env` thì nút nộp mới hoạt
> động (không có dòng này, ô trạng thái DRES luôn hiện `disabled`).
>
> Đây là chốt an toàn cố ý — để một máy đang chạy thử không lỡ tay bắn đáp án
> lên server chấm thật. Nhớ bật trước giờ thi, và kiểm tra ô trạng thái DRES
> chuyển sang xanh.

---

## 3. Đặt tên hiển thị khi nộp bài

Tab **Submission** đồng bộ real-time qua Supabase, nên cả nhóm thấy chung một
bảng đáp án. Mỗi dòng có cột tên người nộp.

Một image dùng chung cho cả team nên tên không thể gắn sẵn — mỗi máy tự đặt
**một lần duy nhất**:

1. Bấm biểu tượng ⚙ ở góc phải thanh trên cùng.
2. Ô **Tên của bạn** nằm ngay đầu màn hình — gõ tên rồi bấm **Lưu tên**
   (hoặc nhấn `Enter`).

Xong. Áp dụng ngay, không cần tải lại trang, và trình duyệt nhớ luôn cho các
lần sau.

> Không đặt tên thì các dòng bạn nộp sẽ hiện là `unknown` — cả nhóm nhìn bảng
> chung sẽ không biết ai nộp dòng nào.

## 4. Dữ liệu của bạn nằm ở đâu

| Đường dẫn | Nội dung |
|---|---|
| `config/.env` | Cấu hình / API key |
| `data/submit_history.json` | Lịch sử các lần nộp lên DRES từ máy này |

Cả hai nằm **ngoài** image, nên nhận bản cập nhật mới không làm mất gì: giải nén
bản mới, chép hai thư mục `config/` và `data/` sang, chạy lại là xong.

---

## 5. Vướng mắc thường gặp

**Cửa sổ đen hiện lên rồi tắt ngay**
→ Bạn đang chạy từ bên trong file zip. Giải nén ra thư mục thật rồi chạy lại.

**SmartScreen / Windows Defender chặn**
→ *More info* → *Run anyway*. File `.bat` và `.ps1` đều là text thuần, mở bằng
Notepad đọc được toàn bộ nội dung.

**Script báo "Ảo hoá (VT-x / AMD-V) đang TẮT trong BIOS"**
→ Khởi động lại, vào BIOS/UEFI (bấm `F2` / `DEL` / `F10` lúc máy vừa bật), bật
mục *Intel VT-x* / *Intel Virtualization Technology* / *AMD-V* / *SVM Mode*, lưu
lại. Không có bước này thì WSL2 và Docker không chạy được.

**"Docker Desktop không khởi động được trong thời gian chờ"**
→ Hầu hết là do máy chưa khởi động lại sau khi cài. Restart rồi chạy lại
`CHAY-APP.bat`. Nếu vẫn lỗi, mở Docker Desktop bằng tay để xem nó báo gì.

**Docker Desktop đòi đăng nhập tài khoản**
→ Không cần. Bấm bỏ qua / *Continue without signing in*.

**Cổng 8000 đã bị chiếm**
→ Mở `docker-compose.yml`, đổi `"127.0.0.1:8000:8000"` thành
`"127.0.0.1:8080:8000"`, chạy lại, rồi mở <http://localhost:8080>.

**Ô trạng thái báo `pe_encoder unreachable`**
→ Notebook Kaggle chưa chạy, hoặc URL cloudflared đã đổi. Chạy lại notebook, lấy
URL mới, cập nhật `PE_ENCODER_URL` (xem mục 2). Tương tự với `NVILA_BASE_URL`
(notebook Colab).

**App báo đang ở mock mode**
→ Trong `config/.env` có `AIC26_MOCK_MODE=true`. Sửa thành `false` rồi bấm
**Đọc lại từ đĩa**.

**Không thấy đáp án của người khác trong tab Submission**
→ Cả nhóm phải dùng chung một `VITE_SUBMISSION_ROOM`. Giá trị này bake trong
image, nên chỉ cần tất cả cùng chạy **cùng một bản zip** là đúng.

**Xem log lỗi**
```
docker compose logs -f
```
(chạy trong thư mục này; trên Windows mở PowerShell tại đây bằng
`Shift + chuột phải` → *Open PowerShell window here*)

**Muốn người khác trong LAN vào được**
→ Sửa `docker-compose.yml` thành `"8000:8000"`. Lưu ý: ai vào được cũng **đổi
được cấu hình và nộp bài dưới tài khoản DRES của bạn** — mặc định đóng là có lý do.

**Gỡ sạch để cài lại từ đầu**
→ Chạy `DUNG-APP.bat`, xoá image trong tab *Images* của Docker Desktop, rồi xoá
hai thư mục `config` và `data`.
