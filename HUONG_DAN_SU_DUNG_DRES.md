# DRES Simulator — Hướng dẫn sử dụng web

Hướng dẫn đầy đủ từng chức năng theo luồng sử dụng. Có **2 vai trò**: **Admin** (ra đề, điều khiển phiên) và **Team** (thi/submit).

> 💡 Muốn vừa làm admin vừa làm team cùng lúc → mở **2 trình duyệt** (hoặc 1 thường + 1 ẩn danh), vì mỗi tab chỉ đăng nhập 1 tài khoản.

## Mục lục
- [1. Đăng nhập](#1-đăng-nhập-login)
- [2. Admin Console](#2-admin-console)
- [3. Team Console](#3-team-console)
- [4. Hiểu kết quả & điểm](#4-hiểu-kết-quả--điểm)
- [5. Luồng chạy thử 1 buổi](#5-luồng-chạy-thử-1-buổi)
- [6. Nối UI truy vấn thật vào](#6-nối-ui-truy-vấn-thật-vào)

---

## 1. Đăng nhập (`/login`)
- Nhập user/mật khẩu. Seed sẵn: **admin / admin123** (admin), **team1 / team123** (team).
- Admin → vào `/admin`, team → vào `/team`.

---

## 2. ADMIN CONSOLE

### 2.1. Dashboard (`/admin`)
Tổng quan: số **Tasks**, **Ready tasks**, **Sessions**, **Active session**; khối "Live session" cho biết phiên đang chạy + task hiện tại; nút nhanh **New Task**, **Manage Sessions**.

### 2.2. Task Bank (`/admin/tasks`) — ngân hàng câu hỏi
Bảng các task: **Code · Title · Type · Status · Dur · GT (✓ nếu ground truth đủ) · Actions**.
- **New Task**: tạo task mới.
- **Import JSON**: chọn file `.json` (mảng hoặc `{tasks:[...]}`, đúng format `data/sample_tasks.json`) → nhập hàng loạt; task nào GT đủ sẽ tự thành **ready**.
- **Import CSV**: chọn file `.csv` (1 dòng = 1 task) — tiện nhập hàng loạt bằng Excel/Google Sheets. Template: [`data/sample_tasks.csv`](../data/sample_tasks.csv). Xem [quy ước cột](#format-csv) bên dưới.
- **Export JSON / Export CSV**: tải toàn bộ task ra file (import lại được).
- Mỗi dòng có: **Edit** (sửa), **Preview** (xem JSON đầy đủ), **Duplicate** (nhân bản → bản nháp), **Archive** (lưu trữ), **Delete** (xoá hẳn).

#### Format CSV

Cột (đúng thứ tự): `task_code, task_type, title, query_text, question_text, media_prompt_url, duration_seconds, difficulty, status, hints, video_id, frame_id, frame_idx, timestamp, answer_text, answer_aliases, epsilon_frames, epsilon_seconds, trake_events`

Các trường lồng nhau dùng quy ước (đều **không bắt buộc**):
- **hints**: các cặp ngăn bởi `|`, mỗi cặp `giây@nội_dung` — vd `0@Một hiện trường cháy|30@Có chữ PCCC`.
- **answer_aliases** (QA): ngăn bởi `|` — vd `15 người|mười lăm người`.
- **trake_events** (TRAKE): 1 chuỗi **JSON array** — vd `[{"event_index":1,"frame_idx":100,"timestamp":3.33}, {"event_index":2,"frame_idx":400}]`.
- Ô số để trống = bỏ qua (`duration_seconds` mặc định 180, `epsilon_frames` mặc định 15).
- Cột nào chứa dấu phẩy thì bọc trong dấu `"..."` (Excel/Sheets tự làm khi xuất CSV).
- `status` để trống cũng được — task có GT đủ sẽ tự thành **ready** khi import.

### 2.3. Task Builder (`/admin/tasks/new` hoặc `…/:id/edit`) — soạn đề
**Trường chung:** `task_code` (để trống = tự sinh qXXX), `task_type` (TKIS/VKIS/QA/TRAKE), `duration_seconds`, `status` (draft/ready/archived), `title`, `query_text`.
- Chọn **QA** → hiện ô **question_text**.
- Chọn **VKIS** → hiện ô **media_prompt_url** (link clip gợi ý).

**Hints (gợi ý theo thời gian):** thêm/xoá từng dòng `at_second` + `text`. Team chỉ thấy hint khi đã trôi đủ số giây đó.

**Ground Truth (đáp án đúng — đổi theo loại):**
- **TKIS/VKIS:** `video_id`, `frame_id`, `frame_idx`, `timestamp`, `epsilon_frames`, `epsilon_seconds`.
- **QA:** như trên + `answer_text` + danh sách **alias** (đáp án chấp nhận thêm, vd "15 người", "mười lăm người").
- **TRAKE:** `video_id`, epsilon + **danh sách sự kiện** (kéo ⠿ để sắp thứ tự; mỗi sự kiện: description, frame_id, frame_idx, timestamp). Có cảnh báo nếu thứ tự không tăng dần.

**Lưu:**
- Lưu **draft**: thoải mái, chưa cần đủ GT.
- Lưu **ready**: phải đủ GT, nếu thiếu sẽ báo lỗi rõ. Chỉ task **ready** mới được đưa vào phiên & kích hoạt.

**Test Submit Against Ground Truth** (chỉ hiện khi đang sửa task đã lưu): nhập 1 payload giả → **Run test judge** → xem ngay **correct/wrong/partial/invalid** + chi tiết, **không** lưu vào lịch sử. Dùng để kiểm GT trước khi thi.

### 2.4. Sessions (`/admin/sessions`) — quản lý phiên
- Tạo phiên: chọn **contest** + nhập **tên** → **Create session**. Chưa có contest thì gõ tên rồi **+ Add contest**.
- Bảng phiên: tên, trạng thái, số task, thời điểm bắt đầu, **Open**.

### 2.5. Session Detail (`/admin/sessions/:id`) — điều khiển phiên (màn hình "trọng tài")
- **Trạng thái** + **task hiện tại**.
- **Nút điều khiển:** **Start** (bắt đầu — tự kích hoạt task đầu), **Pause** (tạm dừng), **Next task →** (sang task kế), **End** (kết thúc).
- **Danh sách task:** kéo **⠿** để đổi thứ tự; mỗi dòng có **Activate** (kích hoạt làm task hiện tại) và **Remove** (gỡ khỏi phiên, không xoá khỏi Task Bank). Task đang chạy được tô xanh.
- **Add a ready task…:** chọn task ready chưa có trong phiên → **Add**.
- **Scoreboard** + **Submissions** (cập nhật ~2.5s/lần, thấy bài nộp của mọi team realtime).

### 2.6. Scoreboard (`/admin/scoreboard`)
Bảng xếp hạng phiên đang chạy: hạng, team, tổng điểm, số ✓/✗/~/!

### 2.7. Submissions (`/admin/submissions`)
Toàn bộ bài nộp của phiên đang chạy + **Export submissions.csv** và **Export event-log.json** (tải log để phân tích sau).

---

## 3. TEAM CONSOLE

> Chỗ này để test submit. UI truy vấn thật của nhóm nên gọi API trực tiếp (xem mục 6).

### 3.1. Home (`/team`)
Tóm tắt phiên đang chạy + task hiện tại + nút sang **Current Task**.

### 3.2. Current Task (`/team/current`) — màn hình chính khi thi
- **Thông tin task:** loại, mã, tiêu đề, `query_text`; câu hỏi (QA); link clip (VKIS); **đồng hồ đếm ngược** + giờ server; **hints** đang hiện.
- **Form nộp (theo loại):**
  - TKIS/VKIS: `video_id`, `frame_id`, `frame_idx`, `timestamp`.
  - QA: thêm `answer`.
  - TRAKE: `video_id` + các dòng **event** (nút **+ event**), mỗi event nhập frame_id/frame_idx/timestamp.
- **Kết quả sau khi nộp:** badge trạng thái, **accuracy**, **score Δ** (điểm cộng), **time bonus**, **wrong count**, message, và **detail JSON** (bung ra xem chấm chi tiết).
- **Lịch sử bài nộp của task hiện tại** + **scoreboard** bên phải.
- Form bị khoá nếu phiên **không** ở trạng thái running.

### 3.3. History (`/team/history`)
Toàn bộ bài nộp của team bạn (mọi task trong phiên).

### 3.4. Scoreboard (`/team/scoreboard`)
Bảng xếp hạng.

---

## 4. Hiểu kết quả & điểm

| Trạng thái | Ý nghĩa |
|---|---|
| **correct** ✓ | Đúng video + frame trong epsilon (và đúng đáp án nếu QA) → **giải xong task**, được điểm. |
| **wrong** ✗ | Sai → **tăng số lần sai**; mỗi lần sai làm điểm về sau của task đó **−100**. |
| **partial** ~ | (Chỉ TRAKE) đúng ≥50% sự kiện → điểm theo accuracy; vẫn nộp tiếp được, **giữ điểm cao nhất**. |
| **invalid** ! | Thiếu field / TRAKE sai thứ tự thời gian / **trùng bài** / **đã giải xong rồi**. |

- Công thức: **điểm = max(0, 500 − 100×số_lần_sai + time_bonus)**, với **time_bonus = 500×(thời_gian_còn_lại / thời_lượng)**.
  → Bài học chiến lược: **nộp sai tốn hơn nộp chậm rất nhiều** → ưu tiên chắc chắn rồi mới nộp.
- **Nộp trùng y hệt:** bị từ chối "Duplicate" nhưng **không bị phạt**.
- **Đã đúng rồi nộp lại:** bị chặn "Task already solved".

---

## 5. Luồng chạy thử 1 buổi (gợi ý làm theo thứ tự)
1. **Admin**: login → Task Bank (đã có sẵn q001–q004 từ seed) → mở 1 task bấm **Test Submit** thử cho quen.
2. **Sessions** → mở **"Training Session 1"** (seed sẵn, đã có 4 task) → **Open**.
3. Bấm **Start** (task q001 tự kích hoạt). Dùng **Activate**/**Next** để chuyển câu.
4. **Cửa sổ khác**: login **team1** → `/team/current` → nộp thử:
   - q001 (TKIS): `video_id=L01_V001`, `frame_idx=123` → **correct**.
   - Thử sai 1 lần xem **wrong count** tăng, rồi nộp đúng xem điểm bị trừ.
5. **Admin** xem **Submissions** + **Scoreboard** nhảy realtime.
6. **Next task** → lặp lại. Xong thì **End** + **Export CSV / event-log**.

---

## 6. Nối UI truy vấn thật vào
Không cần Team Console — gọi thẳng API:
- `import` lại [`frontend/src/lib/dresClient.ts`](../frontend/src/lib/dresClient.ts), hoặc
- `POST /api/submissions` kèm header `X-API-Key: team1-demo-key` (hoặc JWT).

Chi tiết API + ví dụ curl: xem mục 8–9 trong [README.md](../README.md).
