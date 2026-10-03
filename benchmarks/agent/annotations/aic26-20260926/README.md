# Ground truth AIC26 · 26/09/2026

Bộ nhãn ghép `AIC26 - Query.xlsx` với lịch sử DRES ngày 26/9 theo thứ tự và
nội dung, sau đó xem trực tiếp các frame video tại mốc submit và đoạn ngắn xung quanh.
32 dòng workbook thành **30 câu khác nhau** vì dòng 21–23 lặp cùng câu Sagrada Familia.
Kết quả: **15 T-KIS, 12 QA, 3 TRAKE**; không còn câu thiếu nhãn trong workbook này.

| Nguồn nhãn | Số câu | Cách chọn |
|---|---:|---|
| `dres_correct` | 22 | Giữ CORRECT gốc; kiểm tra frame/clip khớp mô tả |
| `owner_confirmed_btc_label_error` | 7 | Chủ dữ liệu xác nhận BTC nhầm đáp án; chỉ nhận những lần WRONG đã đối chiếu |
| `pe_retrieval_visual_annotation` | 1 | Câu đua xe thiếu submit; tìm bằng PE trong S01 và annotate trên video theo yêu cầu chủ dữ liệu |

Mở [bảng Excel](ground_truth_review.xlsx) để xem query, video, timestamp, frame,
answer và ghi chú. [Gallery HTML](review.html) hiển thị ảnh tại mốc và clip của cả
candidate được chọn/lẫn bị loại. Ảnh, clip và PE search responses nằm trong
`evidence/` local, bị git-ignore; [manifest](evidence_manifest.json) giữ SHA-256.
Các file JSON/JSONL là nguồn dữ liệu; HTML/Excel chỉ là bản trình bày.

## Câu TRAKE đua xe · dòng 17

PE tìm được cảnh gốc trong **S01-V011**, thay vì đoạn phát lại quanh 14778.633333s.
Video gốc có 30 FPS, đối chiếu cả ffprobe lẫn metadata keyframe. Frame numbering
theo `frame_idx` của hệ thống, bắt đầu từ 0. Bốn mốc:

| Event | Timestamp (s) | HH:MM:SS | Frame | Tiêu chí |
|---|---:|---|---:|---|
| E1 | 9355.800000 | 02:35:55.800 | 280674 | Tay đua #97 mở lòng bàn tay và năm ngón về camera |
| E2 | 9364.400000 | 02:36:04.400 | 280932 | Hai bàn tay vừa tiếp xúc ở phần dưới khi ghép tim |
| E3 | 9375.433333 | 02:36:15.433 | 281263 | Mép sau bánh xe tay đua xanh vượt mép trước bánh xe #97 |
| E4 | 9398.233333 | 02:36:38.233 | 281947 | Bảng tên trường xanh vừa ra hết mép trái ảnh |

Xem `evidence/row-17/overview.jpg` và các frame native trước/sau mốc.
E2–E4 được xem từng frame ở 30 FPS quanh ranh giới; E1 chọn frame bàn tay mở rõ,
không yêu cầu frame đầu tiên mở tay. Bảng trường thấy chữ `TRƯỜNG CAO ĐẲNG`;
tên riêng bị nhòe nên không tự ghi tên trường. E3 dùng tiêu chí vượt hết cả xe,
ghi rõ để người review biết cách hiểu “vượt hoàn toàn”. Chuyển động nhòe và góc quay
có thể làm ranh giới lệch vài frame. Đây là annotation hình ảnh, **không có verdict
DRES hay khoảng chấp nhận chính thức**. Benchmark dùng tolerance thời gian đã công bố
(mặc định 1 giây), không tự mở rộng khoảng nhãn.

S01 là phạm vi **tìm nhãn** được chủ dữ liệu cung cấp. Dataset hiện không đặt
scope S01 riêng cho câu này; benchmark vẫn tìm toàn corpus cùng cấu hình mặc định.
Similarity PE chỉ dùng tìm candidate; không dùng làm xác suất nhãn đúng.

## Những WRONG-only cần phân biệt các lần submit

| Dòng | Chọn sau khi đối chiếu | Loại / ghi chú |
|---|---|---|
| 13 | S01-V004 tại 3395.9 / 3399.1s: ba nhóm đua xe trong cảnh aerial | Sau audit bỏ 3390.7s chưa thấy đoàn đông; loại 6477.133s là đoạn khác |
| 15 | M08_V016: `ATTAPEU` | Cảnh quà cho học sinh; lower third và speech địa phương hỗ trợ province |
| 24 | L23_V004: tốc độ lớn nhất R.Maikin `46` | Xem thêm clip 85–106s; loại L23_V024/`61` không khớp nhóm đua; giữ answer số |
| 28 | M02_V007: `SADEC` / `LANGHOASADEC` | Cảnh cờ/trái tim trên bè hoa và speech địa phương khớp Sa Đéc |
| 29 | L26_V234: `4` cánh gà | Ingredient card ghi 4 cái; `8` là sau khi chặt, không phải số cánh nguyên liệu |
| 30 | M09_V014: `LONGTHUY` / `LANGCHAILONGTHUY` | Loại video M09_V003 không đúng cảnh giáo viên chuẩn bị dụng cụ |
| 31 | L26_V104: 74.72 → 107.72 → 119 → 134.32s | Bốn pha nấu món chôm chôm/tôm khớp; giữ mốc submit, chưa annotate lại ranh giới động tác |

CORRECT/WRONG gốc luôn được giữ trong `original_dres_verdicts`, không sửa history.
Lỗi DRES/duplicate HTTP 412 không trở thành đáp án. Các câu V-KIS và `kis-1` thiếu
mô tả tương ứng trong workbook không được ghép vào benchmark này. Dòng 17 đã loại
ghép nhầm theo vị trí với V-KIS M01_V006: frame đó là cảnh hoạt hình, không đua xe.

### Audit lại dòng 13 theo bốn hint

[Audit JSON](query13_audit.json) ghi từng hint và các ảnh bằng chứng.
Đoạn S01-V004 quanh **00:56:35.900** khớp nội dung tiếng Việt:
cảnh trên cao mở rộng từ 5 người dẫn đầu tới cặp 2 người và đoàn đông;
ASR tại 3384.619–3409.581s nói khoảng cách dưới 10 giây và khoảng 15 giây;
khi chuyển sang máy quay phía trước khoảng 3407s, thấy một tay đua áo đỏ vươn lên
từ đoàn đông khoảng 3411–3418s. Không tự gán tên tay đua.

3390.7s còn quá sớm và chưa thấy đoàn đông nên đã bỏ khỏi targets.
6477.133s là đoạn khác; transcript gần đó nói khoảng cách **hơn một phút**,
không khớp hint 3. Hai mốc giữ lại là **3395.9s (frame 101877)** và
**3399.1s (frame 101973)**. Đuôi đoàn đông vẫn chạm mép khung hình:
audit xác nhận ba tốp ở cấp đoạn phim, không khẳng định mọi tay đua trong toàn bộ
đoàn nằm trọn trong một frame. Các mốc là seed của đoạn khớp, không phải nhãn
chứng minh riêng lẻ cả bốn hint ở đúng một ảnh.

Bản tiếng Anh trong workbook dịch lệch hai chi tiết: `ahead` phải là **phía sau
tốp đầu**; `from the top three` phải là **từ tốp ba / đoàn đông**. Giữ nguyên
workbook gốc, dùng query tiếng Việt đã lưu để benchmark. Clip có audio và frames
audit nằm trong `evidence/row-13/extra-four-hints-audit/`; transcript chỉ là ASR
hỗ trợ đối chiếu, không phải bản chép tay đã xác minh từng chữ.

## Xuất và ghép vào benchmark

Từ repo root:

```bash
.venv/bin/python -m benchmarks.agent.export_submit_dataset \
  --output benchmarks/agent/runs/aic26-20260926-v3-audited.jsonl --verify-media

.venv/bin/python -m benchmarks.agent.export_dataset \
  --additional-dataset benchmarks/agent/annotations/aic26-20260926/ground_truth.jsonl \
  --group-video-splits \
  --output benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl

.venv/bin/python -m benchmarks.agent.annotations.aic26-20260926.build_review
```

Trên workspace hiện đã tạo các file trên. Bộ riêng 30 câu có 3 dev / 27 test.
Bộ ghép workbook cũ + bộ này có **111 câu: 25 dev / 86 test**, sau khi nhóm video
và giữ các câu đã dùng cho dev ở dev. `final_split_audit` không còn target video
nằm ở cả dev/test; near-duplicate mô tả vẫn cần kiểm tra nếu dùng kết quả cho paper.
Lệnh merge không ghi đè file đã tồn tại; khi chạy lại chọn tên phiên bản khác.
Đổi nhãn cần review/hash mới và output benchmark mới, không resume run cũ.

`review_queue.json` giữ workbook text và sanitized submissions; `source_manifest.json`
giữ hash workbook/history; `review_decisions.json` ghi từng lựa chọn và authorization;
`evidence_index.json` ghi các mốc đã decode. Exporter từ chối review stale, timestamp
không khớp evidence, WRONG override có CORRECT, lỗi DRES, hoặc TRAKE thiếu event.
Ground truth không được gửi trong request tới controller/model.

Các đáp án là **mốc đã submit / đã annotate**, chưa phải toàn bộ khoảng đáp án
chính thức. Những câu có DRES CORRECT và brief review không được quảng bá là đã
kiểm tra chính xác mọi ranh giới first-frame. Chưa chạy benchmark với bộ mới,
chưa gửi submit DRES và chưa gọi model judge để tạo nhãn.
