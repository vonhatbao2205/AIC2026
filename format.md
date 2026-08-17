
Yêu cầu kết quả
Đối với mỗi câu truy vấn, đội thi cần nộp tương ứng một file .csv (comma-separated values file) với mỗi dòng tương ứng với một lần đội dự đoán kết quả. Đội thi có thể nộp file tối đa 100 dòng. Kết quả trên mỗi dòng của đội có format theo từng loại truy vấn:

1. Textual Known Item Search (Textual KIS)
   Format: <Tên file video>, <Frame Idx></frame>

Ví dụ:

L00_V000, 1234
L00_V055, 5555
L01_V028, 25300
2. Question Answering (Q&A)
Format: <Tên file video>, <Frame Idx></frame>, <Answer></answer>

Quy định cho Answer:

Độ dài tối đa: 100 ký tự
Có thể bằng tiếng Việt hoặc tiếng Anh
Được so sánh chính xác về mặt ngữ nghĩa với đáp án
Ví dụ:

L01_V028, 3450, "5"
L02_V011, 1200, "Năm người"
L03_V005, 2800, "Màu đỏ"
3. Temporal Retrieval and Alignment of Key Events (TRAKE)
Format: <Tên file video>, <Frame ID_1></frame>, <Frame ID_2></frame>, ..., <Frame ID_N></frame>

Trong đó:

Frame ID_1, Frame ID_2, ..., Frame ID_N là các keyframe tương ứng với N events trong chuỗi sự kiện
Số lượng Frame ID phải khớp với số events được yêu cầu trong truy vấn
Thứ tự các Frame ID phải tuân theo thứ tự thời gian của các events
Ví dụ (chuỗi 4 events):

L10_V001, 1200, 1850, 2100, 2450
L10_V001, 1180, 1820, 2080, 2420
L11_V003, 5100, 5700, 6200, 6800
Quy chuẩn định dạng CSV
⚠️ Lưu Ý QUAN TRỌNG cho học sinh THPT:

CSV ≠ Excel: Đây là hai định dạng file hoàn toàn khác nhau!

File CSV (.csv): Là file văn bản thuần túy, chỉ chứa dữ liệu được phân cách bằng dấu phẩy
File Excel (.xlsx/.xls): Là file nhị phân phức tạp của Microsoft Excel
PHẢI NỘP FILE .CSV, KHÔNG PHẢI FILE EXCEL!

Cách tạo file CSV đúng:

Từ Excel: File → Save As → chọn "CSV (Comma delimited) (*.csv)"
Từ Google Sheets: File → Download → Comma Separated Values (.csv)
Từ Notepad: Gõ trực tiếp theo format và lưu với đuôi .csv
Từ các text editor: VS Code, Sublime Text, Notepad++
Kiểm tra file CSV:

Có thể mở bằng Notepad và thấy dữ liệu dạng text thuần túy
Kích thước file nhỏ hơn nhiều so với Excel
Đuôi file phải là .csv (KHÔNG phải .xlsx hoặc .xls)
Quy tắc chung:
Encoding: UTF-8
Delimiter: Dấu phẩy (,)
Line ending: CRLF (\r\n) hoặc LF (\n)
Không có header row: File CSV bắt đầu trực tiếp bằng dữ liệu
Xử lý ký tự đặc biệt:
Lưu ý quan trọng: Dấu ngoặc kép chỉ BẮT BUỘC khi answer chứa các ký tự đặc biệt. Nếu answer đơn giản không có ký tự đặc biệt, có thể bỏ qua dấu ngoặc kép.

Dấu phẩy trong answer: BẮT BUỘC bao quanh bằng dấu ngoặc kép

   L01_V028, 3450, "Có 3 người, bao gồm nam và nữ"

Dấu ngoặc kép trong answer: BẮT BUỘC escape bằng double quotes

   L01_V028, 3450, "Anh ấy nói ""Xin chào"""

Xuống dòng trong answer: BẮT BUỘC bao quanh bằng dấu ngoặc kép

   L01_V028, 3450, "Dòng 1
   Dòng 2"

Answer đơn giản: KHÔNG BẮT BUỘC dấu ngoặc kép

   L01_V028, 3450, 5
   L02_V011, 1200, Năm người
   L03_V005, 2800, Màu đỏ

Khoảng trắng đầu/cuối: Được giữ nguyên, không tự động trim

Ví dụ CSV chuẩn cho từng loại:
Textual KIS (query-1-kis.csv):

L00_V000,1234
L00_V055,5555
L01_V028,25300
Q&A (query-2-qa.csv) - Cả hai cách đều đúng:

L01_V028,3450,5
L02_V011,1200,Năm người
L03_V005,2800,"Màu đỏ, rất đẹp"
L04_V012,4100,"Anh ấy nói ""Tuyệt vời"""
HOẶC (với dấu ngoặc kép cho tất cả):

L01_V028,3450,"5"
L02_V011,1200,"Năm người"
L03_V005,2800,"Màu đỏ, rất đẹp"
L04_V012,4100,"Anh ấy nói ""Tuyệt vời"""
TRAKE (query-3-trake.csv - 4 events):

L10_V001,1200,1850,2100,2450
L10_V001,1180,1820,2080,2420
L11_V003,5100,5700,6200,6800
Quy tắc dấu ngoặc kép trong CSV
KHÔNG cần ngoặc kép:
Answer đơn giản: 5, Năm người, Màu đỏ, Ba
Chỉ chứa chữ cái, số, khoảng trắng thông thường
Không có dấu phẩy, ngoặc kép, xuống dòng
BẮT BUỘC có ngoặc kép:
Answer có dấu phẩy: "Có 3 người, bao gồm nam và nữ"
Answer có ngoặc kép: "Anh ấy nói ""Xin chào"""
Answer có xuống dòng: "Dòng 1\nDòng 2"
An toàn nhất:
Để tránh nhầm lẫn, có thể luôn đặt dấu ngoặc kép cho tất cả answer trong Q&A. Cả hai cách đều được hệ thống chấp nhận.
Lỗi thường gặp:
Lưu nhầm file Excel: File có đuôi .xlsx/.xls thay vì .csv
Encoding sai: File hiển thị ký tự lạ khi mở bằng Notepad
Delimiter sai: Sử dụng dấu chấm phẩy (;) thay vì dấu phẩy (,)
Có header: Dòng đầu chứa tiêu đề thay vì dữ liệu
Nộp kết quả cho gói truy vấn
Mỗi đội thi cần đăng ký một tài khoản trên Codabench và đăng ký tham gia vào cuộc thi. BTC sẽ duyệt cho đội tham gia vào cuộc thi nếu tên hoặc thông tin tài khoản Codabench trùng với thông tin đội thi đã đăng ký trước đó với BTC.

Cách chuẩn bị file nộp:
Bước 1: Tạo thư mục có tên submission

Bước 2: Đặt tất cả file CSV kết quả vào trong thư mục submission

Bước 3: Nén thư mục submission thành file .zip

Bước 4 (Tùy chọn): Đổi tên file zip thành tên phù hợp (ví dụ: team_ABC_round1.zip)

Cấu trúc thư mục yêu cầu:
submission/
├── query-1-kis.csv
├── query-2-kis.csv
├── query-3-qa.csv
├── query-4-trake.csv
└── ... (các file CSV khác)
Ví dụ file nộp cuối cùng: team_ABC_round1.zip chứa:

submission/
├── query-1-kis.csv
├── query-2-kis.csv
├── query-3-qa.csv
└── query-4-trake.csv
Lưu ý quan trọng:

PHẢI có thư mục submission bên trong file zip
KHÔNG nén trực tiếp các file CSV - phải nén thư mục submission
Đội thi có thể xem cách đánh giá tại tab Evaluation để biết thêm thông tin về cách tính điểm
Tên file video không có phần đuôi (.mp4)
Frame ID sẽ được so sánh dưới dạng số nguyên
Answer (Q&A) sẽ được so sánh dưới dạng chuỗi chính xác
Answer (Q&A) có độ dài tối đa 100 ký tự
Đối với TRAKE: Số lượng Frame ID phải khớp chính xác với số events yêu cầu
Chỉ chấp nhận file nén định dạng .zip
Khuyến cáo: Tên file zip chỉ nên bao gồm các ký tự chữ hoặc số
Đánh giá và xếp hạng
Kết quả đánh giá trên Public Leaderboard chỉ tính dựa trên 50% đáp án của BTC. Kết quả cuối cùng của đội nộp sẽ được tính trên 100% đáp án và dùng để xếp hạng vòng sơ tuyển tại Private Leaderboard.

Phương pháp tính điểm:
Mỗi gói truy vấn, các đội được phép nộp kết quả tối đa 3 lần. Kết quả được dùng để xếp hạng là kết quả đội nộp lần cuối cùng.

Lưu ý cuối cùng:

Mỗi đội chỉ được dùng duy nhất một tài khoản để nộp bài
Khi nộp sai định dạng vẫn tính là 01 lần nộp
Đội cần lưu ý chọn lựa kết quả nào để nộp lần cuối cùng
Khuyến nghị kiểm tra kỹ format CSV trước khi nộp để tránh lỗi parse
📋 BẢNG TÓM TẮT - NHỮNG ĐIỀU QUAN TRỌNG NHẤT
TIÊU CHÍ	YÊU CẦU	VÍ DỤ
📁 Định dạng file nộp	✅ File .csv thuần túy
❌ KHÔNG phải Excel (.xlsx/.xls)	query-1-kis.csv
📦 Cách đóng gói	File .zip chứa tất cả file CSV	submission.zip
🎯 Format KIS	<video_name>, <frame_id>	L01_V028, 25300
❓ Format Q&A	<video_name>, <frame_id>, "<answer></answer>"	L01_V028, 3450, "5"
⏱️ Format TRAKE	<video_name>, <frame_1>, <frame_2>, ...	L10_V001, 1200, 1850, 2100
💬 Câu trả lời Q&A	Tối đa 100 ký tự, tiếng Việt/Anh	"Năm người"
📊 Số dòng tối đa	100 dòng cho mỗi file CSV	-
🎫 Số lần nộp	Tối đa 3 lần cho mỗi gói truy vấn	-
🏆 Kết quả xếp hạng	Lần nộp cuối cùng được tính điểm	-
📝 Tên file video	KHÔNG có đuôi .mp4	L01_V028 ✅
L01_V028.mp4 ❌
🔢 Frame ID	Số nguyên, không có khoảng trắng thừa	25300 ✅
25 300 ❌
Text Editor an toàn	Notepad, VS Code, Google Sheets	Excel cần Save As → CSV
🚨 5 LỖI THƯỜNG GẶP NHẤT:
LỖI	NGUYÊN NHÂN	CÁCH SỬA
🔴 File không được chấp nhận	Nộp file Excel thay vì CSV	Save As → CSV trong Excel
🔴 Thiếu thư mục submission	Nén trực tiếp file CSV thay vì thư mục	Tạo thư mục submission rồi nén
🔴 Dữ liệu hiển thị lạ	Encoding không phải UTF-8	Chọn UTF-8 khi Save
🔴 Answer bị cắt	Answer có dấu phẩy nhưng thiếu ngoặc kép	"Năm người, gồm nam và nữ"
🔴 TRAKE sai số frame	Thiếu/thừa frame cho các events	Kiểm tra đúng N events
✅ CHECKLIST TRƯỚC KHI NỘP:
 File có đuôi .csv (không phải .xlsx hay .xls)
 Mở file bằng Notepad thấy dữ liệu text thuần túy
 Tên file khớp với tên truy vấn (ví dụ: query-1-kis.csv)
 Format đúng theo loại truy vấn (KIS/Q&A/TRAKE)
 Answer Q&A không quá 100 ký tự
 TRAKE có đúng số frame theo yêu cầu
 Tên video không có đuôi .mp4
 Đã tạo thư mục submission và đặt tất cả CSV vào đó
 File được nén từ thư mục submission (không nén trực tiếp CSV)
 Đã kiểm tra số lần nộp còn lại
