# Hướng dẫn AI: xác minh GT video và tạo ba hint cho PHM

## Nhiệm vụ và phạm vi

Đọc và đối chiếu **cả hai workbook** ở thư mục gốc repository:

- [TKIS_queries.xlsx](../TKIS_queries.xlsx): query gốc và các frame GT để tra cứu.
- [TKIS_72_queries_PHM_3_hints.xlsx](../TKIS_72_queries_PHM_3_hints.xlsx): bộ 72 query có ba delta hint, ba cumulative view và các trường GT ban đầu. Đường dẫn tuyệt đối: `/home/bao/Projects/AIC2026/TKIS_72_queries_PHM_3_hints.xlsx`.

Tìm video tương ứng, xem nội dung để xác minh đáp án, tạo ground truth theo khoảng **`start_ms` / `end_ms`**, rồi kiểm chứng, chỉnh sửa hoặc viết lại ba hint tiết lộ dần từ nội dung thực sự quan sát được. Đây là công việc annotation; **chưa chạy benchmark, chưa chọn tham số PHM và chưa đo gain**.

Không coi việc đọc bảng, xem thumbnail hoặc hỏi VLM một lần là đã kiểm chứng video. Nếu không truy cập/giải mã được media, đánh dấu `unverified` và ghi chính xác phần còn thiếu. Không điền khoảng thời gian hoặc tình tiết bằng suy đoán. Không sửa workbook nguồn.

Các đường dẫn trong hướng dẫn này tính từ root repository. Kiểm tra `AGENTS.md` hiện hành trước khi làm. Dùng môi trường Python có `openpyxl`; đọc workbook với `read_only=True, data_only=True`. Không đọc hay in thông tin đăng nhập từ `.env`, `API_KEY/`, `VBS_Account.txt`; sử dụng media builder/registry và cấu hình sẵn của ứng dụng.

## 1. Kiểm kê nguồn, không kế thừa mẫu số cũ

Trong `TKIS_queries.xlsx`, lần kiểm tra ngày 20/09/2026 thấy:

| Sheet               | Header  | Cột cần đọc                                                                                                                 |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `TKIS tổng hợp` | Hàng 5 | STT, Đợt, Query ID, Nhóm, Nội dung TKIS, Ground truth (nguyên bản), Video ID, Frame ID, Số GT, DBC, Trạng thái, Recall |
| `GT từng frame`  | Hàng 3 | Đợt, Query ID, Nhóm, Video ID, Frame ID, DBC, Trạng thái, Nội dung TKIS                                                   |

Đây là vị trí quan sát được, không phải schema bảo đảm vĩnh viễn: xác nhận header bằng tên cột khi chạy. Sheet tổng hợp có 92 hàng tính cả tiêu đề, sheet frame có 296 hàng. Số đếm trên hàng tổng kết Excel không thay cho audit dữ liệu.

1. Ghi SHA-256 workbook, tên sheet, hàng nguồn, ngày annotation, phiên bản công cụ và commit code. Giữ nguyên chuỗi query và GT gốc trong sidecar.
2. Join hai sheet bằng `Đợt + Query ID`; kiểm tra xung đột nội dung/ID thay vì join theo số thứ tự. Một query có thể có nhiều frame và nhiều video đáp án.
3. Giữ ID dưới dạng string, gồm cả số 0 đầu. Nếu Excel đã lưu ID thành số làm mất số 0, đối chiếu registry rồi ghi phép sửa; không tự pad theo phỏng đoán.
4. Dòng thiếu Query ID được cấp ID ổn định từ sheet/hàng nguồn; để `source_query_id=null`. Không giả vờ ID sinh mới là ID chính thức.
5. Dòng `Số GT=0`, `DBC=0`, thiếu video/frame hoặc query khác TKIS vẫn vào danh sách audit với lý do; không lặng lẽ bỏ. Các nhãn `Recall`/`DBC` cũ không được dùng để chọn query mà PHM làm tốt.
6. Không dùng sẵn mẫu số 67, 72 hoặc split 45/22. Đọc toàn bộ workbook `TKIS_72_queries_PHM_3_hints.xlsx` theo quy trình dưới đây; 72 query đầu vào không đồng nghĩa 72 nhiệm vụ độc lập đã kiểm chứng.

### Đối chiếu workbook 72 query có hint sẵn

Sheet **`72 queries`**, header ở **hàng 1**, có các cột:

| Nhóm | Tên cột |
| --- | --- |
| Query nguồn | `STT`, `Đợt`, `Query ID`, `Nhóm`, `Nội dung TKIS` |
| GT ban đầu | `Ground truth (nguyên bản)`, `Video ID`, `Frame ID`, `start,end`, `ms` |
| Định danh PHM | `PHM Query ID` |
| Hint độc lập | `Hint 1 (delta)`, `Hint 2 (delta)`, `Hint 3 (delta)` |
| Mô tả tích lũy | `Turn 1 cumulative`, `Turn 2 cumulative`, `Turn 3 cumulative` |
| Nguồn hint | `Hint source` |

1. Kiểm kê tất cả dòng của sheet, ghi SHA-256 và hàng nguồn riêng cho workbook này. Nếu `openpyxl` trả `max_row=None` ở read-only mode, đếm bằng cách duyệt `iter_rows(values_only=True)`; không kết luận sheet rỗng.
2. Khi có `Query ID`, đối chiếu với workbook gốc bằng `Đợt + Query ID`, kiểm tra tiếp nội dung query và video. Giữ `PHM Query ID` làm định danh của bản progressive; không ghi đè ID nguồn bằng ID PHM.
3. Khi thiếu `Query ID`, dùng nội dung query, đợt và video để đề xuất liên kết nguồn. Ghi `source_match_status` là `confirmed`, `ambiguous` hoặc `unmatched`; chưa xác nhận thì giữ `source_query_id=null`. Không tự ghép chỉ vì cùng video hoặc cùng số thứ tự hàng.
4. Giữ nguyên `Frame ID`, `start,end` và `ms` trong sidecar. Các khoảng có sẵn chỉ là **GT đề xuất để mở video**: kiểm tra đơn vị, PTS, nội dung và hai biên theo mục 2–3. Không mặc định `start,end` là giây hay ms; tên cột `ms` cũng không thay cho kiểm tra giá trị với duration/video.
5. Giữ bản gốc của cả ba delta hint, ba cumulative view và `Hint source` trước khi sửa. Nguồn `Synthetic rewrite từ full query` không chứng minh đã xem video. Kiểm tra từng mệnh đề với bằng chứng; giữ hint phù hợp, sửa/viết lại hint sai hoặc thiếu bằng chứng theo mục 4, lưu rõ thay đổi và lý do.
6. Kiểm tra cumulative có đúng phép nối các delta hay không. Sau khi sửa delta, sinh lại cumulative từ delta đã duyệt; không giữ cumulative cũ chứa chi tiết đã loại bỏ.
7. Không cộng số dòng của hai workbook thành số nhiệm vụ. Query xuất hiện ở cả hai file tạo một annotation liên kết cả hai nguồn. Dòng không liên kết được vẫn phải có trạng thái xử lý trong inventory, không bị bỏ im lặng.

Đầu ra kiểm kê: `benchmarks/progressive/annotations/source_inventory.jsonl` và `audit_report.md`, có số query nguồn, query thiếu GT, xung đột, trùng lặp, media không tìm thấy.

## 2. Tìm đúng media và hiểu hệ tọa độ thời gian

1. Tra video qua metadata/manifest và URL builder của profile trong backend. Ghi `dataset_id`, `media_revision`, `video_id`, đường dẫn/URL media không chứa token, duration và checksum nếu khả thi. Không dựng đường dẫn V3C theo quy tắc tên AIC.
2. Xác định `Frame ID` là **absolute frame index**, keyframe ordinal, hay ID khác bằng metadata và loader thực tế. Cột frame chỉ là seed để mở video; không mặc định nó chính là đáp án cuối.
3. Dùng `ffprobe` để đọc duration, time base, frame rate và PTS. Chỉ dùng `frame_index / fps` khi đã xác minh video CFR và quy ước index 0/1-based. Với VFR, dùng PTS của frame thực tế. Không mặc định 25 fps; không đổi `25815` thành milliseconds.
4. Nếu file đã remux/cắt hoặc timeline không bắt đầu tại 0, ghi phép ánh xạ PTS sang media playback time. Mọi khoảng GT đầu ra dùng **milliseconds tương đối với đầu video được nộp**.
5. Đối chiếu ảnh keyframe với frame decode tại thời điểm seed để phát hiện sai video, lệch FPS, lệch index hoặc revision media. Chưa giải quyết được thì `needs_review`, không gắn `verified`.

Ví dụ lệnh đọc metadata (thay bằng đường dẫn media đã xác minh, luôn quote shell):

```sh
ffprobe -v error -show_entries format=duration:stream=index,codec_type,time_base,start_time,avg_frame_rate,r_frame_rate -of json '/path/to/video.mp4'
```

Metadata chỉ chứng minh cách quy đổi thời gian; chưa chứng minh nội dung đáp án.

## 3. Xem video và xác minh start/end

Với mỗi query:

1. Đọc nguyên văn query; tách các mệnh đề cần xác minh. Mở tất cả vùng seed, bắt đầu bằng clip lân cận khoảng ±15 giây, mở rộng khi bối cảnh/cảnh còn tiếp diễn. Nếu seed không đúng, tìm tiếp trong video và ghi lại cách tìm.
2. Xem chuyển động và các cut thực tế, không chỉ contact sheet. Có thể tạo contact sheet để định hướng rồi decode/xem clip đầy đủ để xác nhận. Nghe âm thanh khi query phụ thuộc lời nói; OCR/ASR chỉ là gợi ý cần đối chiếu với video.
3. Ghi từng mệnh đề với bằng chứng: thời điểm, frame/clip lưu cục bộ, modality `visual`/`ocr`/`speech`/`audio`, mô tả quan sát. Không suy nghề nghiệp, địa điểm hoặc tên nhân vật chỉ từ ngoại hình.
4. Xác định đoạn mục tiêu là **khoảng mà operator có thể chọn một moment đáp ứng query**. Chọn start là thời điểm bắt đầu điều kiện đáp án, end là thời điểm kết thúc điều kiện đó. Kiểm tra frame ngay trước/sau hai ranh giới. Không mặc định lấy `[min(seed), max(seed)]`, padding ±5 giây, toàn bản tin hay toàn shot.
5. Ghi độ bất định của biên, ví dụ `boundary_uncertainty_ms=200`, và lý do. Nếu chỉ xác định được điểm, giữ `target_points_ms` và để intervals rỗng; không tạo khoảng giả để đủ cột.
6. Giữ nhiều interval rời nhau nếu đều hợp lệ. Không nối chúng thành một khoảng bao trùm cả nội dung không liên quan. Với nhiều video đáp án hợp lệ, giữ nhiều đối tượng `targets`.
7. Phân loại trước khi xem kết quả retrieval: `single_moment` hoặc `multi_shot_context`. Query cần bối cảnh nhiều shot phải lưu riêng **context interval** và **answer interval**; bằng chứng rải khắp video không tự biến mọi frame thành đáp án hợp lệ.
8. Nếu mệnh đề nguồn sai/mâu thuẫn hoặc không quan sát được, lưu nguyên văn và ghi vấn đề. Có thể tạo một bản query sửa dựa trên video, nhưng phải đánh dấu `query_revision`, liên kết query nguồn, ghi lý do và chờ review; không âm thầm thay câu hỏi để khớp video.

Một AI có công cụ thị giác được phép đánh dấu `video_verified` chỉ khi nó thực sự kiểm tra các clip/frame liên tục liên quan và các biên, đồng thời lưu artifact bằng chứng. `review_status` độc lập, mặc định `pending`; chỉ ghi `approved` khi đã có người/reviewer thứ hai duyệt. Nếu môi trường không xem được clip, ghi hạn chế về sampling và yêu cầu review, không tuyên bố đã xem trọn video.

## 4. Tạo hint 1, 2, 3 từ bằng chứng đã xác minh

Tạo **ba delta hint** rồi sinh cumulative bằng phép nối chính xác:

```text
H1 = delta_1
H2 = delta_1 + " " + delta_2
H3 = delta_1 + " " + delta_2 + " " + delta_3
```

- H1: bối cảnh tổng quát nhưng có giá trị tìm kiếm, gồm chủ thể và môi trường quan sát được.
- H2: thêm hành động, quan hệ hoặc bố cục giúp phân biệt hơn.
- H3: thêm chi tiết đặc trưng đã được kiểm chứng (vật thể, màu, trình tự, chữ hiển thị hoặc lời nói).
- Mỗi delta bổ sung thông tin thực sự; không chỉ paraphrase hoặc lặp phiếu H1. Nên viết đủ chủ thể để delta có thể truy hồi độc lập; cumulative cung cấp ngữ cảnh bổ sung.
- Cả ba phải hướng tới cùng đáp án đã xác minh. Mỗi mệnh đề liên kết `evidence_id`; kiểm tra các chi tiết có cùng tồn tại trong answer interval hay chỉ xuất hiện ở context interval.
- Không chứa video ID, frame ID, timestamp đáp án, thứ hạng retriever hay câu như “video đúng là...”. Không giấu thông tin phủ định làm H2/H3 mâu thuẫn H1.
- Không bịa một chi tiết “càng ngày càng cụ thể” nếu video không hỗ trợ. Không đủ ba nhóm chi tiết độc lập thì đánh dấu `insufficient_progressive_evidence`, giữ query ngoài tập 3-hint đã duyệt và giải thích.
- Dùng tiếng Việt như workbook trừ khi có yêu cầu khác. Nếu cần tiếng Anh, lưu cả bản gốc và bản dịch đã review; khóa bản dịch một lần trước thực nghiệm.
- Ghi `hint_source="synthetic_video_verified"` khi đã xem video. Đây vẫn là **synthetic progressive descriptions**, không phải hint VBS thật, log reveal thực hay user study. Không tự đặt thời gian reveal thành dữ kiện lịch sử.
- Mức khó tăng dần ở đây là chủ ý annotation; không chứng minh H1 khó hơn H2 bằng trực giác hoặc kết quả một model. Không sửa hint sau khi thấy PHM thua baseline.

Ví dụ cấu trúc, **không phải annotation của workbook**: H1 “Một người đang làm việc ở bàn bếp.” H2 “Người đó dùng dao cắt rau trên thớt.” H3 “Bên trái thớt có một chiếc bát màu vàng.” Chỉ dùng kiểu mô tả này khi cả ba chi tiết đã có bằng chứng thực tế.

## 5. Schema đầu ra

Ghi một JSON object/query vào `ground_truth.jsonl`. Mẫu sau là template, các giá trị `null` phải được điền từ quan sát hoặc giữ null với trạng thái chưa xác minh:

```json
{
  "query_id": "stable-annotation-id",
  "source_query_id": null,
  "source": {"workbook": "TKIS_queries.xlsx", "sha256": null, "sheet": "TKIS tổng hợp", "row": null},
  "original_query": null,
  "query_revision": 0,
  "dataset_id": null,
  "query_temporal_type": null,
  "time_unit": "ms",
  "interval_convention": "start_inclusive_end_exclusive",
  "targets": [{
    "video_id": null,
    "media_revision": null,
    "duration_ms": null,
    "source_frame_ids": [],
    "frame_id_kind": null,
    "temporal_mapping": {"source": null, "index_base": null, "fps": null},
    "target_points_ms": [],
    "target_intervals": [],
    "context_intervals": []
  }],
  "evidence": [],
  "hint_source": null,
  "hints": [],
  "annotation_status": "unverified",
  "review_status": "pending",
  "annotator": null,
  "reviewer": null,
  "duplicate_group_id": null,
  "issues": [],
  "exclusion_reason": null
}
```

Một `target_intervals` item có `start_ms`, `end_ms`, `boundary_uncertainty_ms`, `evidence_ids`, `justification`. Một evidence item có `evidence_id`, `video_id`, `start_ms`, `end_ms`, `modality`, `observation`, `artifact_path`, `inspection_method`. Một hint item có `hint_id`, `delta_text`, `cumulative_text`, `evidence_ids`, `source`, `revealed_at_ms=null`.

Với query lấy từ workbook 72 dòng, bổ sung `phm_query_id`, `source_match_status` và `source_records[]`. Mỗi source record chứa workbook, SHA-256, sheet, row và các giá trị gốc liên quan; giữ cả hai record nếu đã liên kết với workbook gốc. Lưu `original_hints` gồm ba delta, ba cumulative và `Hint source`, cùng `hint_revision` và `hint_changes[]` để truy được từ hint có sẵn đến hint đã kiểm chứng. Trường `source` trong template chỉ nguồn chính, không thay thế danh sách provenance này.

Tạo thêm `ground_truth_review.xlsx` để người duyệt dễ xem, với sheet `Queries`, `Targets`, `Hints`, `Evidence`, `Issues`. Không nhét nhiều interval vào một ô không có cấu trúc. File XLSX là bản trình bày; JSONL là nguồn máy đọc. Mọi artifact được đặt dưới `benchmarks/progressive/annotations/`, không overwrite workbook gốc.

## 6. Kiểm tra và bàn giao

- Validate `0 <= start_ms < end_ms <= duration_ms`; số đo là ms, không trộn giây/frame. ID giữ string. Kiểm tra từng liên kết evidence/hint/target.
- Cumulative phải bằng phép nối các delta và không chứa nội dung của hint tương lai. Không gọi parser/encoder hoặc tạo cache retrieval với cả H3 khi mô phỏng H1 sau này.
- Kiểm tra mệnh đề nào chưa có evidence, biên quá rộng, nhiều shot không liên tục và nguồn frame không khớp. Sửa từ việc xem video, không từ thứ hạng tìm kiếm.
- Dedup exact query, paraphrase và target tương đương. Không xóa dấu vết: ghi duplicate group, bản đại diện và lý do. Sau này split theo connected group của target video/duplicate; mọi paraphrase và prefix cùng split. Không tạo split evaluation trước khi audit xong.
- Tổng kết rõ: số query nguồn, đủ 3 hint, video verified, approved, needs review, unverified, duplicate và excluded (các loại có thể chồng lấp; giải thích mẫu số). Không báo tổng query nguồn như tổng query đã duyệt.
- Ghi `annotation_decisions.jsonl` append-only: ai, lúc nào, từ trạng thái nào sang trạng thái nào, sửa gì, lý do và bằng chứng. Lưu ảnh/clip biên với tên liên kết query; tránh sao chép toàn bộ corpus nếu không cần.
- Bàn giao `audit_report.md` với danh sách việc cần người duyệt, URL/đường dẫn video thiếu và lỗi GT nguồn. Không tạo bảng benchmark hoặc claim PHM hiệu quả trong công việc này.

**Điều kiện hoàn tất annotation:** mọi query nguồn đều có disposition, mỗi GT được đánh dấu video verified có bằng chứng kiểm tra biên, mỗi hint có liên kết bằng chứng, và phần chưa kiểm chứng được ghi rõ để reviewer xử lý. Nếu media chưa sẵn sàng, hoàn tất inventory/schema/danh sách thiếu; không thay việc xem video bằng suy đoán.
