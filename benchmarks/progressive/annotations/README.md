# GT chuẩn hóa L21–L30

Chủ dữ liệu đã xác nhận bộ này chuẩn hóa từ GT cũ và được dùng cho benchmark. [ground_truth.jsonl](ground_truth.jsonl) là nguồn máy đọc; [ground_truth_review.xlsx](ground_truth_review.xlsx) là bản dễ xem. `target_intervals` chứa GT được chấp nhận theo milliseconds, `candidate_intervals` giữ provenance đề xuất ban đầu; context không phải answer.

**77 record owner-approved, 90 answer intervals; 75 query đủ điều kiện benchmark TKIS.** Một record TRAKE và bản trùng `synthetic-tkis-033` không vào mẫu số TKIS; bốn query chưa có video không đủ điều kiện. Lọc `eligible_for_benchmark=true`, không đếm tổng 81 record hoặc tất cả prefix như nhiệm vụ độc lập.

[owner_approval.json](owner_approval.json) ghi xác nhận của người dùng và digest nội dung. `source_normalized` / `review_status=approved` thể hiện chấp nhận của chủ dữ liệu; không đổi provenance thành `video_verified` hay tuyên bố agent đã xem playback liên tục. Xem [audit_report.md](audit_report.md) để biết trạng thái hiện hành và lịch sử quan sát.

[review.html](review.html) hỗ trợ xem/chỉnh và xuất **nháp mới**; nó không tự ghi đè hoặc tự phê duyệt JSONL. Nếu nhãn/hint thay đổi, digest approval phải được cập nhật sau xác nhận tương ứng; exporter từ chối tự áp dụng approval cũ cho nội dung khác.

## Kiểm tra và tái xuất

Từ repository root:

```sh
backend/.venv/bin/python benchmarks/progressive/annotations/validate_annotations.py
backend/.venv/bin/python benchmarks/progressive/annotations/export_annotations.py
backend/.venv/bin/python benchmarks/progressive/annotations/validate_annotations.py
```

Không chạy `prepare_review.py` lên bộ đã chuẩn hóa: đó là bước khởi tạo queue và có thể ghi đè dữ liệu.

## Artifact cục bộ

Khoảng 1,2 GB clip/ảnh trong `evidence/` được giữ trên máy và không commit vào Git. [evidence_manifest.json](evidence_manifest.json) ghi đường dẫn, kích thước và SHA-256 của toàn bộ artifact để đối chiếu khi chuyển máy. Hai workbook nguồn và mapping/media cần có cục bộ để tái xuất; workbook PHM nguồn đang thuộc ignore rule riêng của repository. Source records/GT/hint gốc đã được giữ trong các JSONL có version control.

Full validation kiểm tra file và hash thực tế. Sau clone chưa có media, có thể dùng `validate_annotations.py --metadata-only` để kiểm tra cấu trúc, approval và tham chiếu manifest; chế độ đó không chứng minh file media có mặt hoặc đúng nội dung. Không tự gọi metadata validation là video verification.
