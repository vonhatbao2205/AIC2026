# Ghi chú biên tập — SOICT 2026

Title mới: **CAD-VR: Evidence-Guided Control for Fast–Slow Agentic Video Event Retrieval**.
Bản tích hợp C+V ngày 05/10/2026. Xem `main.pdf`, `SOURCE_MAP.md` và
`data/ablation_stats.json` để đối chiếu kết quả.

## Kết quả và cách diễn giải

Paper phân biệt rõ hai đợt trên cùng 86 TEST queries:

- Đợt A–F ban đầu: E 58.1%, C 51.2%; E−C +7.0 điểm, CI [−2.3,16.3], p=.21.
- Đợt C/C+V/E bổ sung: 47.7% / 50.0% / 58.1%; đủ 258 lượt, không lỗi/fallback
  trong lượt hoàn tất. E−C +10.5 điểm [2.3,18.6], p=.035 chưa hiệu chỉnh multiple tests.
- E−C+V +8.1 điểm [0.0,16.3], p=.092: **chưa conclusive**. Không được viết đã
  chứng minh gain do adaptive scheduling, hoặc C+V tương đương E.
- C+V−C +2.3 điểm [−3.5,9.3], p=.727. Hai cấu hình chạy agent độc lập.
- So sánh cùng evidence trong C+V: RRF 40/86 → verifier 43/86; sửa 4, hỏng 1;
  +3.5 điểm [−1.2,9.3], p=.375. Đây là phép đo trực tiếp tác động final ordering.
- E mới đạt 0/8 TRAKE, so với 3/8 ban đầu. Tổng R@1 giống nhau không có nghĩa
  chất lượng theo task ổn định. Không giấu kết quả kém thuận lợi này.
- Median C/C+V/E mới là 53.7/55.6/75.8 s; agents/query là 2/2/1.95.
  Không claim deployed compute saving. Có 13 quota interruptions: 314.6 s và
  25 agent calls ngoài các lượt hoàn tất; không tính thời gian chờ quota hồi.

## Đóng góp và giới hạn

Ba contribution: typed evidence-guided control; evidence board + explicit guard
+ audited traces; benchmark và diagnostic evaluation. Group-by-video là thiết kế
phục vụ operator và tiêu chí chẩn đoán, không đứng riêng như thuật toán mới.
Guard ưu tiên evidence sufficiency có thể audit hơn giảm computation.

Giữ negative results D=A, final-order E/F và replay E† (1.29 agents, 35.9 s,
52.3% R@1) trong **đợt ban đầu**. Không diễn giải replay thành live policy.
C+V đã được chạy; giới hạn hiện tại là power/variability và attribution của cả
controller package, thay vì thiếu baseline. Cumulative hints vẫn là independent
fresh runs, không memory giữa các hint và không “learns across hints”.

## Artifacts

- `main.pdf`: 12 trang nội dung và tài liệu tham khảo, giữ format LNCS/CCIS.
- `data/ablation_stats.json`: đủ kết quả, paired intervals, audit và nguồn hash.
- `benchmarks/agent/results/soict-cv-tol1/`: snapshot gọn có report và CSV.
- `tools/benchmark_cv_report.py` ở repo root: tái tính offline, không gọi model.
- Fig. 1 giữ thiết kế/logo đã duyệt. Đủ bảy hình trong main PDF, gồm qualitative example Fig. 7 ở đợt ban đầu;
  đã rút các đoạn trùng ý để giữ đồng thời hình và ablation mới.

Paper directory bị git-ignore; `git status` thường không hiện thay đổi trong đó.
Tên tác giả/cơ quan được giữ nguyên. Chưa tự commit hoặc nộp bản sửa này.
