# Báo cáo Thiết kế lại Pipeline Trích xuất Khung hình Đại diện

**Dự án:** AIC 2026 — Truy vấn nội dung video  
**Phạm vi:** Giai đoạn tiền xử lý — Trích xuất khung hình đại diện  
**Môi trường thực thi:** Kaggle Notebook, GPU T4×2  

---

## Mục lục

1. [Tóm tắt thay đổi](#1-tóm-tắt-thay-đổi)
2. [Pipeline mới — Kiến trúc tổng quan](#2-pipeline-mới--kiến-trúc-tổng-quan)
3. [Mô tả chi tiết từng module](#3-mô-tả-chi-tiết-từng-module)
   - 3.1 Phát hiện cảnh
   - 3.2 Tính tín hiệu nổi bật
   - 3.3 Lấy mẫu thích nghi
   - 3.4 Đọc khung hình từ đĩa
   - 3.5 Chấm điểm ổn định
   - 3.6 Mã hoá đặc trưng thị giác
   - 3.7 Phân cụm và chọn đại diện
   - 3.8 Loại trùng lặp có kiểm soát
   - 3.9 Xuất kết quả
4. [Lý do thay đổi từng thành phần](#4-lý-do-thay-đổi-từng-thành-phần)

---

## 1. Tóm tắt thay đổi

| Module | Trạng thái | Nội dung thay đổi |
|--------|------------|-------------------|
| Phát hiện cảnh | **Sửa lỗi** | Ba lỗi cài đặt: lệch một đơn vị, cảnh đầu không gộp, tốc độ lấy mẫu cứng |
| Tính tín hiệu nổi bật | **Mới hoàn toàn** | Module kết hợp tín hiệu thị giác và âm thanh để định vị sự kiện trong cảnh |
| Lấy mẫu ứng viên | **Thiết kế lại** | Từ phân số cố định sang lấy mẫu thích nghi theo tín hiệu nổi bật |
| Đọc khung hình | Giữ nguyên | — |
| Chấm điểm ổn định | **Sửa lỗi** | Công thức độ sáng không còn phạt nhầm cảnh ngoài trời |
| Mã hoá đặc trưng | Giữ nguyên | — |
| Phân cụm DBSCAN | **Sửa lỗi** | Cập nhật tập đại diện ngay khi giữ điểm nhiễu, tránh trùng lặp ngầm |
| Loại trùng lặp | **Thiết kế lại** | Từ ngưỡng cứng sang độ phủ có suy giảm theo thời gian |
| Xuất kết quả | Giữ nguyên | — |

**Tóm gọn triết lý thay đổi:** Pipeline cũ xử lý từng cảnh như một đơn vị đồng nhất và loại trùng lặp dựa trên hình thức bên ngoài. Pipeline mới thừa nhận rằng *sự kiện quan trọng phân bố không đều trong thời gian* và rằng *tương đồng thị giác không đồng nghĩa với dư thừa ngữ nghĩa*.

---

## 2. Pipeline mới — Kiến trúc tổng quan

```
Video thô
    │
    ▼
┌─────────────────────────────────────────────┐
│ Module 1: Phát hiện cảnh (Sửa lỗi)         │
│  TransNetV2 → PySceneDetect → Histogram     │
└──────────────────────┬──────────────────────┘
                       │  Danh sách cảnh [(đầu, cuối), ...]
                       ▼
┌─────────────────────────────────────────────┐
│ Module 2A: Tính tín hiệu nổi bật (Mới)     │
│  S(t) = ReLU(α·Δ_thị_giác + β·Δ_âm_thanh  │
│              − ε)                           │
└──────────────────────┬──────────────────────┘
                       │  Tín hiệu S(t) theo từng cảnh
                       ▼
┌─────────────────────────────────────────────┐
│ Module 2B: Lấy mẫu thích nghi (Thiết kế    │
│  lại)                                       │
│  Mẫu nền ∪ Mẫu đỉnh S(t) ± δ              │
└──────────────────────┬──────────────────────┘
                       │  Tập chỉ số khung hình ứng viên
                       ▼
┌─────────────────────────────────────────────┐
│ Module 3: Đọc khung hình (Giữ nguyên)      │
│  Duyệt tuyến tính, grab()/retrieve()        │
└──────────────────────┬──────────────────────┘
                       │  Từ điển {chỉ_số: ảnh_RGB}
                       ▼
┌─────────────────────────────────────────────┐
│ Module 4: Chấm điểm ổn định (Sửa lỗi)     │
│  Nét · Sáng (Gaussian) · Tương phản        │
└──────────────────────┬──────────────────────┘
                       │  Điểm ổn định cho từng ứng viên
                       ▼
┌─────────────────────────────────────────────┐
│ Module 5: Mã hoá đặc trưng (Giữ nguyên)   │
│  PE-Core-G14-448 → SigLIP2 → Thủ công      │
└──────────────────────┬──────────────────────┘
                       │  Véc-tơ đặc trưng (1280 chiều)
                       ▼
┌─────────────────────────────────────────────┐
│ Module 6: Phân cụm + Chọn đại diện (Sửa)  │
│  DBSCAN cosine per-cảnh                     │
│  Cập nhật tập đã chọn khi giữ điểm nhiễu   │
└──────────────────────┬──────────────────────┘
                       │  Tập khung hình đại diện thô
                       ▼
┌─────────────────────────────────────────────┐
│ Module 7: Loại trùng lặp (Thiết kế lại)   │
│  Coverage(f,K) = max[cos·exp(−Δt/60)]      │
│  Giữ nếu Coverage < 0,85                   │
└──────────────────────┬──────────────────────┘
                       │  Tập khung hình cuối cùng
                       ▼
┌─────────────────────────────────────────────┐
│ Module 8: Xuất kết quả AIC (Giữ nguyên)   │
│  keyframes/ · map-keyframes/ · embeddings/  │
└─────────────────────────────────────────────┘
```

---

## 3. Mô tả chi tiết từng module

---

### 3.1 Phát hiện cảnh

**Mục đích:** Phân chia video thành các cảnh — đoạn video liên tục giữa hai điểm cắt. Đây là đơn vị xử lý cơ bản của toàn bộ pipeline.

**Kiến trúc phân tầng dự phòng:**

Có ba phương pháp phát hiện điểm cắt, xếp theo thứ tự ưu tiên từ mạnh đến yếu. Bất kỳ thất bại nào ở tầng trên đều tự chuyển sang tầng dưới.

```
TransNetV2  →  PySceneDetect  →  Histogram màu HSV
   (học sâu)    (dựa trên nội dung)  (luôn khả dụng)
```

**Tầng 1 — TransNetV2:** Mô hình học sâu chuyên biệt trả về mảng xác suất $p(t) \in [0,1]$ — xác suất để khung hình thứ $t$ là khung chuyển cảnh. Khung chuyển cảnh là khung hình thuộc về khoảng trung gian giữa hai cảnh, thường bị nhòe hoặc chứa pha trộn hình ảnh.

Thuật toán nhận diện cảnh từ mảng $p(t)$:

```
Nhị phân hoá: p̂(t) = 1 nếu p(t) > 0,5; bằng 0 nếu không
Một cảnh = khoảng liên tục gồm các t có p̂(t) = 0
Ranh giới cảnh = vị trí cuối cùng của cảnh = t_đầu_chuyển - 1
```

**Tầng 2 — PySceneDetect:** Phân tích sự thay đổi nội dung khung hình liên tiếp bằng chỉ số nội dung (content score). Điểm thay đổi vượt ngưỡng `scenedetect_threshold = 27,0` được đánh dấu là điểm cắt.

**Tầng 3 — Histogram màu HSV:** Tính khoảng cách Bhattacharyya giữa biểu đồ phân phối màu HSV của hai khung hình liên tiếp (lấy mẫu ở độ phân giải 320×240). Khi khoảng cách vượt `hist_threshold = 0,45`, một điểm cắt được ghi nhận.

```
Tốc độ lấy mẫu: sample_fps = CONFIG["shot_detection_sample_fps"]
Biểu đồ: không gian HSV, kích thước [50×60]
Khoảng cách: d_Bhattacharyya(hist_t, hist_{t-1})
```

**Gộp cảnh ngắn (`_merge_short`):**

Sau khi có danh sách cảnh thô, những cảnh ngắn hơn `min_shot_sec = 0,6` giây bị gộp vào cảnh liền kề. Quy tắc:

- Nếu cảnh hiện tại quá ngắn và có cảnh trước: gộp vào cảnh trước.
- Nếu cảnh đầu tiên quá ngắn (không có cảnh trước): gộp vào cảnh tiếp theo bằng cách mở rộng cảnh tiếp theo về phía trái.

**Siêu tham số:**

| Tham số | Giá trị | Vai trò |
|---------|---------|---------|
| `scenedetect_threshold` | 27,0 | Độ nhạy PySceneDetect |
| `hist_threshold` | 0,45 | Độ nhạy Histogram |
| `min_shot_sec` | 0,6 | Ngưỡng tối thiểu cảnh (giây) |
| `shot_detection_sample_fps` | 2,0 | Tốc độ lấy mẫu Histogram |

---

### 3.2 Tính tín hiệu nổi bật

**Mục đích:** Xây dựng hàm $S(t)$ theo thời gian cho biết khả năng xuất hiện sự kiện quan trọng tại từng vị trí trong cảnh. Tín hiệu này dùng để hướng dẫn bước lấy mẫu thích nghi.

**Tại sao cần module này:**

Phân phối sự kiện quan trọng trong một cảnh dài không đồng đều. Phần lớn thời gian là trạng thái ổn định; sự kiện quan trọng tập trung tại các điểm chuyển đổi nội dung. Tín hiệu nổi bật là cơ chế "dự báo trước" vị trí cần lấy mẫu dày mà không cần giải mã toàn bộ video ở độ phân giải cao.

**Phương trình tín hiệu nổi bật:**

$$S(t) = \text{ReLU}\!\bigl(\alpha \cdot \Delta_{\text{thị giác}}(t) + \beta \cdot \Delta_{\text{âm thanh}}(t) - \varepsilon\bigr)$$

**Thành phần 1 — Thay đổi thị giác $\Delta_{\text{thị giác}}(t)$:**

Lấy mẫu video ở tốc độ thấp (4–8 khung/giây) với độ phân giải thu nhỏ (64×64 điểm ảnh). Tính khoảng cách Bhattacharyya giữa biểu đồ HSV của khung $t$ và khung $t-1$:

$$\Delta_{\text{thị giác}}(t) = d_{\text{Bhattacharyya}}\bigl(\text{hist}(f_t),\; \text{hist}(f_{t-1})\bigr)$$

Chuẩn hoá per-cảnh: chia cho phân vị 95 của dãy $\Delta_{\text{thị giác}}$ trong cảnh đó, đưa về $[0, 1]$. Chuẩn hoá per-cảnh đảm bảo cả cảnh tĩnh lẫn cảnh hành động đều phát hiện được sự kiện tương đối nổi bật trong ngữ cảnh riêng của chúng.

**Thành phần 2 — Thay đổi âm thanh $\Delta_{\text{âm thanh}}(t)$:**

Trích xuất luồng âm thanh (16kHz, đơn kênh). Tính năng lượng RMS trên từng cửa sổ 100ms. Lấy giá trị tuyệt đối của đạo hàm:

$$\Delta_{\text{âm thanh}}(t) = \left|\frac{d}{dt} \text{RMS}(t)\right|$$

Chuẩn hoá per-cảnh tương tự thành phần thị giác. Tín hiệu này phát hiện: lời nói bắt đầu/kết thúc đột ngột, tiếng vỗ tay, nhạc chuyển cảnh, âm thanh sự kiện.

**Dự phòng không có âm thanh:** Nếu trích xuất âm thanh thất bại (lỗi codec, luồng im lặng), đặt $\beta = 0$ và tiếp tục với tín hiệu thị giác đơn thuần. Pipeline không bị gián đoạn.

**Vai trò của ReLU và $\varepsilon$:**

$\varepsilon$ là ngưỡng nhiễu nền. Khi cả hai tín hiệu ở mức thấp (cảnh tĩnh, âm thanh đều đặn), $S(t) = 0$ — không ghi nhận sự kiện. ReLU đảm bảo chỉ những thay đổi vượt ngưỡng mới đóng góp vào tín hiệu nổi bật, loại bỏ dao động nhỏ vô nghĩa.

**Siêu tham số:**

| Tham số | Giá trị khởi đầu | Vai trò |
|---------|-----------------|---------|
| $\alpha$ | 0,60 | Trọng số tín hiệu thị giác |
| $\beta$ | 0,40 | Trọng số tín hiệu âm thanh |
| $\varepsilon$ | 0,15 | Ngưỡng nhiễu nền |
| Tốc độ lấy mẫu tín hiệu | 4–8 khung/giây | Độ phân giải thời gian của $S(t)$ |

---

### 3.3 Lấy mẫu thích nghi

**Mục đích:** Xác định tập chỉ số khung hình cần giải mã từ mỗi cảnh — đây là "ngân sách" khung hình ứng viên trước khi mã hoá đặc trưng.

**Chiến lược hai lớp:**

Tập khung hình ứng viên cuối cùng là hợp của hai nguồn độc lập:

$$\text{Ứng\_viên} = \text{Mẫu\_nền} \cup \text{Mẫu\_sự\_kiện}$$

**Lớp 1 — Mẫu nền (lưới an toàn):**

Giữ nguyên logic lấy mẫu theo phân số cố định của pipeline cũ. Đảm bảo không có khoảng thời gian nào trong cảnh bị bỏ sót hoàn toàn, kể cả khi $S(t) = 0$ trên toàn cảnh.

| Độ dài cảnh | Số mẫu nền | Vị trí |
|-------------|-----------|--------|
| < 2 giây | 1 | 50% |
| 2–8 giây | 3 | 25%, 50%, 75% |
| 8–20 giây | 5 | 10%, 30%, 50%, 70%, 90% |
| > 20 giây | Đều theo `candidate_fps` | — |

**Lớp 2 — Mẫu sự kiện (phản ứng với $S(t)$):**

Tìm các đỉnh cục bộ của $S(t)$: vị trí $t^*$ thoả mãn $S(t^*) > S(t^*-1)$, $S(t^*) > S(t^*+1)$, và $S(t^*) > 0$. Với mỗi đỉnh, thêm các khung hình trong cửa sổ $[t^* - \delta,\; t^* + \delta]$:

$$\delta = 0{,}25 \text{ giây}$$

Cửa sổ $\delta$ đủ rộng để không bỏ sót sự kiện ngắn nếu đỉnh $S(t)$ hơi lệch pha so với sự kiện thực, nhưng đủ hẹp để không chồng lấn giữa các đỉnh lân cận.

**Vùng đệm ranh giới:**

Hai đầu mỗi cảnh bị loại bỏ vùng đệm có kích thước `boundary_margin_frac × độ_dài_cảnh` để tránh lấy trúng khung chuyển cảnh (thường bị nhòe hoặc nhiễu ánh sáng).

**Giới hạn tổng:** Sau khi hợp hai tập và loại chỉ số trùng, tổng số ứng viên bị giới hạn tại `max_candidates_per_shot = 40`. Khi vượt giới hạn, ưu tiên giữ mẫu sự kiện (điểm $S(t^*)$ cao nhất) rồi đến mẫu nền.

**Siêu tham số:**

| Tham số | Giá trị | Vai trò |
|---------|---------|---------|
| `candidate_fps` | 2,0 | Tốc độ lấy mẫu nền cho cảnh > 20s |
| `max_candidates_per_shot` | 40 | Giới hạn tổng ứng viên/cảnh |
| `boundary_margin_frac` | 0,06 | Tỉ lệ vùng đệm ranh giới |
| $\delta$ | 0,25 giây | Bán kính cửa sổ quanh đỉnh sự kiện |

---

### 3.4 Đọc khung hình từ đĩa

**Mục đích:** Giải mã pixel của các khung hình ứng viên đã xác định ở module trước.

**Chiến lược duyệt tuyến tính:**

Thay vì truy cập ngẫu nhiên theo chỉ số (tốn kém với video nén), hàm duyệt video một lần từ đầu đến cuối. Với mỗi khung hình trên đường đi:

- Gọi `grab()`: đẩy con trỏ nội bộ tiến lên một khung, **không giải mã pixel** — rất nhanh.
- Nếu chỉ số thuộc tập cần lấy: gọi thêm `retrieve()` để giải mã pixel thực sự.
- Nếu không: bỏ qua, tiếp tục.

**Lý do hiệu quả:** Video nén (MP4, MKV) lưu các khung hình chốt (I-frame) toàn bộ và các khung hình delta (P-frame, B-frame) chỉ lưu phần thay đổi. Để giải mã một P-frame cụ thể bằng cách seek ngẫu nhiên, bộ giải mã phải tìm I-frame gần nhất rồi giải mã tuần tự từ đó đến đích — chi phí cao, không ổn định. Duyệt tuyến tính với `grab()` tránh hoàn toàn chi phí này.

Tất cả khung hình được thu nhỏ về `save_max_side = 720` điểm ảnh (cạnh dài nhất) và chuyển sang không gian màu RGB trước khi lưu vào bộ nhớ.

---

### 3.5 Chấm điểm ổn định

**Mục đích:** Đánh giá chất lượng hình ảnh của từng khung hình ứng viên để phân xử khi phải chọn giữa các khung hình tương đương về ngữ nghĩa.

**Điểm ổn định không phải ngưỡng lọc** — không có khung hình nào bị xoá chỉ vì điểm thấp. Nó đóng vai trò tiêu chí phân xử trong ba tình huống: chọn đại diện trong cụm, lọc điểm nhiễu trong cảnh dài, và dự phòng khi không tìm được cụm.

**Ba tiêu chí:**

**Độ nét (trọng số 50%):** Tính phương sai của bộ lọc Laplacian trên ảnh mức xám. Phương sai cao chứng tỏ có nhiều cạnh sắc nét — ảnh rõ, không bị nhòe. Áp dụng log để ổn định thang đo:

$$\text{Nét} = \text{clip}\!\left(\frac{\ln(1 + \sigma^2_\text{Lap}) - \ln(6)}{\ln(1501) - \ln(6)},\; 0,\; 1\right)$$

**Độ sáng (trọng số 30%):** Đánh giá mức sáng trung bình bằng hàm Gaussian tâm tại 0,5 trên thang $[0, 1]$:

$$\text{Sáng} = \exp\!\left(-\frac{(\bar{I} - 0{,}5)^2}{2 \times 0{,}3^2}\right)$$

Hàm Gaussian giảm dần đối xứng quanh mức lý tưởng, phạt nhẹ cả hai phía theo mức độ lệch. Khung hình ở mức trung bình (0,4–0,6) nhận điểm gần tối đa; khung hình quá tối hoặc quá sáng bị phạt tương xứng.

**Độ tương phản (trọng số 20%):** Tính độ lệch chuẩn của ảnh mức xám. Ảnh có dải tông rộng (nhiều chi tiết sáng tối) nhận điểm cao hơn:

$$\text{Tương phản} = \text{clip}\!\left(\frac{\sigma_\text{gray}/255 - 0{,}04}{0{,}30 - 0{,}04},\; 0,\; 1\right)$$

**Điểm tổng hợp:**

$$\text{Ổn định} = 0{,}50 \times \text{Nét} + 0{,}30 \times \text{Sáng} + 0{,}20 \times \text{Tương phản}$$

---

### 3.6 Mã hoá đặc trưng thị giác

**Mục đích:** Biến mỗi khung hình ứng viên thành một véc-tơ số trong không gian đặc trưng nhiều chiều. Khoảng cách giữa hai véc-tơ phản ánh sự tương đồng ngữ nghĩa giữa hai khung hình tương ứng.

**Mô hình chính — PE-Core-G14-448:**

Kiến trúc ViT-G/14 với ảnh đầu vào 448×448 điểm ảnh, đầu ra 1280 chiều. Được phát triển bởi Meta AI, tối ưu hoá cho bài toán truy vấn hình ảnh. Mạnh hơn đáng kể so với các mô hình CLIP thông thường trong không gian truy vấn đa dạng.

**Tăng tốc đa GPU bằng luồng song song:**

PE-Core không tương thích với cơ chế song song dữ liệu thông thường (`DataParallel`) do cách nó ghim bộ nhớ nội bộ theo thiết bị. Giải pháp: khởi tạo một phiên bản mô hình độc lập trên mỗi GPU, chia lô khung hình và chạy song song bằng các luồng Python. Các hoạt động CUDA giải phóng khoá toàn cục (GIL) của Python, cho phép song song thực sự trên phần cứng.

```
Lô N khung hình
        │
   ┌────┴────┐
   │         │
  GPU 0     GPU 1
  N/2 kh   N/2 kh
   │         │
   └────┬────┘
        │
   Ghép & chuẩn hoá L2
```

Khi chế độ đa GPU thất bại, hệ thống tự hạ về một GPU. Tất cả véc-tơ đầu ra được chuẩn hoá về độ dài đơn vị (L2 normalisation) để khoảng cách côsin tương đương với tích vô hướng.

**Chuỗi dự phòng:**

Chuỗi dự phòng là giải pháp kỹ thuật cho điều kiện môi trường Kaggle không ổn định — không phải lựa chọn thuật toán. PE-Core yêu cầu kết nối internet để tải mã nguồn và trọng số; SigLIP2 yêu cầu tải trọng số từ HuggingFace. Khi cả hai thất bại, đặc trưng thủ công (biểu đồ màu HSV + biểu đồ cạnh Sobel, ~272 chiều) đảm bảo pipeline không dừng hoàn toàn. Tuy nhiên, chất lượng phân cụm với đặc trưng thủ công suy giảm rất mạnh vì không có khả năng hiểu ngữ nghĩa.

---

### 3.7 Phân cụm và chọn đại diện

**Mục đích:** Với tập véc-tơ đặc trưng của các khung hình ứng viên trong một cảnh, tìm ra các nhóm nội dung tương đồng và chọn một đại diện chất lượng cao cho mỗi nhóm.

**Thuật toán DBSCAN (phân cụm dựa trên mật độ):**

DBSCAN được chạy độc lập cho từng cảnh. Các điểm trong không gian đặc trưng đủ gần nhau (khoảng cách côsin ≤ `dbscan_eps`) và đủ đông (≥ `dbscan_min_samples`) tạo thành một cụm. Điểm không thuộc cụm nào được gọi là điểm nhiễu.

Ưu điểm của DBSCAN so với K-means: không cần xác định trước số cụm, tự nhiên xử lý được cảnh tĩnh (một cụm lớn) và cảnh năng động (nhiều cụm nhỏ).

**Chọn đại diện trong mỗi cụm:**

Tính tâm cụm (trung bình các véc-tơ, chuẩn hoá về đơn vị). Với mỗi thành viên, tính điểm tổng hợp:

$$\text{Điểm} = 0{,}70 \times \underbrace{\langle \mathbf{e}_i, \mathbf{c} \rangle}_{\text{đại diện}} + 0{,}30 \times \underbrace{s_i}_{\text{ổn định}}$$

Khung hình có điểm cao nhất được chọn làm đại diện cụm. Khung hình đại diện có thể không phải là điểm gần tâm nhất (medoid thuần tuý) — một khung hình hơi lệch tâm nhưng nét sắc hơn sẽ được ưu tiên hơn.

**Xử lý điểm nhiễu trong cảnh dài:**

Với cảnh dài hơn 15 giây, một điểm nhiễu được giữ lại nếu đồng thời thoả mãn:

1. Điểm ổn định $\geq$ `noise_keep_min_stability = 0,62`.
2. Độ tương đồng côsin tối đa với bất kỳ khung đại diện nào đã chọn $< 1 - \varepsilon_{\text{DBSCAN}}$.

Điều kiện 2 đảm bảo điểm nhiễu được giữ thực sự mang nội dung khác biệt. Sau mỗi lần quyết định giữ, tập đại diện đã chọn được cập nhật ngay để điểm nhiễu tiếp theo được so sánh chính xác.

**Giới hạn số khung hình/cảnh:**

| Độ dài cảnh | Tối đa |
|-------------|--------|
| < 4 giây | 1 |
| 4–15 giây | 3 |
| > 15 giây | 5 |

**Siêu tham số:**

| Tham số | Giá trị | Vai trò |
|---------|---------|---------|
| `dbscan_eps` | 0,18 | Ngưỡng khoảng cách côsin để tạo cụm |
| `dbscan_min_samples` | 2 | Số điểm tối thiểu để thành cụm |
| `w_representativeness` | 0,70 | Trọng số độ đại diện |
| `w_stability` | 0,30 | Trọng số độ ổn định |
| `noise_keep_min_stability` | 0,62 | Ngưỡng giữ điểm nhiễu |

---

### 3.8 Loại trùng lặp có kiểm soát

**Mục đích:** Loại bỏ các khung hình thực sự dư thừa ở cấp độ toàn video, trong khi bảo toàn các khung hình có giá trị truy vấn khác nhau dù trông tương đồng.

**Nguyên tắc thiết kế:**

Hai khung hình thực sự dư thừa khi và chỉ khi: chúng tương đồng về nội dung **và** bối cảnh thời gian của chúng cũng tương tự (tức là thuộc cùng một sự kiện liên tục). Hai khung hình tương đồng nhưng cách xa nhau trong thời gian có thể đại diện cho hai lần xuất hiện độc lập của cùng địa điểm — cả hai đều có giá trị truy vấn.

**Hàm độ phủ có suy giảm theo thời gian:**

Với khung hình ứng viên $f$ và tập khung hình đã giữ $K$:

$$\text{Phủ}(f,\, K) = \max_{k \in K}\!\Bigl[\cos(\mathbf{e}_f, \mathbf{e}_k) \times \exp\!\Bigl(-\frac{|t_f - t_k|}{\tau}\Bigr)\Bigr]$$

**Quyết định:** Giữ $f$ nếu $\text{Phủ}(f, K) < \theta$; loại nếu $\geq \theta$.

**Giá trị tham số:**

$$\tau = 60 \text{ giây}, \qquad \theta = 0{,}85$$

Với $\tau = 60$s và video AIC điển hình dài 10–17 phút:

| Khoảng cách thời gian | Suy giảm | Hệ quả thực tế |
|----------------------|---------|----------------|
| 30 giây | 0,61 | Hai khung gần vẫn ảnh hưởng mạnh lên nhau |
| 60 giây | 0,37 | Ảnh hưởng giảm xuống dưới 40% |
| 2 phút | 0,14 | Gần như độc lập |
| 5 phút trở lên | < 0,01 | Hoàn toàn độc lập, luôn được giữ |

**Thứ tự xử lý:**

Trước khi duyệt, các khung hình được sắp xếp theo **điểm ổn định giảm dần**. Khung hình chất lượng cao được thêm vào tập $K$ trước, trở thành neo so sánh. Khung hình kém chất lượng hơn nhưng tương đồng sẽ bị lọc sau. Điều này đảm bảo khi phải chọn giữa hai khung hình gần giống nhau, khung hình nét và rõ hơn được giữ lại — quan trọng cho chất lượng khớp với truy vấn.

**Vai trò của băm tri giác (pHash):**

Băm tri giác (mã rút gọn cấu trúc hình ảnh) không còn tham gia vào quyết định loại trùng lặp. Nó chỉ được dùng làm bộ lọc sơ bộ để tăng tốc: nếu khoảng cách pHash của hai khung hình lớn, chắc chắn chúng không trùng — bỏ qua tính côsin tốn kém hơn. Quyết định cuối cùng chỉ dựa trên hàm độ phủ.

**Siêu tham số:**

| Tham số | Giá trị | Vai trò |
|---------|---------|---------|
| $\tau$ | 60 giây | Hằng số thời gian suy giảm |
| $\theta$ | 0,85 | Ngưỡng độ phủ để loại |

---

### 3.9 Xuất kết quả

Đầu ra tuân theo định dạng chuẩn AIC, tương thích trực tiếp với bước xây dựng chỉ mục vector và nộp bài:

```
/kaggle/working/
├── keyframes/
│   └── <video_id>/
│       ├── 001.jpg
│       ├── 002.jpg
│       └── ...
├── map-keyframes/
│   └── <video_id>.csv         # n, frame_idx, pts_time, fps, shot_id, ...
└── embeddings/
    └── <video_id>.npy          # (N, 1280) float16, hàng i ↔ n=i+1
```

Trường `frame_idx` lưu chỉ số khung hình gốc trong video — dùng trực tiếp để tạo file nộp bài `(video_id, frame_idx)`. Véc-tơ đặc trưng được lưu dưới dạng `float16` để tiết kiệm bộ nhớ, với khoá tra cứu `<video_id>#<n>`.

---

## 4. Lý do thay đổi từng thành phần

---

### 4.1 Phát hiện cảnh — Sửa ba lỗi cài đặt

#### Lỗi 1: Lệch một đơn vị trong `_predictions_to_scenes()`

**Vấn đề của phiên bản cũ:**

Khi TransNetV2 xác định khung hình $t$ là khung chuyển cảnh, hàm cũ đưa khung đó vào cảnh trước:

```python
# Cũ (sai):
if t_prev == 0 and t == 1:
    scenes.append([start, i])   # i là khung chuyển cảnh, không thuộc cảnh trước
```

Kết quả: mỗi cảnh đều kết thúc bằng một khung chuyển cảnh — thường là khung hình nhòe do pha trộn hai cảnh. Các khung hình ứng viên gần cuối cảnh sẽ có điểm ổn định thấp giả tạo, không phản ánh chất lượng thực của nội dung cảnh.

**Phần mới đáp ứng được gì:**

```python
# Mới (đúng):
if t_prev == 0 and t == 1:
    scenes.append([start, i - 1])   # cắt trước khung chuyển cảnh
```

Ranh giới cảnh giờ là khung hình cuối cùng thuộc về nội dung cảnh đó, không phải khung hình chuyển tiếp.

---

#### Lỗi 2: Cảnh đầu tiên không được gộp khi quá ngắn

**Vấn đề của phiên bản cũ:**

`_merge_short()` chỉ gộp cảnh ngắn vào **cảnh trước**. Cảnh đầu tiên không có cảnh trước, nên dù ngắn hơn `min_shot_sec` vẫn tồn tại độc lập — vi phạm ràng buộc tối thiểu.

**Phần mới đáp ứng được gì:**

Sau vòng lặp chính, thêm kiểm tra riêng cho cảnh đầu: nếu ngắn hơn ngưỡng, gộp vào cảnh kế tiếp bằng cách mở rộng cảnh kế về phía trái. Không có ngoại lệ nào trong danh sách cảnh vi phạm `min_shot_sec`.

---

#### Lỗi 3: Tốc độ lấy mẫu histogram cứng hoá

**Vấn đề của phiên bản cũ:**

```python
# Cũ (sai): bỏ qua cấu hình
sample_fps = 4.0
```

`CONFIG` có tham số `shot_detection_sample_fps = 2.0` nhưng hàm histogram không đọc nó. Hành vi thực tế khác với hành vi người dùng cấu hình, gây nhầm lẫn khi tinh chỉnh.

**Phần mới đáp ứng được gì:**

```python
# Mới:
sample_fps = CONFIG["shot_detection_sample_fps"]
```

Hành vi nhất quán với cấu hình. Người dùng có thể tăng `shot_detection_sample_fps` để phát hiện điểm cắt chính xác hơn mà không cần chỉnh code.

---

### 4.2 Lấy mẫu ứng viên — Thiết kế lại

**Vấn đề của phiên bản cũ:**

Pipeline cũ lấy mẫu tại các phân số cố định (25%, 50%, 75%...) bất kể nội dung bên trong cảnh. Điều này dựa trên giả định sai rằng **sự kiện quan trọng phân bố đều theo thời gian**.

Hệ quả định lượng: với `candidate_fps = 2` (khoảng cách lấy mẫu 0,5 giây), xác suất bỏ sót một sự kiện ngẫu nhiên kéo dài 0,3 giây là:

$$P(\text{bỏ sót}) = \frac{0{,}5 - 0{,}3}{0{,}5} = 40\%$$

Với sự kiện 0,1 giây: xác suất bỏ sót tăng lên 80%.

**Phần mới đáp ứng được gì:**

Tín hiệu nổi bật $S(t)$ cho phép lấy mẫu dày tại đúng những điểm có khả năng xảy ra sự kiện. Mẫu nền vẫn được giữ như lưới an toàn. Kết quả: với cùng ngân sách `max_candidates_per_shot`, pipeline mới tập trung tài nguyên vào các khoảng thời gian quan trọng thay vì phân bổ đều vô điều kiện.

---

### 4.3 Chấm điểm ổn định — Sửa công thức độ sáng

**Vấn đề của phiên bản cũ:**

Hàm độ sáng cũ là hàm lều (tent function) tâm tại 0,45:

$$\text{Sáng}_\text{cũ} = \text{clip}\!\left(1 - \frac{|\bar{I} - 0{,}45|}{0{,}45},\; 0,\; 1\right)$$

Điểm bằng 0 khi $\bar{I} \geq 0{,}90$ (229/255). Điều này phạt nặng mọi cảnh quay ngoài trời ban ngày, mặt nước phản chiếu ánh sáng, hoặc cảnh studio sáng — tất cả đều phổ biến trong dữ liệu AIC.

**Phần mới đáp ứng được gì:**

Hàm Gaussian tâm tại 0,5 phạt theo mức độ lệch, không phạt nhị phân khi vượt ngưỡng. Khung hình sáng tự nhiên (0,6–0,75) nhận điểm 0,7–0,9 thay vì 0,0–0,3. Mức lý tưởng được đặt tại 0,5 (trung tính thực sự) thay vì 0,45 (lệch về tối).

---

### 4.4 Phân cụm DBSCAN — Sửa lỗi cập nhật tập đại diện

**Vấn đề của phiên bản cũ:**

Khi giữ lại các điểm nhiễu trong cảnh dài, véc-tơ so sánh `chosen_emb` được tính một lần trước vòng lặp và không bao giờ cập nhật:

```python
# Cũ (sai): tính một lần, không cập nhật
chosen_emb = np.stack([c.embedding for c in chosen])
for i in range(len(cands)):
    if labels[i] != -1: continue
    # chosen_emb đã lỗi thời khi c được thêm vào chosen
    if (chosen_emb @ c.embedding).max() < nguong:
        chosen.append(c)
```

Hệ quả: hai điểm nhiễu tương đồng nhau (cùng cụm nhỏ không đủ `min_samples`) đều vượt qua kiểm tra vì cả hai đều xa tập đại diện ban đầu. Chúng đều được giữ lại dù thực chất là trùng lặp. Bước loại trùng lặp phía sau phải xử lý hậu quả này.

**Phần mới đáp ứng được gì:**

```python
# Mới: cập nhật ngay sau mỗi lần giữ
danh_sach_vecto = [c.embedding for c in chosen]
for i in range(len(cands)):
    ...
    vecto_tap = np.stack(danh_sach_vecto)
    if (vecto_tap @ c.embedding).max() < nguong:
        chosen.append(c)
        danh_sach_vecto.append(c.embedding)  # cập nhật ngay
```

Điểm nhiễu thứ hai sẽ thấy điểm nhiễu thứ nhất đã được thêm vào tập so sánh và bị lọc đúng cách.

---

### 4.5 Loại trùng lặp — Thiết kế lại

**Vấn đề của phiên bản cũ:**

Pipeline cũ dùng điều kiện:

```
Loại nếu: (cosine ≥ 0,94) HOẶC (khoảng_cách_pHash ≤ 4)
```

Điều kiện này có hai vấn đề nghiêm trọng:

**Vấn đề 1 — Bỏ qua yếu tố thời gian:** Hai khung hình giống nhau cách nhau 2 giây và cách nhau 15 phút được đối xử giống hệt nhau. Khung hình cách xa 15 phút có thể đại diện cho hai lần xuất hiện độc lập của cùng địa điểm — trong AIC, cả hai đều là câu trả lời hợp lệ cho các truy vấn khác nhau.

**Vấn đề 2 — pHash không đo ngữ nghĩa:** Băm tri giác đo sự tương đồng cấu trúc hình ảnh cấp thấp (bố cục, phân bố sáng tối). Hai khung hình của cùng một phòng họp — một trống, một có người ngồi — có bố cục tổng thể giống nhau. Với ngưỡng pHash ≤ 4 và điều kiện "hoặc", khung hình có người có thể bị loại dù nó là câu trả lời chính xác cho truy vấn "cảnh có người trong phòng họp".

**Phần mới đáp ứng được gì:**

Hàm độ phủ kết hợp tương đồng cosine với suy giảm thời gian tự nhiên giải quyết cả hai vấn đề:

- Yếu tố thời gian: $\exp(-|t_f - t_k|/\tau)$ tự động giảm ảnh hưởng tương đồng theo khoảng cách thời gian. Khung hình cách 5 phút trở lên gần như luôn được giữ bất kể tương đồng thị giác.
- Không dùng pHash để quyết định: loại bỏ nguồn dương tính giả chính trong điều kiện cũ. pHash chỉ còn dùng để tăng tốc (bộ lọc loại trừ nhanh).
- Ngưỡng $\theta = 0,85$ thiên về giữ lại (tránh mất frame quan trọng), phù hợp với mục tiêu tối đa hoá độ phủ truy vấn của AIC.

---

*Kết thúc báo cáo.*
