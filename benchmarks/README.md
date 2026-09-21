# Benchmark AIC26

Bộ script sinh ra toàn bộ số liệu trong Chương 10 của báo cáo kỹ thuật
(`AIC26_Technical_Report/`). Mục tiêu là mỗi con số trong báo cáo đều truy được
về một file JSON có ghi cấu hình và thời điểm chạy, thay vì chỉ còn lại bảng tổng
hợp như các lần đánh giá trước.

## Chuẩn bị

Backend phải chạy ở chế độ `live` tại cổng 8000 và `/api/health` phải báo `ok`
cho Elastic, Milvus và PE encoder.

```bash
export LD_LIBRARY_PATH=/nix/store/hngmi01i8wgi25a0byrxcn4ysz5j79mw-gcc-15.2.0-lib/lib:$LD_LIBRARY_PATH
cd backend && PYTHONPATH=. ./.venv/bin/python -m uvicorn app.main:app --port 8000 &
curl -s localhost:8000/api/health | python3 -m json.tool
```

## Chạy

```bash
# Kiểm tra việc đọc ground truth trước khi chạy các bước tốn thời gian
backend/.venv/bin/python benchmarks/gt_loader.py

# T-KIS: 67 query, ablation kênh
for c in full pe_only no_speech no_audio; do
  backend/.venv/bin/python benchmarks/run_tkis.py --config $c
done

# TRAKE: 7 query có GT + 4 câu chung kết
backend/.venv/bin/python benchmarks/run_trake.py

# Định tuyến query có ảnh hưởng tới thứ hạng không
backend/.venv/bin/python benchmarks/run_routing_ablation.py

# Độ trễ — luôn dùng --interleave, xem cảnh báo bên dưới
backend/.venv/bin/python benchmarks/run_latency.py -n 25 --interleave

# In ra đúng các con số xuất hiện trong chương Thực nghiệm của báo cáo
backend/.venv/bin/python benchmarks/report_metrics.py
```

Kết quả ghi vào `benchmarks/runs/*.json`.

## Bộ sinh 100 đáp án

Chấm theo đúng công thức của vòng sơ tuyển
(`Final = (R@1 + R@5 + R@20 + R@50 + R@100) / 5`), tách làm hai bước để việc dò
tham số không phải gọi lại Milvus/Elastic hàng nghìn lần.

```bash
# 1. Chụp kho ứng viên live một lần cho mọi truy vấn có GT ở phía L21-L30
backend/.venv/bin/python benchmarks/cache_pools.py --profile infoshotpp --side L --scope auto

# 2. Đọc thang epsilon ra từ phân bố sai số truy xuất-GT (đừng đặt theo cảm tính)
backend/.venv/bin/python benchmarks/run_answer_gen.py --mode calib

# 3. Chấm bộ tham số đang ship
backend/.venv/bin/python benchmarks/run_answer_gen.py --mode eval

# 4. Dò tham số: leo đồi từng toạ độ, nhiều điểm khởi đầu ngẫu nhiên
backend/.venv/bin/python benchmarks/run_answer_gen.py --mode sweep --restarts 30 --passes 6

# 5. Kiểm chứng chéo: bộ tham số có tổng quát hoá không hay chỉ vừa khít 21 câu
backend/.venv/bin/python benchmarks/run_answer_gen.py --mode cv --restarts 4

# 6. Mô phỏng đầu danh sách do người chấm tay, quét phạt video đã chiếm chỗ
backend/.venv/bin/python benchmarks/run_answer_gen.py --mode taken --head 3
```

**Dung sai chấm điểm.** Ground truth của nhóm là các frame rời rạc, không phải
đoạn `[s, e]` mà BTC dùng để chấm. Vì vậy mọi bảng đều in nhiều mức dung sai, và
mục tiêu tối ưu mặc định là **trung bình Final trên tol = {0, 12, 25, 50, 100}
frame** thay vì một mức duy nhất. Lý do rất thực tế: tối ưu ở tol = 0 dạy thuật
toán liệt kê keyframe, tối ưu ở tol = 200 dạy nó rải epsilon thật thưa — mà
không ai biết trước đoạn chấp nhận của BTC rộng bao nhiêu. Đổi bằng `--tol`.

**Cực trị địa phương là có thật.** Leo đồi một lần từ tham số mặc định dừng lại
sau hai vòng ở 0,5600; ba mươi điểm khởi đầu ngẫu nhiên rải từ 0,539 đến 0,590.
Chênh 0,01 trên 21 câu chỉ là **một câu nhảy một mốc**, nên `--mode sweep` không
lấy thẳng cấu hình cao nhất: nó khử trùng lặp toàn bộ log rồi chọn từng tham số
theo điểm trung bình trong nhóm cấu hình dẫn đầu, và chỉ dùng kết quả đó khi nó
không kém hơn cấu hình cao nhất.

**Đừng đuổi theo phần chênh dưới ngưỡng nhiễu.** Chạy lại toàn bộ quá trình dò
sau khi sửa thuật toán cho ra một cấu hình khác đạt 0,5943 so với 0,5905 của bộ
tham số đang ship (`runs/answer_gen_sweep_kis_rerun.json`). Chênh 0,0038 nhỏ hơn
**một nửa** bước lượng tử của thước đo (một câu nhảy một mốc = 0,0095), và cấu
hình mới đổi phần hơn ở tol 50-200 lấy phần kém ở tol 12-25 — đúng vùng chứa ví
dụ đoạn `[500, 510]` (11 frame) mà BTC đưa trong thể lệ. Vì vậy bộ tham số cũ
được giữ nguyên. Kiểm chứng chéo mới là căn cứ để đổi tham số, không phải một
con số in-sample cao hơn.

| File kết quả | Nội dung |
|---|---|
| `runs/answer_gen_eval_shipped.json` | Điểm của bộ tham số đang ship, chi tiết từng câu |
| `runs/answer_gen_cv.json` | Kiểm chứng chéo 3-fold × 2: tuned 0,515 vs mặc định 0,484 trên phần held-out |
| `runs/answer_gen_sweep_kis_rerun.json` | Lần dò lại sau khi sửa thuật toán (0,5943 — trong ngưỡng nhiễu, không dùng) |
| `runs/answer_gen_sweep_kis_wide.json` | Dò chỉ trên tol 25-200: cho thấy tối ưu một mức làm sập tol 0 xuống 0,029 |

## Các file

| File | Việc |
|---|---|
| `gt_loader.py` | Đọc ba workbook Excel, ưu tiên sheet chi tiết, ghi lại lý do loại từng query |
| `run_tkis.py` | Chạy T-KIS, tính Hit@K và MRR ở mức video lẫn mức khung hình |
| `run_trake.py` | Chạy TRAKE, chấm theo nhiều phương án GT và nhiều mức dung sai |
| `run_routing_ablation.py` | Chạy lại các query bị gán nhãn sai với `query_type_hint` ép sẵn |
| `run_latency.py` | Đo độ trễ theo từng cấu hình kênh |
| `report_metrics.py` | Tính đúng các bảng số trong chương Thực nghiệm của báo cáo |
| `cache_pools.py` | Chụp kho ứng viên (frame đã fuse + chuỗi TRAKE) của mọi truy vấn có GT xuống `benchmarks/cache/` |
| `run_answer_gen.py` | Chấm / dò tham số / kiểm chứng chéo bộ sinh 100 đáp án (`app/answer_gen.py`) |

## Tập đánh giá chính

Báo cáo lấy **tập truy vấn T-KIS mô tả một khoảnh khắc đơn** (45 câu) làm tập
chính, và tách riêng 22 câu mô tả một chuỗi nhiều mốc nối tiếp.

Lý do: T-KIS được định nghĩa là tìm *một* khoảnh khắc đã biết. Những câu liệt kê
nhiều hành động nối tiếp về bản chất là bài toán chuỗi sự kiện (TRAKE), chỉ là
chúng bị xếp nhầm nhóm khi soạn bộ truy vấn. Chênh lệch giữa hai nhóm rất lớn ở
R@1 (60,00% so với 18,18%) nhưng gần như bằng nhau ở R@100 — tức hệ thống vẫn tìm
ra đúng video cho nhóm mô tả chuỗi, chỉ không xếp được lên đầu.

Tiêu chí phân loại đọc nội dung câu hỏi và **không** phụ thuộc thứ hạng hệ thống
trả về. Đây là điểm quan trọng: nếu chia tập theo việc câu nào đạt điểm cao thì
con số báo cáo sẽ mất ý nghĩa. Cả hai nhóm đều được báo cáo kết quả.

## Hai điều đã làm hỏng kết quả một lần

**Đo độ trễ theo khối cho kết luận ngược.** Nếu chạy hết 25 request của cấu hình
A rồi mới sang cấu hình B, cấu hình chạy trước sẽ gánh chi phí làm nóng kết nối
tới Elastic Cloud và Milvus. Lần đầu chúng tôi đo được "cấu hình đầy đủ nhanh
ngang cấu hình chỉ có PE"; chạy lại với thứ tự đảo ngược thì cấu hình đầy đủ
thành chậm hơn gấp đôi. Dùng `--interleave` để mỗi vòng gọi lần lượt tất cả cấu
hình.

**`query_type_hint` có tác dụng, nhưng không tác dụng lên thứ hạng.** Khi thấy
`run_routing_ablation.py` trả về kết quả giống hệt baseline, phản xạ đầu tiên là
nghĩ tham số bị bỏ qua. Thực tế `query_type` trong JSON trả về có đổi — chỉ là
`/api/search` không dùng trường đó để quyết định bật kênh nào. Nếu sửa script,
hãy kiểm tra `forced_type` trong output trước khi kết luận có bug.

## Cảnh báo: `runs/tkis_full_llm.json` không phải kết quả của LLM

File này được sinh bằng `--config full_llm` (tức `use_llm=true`), nhưng kết quả
giống hệt `tkis_full.json` đến từng thứ hạng. Nguyên nhân: model cấu hình trong
`NVIDIA_MODEL` (`qwen/qwen3-next-80b-a3b-instruct`) đã hết vòng đời từ
2026-07-27 và trả về HTTP 410. `QueryParser._llm_parse` bắt mọi lỗi rồi trả
`None` nên hệ thống lặng lẽ quay về heuristic.

```bash
# Kiểm tra nhanh xem model còn sống không
curl -s -X POST "$NVIDIA_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -d "{\"model\":\"$NVIDIA_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":5}"
```

Các model thay thế còn sống trên cùng endpoint đều quá chậm cho phân tích truy
vấn: `meta/llama-3.3-70b-instruct` và `meta/llama-3.1-70b-instruct` không xong
trong 180 s với schema đầy đủ, `nvidia/llama-3.3-nemotron-super-49b-v1.5` mất
32 s với schema rút gọn (timeout mặc định của client là 40 s). Vì vậy mọi số
liệu trong báo cáo đều đến từ bộ phân tích heuristic.

## Ghi chú về ground truth

- T-KIS: 67/87 query có GT dùng được, 272 cặp `(video, frame)` duy nhất. Giữ lại
  các dòng `DBC=0` vì đó là câu khó có GT, loại đi sẽ làm tập đánh giá dễ hơn thực tế.
- TRAKE: 7/11 query có GT; hai query có hai phương án chuỗi nên phải chấm theo
  `max` trên các phương án.
- QA: 9/15 query có GT nhưng chưa có chương trình chấm — đáp án phụ thuộc video và
  có alias, nên chưa có số trong báo cáo.

Progressive PHM benchmark: see [progressive/README.md](progressive/README.md) for frozen translations, target-video splits, fair Hint-RRF, dev/evaluation execution and metrics.
