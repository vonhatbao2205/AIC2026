# NVILA-8B cho QA — phương pháp, triển khai Colab và tích hợp AIC26

## 1. Phương pháp được áp dụng

Thiết kế bám theo pipeline QA của đội NII-UIT tại VBS 2026 trong
[`Nvila/_MMM2026__NII_UIT_at_VBS2026__Towards_Effective_Visual_Question_Answering_for_Interactive_and_Multimodal_Video_Retrieval.pdf`](Nvila/_MMM2026__NII_UIT_at_VBS2026__Towards_Effective_Visual_Question_Answering_for_Interactive_and_Multimodal_Video_Retrieval.pdf):

1. **Answer Span/Hotspot Prediction**: lọc các vùng thời gian có khả năng chứa câu trả lời.
2. **Candidate Answer Suggestion**: sinh nhiều đáp án ngắn có dẫn chứng thay vì trả một câu tự do duy nhất.
3. **In-video verification**: đưa operator đến đúng frame/video/timeline để kiểm chứng.
4. **Human in the loop**: model chỉ hiển thị lựa chọn; chỉ sau khi operator click
   thì đáp án mới được điền nháp và frame dẫn chứng mới được mở.

Hệ thống hiện chưa có dense caption cho toàn bộ keyframe. Vì vậy, bước đầu được
thay bằng cách cho NVILA-8B nhìn trực tiếp một candidate pack đa dạng lấy từ kết
quả PE-G14 + OCR + ASR + audio. Đây là adaptation có chủ đích, không giả định
rằng OCR/ASR là ground truth. Kiến trúc và khả năng multi-image/video của model
được đối chiếu với [`Nvila/2412.04468v3.pdf`](Nvila/2412.04468v3.pdf),
[repository chính thức](https://github.com/NVlabs/VILA) và
[checkpoint NVILA-8B-hf](https://huggingface.co/Efficient-Large-Model/NVILA-8B-hf).

## 2. Luồng chạy thực tế

```text
QA query
   │
   ├─ backend retrieval: PE-G14 + OCR + ASR + audio → RRF/group-by-video
   │
   ├─ frontend chọn tối đa 12 frame
   │    • frame operator đang xem luôn là C01 và không bị lặp
   │    • từ C02, lấy đủ quota frame đa dạng của top video trước khi sang video khác
   │    • tối đa 3 frame/video; C01 được tính vào quota của video chứa nó
   │    • ưu tiên giãn cách ≥2 giây để giảm frame gần trùng
   │
   └─ POST /api/qa/analyze (chỉ canonical keyframe IDs)
         │
         ├─ backend dựng lại trusted R2 image URLs
         ├─ Colab NVILA pass 1: 3–5 answer-bearing hypotheses
         ├─ Colab NVILA pass 2: 3–5 grounded answer alternatives
         ├─ backend DeepSeek web_search khi cần tri thức ngoài
         ├─ Colab NVILA pass 3: kiểm tra visual consistency của đáp án web
         └─ UI: toàn bộ input + đáp án/citation + frame/timeline dẫn chứng
                                      │
                                      └─ operator kiểm chứng → Submit Guard
```

Các pass dùng greedy decoding để kết quả ổn định. Output JSON được lọc ở worker
và backend:
worker chỉ giữ candidate ID hợp lệ; backend tiếp tục canonicalize, bỏ output
confidence 0, placeholder và mọi ID model tự bịa. Stage web chỉ nhận question,
OCR/ASR/audio cue và các hypothesis của NVILA; nó không nhận quyền tạo frame.
Sau stage web, đáp án và citation được gửi ngược về NVILA cùng tối đa 8 hotspot
images. NVILA chỉ kết luận `supported`, `contradicted` hoặc `insufficient` về
tính nhất quán thị giác; nó không thay thế việc đánh giá độ uy tín của website.

## 3. Chạy worker trên Google Colab A100

Mở [`aic26_nvila8b_qa_colab_server.ipynb`](aic26_nvila8b_qa_colab_server.ipynb),
chọn **Runtime → Change runtime type → A100 GPU**, rồi chạy từ trên xuống.

Colab Secrets:

| Secret | Bắt buộc | Công dụng |
|---|---:|---|
| `AIC26_NVILA_TOKEN` | Có khi dùng live | Bearer token riêng giữa backend và worker |
| `HF_TOKEN` | Không | Tải checkpoint ổn định hơn; model hiện là public |

Notebook tải `cloudflared` thẳng vào `/content/cloudflared` và gọi bằng absolute
path, nên không phụ thuộc `apt`, `nix-index` hay PATH. Notebook không upload
dataset/model/output. NVILA tải candidate image qua URL R2 mà backend dựng sẵn.

Trên A100, model chạy native **BF16** với SDPA. Notebook bật TF32 cho các phép
toán FP32 phù hợp, giới hạn 12 candidate, 5 hotspot và cache 64 request. Giới
hạn 12 là lựa chọn chất lượng/độ trễ, không phải giới hạn VRAM của A100. Notebook
không dùng lượng tử hóa 4-bit để tránh giảm chất lượng và các nhánh load tùy phiên bản.

Cell cuối in ra `PUBLIC_URL`, ví dụ `https://random-words.trycloudflare.com`.
Quick tunnel đổi URL mỗi lần restart và Colab phải còn kết nối.

## 4. Nối backend với worker

Thêm vào `backend/.env`, rồi restart FastAPI:

```dotenv
NVILA_BASE_URL=https://random-words.trycloudflare.com
NVILA_TOKEN=<đúng giá trị AIC26_NVILA_TOKEN trên Colab>
NVILA_TIMEOUT_SECONDS=240
NVILA_MAX_CANDIDATES=12

# Optional: world-knowledge completion after NVILA
DEEPSEEK_API_KEY=<key tạo tại platform.deepseek.com>
DEEPSEEK_GROUNDING_MODEL=deepseek-v4-flash
DEEPSEEK_GROUNDING_ENABLED=true
DEEPSEEK_GROUNDING_MAX_OUTPUT_TOKENS=8000
DEEPSEEK_GROUNDING_REASONING_EFFORT=high
DEEPSEEK_GROUNDING_AUTO_THRESHOLD=0.55
```

Không đưa `NVILA_TOKEN`, `HF_TOKEN`, `DEEPSEEK_API_KEY` hay bất kỳ secret nào vào
frontend. `DEEPSEEK_API_KEY` cũng không cần đặt trong Colab. Browser chỉ gọi
backend `/api/qa/analyze`; backend mới gọi Colab và DeepSeek.

Worker Colab có hai route nội bộ dùng cùng Bearer token: `POST /qa/analyze` cho
pass 1–2 và `POST /qa/verify-grounded` cho pass 3. Backend gọi cả hai; frontend
không gọi worker trực tiếp.

DeepSeek dùng built-in `web_search` chạy phía server, không cần search engine
riêng. Tool này **chỉ có trên Responses API** (`POST /responses`) và **chỉ chạy
với `deepseek-v4-flash`**; `/chat/completions` trả lỗi `unknown variant` cho mọi
tool khác `function`. Tài liệu chính thức:
[DeepSeek Responses API](https://api-docs.deepseek.com/api/create-response/).

Thinking bật mặc định và dùng chung `max_output_tokens` với câu trả lời — đặt
budget quá thấp sẽ nhận về `status: "incomplete"` với message rỗng.

Kiểm tra kết nối:

```bash
curl http://localhost:8000/api/health
```

Khi worker thật sự trả health thành công,
`capabilities.qa_nvila`, `qa_hotspot_prediction` và `qa_candidate_answers` là
`true`. Worker tắt hoặc tunnel hết hạn không làm hỏng core retrieval; UI báo
worker offline và QA vẫn có thể trả lời thủ công.

## 5. API contract

Frontend gửi:

```json
{
  "question": "Người dẫn chương trình đang cầm vật gì?",
  "candidates": [
    {
      "submit_keyframe_id": "K01/K01_V001/001",
      "frame_idx": 125,
      "pts_time": 5.0,
      "retrieval_score": 0.91,
      "evidence": [{"type": "ocr", "text": "..."}]
    }
  ],
  "max_answers": 5,
  "web_grounding": "auto"
}
```

Backend trả `candidate_answers`, `hotspots`, `best_answer`,
`best_submit_keyframe_id`, `uncertainty`, model và latency. Mỗi đáp án có
`source` (`nvila`, `google`, `hybrid`), `supporting_frames`, `web_sources` và
`visual_verification`; `web_grounding` chứa search queries, citation, trạng thái,
verdict pass 3, rejected answers và latency.

`auto` gọi search khi đáp án visual không có/độ tin cậy thấp, hoặc question cần
resolve tên người, thương hiệu, cửa hàng, công ty hay địa danh và ASR/OCR có clue.
`on` buộc thử search; `off` tắt. Ví dụ ASR chỉ nói “logo Disney”, stage web có
thể resolve canonical form “Walt Disney” rồi áp dụng format mà question yêu cầu.
Backend sau đó gửi “Walt Disney” về NVILA: nếu clue Disney phù hợp thì giữ/rank;
nếu hình hoặc ASR mâu thuẫn thì loại; nếu không đủ chứng cứ thì cap confidence.

## 6. Cách dùng trong UI

1. Chọn task **QA** và search như bình thường.
2. Có thể chọn frame đáng chú ý; frame được bảo đảm có mặt nhưng không được ưu
   tiên vị trí hơn candidate của video khác.
3. Bấm **Analyze visual candidates**.
4. Kiểm tra đủ 12 input candidates, nhiều answer option, badge nguồn và citation.
5. Click một answer option để điền nháp và mở frame dẫn chứng; hệ thống không tự chọn.
6. Xem badge `NV`, hotspot trên timeline, video và OCR/ASR liên quan.
7. Sửa đáp án nếu cần rồi submit qua guard.

Model không tự submit. Confidence là tự đánh giá của model, không phải xác suất
đã hiệu chuẩn. Với câu hỏi cần sự kiện kéo dài, retrieval phải đưa đủ frame vào
candidate pack; nếu chưa đủ, mở rộng query hoặc tìm trong video theo timeline.

## 7. Vận hành và bảo mật

- Backend không nhận `image_url` từ browser cho luồng này; URL được dựng từ
  `submit_keyframe_id` hợp lệ, giảm nguy cơ SSRF qua worker.
- Worker giới hạn số candidate, kích thước ảnh, timeout download và yêu cầu
  Bearer token.
- Không log hoặc trả secret về frontend.
- DeepSeek không trả widget search suggestions; trường này luôn rỗng.
  citation mở sang tab mới. Web result không bao giờ được dùng như frame submit.
- Quick tunnel phù hợp cho phiên thi/thử nghiệm. Muốn URL ổn định cần named
  Cloudflare Tunnel hoặc GPU host cố định.
- Checkpoint có giấy phép hướng nghiên cứu/non-commercial; kiểm tra lại license
  trước mục đích khác.

## 8. Kiểm thử

```bash
cd backend
PYTHONPATH=. python -m pytest -q

cd ../frontend
npm test
npm run build
```

Mock mode có kết quả NVILA cố định để test contract/UI mà không gọi mạng. Colab
notebook có cell smoke test tùy chọn (`RUN_SMOKE_TEST=True`) để kiểm tra trọn
luồng ảnh → hai pass QA → pass 3 verification → JSON trước khi nối backend.
