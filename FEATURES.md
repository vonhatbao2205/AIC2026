# AIC26 Retrieval Console — Danh mục tính năng

Toàn bộ tính năng hiện có của hệ thống truy hồi video AIC26 (backend FastAPI + frontend React/Vite). Tài liệu vận hành/contract chi tiết: [RETRIEVAL_APP_HANDOFF.md](RETRIEVAL_APP_HANDOFF.md).

---

## 1. Kiến trúc tổng quan

- **Backend** `backend/` — FastAPI, adapter tách module: Elastic, Milvus, PE encoder, GLAP encoder; service: search / trake / timeline / submit; logic thuần (identity, media, fusion, trake, scoring) test được độc lập.
- **Frontend** `frontend/` — React + Vite + TypeScript. Frontend **chỉ gọi backend**, không giữ secret.
- **Models/infra** (Kaggle/Colab + cloud): PE-Core-G14 (ảnh+text), GLAP (audio↔text), NVILA-8B (visual QA), Milvus/Zilliz (vector), Elastic Cloud (OCR/speech/audio/keyframe-map), Cloudflare R2 (media), NVIDIA NIM (LLM parse/expansion), faster-whisper (voice STT).
- **Mock mode** (`AIC26_MOCK_MODE=true`): mọi adapter trả fixture cố định → chạy UI/test không cần service thật.

## 2. Quy tắc định danh (bất biến)
- `submit_keyframe_id` = `"<category>/<video_id>/<frame_3_digits>"` (vd `K01/K01_V001/001`), cũng là `image_id`.
- **Submit dùng `video_id` + `frame_idx`** (đúng format DRES) — KHÔNG dùng `image_path`/`source_path`.
- Chuẩn hoá shard **L26_a/L26_b → L26**.
- Media URL build từ `submit_keyframe_id`/`video_id`+`keyframe_n`.

## 3. Hai chế độ giao diện
- **Console (đầy đủ)** ⇄ **Simple** — nút chuyển góc trên, nhớ lựa chọn trong `localStorage`.
- **Simple**: 1 ô query + thanh trượt số keyframe → lưới keyframe theo cosine (vector-only, không parser/fusion).
- **Console**: dashboard 3 cột keyboard-first (query/kênh/understanding · kết quả+video+timeline · detail/TRAKE/history).

## 4. Dark / Light mode
- Toggle sun/moon (SVG) ở top bar + Simple. Lưu localStorage, lần đầu theo `prefers-color-scheme`.
- **Soft dark** (không OLED gắt) + light dịu; ít border, dùng elevation; scrollbar mảnh; chữ rõ.

## 5. Hiểu & định tuyến query (Query understanding)
- **Heuristic (mặc định, ~0ms)**: phân loại query_type, bật kênh theo dấu hiệu (OCR-likely/speech/audio), tách negation, tách event TRAKE.
- **LLM tùy chọn** (toggle **🧠 LLM on**): NVIDIA NIM (mặc định `qwen/qwen3-next-80b-a3b-instruct`) trả JSON routing đầy đủ. Có cache theo query; JSON retry 1 lần; lỗi/chậm → fallback heuristic.
  - **Slim-schema** (opt-in `SLIM_PARSE=true`): nhanh ~2× nhưng A/B cho thấy đôi khi sai query_type → mặc định TẮT.
- **Dịch VI→EN** (mặc định bật, `TRANSLATE_TO_EN`): khi LLM off, dịch query → `image_pe` (PE cần tiếng Anh); OCR/speech giữ tiếng Việt. TRAKE dịch từng event.
  - **Checkbox "Dịch VI→EN"** cạnh nút Search (cả Console lẫn Simple, mặc định **bật**): tắt = tìm **đúng chữ đã gõ**, không gọi dịch, và nếu LLM parser đang bật thì bản rewrite tiếng Anh của nó cũng bị bỏ (query thị giác quay về tiếng Việt gốc). Dùng khi query đã là tiếng Anh, có tên riêng, hoặc bị dịch sai. Gửi kèm `translate` trong `/api/search`, `/api/search/simple`, `/api/search/trake`, `/api/query/parse`, `/api/answers/generate`. `TRANSLATE_TO_EN=false` vẫn tắt toàn cục — checkbox chỉ tắt được chứ không bật ngược lại.
- **Query expansion** (toggle **🔎 Expand**): model nhanh (`NVIDIA_FAST_MODEL`, vd `meta/llama-3.1-8b-instruct`) sinh 2-3 mô tả thị giác/ query → backend search từng biến thể, **fuse max-cosine** → tăng recall cho concept khó (vd "The Thing"). Lọc bỏ biến thể không phải tiếng Anh.
- **Panel Query Understanding**: type+confidence, EN visual (clamp "xem thêm"), kênh được route, negations.

## 5b. Phạm vi tìm kiếm (Search scope — lọc theo thư mục dữ liệu)
Bộ lọc **📂 phạm vi** cạnh nút Search: bấm vào ra **danh sách checkbox** các thư mục dataset, tick nhiều thư mục tuỳ ý. Áp dụng cho lần Search **kế tiếp** (giống slider retrieval depth, không tự chạy lại).
- **Danh mục theo profile**: `btc` = L21–L30 + K01–K20 (30 thư mục); `infoshotpp` = L21–L30 (10 thư mục — không có K nên không hiện). Đổi profile sẽ tự bỏ thư mục profile mới không có.
- **3 chế độ**: **Tự động** (mặc định — heuristic chủ đề), **Tất cả**, **Tùy chọn** (đúng những gì đã tick). Tick 1 ô bất kỳ = chuyển sang Tùy chọn.
- **Heuristic chủ đề** (`backend/app/scope.py`, giống query routing): mỗi thư mục là **một chương trình**, nên query nói về chủ đề gì thì đã chỉ ra thư mục — `đầu bếp / công thức / nước sốt` → L26, `đua xe đạp / tay đua / chặng đua` → L23, `múa lân / lân` → L24, `ôn thi / đề thi / trắc nghiệm` → L25, `làng nghề / nghệ nhân` → L27, `mekong / miền tây / chợ nổi` → L28+L29, `từ thiện / trao quà / lan tỏa` → L30, `thời sự / bản tin / 60 giây` → L21+L22+K01–K20.
- **Chỉ dùng từ khoá về ĐỊNH DẠNG chương trình, không dùng từ về chủ thể**: `tôm`, `bánh`, `cá` bị loại khỏi từ điển vì bản tin thời sự cũng quay đồ ăn — để chúng vào làm heuristic chọn nhầm L26 cho 6 query K-side.
- **Không bao giờ loại các thư mục "mọi chủ đề"** (L21, L22, L30, K01–K20): thời sự 60 giây đưa tin về nấu ăn/đua xe đạp/múa lân, L30 là clip ngắn tự gửi nên chứa mọi thứ. Đo trên 104 query ground truth (`TKIS/QA/TRAKE_queries.xlsx`): heuristic kích hoạt 22 lần, **thư mục chứa đáp án luôn nằm trong phạm vi**; bỏ luật này thì mất đáp án 6 lần.
- **Match có dấu** khi query có dấu: fold cả 2 phía làm `vẫy tay đưa` trùng `tay đua` (L23) và `cụ lão` trùng `cù lao` (L28) → chọn nhầm. Query gõ không dấu (`nau an`) vẫn match qua fold.
- **Nút "Chỉ L26"**: một click để bỏ luôn phần thời sự/L30 khi thao tác viên chắc chắn (chuyển sang Tùy chọn với đúng thư mục của chủ đề).
- **Push-down, không phải lọc sau**: phạm vi đi thẳng vào Milvus (`video_id like "L26_%"`) và Elastic (`prefix` trên `video_id`, trong `filter` nên không đổi điểm), nên `top_k` được lấp **từ trong phạm vi** thay vì lấy toàn cục rồi bị gọt. Vẫn lọc lại sau fusion cho chắc. Áp dụng cho `/api/search`, `/api/search/trake`, `/api/search/canvas`, `/api/search/simple` — TRAKE dùng **một phạm vi cho cả chuỗi** (mọi event phải cùng 1 video).
- Chọn hết = không lọc: backend chuẩn hoá về "không filter" để adapter không phải dựng mệnh đề vô nghĩa.

## 6. Kênh truy hồi (Retrieval channels)
- **image_pe (VECTOR)**: PE-Core-G14 encode query → Milvus `aic26_image_peg14_v1` (COSINE). Hỗ trợ **multi-variant max-fusion** (query expansion).
- **image_qwen (VECTOR, InfoShot++)**: Qwen3-VL-Embedding-8B encode query → Milvus `aic26_image_qwen3vl8b_infoshotpp_v3` (COSINE, 4096-d native). Ô chọn **Image embedding** trong query panel cho phép chạy PE, Qwen, hoặc cả hai; chọn cả hai thì hai kênh **fuse bằng RRF**, không bao giờ cộng cosine của hai không gian khác nhau. BTC không có vector Qwen nên request BTC bị từ chối 422 thay vì âm thầm trả kết quả PE.
- **OCR**: Elastic `text_clean`/`text_nfc`/`text_clean_fold` (fuzzy) + filter `hour`/`clock`. Guard: query rỗng → trả rỗng (không match-all).
- **speech**: Elastic ASR `text`; hạ điểm `confidence_bucket` low/mid, `segment_role` intro/preview.
- **audio**: Elastic `top1_label`/`tag_labels`/`caption` (drop stoplist, hạ generic/vietnamese_asr) **FUSE (rank-RRF) với GLAP audio-vector** (`aic26_audio_glap_v1`, COSINE) khi GLAP bật. Frame do vector tìm có **badge `GLAP`** trên kết quả + evidence.
- Toggle bật/tắt từng kênh thủ công (force/disable), hiển thị weight + "auto".

## 7. Fusion & hiển thị kết quả
- **RRF (k=60)** trên `submit_keyframe_id`, có trọng số kênh; gom evidence mọi kênh.
- **Group-by-video** với điểm chuẩn hoá **coverage-first**: `0.75·best + 0.15·mean_top3 + 0.10·cluster_support` (đều [0,1]); video có frame mạnh không bị video nhiều-frame-yếu chôn. Cờ **AMBIGUOUS** khi top frame tách cụm thời gian xa.
- **2 chế độ xem**: **Group by video** (mở sẵn tất cả group, badge kênh, feedback) ⇄ **Flat top-K** (lưới phẳng theo điểm). Bù `pts_time`/`frame_idx`/`fps` bằng 1 lần Elastic `_mget`.
- **Frame trong group xếp theo thời gian — MẶC ĐỊNH.** Việc operator làm với một group là *đọc* một cảnh, mà strip lộn xộn thứ tự cảnh khó phán đoán hơn hẳn strip lộn xộn thứ hạng. Nút **↺** đổi về thứ tự độ liên quan, **⏱** trả lại mặc định; áp dụng cho **đúng group đó**, các group khác giữ nguyên; search mới thì về lại mặc định.
  - **Chỉ gắn nhãn cho trường hợp ngoại lệ** (`theo độ liên quan`) — dán nhãn lên mọi group là nhiễu, operator sẽ ngừng đọc nó.
  - **Đánh dấu đỏ `#1 #2 #3` các frame mạnh nhất trong group.** Cái mà thứ tự thời gian lấy đi chính là thứ hạng: lý do video này có mặt trên màn hình bị rải khắp strip. Nên phải đánh dấu tại chỗ. Selection (viền accent) vẫn thắng vì nó nói cái mà submit guard sắp thao tác.
  - Xếp theo `pts_time`, thiếu thì `frame_idx`, thiếu nữa thì `keyframe_n` (cả 3 đều tăng theo thời gian trong 1 video). Frame không có thời gian nào thì **giữ nguyên thứ tự relevance ở cuối** — không biết vị trí thì không phải là vị trí 0.
  - Ranking gốc **không bị đổi**: thứ tự hiển thị là view phái sinh, nên đổi qua lại chỉ là thêm/bớt id trong một set (không có bản copy nào để bị stale).
  - **Selection bám theo frame, không bám theo vị trí**: `selectedFrame` là chỉ số trong strip, nên khi đổi thứ tự thì con trỏ được trỏ lại đúng frame cũ — nếu không, Detail / timeline / submit guard sẽ mô tả một frame khác với frame đang được highlight.
  - **TRAKE bị loại hẳn khỏi cả mặc định lẫn nút bấm**: ở TRAKE mỗi frame LÀ event thứ i, và DP đã đảm bảo thời gian tăng dần, nên sắp xếp strip chỉ có thể làm rối cách hiểu chứ không đổi được gì.
- **Channel attribution**: mỗi frame có badge kênh + "explain match" (evidence snippet, điểm, thời gian).
- **Relevance feedback**: nút **More like this** (boost video) / **Exclude** (loại frame) → re-rank phiên search.

## 8. 4 dạng tác vụ
- **T-KIS**: hỗ trợ **Append hint** (gộp hint tích lũy).
- **QA**: retrieval lấy tối đa 12 frame; frame user đang xem luôn là **C01** và được tính vào quota của video đó. Từ C02, candidate được xếp theo block ưu tiên `video_score`: lấy đủ tối đa 3 frame đa dạng của video đứng đầu trước rồi mới sang video hạng tiếp theo. Sau đó NVILA-8B chạy **3–5 hotspot hypotheses → 3–5 visual answer options** → DeepSeek web search → **NVILA pass 3**. `contradicted` bị loại; `insufficient/unverified` bị hạ confidence.
- **V-KIS**: như KIS hình ảnh, **cộng canvas vẽ bố cục** (chi tiết §8b).
- **TRAKE** (chi tiết §9).

## 8b. Canvas V-KIS (vẽ bố cục → tìm frame)
- **Bảng vẽ 16:9 nằm ở cột giữa** (cột rộng nhất) khi chọn task V-KIS, cao tối đa 46vh, thu gọn được bằng nút ▾. Toàn bộ mặt vẽ là một `<canvas>` thật.
- **3 công cụ**: `✥` chọn/kéo/resize object · `✎` **vẽ tự do** (bầu trời, cánh đồng, mặt nước…) với 16 màu + độ dày nét + hoàn tác từng nét · `⌫` tẩy nét.
- **Object vẽ bằng hình cụ thể**, không phải ô chữ: người/đám đông/xe/xe hai bánh/thuyền/máy bay/tòa nhà/màn hình/micro/cờ/biển-giấy tờ/cây/phong cảnh/lửa/bàn ghế/bát đĩa + fallback. Silhouette là thứ làm ảnh render có nghĩa với PE image encoder.
- Palette object lấy từ `GET /api/canvas/palette` = **đúng vocabulary OD đã prompt**; màu đúng 16 màu `aic16-lab-v1`; class OD không trích màu (vd `crowd`) thì picker tự ẩn.
- **required/optional** mỗi object; `Rough` (mặc định, ưu tiên nhãn + vị trí tương đối) ⇄ `Precise` (tăng IoU/size/màu).
- **3 kênh, cùng một canvas**:
  1. **OD spatial** trên `aic26_od_frames_v1` + **Hungarian one-to-one** giữa object vẽ và detection thật (2 người vẽ không thể cùng khớp 1 detection);
  2. **PE text** sinh bằng rule từ canvas JSON (0 ms, không hallucinate) → Milvus;
  3. **PE image** (bật/tắt bằng ô `PE image`): render canvas thành PNG **không có lưới/handle/nhãn**, gửi `{PE_ENCODER_URL}/encode-image` → Milvus cùng collection keyframe. Weight thấp (0.2) — cứu những thứ OD không có nhãn (cánh đồng, bầu trời) chứ không dẫn dắt xếp hạng.
  Ba kênh fuse RRF như search thường. Không gọi VLM.
- **Overlay giải thích**: detection khớp được vẽ đè lên keyframe trong Detail đúng màu với box đã vẽ, kèm coverage %, conf, vị trí, cờ lệch màu và danh sách object không tìm thấy.
- **PE text hiển thị lại** cho operator kiểm chứng câu truy vấn được sinh.
- PE server cần route `/encode-image` (cell mục 7 trong `model-setup-backend.ipynb`, hot-add không phải restart). Thiếu route → backend chỉ cảnh báo, 2 kênh kia vẫn chạy.

## 9. TRAKE (chuỗi sự kiện)
- **Tách event**: nhận `E1:/E2:`, `sự kiện 1:`, đánh số, từ nối ("sau đó/rồi/…").
- **Đơn vị của TRAKE là VIDEO, không phải keyframe.** Ngay sau pass 1, kết quả từng event được gấp thành `video_id → ứng viên mỗi event` (`build_video_event_map`). Bản đồ đó — chứ không phải danh sách chuỗi — là biểu diễn trung gian mà mọi bước sau đọc: chọn video cho pass 2, DP, bản đồ nhiệt và xếp hạng. Trước đây mỗi bước tự group theo video một lần, và mọi ứng viên không nằm trong chuỗi tốt nhất bị vứt **trước khi** giao diện nhìn thấy.
- **Temporal NMS trước DP**: 12 frame cách nhau 0,2 s là **một** khoảnh khắc. Lấy top-N theo điểm tiêu hết ngân sách vào một cụm và loại mất cảnh khác ở phút sau — đúng cái chuỗi cần. Nay giữ tối đa 8 **đỉnh cách nhau ≥ 2 s**, rồi mới lấp nốt bằng các frame gần kề (một frame sớm hơn 0,3 s vẫn có thể là frame duy nhất chèn vừa thứ tự).
- **Two-pass retrieval**:
  1. Truy hồi từng event (song song, pool rộng `top_k≈400`).
  2. **DP sơ bộ quyết định pass 2** (`select_pass2_gaps`): "event thiếu" là event **DP chưa xếp được vào chuỗi**, KHÔNG phải event không có ứng viên. Video có E1 chỉ khớp ở 90 s và E2 chỉ khớp ở 10 s thì mọi event đều có ứng viên nhưng chuỗi vẫn 1/2 — và đây mới là lỗi phổ biến (retrieval tìm đúng hành động nhưng ở sai đoạn video). Hỏi "event này đã có ứng viên chưa?" khiến pass 2 mù với nó: video trông như đã phủ đủ nên không bao giờ được search lại.
  3. **Chọn video cho pass 2**: tier theo số event **chuỗi** còn thiếu (thiếu 1 → chỉ cách một truy vấn là đủ chuỗi, đi trước), trong cùng tier xếp theo `trake_video_score` sơ bộ. Luật cũ xếp theo *số* event có ứng viên nên coi "phủ E1,E2,E3 rất yếu" ngang "phủ E1,E2,E3 rất mạnh"; video thiếu nhiều event rơi xuống cuối và chỉ chạy khi còn ngân sách.
  4. **In-video fill đưa GIẢ THUYẾT, DP mới là nơi chọn**: search lại đúng video đó (Milvus filter `video_id`) cho event thiếu, có ngưỡng tương đồng tương đối, rồi **thêm tối đa 3 ứng viên/gap** cho DP giải. Trước đây tự chọn một hit rồi mới đưa DP — mà retrieval đã trả top-3 rồi, và chốt sớm thì vứt phần còn lại trước khi DP kịp nhìn: E2 mạnh nhất ở 80 s cộng E3 mạnh nhất ở 20 s là chuỗi chết, trong khi E2@30 + E3@70 thì đủ chuỗi. Frame fill được đánh dấu `via_fill`.
  - Cửa sổ thời gian `[min(candidate event trước), max(candidate event sau)]` chỉ là **điều kiện cần, không đủ** — E1@90, E2@100 thì ứng viên 95 vẫn lọt cửa mà không thể đứng sau E2. Nên nó chỉ dùng để **xếp thứ tự ưu tiên** các giả thuyết, không dùng để chọn.
  5. **Chất lượng fill là tỉ lệ, không phải thứ hạng**: chạy PE+Qwen thì RRF chỉ quyết định **thứ tự** (hai không gian vector không so cosine trực tiếp được), còn `fill_quality` giữ nguyên nghĩa model-local `score / hit tốt nhất của model đó`. Chuẩn hoá RRF theo đỉnh của chính gap khiến mọi ứng viên top thành `1.0` — frame vừa lách qua ngưỡng 45% bị báo cho operator là khớp hoàn hảo.
- **DP tối ưu**: max-weight strictly-increasing chain theo (event_index, pts_time) → 1 frame/event, thời gian tăng dần. Trạng thái so sánh theo **`(bằng chứng thật, coverage, chất lượng)`** — **đúng từng số hạng** với thứ tự mà `trake_video_score` mã hoá (`C·B² + T·B + Q`), nên chuỗi DP giao ra chính là chuỗi mà xếp hạng sắp chấm. Cho phép partial.
  - **Thứ tự quan trọng, không chỉ các số hạng.** `(coverage, real, …)` nhìn thì gần giống mà không phải: nếu E1/E4/E5 có bằng chứng thật còn E2/E3 chỉ fill được ở sau E4, nó sẽ chọn `E1 + fill E2 + fill E3 + E5` (coverage 4, real 2) thay vì `E1 + E4 + E5` (coverage 3, real 3) — rồi bảng xếp hạng chấm video đó theo `confident_coverage = 2`, tức trừ điểm chính vì cú đổi mà DP làm hộ nó. "Bằng chứng thật đè coverage do fill" là luật giữa các video, nên phải là luật bên trong một video luôn.
  - **"Bằng chứng thật" phải là một chiều riêng, không thể suy ra từ điểm số.** Fill được `0,02 × quality`, còn frame chỉ một kênh tìm ra ở rank 0 được `1/(60+1) = 0,0164`: **fill mạnh thắng bằng chứng thật trên tổng điểm thô**.
  - **Chất lượng dùng `strength` chuẩn hoá theo từng event**, không dùng điểm thô, để một event có thang điểm cao không tự quyết định cả chuỗi.
  - **Đánh đổi đã biết**: khi fill *thay chỗ* bằng chứng thật, DP giữ chuỗi thật ngắn hơn, nên video đó có thể **không sinh ra dòng nộp nào** (một dòng TRAKE chỉ hợp lệ khi đủ N event). Đây là lựa chọn có chủ đích: chuỗi kia dù đủ N thì cũng bị chính bảng xếp hạng dìm xuống vì thiếu bằng chứng thật.
- **`trake_video_score`** — một số duy nhất trong [0,1] mã hoá đúng thứ tự từ điển `(confident_coverage, coverage, chất lượng chuỗi)`. Với `B = n+1`: `(C·B² + T·B + Q) / (n·B² + n·B + 1)`, `Q = 0,70·mean + 0,30·min` các cường độ đã chuẩn hoá. Thêm **một** event phủ thật luôn thắng mọi khác biệt về cosine — đó là khác biệt cốt lõi giữa xếp hạng TRAKE và T-KIS. `min` có trọng số riêng vì chuỗi "3 event xuất sắc + 1 event vô vọng" không được đè chuỗi 4 event đều tốt.
- **`Q` tính trên MỌI mắt xích của chuỗi, kể cả fill** (khác `mean_score`/`min_score` của chuỗi — hai số đó cố tình chỉ nói về bằng chứng thật). `confident_coverage` đã thống trị xếp hạng rồi, nên loại fill khỏi `Q` chỉ vứt đi tín hiệu duy nhất phân biệt hai video cùng `C` và `T`: chuỗi khép lại bằng một fill thuyết phục hơn hẳn chuỗi khép lại bằng fill vừa đủ qua ngưỡng.
- **Xếp hạng theo evidence thật**: `confident_coverage` (event do retrieval thật phủ, **loại** event fill) thống trị → video 4/4 "giả" (nhiều fill) không đè được video có nhiều bằng chứng thật; clean full-coverage luôn lên đầu.
- **Cường độ chuẩn hoá theo TỪNG event**: `strength = score / G_e` với `G_e` là điểm cao nhất event đó đạt được trong toàn pool. Mỗi event là một truy vấn khác nhau, độ khó và mix kênh khác nhau — 0,020 có thể là gần hoàn hảo với E2 nhưng yếu với E1. Frame fill dùng chính `fill_quality` của nó (tỉ lệ cosine), không phải thang RRF.
- **Kết quả là thẻ VIDEO** (`TrakeVideoResults`): mỗi thẻ có **chuỗi ở trên** (một ảnh cho mỗi Ei — frame DP chọn; nếu DP không xếp được event đó thì vẫn hiện ứng viên mạnh nhất, badge `⚠ ngoài chuỗi`) và **mọi ứng viên ở dưới**, đặt đúng vị trí trên trục thời gian video. Backend chỉ gửi các đỉnh thưa, frontend vẽ.
- **Bản đồ nhiệt hiện THẲNG frame, không phải vạch màu**: mỗi ứng viên là chính keyframe của nó, hover phóng to. Trước đó muốn biết một phương án trông thế nào phải load video → tua → pause; nay chỉ cần liếc. Cụm frame chồng nhau là **thông tin** (đúng là cùng một khoảnh khắc), không phải lỗi layout.
- **Kéo-thả frame** (`text/x-trake-peak`): kéo bất kỳ frame nào từ bản đồ nhiệt **vào ô event ở sidebar** (nộp luôn frame đó — không cần load video) hoặc **lên thẻ Ei của chuỗi** (thay frame DP đã chọn, badge `đã chọn tay`, nút `↺` trả về lựa chọn của DP vì kết quả backend không bao giờ bị sửa). Ảnh đại diện trên chuỗi cũng kéo được xuống sidebar.
- **Một dòng TRAKE chỉ được lấy frame từ MỘT video — chặn ở hai tầng.** Dòng nộp nêu tên *một* video (lấy từ ô đầu tiên có frame) rồi liệt kê frame_idx dưới đó, nên frame của video khác sẽ được nộp **như thể** nó thuộc video này: một đáp án sai mà đúng format hoàn hảo, không tầng nào phía sau bắt được.
  - **Lúc gán**: kéo/gán frame khác video vào ô sẽ bị từ chối kèm toast nói rõ chuỗi đang dùng video nào.
  - **Ở submit guard**: kiểm tra lại một cách độc lập và **chặn nút nộp** (không chỉ cảnh báo), vì đó là tầng phải giữ nếu sau này có đường nào khác dựng được chuỗi mà bước gán không thấy.
- **Guard chặn cả chuỗi sai thứ tự thời gian**, không chỉ hiện cảnh báo: parser của BTC loại dòng sai thứ tự và **một dòng bị loại chặn cả bài nộp**.
- **Cảnh báo sai thứ tự ngay trên thẻ**: đổi frame bằng tay có thể phá thứ tự tăng dần; parser của BTC loại dòng sai thứ tự và **một dòng bị loại chặn cả bài nộp**, nên phải báo tại chỗ chứ không đợi tới lúc submit.
- **Đọc bản đồ nhiệt = việc DP không làm được**: DP chỉ biết chuỗi *xếp được*, không biết nó *hợp lý*. Đường chéo đẹp (E1 sớm → E2 → E3) đọc ra ngay là chuỗi thật; E1 nằm cách cụm E2-E3-E4 vài phút là dấu hiệu đáng kiểm tra trước khi nộp.
- **Click đỉnh nhiệt / ảnh đại diện Ei** → chọn frame đó + **arm slot Ei**. Đỉnh đang kiểm chứng **trở thành frame đang chọn**, nên detail panel, marker trên timeline và neighbour anchor đều nói về đúng khoảnh khắc đó. **Click KHÔNG mở video**: chọn một khoảnh khắc và xem nó là hai quyết định khác nhau, thẻ đã hiện sẵn frame rồi, nên một cú click không được tốn một lần tải video và một cú nhảy layout. Bấm `v` mới mở, và nó mở đậu đúng tại khoảnh khắc vừa chọn.
- **Trục thời gian là độ dài thật của video** (`duration_s`, lấy bằng một aggregation trên keyframe map cho cả trang kết quả). Vẽ theo đỉnh cuối cùng sẽ âm thầm phóng "cả 4 event nằm trong 1/3 đầu video" thành "trải khắp video".
- **Đỉnh nhiệt là control, không phải biểu đồ**: frame 72×44 px, vùng bấm rộng hơn khung ảnh và **cố định** (không co theo điểm), nên ứng viên mờ cũng dễ bấm/kéo như ứng viên mạnh — quan trọng trên laptop 13" khi đang chạy đua với đồng hồ. Điểm số làm mờ **viền**, không làm mờ ảnh: ứng viên yếu vẫn là frame operator phải đọc được.
- **Cụm frame sát nhau xếp sang lane thứ hai, không đè lên nhau** (`lib/trakeHeatLayout.ts`). Đặt frame thuần theo timestamp thì trung thực nhưng không dùng được: ba ứng viên cách nhau 11 giây trong video 10 phút nằm chồng lên nhau, chỉ frame vẽ sau cùng bấm được, hai frame dưới **không với tới** — mà đó đúng là lúc operator cần so sánh chúng nhất. Chật quá cả hai lane thì frame mới bị đẩy dọc theo lane, và **mỗi ứng viên luôn để lại một vạch mốc ở đúng thời điểm thật** nên hàng vẫn đọc ra được mẫu thời gian.
- **TRAKE không có nút ⬆/⬇ và không có FeedbackBar**: `/api/search/trake` không nhận feedback, nên các nút đó chạy lại search mà không đổi gì trong khi báo với operator là đã đổi. Control không có tác dụng còn tệ hơn không có control.
- **Badge coverage** `4/4 / 3/4 events` + chip `+N in-video` khi có fill + `q` chất lượng chuỗi + cảnh báo `⚠ yếu N%` (event yếu nhất) và `⚠ N s giữa 2 event` (chuỗi bị nén vào một khoảnh khắc — chẩn đoán, **không** vào xếp hạng).
- **Submit nhanh**: video đủ event → nút **Submit sequence ↵** điền hết slot + mở guard.
- **Frame-pick là tính năng global** (phím `v` mở video): lấy **frame_idx = round(pts×fps) ngay tại điểm đang xem** (KHÔNG snap về BTC keyframe) và thumbnail trực tiếp từ video (canvas). T-KIS/QA/V-KIS tự dùng raw frame này làm submit target (có thể chuyển lại result keyframe); TRAKE kéo thả/Enter để gán vào event slot.
  - **Origin video tách khỏi origin keyframe.** L21–L30 phát video qua named Cloudflare tunnel: đo trên 10 video, TTFB median **63 ms so với 834 ms** của HF bucket ở range đầu và **59 ms so với 831 ms** khi tua giữa file, throughput 13,1 MB/s so với 1,2 MB/s. HF trả **302 mỗi range request** rồi mới sang CDN, mà player tua thì phát range liên tục nên trả cái round-trip đó liên tục.
  - **HF vẫn là fallback, không bị bỏ.** Nhanh và luôn-sống là hai tính chất khác nhau: tunnel kết thúc ở một máy có thể bị tắt, còn bucket là hosting luôn bật. `/api/health` trả về cả hai origin; video lỗi tải thì player **thử lại đúng đường dẫn đó ở origin kia, đúng một lần, mang theo vị trí đang xem**, rồi console mở các video sau thẳng từ fallback thay vì mỗi clip lại tốn một request để phát hiện lại là origin chính vẫn chết. Cấu hình hai origin trùng nhau bị từ chối — đó là single point of failure khoác áo dự phòng.
  - **Mở video thì tự cuộn tới nó.** Player nằm bên trong đúng thẻ kết quả của nó, nên một video có hàng trăm keyframe sẽ đẩy player xuống dưới màn hình — bấm `v` rồi phải tự lướt đi tìm, mất mấy giây thật khi đang chạy đua đồng hồ. Áp dụng cho cả dải keyframe lân cận (`k`) vì nó nằm cùng khối. Tôn trọng `prefers-reduced-motion`.
  - **Video mở ra ở trạng thái DỪNG ngay tại frame đang chọn, không tự chạy.** Autoplay đi thẳng khỏi đúng cái frame vừa mở ra để xem, nên muốn lấy nó lại phải đuổi theo playhead bằng một lần pause. `Space` chạy/dừng khi thật sự muốn xem.
  - **Mọi lần tua khi video đang dừng CHÍNH LÀ một lần chọn frame** — ghi nhận ngay, không cần chạy rồi pause. Trước đây capture chỉ gắn vào sự kiện `pause`, nên chọn frame nghĩa là cho chạy rồi bấm dừng, và cái nhận về là chỗ video đã chạy tới lúc lệnh dừng có hiệu lực, **không phải frame đã tua tới** → lệch `frame_idx`. Tinh chỉnh bằng `a`/`d`, mũi tên, click timeline hay click frame trên bản đồ nhiệt đều là tua trên video đang dừng, nên đây đúng là đường phải chính xác nhất.
  - Tua **trong lúc đang chạy** thì không ghi nhận: đó là thao tác di chuyển, không phải chọn frame.
- **Kéo-thả keyframe giữa các ô event** để sắp xếp lại thứ tự (swap); validate thứ tự tăng dần, cảnh báo `⚠ order`.
- **Auto-fill E1..En** từ video đang chọn; marker event màu trên timeline.

## 10. Submit lên DRES (Client API v2) & lịch sử
- **Đăng nhập & phiên**: backend `POST /api/v2/login` (user/password trong `backend/.env`) → `sessionId`, mọi call sau đính `?session=…`; gặp 401/403 thì tự login lại **1 lần** rồi retry (hết hạn phiên không làm mất lượt nộp). Frontend không bao giờ thấy mật khẩu.
- **Đề bài tự fetch từ DRES**: panel "Đề bài" ở đầu cột trái hiện nguyên văn câu hỏi/mô tả của task đang mở (endpoint `…/template/task/{taskTemplateId}/hint` — nằm trong full `/openapi.json`, KHÔNG có trong `clientapi.json`), kèm media hint (ảnh/clip cho V-KIS). **Tự điền vào ô truy vấn** khi ô còn trống hoặc còn đề bài của task cũ, **không bao giờ đè chữ operator đã gõ**; có nút `→ dùng làm truy vấn` / `copy` / `⟳`. Cache theo task template nên poll không tải lại media.
- **DRES bar** (dưới top bar): trạng thái kết nối, user, chọn evaluation run (mặc định *auto* theo loại truy vấn: `tkis` / `qa` / `vkis` / `trake`), **tên task đang mở**, `taskStatus`, đồng hồ đếm ngược; poll 5 s + nút refresh/reconnect.
- **Định dạng answerSets đúng chuẩn v2** (`{"answerSets":[{"taskName","answers":[…]}]}`):
  - **T-KIS / V-KIS**: 1 answer `{mediaItemName, start, end}` — **milli-giây**, `mediaItemName` không có `.mp4`/thư mục.
  - **TRAKE**: mỗi event 1 answer temporal, đúng thứ tự, trong cùng answer set.
  - **QA**: **một** answer mang cả bốn trường `{mediaItemName, start, end, text}` — đúng ví dụ payload BTC đưa: xác định đoạn video **và** trả lời. (Vẫn có mode `text only` để đè khi cần.)
  - Cửa sổ = thời điểm chọn ± `DRES_SEGMENT_PAD_MS` (mặc định 500 ms; range dài 0 ms có thể bị loại bởi overlap check). Sửa được ngay trong guard, hoặc gửi thẳng `start_ms`/`end_ms`.
- **`taskName` lấy từ server** (`currentTask`) tại thời điểm nộp → luôn khớp task đang mở; operator vẫn có thể gõ đè.
- **Submit guard** trước mọi submit: thumbnail, video_id, frame_idx, nguồn raw paused frame hoặc btc id, thời gian, evidence OCR/ASR/audio, ô answer (QA), danh sách frame_idx có thứ tự (TRAKE), **+ evaluation/task đích, answer mode, ± ms, và JSON payload y hệt cái sẽ gửi** (`POST /api/submit/preview`, không gửi gì lên DRES).
- **Chặn sai định dạng tại chỗ**: QA thiếu text / frame không suy ra được mốc thời gian → nút submit bị khoá kèm lý do, không tốn 1 lần nộp sai.
- **Chống nộp trùng** (scope = `{evaluationId}/{taskName}`): dedup theo `video_id:frame_idx` (KIS) / `video_id|f1,f2,…` (TRAKE) / `video_id:frame_idx|text:<đáp án>` (QA — đổi đáp án hoặc đổi đoạn video đều là lượt mới; chỉ lặp y hệt cặp (đoạn, đáp án) mới bị chặn) → 409 + nút "Submit anyway".
- **Chặn lệch định dạng**: `answer as` reset về `auto` mỗi khi đổi loại truy vấn / đổi task (trước đây bị dính → QA im lặng gửi thiếu `text`), và backend so `taskType` DRES trả về với shape đang dựng → cảnh báo đỏ trong guard nếu lệch.
- **Xoá lịch sử**: mỗi mục có **checkbox** (chọn nhiều → "Xoá đã chọn") và nút **✕** xoá riêng 1 mục; nút `clear…` để xoá **cả task đang mở** hoặc **tất cả** (có xác nhận 2 bước); mọi mục bị xoá được ghi ra file backup `submit_history.<ts>.bak.json` cạnh file gốc. Lưu ý: xoá cũng xoá luôn trí nhớ dedup (bài đã nộp sẽ không còn bị cảnh báo trùng); bài đã lên DRES thì vẫn còn nguyên trên server.
- **Submit history sidebar**: loại task, frame, trạng thái (local / dres_ok / dres_error), **verdict DRES** (CORRECT/WRONG/…), cờ dup, answer; đồng hồ "Wrong" trên top bar đếm verdict WRONG.
- **Mock mode không bao giờ gọi DRES thật** (`has_dres` = false khi `AIC26_MOCK_MODE=true`).

## 10b. Bộ sinh 100 đáp án (tab Submission)
Vòng sơ tuyển chấm `Final = (R@1 + R@5 + R@20 + R@50 + R@100) / 5`, mỗi truy vấn nộp tối đa 100 dòng. Nên danh sách đáp án **không phải** "100 frame điểm cao nhất" mà là một bài phân bổ ngân sách xếp hạng giữa hai kiểu sai: **sai video** và **đúng video nhưng lệch thời điểm**.
- **Hàm lợi ích** (`backend/app/answer_gen.py`): tại vị trí `r`, chọn ứng viên `c` của video `V` có `U_r(V,c) = π_B(V) · E(c|V) · N(c|A_V)` lớn nhất.
  - `π_B(V)` — softmax của `video_score` ở nhiệt độ `T_B` của mốc hiện tại. `T` tăng dần theo mốc 1 → 5 → 20 → 50 → 100, nên danh sách tự chuyển từ khai thác video đầu sang phân tán rủi ro mà **không cần quota cứng** cho từng video.
  - `E(c|V)` — bằng chứng thời gian **trong nội bộ video**: mỏ neo giữ điểm truy xuất của nó, điểm dịch thừa hưởng rồi giảm theo bậc `|ε|`.
  - `N(c|A_V)` — độ mới: nhân kernel Gauss cho từng đáp án đã chọn của cùng video. Vùng nào đã phủ thì rơi điểm, nên video mạnh tự nhường chỗ khi hết thứ mới để nói.
- **Hai loại ứng viên cạnh tranh trong cùng một bảng xếp hạng**:
  - **Mỏ neo (anchor)** — các giả thuyết thời gian còn lại sau **temporal NMS** trên frame đã fuse ("sự kiện có thể nằm ở một vùng khác hẳn trong video này").
  - **Điểm dịch (offset)** — `a ± ε` ("đúng vùng rồi, nhưng keyframe được truy xuất nằm ngoài đoạn chấp nhận"). Thang `ε` **lấy từ phân vị sai số truy xuất–GT đo trên tập phát triển**, không đặt theo cảm tính (`--mode calib`).
  - **Snap về keyframe thật**: đo trên tập phát triển, trung vị `|frame truy xuất gần nhất − GT|` bằng **0** — phần lớn frame ground truth CHÍNH LÀ keyframe đã trích. Nên một điểm dịch rơi gần keyframe sẽ được kéo về đúng keyframe đó (`snap_offsets`, `snap_radius_frames`): vừa thăm dò đoạn chấp nhận, vừa lấy lại các keyframe mà NMS đã gộp — có xếp hạng chứ không liệt kê bừa.
- **Cờ `ambiguous` không hạ điểm video**: nó nói bất định nằm ở *thời điểm*, nên video đó được mở thêm mỏ neo và bớt điểm dịch ở mỗi mốc.
- **Xác định (deterministic)**: cùng đầu vào + cùng tham số ⇒ cùng danh sách, không có ngẫu nhiên. Mỗi đáp án lưu kèm lý do nội bộ (`π`, `E`, `N`, loại ứng viên) để gỡ lỗi và hiệu chỉnh — không nộp lên.
- **TRAKE** dùng cùng cơ chế nhưng ứng viên là **cả chuỗi**: `π_B` chia vị trí giữa các video, `ε` dịch cả chuỗi (lệch đồng hồ) hoặc từng event một (một mốc bị định vị sai), độ mới đo trên dịch chuyển trung bình mỗi event.
  - **Chỉ chuỗi đủ N sự kiện mới được thành đáp án.** Một dòng TRAKE là `<video>,<f1>,…,<fN>` với N do đề bài quy định, nên chuỗi tìm được 3/4 khoảnh khắc *không phải* đáp án ngắn hơn — nó không phải một dòng hợp lệ. Bộ ghép chuỗi cố tình trả cả chuỗi thiếu (để xem trong console) và **xếp hạng theo confident coverage**, nên chuỗi 3/4 thường đứng trên chuỗi đủ: đo trên tập phát triển, 45/50 chuỗi là chuỗi thiếu và có tới **55/100 dòng sai số frame**. N lấy từ đề bài (client gửi `event_count`), không suy ra từ chuỗi — nếu mọi chuỗi đều thiếu thì suy ra sẽ cho ra dòng 3 frame cho câu 4 sự kiện.
  - **Mọi dòng luôn tăng dần theo thời gian.** Dịch riêng một event có thể đẩy nó vượt event kế bên; dòng như vậy bị BTC loại nên không được coi là ứng viên (trước đây 13/100 dòng sai thứ tự). Chuỗi thiếu `frame_idx` cũng bị loại thay vì quy về frame 0.
  - Không có chuỗi nào đủ N thì **không sinh dòng nào** và ghi rõ lý do — nộp thiếu đáp án một câu còn cứu được, nộp một dòng sai định dạng thì hỏng cả file.
- **Q&A: bộ sinh chỉ xếp *chỗ để nhìn*, không nghĩ ra đáp án.** Dòng sinh ra có sẵn `video_id` + `frame_idx` (2/3 điều kiện tính điểm) nhưng cột answer để trống, vì đó là phán đoán của người. Nên tab Submission có **một ô answer ở cấp câu hỏi**: gõ một lần, bấm "Điền vào N dòng" là ghi vào tất cả các dòng bằng **một** lượt ghi (không phải N lượt ghi Supabase). Ô này tự điền sẵn nếu mọi dòng đang dùng chung một đáp án; nếu các dòng cố tình khác nhau (đoán nhiều phương án trên cùng frame là chiến thuật hợp lệ) thì để trống.
- **Thiếu answer báo một lần cho cả câu, không phải 100 lần.** Thiếu answer là tính chất của *câu hỏi* chứ không phải của từng dòng, và sửa bằng **một** thao tác. Khi mọi dòng đều trống thì gộp thành một lỗi `chưa có answer — cả N dòng sẽ bị chặn` (vẫn là lỗi chặn export, đúng luật BTC) kèm nhãn `⚠ chưa có answer` ở đầu câu; chỉ khi **một số** dòng đã có answer thì mới liệt kê từng dòng, vì lúc đó "dòng nào trống" mới là thông tin thật.
- **Nút "✨ Tự sinh đáp án"** trong tab Submission: chạy lần lượt từng câu trong gói, ghi thẳng vào bảng đáp án (một lần ghi Supabase cho cả 100 dòng, không phải 100 lần). Có nút "✨ sinh lại" cho từng câu.
- **Thứ tự dòng trong file nộp = thứ tự bộ sinh xếp ra.** `created_at` chính là khoá sắp xếp, nên nó phải **duy nhất theo từng dòng**: `nextCreatedAt()` cấp mốc tăng dần 1 ms/dòng, và `rowToRecord` **gửi kèm `created_at`** thay vì để cột `default now()` quyết định — `now()` là thời điểm *transaction* nên một lệnh INSERT 100 dòng đóng dấu y hệt nhau cho cả 100. Trước khi sửa, cả khối rơi xuống tie-break theo uuid: đo trên file nộp thật, video mạnh nhất nằm ở **hạng trung bình 50/100** — đúng bằng ngẫu nhiên, tức toàn bộ xếp hạng bị vứt đi trước khi export.
- **So sánh mốc thời gian theo *thời điểm*, không theo chuỗi.** Client ghi `…T19:04:00.123Z`, PostgREST trả `…T19:04:00.123456+00:00`, và **bỏ hẳn phần thập phân khi micro-giây bằng 0** → `…T19:04:00+00:00` so như chuỗi lại đứng SAU `…T19:04:00.500Z`, tức muộn hơn nửa giây so với sự thật. `recordToRow` chuẩn hoá về một dạng duy nhất, `sortRows` so bằng `Date.parse`.
- **Đáp án chấm tay luôn đứng trước và không bao giờ bị máy xoá.** Bộ sinh chỉ sở hữu các dòng `source: "generated"` — sinh lại thay đúng khối đó, mọi dòng do người chấm (`submit` từ console, `manual` gõ tay) được giữ nguyên. Lý do là điểm: `R@k` lấy max trong k đáp án đầu, nên một frame bạn đã nhìn tận mắt đặt ở **hạng 1 ăn trọn câu (Final 1.0)**, còn cũng frame đó ở **hạng 100 chỉ được 0.2**. Dòng tay có `created_at` cũ hơn nên tự đứng trước; k dòng tay ăn vào ngân sách nên câu vẫn dừng đúng ở 100 dòng, và ứng viên máy sinh trùng frame với dòng tay bị loại để không phí vị trí (máy rất hay xếp đúng frame bạn đã chọn lên đầu).
- **Import lại `submission.zip` đã export** (nút "⭳ Import đáp án"): đọc ngược đúng định dạng đã ghi ra — dùng để khôi phục sau khi mất localStorage, nạp file của đồng đội, hoặc mở lại bài nộp cũ để sửa.
  - Không thể parse một dòng nếu không biết loại câu: `L01_V028,1200,5` vừa là TRAKE 2 event vừa là QA trả lời "5". Loại câu lấy từ gói câu hỏi đang mở, thiếu thì lấy từ hậu tố `-kis|-qa|-trake` trong tên file.
  - Có xử lý CSV quoting đúng như lúc ghi (đáp án QA chứa dấu phẩy hoặc dấu nháy vẫn đọc lại nguyên vẹn).
  - **Không ghi gì cho tới khi bạn chọn**: xem trước số câu / số dòng, chọn `chỉ điền câu đang trống` (mặc định, không xoá gì) hay `thay thế`, và nếu thay thế thì nói rõ **sẽ xoá bao nhiêu dòng đang có**. Đáp án là thứ duy nhất trong app không tính lại được.
  - Dòng nhập vào mang `source: "manual"` nên bộ sinh coi như đáp án chấm tay: giữ nguyên, không bao giờ đè.
  - Dòng hỏng được **báo ra** chứ không im lặng bỏ; câu không có trong gói hiện tại vẫn nạp được nhưng cảnh báo rõ là sẽ không được export.
- **Đáp án đã có được truyền vào bộ sinh như độ phủ ban đầu (`taken`).** Danh sách được chấm như một khối, nên một vị trí đã tiêu cho một khoảnh khắc là *độ phủ*, không phải chỗ trống: máy không phát lại dòng đó, không tiêu thêm vị trí quanh đúng khoảnh khắc đó, và **băng tần chạy tiếp** thay vì khởi động lại từ hạng 1. Đây là chỗ sửa đáng kể — dòng đầu tiên sau 6 đáp án tay là hạng 7 (đã phải đa dạng hoá) chứ không phải hạng 1 (dồn hết vào một giả thuyết). Đo trên tập dev: sinh phần đuôi sau `taken` cho ra **đúng** danh sách mà một lần chạy đầy đủ sẽ cho ở các vị trí đó (0.5905 ở mọi mức head), trong khi bản chưa sửa tụt xuống 0.5848 khi head=6.
- **Phạt video đã chiếm chỗ (`taken_video_penalty`) mặc định TẮT — đây là kết quả đo âm.** Giả thuyết "video mà người ta đã đặt cược thì vị trí kế tiếp không nên đặt tiếp vào đó" nghe hợp lý nhưng không sống sót qua dữ liệu: mô phỏng đầu danh sách do người chấm, head=1 chỉ +0.002 (dưới ngưỡng nhiễu 0.0095 = một câu nhảy một mốc), còn head=3 và head=6 thì **giảm đơn điệu**. Cơ chế vẫn giữ và vẫn chỉnh được cho mỗi request, nhưng không bật sẵn.
- Ô "chỉ câu máy chưa sinh" nay tính theo **dòng do máy sinh**, không phải theo "câu đã có đáp án": một câu mới chỉ có dòng chấm tay vẫn đáng được sinh thêm phần đuôi bảo hiểm.
- **Hiệu chỉnh tham số**: `benchmarks/cache_pools.py` chụp kho ứng viên live một lần, `benchmarks/run_answer_gen.py` dò tham số offline (coordinate descent nhiều điểm khởi đầu), chấm bằng đúng công thức Final Score của BTC. Xem `benchmarks/README.md`.

## 11. Video & timeline
- **Video inline**: phím **`v`** trên keyframe đang chọn → video hiện **ngay dưới group đó**, **tua đúng keyframe**; chọn keyframe khác cùng video → chỉ seek (không reload, load 1 lần/video); chọn keyframe khác video → tự ẩn. Video **fill khung** (object-fit cover, bỏ viền đen). `Space` = play/pause gốc.
- **Timeline keyframe filmstrip** + playhead + marker event TRAKE (đã bỏ track OCR/speech/audio/heatmap để nhanh); click seek.
- Endpoint `GET /api/videos/{id}/timeline` trả keyframes (sorted, frame_idx/pts_time/url); `POST .../snap` snap raw→keyframe (cho frame-pick).

## 12. Voice input
- Nút **🎙 voice** / phím **Ctrl+M** (mọi nơi).
- **Web Speech API** (Chrome/Edge): interim realtime (`…đang nói`), nói liên tục, lang vi-VN.
- **Whisper backend** (`POST /api/transcribe`, faster-whisper, `WHISPER_MODEL`): chạy **mọi trình duyệt kể cả Brave** (Brave tắt Web Speech → tự fallback Whisper). Trả transcript VI + bản dịch EN.
- **Dịch sau khi nói theo đúng checkbox "Dịch VI→EN"**: bật → Whisper trả sẵn bản EN (Web Speech gọi `/api/translate`); tắt → giữ **nguyên lời nói** trong ô query (`/api/transcribe?translate=false` trả `text_en: null`). Trước đây voice input luôn dịch, không có cách nào giữ lại tiếng Việt.

## 13. Phím tắt
`/` focus query · `Enter` search / mở guard / xác nhận / gán paused frame cho TRAKE · `↑↓` chọn video · `←→` chọn frame · `v` hiện/ẩn video · `Space` play/pause và capture raw frame · `Tab` đổi vùng · `T` ẩn/hiện timeline · `Esc` đóng modal/xóa paused frame · `Ctrl+M` voice · `Ctrl+/` **bảng phím tắt**. Nút **?** ở top bar mở keymap.

## 14. Hiệu năng & độ bền
- **Latency badge**: parse/fusion/total ms.
- **Song song hoá**: kênh search, TRAKE per-event + pass-2, timeline (4 query Elastic), enrich 1 `_mget`.
- **Chịu lỗi từng kênh**: kênh chết (PE/GLAP down) → bỏ qua + cảnh báo, kênh khác vẫn chạy (không sập search).
- **Fallback**: dịch lỗi → giữ bản gốc; GLAP/Whisper/PE không có → fallback rõ ràng; cache parser + timeline (client).
- **Health** `GET /api/health`: trạng thái Elastic/Milvus/PE + `capabilities` (llm_query_parser, audio_vector_search, dres_submit, dres_connected, vlm_*) + warnings. UI hiện banner mock/lỗi.

## 15. API endpoints
| Method | Path | Mục đích |
|---|---|---|
| GET | `/api/health` | trạng thái service + capabilities |
| POST | `/api/query/parse` | routing JSON (heuristic/LLM) |
| GET | `/api/search/scope` | danh mục thư mục của profile + gợi ý heuristic cho `query` |
| POST | `/api/search` | search đầy đủ (group-by-video); nhận `scope`, trả lại `scope` đã áp dụng |
| POST | `/api/search/simple` | vector-only flat top-K |
| POST | `/api/search/trake` | chuỗi event (two-pass + DP) |
| POST | `/api/answers/generate` | danh sách tối đa 100 đáp án có thứ tự cho 1 truy vấn (§10b); truyền `groups`/`sequences` để xếp lại kết quả đang có trên màn hình thay vì search lại; `event_count` là số frame mỗi dòng TRAKE, lấy từ đề bài |
| GET | `/api/canvas/palette` | vocabulary + 16 màu cho canvas V-KIS |
| POST | `/api/search/canvas` | canvas JSON → OD spatial + PE text (RRF) |
| POST | `/api/qa/analyze` | NVILA visual QA + optional DeepSeek web-search grounding/citations |
| GET | `/api/keyframes/{submit_keyframe_id:path}` | chuẩn hoá id + URL + timing |
| GET | `/api/videos/{video_id}/timeline` | keyframes timeline |
| POST | `/api/videos/{video_id}/snap` | snap raw→keyframe |
| GET | `/api/dres/status` | kết nối DRES + user đang đăng nhập |
| POST | `/api/dres/login` | đăng nhập lại (phiên hết hạn) |
| GET | `/api/dres/evaluations` | các run + task đang mở + `taskStatus`/`timeLeft` |
| GET | `/api/dres/current-task` | task đang mở của 1 run |
| GET | `/api/dres/task-hint` | **đề bài** của task đang mở (text + media hint) |
| POST | `/api/submit/preview` | dựng đúng body DRES (không gửi) + cảnh báo trùng |
| POST | `/api/submit` | submit DRES v2 (answerSets) + dedup guard |
| GET | `/api/submit/history` | lịch sử submit + verdict |
| DELETE | `/api/submit/history` | xoá log local: `?ids=a,b` (từng mục), `?task_id=` (1 task), không tham số = tất cả; luôn backup |
| POST | `/api/translate` | dịch VI→EN |
| POST | `/api/transcribe` | Whisper STT (audio→text + EN); `?translate=false` → `text_en: null`, giữ nguyên lời nói |

## 16. Cấu hình (env, xem `backend/.env.example`)
`ELASTIC_ENDPOINT/API_KEY` · `MILVUS_ENDPOINT/TOKEN` · `PE_ENCODER_URL`(+`_TOKEN`) · `GLAP_ENCODER_URL` (mặc định = PE) · `NVILA_BASE_URL/TOKEN` · `NVILA_TIMEOUT_SECONDS/MAX_CANDIDATES` · `DEEPSEEK_API_KEY` · `DEEPSEEK_GROUNDING_*` · `MEDIA_BASE_URL` · `NVIDIA_API_KEY/BASE_URL` · `NVIDIA_MODEL` (parse) · `NVIDIA_FAST_MODEL` (expansion) · `SLIM_PARSE` · `TRANSLATE_TO_EN` · `WHISPER_MODEL` · `DRES_BASE_URL` · `DRES_USERNAME/PASSWORD` · `DRES_SESSION` · `DRES_EVALUATION_ID` · `DRES_SEGMENT_PAD_MS` · `IDX_*` · `MILVUS_IMAGE_COLLECTION` · `MILVUS_AUDIO_COLLECTION` · `AIC26_MOCK_MODE` · `CORS_ORIGINS`.

## 17. Models & dữ liệu
- **PE-Core-G14-448** (1280-d) — ảnh + text, Kaggle FastAPI + cloudflared (`model-setup-backend.ipynb`).
- **GLAP `mispeech/GLAP`** (1024-d) — audio↔text, text encoder chạy **CPU** (`/encode-audio-text`); cùng notebook PE hoặc `glap-encoder-kaggle.ipynb` riêng.
- **Qwen3-VL-Embedding-8B** (4096-d native) — text query encoder, Colab A100 40/80 GB BF16 + FA2 (`Qwen3VL-Embedding-8B/Qwen3_VL_Embedding_8B_Text_Encoder_Server_Colab_A100.ipynb`). Ảnh đã encode sẵn offline; instruction query khóa ở `Retrieve images or text relevant to the user's query.`
- **Milvus**: `aic26_image_peg14_v1` (382,299), `aic26_audio_glap_v1` (466,996), `aic26_image_peg14_infoshotpp_v1` và `aic26_image_qwen3vl8b_infoshotpp_v3` (1,339,055 keyframe InfoShot++).
- **Elastic**: `aic26_keyframe_map_v1`, `aic26_ocr_keyframes_v1`, `aic26_speech_segments_v1`, `aic26_audio_windows_v1`.
- **NVIDIA NIM**: qwen3-next (parse), llama-3.1-8b (expansion). **faster-whisper** (voice).
- **NVILA-8B**: Colab A100 BF16 worker (`aic26_nvila8b_qa_colab_server.ipynb`), multi-image QA hai pass cộng post-search visual verification, không cần caption keyframe tạo sẵn.
- **DeepSeek `deepseek-v4-flash` + built-in `web_search`**: backend-only knowledge grounding sau NVILA, gọi qua **Responses API** (`POST /responses`) — `/chat/completions` từ chối mọi tool không phải `function`, và chỉ `deepseek-v4-flash` hỗ trợ tool này. Query và các trang model thực sự mở được trả về UI để operator kiểm chứng.

## 18. Chưa làm (cố ý) / điểm mở rộng
- **Chưa có**: general Qwen3-VL reranker cho mọi task, structured VLM caption offline, object/scene/action tag filter, SigLIP/EVA ensemble, true clip-query V-KIS. QA đã có NVILA trực tiếp nhìn candidate, nhưng chưa có dense-caption Answer Span Prediction trên toàn corpus.
- **Điểm mở rộng**: thêm `RerankerClient` + bước rerank sau fusion (flip `capabilities.vlm_rerank`); object filter qua parser `filters` + Elastic; query-vector morphing cho feedback.

## 19. Chạy & test
```bash
# Backend (mock, không cần service)
cd backend && AIC26_MOCK_MODE=true PYTHONPATH=. ../.venv/bin/python -m uvicorn app.main:app --port 8000
# Backend (live: đọc .env)
cd backend && PYTHONPATH=. ../.venv/bin/python -m uvicorn app.main:app --port 8000
# Frontend
cd frontend && npm install && npm run dev      # proxy /api -> :8000

# Tests
cd backend && PYTHONPATH=. ../.venv/bin/python -m pytest -q     # 88
cd frontend && npm run test                                    # 18
```
