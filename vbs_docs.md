Đã đọc xong hệ thống và khảo sát trực tiếp server V3C. Dưới đây là kết quả.

## ⚠️ Deadline trước



| Mốc | Ngày |

|---|---|

| **Nộp demo paper VBS 2027** | **22/09/2026** — còn **3 ngày** |

| Notification | 09/10/2026 |

| Camera-ready | 01/11/2026 |

| Thi đấu | 05/01/2027 @ MMM 2027, Siem Reap |



Format: **6+2 trang Springer LNCS** (2 trang chỉ cho references), nộp qua hệ thống MMM 2027.



**Điểm chiến lược quan trọng:** paper VBS là *demo paper mô tả công cụ*, không phải paper kết quả. Bạn **không cần** index xong V3C trước 22/9 — chỉ cần mô tả hệ thống và cách nó sẽ chạy trên V3C. Deadline thực để có data chạy được là **05/01/2027**. Đây là tin tốt, vì khối lượng tính toán bên dưới không thể xong trong 3 ngày.



---



## 1. Data V3C thực sự có trên server



Tôi đã kết nối SFTP và đo trực tiếp (không lấy từ trí nhớ):



| Shard | Videos | Shots | Thời lượng |

|---|---:|---:|---:|

| V3C1 | 7,475 | 1,082,657 | 1,000 h |

| V3C2 | 9,760 | 1,425,452 | ~1,300 h |

| V3C3 | 11,215 | 1,635,573 | ~1,000 h |

| **Tổng** | **28,450** | **4,143,682** | **~3,300 h** |



**Cấu trúc thư mục:**



```

/V3C/V3C1/  videos/<id>/<id>.mp4 + <id>.info.json + <id>.description

            keyframes/<id>.tgz          ← 1,073 GB tổng (7,475 archive)

            keyframes-jpg.tgz           ← 440 GB (1 archive khổng lồ)

            thumbnails.tar              ← 49.4 GB (1 archive)

            msb.tar.gz (12 MB) · info.tar.gz (3 MB)



/V3C/V3C2/  videos/<id>/ · keyframes/<id>/ · thumbnails/<id>/ · msb/<id>.tsv · info/<id>.json

/V3C/V3C3/  (giống V3C2)



/V3C/small/      28,450 mp4 — 976 GB   (bản downscale)

/V3C/verysmall/  28,305 mp4 — 102 GB   (bản downscale mạnh)  ← khả thi để host

```



**Định dạng đã verify:**



- **MSB** (master shot boundary — chính là đơn vị chấm điểm của VBS): TSV `startframe / starttime / endframe / endtime`, thời gian tính bằng **giây**.

- **Shot id**: `shot<videoid>_<n>`, keyframe là `shot<videoid>_<n>_RKF.png` (~600 KB), thumbnail `shot<videoid>_<n>.png` (~38 KB). V3C1 cũng dùng đúng naming này bên trong `.tgz`.

- **info.json**: metadata Vimeo — `title`, `description` (HTML), `tags[]`, `categories[]`, `duration`, `width/height`, `channel`, `uploadDate`, `license`.

- Video id là **5 chữ số phẳng** (`00047`, `13596`) — **không có category/folder** như `L26`/`K01`.



**Thống kê V3C1 từ info.json:** trung bình 8.0 phút/video; phần lớn 1280×720 (3,976) và 1920×1080 (1,752); license toàn bộ là Creative Commons; 908/7,475 video **không có tag** nào; category phổ biến: narrative, travel, sports, documentary, music, journalism.



---



## 2. Data CÒN THIẾU — phải tự sinh toàn bộ



V3C chỉ cho **video + shot boundary + keyframe + metadata Vimeo**. Mọi thứ hệ thống bạn đang đọc từ Elastic/Milvus đều **không tồn tại** và phải tự build:



| Kênh trong hệ thống | V3C có? | Phải làm gì |

|---|---|---|

| `image_pe` (PE-Core-G14 1280-d) | ❌ | Encode **4.14 M keyframe** → Milvus |

| `image_qwen` (Qwen3-VL-Emb-8B 4096-d) | ❌ | Encode 4.14 M keyframe (rất nặng) |

| `tara` (clip 3584-d) | ❌ | Encode theo clip |

| `speech` | ❌ | **V3C không có subtitle/ASR gì cả.** Whisper trên 3,300 h, tiếng **Anh** đa ngôn ngữ |

| `ocr` | ❌ | Chạy OCR trên 4.14 M keyframe |

| `audio` (Elastic tags) + GLAP vector | ❌ | Audio tagging trên 3,300 h |

| `object_layout` / `canvas_image` (V-KIS) | ❌ | OD trên 4.14 M keyframe |

| `keyframe_map` (Elastic) | ⚠️ **Miễn phí** | Dựng thẳng từ MSB TSV — đã có sẵn start/end frame + time chính xác |

| Metadata text channel | ⚠️ **Miễn phí, kênh MỚI** | `title`/`description`/`tags` từ info.json — V3C có mà AIC không có |



**Hai điểm đáng chú ý:**



- `keyframe_map` **cho không**: MSB đã cho `(video, shot_n, start_ms, end_ms)` chính xác hơn hẳn keyframe map bạn phải tự suy ở AIC. Và vì VBS chấm theo **khoảng ms trong video**, nộp thẳng cửa sổ shot từ MSB sẽ chính xác hơn cách `DRES_SEGMENT_PAD_MS ±500ms` quanh một instant.

- **Metadata Vimeo là kênh retrieval mới hoàn toàn** mà hệ thống hiện chưa có. Với ~87% video có tags và title/description tiếng Anh do người upload viết, đây là một kênh Elastic full-text rẻ, mạnh, và là thứ đáng viết vào paper.



---



## 3. Chỗ code phải sửa



Hệ thống có sẵn một **seam rất sạch** để cắm profile thứ ba: `Settings.for_retrieval_database()` tại [config.py:316](backend/app/config.py#L316) đã tách `btc` / `infoshotpp` theo Milvus endpoint, index Elastic, và media origin. Thêm `"v3c"` vào đây là đúng chỗ.



Nhưng có **4 chỗ giả định cứng theo dataset AIC** sẽ vỡ:



1. **Identity** — [identity.py:41](backend/app/adapters/../identity.py#L41): `submit_keyframe_id = "<category>/<video_id>/<frame>"`, và `group_from_video_id()` cắt theo `_`. Với V3C id `00047` thì `split("_")[0]` trả về chính `00047` → group = video id. Cần category tổng hợp (đề xuất: shard `V3C1`/`V3C2`/`V3C3`) → `V3C1/00047/001`.

   *Lưu ý nhỏ:* 30 video V3C1 có **>999 shot** (max 5,011). `f"{n:03d}"` không cắt chuỗi nên vẫn chạy đúng, miễn là tên file keyframe trên storage khớp — nhưng phải thống nhất, không được pad 3 chữ số ở một chỗ và không pad ở chỗ khác.



2. **Media URL** — [media.py:24](backend/app/media.py#L24) hardcode layout `Keyframes/Keyframes_{group}/{video_id}/{n:03d}.jpg`. V3C là `shot<vid>_<n>_RKF.png`. Cần builder theo profile.



3. **Scope** — [scope.py](backend/app/scope.py) hardcode L21–L30 + K01–K20 kèm nhãn chương trình tiếng Việt và heuristic topic. V3C không có khái niệm này. Thay thế tự nhiên: facet theo **Vimeo categories** (travel/sports/documentary/music/…) đọc từ info.json — giữ nguyên cơ chế, đổi nguồn facet.



4. **Ngôn ngữ** — [query_parser.py](backend/app/query_parser.py) dùng lexicon tiếng Việt và [translate.py](backend/app/translate.py) dịch VI→EN. VBS query là **tiếng Anh**. Đường dịch phải thành no-op (đã có `_looks_english()` sẵn), và lexicon heuristic cần bản tiếng Anh — nếu không thì khi `NVIDIA_API_KEY` chết, fallback sẽ route sai hoàn toàn.



**Tin tốt về DRES:** [dres_client.py](backend/app/adapters/dres_client.py) đã đúng chuẩn **Client API v2** — chính là API VBS dùng. `answerSets` / `mediaItemName` / `start`–`end` ms trong [submit_service.py](backend/app/services/submit_service.py) dùng được ngay, chỉ cần `mediaItemName = "00047"`.



**Task mismatch:**

- **TRAKE** (`trake.py`, 718 dòng + DP chain) — VBS **không có** task này. Công sức lớn nhất của hệ thống không ghi điểm ở VBS.

- **AVS** (Ad-hoc Video Search) — VBS **có**, hệ thống bạn **không có**. AVS cần nộp hàng trăm–hàng nghìn shot, chấm live, thưởng recall + diversity giữa các video. Đây là mô hình tương tác khác hẳn KIS. Nhưng `answer_gen.py` (bộ sinh 100 đáp án theo rank-budget với novelty + temporal NMS) **gần như đúng là thuật toán AVS cần** — đó là seam tái dùng tốt nhất bạn đang có.

- CFP nhấn mạnh **KIS textual và chat/panel là chính, KISV là thiểu số**, cộng thêm **VQA**. Không có novice session, chỉ developer thi.



---



## 4. Khối lượng — chốt bằng con số



- **Embedding**: 4.14 M keyframe. PE-Core-G14-448 trên T4 khoảng 30–60 ảnh/s → **19–38 giờ GPU** cho một model. Qwen3-VL-Emb-8B nặng hơn nhiều bậc. Quota Kaggle free (12h/session, 30h/tuần) **không đủ** — cần GPU thuê.

- **ASR**: 3,300 h audio. Whisper large-v3 với batching tốt vẫn là hàng trăm giờ GPU.

- **Storage keyframe**: thumbnails V3C2+V3C3 ≈ 116 GB, cộng V3C1 thumbnails.tar 49 GB → **~165 GB** phải host trên R2/HF. Bản `keyframes/` full-res PNG (~600 KB/shot) là **~2.4 TB** — không khả thi, phải dùng thumbnails hoặc tự re-encode JPEG.

- **Video để phát trong console**: `verysmall/` **102 GB** là lựa chọn duy nhất hợp lý (bản `small/` 976 GB quá lớn).



---



## 5. Đề xuất lộ trình



**Từ giờ đến 22/9 (3 ngày) — chỉ viết paper:**

Paper mô tả hệ thống đã có (multi-channel RRF, Qwen reranker, submit guard, console keyboard-first) + kế hoạch thích ứng V3C. Bạn đã có [Paper/technical-report/main.tex](Paper/technical-report/main.tex) 6 trang tiếng Anh — cần chuyển sang **LNCS** và viết lại phần task cho đúng VBS (bỏ TRAKE, thêm AVS + VQA). Đây là việc khả thi trong 3 ngày.



**Từ 22/9 đến 05/01 — build data (ưu tiên theo tỉ lệ giá trị/chi phí):**



1. **Miễn phí, làm ngay**: MSB → Elastic keyframe_map; info.json → Elastic metadata channel (title/description/tags). Hai thứ này cho bạn một hệ thống **chạy được ngay** chỉ với text search, không cần một giờ GPU nào.

2. **Bắt buộc**: tải `verysmall/` (102 GB) + thumbnails (~165 GB), host lên R2.

3. **Blocker chính**: PE-Core-G14 trên 4.14 M keyframe. Đây là đường tới hạn — bắt đầu sớm nhất có thể.

4. **Sau đó**: ASR (Whisper) → OCR → audio tags. Bỏ OD/canvas trừ khi dư thời gian, vì KISV là thiểu số ở VBS.

5. **Code**: thêm profile `v3c` + identity/media builder theo profile; chuyển `answer_gen` thành AVS submitter.



---



Hai điểm tôi muốn bạn quyết trước khi tôi đi tiếp: **(a)** có dùng cả 3 shard V3C1+2+3 không, hay chỉ V3C1 (1,08 M shot — bằng ¼ chi phí GPU); **(b)** ưu tiên 3 ngày tới là viết paper LNCS, hay bắt đầu download/ingest data. Tôi có thể publish bản phân tích này thành một trang chia sẻ cho team nếu bạn cần.
