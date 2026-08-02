# Đánh giá ngắn output Claude về Speech Extraction Pipeline

## 1. Kết luận nhanh

Claude đánh giá pipeline speech hiện tại **đủ tốt để giữ làm nhánh ASR chính**. Không có lý do mạnh để chạy lại toàn bộ corpus hoặc đổi model ở tầng extraction.

Điểm quan trọng nhất: các vấn đề lớn **không nằm ở ASR/PhoWhisper/WhisperX**, mà nằm ở tầng **indexing và temporal fusion**. Vì vậy hướng sửa đúng là: giữ output JSON hiện tại, rồi bổ sung xử lý hậu kỳ trước khi đưa vào Elasticsearch và fusion.

## 2. Nhận định đáng giữ từ Claude

- **Không cần chạy lại ASR toàn corpus.**
- **PhoWhisper-large + WhisperX** là lựa chọn hợp lý cho video chủ yếu tiếng Việt.
- `avg_word_score` nên được dùng ở tầng index để lọc hoặc hạ trọng số segment kém, không nên lọc cứng ngay khi extract.
- Output có `text`, `start`, `end`, `words[]` đã đủ làm nền cho BM25 và temporal fusion.
- ASR không có nhiệm vụ bắt tiếng nhạc, piano, còi xe; phần đó phải thuộc nhánh audio riêng như CLAP/audio tagging.
- Lỗi transcript chủ yếu là lỗi đồng âm, sai dấu, sai tên nước ngoài, nhưng phần lớn không phá hỏng retrieval nếu có index tốt.

## 3. Các lỗi / thiếu sót cần sửa ngay

### M1. Elasticsearch mapping thiếu `avg_word_score`

**Vấn đề:**  
Claude chỉ ra rằng nếu ES mapping không lưu `avg_word_score`, thì chính sách “lọc ở tầng index” không thể thực thi.

**Tác động:**  
Segment rác hoặc alignment kém vẫn có thể lọt vào kết quả retrieval.

**Cách sửa:**

- Thêm field `avg_word_score: float`.
- Thêm `duration: float`.
- Có thể thêm `confidence_bucket: high | mid | low`.
- Khi query:
  - drop segment có `avg_word_score < 0.40`;
  - hạ trọng số segment `0.40 <= avg_word_score < 0.50`;
  - giữ nguyên segment `>= 0.50`.

**Có cần chạy lại ASR không?**  
Không. Chỉ cần đọc lại JSON cũ và build lại index.

---

### M2. Analyzer tiếng Việt trong Elasticsearch chưa chắc chạy được

**Vấn đề:**  
Nếu mapping dùng `vi_tokenizer` nhưng Elasticsearch chưa cài plugin tương ứng, index creation sẽ lỗi.

**Tác động:**  
Không build được index hoặc tokenization tiếng Việt kém.

**Cách sửa:**

Chọn một trong hai hướng:

1. Cài plugin tokenizer tiếng Việt cho Elasticsearch.
2. Hoặc tokenize tiếng Việt bên ngoài bằng underthesea/VnCoreNLP, sau đó đưa token đã tách vào field riêng dùng `whitespace analyzer`.

**Có cần chạy lại ASR không?**  
Không. Đây là lỗi tầng index.

---

### M3. Segment teaser/preview tạo “neo thời gian giả”

**Vấn đề:**  
Các segment đầu bản tin thường tóm tắt nhiều chủ đề. Chúng chứa keyword đúng nhưng timestamp không trỏ tới phần nội dung chính.

Ví dụ: teaser đầu video có thể nhắc “ghép tim”, “cháy rừng”, “sụt lún”, nhưng phần thân bài thực sự xuất hiện sau đó vài phút.

**Tác động:**  
Fusion với keyframe/audio có thể sai thời điểm, vì query match ở teaser thay vì đoạn nội dung thật.

**Cách sửa:**

- Gắn nhãn `segment_role`:
  - `intro`
  - `preview`
  - `body`
- Hạ trọng số các segment `intro/preview` khi dùng làm temporal anchor.
- Ưu tiên segment `body` khi join với keyframe/audio.
- Có thể tách teaser thành nhiều câu nhỏ nếu cần.

**Có cần chạy lại ASR không?**  
Không.

---

## 4. Các cải tiến nên làm tiếp theo

### S1. Chuẩn hóa số

**Vấn đề:**  
ASR thường xuất số dạng chữ, ví dụ:

- “hai ngàn không trăm hai mươi bốn”
- “năm phẩy hai mươi lăm phần trăm”

Query của người dùng có thể nhập:

- `2024`
- `5.25%`

**Cách sửa:**  
Tạo thêm field `text_normalized` để chuyển số đọc bằng chữ sang dạng số.

---

### S2. Re-segment ở tầng index

**Vấn đề:**  
Một số segment dài 20–30 giây, chứa quá nhiều từ. Điều này làm retrieval và fusion kém chính xác.

**Cách sửa:**  
Dùng `words[]` để cắt lại thành cửa sổ nhỏ hơn, khoảng 8–12 giây.

Output mới nên có dạng:

```json
{
  "segment_id": "L21_V001_00031739",
  "video_id": "L21_V001",
  "start": 31.739,
  "end": 43.739,
  "text": "...",
  "avg_word_score": 0.58,
  "source_segment_start": 31.739,
  "source_segment_end": 45.370
}
```

---

### S3. Thêm synonym / alias cho các lỗi ASR phổ biến

Nên thêm alias cho các nhóm dễ sai:

- `cá tra` ↔ `cá trà`
- `thiên tai` ↔ `thiên tài`
- `Fed` ↔ `ngân hàng trung ương mỹ` ↔ `cục dự trữ liên bang`
- `ECMO` ↔ các dạng phiên âm sai của ASR
- tên địa danh nước ngoài dễ bị phiên âm sai

Việc này giúp tăng recall mà không cần đổi model.

---

### S4. Bổ sung field phục vụ fusion

Nên thêm các field:

```json
{
  "segment_id": "...",
  "duration": 12.0,
  "source_video_path": "...",
  "confidence_bucket": "high",
  "segment_role": "body"
}
```

Các field này giúp join speech ↔ audio ↔ keyframe ổn định hơn.

---

### S5. Thêm text embedding tiếng Việt

BM25 đủ tốt cho query keyword, nhưng yếu với query diễn đạt lại. Nên thêm một retriever semantic tiếng Việt rồi fuse bằng RRF với BM25.

BM25 dùng cho keyword chính xác.  
Embedding dùng cho paraphrase.  
RRF dùng để gộp rank.

## 5. Những điểm trong output Claude cần kiểm chứng lại

Claude đưa ra nhiều con số khá cụ thể như “55 segment”, “20.9 phút”, “median 22.25s”, “0 overlap”, “0 từ sai thứ tự”. Các nhận định này có vẻ hợp lý, nhưng nếu dùng trong báo cáo chính thức thì nên tự chạy script kiểm chứng lại từ file JSON.

Không nên bê nguyên các con số định lượng vào báo cáo nếu chưa tự reproduce.

## 6. Thứ tự sửa đề xuất

Ưu tiên thực hiện theo thứ tự:

1. Thêm `avg_word_score`, `duration`, `segment_id` vào index.
2. Chốt tokenizer tiếng Việt cho Elasticsearch.
3. Gắn nhãn hoặc hạ trọng số teaser/preview.
4. Re-segment thành window 8–12 giây từ `words[]`.
5. Thêm chuẩn hóa số.
6. Thêm synonym/alias cho lỗi ASR phổ biến.
7. Thêm text embedding tiếng Việt và fuse với BM25 bằng RRF.
8. Sau đó mới ráp với nhánh audio CLAP/audio tagging.

## 7. Kết luận

Output Claude có thể dùng làm review kỹ thuật nền. Tuy nhiên, phần cần biến thành hành động là:

- **không sửa extraction;**
- **không chạy lại ASR;**
- **sửa tầng index trước;**
- **sửa temporal fusion sau;**
- **chuẩn bị schema đủ tốt để speech branch join được với audio branch.**

Trọng tâm hiện tại không phải “ASR có đủ tốt chưa”, mà là “làm sao biến JSON ASR hiện tại thành index retrieval và temporal anchor đáng tin”.
