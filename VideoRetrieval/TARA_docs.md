Tui đã đọc toàn bộ **36 query của đợt 3** trong file zip và đối chiếu chúng với failure mode của hệ thống hiện tại, rồi so với các model video-retrieval mới nhất mà mình vừa tìm. Kết luận khá rõ:

> **Nếu chỉ được thêm đúng 1 model để sửa điểm yếu của đợt 3, tui chọn TARA trước WeMM.**
>
> **Nếu được thêm 2 nhánh video retrieval, tui chọn TARA + WeMM-2B.**

Lý do không phải TARA có benchmark tổng quát cao nhất, mà vì **dataset đợt 3 của bạn có rất nhiều query cần hiểu diễn tiến và thứ tự hành động**, đặc biệt là cooking và TRAKE. Đó chính xác là loại lỗi mà TARA được train để giải quyết.

## 1. Bộ đề đợt 3 thực sự rất “temporal-heavy”

Trong 36 query có:

* **26 KIS**
* **8 QA**
* **2 TRAKE**
* khoảng **13/36 query liên quan trực tiếp đến nấu ăn/thực phẩm**
* và tui đếm được khoảng **15 query mà thứ tự hành động là thông tin quan trọng**, chưa tính những query chỉ có nhiều cảnh nối tiếp.

Riêng cooking đã có một pattern rất rõ:

| Query              | Nội dung quyết định                                                 | Frame embedding có đủ không? |
| ------------------ | ----------------------------------------------------------------------- | -------------------------------- |
| **1**        | đổ trứng → cắt đậu hũ → cho đậu hũ → khuấy                | ❌                               |
| **3**        | nguyên liệu cam → dạng ống → rau/nấm → thịt                    | ❌                               |
| **11**       | lấy thành phẩm → cho nguyên liệu mới → thành phẩm nở         | ❌                               |
| **12**       | đổ chất lỏng → xóc → lửa hồng → chuyển nồi → lặp lại     | ❌                               |
| **15**       | rửa/để ráo → bột → gia vị → nguyên liệu xanh/vàng → chiên | ❌                               |
| **16**       | cho nguyên liệu → đảo → đổi màu → thêm nước/gia vị        | ❌                               |
| **17**       | chiên sơ → bổ đôi → lấy lõi → đặt lên dụng cụ            | ❌                               |
| **21 TRAKE** | pha sốt → cắt đầu tôm → tôm chạm đĩa → tép cam thứ 4      | **Rất không đủ**       |
| **25**       | nhúng thịt → cho vào tô → chan nước dùng                       | ❌                               |
| **29**       | múc canh đúng 2 lần                                                 | ❌, cần temporal/counting       |

Trong khi Q24, Q27, Q31 lại khác: Q24 cần biết **lượng nước tương** nên ASR/OCR/VQA quan trọng hơn; Q27 cần đếm màu bột; Q31 chủ yếu là visual semantics.

Nghĩa là không nên kết luận:

> “Cooking yếu → cần model hiểu đồ ăn tốt hơn.”

Thực ra vấn đề lớn hơn là:

> **Cooking trong đề này được dùng để tạo những chuỗi hành động rất giống nhau về object/scene nhưng khác thứ tự và trạng thái.**

Một frame có thể thấy:

```text
pot
shrimp
vegetables
chef
bowl
```

nhưng không biết:

```text
A → B → C → D
```

hay:

```text
C → A → D → B
```

---

# 2. Chính benchmark của repo bạn cũng đã báo trước vấn đề này

Trong benchmark documentation hiện tại, team đã tách T-KIS thành:

* 45 query mô tả **một moment**
* 22 query mô tả **một sequence**

và ghi nhận:

**R@1: 60.00% → 18.18%**

trong khi R@100 của hai nhóm lại gần nhau.

Đây là tín hiệu rất mạnh.

Nó nói rằng hệ thống thường:

> **tìm được đúng video**, nhưng **không biết xếp đúng video lên đầu khi query chứa chuỗi hành động**.

Đó chính xác là failure mode của frame-level retrieval.

Current retrieval channels của repo cũng không có một `video` embedding thật sự; hiện là image PE/Qwen, OCR, speech, audio, similarity và object/canvas.

Do đó tui nghĩ việc thêm video-text embedding **không phải optimization nhỏ nữa**, mà là một modality đang thiếu.

---

# 3. TARA phù hợp nhất với chính bộ query này

**TARA – Text Adapted Retrieval Alignment**, ECCV 2026, được xây trên Tarsier2 và tạo **video-text embedding 3584-d**. Checkpoint và code hiện đã public. ([GitHub][1])

Điểm đặc biệt là tác giả không chỉ benchmark “man cooking food” kiểu MSR-VTT.

Họ chủ động nghiên cứu **temporal nuance**, tức các video có:

* gần như cùng visual content;
* cùng objects;
* cùng actions;
* nhưng **thứ tự/thời gian ngược nhau**.

Ví dụ canonical của họ là:

```text
opening a door
vs.
closing a door
```

frame content gần giống nhau, nhưng temporal direction khác. ([Oxford Robotics][2])

Đây gần như cùng bản chất với:

```text
Q25:
nhúng thịt → cho vào tô → chan nước

vs.

chan nước → cho thịt → ...
```

hay Q21:

```text
mix sauce
→ cut shrimp head
→ plate shrimp
→ garnish orange
```

---

# 4. Evidence temporal của TARA rất mạnh

Trên **RTime**, nơi model phải phân biệt video/caption đúng với phiên bản đảo thời gian:

| Model           |        T2V R@1 |
| --------------- | -------------: |
| InternVideo2-1B |           50.0 |
| Qwen2.5-VL      |           53.4 |
| Tarsier2 base   |           58.8 |
| ArrowRL         |           55.6 |
| **TARA**  | **67.2** |

([Oxford Robotics][2])

Quan trọng hơn với cooking: benchmark CiA có cả **EPIC-KITCHENS**, tức dataset tập trung vào các thao tác đời thường/nhà bếp.

Trên EPIC temporal-chiral split:

| Model             |         Chiral |         Static |            All |
| ----------------- | -------------: | -------------: | -------------: |
| InternVideo2      |           48.3 |           22.1 |            8.8 |
| Qwen3VL-Embedding |           62.1 |           28.6 |           20.6 |
| Tarsier2          |           67.4 |           22.0 |           15.3 |
| **TARA**    | **81.1** | **45.6** | **38.9** |

([Oxford Robotics][2])

**Đây là lý do tui đổi priority so với câu trả lời trước.**

Nếu chỉ nhìn general text→video benchmark, WeMM cực hấp dẫn.

Nhưng sau khi đọc **chính bộ đề đợt 3 của bạn**, TARA match failure mode tốt hơn.

---

# 5. Còn WeMM-Embedding thì sao?

**Vẫn rất đáng thêm. Nhưng vai trò khác TARA.**

WeMM-2B là candidate rất mạnh cho:

> **general-purpose text → video retrieval**

Nó nhận trực tiếp video, text, image và interleaved input.

WeMM-2B còn vượt Qwen3-VL-Embedding-8B trên nhiều video benchmark dù nhỏ hơn nhiều. Ví dụ MMEB-v2 Video:

**Qwen3VL-8B: 67.1
WeMM-2B: 70.8**

và Video Retrieval:

**58.7 → 63.4**.

Trên VATEX/MSR-VTT/YouCook2 text→video:

**Qwen3VL-8B mean: 64.6
WeMM-2B: 70.4
WeMM-9B: 72.9**.

YouCook2 đặc biệt cũng liên quan cooking.

Nhưng WeMM paper **không benchmark arrow-of-time/chiral actions trực tiếp như TARA**.

Vì vậy tui xếp:

```text
General video semantics:
WeMM > TARA

Fine-grained temporal order:
TARA > WeMM   [evidence hiện tại mạnh hơn]

Compute efficiency:
WeMM-2B >> TARA-8B
```

---

# 6. MARS thì rất mạnh, nhưng tui chưa chọn cho production lúc này

MARS mới xuất hiện ngày **2/9/2026** và report SOTA trên 4 standard text-video retrieval benchmarks.

Ý tưởng rất hay: thay vì ép cả video vào một vector, nó tạo **nhiều representation slots từ nhiều decoder layers**, để các slot giữ những cue khác nhau như object/action/temporal structure. ([arXiv][3])

Về research:

**MARS rất đáng benchmark.**

Nhưng đối với hệ thống AIC/VBS cần triển khai ngay:

**TARA/WeMM thuận tiện hơn**, vì:

* TARA có checkpoint public;
* WeMM có 2B/4B/9B public;
* cả hai có inference workflow tương đối rõ;
* MARS multi-slot cần thay đổi ANN/index/fusion nhiều hơn.

Nó không còn đơn giản:

```text
query vector ↔ video vector
```

mà là:

```text
[q1 q2 q3 q4]
      ↕
[v1 v2 v3 v4]
      ↓
slot aggregation
```

([arXiv][3])

Với deadline hiện tại, tui không chọn nó đầu tiên.

---

# 7. ViLL-E rất hợp về mặt research nhưng deployment chưa đẹp bằng TARA/WeMM

ViLL-E là **ACL 2026 Outstanding Paper**, train chung:

* video retrieval;
* temporal localization;
* video-text matching;
* QA.

Nó report khoảng **+7% temporal localization** so với các VideoLLM khác và cải thiện video retrieval tới ~4% so với dual encoders. ([ACL Anthology][4])

Về mặt capability thì rất hợp.

Nhưng nếu mục tiêu là:

> “Tuần này tui muốn thêm model vào pipeline”

thì TARA và WeMM hiện có đường triển khai rõ hơn.

---

# 8. Ranking của tui sau khi xem bộ đề thật

Nếu tiêu chí là **khắc phục đợt 3 AIC**, không phải leaderboard abstract:

| Candidate         | Cooking sequence |  Before/after |   General KIS | TRAKE support | Deployability |     Priority |
| ----------------- | ---------------: | ------------: | ------------: | ------------: | ------------: | -----------: |
| **TARA**    |    **5/5** | **5/5** |           4/5 | **5/5** |           4/5 | **#1** |
| **WeMM-2B** |            4.5/5 |      3.5–4/5 | **5/5** |           4/5 | **5/5** | **#2** |
| ViLL-E            |            4.5/5 |         4.5/5 |         4.5/5 |         4.5/5 |         2.5/5 |           #3 |
| MARS              |              4/5 |           4/5 | **5/5** |         3.5/5 |         2.5/5 |           #4 |
| InternVideo2-S2   |              4/5 |           3/5 |         4.5/5 |           3/5 |           4/5 |     baseline |

Và **InternVideo3 tui vẫn không chọn làm dense retriever chính**. Nó phù hợp hơn với downstream video reasoning/verification.

---

# 9. Nhưng thêm TARA không đồng nghĩa TRAKE sẽ tự nhiên được giải quyết

Đây là điểm rất quan trọng.

Q21 yêu cầu:

```text
E1 bắt đầu trộn sốt
E2 bắt đầu cắt đầu tôm
E3 con tôm đầu tiên chạm đĩa
E4 tép cam thứ tư được đặt lên đĩa
```

Một embedding cho **toàn bộ video** chỉ nói:

> “video này có vẻ đúng.”

Nó **không trả về timestamp chính xác E1–E4**.

TRAKE của bạn vẫn phải dùng:

```text
query
 ↓
event decomposition
 ↓
E1 → retrieve temporal clips
E2 → retrieve temporal clips
E3 → retrieve temporal clips
E4 → retrieve temporal clips
 ↓
existing temporal DP
 ↓
best ordered sequence
```

Tức là TARA nên thay đổi **candidate generation**, không thay thuật toán TRAKE hiện tại.

---

# 10. Cách index mới mới là thứ quyết định model có phát huy hay không

Nếu bạn làm:

```text
TARA(video toàn bộ 10 phút)
→ 1 vector
```

thì sẽ rất phí.

Nếu làm:

```text
TARA(keyframe.jpg)
```

thì còn tệ hơn: lại biến video model thành image model.

Tui đề xuất **multi-scale temporal clip index**:

```text
VIDEO
 │
 ├─ Shot/window A ─── TARA vector
 ├─ Shot/window B ─── TARA vector
 ├─ Shot/window C ─── TARA vector
 │
 ├─ A+B+C ─────────── TARA vector
 ├─ B+C+D ─────────── TARA vector
 └─ C+D+E ─────────── TARA vector
```

Mỗi record:

```text
video_id
start_time
end_time
start_frame
end_frame
tara_embedding
```

Sau đó query:

> “đổ trứng, sau đó cắt đậu hũ rồi cho vào nồi”

được match trực tiếp với một **ordered clip**, chứ không với một ảnh nồi súp.

---

# 11. Với TRAKE, dùng window nhỏ hơn

Ví dụ Q21:

```text
                 video
───────────────────────────────────────

     E1       E2               E3   E4
     ▲        ▲                ▲    ▲
```

Bạn cần:

```text
TARA(query E1) → top temporal clips
TARA(query E2) → top temporal clips
TARA(query E3) → top temporal clips
TARA(query E4) → top temporal clips

                 ↓

existing DP:
t1 < t2 < t3 < t4
```

Đây là kiến trúc rất đẹp vì **TARA xử lý semantics/temporal action**, còn TRAKE DP xử lý **global chronology**.

Hai cái bổ sung nhau, không trùng nhau.

---

# 12. Còn KIS sequence thì dùng cả full query

Ví dụ Q3:

```text
orange vegetable
→ tubular ingredient
→ green vegetables + mushroom
→ sliced meat
```

Tui sẽ chạy song song:

```text
                     Q3
                      │
        ┌─────────────┴──────────────┐
        │                            │
 full sequence query           event decomposition
        │                            │
      TARA                E1 E2 E3 E4 retrieval
        │                            │
        └────────────┬───────────────┘
                     ↓
                 sequence score
```

Có thể định nghĩa:

$$
S(v)
=
\alpha S_{\text{whole-sequence}}
+
(1-\alpha)S_{\text{event-chain}}
$$

Rồi fusion nó với PE/Qwen/OCR/ASR bằng RRF.

---

# 13. Architecture tui sẽ đưa vào repo

Current:

```text
               QUERY
                 │
        ┌────────┼────────┐
        │        │        │
       PE      Qwen     OCR/ASR
        │        │        │
      frame    frame      text
        └────────┼────────┘
                 ↓
                RRF
```

Đổi thành:

```text
                        QUERY
                          │
         ┌────────────────┼────────────────┐
         │                │                │
      STATIC           TEMPORAL          TEXT/AUDIO
         │                │                │
    PE + Qwen           TARA         OCR/ASR/GLAP
         │                │                │
      keyframes      video clips           │
         │                │                │
         └────────────────┼────────────────┘
                          ↓
                         RRF
                          │
               ┌──────────┴──────────┐
               │                     │
              KIS                  TRAKE
               │                     │
          normal rank          event-wise clips
                                     │
                                     ↓
                               temporal DP
```

**Đây là model/module tui nghĩ repo đang thiếu nhất.**

---

# 14. Nếu resource đủ, thêm WeMM sau TARA

Lúc đó architecture cuối sẽ đẹp hơn:

```text
                   VIDEO RETRIEVAL
                         │
               ┌─────────┴─────────┐
               │                   │
             WeMM                TARA
               │                   │
      semantic/general       temporal nuance
               │                   │
               └─────────┬─────────┘
                         ↓
                        RRF
```

WeMM giải:

> “video nào nói chung giống mô tả này?”

TARA giải:

> “video nào có **đúng hành động và đúng chiều/thứ tự thời gian**?”

Đó là complementarity tốt hơn rất nhiều so với:

```text
PE + Qwen + thêm một image model
```

---

# 15. Một điều còn hay hơn: literature đã thử TARA + Qwen ensemble

Tác giả TARA report luôn:

**TARA ⊕ Qwen3VL-Embedding**

và ensemble cải thiện Qwen standalone trên nhiều benchmark:

* MSR: `53.8 → 54.5`
* MSVD: `87.2 → 88.4`
* VATEX: `64.8 → 66.2`
* SSv2 classification: `76.9 → 81.4`

([Oxford Robotics][2])

Đây rất sát với architecture hiện tại của bạn vì **bạn đã có Qwen3-VL-Embedding**.

Tức không phải tui chỉ suy đoán rằng hai model complementary; paper TARA đã có evidence ensemble.

---

# 16. Điều tui sẽ làm ngay cho team

**Phase 1 — không cần re-index full corpus:**

Lấy khoảng **50–100 video**, cố tình gồm cooking/action sequence.

Tạo các query:

```text
positive:
cut onion → fry onion

hard negative:
fry onion → cut onion
```

hoặc dùng chính query đợt 3 rồi tạo:

```text
Q3 original:
orange → tube → vegetables → meat

Q3 reversed:
meat → vegetables → tube → orange
```

Chạy:

```text
PE
Qwen
WeMM-2B
TARA
```

rồi đo:

$$
Acc_{\text{order}}
$$

đặc biệt trên positive vs reversed-order hard negatives.

**Nếu TARA không thắng ở benchmark này thì đừng index cả dataset.**

---

# Quyết định cuối cùng

Sau khi xem bộ đề đợt 3, tui sẽ thay recommendation thành:

> **P0 = TARA**
>
> **P1 = WeMM-2B**
>
> **P2 = MARS/ViLL-E để nghiên cứu tiếp**

Không phải:

> WeMM first rồi TARA second.

**Đối với VBS general retrieval**, WeMM có thể là lựa chọn first-stage cực tốt.

Nhưng **đối với chính failure của AIC đợt 3 — cooking sequences + TRAKE + before/after — TARA là model có evidence trực tiếp mạnh nhất mà tui tìm được hiện tại.**

Ngoài ra, TARA benchmark temporal còn bao gồm **EPIC-KITCHENS**, và khoảng **10 KIS cooking + 1 TRAKE cooking** trong bộ đề của bạn khiến overlap về domain/action type khá đáng chú ý. ([Oxford Robotics][2])

Bước tiếp theo đáng làm nhất là tui có thể thiết kế cho bạn **benchmark “AIC Temporal Retrieval Test” từ chính 36 query này**: tách từng query thành event sequence, tạo reversed-order hard negatives và đưa ra protocol để benchmark **TARA vs WeMM vs Qwen hiện tại**, từ đó quyết định bằng số liệu trước khi bạn bỏ hàng giờ GPU để index toàn bộ corpus.


Lúc thi,  **TARA không trả “đáp án” trực tiếp** . Nó trả về  **embedding vector 3584 chiều cho text query** , rồi hệ thống dùng vector đó để search trong toàn bộ clip embeddings mà bạn đang build.

Trong notebook của bạn, contract query đã ghi rất rõ: query được đưa qua `TARA.encode_text(...)`, sau đó L2-normalize FP32, dùng **cosine similarity** để so với clip embeddings. Query tiếng Việt cũng được định hướng **dịch sang tiếng Anh trước** vì TARA được fine-tune trên dữ liệu tiếng Anh.

Ví dụ lúc thi có query:

```text
Người đầu bếp cắt đầu tôm, sau đó đặt tôm lên đĩa
```

Pipeline sẽ thành:

```text
Vietnamese query
   ↓
translate
   ↓
"A chef cuts the head off a shrimp, then places the shrimp on a plate"
   ↓
TARA.encode_text(...)
   ↓
3584-d vector
   ↓
L2 normalize
   ↓
cosine search với tất cả clip embeddings
   ↓
top-K clips
```

Kết quả trả về thực tế sẽ kiểu:

```text
rank 1
video_id: L26_V183
scale: sequence
start_time: 124.0
end_time: 148.0
score: 0.71

rank 2
video_id: L25_V047
scale: event
start_time: 86.0
end_time: 94.0
score: 0.68

rank 3
video_id: L26_V019
scale: scene
start_time: 210.0
end_time: 282.0
score: 0.65
```

Vì mỗi row trong index của bạn lưu luôn `clip_id`, `video_id`, `scale`, `start_time`, `end_time`, `frame_indices` và embedding, nên retrieval có thể trả về chính xác **clip nào trong video nào** chứ không chỉ video ID.

Điểm hay là bạn đang có 3 scale:

```text
event     8s
sequence 24s
scene    72s
```

nên cùng một query có thể match ở nhiều mức:

```text
"cuts shrimp head"
    → event clip

"cuts shrimp, then plates it"
    → sequence clip

"chef preparing seafood in kitchen"
    → scene clip
```

Nhưng lúc thi tui  **không khuyên dùng raw cosine score của TARA trộn thẳng với PE/Qwen** . Notebook của bạn cũng đã ghi đúng chỗ này: các embedding space có phân phối score khác nhau, nên nên fuse bằng  **rank/RRF** , không cộng cosine trực tiếp.

Kiến trúc runtime hợp lý sẽ là:

```text
                    query
                      │
          ┌───────────┼───────────┐
          │           │           │
         PE          Qwen        TARA
          │           │           │
      keyframes    keyframes     clips
          │           │           │
          └─────── RRF / fusion ──┘
                      │
                 ranked videos
                      │
                human inspect
                      │
                    submit
```

Với  **TRAKE** , TARA còn hữu ích hơn nếu bạn tách query thành event:

```text
E1: mix sauce
E2: cut shrimp head
E3: place shrimp on plate
E4: add orange garnish
```

rồi:

```text
TARA(E1) → top clips
TARA(E2) → top clips
TARA(E3) → top clips
TARA(E4) → top clips
```

sau đó dùng temporal DP hiện tại của bạn để ép:

```text
t1 < t2 < t3 < t4
```

Tức là  **TARA tìm candidate temporal clip tốt hơn** , còn TRAKE solver đảm bảo đúng thứ tự toàn chuỗi.

Nếu bạn muốn hình dung rất ngắn thì:

> **PE/Qwen trả frame nào giống query.**
> **TARA trả clip nào có chuỗi hành động giống query.**

Đó là lý do TARA đặc biệt đáng giá cho các câu cooking/sequence mà hệ thống cũ của bạn yếu.


Đúng theo nghĩa  **đủ hợp lý để chạy full và đem thi** , nhưng tui sẽ không gọi `8s / 24s / 72s` là “đã chứng minh tối ưu tuyệt đối”. Với loại query AIC của bạn, đây là một  **multi-scale design khá cân bằng** , và hiện tại  **không đáng dừng job để đổi** .

TARA mặc định chỉ lấy  **8 frame cho mỗi clip** , nên ba mức hiện tại tương ứng gần như:

| Scale        |        Window | Khoảng cách trung bình giữa frame | Vai trò                               |
| ------------ | ------------: | ------------------------------------: | -------------------------------------- |
| `event`    |  **8s** |                                 ~1.1s | thao tác đơn, chi tiết động tác |
| `sequence` | **24s** |                                 ~3.4s | chuỗi 2–4 hành động               |
| `scene`    | **72s** |                                ~10.3s | ngữ cảnh rộng / long scene          |

Notebook của bạn cũng đang khóa đúng ba scale này với overlap 50%.

Với query đợt 3 thì `8s` và `24s` là hai mức quan trọng nhất. Ví dụ:

```text
"cắt đầu tôm"
→ 8s

"cắt đầu tôm → đặt tôm lên đĩa → thêm garnish"
→ 24s
```

Đặc biệt cooking thường có động tác khá nhanh, nên `8s` giúp không làm loãng temporal signal. `24s` thì đủ dài để chứa vài bước liên tiếp mà vẫn chưa quá sparse.

`72s` thì khác. Nó không nên được hiểu là một temporal window mạnh cho query kiểu A→B→C, vì 8 frame trên 72 giây nghĩa là model chỉ nhìn khoảng mỗi 10 giây một frame. Một hành động 2–3 giây hoàn toàn có thể bị bỏ qua. Nhưng nó vẫn hữu ích cho query dạng:

```text
"một người đang nấu ăn trong bếp ngoài trời"

"người đàn ông chuẩn bị nhiều món ăn quanh một bàn lớn"

"cảnh lớp học với giáo viên trình bày..."
```

nên tui vẫn giữ nó làm  **context/scene retrieval** .

Điểm quan trọng hơn việc đổi duration là  **không cho ba scale đóng góp giống nhau** . Lúc retrieval nên làm kiểu:

```text
query
  ↓
classify intent

single action
  → event 8s: HIGH
  → sequence 24s: medium
  → scene 72s: low

multi-step / before-after / TRAKE
  → event 8s: HIGH
  → sequence 24s: VERY HIGH
  → scene 72s: very low

broad scene/context
  → event 8s: medium
  → sequence 24s: HIGH
  → scene 72s: HIGH
```

Nếu dùng RRF, thậm chí có thể retrieve riêng mỗi scale rồi fuse thay vì quăng toàn bộ 168k vectors vào một ranking mà không biết scale.

Một phương án khác đáng benchmark sau này là:

```text
8s / 24s / 48s
```

vì 48 giây cho 8 frame sẽ còn khoảng ~6.9 giây/frame, bớt sparse đáng kể so với 72 giây. Nhưng  **chưa có bằng chứng rằng 48s sẽ thắng 72s trên AIC của bạn** , nên không có lý do hợp lý để bỏ job hiện tại và re-index lại.

Tui sẽ làm theo chiến lược:

> **Giữ `8 / 24 / 72` cho full corpus hiện tại.**
> Sau khi xong, benchmark riêng từng scale trên query vòng 3: `event-only`, `sequence-only`, `scene-only`, rồi `fusion`.

Nếu kết quả cho thấy `scene=72s` gần như không có unique hits, lúc đó mới cân nhắc bỏ hoặc thay bằng `48s` ở iteration sau.

Vì vậy với job đang chạy hiện tại: **cứ để chạy. `8s / 24s / 72s` đủ tốt để coi là production configuration cho phiên bản đầu dùng thi.** Phần cần tối ưu tiếp theo nên là  **scale-aware retrieval/fusion** , chứ không phải dừng toàn bộ index để đổi window ngay bây giờ.


Đúng hướng nhất là **đưa TARA thành retrieval channel thứ ba trong Milvus**, song song với PE-Core và Qwen3-VL, nhưng **không nên nhét TARA vào cùng collection keyframe**. Lý do là đơn vị retrieval khác nhau: PE/Qwen đang index **frame/keyframe**, còn TARA index **temporal clip có start/end time**.

## 1. Kiến trúc sau khi thêm TARA

Hệ thống nên thành:

```text
                           USER QUERY
                               │
                    normalize / translate
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
        PE-Core           Qwen3-VL              TARA
     text embedding     text embedding       text embedding
           │                   │                3584-d
           ▼                   ▼                   ▼
   Milvus PE index      Milvus Qwen index    Milvus TARA index
      keyframes            keyframes              clips
           │                   │                   │
           │                   │          8s / 24s / 72s
           │                   │                   │
           └───────────────────┼───────────────────┘
                               ▼
                         rank normalization
                               │
                              RRF
                               │
                               ▼
                       VIDEO-LEVEL RANKING
                               │
                   best frame + best TARA clip
                               │
                               ▼
                              UI
```

TARA **không thay PE/Qwen**. Nó giải quyết phần mà hai channel kia yếu hơn:

```text
PE-Core      → visual appearance / object / scene
Qwen3-VL     → semantic understanding
TARA         → action + temporal order / sequence
```

Đây cũng là đúng failure mode mà benchmark AIC của hệ thống hiện tại đang gặp ở sequence queries.

---

# 2. Nên tạo collection riêng cho TARA

Tui đề xuất:

```text
aic_pe_keyframes
aic_qwen_keyframes
aic_tara_clips_v1
```

không phải:

```text
aic_embeddings
 ├ PE vector
 ├ Qwen vector
 └ TARA vector
```

Vì PE/Qwen có identity kiểu:

```text
video_id + frame_idx
```

còn TARA là:

```text
video_id
start_time
end_time
scale
frame_indices[8]
```

Một TARA result ví dụ:

```text
clip_id:
L26_V183__sequence__000037

video_id:
L26_V183

scale:
sequence

start_time:
444.0

end_time:
468.0

embedding:
float[3584]
```

Đây là một retrieval entity hoàn toàn khác keyframe.

---

# 3. Milvus schema tui khuyên dùng

Một collection duy nhất cho cả ba scale TARA:

```python
aic_tara_clips_v1
```

với schema logic:

```text
clip_id          VARCHAR PRIMARY KEY
video_id         VARCHAR
category         VARCHAR

scale            VARCHAR
                 event | sequence | scene

scale_index      INT64

start_time       FLOAT
end_time         FLOAT

fps              FLOAT
duration_sec     FLOAT

frame_indices    ARRAY<INT64>   # optional nhưng nên giữ

embedding        FLOAT_VECTOR(3584)
```

Nếu Milvus deployment hiện tại không tiện lưu array thì `frame_indices` không bắt buộc cho online retrieval. Bạn có thể giữ nó trong Parquet artifact và trong Milvus chỉ lưu:

```text
clip_id
video_id
scale
start_time
end_time
embedding
```

Như vậy đủ cho UI và search.

Tui sẽ **không lưu 3 vector fields event/sequence/scene**.

Tất cả đều là TARA 3584-d trong cùng embedding space nên chỉ cần:

```text
embedding FLOAT_VECTOR(3584)
```

và phân biệt bằng scalar:

```text
scale = event
scale = sequence
scale = scene
```

---

# 4. Import 168,536 clip embeddings vào Milvus

Data hiện tại của bạn đã rất thuận lợi:

```text
Hugging Face Bucket
   ↓
Parquet shards
   ↓
168,536 rows
   ↓
3584-d FP32 L2-normalized
```

Pipeline ingestion:

```text
HF bucket
   │
   ├─ L21/*.parquet
   ├─ L22/*.parquet
   ...
   └─ L30/*.parquet
          │
          ▼
     read parquet
          │
          ▼
      validate
     dim = 3584
     norm ≈ 1
     unique clip_id
          │
          ▼
     Milvus insert
          │
          ▼
 aic_tara_clips_v1
```

Tui sẽ insert batch khoảng:

```text
1,000–5,000 rows / insert
```

không cần gom hết 168k rows vào RAM một lần.

Quan trọng hơn là ingestion nên **idempotent**. Tốt nhất dùng collection version:

```text
aic_tara_clips_v1
```

build xong và audit rồi mới cho backend sử dụng.

Đừng ghi trực tiếp vào collection production đang query rồi vừa insert vừa search.

---

# 5. Metric trong Milvus

Embedding của bạn đã:

```text
FP32
L2 normalized
```

nên:

$$
\cos(q,x)=q^\top x
$$

Bạn có thể dùng:

```text
COSINE
```

hoặc:

```text
IP
```

về mặt ranking sẽ tương đương khi cả query và corpus đều normalized.

Tui nghiêng về **COSINE** vì rõ semantic contract hơn:

```python
metric_type = "COSINE"
```

và lúc query vẫn L2-normalize text embedding để đảm bảo contract.

---

# 6. Không nên search ba scale TARA chung một lần

Đây là điểm rất quan trọng.

Bạn có:

```text
event     8s
sequence 24s
scene    72s
```

Nếu chỉ:

```python
search(
    collection="aic_tara_clips_v1",
    vector=query,
    limit=100
)
```

thì ba loại clip cạnh tranh trực tiếp.

Điều đó không lý tưởng bởi distribution và số lượng clip từng scale khác nhau.

Thay vào đó tui sẽ chạy **3 filtered searches song song**:

```python
event_hits = milvus.search(
    vector=tara_query,
    filter='scale == "event"',
    limit=100,
)

sequence_hits = milvus.search(
    vector=tara_query,
    filter='scale == "sequence"',
    limit=100,
)

scene_hits = milvus.search(
    vector=tara_query,
    filter='scale == "scene"',
    limit=100,
)
```

Thời gian tăng không nhiều nếu Milvus query parallel.

Sau đó mới fuse:

```text
event ranking
sequence ranking
scene ranking
      │
      ▼
 TARA-scale RRF
      │
      ▼
TARA final ranking
```

Như vậy bạn kiểm soát được contribution của từng scale.

---

# 7. Một vấn đề quan trọng: clip duplication

TARA của bạn dùng 50% overlap.

Ví dụ cùng một action có thể tạo:

```text
L26_V001
8s [20,28]
8s [24,32]
8s [28,36]

24s [12,36]
24s [24,48]
```

Nếu search raw top-100, một video rất tốt có thể chiếm:

```text
rank 1
rank 2
rank 3
rank 7
rank 11
...
```

Nếu sau đó fuse trực tiếp với PE/Qwen thì video đó được hưởng lợi **nhiều lần chỉ vì có nhiều overlapping windows**.

Không nên làm vậy.

TARA retrieval cần bước:

```text
clip ranking
     ↓
group by video_id
     ↓
keep best clip / scale
     ↓
video ranking
```

Ví dụ:

```text
Raw:

#1 L26_V183 sequence 120–144  0.73
#2 L26_V183 event    128–136  0.72
#3 L25_V044 sequence 240–264  0.69
#4 L26_V183 event    132–140  0.68
```

sau collapse:

```text
#1 L26_V183  best=0.73
#2 L25_V044  best=0.69
```

nhưng vẫn giữ metadata:

```json
{
  "video_id": "L26_V183",
  "best_tara_clip": {
    "scale": "sequence",
    "start": 120,
    "end": 144,
    "score": 0.73
  }
}
```

Đây sẽ rất hữu ích cho UI.

---

# 8. TARA-scale fusion

Có hai tầng fusion.

Tầng 1:

```text
8s
24s
72s
 ↓
TARA ranking
```

Tầng 2:

```text
PE
Qwen
TARA
 ↓
final ranking
```

Tui sẽ dùng RRF cả hai tầng.

Công thức:

$$
RRF(d)=
\sum_i
\frac{w_i}{k+r_i(d)}
$$

Không nên:

```text
0.4 × PE cosine
+ 0.3 × Qwen cosine
+ 0.3 × TARA cosine
```

vì score distribution của ba embedding spaces hoàn toàn khác nhau.

---

# 9. Ban đầu chưa cần query classifier

Phiên bản đầu tui sẽ giữ đơn giản:

```text
event      weight 1
sequence   weight 1
scene      weight 1
```

và RRF.

Sau khi benchmark mới chuyển thành adaptive weights.

Ví dụ sau này có thể là:

| Query                 |   8s |      24s |         72s |
| --------------------- | ---: | -------: | ----------: |
| single action         |  cao |     vừa |       thấp |
| multi-action sequence |  cao | rất cao |       thấp |
| general scene         | vừa |      cao |         cao |
| TRAKE                 |  cao | rất cao | gần như 0 |

Nhưng **đừng hard-code bảng này trước khi benchmark round-3**.

---

# 10. Fusion với PE + Qwen

Backend hiện tại của bạn đã có logic RRF giữa PE và Qwen, nên TARA nên trở thành channel thứ ba:

```text
PE video rank:
V15
V72
V30
...

Qwen video rank:
V72
V15
V91
...

TARA video rank:
V30
V15
V72
...

             ↓

          RRF

             ↓

V15
V72
V30
...
```

Pseudo-code:

```python
rankings = {
    "pe": pe_results,
    "qwen": qwen_results,
    "tara": tara_results,
}

final = weighted_rrf(
    rankings,
    weights={
        "pe": 1.0,
        "qwen": 1.0,
        "tara": 1.0,
    },
)
```

**Bắt đầu 1:1:1**, benchmark xong mới tune.

---

# 11. Query path của TARA

Query tiếng Việt:

```text
Một người cắt đầu tôm rồi đặt tôm lên đĩa
```

pipeline:

```text
Vietnamese
    ↓
translator
    ↓
A person cuts the head off a shrimp
then places the shrimp onto a plate
    ↓
TARA.encode_text()
    ↓
3584 float
    ↓
L2 normalize
    ↓
Milvus
```

Output Milvus:

```text
video_id
clip_id
scale
start_time
end_time
distance/score
```

Từ đó frontend có thể jump thẳng:

```text
video = L26_V183
seek = 124s
```

Đây là advantage khá lớn so với video-level embedding.

---

# 12. TARA text encoder nên chạy dưới dạng service riêng

Đừng load model mỗi lần query.

TARA ~8.3B parameters nên runtime nên:

```text
tara-service
    │
    ├ model loaded once
    ├ GPU resident
    └ /embed_text
```

Ví dụ API:

```http
POST /embed/tara
```

input:

```json
{
  "texts": [
    "a man cuts a shrimp and then places it on a plate"
  ]
}
```

output:

```json
{
  "embeddings": [
    [...]
  ]
}
```

Backend retrieval:

```text
frontend
  ↓
FastAPI
  ↓
translator
  ↓
TARA service
  ↓
Milvus
```

Tui còn khuyên cache theo normalized query:

```text
sha256(normalized English query)
    ↓
Redis / memory cache
    ↓
3584-d embedding
```

Trong thi, query thường được search nhiều lần khi user chỉnh filters/UI nên cache rất có ích.

---

# 13. Có thể encode nhiều query variant cùng lúc

Đây là chỗ TARA khá hay.

Thay vì một query:

```text
the chef prepares shrimp
```

LLM query expansion có thể sinh:

```text
cuts the head off a shrimp
removes the shrimp head
cuts shrimp then places it on a plate
prepares shrimp and transfers it to a plate
```

TARA hỗ trợ batch text embedding.

Bạn encode:

```text
Q1
Q2
Q3
Q4
```

rồi search Milvus song song và RRF kết quả.

Đặc biệt useful cho query AIC dài/mô tả nhiều bước.

---

# 14. KIS và TRAKE phải dùng TARA khác nhau

### KIS

Có thể dùng full query:

```text
Query
 ↓
TARA
 ↓
top clips
 ↓
video ranking
```

### TRAKE

Không nên chỉ:

```text
whole TRAKE description
→ one TARA vector
```

Nên decomposition:

```text
E1: người lấy quả cà chua
E2: cắt cà chua
E3: cho vào chảo
E4: khuấy
```

encode thành:

$$
q_1,q_2,q_3,q_4
$$

Milvus:

```text
q1 → clips + timestamp
q2 → clips + timestamp
q3 → clips + timestamp
q4 → clips + timestamp
```

group theo:

```text
video_id
```

sau đó đưa vào **TRAKE temporal DP mà hệ thống của bạn đã có**:

$$
t_1<t_2<t_3<t_4
$$

Đây mới là cách tận dụng TARA tối đa.

---

# 15. TARA cũng có thể cải thiện QA

Pipeline QA hiện tại có thể dùng:

```text
query
 ↓
TARA
 ↓
top temporal clips
 ↓
NVILA / VLM
 ↓
answer
```

Thay vì đưa VLM:

```text
whole 10-minute video
```

TARA đưa candidate:

```text
132–156 sec
```

rồi QA model chỉ xử lý vùng đó.

Có thể giảm inference cost đáng kể.

---

# 16. Milvus index nên chọn gì?

Với:

```text
168,536 vectors
× 3584 dimensions
≈ 2.25 GiB FP32 raw
```

đây thực ra chưa phải corpus quá lớn.

Tui sẽ không vội quantize.

Nếu PE/Qwen hiện tại dùng một index type đã ổn trong Milvus, **iteration đầu tiên nên dùng cùng operational pattern** để giảm số biến thay đổi.

Nếu thiết kế mới hoàn toàn, tui sẽ benchmark:

```text
FLAT / exact baseline
vs
HNSW
vs
IVF_FLAT
```

trên chính TARA corpus.

Exact baseline cực kỳ quan trọng vì nó cho biết approximate index làm mất bao nhiêu retrieval quality.

Không nên chọn IVF_PQ ngay chỉ vì tiết kiệm RAM; 2.25 GiB vector data chưa đủ lớn để chấp nhận quality loss khi chưa benchmark.

---

# 17. Collection versioning

Tui rất khuyên:

```text
aic_tara_clips_v1
```

thay vì:

```text
tara
```

Sau này nếu bạn thử:

```text
8 / 24 / 48
```

thì:

```text
aic_tara_clips_v2
```

không overwrite v1.

Có thể dùng alias:

```text
aic_tara_clips
          │
          └── aic_tara_clips_v1
```

Sau này validation v2 xong mới swap alias.

Điều này tránh phá hệ thống thi.

---

# 18. Backend response nên thay đổi một chút

Hiện result PE/Qwen có thể thiên về:

```json
{
  "video_id": "...",
  "frame_idx": 12345
}
```

Sau TARA nên hỗ trợ:

```json
{
  "video_id": "L26_V183",
  "score": 0.01234,

  "best_frame": {
    "frame_idx": 12345,
    "source": "qwen"
  },

  "best_clip": {
    "start_time": 124.0,
    "end_time": 148.0,
    "scale": "sequence",
    "source": "tara"
  },

  "sources": [
    "pe",
    "qwen",
    "tara"
  ]
}
```

Frontend có thể hiển thị:

```text
TARA: 02:04 → 02:28
```

và click một cái là video seek thẳng vào vùng đó.

---

# 19. Flow tui đề xuất cho repo AIC2026

Toàn pipeline cuối cùng:

```text
                     QUERY
                       │
                 query parser
                       │
               VI → EN translator
                       │
        ┌──────────────┼───────────────┐
        │              │               │
        ▼              ▼               ▼
      PE text       Qwen text       TARA text
        │              │              3584
        ▼              ▼               │
    Milvus PE      Milvus Qwen     Milvus TARA
        │              │               │
   keyframes       keyframes      ┌────┼────┐
                                  │    │    │
                                  8s  24s  72s
                                  │    │    │
                                  └────┼────┘
                                       │
                                  TARA RRF
                                       │
                                  collapse video
        │              │               │
        └──────────────┼───────────────┘
                       │
                  Global RRF
                       │
               relevance feedback
                       │
                     TRAKE
                 (when needed)
                       │
                       ▼
                      UI
```

---

# Cách tui sẽ triển khai thực tế

Thứ tự implementation hợp lý nhất là:

1. tạo `aic_tara_clips_v1`;
2. import toàn bộ **168,536 embeddings** từ HF Parquet vào Milvus;
3. viết `/embed/tara` text service;
4. viết `search_tara()` với 3 filtered searches `event/sequence/scene`;
5. collapse clip → video;
6. TARA-scale RRF;
7. thêm `tara_results` vào RRF PE/Qwen hiện tại;
8. frontend hiện best temporal clip;
9. benchmark round-3;
10. sau đó mới tune weights và TRAKE integration.

Điểm quan trọng nhất là **đừng xem TARA chỉ là “thêm một vector field nữa”**. PE/Qwen retrieval entity là keyframe, còn TARA retrieval entity là **time interval**. Vì thế Milvus storage tương đối đơn giản; phần cần làm cẩn thận thực sự nằm ở **clip→video aggregation, scale fusion và global RRF**.

Nếu làm đúng kiến trúc này thì TARA có thể được thêm vào hệ thống hiện tại mà **không cần phá pipeline PE/Qwen đang ổn**, đồng thời TRAKE và UI có thể sử dụng timestamp trực tiếp từ TARA.
