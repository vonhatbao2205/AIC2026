
## 2. P0 quan trọng nhất: Progressive Hint + Early Submit Confidence

Đây là module mình thấy **thiếu rõ nhất** so với objective chung kết.

Hiện `previous_hints` về bản chất được nối vào query:

```

```

```
hint1
  ↓
hint1 + hint2
  ↓
hint1 + hint2 + hint3
```

rồi search lại.

Nhưng BTC đang biến nó thành một **sequential decision problem**:

```

```

```
Hint 1
 ↓
Có candidate A
 ↓
Submit ngay?
hay chờ Hint 2?
 ↓
Hint 2
 ↓
A vẫn đứng đầu?
 ↓
độ tin cậy tăng
```

Mình sẽ tạo module riêng:

```

```

```
ProgressiveQuerySession
    ├── hints[]
    ├── rankings_by_hint[]
    ├── candidate_trajectory
    ├── confidence
    └── submit_recommendation
```

### Candidate trajectory

Ví dụ:

```

```

```
                  Hint 1   Hint 2   Hint 3
L26_V004             #2       #1       #1
L21_V018             #1       #4      #11
L30_V007             #4       #3       #6
```

L26\_V004 không chỉ đang rank #1.

Nó còn:

- rank tăng theo hint,
- giữ #1 liên tục,
- PE đồng ý,
- Qwen đồng ý,
- reranker đồng ý,
- temporal cluster tập trung.

Đây là tín hiệu **mạnh hơn rất nhiều** so với RRF score tuyệt đối.

### Features nên dùng

Không cần neural network lớn. Với dữ liệu team hiện có, logistic regression / calibrated GBDT thậm chí isotonic regression là đủ.

Một vector có thể là:

```

```

```
top1_rank_stability
top1_vs_top2_margin
PE_rank
Qwen_rank
PE_Qwen_overlap@K
reranker_margin
channel_agreement_count
video_cluster_support
temporal_dispersion
ambiguous
scope_match
hints_survived
rank_improvement
rank_entropy
```

Output:

```

```

```
P(correct video) = 0.96
P(correct region) = 0.91

Recommendation:
STRONG — submit
```

hoặc:

```

```

```
P(correct video) = 0.68

HOLD
Reason:
Qwen #1 nhưng PE #17
top candidate mới xuất hiện ở hint hiện tại
```

### Không nên auto-submit

Ít nhất ở vòng cuối này mình sẽ để:

```

```

```
🟢 STRONG
🟡 HOLD
🔴 UNCERTAIN
```

và con người quyết định.

Vì false positive của confidence module ảnh hưởng trực tiếp số lần submit sai.

Đây là trường hợp rất hợp với dữ liệu hai đợt thi trước: replay lại các query theo từng phần mô tả/hint rồi xem **thời điểm sớm nhất nào hệ thống đã có đúng video/keyframe một cách ổn định**.

Metric mới nên là:

```

```

```
Time-to-stable-correct
Wrong-submit rate
Prefix Hit@1
Prefix MRR
Calibration / Brier score
```

chứ không chỉ Hit\@100.
Kết hợp những thứ trên và research kỹ lại để tối ưu thuật toán progressive hint memory

Sau khi đối chiếu **pipeline hiện tại của repo**, benchmark của team và các hướng nghiên cứu gần đây về interactive KIS / conversational retrieval / query-performance prediction, mình sẽ thay thiết kế Progressive Hint ban đầu bằng một phiên bản chặt hơn:

> **Progressive Hint Memory không nên là “concatenate hint”, cũng không nên là hard-AND các hint.**
> Kiến trúc phù hợp nhất với AIC hiện tại là **multi-view retrieval theo từng hint + video-level evidence memory + soft conjunction + trajectory reflection + survivor rescue**, rồi huấn luyện **confidence model riêng** để quyết định `STRONG / HOLD / UNCERTAIN`.

Mình tạm gọi nó là **PHM-v2 — Progressive Hint Memory v2**.

---

# 1. Vấn đề thật sự của pipeline hiện tại

Trong `heuristic_parse()`, repo hiện làm:

```python
previous_hints = previous_hints or []
combined = " ".join([*previous_hints, query]).strip()
```

và sau đó visual query của PE, các cue OCR/speech/audio… đều chủ yếu được xây từ `combined`.

LLM path cũng được truyền `Previous accumulated hints`, còn cache key chứa toàn bộ `previous_hints`.

Tức bản chất hiện tại là:

```text
H1
 ↓
retrieve(H1)

H2
 ↓
retrieve(H1 + H2)

H3
 ↓
retrieve(H1 + H2 + H3)
```

Điều thú vị là đây cũng gần cách VBS phát textual KIS: query được tiết lộ qua ba round và text mới được append vào text cũ. Nhưng paper ICMR 2025 về interactive KIS cho thấy chính bài toán này hưởng lợi mạnh từ việc giữ **state qua nhiều vòng** thay vì coi mỗi round như một query độc lập; trên 17 textual-KIS của VBS, hệ thống của họ tăng R@1 từ 10/17 ở round 1 lên 14/17 rồi 16/17 ở round 3. ([arXiv][1])

Đồng thời, nghiên cứu conversational search cho thấy lấy toàn bộ history một cách không chọn lọc có thể gây **topic/query drift**; selective contextual history hiệu quả hơn vì chỉ đưa phần history cần thiết để giải ambiguity/coreference. ([ScienceDirect][2])

Đây chính xác là lý do mình **không muốn đơn giản đổi từ concatenate sang “chỉ search hint mới”**. Cả hai cực đều chưa tối ưu.

---

# 2. Kiến trúc mình khuyên dùng: hai pathway

Mình sẽ học một ý rất hợp lý từ các interactive retrieval system gần đây: tách **intent processing** khỏi **trajectory/reflection**.

```text
                  NEW HINT H_t
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
      INTENT PATH           REFLECTION PATH
            │                     │
    search new evidence       memory/history
            │                     │
   PE / Qwen / OCR ...       ranks H1..Ht
            │                     │
            └──────────┬──────────┘
                       ▼
                VIDEO MEMORY
                       │
                 session rank
                       │
          ┌────────────┴───────────┐
          ▼                        ▼
      candidates               confidence
                                   │
                           STRONG / HOLD
```

Điểm quan trọng:

**Intent Path hỏi:** “Hint mới cung cấp evidence gì?”

**Reflection Path hỏi:** “Ranking của chúng ta đã tiến triển như thế nào qua các hint?”

Hai cái không nên trộn thành một embedding.

---

# 3. Một hint không phải một query: dùng 3 query views

Ở round \(i\), mình sẽ tạo tối đa ba view.

```text
Q_delta:
    chỉ thông tin MỚI xuất hiện ở hint i

Q_context:
    hint mới + context tối thiểu cần thiết từ hint trước

Q_cumulative:
    toàn bộ description H1...Hi
```

Nhưng trọng số khác nhau:

$$
a_\Delta > a_{ctx} > a_{cum}
$$

Khởi điểm hợp lý để benchmark:

```text
delta              1.00
selective context  0.65–0.80
cumulative guard   0.20–0.35
```

Đây chỉ là seed để grid-search/CV, không phải giá trị mình khuyên hard-code cuối cùng.

### Ví dụ

H1:

```text
Một người đàn ông đứng trong bếp
```

H2:

```text
Phía sau ông ấy là một chiếc tủ lạnh trắng
```

Không thể search H2 hoàn toàn độc lập vì `"ông ấy"` thiếu antecedent.

Ta tạo:

```text
delta:
white refrigerator behind a man

context:
man standing in kitchen with white refrigerator behind him

cumulative:
man standing in kitchen, white refrigerator behind him
```

Nhưng nếu H3 là:

```text
Trên bàn có một chai nước màu đỏ
```

thì:

```text
delta:
red bottle on a table
```

đã rất informative và không cần kéo toàn bộ description cũ vào embedding.

Nghiên cứu selective-history cũng ủng hộ kiểu lựa chọn context theo nhu cầu thay vì luôn dùng full history. ([ScienceDirect][2])

---

# 4. Nếu BTC gửi cumulative hint thay vì delta thì phải tự diff

Đây là chi tiết nhỏ nhưng rất quan trọng.

Giả sử frontend nhận:

```text
Round 1:
A man is sitting at a table.

Round 2:
A man is sitting at a table.
There is a red bottle in front of him.

Round 3:
A man is sitting at a table.
There is a red bottle in front of him.
A white refrigerator is visible behind him.
```

Không được coi toàn bộ Round 3 là hint mới.

Session nên lưu:

```python
HintTurn(
    raw_text=...,
    delta_text=...,
    cumulative_text=...,
)
```

và lấy sentence/span difference:

```text
H1 delta = man sitting at table
H2 delta = red bottle...
H3 delta = white refrigerator...
```

Như vậy cùng một evidence không bị “vote” ba lần chỉ vì BTC append text.

---

# 5. Mỗi hint retrieval riêng, nhưng memory phải ở VIDEO level

Đây là điểm mình giữ nguyên từ phân tích trước.

Repo hiện đã có `group_by_video()` rất tốt:

$$
videoScore=
0.75\,best+
0.15\,meanTop+
0.10\,cluster
$$

và còn tính `timestamp_dispersion`, `ambiguous`, cluster support.

Nhưng score đó chỉ tồn tại **trong một lần search**.

PHM cần state:

```python
ProgressiveQuerySession
    hints
    turns
    candidates
    ranking_history
    top1_history
    confidence
    recommendation
```

Một candidate:

```python
CandidateMemory
    video_id

    rank_by_hint
    rank_pe_by_hint
    rank_qwen_by_hint
    rank_reranker_by_hint

    evidence_by_hint

    best_frames_by_hint
    timestamps_by_hint

    first_seen_turn
    best_rank
    current_rank

    top1_streak
    top5_streak
    survival_streak

    rank_slope
    rank_volatility

    temporal_coherence
    ambiguous
```

Đến H3, video không mất H1/H2 evidence chỉ vì H3 retrieval của nó yếu.

---

# 6. Không hard-AND hint coverage

Đây là thay đổi quan trọng nhất so với công thức mình đề xuất trước.

Ban đầu ta có thể nghĩ:

```text
video A: match 4/4 hint
video B: match 3/4 hint

→ A luôn thắng
```

Nhưng thực tế có các ca:

```text
H1: "một người đàn ông"
```

H1 gần như vô dụng.

Hoặc:

```text
H3: "phía sau có bảng màu trắng"
```

video đúng có bảng đó nhưng đúng keyframe không lọt top-K.

Hoặc một hint mô tả context của shot trước/sau chứ không phải cùng frame.

Hard coverage sẽ xử lý cả ba sai.

### Mình chọn Soft Product of Experts

Cho evidence của video \(v\) ở hint \(i\):

$$
e_i(v)\in[0,1]
$$

Ta tính memory:

$$
M_t(v)=
\exp\left[
\frac{
\sum_{i=1}^{t}w_i
\log\left(
\epsilon+(1-\epsilon)e_i(v)
\right)
}{
\sum_i w_i
}
\right]
$$

Trong đó:

* \(e_i(v)\) = evidence của hint \(i\);
* \(w_i\) = reliability/informativeness của hint;
* \(\epsilon\) = floor chống việc một retrieval miss biến score thành 0.

Ví dụ thử:

$$
\epsilon=0.1
$$

Video A:

```text
0.8  0.7  0.8
```

sẽ rất mạnh.

Video B:

```text
0.95  0.00  0.95
```

bị phạt đáng kể, nhưng **không chết hoàn toàn**.

Đó chính là behavior ta cần.

Mình sẽ sweep:

$$
\epsilon \in \{0.05,0.10,0.15,0.20\}
$$

trên historical replay.

---

# 7. `e_i(v)` không được lấy PE cosine + Qwen cosine

Repo hiện đang xử lý chuyện này đúng: PE và Qwen ở hai embedding spaces khác nhau, khi không rerank chúng gặp nhau bằng rank fusion; khi rerank thì union candidate trước rồi Qwen reranker tạo một visual ranking.

PHM cũng phải giữ invariant đó.

Mình sẽ biến mỗi hint thành một **video ranking**, rồi convert rank → evidence.

Một cách rất ổn:

$$
e_i(v)
=
\frac{k+1}{k+r_i(v)}
$$

sau khi đã fusion các query-view/ranking.

Nếu video rank #1:

$$
e=1
$$

Nếu rank thấp thì giảm từ từ.

Hoặc dùng normalized video-RRF.

Điểm quan trọng là:

> **Progressive memory nên tích lũy ranks/evidence chuẩn hóa, không tích lũy raw model scores.**

Điều này còn hữu ích cho confidence vì QPP literature cho thấy score distribution phụ thuộc retriever; lấy score absolute từ hai model khác nhau rất dễ tạo feature sai nghĩa. ([arXiv][3])

---

# 8. Hint reliability: đừng học phức tạp ngay

Một phiên bản quá tham vọng sẽ đặt:

$$
w_i=f(\text{entropy, PE agreement, Qwen, novelty, margin,...})
$$

ngay từ đầu.

Mình không làm vậy cho final.

Dataset của repo hiện vẫn khá nhỏ: benchmark T-KIS có 67 query dùng được, trong đó 45 câu single-moment và 22 câu thực chất chứa context/sequence; chính README cũng đã cảnh báo rằng những cải thiện rất nhỏ dễ chỉ là noise hoặc local optimum.

Research về conversational QPP cũng thấy supervised QPP chỉ vượt rõ unsupervised khi có training set đủ lớn. ([ResearchGate][4])

Vì vậy:

### PHM-v1

```text
w_i = 1
```

cho mọi hint thực sự mới.

Chỉ giảm weight khi text bị lặp gần như nguyên xi.

### PHM-v2 sau benchmark

Ta mới thử:

$$
w_i=
f(
discriminativeness,
query\ clarity,
channel\ agreement
)
$$

Một hint generic:

```text
"a person is visible"
```

sẽ có top ranking rất phẳng → reliability thấp.

Một hint:

```text
"yellow excavator beside a blue bus"
```

tạo top-list rất concentrated → reliability cao.

Đây là một dạng QPP rất nhẹ.

---

# 9. Candidate trajectory phải là first-class evidence

Đây là phần mình đồng ý hoàn toàn với ý tưởng bạn đưa.

Ví dụ:

| Candidate      |          H1 | H2 | H3 |          H4 | Diễn giải                      |
| -------------- | ----------: | -: | -: | ----------: | -------------------------------- |
| **V004** |          18 |  6 |  2 | **1** | hội tụ rất đẹp              |
| V018           | **1** |  4 | 11 |          19 | drift khỏi target               |
| V007           |           4 |  3 |  6 |           5 | plausible nhưng không hội tụ |

`V004` mạnh hơn một candidate tình cờ vừa xuất hiện #1 ở H4.

Các feature mình ưu tiên:

```text
current_rank
best_rank
reciprocal_rank

top1_streak
top3_streak
top10_survival

first_seen_turn

rank_slope
rank_volatility

top1_changed_this_turn

top1_vs_top2_margin

PE_Qwen_agreement

ranking_stability
```

### Ranking stability nên dùng RBO

Jaccard@10:

```text
{top10 H2} ∩ {top10 H3}
```

không biết #1 quan trọng hơn #10.

**Rank-Biased Overlap (RBO)** được thiết kế đúng cho việc so hai ranked prefixes và đặt trọng số lớn hơn lên phần đầu ranking. ([blog.mobile.codalism.com][5])

Ta có thể tính:

$$
RBO(R_{t-1},R_t)
$$

với \(p\) khoảng `0.8–0.9`.

Ví dụ:

```text
RBO = 0.88
top1 unchanged
margin grows
```

→ ranking đang hội tụ.

Ngược lại:

```text
RBO = 0.31
top1 changed
new top1 chưa từng top20
```

→ HOLD.

---

# 10. Nhưng trajectory không nên quyết định ranking quá mạnh

Đây là nuance quan trọng.

Candidate đúng có thể chỉ được một hint cực kỳ discriminative “cứu” ở round cuối:

```text
H1 #312
H2 #95
H3 #1
```

Nếu penalize trajectory quá mạnh thì target thật không lên được.

Do đó trong bản final-safe:

```text
semantic memory
    ↓
quyết định ranking chính

trajectory
    ↓
confidence + tie breaker
```

chứ không phải:

```text
trajectory bonus × 10
→ ép video cũ giữ top
```

Progressive memory phải cho phép **belief revision**.

---

# 11. Survivor Rescue: đây có thể là cải tiến ranking tốt nhất

Phần này cực kỳ hợp với code hiện tại.

Giả sử:

```text
H1:
correct video V123 rank #2

H2 global:
V123 không có frame trong top100
```

Không có nghĩa V123 không match H2.

Có thể frame H2 của V123 rank global #137.

### PHM nên làm

```text
global search H2
       │
       ├──── new global candidates
       │
       ▼
memory top survivors
V123 V055 V210...
       │
       ▼
"V123 có match H2 ở đâu trong video không?"
       │
    local search
       │
       ▼
PE/Qwen restricted to video_id=V123
```

Điều hay là **repo đã có primitive này**.

`MilvusClient.search_image()` và `search_qwen_image()` đã nhận:

```python
video_id: str | None
```

và comment ghi rõ đây là cơ chế dùng cho TRAKE pass-2 để tìm frame của một event bên trong một video.

Chỉ cần expose nó lên `_search_visual_model()`.

Mình đề xuất mỗi hint:

```text
GLOBAL
top 100–150 frames

+

RESCUE
top 8–12 session videos
chỉ rescue video thiếu evidence ở hint mới
top 3–5 local frames/video
```

Không search local tất cả 300 candidate.

---

# 12. Đây chính là “soft search-space pruning”

Paper Robust Relevance Feedback cho thấy search-space pruning tăng R@1 rõ rệt trong setup của họ. ([arXiv][1])

Nhưng mình không hard-prune corpus trong AIC.

Thay vào đó:

```text
global branch:
    luôn có khả năng phát hiện video mới

memory branch:
    search sâu các survivor cũ
```

Tức:

```text
exploration + exploitation
```

Một hint mới luôn có quyền đưa một video hoàn toàn mới vào top.

Nhưng những video đã tích lũy evidence không bị bỏ chỉ vì một global top-K miss.

Đây là compromise mình cho là rất mạnh.

---

# 13. Candidate memory chỉ nên giữ khoảng 200–300 video

Không cần lưu mọi video từng retrieve.

Một policy đơn giản:

```text
current global Top-N
∪
previous session Top-N
∪
all candidates từng vào Top-K_small
```

rồi cap 200–300.

Ví dụ:

```text
global keep          150
historical survivors 100
protected top20       ∞ trong session
```

Sau merge cap về 250.

Không hard-remove những video từng rất mạnh nếu chỉ vừa tụt một vòng.

---

# 14. Temporal coherence: có giá trị, nhưng phải biết query thuộc kiểu nào

Repo hiện đã tính:

```text
timestamp_dispersion
cluster_support
ambiguous
```

và mặc định coi gap >30 giây là ambiguous.

Progressive memory có thể tận dụng mạnh hơn.

Giả sử các hint của video A match ở:

```text
H1  04:20
H2  04:24
H3  04:27
H4  04:23
```

Rất mạnh.

Video B:

```text
H1  01:10
H2  05:42
H3  09:03
H4  13:27
```

Nếu đề mô tả **một khoảnh khắc**, B rất đáng nghi.

Có thể dùng:

$$
T(v)=
\frac{
\sum_i w_i
e^{-|t_i-\tilde t|/\sigma}
}{
\sum_i w_i
}
$$

với \(\tilde t\) là weighted median timestamp.

### Nhưng không hard enforce

Benchmark hiện tại của repo đã cho một tín hiệu rất quan trọng: 22 query mô tả nhiều hành động có R@1 chỉ ~18%, nhưng R@100 gần tương đương nhóm single-moment.

Vì vậy PHM cần:

```text
temporal_mode = same_scene
```

hoặc:

```text
temporal_mode = multi_shot_context
```

`same_scene` → coherence bonus.

`multi_shot_context` → không penalize distant timestamps; thậm chí sau này có thể dùng TRAKE-style ordering.

Trước final, mình chỉ dùng temporal coherence như **tie-breaker/confidence feature**, không làm hard filter.

---

# 15. Công thức ranking mình sẽ ship trước

Sau tất cả cân nhắc, mình không khuyên làm một công thức 15 weight.

Bản production đầu tiên nên đơn giản.

Cho mỗi hint:

$$
e_i(v)=\text{rank-normalized evidence}
$$

Memory:

$$
M_t(v)
=
GeoMean_w(
\epsilon+(1-\epsilon)e_i(v)
)
$$

Current hint:

$$
L_t(v)=e_t(v)
$$

Temporal:

$$
T_t(v)\in[0,1]
$$

Final ranking energy:

$$
S_t(v)
=
\alpha M_t(v)
+
\beta L_t(v)
+
\gamma T_t(v)
$$

Seed để sweep nhỏ:

```text
α memory        0.65–0.80
β latest hint   0.15–0.30
γ temporal      0.00–0.10
```

Mình sẽ bắt đầu:

```text
0.70 / 0.25 / 0.05
```

nhưng **không ship con số này nếu chưa replay CV**.

Trajectory không nằm trong formula chính ở v1.

Nó đi sang confidence model.

---

# 16. Early Submit Confidence phải tách hoàn toàn khỏi ranking

Đây là nguyên tắc kiến trúc mình nhấn mạnh.

```text
SessionScore(V004) = 0.79
```

không có nghĩa:

```text
P(V004 đúng) = 79%
```

RRF cũng không phải probability.

Do đó:

```text
retrieval/ranking
        ↓
feature extraction
        ↓
confidence model
        ↓
P(correct)
```

---

# 17. Mình sẽ có hai confidence model

Không chỉ một.

## Video confidence

$$
P_v=P(\text{Top1 video đúng}\mid x)
$$

## Region confidence

$$
P_r=P(\text{chosen region/frame đúng}\mid x)
$$

Vì hoàn toàn có ca:

```text
P(video correct)  = 0.97
P(region correct) = 0.62
```

→ không nên submit frame hiện tại; nên mở video/local refine.

Hoặc:

```text
0.96 / 0.94
```

→ STRONG.

---

# 18. Feature cho `P_video`

Một compact feature vector khoảng 15–25 chiều là đủ:

```text
session top1/top2 margin
session top1/top5 margin

latest top1 rank evidence
memory score
weighted hint coverage
weakest-hint evidence

top1 streak
top3 survival streak
rank slope
rank volatility
first_seen_turn

RBO(t-1,t)

PE rank
Qwen rank
PE-Qwen overlap/RBO
reranker rank/margin

channel agreement count

temporal coherence
timestamp dispersion
ambiguous

rescued_this_turn
new_top1_this_turn
```

Không cần VLM hay neural network nữa.

---

# 19. Logistic regression trước, không GBDT trước

Ở đây mình thay đổi nhẹ đề xuất ban đầu của bạn.

Với dữ liệu hiện tại, mình ưu tiên:

```text
L2 Logistic Regression
```

trước:

```text
GBDT
```

và chắc chắn trước isotonic.

Lý do là prefixes của cùng query **không độc lập** và số query thực sự khá nhỏ. QPP research cho thấy supervised predictors cần lượng training data đáng kể mới thể hiện ưu thế ổn định. ([ResearchGate][4])

Đặc biệt, tài liệu calibration của scikit-learn cảnh báo **isotonic dễ overfit khi dataset nhỏ**, và thường chỉ thực sự phù hợp khi có khoảng trên 1.000 sample calibration; sigmoid phù hợp hơn với sample nhỏ. ([Scikit-learn][6])

Với tình huống này:

```text
logistic regression
        ↓
reliability check
```

Nếu cần post-calibration:

```text
sigmoid / Platt
```

Mình chưa dùng isotonic.

---

# 20. Tuyệt đối không random split prefix

Ví dụ query Q17 có:

```text
Q17-H1
Q17-H2
Q17-H3
```

Nếu:

```text
H1 train
H2 train
H3 test
```

thì leakage rất nặng.

Split phải theo:

```text
query session
```

tốt hơn nữa:

```text
target video
```

nếu nhiều query có chung target.

Ví dụ:

```text
GroupKFold(target_video)
```

Toàn bộ prefixes của một target cùng ở train hoặc test.

Repo của bạn đã có tư duy CV rất đúng trong answer generator; README hiện cũng cảnh báo cụ thể việc chasing tiny in-sample gain.

---

# 21. `STRONG` threshold không được chọn kiểu `p > 0.9`

Nên chọn bằng **risk–coverage**.

Với threshold \(\theta\):

$$
Coverage(\theta)
=
\frac{\# recommended}{\# prefixes}
$$

$$
Risk(\theta)
=
\frac{\# wrong\ recommended}{\# recommended}
$$

Ví dụ:

```text
θ      Coverage    Wrong risk
0.80     62%          8%
0.90     41%          3%
0.95     28%          1%
```

Ta chọn threshold dựa vào cost thực tế của submit sai.

Local simulator của repo hiện mô hình hóa penalty submit sai lớn hơn lợi ích time bonus, nên nó khuyến khích conservative submit; nhưng file cũng ghi rõ simulator **không phải official BTC server**, do đó threshold cuối phải dùng scoring rule chính thức của final nếu có.

---

# 22. Recommendation policy mình sẽ dùng

Không auto-submit.

```text
                ┌──────────────┐
                │ confidence   │
                └──────┬───────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      STRONG          HOLD        UNCERTAIN
```

`STRONG` không chỉ dựa vào \(P_v\).

Nó nên yêu cầu:

```text
P_video cao
P_region cao

AND

không ambiguous nặng

AND

top1 margin đủ lớn
```

Với Hint ≥2, thêm một stability gate:

```text
top1 streak >= 2
```

**nhưng không bắt buộc streak ở H1.**

Nếu H1 đã:

```text
PE #1
Qwen #1
reranker #1
giãn cách top2 cực lớn
local region rõ ràng
```

thì system vẫn phải có quyền báo STRONG ngay.

Nếu không, ta sẽ tự phá mục tiêu early submission.

---

# 23. Stage 2 sau này mới làm Optimal Stopping

Khi đã có nhiều replay data và biết scoring chính thức, có thể nâng từ classifier sang sequential decision.

Submit hiện tại:

$$
U_{submit}(t)=
p_tR(t)-(1-p_t)C_{wrong}
$$

Chờ hint:

$$
U_{wait}(t)=
E[V_{t+1}\mid state_t]
$$

Submit khi:

$$
U_{submit}>U_{wait}
$$

Đây mới là formulation “đúng toán” của chung kết.

Nhưng mình **không ship cái này trước** nếu chỉ có hai round historical data.

Risk-controlled `STRONG/HOLD` sẽ robust hơn.

---

# 24. Replay dataset cần xây như thế nào

Đây là phần quyết định module thành công hay không.

Đối với mỗi task cũ:

```text
Query Q
 ├ H1
 ├ H2
 ├ H3
 └ GT
```

Replay:

```text
H1
↓
run PHM
↓
save complete session state

H2
↓
update PHM
↓
save state

H3
↓
update PHM
↓
save state
```

Mỗi prefix trở thành một sample:

```text
Q17 / turn1 / top1 wrong
Q17 / turn2 / top1 correct
Q17 / turn3 / top1 correct
```

Nhưng split vẫn theo Q17, không theo sample.

### Nếu chỉ còn final descriptions mà không còn historical reveal

Có thể synthetic-split câu thành H1/H2/H3 để **test ranking algorithm**.

Nhưng mình sẽ **không dùng synthetic prefixes để calibrate confidence threshold**.

Distribution của hint do mình tự split khác hint thật của BTC.

---

# 25. Metric mới: `Time-to-stable-correct` cực kỳ đúng

Mình đồng ý với metric bạn đề xuất và sẽ định nghĩa chính xác:

$$
TSC(Q)
=
\min t
$$

sao cho:

```text
rank_t(GT)=1
rank_{t+1}(GT)=1
...
rank_T(GT)=1
```

Ví dụ:

```text
GT ranks:
7 → 1 → 3 → 1
```

`first correct = 2`

nhưng:

```text
time-to-stable-correct = 4
```

Cái thứ hai quan trọng hơn cho early submit.

Benchmark suite final nên đo:

```text
Prefix Hit@1
Prefix Hit@5
Prefix MRR

First-correct turn
Stable-correct turn

GT-video memory survival

Top1 churn

STRONG coverage
STRONG wrong-submit rate

Brier
log loss
reliability curve

risk-coverage / AURC

local-rescue recovery
latency overhead
```

Brier hữu ích nhưng không nên dùng một mình: tài liệu calibration lưu ý Brier đồng thời chịu ảnh hưởng bởi calibration lẫn discrimination. Reliability diagram phải được xem cùng nó. ([Scikit-learn][6])

---

# 26. Ablation mình sẽ chạy

Đừng tune tất cả cùng lúc.

```text
A current
  cumulative concat

B
  delta-only

C
  delta + cumulative guard

D
  + progressive soft memory

E
  + survivor rescue

F
  + temporal tie-break

G
  + trajectory features

H
  + confidence / STRONG policy
```

Nếu:

```text
C > A
D > C
E > D
```

ổn định qua held-out sessions thì mới giữ.

Nếu một bước chỉ:

```text
+0.003 MRR
```

trong khi CV đảo dấu, bỏ.

Chính benchmark hiện tại của repo đã chứng minh việc tối ưu trên ít query dễ tạo “win” nhỏ hơn độ phân giải thật của metric.

---

# 27. Cách tích hợp vào repo hiện tại

Mình sẽ **không sửa thẳng `/api/search` ngay**.

Thêm:

```text
backend/app/progressive.py
```

chứa pure logic:

```text
HintTurn
CandidateMemory
ProgressiveSession

derive_delta()
rank_to_evidence()
soft_memory_score()
trajectory_features()
RBO()
temporal_coherence()
```

Thêm:

```text
backend/app/services/progressive_service.py
```

orchestration:

```text
current hint
→ query views
→ existing SearchService
→ global ranking
→ survivor rescue
→ update memory
→ session ranking
```

Thêm:

```text
backend/app/confidence.py
```

cho:

```text
extract_features()
video_confidence()
region_confidence()
recommendation()
```

API:

```text
POST /api/search/progressive
```

hoặc:

```text
POST /api/progressive/update
```

Response:

```json
{
  "turn": 3,
  "ranking": [],
  "trajectory": {},
  "confidence": {
    "video": 0.94,
    "region": 0.91,
    "recommendation": "strong"
  }
}
```

---

# 28. Một refactor quan trọng trong QueryParser

Không được tiếp tục để:

```python
combined
```

vừa quyết định routing, vừa trở thành retrieval query.

Mình tách:

```text
routing_context
retrieval_text
```

Ví dụ:

```python
routing_context = H1 + H2 + H3
retrieval_text = H3_delta
```

Cumulative history vẫn hữu ích để biết:

```text
đây là QA?
có OCR không?
scope nào?
đại từ nói về ai?
```

nhưng visual embedding hiện tại nên nhận query-view mà ProgressiveService lựa chọn.

Như vậy history không bị mất; chỉ **không còn ép tất cả history vào một vector**.

---

# 29. Local survivor search gần như đã có sẵn

Hiện:

```python
_search_visual_model(
    model,
    queries,
    k,
    categories
)
```

chỉ cần mở rộng:

```python
_search_visual_model(
    ...,
    video_id=None
)
```

rồi pass xuống:

```python
milvus.search_image(..., video_id=video_id)
milvus.search_qwen_image(..., video_id=video_id)
```

vì adapter Milvus hiện đã implement filter này sẵn.

Đây là lý do mình đánh giá **survivor rescue có ROI cực cao**: không phải tạo index/model mới.

---

# 30. Phiên bản mình thực sự dám ship cho final

Nếu mục tiêu là **tối đa reliability chứ không phải research đẹp**, mình sẽ không implement tất cả một lúc.

```text
PHM Final-safe v1

New hint
   │
   ├─ delta query ──────────────── weight 1.0
   │
   └─ cumulative guard ────────── low weight
                 │
                 ▼
       existing PE/Qwen pipeline
                 │
                 ▼
          video ranking
                 │
          ┌──────┴──────┐
          │             │
     global new     survivor rescue
     candidates      top 8–12 video
          │             │
          └──────┬──────┘
                 ▼
        soft-product memory
                 │
                 ▼
           session ranking
                 │
          trajectory record
                 │
                 ▼
       confidence features
                 │
                 ▼
      STRONG / HOLD / UNCERTAIN
```

Cụ thể, vòng đầu mình chỉ ship:

```text
delta extraction
cumulative low-weight guard
per-hint video memory
soft geometric conjunction
survivor rescue
trajectory logging/display
```

Sau khi replay historical rounds chứng minh ranking tốt hơn current concatenation, mới bật:

```text
logistic confidence
risk-calibrated STRONG threshold
```

Sau nữa mới cân nhắc:

```text
learned hint reliability
optimal stopping
```

---

# Kết luận quan trọng nhất

Với pipeline của repo hiện tại, **Progressive Hint Memory không nên là một retriever mới**.

Nó nên là tầng nằm **trên** PE/Qwen/RRF:

```text
existing retrieval
       ↓
per-turn evidence
       ↓
persistent video belief
       ↓
trajectory reflection
       ↓
confidence
```

Và có ba nguyên tắc mình sẽ giữ rất cứng:

> **Không concatenate history thành query chính.**
> Search hint mới riêng, dùng selective context và cumulative chỉ như những query view bổ sung.

> **Không hard-AND hint coverage.**
> Dùng soft conjunction để một retrieval miss không giết video đúng.

> **Không dùng session score như confidence.**
> Confidence là một model/calibration layer riêng, được đánh giá bằng held-out sequential replay và risk–coverage.

Trong các thay đổi mình đã xem của AIC2026, mình đánh giá **`soft progressive memory + survivor rescue` là phần đáng implement trước**, thậm chí trước confidence model. Nó có khả năng trực tiếp đưa correct video từ kiểu `#8 → #4 → #1` sớm hơn, trong khi confidence chỉ giúp hệ thống biết **khi nào nên tin** sự hội tụ đó.

Nếu triển khai theo hướng này, nó cũng tận dụng đúng những gì repo đã làm tốt: PE/Qwen rank fusion, Qwen reranker, video grouping, temporal ambiguity và đặc biệt là video-restricted Milvus search đã tồn tại cho TRAKE. ([arXiv][1])

[1]: https://arxiv.org/abs/2505.15128
[2]: https://www.sciencedirect.com/science/article/pii/S2543925122001231?utm_source=chatgpt.com
[3]: https://arxiv.org/abs/2305.10923?utm_source=chatgpt.com
[4]: https://www.researchgate.net/publication/370870050_Query_Performance_Prediction_From_Ad-hoc_to_Conversational_Search?utm_source=chatgpt.com
[5]: https://blog.mobile.codalism.com/research/papers/wmz10_tois.pdf?utm_source=chatgpt.com
[6]: https://scikit-learn.org/stable/modules/calibration.html?utm_source=chatgpt.com
> From: https://chatgpt.com/c/6aafb101-0d68-83ec-8f7d-c3c062fd3ba7

# you asked

message time: 2026-09-20 17:10:44

Đọc cái này và xem idea của nó là gì, có khả năng accept cho paper vbs sắp tới không, và tích hợp với repo Aic 2026 hiện tại của tui như thế nào

---

# chatgpt response

Tui sẽ tách làm 3 phần: (1) bóc đúng contribution/idea của paper này, (2) đối chiếu với CFP VBS sắp tới để đánh giá mức độ phù hợp và rủi ro reject, (3) map nó vào kiến trúc repo AIC 2026 hiện tại của bạn thành các module/cụm thay đổi cụ thể.

Tui đã kiểm tra luôn bản `main` hiện tại của repo private `vonhatbao2205/AIC2026`: commit mới nhất hôm nay là `24a3260` (“integrate TARA clip retrieval”). Điểm quan trọng là repo bây giờ đã có PE + Qwen image retrieval, OCR/ASR/audio, TARA, RRF, reranker, TRAKE và DRES, nên paper này **không cần thay retrieval stack**; nó hợp nhất khi được thêm như một lớp “inspection” hậu truy hồi.

Tui đã đọc kỹ paper 7 trang, đối chiếu CFP chính thức VBS 2027 và kiểm tra luôn `main` hiện tại của repo private `vonhatbao2205/AIC2026` ở commit mới nhất `24a3260` ngày 20/09/2026 (`feat: integrate TARA clip retrieval`).

Kết luận quan trọng nhất: **idea này hợp với VBS, nhưng bản paper hiện tại chưa ở trạng thái nên submit nguyên xi**. Vấn đề không phải idea dở, mà paper hiện đang chứng minh một **controller độc lập**, trong khi VBS 2027 yêu cầu một **video search tool thực sự, có mô tả hệ thống, screenshot và interactive workflow**. CFP hiện có deadline được gia hạn đến **22/09/2026**. ([Video Browser Showdown](https://videobrowsershowdown.org/call-for-papers/?utm_source=chatgpt.com))

## 1. Idea thực sự của paper là gì?

Paper không đề xuất một retrieval model mới. Nó giải quyết một vấn đề **systems/interaction architecture**.

Giả sử search hiện tại của bạn trả về:

```text
Query Q1
  ↓
PE + Qwen + OCR + ASR + Audio + TARA
  ↓
RRF / rerank
  ↓
100 candidate frames
```

Sau đó người dùng muốn chạy một bước đắt tiền hơn trên vài candidate, chẳng hạn:

- VLM kiểm tra kỹ frame;
- phân tích crop;
- VQA;
- temporal inspection;
- object reasoning;
- caption chi tiết;
- visual verification.

Vấn đề là bước này có thể mất vài giây. Trong lúc nó đang chạy, operator đã đổi:

```text
Q1 → Q2
```

Nhưng worker của Q1 sau đó mới trả kết quả.

Nếu hệ thống chỉ dùng:

```text
frame_id = XXX
```

thì frame đó có thể cũng xuất hiện trong Q2 và output cũ có nguy cơ bị gắn nhầm vào trạng thái hiện tại.

Paper giải quyết bằng:

```text
generation g
+
budget B
+
pending work set Pg
+
versioned cache
```

Một result chỉ được nhận khi:

$$
g' = g
\land
state = ACTIVE
\land
k \in P_g
$$

Tức là result phải:

1. thuộc đúng query generation hiện tại;
2. request vẫn đang active;
3. work item vẫn thực sự đang pending.

Đồng thời baseline ranking **không bao giờ bị optional analysis sửa hoặc phá**. Auxiliary evidence được giữ ở một store riêng. Đây chính là contribution trung tâm được paper mô tả.

---

Paper còn thêm một ý khá hay về resource control.

Mỗi generation có budget:

$$
|A_g|\le B
$$

và work identity không đơn giản chỉ là frame:

```text
media_revision
frame_id
crop_coordinates
encoder_revision
preprocessing_revision
```

Cache có capacity riêng `K`, sử dụng LRU. Cache hit vẫn chiếm một slot trong `B`, tức `B` đo **scope mà user được inspect**, không phải GPU compute thực sự.

Điểm này khá sạch về semantics.

### Nói ngắn gọn

Paper này đang nói:

> Retrieval và expensive inspection là hai lifecycle khác nhau. Retrieval tạo một immutable candidate snapshot; inspection được cấp budget, chạy async, có generation ownership, cache và failure isolation.

Đây là **systems contribution**, không phải retrieval-algorithm contribution.

---

# 2. Contribution có đủ mới không?

Đây là điểm cần nhìn rất rõ.

Ngay chính paper cũng thừa nhận generation IDs, resource budgets và LRU caching đều là các engineering mechanisms đã biết; contribution nằm ở **cách tổ hợp chúng thành một explicit integration contract cho interactive video search**.

Do đó:

**Nếu submit regular research paper:** novelty sẽ khá yếu.

**Nếu submit VBS extended demo/system paper:** kiểu contribution này lại hợp hơn nhiều.

VBS 2027 không yêu cầu mọi paper phải phát minh retrieval algorithm mới. CFP nói paper phải là extended demo paper mô tả video search tool và cách nó hỗ trợ interactive search. ([Video Browser Showdown](https://videobrowsershowdown.org/call-for-papers/?utm_source=chatgpt.com))

Các accepted VBS paper trước cũng thường là kiểu system improvement. Ví dụ Exquisitor VBS 2026 thêm sequence-chain temporal query và in-video search vào một interactive system đã tồn tại. ([ITU Researcher Portal](https://researcher.itu.dk/en/publications/exquisitor-atthevideo-browser-showdown-2026-temporal-queries-revi/?utm_source=chatgpt.com)) Fusionista2.0 thì tập trung cả UI/UX, search optimization và reranking chứ không phải một standalone SOTA retrieval model. ([ResearchGate](https://www.researchgate.net/publication/400556804_Fusionista20_Efficiency_Retrieval_System_for_Large-Scale_Datasets?utm_source=chatgpt.com))

Vì vậy **idea fit track**.

---

# 3. Nhưng bản hiện tại có một vấn đề rất lớn

Đây là phần tui lo nhất nếu bạn submit đúng PDF này.

Paper tự nói rằng evaluation chỉ gồm:

> seven deterministic traces

với synthetic payload, không có video corpus hay relevance labels.

Nặng hơn nữa, Section 5.2 tự ghi rằng:

- chưa đo retrieval accuracy;
- chưa đo model latency;
- chưa test concurrent service behavior;
- controller hiện độc lập khỏi inherited application;
- visual workers chưa tích hợp;
- frontend chưa tích hợp;
- DRES interoperability chưa tích hợp;
- không có dataset evaluation;
- complete VBS validation vẫn cần working interface và submission path.

Trong khi CFP VBS 2027 yêu cầu paper:

> detailed description of the video search tool **including a screenshot** and how it supports interactive video search. ([Video Browser Showdown](https://videobrowsershowdown.org/call-for-papers/?utm_source=chatgpt.com))

PDF hiện tại chỉ có architecture figure, không phải screenshot của working retrieval system.

Đây là một mismatch trực tiếp với CFP.

---

# 4. Khả năng accept hiện tại

Tui sẽ không bịa ra kiểu “62.7%” vì VBS không công bố acceptance-rate đủ để suy ra con số đó.

Nhưng nếu review theo trạng thái:

| Phiên bản                                                      | Đánh giá                                      |
| ---------------------------------------------------------------- | ------------------------------------------------ |
| **PDF hiện tại submit nguyên trạng**                   | **rủi ro reject cao**                     |
| Controller + AIC system nhưng chỉ nói tích hợp, không demo | Borderline                                       |
| Controller chạy thật trong AIC2026 + screenshot + real worker  | Khá hợp VBS                                    |
| Trên + end-to-end experiment/latency/failure trace              | **Paper trở nên thuyết phục rõ rệt** |

Điểm yếu hiện tại không phải:

> “idea này quá đơn giản.”

Mà là:

> **“You submitted a coordination component to a track asking for an interactive video retrieval system.”**

Đó là hai chuyện khác nhau.

Tui **không khuyên submit nguyên PDF hiện tại**.

---

# 5. Tin tốt: repo AIC2026 hiện tại gần như chính là thứ paper đang thiếu

Current repo của bạn giờ đã mạnh hơn nhiều so với baseline được paper mô tả.

Commit `24a3260` hiện có ít nhất:

```text
PE-Core-G14
Qwen3-VL-Embedding-8B
TARA video-clip retrieval
OCR
speech
audio / GLAP
RRF
Qwen visual reranker
TRAKE
NVILA QA
query parsing
group-by-video
media playback
DRES submission
React UI
```

Đặc biệt commit mới hôm nay đã có:

```text
backend/app/adapters/tara_encoder.py
backend/app/tara_fusion.py
VideoRetrieval/TARA_INTEGRATION.md
backend/tests/test_tara.py
```

và `SearchService` đang có:

```python
PE
Qwen
TARA
OCR
speech
audio
      ↓
fusion / rerank / grouping
```

Vậy không cần biến controller thành retrieval engine.

Nên đặt nó **sau retrieval**.

---

# 6. Architecture tui đề xuất cho AIC2026

Hiện tại:

```text
                           ┌── PE
                           ├── Qwen
Query → Parser / Translate ├── OCR
                           ├── ASR
                           ├── Audio
                           └── TARA
                                 │
                                 ↓
                          RRF / rerank
                                 │
                                 ↓
                       baseline candidates
                                 │
                                 ↓
                                UI
```

Sau khi thêm paper:

```text
                           RETRIEVAL BASELINE
            ┌─────────────────────────────────────┐
            │ PE / Qwen / OCR / ASR / Audio/TARA │
            │                 ↓                   │
Query ─────→│             RRF/rerank              │
            │                 ↓                   │
            │       immutable baseline L_g        │
            └─────────────────┬───────────────────┘
                              │
                         shown immediately
                              │
                              ▼
                    ┌──────────────────┐
                    │ Search Result UI │
                    └────────┬─────────┘
                             │
                     user clicks Inspect
                             │
                             ▼
                   ┌────────────────────┐
                   │ Inspection         │
                   │ Controller         │
                   │                    │
                   │ session_id         │
                   │ generation g       │
                   │ budget B           │
                   │ pending P_g        │
                   │ cache K            │
                   └─────────┬──────────┘
                             │
              ┌──────────────┴──────────────┐
              ↓                             ↓
       NVILA / VLM                  future crop/clip
       detail worker                     worker
              │                             │
              └──────────────┬──────────────┘
                             ↓
                     completion guard
                             ↓
                    auxiliary evidence
                             │
                             ▼
               ┌─────────────────────────┐
               │ Evidence / Detail Panel │
               └─────────────────────────┘

                 baseline ranking unchanged
```

Đây mới thực sự khớp Fig. 1 của paper.

---

# 7. Quan trọng: đừng nhét controller vào `SearchService`

Tui sẽ **không sửa logic ranking của**:

```text
backend/app/services/search_service.py
```

để inspection result quay ngược lại reorder baseline.

Paper định nghĩa rất rõ:

```text
baseline candidate snapshot
≠
inspection evidence
```

Inspection branch không được erase/reorder baseline.

Trong AIC2026, hãy định nghĩa:

```text
baseline L_g
=
output cuối cùng của current search pipeline
sau PE/Qwen/TARA/RRF/reranking nếu operator đã bật nó.
```

Tức reranker hiện tại vẫn có thể là **một phần baseline retrieval**.

Sau khi `L_g` được tạo:

```text
L_g = immutable
```

Inspection mới bắt đầu.

---

# 8. Những file nên thêm

Tui sẽ tạo một package độc lập:

```text
backend/app/inspection/
├── __init__.py
├── controller.py
├── models.py
├── service.py
└── cache.py
```

### `controller.py`

Chứa đúng state machine của paper.

Ví dụ conceptual API:

```python
begin_generation(...)
admit(...)
complete(...)
fail(...)
close(...)
cancel(...)
snapshot(...)
```

Không chứa PE, Qwen, NVILA hay TARA.

Điều này cực kỳ quan trọng cho paper vì bạn vẫn có thể nói:

> controller is model-independent.

---

### `models.py`

Ví dụ:

```python
WorkKey:
    media_revision
    submit_keyframe_id
    crop
    worker_revision
    preprocessing_revision
```

và:

```python
InspectionEvidence:
    work_key
    generation
    worker
    payload
    latency_ms
```

---

### `service.py`

Đây mới là layer gắn controller với AIC.

```text
controller
   ↓
worker adapter
   ↓
existing AIC infrastructure
```

---

# 9. Worker đầu tiên nên là gì?

Để kịp paper, **không nên train thêm model mới**.

Repo hiện đã có NVILA-8B visual QA/copilot path.

Tui sẽ reuse nó làm:

```text
DetailInspectionWorker
```

Ví dụ operator tìm:

> “a man in a red shirt repairing a bicycle”

Baseline trả:

```text
frame A
frame B
frame C
...
```

User chọn A rồi Inspect.

NVILA worker trả evidence kiểu:

```text
Candidate: A

Visible subjects:
- one person
- red upper clothing

Objects:
- bicycle
- wheel/tool-like object

Query support:
- person + bicycle strongly supported
- repair action uncertain

Overall evidence:
PARTIAL_MATCH
```

Quan trọng:

**đừng cho kết quả đó tự động đẩy frame A lên rank 1**.

Nó chỉ xuất hiện ở:

```text
Inspection Evidence
```

để operator quyết định.

Đó chính xác là claim của paper:

> ownership ≠ relevance.

Controller xác định result thuộc request nào; việc operator sử dụng evidence thế nào vẫn thuộc presentation layer.

---

# 10. TARA mới của bạn cũng làm paper này mạnh hơn

Commit hôm nay thêm TARA rất hữu ích cho architecture này.

Ví dụ baseline:

```text
TARA
 ↓
candidate clip:
video=V028
start=213s
end=237s
```

Sau đó Detail Inspection có thể inspect chính clip đó:

```text
candidate clip
    ↓
NVILA / video worker
    ↓
detailed temporal evidence
```

Như vậy paper không còn chỉ nói:

> frame inspection.

Mà thành:

> **budgeted candidate-level frame/clip inspection on top of multimodal retrieval.**

Cái này hợp VBS hơn đáng kể.

---

# 11. Frontend nên thay đổi thế nào?

Repo hiện tại có:

```text
frontend/src/FullConsole.tsx
frontend/src/components/DetailPanel.tsx
```

Tui sẽ thêm nút:

```text
🔍 Inspect
```

ở DetailPanel.

UI:

```text
┌───────────────────────────────────┐
│ Candidate                         │
│ [frame]                           │
│                                   │
│ video: V028                       │
│ time: 03:42                       │
│ PE + Qwen + TARA                  │
│                                   │
│ [Inspect detail]                  │
└───────────────────────────────────┘
```

Sau khi click:

```text
Inspection
──────────────
Budget: 3 / 5

Status:
✓ NVILA analysis

Evidence:
person: yes
red shirt: yes
bicycle: yes
repairing: uncertain
```

Baseline list bên trái **không đổi**.

Đây sẽ là screenshot cực đẹp cho paper vì người review nhìn một phát là hiểu contribution.

---

# 12. Generation phải gắn với query lifecycle của frontend

Ví dụ:

```text
generation 41
query = "man fixing bicycle"
```

Operator request 5 inspections.

Trong lúc worker chạy:

```text
generation 42
query = "woman driving a red car"
```

Backend phải ngay lập tức:

```text
generation 41 → CANCELLED
generation 42 → ACTIVE
```

Nếu worker trả:

```json
{
  "generation": 41,
  "candidate": "V001/frame_120"
}
```

thì:

```text
DROP
```

dù frame đó tình cờ cũng tồn tại trong result của generation 42.

Đây chính là scenario mạnh nhất trong paper.

---

# 13. Nhưng repo của bạn có multi-user → paper hiện tại phải mở rộng một chút

Đây là vấn đề production mà paper hiện tại chưa giải quyết.

Paper giả định:

> serialized controller calls.

Nhưng AIC/VBS của bạn có thể có nhiều operator dùng chung backend.

Không nên chỉ dùng:

```text
generation = 1,2,3,...
```

toàn server.

Nên dùng:

```text
operator_session_id
+
generation
```

Ví dụ:

```text
Bảo / session-A / generation-14
Quân / session-B / generation-7
Linh / session-C / generation-21
```

Backend:

```python
controllers: dict[SessionId, InspectionController]
```

và mỗi session có:

```python
asyncio.Lock()
```

Điều này vừa làm implementation đúng hơn, vừa giải quyết một limitation mà paper hiện đang tự thừa nhận.

Đây có thể trở thành **một revision có giá trị cho paper**.

---

# 14. API tối thiểu

Không cần thiết kế phức tạp.

Tui sẽ làm:

```http
POST /api/inspection/start
```

Input:

```json
{
  "session_id": "...",
  "candidate_ids": [...],
  "budget": 5
}
```

Output:

```json
{
  "generation": 42
}
```

---

```http
POST /api/inspection/{generation}/admit
```

```json
{
  "submit_keyframe_id": "...",
  "worker": "nvila"
}
```

---

```http
GET /api/inspection/{generation}
```

return:

```json
{
  "state": "ACTIVE",
  "budget": 5,
  "admitted": 3,
  "pending": 1,
  "completed": 2,
  "evidence": [...]
}
```

Và:

```http
POST /close
POST /cancel
```

Đủ cho paper.

---

# 15. Một thứ đặc biệt không nên làm

Đừng implement:

```text
inspection score
      ↓
automatically blend vào RRF
      ↓
reorder result
```

Nếu làm vậy bạn phá central claim của paper.

Paper nói rõ auxiliary completion không có implicit authority lên current search state.

Nếu sau này muốn operator bấm:

```text
Apply evidence to ranking
```

thì đó nên là **explicit action tạo một generation/search mới**.

Ví dụ:

```text
generation 42
       ↓
operator chooses refine
       ↓
generation 43
```

Rất sạch.

---

# 16. Evaluation nên sửa thế nào trước VBS submission?

Hiện tại Table 1 chỉ có 7 trace:

```text
budget + duplicate
query replacement
cancellation
worker failure
versioned reuse
request closure
zero budget
```

và cả 7 pass.

Giữ nguyên phần đó.

Nhưng thêm một subsection:

## End-to-End Application Validation

Chạy trên AIC2026 thật.

Ít nhất đo:

### A. Real stale-completion test

```text
Q1
↓
launch 5 NVILA jobs
↓
switch Q2 before completion
↓
wait
```

Report:

```text
old results accepted = 0
baseline Q2 changed = 0
```

---

### B. Failure isolation

Tắt worker giữa chừng:

```text
baseline retained = true
completed evidence retained = true
failed evidence = absent
```

---

### C. Budget correctness

Ví dụ:

```text
B = 5
20 inspect requests
```

expect:

```text
unique admitted <= 5
```

---

### D. Cache

Record:

```text
cold worker latency
cache-hit latency
```

và cache revision invalidation.

---

### E. Overhead

Measure controller overhead:

```text
admit()
complete()
snapshot()
```

Tui kỳ vọng rất nhỏ nhưng **đo rồi mới ghi**.

---

# 17. Nếu còn thời gian, thêm một usefulness experiment

Không cần đòi mAP/SOTA retrieval.

Một experiment VBS-style phù hợp hơn:

```text
20 archived search tasks

Baseline only
vs
Baseline + Detail Inspection
```

Measure:

```text
time-to-confirm candidate
number of inspected candidates
successful task completion
```

Ngay cả một study nhỏ cũng mạnh hơn rất nhiều so với:

> seven synthetic traces only.

Các VBS system paper trước thực sự thường trình bày hiệu quả của interactive feature chứ không chỉ correctness của một internal class; Fusionista2.0 chẳng hạn báo cáo cả system efficiency/usability improvements. ([ResearchGate](https://www.researchgate.net/publication/400556804_Fusionista20_Efficiency_Retrieval_System_for_Large-Scale_Datasets?utm_source=chatgpt.com))

---

# 18. Paper nên đổi structure

Bản hiện tại:

```text
Introduction
Architecture
Budget/Lifecycle Contract
Architectural Validation
Implications for VBS
Conclusion
```

Tui sẽ đổi thành:

```text
1 Introduction

2 AIC2026 Interactive Video Retrieval System
  2.1 Multimodal retrieval
  2.2 TARA temporal retrieval
  2.3 User interface and DRES

3 Budgeted Detail Inspection
  3.1 Baseline / auxiliary separation
  3.2 Admission budget
  3.3 Generation ownership
  3.4 Versioned cache

4 Integration into the Interactive System
  4.1 Inspection worker
  4.2 Frontend workflow
  4.3 Multi-user lifecycle
  4.4 Failure handling

5 Evaluation
  5.1 Seven deterministic traces
  5.2 End-to-end application validation
  5.3 Latency/cache/resource measurements

6 VBS Workflow

7 Conclusion
```

Cái khác biệt lớn là reviewer không còn nhìn thấy:

> standalone controller looking for somewhere to be used.

Mà thấy:

> **working VBS system with a new reliability/resource-management feature.**

---

# 19. Figure 1 cũng nên đổi

Architecture figure hiện tại ghi dashed box:

> interfaces awaiting application integration.

**Bản submit cuối không nên còn câu này.**

Nó gần như tự nói với reviewer:

> system chưa hoàn thiện.

Thay bằng actual integrated architecture:

```text
AIC Retrieval
 PE Qwen OCR ASR GLAP TARA
           ↓
          RRF
           ↓
 Candidate Snapshot
           ↓
 ┌───────────────────────┐
 │ Inspection Controller │
 └───────────┬───────────┘
             ↓
       NVILA worker
             ↓
       Evidence Panel
             ↓
          Operator
             ↓
           DRES
```

Và thêm **actual UI screenshot** riêng.

---

# 20. Điểm paper này có thể trở nên khá hay

Sau integration, story sẽ rất coherent:

### Problem

Modern VBS system ngày càng có nhiều expensive second-stage models:

```text
VLM
reranker
video model
VQA
crop analysis
temporal verification
```

nhưng interactive search có query churn rất nhanh.

### Existing approach

Thường implementation kiểu:

```python
async worker → callback → update UI
```

không có explicit ownership/resource semantics.

### Your contribution

Một model-independent lifecycle:

```text
immutable retrieval snapshot
+
budgeted optional analysis
+
generation ownership
+
versioned cache
+
failure isolation
```

### Demonstration

Chạy thật trên multimodal AIC/VBS system với:

```text
PE
Qwen
TARA
OCR
ASR
audio
NVILA
DRES
```

Đây là story mạnh hơn bản paper hiện tại **rất nhiều**.

---

# 21. Đánh giá cuối cùng của tui

Nếu coi đây là **research algorithm paper**, contribution chưa đủ mạnh: generation IDs, LRU và budgets đều không mới, và paper cũng thừa nhận điều đó.

Nhưng nếu coi đây đúng là **VBS system/demo paper**, thì hướng này có lý.

Vấn đề là current PDF đang đứng giữa hai thế giới:

```text
quá engineering để thành algorithm paper
nhưng chưa integrated đủ để thành VBS system paper.
```

Đó là lý do tui đánh giá bản hiện tại có rủi ro reject cao.

Trong khi repo AIC2026 hiện tại lại giải quyết gần như toàn bộ vấn đề đó. Bạn đã có retrieval, UI, video playback, TARA và DRES. Phần còn thiếu chủ yếu là:

```text
InspectionController
        +
API glue
        +
NVILA worker adapter
        +
DetailPanel UI
        +
end-to-end experiment
        +
screenshot
```

Chứ **không phải xây một system mới từ đầu**.

### Nếu tui đang làm paper này với repo của bạn

Tui sẽ freeze retrieval stack hiện tại tại:

```text
24a3260
```

và **không đụng thêm PE/Qwen/TARA/RRF**.

Trong thời gian còn lại chỉ tập trung:

```text
1. Port controller vào backend
2. Gắn NVILA làm real inspection worker
3. Thêm generation/session lifecycle
4. Add Inspect UI + budget counter
5. Run 7 existing traces
6. Run 3–4 end-to-end real traces
7. Chụp screenshot
8. Rewrite Sections 2, 4, 5
9. Bỏ mọi câu "awaiting integration"
```

Nếu hoàn thành được đúng scope này, **tui cho rằng việc submit VBS 2027 trở nên hợp lý và paper có một câu chuyện hệ thống đủ rõ để peer review nghiêm túc**, thay vì current version vốn rất dễ bị reviewer bắt ngay lỗi “controller only, no actual VBS application”.

Nếu bạn muốn, bước tiếp theo tui có thể **đọc trực tiếp từng file của commit `24a3260` rồi thiết kế patch integration cụ thể tới mức file nào thêm class gì, endpoint nào, React state nào và test nào cần viết**, để bạn/Codex triển khai thẳng trong repo.

---

# you asked

message time: 2026-09-20 17:22:17

có nhất thiết phải xử lý 1 phần data v3c hay không, hay paper có thể dùng 1 data khác và chỉ cần mô tả hệ thống là được, có trường hợp nào giống vậy không

---

# chatgpt response

Có thể không nhất thiết phải xử lý một phần V3C, nhưng điểm mấu chốt là paper phải chứng minh được **tool chạy thật cho interactive video search**, chứ không chỉ mô tả architecture trên giấy. Tui sẽ kiểm tra CFP VBS hiện tại và vài paper VBS trước để xem họ dùng data gì và mức evaluation tối thiểu ra sao.

Không, **không có yêu cầu trong CFP VBS 2027 rằng paper phải xử lý V3C hoặc phải có experiment trên V3C**. Yêu cầu của paper là một *extended demo paper* mô tả chi tiết video search tool, có screenshot, và giải thích cách tool hỗ trợ interactive video search. ([Video Browser Showdown](https://videobrowsershowdown.org/call-for-papers/?utm_source=chatgpt.com))

Có một điểm cần tách rất rõ:

- **Để paper được review/accept:** không bắt buộc evaluation trên V3C.
- **Để thực sự thi VBS 2027:** cuối cùng system phải chạy được trên collection chính thức. Năm 2027 họ dùng toàn bộ V3C1+V3C2+V3C3 cùng Marine và GynSurg, và CFP nói rõ **đa số task sẽ lấy từ V3C**. ([Video Browser Showdown](https://videobrowsershowdown.org/call-for-papers/?utm_source=chatgpt.com))

## Có tiền lệ dùng data khác không?

Có, và khá rõ.

**ViFi tại VBS 2025** là paper VBS đã được accept/publish ở MMM 2025. Trong abstract, họ đánh giá SigLIP so với CLIP bằng **MSCOCO**, tức một benchmark bên ngoài V3C. ([ResearchGate](https://www.researchgate.net/publication/387636161_ViFi_A_Video_Finding_System_at_Video_Browser_Showdown_2025?utm_source=chatgpt.com)) Điều này không có nghĩa ViFi không chuẩn bị V3C cho cuộc thi; nó cho thấy **experiment trong VBS paper không bắt buộc phải dùng V3C**.

**VideoEase at VBS 2025** cũng là paper đã qua peer review. Họ không cần chạy toàn bộ dataset mới để làm evaluation; paper dùng **các query từ VBS 2024** để đánh giá việc fusion nhiều embedding model và đo answer rank. ([DORAS](https://doras.dcu.ie/30858/1/MMM_VBS25_Linh.pdf?utm_source=chatgpt.com))

Thậm chí có nhiều VBS paper thiên rất mạnh về **system description**. Chẳng hạn **diveXplore at VBS 2024** chủ yếu trình bày kiến trúc mới, OpenCLIP retrieval, query server, UI và exploration view. Abstract của paper tập trung vào những thay đổi hệ thống đó chứ không xây dựng câu chuyện như một benchmark retrieval paper thông thường. ([arXiv](https://arxiv.org/abs/2508.20560?utm_source=chatgpt.com)) **VERGE in VBS 2024** cũng được mô tả như một interactive system với retrieval methods, fusion, reranking và web UI. ([Zenodo](https://zenodo.org/records/10652893?utm_source=chatgpt.com))

Nói cách khác, VBS paper là **demo/system paper**, không phải bắt buộc phải giống regular research paper kiểu:

```text
Dataset → train → test → mAP → SOTA
```

---

## Đối với paper của bạn, tui nghĩ không nên lao vào xử lý V3C lúc này

Deadline hiện là **22/09/2026**, tức còn rất ít thời gian. ([Video Browser Showdown](https://videobrowsershowdown.org/call-for-papers/important-dates/?utm_source=chatgpt.com))

Với contribution của paper:

```text
Budgeted Detail Inspection

baseline retrieval
       ↓
immutable candidate snapshot
       ↓
budgeted asynchronous inspection
       ↓
generation ownership
       ↓
cache / failure isolation
       ↓
inspection evidence
```

thì bản chất contribution **không phụ thuộc dataset**.

Bạn không cần chứng minh:

> “retrieval của chúng tôi tốt trên V3C.”

Bạn cần chứng minh:

> “khi một interactive video retrieval system gọi expensive auxiliary analysis, controller đảm bảo bounded work, correct request ownership, stale-result rejection và baseline preservation.”

Đó là một **systems property**.

Vì vậy experiment hợp lý hơn là:

```text
AIC2026 current dataset
        ↓
real retrieval
        ↓
top candidates
        ↓
real detail inspection
        ↓
switch query while jobs running
        ↓
verify stale completions rejected
```

Dataset chỉ đóng vai trò tạo ra video/candidate thật.

---

# Tui sẽ chọn phương án này cho paper

### 1. Dùng chính data hiện AIC2026 đang chạy

Tức **InfoShot++ / AIC data hiện tại**, bởi vì toàn bộ stack đã hoạt động:

```text
PE
Qwen
TARA
OCR
ASR
Audio
RRF
reranker
UI
DRES-like submission
```

Sau đó integrate controller.

Paper ghi rõ:

> We integrate the proposed controller into an existing multimodal interactive video retrieval system and evaluate the lifecycle properties on a working video collection.

Không cần giả vờ đó là V3C.

---

### 2. Thay 7 synthetic traces bằng hoặc bổ sung real application traces

Ví dụ lấy khoảng:

```text
50–200 videos
```

từ data hiện tại cũng đủ cho **integration validation**.

Bạn không cần 10,000 video vì property:

```text
old generation result must be rejected
```

không thay đổi giữa:

```text
100 videos
```

và

```text
28,450 videos.
```

Scale của collection không phải independent variable mà paper đang nghiên cứu.

Có thể làm:

| Experiment            |  Data cần |
| --------------------- | ---------: |
| stale completion      | vài video |
| budget enforcement    | vài video |
| cache reuse           | vài video |
| worker failure        | vài video |
| query replacement     | vài video |
| baseline preservation | vài video |
| latency overhead      | vài video |

Toàn bộ contribution chính có thể kiểm nghiệm mà **không cần V3C3**.

---

# Nếu muốn tăng credibility thêm, có thể dùng một public dataset nhỏ

Không bắt buộc, nhưng nếu reviewer hỏi:

> “Why are you evaluating only on an AIC-specific collection?”

thì có thể thêm một dataset public nhỏ.

Ví dụ không cần đánh giá retrieval accuracy; chỉ dùng nó để chứng minh:

```text
controller does not depend on collection
```

Paper có thể có:

```text
Collection A: AIC/InfoShot++
Collection B: public video collection
```

và report:

```text
Lifecycle assertions       A       B
--------------------------------------
Budget                     ✓       ✓
Stale rejection            ✓       ✓
Failure isolation           ✓       ✓
Cache revision              ✓       ✓
Baseline preservation       ✓       ✓
```

Nhưng với deadline hiện tại, tui còn thấy **một dataset thực chạy end-to-end** tốt hơn là hai dataset nhưng integration nửa vời.

---

# Vậy V3C xuất hiện trong paper như thế nào?

Tui vẫn sẽ dành một subsection ngắn:

### VBS Deployment

Ví dụ nội dung conceptually:

> The controller is collection-independent and operates on candidate identifiers supplied by the retrieval layer. Consequently, deployment on V3C does not require changes to the lifecycle contract. The VBS deployment requires replacing the collection-specific retrieval/media adapters and indexing the official V3C collection, while generation ownership, admission budgeting, caching, and evidence handling remain unchanged.

Điều này hợp với paper hiện tại vì ngay thiết kế ban đầu của bạn cũng đã tách:

```text
retrieval adapter
worker adapter
client
controller
```

ra khỏi nhau.

Nhưng có một câu **không nên giữ**:

> “visual workers, frontend wiring, and DRES interoperability remain integration work.”

vì nó làm paper trông như chưa có application. Paper hiện tại đang tự thừa nhận đúng điều này.

Hãy integrate **AIC application thật** trước. V3C có thể là deployment dataset về sau.

---

# Nhưng đến January 2027 thì khác

Nếu paper được accept, lúc thi thật vào **5/01/2027**, bạn không thể chỉ mang AIC dataset đi thi.

VBS 2027 dùng:

```text
V3C1
+
V3C2
+
V3C3
+
Marine
+
GynSurg
```

và organizers ghi rõ phần lớn task đến từ V3C. ([Video Browser Showdown](https://videobrowsershowdown.org/call-for-papers/?utm_source=chatgpt.com))

Cho nên timeline hợp lý là:

```text
NOW — paper deadline
│
├── AIC2026 data
├── integrate controller
├── real worker
├── UI screenshot
├── end-to-end validation
└── submit paper
        │
        ▼
Oct 9 — notification
        │
        ▼
camera ready / competition preparation
        │
├── index V3C1
├── V3C2
├── V3C3
├── integrate official metadata
├── DRES
└── performance tuning
        │
        ▼
Jan 5 — VBS 2027
```

Tức là **không cần giải bài toán 8.7 TB V3C ngay trong hai ngày trước deadline**. Official site hiện liệt kê V3C1+2+3 tổng cộng khoảng 28,450 video, 3,801 giờ và 8.7 TB, nên đây rõ ràng là một deployment/preprocessing task lớn. ([Video Browser Showdown](https://videobrowsershowdown.org/about-vbs/existing-data-and-tools/?utm_source=chatgpt.com))

---

## Một nuance rất quan trọng

Tui sẽ không viết trong paper:

> “Our system has been validated for VBS 2027.”

nếu chưa chạy V3C.

Mà viết chính xác hơn:

> “We validate the integrated interactive workflow on the currently deployed collection and design the controller independently of collection-specific retrieval and media adapters.”

Sau đó:

> “Deployment to the official VBS collections changes the retrieval/media adapters, not the controller contract.”

Claim như vậy vừa đúng dữ liệu bạn có, vừa không overclaim.

---

## So với các precedent

Pattern chung có thể nhìn như sau:

| VBS paper                 | Evaluation                                                                                                                                                                                                  |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ViFi 2025**       | Có comparison trên**MSCOCO**, không chỉ V3C ([ResearchGate](https://www.researchgate.net/publication/387636161_ViFi_A_Video_Finding_System_at_Video_Browser_Showdown_2025?utm_source=chatgpt.com)) |
| **VideoEase 2025**  | Replay**VBS 2024 queries**, không cần evaluation toàn V3C3 mới ([DORAS](https://doras.dcu.ie/30858/?utm_source=chatgpt.com))                                                                       |
| **diveXplore 2024** | Nặng về architecture + UI + system integration ([arXiv](https://arxiv.org/abs/2508.20560?utm_source=chatgpt.com))                                                                                          |
| **VERGE 2024**      | Chủ yếu trình bày interactive retrieval system, fusion/reranking/UI ([Zenodo](https://zenodo.org/records/10652893?utm_source=chatgpt.com))                                                               |
| **Paper của bạn** | Có thể dùng AIC collection + controller traces + real end-to-end inspection                                                                                                                              |

Cho nên **có precedent khá rõ cho cách làm này**.

---

### Tui sẽ đổi recommendation so với câu trả lời trước một chút

**Không cần xử lý V3C ngay để cứu paper.**

Với hai ngày còn lại, giá trị cao hơn nhiều là:

```text
controller standalone
        ↓
integrate thật vào AIC2026
        ↓
real candidate
        ↓
real inspection worker
        ↓
UI screenshot
        ↓
end-to-end lifecycle experiment
```

thay vì:

```text
download/process một ít V3C
        ↓
chưa integrate controller hoàn chỉnh
```

Paper VBS cần **working interactive tool** hơn là cần cái tên `V3C` xuất hiện trong bảng experiment.

Và official CFP thực sự hỗ trợ cách hiểu này: điều kiện paper họ viết là **“detailed description of the video search tool (including a screenshot) and how it supports interactive search in video data”**, chứ không ghi “evaluation must be conducted on V3C.” ([Video Browser Showdown](https://videobrowsershowdown.org/call-for-papers/?utm_source=chatgpt.com))

Nếu xét đúng trạng thái repo + paper của bạn hiện tại, **tui sẽ dùng data AIC hiện có cho bản submission ngày 22/9 và dành V3C cho giai đoạn chuẩn bị competition/camera-ready**.

---

# you asked

message time: 2026-09-20 17:29:08

xem idea này với cái idea kia thì cái nào ổn hơn để có khả năng accept cao hơn

---

# chatgpt response

Tui sẽ so hai idea theo đúng tiêu chí reviewer VBS dễ quan tâm nhất: **độ mới, fit với VBS, mức chứng minh được bằng experiment, độ hoàn thiện hệ thống, và rủi ro bị xem là “engineering only”**. Sau đó tui sẽ chốt hướng nào đáng đặt cược hơn cho submission hiện tại.

Nếu đặt hai idea cạnh nhau với mục tiêu duy nhất là **tăng khả năng paper VBS 2027 được accept**, tui sẽ ưu tiên **Progressive Hint Memory (PHM) + Early-Submit Confidence** hơn **Budgeted Detail Inspection (BDI)**.

Nếu cả hai đều được implement và evaluate ở mức tương đương, khoảng cách theo tui là khá rõ.

| Tiêu chí                           |   Progressive Hint Memory | Budgeted Detail Inspection |
| ------------------------------------ | ------------------------: | -------------------------: |
| Fit trực tiếp với VBS 2027        |        **Rất cao** |                       Khá |
| Novelty thuật toán                 |       **Khá–cao** |                Thấp–khá |
| Dễ kể thành research story        |       **Rất tốt** |                  Khó hơn |
| Dễ có quantitative evaluation      |       **Rất tốt** |                Trung bình |
| Gắn với AIC2026 hiện tại         | **Rất tự nhiên** |                 Tự nhiên |
| Reviewer hiểu “tại sao cần nó” |  **Ngay lập tức** |          Cần giải thích |
| Nguy cơ bị gọi là engineering    |        Thấp–trung bình |              **Cao** |
| Related-work collision               |               Trung bình |         Thấp–trung bình |
| Tiềm năng cải thiện thi thật    |             **Cao** |      Chủ yếu reliability |
| Lựa chọn của tui cho VBS paper    |       **Ưu tiên** |                     Backup |

### Vì sao PHM hợp VBS hơn hẳn

VBS 2027 nói rõ KISV sẽ là thiểu số, còn **KIST, KISC, VQA và AVS sẽ nhiều hơn**. Đặc biệt textual KIS vốn có mô tả được cung cấp ngày càng chi tiết trong quá trình search. ([Video Browser Showdown](https://videobrowsershowdown.org/call-for-papers/?utm_source=chatgpt.com))

PHM đánh thẳng vào chính interaction đó:

```text
Hint 1
   ↓
retrieve + remember evidence
   ↓
Hint 2
   ↓
update belief, không search lại từ zero
   ↓
Hint 3
   ↓
ranking converges
   ↓
confidence → submit hay chờ?
```

Trong file bạn đưa, idea cuối cùng cũng đã tiến xa hơn một ý tưởng kiểu “cộng score qua hint”. Nó tách thành **multi-view retrieval + video-level evidence memory + soft conjunction + trajectory reflection + survivor rescue**, đồng thời confidence được tách khỏi ranking.

Đây là một research problem VBS rất dễ hiểu:

> Existing retrieval treats an incrementally revealed description as a growing query; we instead model textual KIS as sequential evidence accumulation over candidate videos.

Một reviewer đọc 2 câu là hiểu problem.

---

## PHM có contribution “paper-like” hơn

PHM hiện có ít nhất bốn contribution có thể formalize.

Thứ nhất là **multi-view hint processing**. Mỗi round không chỉ concatenate toàn bộ history mà tạo:

```text
Q_delta       = thông tin mới
Q_context     = thông tin mới + context tối thiểu
Q_cumulative  = toàn bộ description
```

với vai trò khác nhau.

Thứ hai là **video-level soft evidence accumulation** thay vì hard intersection. Bạn đã có formulation kiểu soft Product-of-Experts:

$$
M_t(v)=
\exp\left(
\frac{
\sum_i w_i\log[\epsilon+(1-\epsilon)e_i(v)]
}{
\sum_iw_i
}
\right)
$$

Nó có lý do rõ ràng: một hint retrieval miss không được phép giết hoàn toàn video đúng.

Thứ ba, theo tui đây là phần mạnh nhất, là **Survivor Rescue**:

```text
new hint
   ├── global retrieval → discover new videos
   │
   └── local retrieval inside strong historical candidates
                         → rescue old videos
```

Tức exploration + exploitation. Một video đúng đã rank #2 ở Hint 1 không bị mất chỉ vì frame phù hợp Hint 2 đứng #137 toàn corpus; hệ thống có thể search sâu trong video đó.

Thứ tư là **confidence/early stopping tách khỏi ranking**:

```text
ranking score
    ≠
probability correct

ranking
   ↓
trajectory features
   ↓
confidence model
   ↓
STRONG / HOLD
```

Những phần này kết hợp lại thành một method, chứ không chỉ một feature UI.

---

# Còn BDI yếu ở đâu?

BDI có một ý kiến trúc rất sạch:

```text
baseline retrieval
      │
      ├──────────────→ immutable baseline
      │
      ↓
optional inspection
      ↓
generation ownership
budget
cache
failure isolation
```

Nhưng chính paper cũng tự nói rằng generation identifiers, budgets và LRU caching đều là **established engineering mechanisms**; contribution là composition của chúng thành integration contract.

Reviewer có thể hỏi rất thẳng:

> “What is scientifically novel beyond standard request versioning, bounded queues, and caching?”

Đó là câu hỏi khó trả lời hơn nhiều.

PHM thì reviewer có thể disagree với công thức, nhưng ít nhất họ thấy ngay **retrieval problem mới mà method đang cố giải**.

---

# BDI hiện còn mắc một vấn đề lớn hơn: evaluation

BDI hiện chứng minh bằng 7 deterministic event traces, dùng synthetic payload, không video corpus và không relevance labels.

Paper còn tự ghi rằng chưa đo:

- retrieval accuracy;
- model latency;
- concurrent service behavior;
- working frontend integration;
- DRES interoperability;
- dataset evaluation.

Trong khi CFP yêu cầu một **extended demo paper mô tả video search tool, có screenshot, và cách nó hỗ trợ interactive video search**. ([Video Browser Showdown](https://videobrowsershowdown.org/call-for-papers/?utm_source=chatgpt.com))

BDI có thể sửa được vấn đề này bằng cách integrate vào AIC2026, nhưng kể cả sau khi sửa, contribution trung tâm vẫn thiên về **system correctness/reliability**.

PHM tự nhiên hơn với kiểu paper:

```text
problem
→ method
→ ablation
→ retrieval improvement
→ interactive consequence
→ integration in system
```

Đó là cấu trúc review dễ hơn.

---

# Tuy nhiên PHM có một đối thủ literature cần xử lý rất cẩn thận

Đây là rủi ro lớn nhất của PHM.

ICMR 2025 đã có **Robust Relevance Feedback for Interactive Known-Item Video Search**. Họ còn evaluate trên **17 textual-KIS queries của VBS 2022–2024**, và chính các query này được đưa ra trong ba rounds với thông tin tăng dần. ([arXiv](https://arxiv.org/abs/2505.15128?utm_source=chatgpt.com))

Nếu viết PHM kiểu:

> “We are the first to exploit multiple rounds of textual KIS.”

thì rất nguy hiểm.

Không nên claim như vậy.

Điểm khác biệt phải được đóng đinh là:

### Existing relevance-feedback work

```text
initial query
   ↓
display candidates
   ↓
USER feedback/judgment
   ↓
representation refinement
```

Paper ICMR sử dụng pairwise relative judgments, multiple sub-perceptions và predictive user model để lọc feedback không phù hợp. ([arXiv](https://arxiv.org/abs/2505.15128?utm_source=chatgpt.com))

### PHM của bạn

```text
progressively revealed textual hints
        ↓
delta/context/cumulative retrieval
        ↓
persistent VIDEO evidence
        ↓
soft multi-hint conjunction
        ↓
survivor-local rescue
        ↓
trajectory-based confidence
```

**không cần user relevance judgment để tạo state transition.**

Đó mới là differentiation.

---

# Và thực tế đã có precedent rất đẹp cho loại paper PHM

Exquisitor VBS 2026 không xây một retrieval engine hoàn toàn mới. Họ phát hiện một weakness từ VBS 2025 rồi thêm:

- sequence-chain method + RRF cho temporal queries;
- in-video search cho QA.

Paper đó được accept vào VBS 2026. ([ITU Researcher Portal](https://researcher.itu.dk/en/publications/exquisitor-atthevideo-browser-showdown-2026-temporal-queries-revi/?utm_source=chatgpt.com))

Thậm chí Exquisitor đứng thứ **3 overall tại VBS 2026**. ([Video Browser Showdown](https://videobrowsershowdown.org/hall-of-fame/?utm_source=chatgpt.com))

Đây khá giống strategy mà PHM có thể dùng:

> “Our existing system retrieves well, but progressive textual KIS is handled incorrectly as repeated/cumulative independent retrieval. We introduce stateful progressive evidence retrieval.”

Đây là kiểu incremental system contribution hoàn toàn phù hợp với track.

---

# Nếu là reviewer, tui sẽ nhìn hai submission như thế này

### BDI

> Interesting engineering design. But why is this a multimedia retrieval contribution rather than generic asynchronous system orchestration?

Đây là câu hỏi nguy hiểm.

### PHM

> Progressive textual descriptions are central to KIS. Does keeping video-level evidence across rounds improve retrieval and enable earlier reliable decisions?

Đây là câu hỏi bạn **có thể trả lời bằng experiment**.

Và đó là khác biệt rất lớn.

---

# PHM còn có lợi thế: experiment cực kỳ dễ thiết kế

Bạn có thể biến historical tasks thành:

```text
Q1
├── Hint 1
├── Hint 2
└── Hint 3
```

Với mỗi prefix đo:

$$
Hit@1_t,\quad MRR_t,\quad Recall@K_t
$$

và đặc biệt metric mà file đã đề xuất:

$$
TSC(Q)
=
\min t
$$

sao cho ground truth đạt rank #1 từ round $t$ đến hết. Đây là **Time-to-Stable-Correct**.

Baseline rất rõ:

```text
B0: current cumulative concat
B1: latest hint only
B2: RRF across hints
B3: PHM without rescue
B4: PHM + rescue
```

Rồi ablation:

```text
PHM full
- Q_delta
- cumulative guard
- soft conjunction
- survivor rescue
- temporal feature
```

Reviewer nhìn table là hiểu contribution.

BDI không có loại ablation retrieval tự nhiên như vậy.

---

# Một paper PHM mạnh có thể có result table kiểu này

Giả sử sau này số liệu thực ra như sau — **đây chỉ là cấu trúc minh họa, không phải result hiện có**:

| Method           |        H1 R@1 |        H2 R@1 |        H3 R@1 |           MRR | Stable-correct |
| ---------------- | ------------: | ------------: | ------------: | ------------: | -------------: |
| Cumulative query |           .41 |           .55 |           .66 |           .71 |           2.42 |
| Latest hint      |           .37 |           .51 |           .61 |           .67 |           2.55 |
| Hint-RRF         |           .43 |           .60 |           .70 |           .74 |           2.29 |
| PHM              | **.47** | **.68** | **.78** | **.81** | **1.91** |

Rồi:

| Ablation            | H3 R@1 |
| ------------------- | -----: |
| PHM full            |    .78 |
| − survivor rescue  |    .72 |
| − soft conjunction |    .69 |
| − delta view       |    .73 |

Một paper có bảng như vậy dễ defend hơn rất nhiều so với:

```text
7/7 state-machine traces passed.
```

---

# Nhưng có một điều kiện cực quan trọng

Nếu **PHM chỉ tồn tại trong file Markdown** còn BDI đã có implementation + paper + tests, thì với deadline **22/09/2026** hiện tại, comparison không còn đơn giản.

Deadline submission đang là **22 September 2026**. ([Video Browser Showdown](https://videobrowsershowdown.org/call-for-papers/important-dates/?utm_source=chatgpt.com))

Do đó:

### Nếu PHM chưa có code/result nào

BDI hiện **an toàn hơn về khả năng hoàn thành submission**, mặc dù idea yếu hơn.

### Nếu trong thời gian còn lại bạn có thể có ít nhất:

```text
PHM session implementation
+
current cumulative baseline
+
historical progressive queries
+
R@1 / MRR / Recall@K
+
1–2 ablation
+
UI screenshot
```

thì **tui sẽ chuyển paper sang PHM**.

Không cần implement confidence model hoàn hảo.

---

# Tui thậm chí sẽ giảm scope PHM để paper mạnh hơn

Đừng cố nhét toàn bộ file 7.000 dòng vào một paper 6 trang.

Core paper chỉ nên là:

```text
Progressive Hint Memory
│
├── Multi-view hint retrieval
│    delta / context / cumulative
│
├── Video-level evidence memory
│
├── Soft cross-hint aggregation
│
└── Survivor rescue
```

Và xem:

```text
trajectory confidence
early submit
```

là **application/analysis**, không phải contribution chính.

Tức title có thể gần kiểu:

> **Stateful Progressive-Hint Retrieval for Interactive Known-Item Video Search**

hoặc

> **Progressive Hint Memory for Interactive Video Known-Item Search**

Story rất gọn.

---

# Cái nào tui sẽ submit?

**PHM.**

Nếu cả hai ở cùng mức hoàn thiện, tui xếp:

$$
\boxed{\text{PHM} > \text{BDI}}
$$

khá rõ cho **VBS 2027**.

Không phải vì BDI là idea tệ. BDI thực ra là một module infrastructure tốt và **nên giữ lại trong AIC2026**.

Nhưng vai trò hợp lý hơn của chúng là:

```text
MAIN PAPER CONTRIBUTION
        │
        ▼
Progressive Hint Memory
        │
        ▼
retrieval over progressive KIST
        │
        ▼
candidate ranking
        │
        ├─────────────→ BDI
        │               optional reliable inspection
        │
        ▼
confidence / operator
```

Tức **PHM làm scientific/retrieval contribution chính**, còn BDI có thể được nhắc như architecture feature của system hoặc để dành thành một systems paper khác.

Nếu phải chọn đúng **một idea để cược submission hiện tại**, tui sẽ chọn **Progressive Hint Memory + Survivor Rescue**, với Early-Submit Confidence là secondary contribution. Đây vừa sát task VBS hơn, vừa có hypothesis định lượng rõ, vừa dễ làm ablation, và ít bị reviewer quy thành “standard software engineering mechanisms” hơn BDI.

---

# you asked

message time: 2026-09-20 17:39:17

hiện tại do không có nhiều thời gian nên PHM tui không thể tạo vài nghìn query và train được thì có sao không

---

# chatgpt response

Không sao. Với PHM, **không có vài nghìn query để train không phải deal-breaker** — miễn là bạn **đừng biến PHM thành một method phụ thuộc supervised training**.

Thậm chí với trạng thái hiện tại, tui nghĩ hướng an toàn hơn cho paper là:

> **PHM core = training-free / parameter-light retrieval method**
> **Confidence model = optional extension, không phải contribution bắt buộc.**

Điều này còn khớp với chính thiết kế trong tài liệu của bạn: vì benchmark hiện chỉ có khoảng **67 T-KIS query**, file đã chủ động đề xuất PHM-v1 dùng `w_i = 1`, chưa học hint reliability phức tạp; supervised weighting chỉ để sau khi có thêm benchmark.

## Tui sẽ cắt PHM xuống như này

Core PHM không cần train:

```text
Progressive textual hints
        ↓
1. Extract delta / contextual / cumulative views
        ↓
2. Retrieve each hint/view independently
        ↓
3. Convert ranks → normalized evidence
        ↓
4. Store evidence at VIDEO level
        ↓
5. Soft aggregate evidence across hints
        ↓
6. Survivor Rescue inside strong previous videos
        ↓
7. Updated ranking
```

Trong đó tất cả có thể deterministic.

Ví dụ rank evidence:

$$
e_i(v)=\frac{k+1}{k+r_i(v)}
$$

và memory:

$$
M_t(v)=
\exp\left[
\frac{1}{t}
\sum_{i=1}^{t}
\log\left(
\epsilon+(1-\epsilon)e_i(v)
\right)
\right]
$$

rồi:

$$
S_t(v)
=
\alpha M_t(v)
+
\beta e_t(v)
+
\gamma T_t(v)
$$

Bạn chỉ có vài hyperparameter như:

```text
ε
α
β
γ
top-N survivor
local rescue depth
```

Chúng **không đòi vài nghìn query để train**.

Bạn có thể chọn một cấu hình hợp lý trước, rồi làm **small sensitivity analysis** thay vì “training”.

---

## Ví dụ thay vì train α, β

Đừng nói:

> We trained α=0.70 and β=0.25.

Mà làm:

```text
α ∈ {0.5, 0.6, 0.7, 0.8}
β = 1 - α
```

và báo:

```text
performance ổn định trong khoảng α = 0.6–0.8
```

Nếu đúng.

Đây thậm chí còn đẹp hơn cho paper vì chứng minh method **không nhạy với một parameter được tune đặc biệt trên tập nhỏ**.

---

# Cái cần data không phải là PHM core

Phần thực sự cần nhiều data là:

```text
Early Submit Confidence
```

bởi vì nếu bạn muốn output:

$$
P(\text{Top1 correct})=0.93
$$

thì đây là một calibrated probability.

Muốn claim con số đó nghiêm túc thì cần dataset đủ lớn để:

```text
train
+
validation
+
calibration
+
held-out evaluation
```

File của bạn cũng đã nhận ra vấn đề này: các prefix của cùng query không độc lập và dataset nhỏ, nên nếu làm supervised thì chỉ đề xuất L2 logistic regression, split theo **query session/target video**, tuyệt đối không random-split các prefix.

Với thời gian hiện tại, tui sẽ **bỏ calibrated Early Submit Confidence khỏi contribution chính**.

---

# Thay Early Submit Confidence bằng training-free Stability Indicator

Cái này rất hợp.

Thay vì:

```text
P(correct) = 93.7%
```

hãy chỉ hiển thị các tín hiệu quan sát được:

```text
Candidate V004

Current rank       #1
Previous ranks     #18 → #6 → #2 → #1
Top-1 streak       1
PE/Qwen agreement  high
Hint evidence      4/4
Ranking stability  high

Status: STABLE
```

Hoặc:

```text
Candidate V018

#1 → #4 → #11 → #19

Status: UNSTABLE
```

Không claim đó là probability.

Có thể định nghĩa một deterministic indicator từ:

- rank trajectory;
- Top-K survival;
- RBO giữa hai rankings;
- PE/Qwen agreement;
- top1/top2 margin;
- hint evidence.

Rồi gọi nó là:

> **ranking stability indicator**

chứ không phải:

> confidence probability.

Như vậy **không cần train**.

---

# Tui thậm chí nghĩ paper sẽ sạch hơn

Paper ban đầu hơi tham:

```text
PHM
+
Survivor Rescue
+
trajectory
+
confidence calibration
+
early stopping
```

Với 6+2 trang và deadline gần, quá nhiều.

Paper nên tập trung:

## Contribution 1 — Progressive Hint Decomposition

Thay vì:

$$
H_1+H_2+H_3
$$

thành một query ngày càng dài, mỗi round giữ:

```text
new evidence
minimal context
full accumulated description
```

---

## Contribution 2 — Persistent Video-Level Evidence Memory

Một video không mất evidence từ H1 chỉ vì H2 global retrieval không retrieve nó.

```text
H1 evidence ─┐
H2 evidence ─┼─→ memory(video)
H3 evidence ─┘
```

---

## Contribution 3 — Survivor Rescue

Đây là phần tui sẽ nhấn mạnh nhất:

```text
               new hint
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
global corpus search    historical survivors
                            │
                            ▼
                     in-video retrieval
        │                   │
        └─────────┬─────────┘
                  ▼
              updated rank
```

Method này không cần học parameter bằng neural model.

Và nó tạo một hypothesis rất dễ kiểm chứng:

> Does persistent candidate memory plus local survivor rescue improve ranking across progressive textual hints compared with cumulative-query retrieval?

Đây là một câu research rất rõ.

---

# Bạn cần bao nhiêu query?

Không có con số magic kiểu “phải 1.000”.

Đặc biệt đây là **VBS extended demo paper**, không phải NeurIPS benchmark paper. CFP yêu cầu một hệ thống video search được mô tả chi tiết, có screenshot và hỗ trợ interactive retrieval; không đặt yêu cầu phải có một training corpus lớn hay phải train model riêng. ([Video Browser Showdown](https://videobrowsershowdown.org/call-for-papers/?utm_source=chatgpt.com))

Có một precedent khá quan trọng: paper **Robust Relevance Feedback for Interactive KIS** năm 2025 là một nghiên cứu retrieval đầy đủ, và evaluation được thực hiện trên V3C; tức interactive KIS research không mặc định đòi một learned model từ hàng nghìn progressive queries. ([arXiv](https://arxiv.org/abs/2505.15128?utm_source=chatgpt.com))

Với PHM hiện tại, nếu bạn có:

```text
~50–100 query có GT tốt
```

tui vẫn nghĩ đủ để làm **exploratory / system-level evaluation**, với điều kiện không overclaim.

67 query hiện tại không phải lý tưởng, nhưng **dùng được**.

---

# Với 67 query, cách evaluation quan trọng hơn số lượng

Giả sử mỗi query có 3 progressive hints:

```text
67 query × 3 stages
```

bạn có 201 observations để vẽ behavior theo round.

Nhưng nhớ:

> Không được gọi đó là 201 independent queries.

Các prefix cùng một query correlated.

Tức các metric nên tính **ở query level**.

Ví dụ:

| Method      | H1 R@1 | H2 R@1 | H3 R@1 | Final MRR |
| ----------- | -----: | -----: | -----: | --------: |
| cumulative  |    ... |    ... |    ... |       ... |
| latest hint |    ... |    ... |    ... |       ... |
| Hint-RRF    |    ... |    ... |    ... |       ... |
| PHM         |    ... |    ... |    ... |       ... |

Rồi bootstrap/query-level CI nếu kịp.

---

# Quan trọng nhất: baseline

Bạn không cần train dataset khổng lồ nếu baseline đủ thuyết phục.

Tui sẽ benchmark ít nhất 4:

```text
B1: Cumulative
H1
H1+H2
H1+H2+H3
```

Đây chính là current AIC behavior.

```text
B2: Latest Hint
H1
H2
H3
```

```text
B3: Independent Hint RRF
retrieve H1, H2, H3 independently
→ fuse ranks
```

Và:

```text
OURS:
PHM
+ persistent video memory
+ soft conjunction
+ survivor rescue
```

Đặc biệt B3 rất quan trọng.

Reviewer có thể hỏi:

> Why isn't this simply RRF over all previous hint rankings?

Bạn phải trả lời bằng experiment.

---

# Ablation cũng không cần training

Rất tiện.

```text
PHM full
```

so với:

```text
PHM – survivor rescue
```

```text
PHM – delta view
```

```text
PHM with arithmetic mean
vs
PHM with soft geometric aggregation
```

Nếu `survivor rescue` mang lại gain rõ ràng thì story rất mạnh.

---

# Nếu hint thật quá ít thì sao?

Trong file bạn có một phương án hợp lý:

Nếu chỉ có final descriptions mà không còn progressive reveal thực, có thể **synthetic-split description thành H1/H2/H3 để test ranking**, nhưng **không dùng synthetic prefixes để calibrate confidence**, vì distribution khác hint thật.

Tui đồng ý với giới hạn đó.

Ví dụ một description:

> A man sits in a kitchen. A white refrigerator is behind him. He holds a watermelon.

có thể thành:

```text
H1:
A man sits indoors.

H2:
A white refrigerator is behind him.

H3:
He is holding a watermelon.
```

Dùng để kiểm tra:

```text
cumulative
vs
PHM
```

được.

Nhưng không được train classifier rồi claim:

```text
93% confidence means 93% probability correct
```

từ data synthetic đó.

---

# Còn một lựa chọn rất hay: không “train”, chỉ “tune once”

Nếu bạn lo reviewer hỏi parameter lấy đâu ra, có thể chia 67 query như:

```text
Development: 20
Evaluation: 47
```

Dùng 20 query để chọn một cấu hình:

```text
ε
α
rescue_top_n
```

rồi **freeze**.

47 query còn lại chạy đúng một lần.

Không phải phương án statistical lý tưởng, nhưng scientific hơn nhiều so với tune trực tiếp trên toàn bộ 67 rồi report toàn bộ 67.

Nếu dataset quá nhỏ để split như vậy, dùng leave-one-query/group CV cho parameter sensitivity, nhưng **đừng train model phức tạp**.

---

# Tui sẽ thay scope paper thành

> **A training-free stateful retrieval method for progressively revealed textual descriptions.**

Câu đó theo tui còn attractive.

Nó biến limitation:

> “Chúng tôi không có vài nghìn query để train.”

thành một design decision:

> “Our method requires no task-specific learned ranker or progressive-hint training corpus.”

Pipeline:

```text
Existing pretrained retrievers
PE / Qwen / TARA
        │
        ▼
PHM
(no additional model training)
        │
        ├── hint decomposition
        ├── rank-normalized evidence
        ├── persistent video memory
        └── survivor rescue
        │
        ▼
interactive result
```

Tức bạn tận dụng pretrained model đã có nhưng **PHM bản thân training-free**.

---

## Nếu chỉ còn rất ít thời gian, tui sẽ làm đúng 5 thứ

1. Implement **persistent video memory**.
2. Implement **independent delta/current-hint retrieval**.
3. Implement **Survivor Rescue** bằng search restricted theo `video_id` mà repo của bạn đã có.
4. Benchmark `cumulative vs hint-RRF vs PHM`.
5. Làm một ablation `PHM – rescue`.

**Bỏ hẳn learned confidence model ra khỏi submission hiện tại.**

Nếu còn chỗ trong paper thì chỉ nói:

> Candidate trajectories are exposed to the operator as stability cues; calibrated early-submission prediction is left for future work.

Như vậy scope vừa đủ, không cần tạo hàng nghìn query và cũng tránh tình trạng lấy 67 query để train một classifier rồi reviewer bắt ngay overfitting/leakage.

Với tình trạng hiện tại, tui thậm chí **thích phiên bản PHM training-free này hơn phiên bản PHM + learned confidence**, vì claim nhỏ hơn nhưng dễ bảo vệ hơn nhiều.

---

# you asked

message time: 2026-09-20 17:47:38

PHM chỉ cần thêm thuật toán vào chứ không cần thêm bước tiền xử lý đúng không

---

# chatgpt response

Đúng. Với phiên bản **PHM training-free** mà mình vừa chốt, bạn **không cần thêm một pipeline tiền xử lý dataset mới** theo kiểu re-encode video, tạo embedding mới hay xây index mới.

PHM chủ yếu là một **thuật toán/stateful ranking layer chạy ở query-time** nằm trên retrieval stack hiện tại:

```text
PE / Qwen / OCR / Speech / TARA
              ↓
        retrieval hiện có
              ↓
             PHM
              ↓
   video-level memory/ranking
              ↓
              UI
```

Những thứ PHM cần thêm chủ yếu là xử lý online:

- tách hint mới thành `delta / context / cumulative`;
- lưu state của từng progressive-query session;
- lưu evidence/rank của từng video qua các hint;
- tính soft aggregation;
- chạy Survivor Rescue trên các video mạnh từ vòng trước;
- rerank lại candidate list.

Thiết kế trong tài liệu cũng xác định PHM là `ProgressiveQuerySession` + `CandidateMemory`, tức state được tích lũy qua từng hint chứ không phải một feature phải tính trước cho toàn corpus.

### Quan trọng nhất: không cần tạo embedding mới

Bạn vẫn dùng nguyên:

```text
PE embedding index
Qwen3-VL embedding index
OCR/ASR index
TARA index
```

PHM chỉ lấy **rank/evidence** từ các retriever đó. Thậm chí tài liệu còn nhấn mạnh không được cộng raw cosine của PE và Qwen vì chúng thuộc hai embedding space khác nhau; PHM nên tích lũy rank-normalized evidence.

Do đó không có bước kiểu:

```text
download toàn dataset
→ decode video
→ extract thêm keyframes
→ chạy model toàn corpus
→ tạo embedding mới
→ upload collection mới
```

cho PHM.

## Survivor Rescue cũng không cần preprocess mới

Đây là chỗ rất thuận lợi với repo hiện tại.

Ý tưởng là:

```text
Hint 1
→ V123 rank #2

Hint 2 global search
→ V123 bị miss

PHM:
→ search Hint 2 riêng trong V123
→ tìm frame phù hợp trong video đó
→ giữ/rescue V123
```

Tài liệu của bạn ghi rõ repo hiện đã có primitive search với `video_id`, vốn đang được dùng theo tư duy tương tự cho TRAKE pass-2.

Nên Survivor Rescue chỉ cần gọi index hiện tại với filter:

```python
video_id="V123"
```

không cần tạo local index riêng cho từng video.

---

Có một loại “preprocessing” rất nhẹ nhưng đó là **query preprocessing**, không phải dataset preprocessing:

```text
Round 1:
"A man is sitting at a table"

Round 2:
"A man is sitting at a table.
There is a red bottle in front of him."
```

PHM phải phát hiện:

```text
delta H2 =
"There is a red bottle in front of him."
```

để không tính lại evidence cũ hai lần. File PHM có đề xuất lưu `raw_text`, `delta_text`, `cumulative_text` cho mỗi turn.

Cái này chỉ là string/sentence processing lúc nhận hint, gần như không tốn compute.

### Vậy kiến trúc thực tế sẽ là

```text
                    OFFLINE
────────────────────────────────────
Existing AIC preprocessing
      │
      ├── PE embeddings
      ├── Qwen embeddings
      ├── OCR
      ├── ASR
      ├── Audio
      └── TARA
              │
              ▼
      Existing databases/indexes


                   ONLINE
────────────────────────────────────
Hint H_t
   │
   ▼
Hint differencing
   │
   ├── delta
   ├── context
   └── cumulative
   │
   ▼
Existing SearchService
   │
   ▼
per-hint video rankings
   │
   ▼
┌────────────────────────┐
│ PHM Session            │
│                        │
│ video evidence memory  │
│ soft aggregation       │
│ survivor list          │
│ rank trajectory        │
└────────────┬───────────┘
             │
     survivor rescue
             │
             ▼
 Existing Milvus search
 filter video_id=...
             │
             ▼
       final ranking
```

Nên nếu mục tiêu là **implement PHM vào AIC2026**, phần nặng không nằm ở data preprocessing. Phần chính là sửa orchestration/search logic và thêm session state.

Có một ngoại lệ: nếu index hiện tại **không lưu `video_id` hoặc không hỗ trợ filtered retrieval** thì mới phải chỉnh index/schema. Nhưng theo chính phân tích PHM của repo hiện tại, primitive `search_image(..., video_id=...)` và tương đương Qwen đã tồn tại, nên trường hợp của bạn **không cần làm lại index**.

Vì vậy về mặt thời gian, PHM thuận lợi hơn khá nhiều: **đây gần như là thay đổi thuật toán + backend state, không phải một preprocessing project mới.**
