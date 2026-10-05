> From: https://chatgpt.com/c/6abff06f-7d78-83ec-82b5-5c67e437dc9e

# you asked

message time: 2026-10-03 00:57:51

Đọc repo aic 2026 của tui. Thì hiện tại tui muốn viết paper soict 2026 sắp tới, và tui muốn nâng cấp hệ thống agent của tui, tích hợp thêm model jev 1.13 model vào hệ  thống agent và mở rộng tool và kien61 trúc agent, giúp tui research hướng và nâng cấp hệ thống agent

---

# chatgpt response

Mình sẽ làm theo 3 lớp: đọc kiến trúc agent hiện tại trong repo AIC 2026, xác định chính xác “JEV 1.13” là model nào và khả năng của nó, rồi đối chiếu literature gần đây để đề xuất một kiến trúc agent có contribution đủ rõ cho SOICT — không chỉ “thêm một model vào ensemble”.

Mình đã xác nhận repo hiện tại vẫn đúng mô hình “agent sidecar”: Codex + Claude chạy song song, dùng chung MCP tools, nhưng kết quả agent chưa đi vào main ranking. Điểm này quan trọng vì hướng paper nên biến agent từ “2 model cùng đi tìm” thành một cơ chế **phân vai + kiểm chứng + điều phối search có thể đo được**, nếu không novelty sẽ khá mỏng.

Jev 1.13 đúng là một mảnh ghép khá hợp, nhưng **không nên dùng nó như agent thứ ba kiểu Claude/Codex**. Nó là “decision model”: nhận state + các câu hỏi có cấu trúc rồi trả xác suất/quyết định, rất nhanh, không sinh văn bản và không xem ảnh trực tiếp. Vì vậy vai trò mạnh nhất của nó trong hệ thống của bạn là **controller/judge/router** nằm giữa retrieval và các agent đắt tiền.

Mình đã đọc lại `vonhatbao2205/AIC2026` ở `main` hiện tại. Với trạng thái repo bây giờ, hướng tốt nhất không phải là **“thêm Jev 1.13 thành agent thứ ba cạnh Codex + Claude”**. Nếu làm vậy contribution sẽ khá yếu.

Hướng đáng làm hơn là biến Jev thành **decision layer điều phối toàn bộ hệ thống agent**, theo kiến trúc **fast–slow / System-1–System-2**, trong đó Jev quyết định *khi nào cần agent, agent nào cần chạy, tool nào đáng gọi tiếp, candidate nào đủ bằng chứng để dừng*. Codex và Claude trở thành các System-2 search agents chỉ được kích hoạt khi cần.

## 1. Vấn đề của agent architecture hiện tại

Repo hiện tại của bạn có thiết kế khá tốt cho competition:

```text
User query
    │
    ├──────── Main Retrieval
    │         PE / Qwen / OCR / ASR / Audio / TARA
    │         → RRF
    │         → result ~ vài giây
    │
    ├──────── Codex Agent ─┐
    │                      ├─ independent search
    └──────── Claude Agent ┘
                           ↓
                      agent candidates
```

Trong `backend/app/agent/runs.py` hiện vẫn hard-code:

```python
AGENTS = ("codex", "claude")
```

và hai agent:

- chạy song song;
- dùng **cùng prompt**;
- dùng cùng bộ MCP tools;
- không chia vai trò;
- không có shared evidence state;
- không quyết định agent nào thực sự cần thiết;
- confidence của candidate chủ yếu do chính agent tự khai;
- `/api/search` không chờ agent và agent candidate không tham gia main ranking.

Bộ tool hiện tại lại đã khá mạnh:

`search`, `list_videos`, `folder_frames`, `video_outline`, `video_frames`, `view_frames`, `video_text`, `report_candidate`.

Vấn đề không còn là “agent có đủ tool không”, mà là **agent sử dụng tool thế nào**.

Ví dụ query đơn giản về OCR vẫn chạy cả Claude + Codex. Query khó về relation cũng chạy cả hai với cùng chiến lược. Hai agent thường có thể dò cùng candidate, gọi lại cùng `search`, `video_outline`, `video_frames` → tốn khoảng thời gian System-2 rất lớn.

Đây chính là chỗ Jev hợp với hệ thống.

---

# 2. Tại sao Jev 1.13 phù hợp

Jev không phải chatbot thông thường. TypeSafe mô tả nó như một **System One decision model**: nhận state có cấu trúc và trả về các typed decisions/probabilities thay vì sinh một chuỗi văn bản dài. Họ định vị nó cho các tác vụ classify, route, score, verify và branching trong software workflow. ([TypeSafe AI](https://typesafe.ai/blog/introducing-system-one-models-and-jev?utm_source=chatgpt.com))

Do đó:

```text
❌ Không nên:

Jev Agent
Claude Agent
Codex Agent
     ↓
3 agent cùng search
```

Mà nên:

```text
                  Jev 1.13
             Fast Decision Layer
                   │
           ┌───────┴────────┐
           ↓                ↓
       easy query        hard query
           │                │
      Retrieval       System-2 agents
           │          Codex / Claude
           └──────┬─────────┘
                  ↓
              Jev verify
                  ↓
               STOP
```

Điểm hay là Jev cực kỳ phù hợp với bài toán **“trong 8–15 action thì action nào nên thực hiện tiếp?”**, vì đây là decision problem hơn là generation problem.

---

# 3. Nhưng chỉ dùng Jev reranking thì KHÔNG đủ novelty

Mình đã kiểm tra phần này khá kỹ.

Đã có project dùng Jev trực tiếp phía sau video embeddings:

> embeddings retrieve → Jev decides/reranks.

Họ đã thử trên MSR-VTT và nhận thấy Jev có thể giúp với compositional/relational queries, nhưng kết quả reranking phụ thuộc mạnh vào representation/caption đầu vào. ([GitHub](https://github.com/Anmol-Srv/jev-video-search?utm_source=chatgpt.com))

Jev cũng đã được thử cho document/RAG reranking và cho thấy có thể cải thiện thứ hạng candidate trong một số benchmark. ([GitHub](https://github.com/erendikmenn/jev-rag-benchmark?utm_source=chatgpt.com))

Vì vậy paper kiểu:

> “We integrate Jev 1.13 into video retrieval for reranking.”

mình sẽ **không chọn**.

---

# 4. Và “multi-agent video search” cũng không còn mới

Đây là mối đe dọa novelty lớn hơn.

**Adaptive Multi-Agent Reasoning for Text-to-Video Retrieval — ICMR 2026** đã có Retrieval Agent + Contextual Reasoning Agent + Query Reformulation Agent + Orchestration Agent. Orchestrator quyết định các vòng reasoning/search tiếp theo. ([arXiv](https://arxiv.org/abs/2602.19040?utm_source=chatgpt.com))

**Deep Video Discovery — NeurIPS 2025** đã cho LLM tự chọn search tools trên multi-granular video database và iteratively refine search. ([NeurIPS Proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/8190b210e9808e54ee16263b673a847d-Abstract-Conference.html?utm_source=chatgpt.com))

**VideoMind — ICLR 2026** còn chia Planner / Grounder / Verifier / Answerer. ([VideoMind](https://videomind.github.io/?utm_source=chatgpt.com))

**LensWalk — CVPR 2026** cho agent quyết định temporal range và sampling density để xem video tiếp theo. ([The Computer Vision Foundation](https://openaccess.thecvf.com/content/CVPR2026/html/Li_LensWalk_Agentic_Video_Understanding_by_Planning_How_You_See_in_CVPR_2026_paper.html?utm_source=chatgpt.com))

Ngay SOICT 2025 cũng đã có cả *Towards Conversational Video Retrieval with an Intelligent Search Agent* và *Applying LLM Agents for Automated Lifelog Retrieval*. ([SOICT 2026](https://soict.org/wp-content/uploads/2025/12/SOICT2025-ProgramBook.pdf?utm_source=chatgpt.com))

Cho nên:

> “Codex + Claude + Jev multi-agent orchestration”

vẫn chưa đủ.

---

# 5. Hướng mình đề xuất: **CAD-VR**

Mình tạm gọi kiến trúc là:

**CAD-VR — Calibrated Adaptive Decisioning for Agentic Video Retrieval**

Core contribution:

> **A calibrated fast–slow agentic retrieval architecture where a non-generative decision model continuously decides whether expensive LLM reasoning is necessary, which evidence is missing, which search action has the highest expected value, and when retrieval can safely terminate.**

Đây mới là sự khác biệt.

```text
                         QUERY
                           │
                           ▼
                 ┌─────────────────┐
                 │ Query Compiler  │
                 │ constraints Cq  │
                 └────────┬────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      Main Retrieval              Jev Router
 PE/Qwen/OCR/ASR/TARA          System-1 (~fast)
             │                         │
             └───────────┬─────────────┘
                         ▼
                 EVIDENCE BOARD
                         │
                  Jev Decision
            ┌────────────┼────────────┐
            │            │            │
          STOP      Codex Scout   Claude Investigator
            │            │            │
            │       tools/search      │
            │            └─────┬──────┘
            │                  ▼
            │           Evidence Board
            │                  │
            │                  ▼
            │            Jev Verifier
            │                  │
            └──────────────────┘
                         │
                         ▼
                    FINAL RANK
```

Điểm quan trọng: **Jev không search video**.

Nó quyết định **search policy**.

---

# 6. Contribution mạnh nhất: Query Constraint Graph

Đừng đưa nguyên query cho Jev kiểu:

```text
"một người đàn ông mặc áo vàng..."
```

Hãy compile query thành constraints.

Ví dụ:

```json
{
  "subject": "man",
  "clothing": "yellow shirt",
  "object": "bicycle",
  "action": "repairing",
  "location": "roadside",
  "relation": "man interacts with bicycle"
}
```

TRAKE có thể thành:

```text
GLOBAL:
  subject = red lion

E1:
  jumps

E2:
  approaches crowd

E3:
  turns around
```

Candidate A có evidence:

```text
subject          0.96
yellow-shirt     0.91
bicycle          0.98
repair-action    0.42   ← uncertainty
roadside         0.89
```

Jev không cần hỏi:

> Candidate này có đúng không?

Mà hỏi song song:

```text
Does candidate satisfy subject?
Does it satisfy clothing?
Does it satisfy action?
Does it satisfy location?
Does it satisfy relation?
```

Sau đó **code** tổng hợp probabilities.

Ví dụ:

$$
S(v,q)=
\exp\left(
\frac{\sum_i w_i \log(p_i+\epsilon)}
{\sum_i w_i}
\right)
$$

Geometric aggregation rất phù hợp retrieval vì:

```text
man ✓
yellow ✓
bicycle ✓
road ✓
but repairing ✗
```

không nên vẫn được score cao chỉ vì 4/5 thuộc tính đúng.

Đây trực tiếp giải quyết một failure mode mà embedding retrieval thường gặp: **query có nhiều constraint nhưng candidate chỉ match nouns/global semantics**.

---

# 7. Jev nên có 3 nhiệm vụ khác nhau

### Jev-G0 — Query Router

Ngay sau query:

```text
questions:

query_modality:
 visual / OCR / speech / metadata / temporal / mixed

query_complexity:
 simple / compositional / temporal

primary_route:
 retrieval_only
 visual_search
 text_search
 browse
 agent_reasoning

slow_agent_required:
 yes / no
```

Một call có thể quyết định nhiều field.

---

### Jev-G1 — Candidate Gate

Sau main retrieval:

```text
query
+
top candidates
+
scores by PE/Qwen/OCR/ASR
+
OCR
+
speech
+
metadata
+
VLM textual descriptions
```

Jev đánh giá:

```text
candidate A relevance
candidate B relevance

candidate A constraint coverage
candidate B constraint coverage

is top candidate sufficiently supported?
```

Nếu candidate rất rõ:

```text
Main retrieval
     ↓
Jev 0.94
     ↓
STOP
```

Không cần Codex/Claude.

---

### Jev-G2 — Escalation Controller

Nếu uncertainty cao:

```text
Jev choice:

SEARCH_VISUAL
SEARCH_OCR
SEARCH_SPEECH
BROWSE_FOLDER
INSPECT_VIDEO
ZOOM_TEMPORAL
COMPARE_CANDIDATES
CALL_CODEX
CALL_CLAUDE
CALL_BOTH
STOP
```

Đây mới là chỗ rất hợp với decision model.

---

# 8. Codex và Claude cũng phải đổi vai

Hiện tại `prompt.py` cố tình dùng gần như cùng prompt cho cả hai.

Cho competition thì hợp lý vì diversity tự nhiên.

Cho paper thì chưa đủ principled.

Mình sẽ đổi thành:

| Component                     | Vai trò                                                            |
| ----------------------------- | ------------------------------------------------------------------- |
| **Codex Scout**         | high-recall exploration, query decomposition, search/browse nhanh   |
| **Claude Investigator** | inspect candidate sâu, relation/temporal reasoning, counterexample |
| **Jev Controller**      | routing, confidence, action selection, stopping                     |
| **Jev Verifier**        | constraint-level candidate scoring                                  |
| **Main retrieval**      | cheap high-recall candidate generator                               |

Claude **không chạy ngay từ đầu**.

Ví dụ:

```text
Retrieval
    ↓
Jev unsure
    ↓
Codex Scout
    ↓
Jev still unsure?
    │
 NO ─────→ STOP
    │
 YES
    ↓
Claude Investigator
```

Như vậy bạn có một **adaptive computational cascade**.

---

# 9. Thêm Evidence Board, nhưng KHÔNG phải Progressive Memory

Cái này rất quan trọng vì SOICT không được reuse Progressive Memory của VBS.

Evidence Board chỉ tồn tại **trong một query**:

```python
EvidenceBoard:
    query_constraints

    candidate_1:
        retrieval_scores
        inspected_frames
        ocr
        speech
        satisfied_constraints
        rejected_constraints
        agent_sources

    candidate_2:
        ...
```

Kết thúc query:

```text
destroy EvidenceBoard
```

Không có:

```text
previous query
previous hint
cross-query history
progressive hint memory
```

Mình thậm chí sẽ disable `previous_hints` trong experimental SOICT path.

Vậy kiến trúc này hoàn toàn tách khỏi PHM/VBS của bạn.

---

# 10. Bộ tool hiện tại nên mở rộng như sau

Không nên thêm 20 tool linh tinh. Mình chỉ thêm khoảng 5 abstraction-level tools:

| Tool mới                 | Chức năng                                                                       |
| ------------------------- | --------------------------------------------------------------------------------- |
| `candidate_evidence`    | gom PE/Qwen/OCR/ASR/metadata/neighbor evidence cho một candidate                 |
| `temporal_neighborhood` | lấy context trước/sau candidate thay vì agent tự gọi nhiều`video_frames` |
| `compare_candidates`    | một contact sheet A/B/C + evidence để verifier phân biệt                     |
| `constraint_probe`      | search chỉ cho**một missing constraint**                                  |
| `inspect_candidate`     | wrapper của outline + frames + text quanh một moment                            |

Ví dụ agent phát hiện:

```text
Candidate A:
man ✓
bicycle ✓
yellow shirt ?
```

thì không search lại cả query.

Nó gọi:

```text
constraint_probe(
    candidate=A,
    constraint="yellow shirt"
)
```

Đây là **evidence-directed search**.

---

# 11. Một cải tiến nữa rất đáng làm: disagreement-driven verification

Hiện tại hai agent độc lập tìm candidate.

Thay vì chỉ hiện:

```text
Codex: L21_V012/033
Claude: L21_V017/041
```

hãy biến disagreement thành signal.

```text
Codex = A
Claude = B

        ↓

Jev:
P(A) = .52
P(B) = .48

        ↓

compare_candidates(A, B)

        ↓

Claude verifier ONLY
```

Ngược lại:

```text
Codex = A
Claude = A
retrieval = A
Jev = .96

→ STOP
```

Tức là compute được phân bổ theo **uncertainty/disagreement**, không phải fixed workflow.

---

# 12. Đây mới là novelty statement có thể viết trong paper

Mình sẽ tránh claim “first multi-agent video retrieval”.

Thay vào đó:

> *We investigate calibrated decision models as a machine-native control layer for agentic video retrieval. Unlike existing approaches that rely on generative LLMs for both orchestration and reasoning, CAD-VR separates fast probabilistic control from expensive open-ended search. A System-One controller performs query routing, constraint-level candidate verification, adaptive agent escalation, and evidence-aware stopping, while System-Two LLM agents are invoked only for unresolved evidence.*

Trong các work mình tìm được, mình **chưa thấy kiến trúc video retrieval nào sử dụng một non-generative calibrated decision model làm closed-loop controller cho System-2 video-search agents**. Nhưng đây nên được viết là research gap cần empirical validation, không nên claim tuyệt đối “world first”.

Đặc biệt vì fast/slow agent routing nói chung đã tồn tại ở các domain khác. PRIME dùng System-1/System-2 cho reasoning, và RACER nghiên cứu calibrated model routing; budget-aware tool agents cũng đã được nghiên cứu. ([AAAI Publications](https://ojs.aaai.org/index.php/AAAI/article/view/40612?utm_source=chatgpt.com))

Điểm bạn phải chứng minh là:

> **video-specific evidence + constraint verification + agent/tool escalation**.

---

# 13. Experiment mới là thứ quyết định paper có mạnh hay không

Đừng chỉ report R@1.

Mình sẽ benchmark 6 configurations:

| ID          | System                            |
| ----------- | --------------------------------- |
| A           | current hybrid retrieval          |
| B           | retrieval + always Codex          |
| C           | retrieval + always Codex + Claude |
| D           | retrieval + Jev reranking         |
| E           | Jev routing + adaptive agents     |
| **F** | **full CAD-VR**             |

Metrics quan trọng:

```text
R@1
R@5
MRR

Agent Rescue@1
Agent Rescue@5

Time-to-first-correct
p50 latency
p95 latency

tool calls/query
Codex calls/query
Claude calls/query

tokens/query
$/query

% queries solved without System-2

Jev ECE
Jev Brier score
```

Đặc biệt figure đẹp nhất của paper sẽ không phải chỉ là Recall.

Nó nên là:

```text
                     Accuracy
                       ▲
                       │                 Full Claude+Codex
                       │              ●
                       │         CAD-VR ●
                       │
                       │
                       │   Retrieval ●
                       └────────────────────────► Cost / latency
```

Nếu CAD-VR đạt:

```text
Always Codex+Claude:
R@1 = 0.72
latency = 52s
agent calls = 2.0

CAD-VR:
R@1 = 0.71–0.73
latency = 15–25s
agent calls = 0.7
```

thì đây là một result rất dễ kể thành contribution.

Nếu đồng thời Recall tăng nhờ constraint verification thì càng tốt.

---

# 14. Một ablation rất quan trọng

Phải tách:

```text
Jev là model mạnh
```

khỏi:

```text
kiến trúc của mình mạnh
```

Do đó so:

```text
LLM Router
vs
rule-based router
vs
Jev Router
```

và:

```text
no verifier
Jev verifier
Claude verifier
```

Nếu Jev cho gần quality của Claude nhưng nhanh/rẻ hơn nhiều thì paper có story rất rõ.

TypeSafe tự report latency rất thấp cho System-One workloads, nhưng đừng lấy vendor number làm result paper; hãy đo ngay trên infrastructure của bạn. ([TypeSafe AI](https://typesafe.ai/blog/introducing-system-one-models-and-jev?utm_source=chatgpt.com))

---

# 15. Mapping trực tiếp vào repo

Mình sẽ refactor theo structure này:

```text
backend/app/agent/
│
├── runs.py
├── cli.py
├── prompt.py
├── tools.py
├── mcp_server.py
│
├── decision/
│   ├── jev_client.py
│   ├── router.py
│   ├── verifier.py
│   ├── stopping.py
│   └── calibration.py
│
├── evidence/
│   ├── board.py
│   ├── constraints.py
│   └── compiler.py
│
└── roles/
    ├── scout.py
    └── investigator.py
```

Và flow trong `runs.py` từ:

```python
for name in ("codex", "claude"):
    create_task(run_agent(name))
```

thành:

```text
main retrieval
      ↓
compile evidence
      ↓
Jev route
      ↓
execute selected action
      ↓
update EvidenceBoard
      ↓
Jev evaluate
      ↓
STOP / next action / escalate
```

Jev **không cần đưa vào `AgentName`**.

Nó là controller service.

Đó là distinction kiến trúc rất quan trọng.

---

# 16. Nếu chỉ còn thời gian rất ngắn trước SOICT

Trang SOICT hiện ghi deadline cho **Lifelogging, Event Retrieval, and Personal Data Analytics track là 05/10/2026**. Hôm nay là 03/10/2026, nên nếu đây là track bạn đang nhắm tới thì không nên cố implement một kiến trúc 15 module ngay. ([SOICT 2026](https://soict.org/proceedings/?utm_source=chatgpt.com))

MVP mình sẽ làm theo đúng thứ tự này:

1. **Jev router** — quyết định `main_only / codex / claude / both`.
2. Tách prompt thành **Codex Scout / Claude Investigator**.
3. **Jev candidate verifier** trên top candidates.
4. Shared per-query **Evidence Board**.
5. Benchmark current always-two-agent vs adaptive.
6. Sau đó mới thêm constraint-directed tools.

Chỉ 4 thay đổi đầu tiên đã đủ tạo architecture mới rõ ràng.

---

## Kết luận

Nếu paper là:

> **AIC retrieval + Codex + Claude + Jev**

thì mình đánh giá novelty thấp.

Nếu paper là:

> **Calibrated Fast–Slow Agentic Video Retrieval: a probabilistic System-One controller that dynamically routes tools and System-Two LLM agents, verifies query constraints, and stops search under calibrated uncertainty**

thì story nghiên cứu mạnh hơn hẳn, đồng thời **không đụng Progressive Memory của VBS**.

Nó tận dụng đúng điểm mạnh nhất của repo hiện tại: main retrieval của bạn đã nhanh và đa modality; Claude/Codex đã thực sự rescue được những query khó trong chung kết. Contribution mới không cần xây retrieval model khác — nó nghiên cứu **khi nào và bằng cách nào nên dùng reasoning agents**, và chứng minh trade-off **accuracy–latency–cost** một cách định lượng. Đây cũng khác trực tiếp với ICMR 2026 Adaptive Multi-Agent Retrieval, vốn vẫn dùng MLLM orchestration + reasoning/history, thay vì calibrated non-generative control. ([arXiv](https://arxiv.org/abs/2602.19040?utm_source=chatgpt.com))

Nếu triển khai hướng này, tên paper mình thấy khá khớp là **“CAD-VR: Calibrated Fast–Slow Decision Routing for Agentic Video Retrieval”**.

---

# you asked

message time: 2026-10-03 01:20:56

jev dùng api key trên open router

---

# chatgpt response

Đúng. Nếu bạn định tích hợp Jev 1.13 vào AIC 2026 thì **không cần host Jev local hay đăng ký TypeSafe riêng**. Có thể dùng thẳng **OpenRouter API key**.

Thông tin hiện tại trên OpenRouter:

- Model ID cố định: `typesafe/jev-1.13`
- Hoặc alias luôn theo bản mới nhất: `~typesafe/jev-latest`
- Endpoint riêng cho decision model: `POST https://openrouter.ai/api/alpha/decisions`
- Context: 32K
- Giá hiện tại: **$0.042 / 1M input tokens, $0 output**
- Jev không generate text; nó trả structured decisions/probabilities. ([OpenRouter](https://openrouter.ai/typesafe/jev-1.13/api?utm_source=chatgpt.com))

Với paper SOICT, mình khuyên **pin `typesafe/jev-1.13`**, đừng dùng `jev-latest`, để experiment reproducible.

Trong repo của bạn có thể thêm:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
JEV_MODEL=typesafe/jev-1.13
JEV_BASE_URL=https://openrouter.ai/api
```

Và kiến trúc vẫn giữ như mình đề xuất:

```text
                  Query
                    │
           Main Retrieval
     PE/Qwen/OCR/ASR/TARA
                    │
                    ▼
             Evidence State
                    │
                    ▼
        Jev 1.13 via OpenRouter
          System-1 Controller
                    │
       ┌────────────┼────────────┐
       │            │            │
      STOP      Codex Scout   Claude Investigator
       │            │            │
       └────────────┴─────┬──────┘
                          ▼
                    Evidence update
                          │
                    Jev verification
                          │
                    STOP / continue
```

Đặc biệt, **đừng gọi Jev qua `/chat/completions` như DeepSeek/OpenAI-style chatbot**. Jev được OpenRouter expose qua **Decisions API** và hỗ trợ những kiểu decision như yes/no (`noul`), lựa chọn (`choice`) và ordered score (`score`). ([OpenRouter](https://openrouter.ai/blog/insights/what-is-jev/?utm_source=chatgpt.com))

Ví dụ với AIC, thay vì bắt Jev generate:

> "I think you should call Claude."

ta gửi state kiểu:

```json
{
  "query": "a man in yellow shirt repairing a bicycle",
  "retrieval": {
    "top1_score": 0.72,
    "top2_score": 0.70,
    "ocr_match": 0.03,
    "speech_match": 0.08
  },
  "candidate_evidence": {
    "person": true,
    "yellow_shirt": true,
    "bicycle": true,
    "repairing": "uncertain"
  }
}
```

rồi đặt decision:

```text
next_action ∈ {
    STOP,
    SEARCH_VISUAL,
    SEARCH_OCR,
    SEARCH_SPEECH,
    CODEX_SCOUT,
    CLAUDE_VERIFY,
    BOTH_AGENTS
}
```

Jev trả probability cho các choice, rồi **Python của bạn tự quyết định**.

Chẳng hạn:

```text
STOP                0.04
SEARCH_VISUAL       0.09
CODEX_SCOUT         0.21
CLAUDE_VERIFY       0.61
BOTH_AGENTS         0.05
```

→ controller gọi Claude vì phần `"repairing"` cần visual reasoning sâu hơn.

Đây cũng chính xác là loại use case OpenRouter mô tả cho Jev: **routing, candidate ranking, verification và gating agent tool calls**, thay vì sinh prose. ([OpenRouter](https://openrouter.ai/blog/tutorials/how-to-use-jev/?utm_source=chatgpt.com))

Và chi phí Jev gần như không đáng kể so với Claude/Codex. Ví dụ nếu mỗi decision state khoảng 2,000 input tokens:

$$
2000/10^6 \times \$0.042
=
\$0.000084
$$

tức khoảng **$0.000084/call**. Ngay cả 10 Jev decision/query cũng chỉ khoảng `$0.00084`.

Vì vậy mình còn muốn điều chỉnh kiến trúc SOICT một chút: **cho Jev chạy nhiều lần trong loop cũng được**, thay vì chỉ một router đầu vào:

```text
Retrieve
   ↓
Jev decision #1
   ↓
Codex search
   ↓
Jev decision #2
   ↓
inspect_candidate
   ↓
Jev decision #3
   ↓
Claude verify
   ↓
Jev decision #4
   ↓
STOP
```

Đây mới đúng tinh thần **closed-loop decision controller** và tận dụng lợi thế chi phí/latency của Jev.

Quan trọng nữa: OpenRouter hiện còn có riêng một model **Jev Router**, nhưng với paper của bạn mình **không chọn Jev Router**. Mình sẽ dùng `typesafe/jev-1.13` trực tiếp và **tự định nghĩa search action space của AIC**, vì chính controller/action policy đó mới là phần contribution của paper. ([OpenRouter](https://openrouter.ai/typesafe?utm_source=chatgpt.com))

Nếu tiếp tục theo hướng này, bước hợp lý tiếp theo là mình có thể **đọc sâu `backend/app/agent/` hiện tại và thiết kế chính xác CAD-VR v1 cho repo AIC 2026**, bao gồm `jev_client.py`, schema của từng decision, EvidenceBoard, 5 tool mới, flow Codex Scout → Claude Verifier, và benchmark/ablation để sau đó code thẳng vào repo.

---

# you asked

message time: 2026-10-03 02:26:18

liệu module 8 có làm hệ thống hien tại yếu di không khi tách codex và claude ra làm 2 prompt riêng biệt hay vì 1 prompt duy nhất ?

---

# chatgpt response

Có thể yếu đi **nếu bạn tách vai trò quá cứng**.

Hiện tại Codex và Claude cùng nhận gần như một prompt, cùng toolset, nên lợi thế lớn là **redundancy + independent diversity**: nếu một con hiểu query sai hoặc đi sai nhánh, con kia vẫn có thể tự tìm từ đầu và cứu lại. Nếu bạn ép kiểu:

```text
Codex = chỉ search
Claude = chỉ verify
```

thì bạn có thể mất chính khả năng rescue này.

Mình khuyên **không tách thành hai prompt hoàn toàn khác nhau**, mà dùng:

```text
Shared Core Prompt
+ Role Bias
```

Cụ thể, cả hai vẫn giữ đầy đủ:

- hiểu toàn bộ query;
- dùng toàn bộ tools;
- tự search nếu cần;
- tự inspect candidate;
- tự report candidate;
- được phép bỏ qua role nếu bằng chứng yêu cầu.

Khác nhau chỉ ở **ưu tiên chiến lược**.

Ví dụ:

```text
COMMON CORE
- Understand all constraints.
- Search independently.
- Verify every important constraint.
- You may use any available tool.
- Do not trust another agent's conclusion blindly.
- If your assigned strategy fails, switch strategies.

CODEX BIAS
- Prefer broad exploration.
- Generate several search formulations.
- Explore OCR / speech / visual / folder metadata.
- Maximize candidate recall.
- Quickly identify 3–5 plausible videos.

CLAUDE BIAS
- Prefer deep inspection.
- Compare competing candidates.
- Check relational / temporal / subtle visual constraints.
- Search independently when current candidates are insufficient.
- Look actively for contradictions.
```

Tức là:

```text
             SAME CAPABILITY SET
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
       Codex                Claude
  exploration bias     verification bias
```

chứ không phải:

```text
Codex               Claude
search only         verify only
```

## Vì sao cách này tốt hơn

Giả sử query:

> một người đàn ông áo vàng đang sửa xe đạp bên đường.

Main retrieval ra sai video.

Nếu hard-role:

```text
Codex finds wrong A
      ↓
Claude only verifies A
      ↓
A rejected
      ↓
nothing
```

Hệ thống có thể tệ hơn hiện tại.

Với soft-role:

```text
Codex finds wrong A
      ↓
Claude checks A
      ↓
A rejected
      ↓
Claude allowed to search independently
      ↓
finds B
```

Bạn vẫn giữ khả năng rescue.

---

## Thậm chí với Jev mình cũng không để nó quyết định role tuyệt đối

Không nên:

```text
Jev → CODEX
```

rồi cấm Claude hoàn toàn.

Mình sẽ để Jev ra policy kiểu:

```text
primary_agent = codex
secondary_agent = claude
secondary_trigger = uncertainty > 0.35
```

Ví dụ:

```text
Main retrieval
      ↓
Jev

P(stop)         = 0.11
P(codex_first)  = 0.67
P(claude_first) = 0.17
P(both)         = 0.05
```

→ chạy Codex trước.

Sau Codex:

```text
candidate confidence = 0.58
constraint disagreement = high
```

Jev lại quyết định:

```text
CALL_CLAUDE = 0.82
```

→ Claude vào.

Đây là **adaptive specialization**, không phải rigid specialization.

---

# Điểm mình sẽ thay đổi so với Module 8 trước đó

Thay vì gọi:

```text
Codex Scout
Claude Investigator
```

như hai role cố định, mình sẽ định nghĩa:

```text
Codex
default_policy = exploration

Claude
default_policy = verification
```

Từ **role** thành **policy prior**.

Điểm này khá quan trọng về mặt paper.

Bạn có thể formalize:

$$
\pi_a(u \mid s)
$$

với:

- $a$: agent
- $u$: tool/action
- $s$: evidence state

Codex có prior cao hơn cho:

```text
search
list_videos
video_outline
constraint_probe
```

Claude có prior cao hơn cho:

```text
view_frames
compare_candidates
video_text
temporal_neighborhood
```

Nhưng **không action nào bị cấm**.

---

# Cách tốt nhất cho AIC hiện tại của bạn

Mình sẽ dùng kiến trúc này:

```text
                         Query
                           │
                    Main Retrieval
                           │
                     Evidence Board
                           │
                           ▼
                         Jev
                           │
             ┌─────────────┼─────────────┐
             │             │             │
           STOP       Codex first    Claude first
                           │             │
                           ▼             ▼
                    soft-role prompt
                           │
                     Evidence update
                           │
                           ▼
                         Jev
                           │
                 ┌─────────┴─────────┐
                 │                   │
               STOP          call other agent
```

Và vẫn giữ một mode:

```text
PARALLEL_BOTH
```

cho query khó.

Ví dụ query có:

- nhiều relation;
- nhiều event;
- multimodal cues;
- retrieval score rất sát nhau;
- TRAKE;
- candidate disagreement lớn.

Thì Jev có thể quyết định:

```text
Codex || Claude
```

giống hệ thống hiện tại.

---

# Thứ cần benchmark

Bạn nên làm ablation này vì nó vừa trả lời câu hỏi của bạn, vừa thành experiment rất đẹp cho SOICT:

| Variant | Prompt                                     |
| ------- | ------------------------------------------ |
| A       | Current same prompt                        |
| B       | Hard role split                            |
| C       | Shared core + soft specialization          |
| D       | Soft specialization + Jev adaptive routing |

Đo:

```text
R@1
R@5
rescue rate
agent disagreement
tool calls
latency
cost
```

Mình dự đoán thực tế:

```text
Hard split
→ giảm cost
→ có nguy cơ giảm recall/rescue

Same prompt
→ recall tốt
→ cost cao, duplicate search nhiều

Soft specialization
→ giữ gần hết recall
→ giảm duplicate work

Soft specialization + Jev
→ tốt nhất nếu router đủ calibrated
```

Và **nếu experiment cho thấy same-prompt vẫn tốt hơn**, bạn hoàn toàn có thể giữ same prompt. Contribution của paper **không phụ thuộc vào việc bắt buộc tách prompt**. Core contribution vẫn có thể là Jev adaptive controller + evidence-based stopping.

Nói ngắn gọn: **mình không khuyên tách Codex và Claude thành hai agent bị khóa chức năng. Giữ cùng capability, cùng shared core prompt, chỉ thêm bias chiến lược khác nhau.** Cách này an toàn hơn cho performance hiện tại và vẫn tạo được diversity có chủ đích cho paper.

---

# you asked

message time: 2026-10-03 02:29:24

module 9 có thật sự quan trọng không, vì nó chỉ là thêm phần hiển thị ở UI, làm rối UI khiến user khó theo dõi ?

---

# chatgpt response

Nếu module 9 bạn đang nói là **Evidence Board/hiển thị evidence state ra UI**, thì phần **hiển thị UI không quan trọng** và mình cũng không khuyên đưa nó thành một panel riêng.

Điểm cần giữ là **Evidence Board ở backend**, không phải UI.

Hiện tại bạn có thể làm:

```text
Query
  ↓
Main retrieval
  ↓
Codex / Claude
  ↓
Jev
```

nhưng nếu Jev không có một state chuẩn hóa thì mỗi vòng nó phải đọc lại kết quả thô, rất khó biết:

```text
candidate nào đã kiểm tra
constraint nào đã xác nhận
constraint nào còn thiếu
agent nào đã tìm candidate đó
tool nào đã dùng
candidate nào đã bị loại và vì sao
```

Evidence Board giải quyết chuyện đó:

```text
EvidenceBoard (backend only)

Query constraints:
  person = man
  clothes = yellow
  action = repairing
  object = bicycle

Candidate A
  visual_score      = 0.81
  Codex support     = true
  Claude support    = false

  person            = 0.97
  yellow            = 0.92
  bicycle           = 0.95
  repairing         = 0.41   ← unresolved

Candidate B
  ...
```

Jev nhìn state này rồi quyết định:

```text
repairing còn uncertain
→ inspect Candidate A thêm

hoặc

Candidate B evidence mạnh hơn
→ verify B

hoặc

đủ confidence
→ STOP
```

### Nhưng user không cần nhìn thấy tất cả thứ đó

UI hiện tại của AIC vốn đã có:

- main result;
- timeline;
- Agents strip;
- Codex;
- Claude;
- sticky;
- submission controls;
- TRAKE events;
- player.

Nếu thêm:

```text
Evidence Board
constraint matrix
agent confidence
Jev confidence
tool history
decision graph
```

thì đúng là rất dễ thành:

```text
              SEARCH UI

results       evidence
results       constraint
agents        reasoning
timeline      Jev
player        tool trace
submission    state
```

→ operator phải đọc quá nhiều thứ trong khi mục tiêu thật sự chỉ là **tìm frame đúng càng nhanh càng tốt**.

Mình sẽ không làm vậy.

---

## Kiến trúc nên là

```text
                   BACKEND

                   Query
                     ↓
              Query constraints
                     ↓
                Retrieval
                     ↓
              Evidence Board
                     ↓
                 Jev
              ↙       ↘
          Codex       Claude
              ↘       ↙
              Evidence Board
                     ↓
                 Jev
                     ↓
              final candidates


────────────────────────────────────

                    UI

              Main candidates

        Agent candidates / consensus

                Search status
```

Evidence Board chỉ là **machine state**.

Không phải operator interface.

---

# UI chỉ cần expose 2–3 signal

Ví dụ candidate card:

```text
L21_V083 — 02:17

Claude + Codex
High confidence

[Play] [Submit]
```

hoặc:

```text
L21_V083 — 02:17

Agent verified
Missing: action detail

[Inspect] [Submit]
```

Thậm chí `"Missing: action detail"` cũng có thể bỏ.

Một version còn đơn giản hơn:

```text
Agents

✓ Codex + Claude
L21_V083 / 02:17

✓ Claude
L22_V014 / 05:43
```

User không cần biết:

```text
Jev: P(candidate)=0.8733
constraint #4=0.442
action entropy=0.36
next-action score=...
```

Những thứ này để **logging/experiment**, không phải competition UI.

---

# Về paper cũng vậy

Reviewer không cần bạn có fancy Evidence UI.

Contribution nằm ở:

> internal evidence state enables adaptive agent routing and calibrated stopping.

Không phải:

> We introduce an evidence visualization panel.

Do đó paper architecture:

```text
Evidence State
     ↓
Jev Controller
     ↓
Agent action
     ↓
Evidence update
```

là đủ.

UI không phải contribution.

---

# Mình thậm chí sẽ chia thành 3 mức

### Production UI

Cực kỳ tối giản:

```text
candidate
agent source
confidence badge
```

### Debug mode

Có thể bật bằng:

```text
?debug_agent=true
```

hoặc developer toggle.

Hiện:

```text
Jev routing
tool calls
constraint scores
agent states
latency
```

### Experiment logging

Không cần UI:

```json
{
  "query_id": 17,
  "jev_decisions": [...],
  "codex_calls": 4,
  "claude_calls": 2,
  "constraints": {...},
  "latency_ms": 18320,
  "correct_rank": 1
}
```

Paper benchmark đọc log này.

---

# Nếu muốn tối giản thêm nữa

Bạn thậm chí có thể **không gọi nó là một module riêng** trong implementation.

Thay vì:

```text
Module 9: Evidence Board
```

hãy coi nó là:

```python
AgentRunState
```

bên trong controller:

```python
class AgentRunState:
    query
    constraints
    candidates
    evidence
    decisions
```

Tức là nó chỉ là một data structure hỗ trợ CAD-VR.

---

## Vậy module 9 có quan trọng không?

**UI Evidence Board: không quan trọng → mình sẽ bỏ.**

**Internal Evidence State: khá quan trọng → nên giữ.**

Nó giúp Jev/controller tránh trở thành một router stateless chỉ quyết định dựa trên một snapshot duy nhất.

Nếu bạn muốn kiến trúc gọn cho SOICT, mình sẽ rút hệ thống còn:

```text
1. Main multimodal retrieval

2. Query constraint compiler

3. Jev adaptive controller

4. Codex + Claude
   shared capabilities
   soft specialization

5. Evidence-aware verification

6. Adaptive stopping

7. Existing AIC UI
```

Không thêm panel mới, không làm UI rối lên. Đây theo mình là phiên bản hợp lý hơn cho cả **paper lẫn hệ thống thực tế**.

---

# you asked

message time: 2026-10-03 02:32:11

module 10 thêm tool vào thì có làm giảm độ chính xác so với tool đã có hiện tại không, liệu có reduancy không

---

# chatgpt response

Có. Nếu Module 10 được triển khai theo kiểu **thêm thẳng 5 tool mới vào MCP toolset hiện tại**, mình nghĩ có khả năng làm hệ thống yếu đi, dù từng tool riêng lẻ có vẻ hữu ích.

Lý do chính không phải tool “xấu”, mà là **tool-choice entropy** tăng. Hiện Codex/Claude đã có bộ tool khá rõ:

```text
search
list_videos
folder_frames
video_outline
video_frames
view_frames
video_text
report_candidate
```

Đây là một abstraction tương đối sạch: search → chọn video → inspect timeline → xem frame → đọc text → report.

Nếu thêm:

```text
candidate_evidence
temporal_neighborhood
compare_candidates
constraint_probe
inspect_candidate
```

thì rất nhiều action bắt đầu overlap nhau.

Ví dụ:

```text
inspect_candidate
≈ video_outline
 + video_frames
 + view_frames
 + video_text
```

và:

```text
temporal_neighborhood
≈ video_frames(start/end)
```

và:

```text
constraint_probe
≈ search(query=missing_constraint)
```

Khi đó agent phải reasoning thêm một bước:

> Tôi nên gọi `video_frames`, `temporal_neighborhood`, `inspect_candidate`, hay `candidate_evidence`?

Đó là overhead không có trong hệ thống hiện tại.

## Mình sẽ sửa Module 10

Không thêm cả 5 tool. Với repo của bạn, mình chỉ expose **1–2 tool mới thực sự orthogonal**.

| Đề xuất ban đầu      | Nên làm gì                        | Lý do                                            |
| ------------------------- | ------------------------------------ | ------------------------------------------------- |
| `candidate_evidence`    | **Backend only**               | Jev/controller dùng, LLM không cần gọi        |
| `temporal_neighborhood` | **Gộp vào `video_frames`** | Chức năng gần như đã có                    |
| `compare_candidates`    | **Giữ**                       | Hiện tại chưa có abstraction tương đương |
| `constraint_probe`      | **Có thể giữ**              | Có giá trị cho compositional query             |
| `inspect_candidate`     | **Không expose**              | Quá redundant với 4 tool hiện tại             |

Vậy toolset agent chỉ từ:

```text
8 tools
```

thành khoảng:

```text
9–10 tools
```

chứ không phải 13.

---

### `candidate_evidence` nên nằm bên trong controller

Ví dụ backend tự xây:

```text
Candidate A
├─ PE score
├─ Qwen score
├─ OCR hit
├─ speech hit
├─ agent support
└─ inspected evidence
```

rồi Jev đọc.

Codex/Claude không cần biết có tool này.

---

### `temporal_neighborhood` nên là option của `video_frames`

Thay vì:

```text
video_frames(...)
temporal_neighborhood(...)
```

sửa API hiện tại thành kiểu:

```python
video_frames(
    video_id,
    center_time=127.0,
    before=10,
    after=10,
    density="medium"
)
```

Một primitive là đủ.

---

### `inspect_candidate` cũng không cần tồn tại dưới dạng tool

Controller có thể tự chạy macro:

```text
video_outline
     ↓
video_frames
     ↓
video_text
```

nhưng LLM vẫn nhìn thấy các primitive hiện tại.

Điều này tốt hơn vì khi cần linh hoạt, agent vẫn có thể chọn:

```text
chỉ frame
hoặc
chỉ speech
hoặc
chỉ outline
```

thay vì `inspect_candidate` luôn tải tất cả.

---

# Tool mới mình thấy đáng giữ nhất: `compare_candidates`

Tool này giải quyết một failure mode mà tool hiện tại chưa giải quyết tốt.

Ví dụ:

```text
Candidate A
Candidate B
Candidate C
```

hiện Claude phải:

```text
view_frames(A)
view_frames(B)
view_frames(C)
```

rồi tự nhớ ba kết quả.

Thay bằng:

```text
compare_candidates(A, B, C)
```

trả một contact sheet:

```text
┌──────── A ────────┬──────── B ────────┬──────── C ────────┐
│ frame before      │ frame before      │ frame before      │
│ candidate         │ candidate         │ candidate         │
│ frame after       │ frame after       │ frame after       │
└───────────────────┴───────────────────┴───────────────────┘
```

Claude có thể trực tiếp trả lời:

> A có áo vàng nhưng không sửa xe.
> B có người đang sửa xe nhưng áo trắng.
> C thỏa cả hai.

Đây là tool **không redundant**, vì nó thay đổi unit of reasoning từ:

```text
inspect one candidate
```

sang:

```text
discriminate between competing candidates
```

Rất hợp với verifier.

---

# `constraint_probe` thì chỉ nên thêm nếu thiết kế đúng

Không nên làm:

```python
constraint_probe("yellow shirt")
```

rồi bên trong chỉ gọi:

```python
search("yellow shirt")
```

vì thế thì hoàn toàn redundant.

Nó chỉ đáng tồn tại nếu semantics là:

> kiểm tra **một constraint cụ thể trong phạm vi candidate/video hiện tại**.

Ví dụ:

```python
constraint_probe(
    video_id="L21_V083",
    around_time=137.2,
    constraint="the man is repairing the bicycle"
)
```

backend có thể lấy:

```text
±15 sec frames
OCR
speech
neighboring keyframes
```

rồi trả evidence tập trung vào constraint đó.

Như vậy:

```text
global search
```

và:

```text
local evidence probe
```

là hai operation khác nhau.

---

# Quan trọng hơn: Jev cũng không nên thấy toàn bộ low-level tool space

Kiến trúc tốt hơn là:

```text
                   Jev
                    │
          chooses ACTION CLASS
                    │
      ┌─────────────┼──────────────┐
      ↓             ↓              ↓
   SEARCH        INSPECT        COMPARE
      │             │              │
      ▼             ▼              ▼
 existing       existing       compare_
 search          tools         candidates
```

Jev quyết định:

```text
SEARCH_MORE
INSPECT_TOP1
COMPARE_TOP2
CALL_CODEX
CALL_CLAUDE
STOP
```

sau đó controller map action xuống tool cụ thể.

Jev không cần chọn giữa:

```text
video_frames
view_frames
video_text
video_outline
temporal_neighborhood
inspect_candidate
```

Điều đó giảm action-space đáng kể.

---

# Có nguy cơ giảm accuracy thật không?

Có 3 loại nguy cơ.

**Một là tool confusion.** Hai tool gần giống nhau khiến agent chọn sai abstraction hoặc gọi dư.

**Hai là correlated evidence.** Nếu `candidate_evidence`, `inspect_candidate`, `constraint_probe` thực chất đều đọc cùng frame/OCR, hệ thống có vẻ có “nhiều bằng chứng”, nhưng thực ra là cùng một bằng chứng được đóng gói ba lần. Jev có thể trở nên overconfident nếu không xử lý source provenance.

**Ba là macro quá mạnh.** Một tool như `inspect_candidate` trả quá nhiều dữ liệu một lần có thể khiến Claude/Codex bị distraction, trong khi tool primitives hiện tại cho phép agent zoom dần.

Đó là lý do mình sẽ ưu tiên **small orthogonal toolset**.

---

## Kiến trúc mình chốt lại cho AIC

Agent-facing tools:

```text
search
list_videos
folder_frames
video_outline
video_frames
view_frames
video_text

compare_candidates       ← NEW
constraint_probe         ← NEW, optional

report_candidate
```

Backend-only:

```text
EvidenceState
candidate_evidence
tool history
constraint scores
Jev state
```

Và **không có**:

```text
inspect_candidate
temporal_neighborhood
```

dưới dạng top-level MCP tool.

Mình nghĩ đây tốt hơn Module 10 ban đầu: **không cố tăng số lượng tool để tạo novelty**. Novelty của paper nằm ở *Jev adaptive controller + calibrated escalation + constraint-aware verification*, không phải “chúng tôi có nhiều tool hơn”.

Thậm chí ablation `current 8 tools` vs `+ compare` vs `+ compare + constraint_probe` sẽ rất đáng làm. Nếu tool mới không cải thiện Recall/Rescue Rate hoặc chỉ tăng latency, bỏ nó khỏi final system luôn.

---

# you asked

message time: 2026-10-03 02:37:11

module 11 nó có làm tăng latency không, có thưc sự cần thiết không

---

# chatgpt response

Có, **Module 11 — disagreement-driven verification** có thể làm tăng latency nếu bạn biến nó thành một bước bắt buộc sau mỗi query. Và với hệ thống hiện tại, mình không xem nó là module cốt lõi bắt buộc.

Điểm quan trọng là phân biệt hai thứ:

- **phát hiện disagreement**: gần như miễn phí, chỉ so candidate/result đã có.
- **xử lý disagreement bằng thêm một vòng agent/tool call**: đây mới là phần tăng latency.

Ví dụ hiện tại:

```text
Codex ─┐
       ├─> candidates
Claude ┘
```

Bạn chỉ cần kiểm:

```text
Codex top1 == Claude top1 ?
```

hoặc mềm hơn:

```text
same video?
same temporal neighborhood?
confidence gap?
```

Phần này gần như không đáng kể.

Nhưng nếu Module 11 làm:

```text
Codex = A
Claude = B
    ↓
compare_candidates(A, B)
    ↓
Claude verify again
    ↓
Jev verify again
```

thì rõ ràng latency tăng.

---

## Mình sẽ không để Module 11 chạy mặc định

Thay vì:

```text
Codex + Claude
      ↓
disagreement verification
      ↓
final
```

hãy để:

```text
Codex + Claude
      ↓
agreement?
  ┌───────┴───────┐
 yes              no
  │                │
 STOP         check uncertainty
                    │
              ┌─────┴─────┐
             low          high
              │            │
             STOP       verify
```

Tức là disagreement chỉ là **trigger signal**.

---

# Khi nào disagreement đáng verify?

Ví dụ:

```text
Codex: L21_V083 / 132s
Claude: L21_V083 / 136s
```

Thực chất hai agent đã agree ở video/moment.

Không cần verify thêm.

Hoặc:

```text
Codex: A confidence 0.91
Claude: B confidence 0.34
main retrieval: A
Jev: A = 0.94
```

Cũng không cần.

Chỉ khi kiểu:

```text
Codex → A 0.78
Claude → B 0.81

main retrieval:
A = 0.73
B = 0.72

Jev:
A = 0.52
B = 0.48
```

thì mới đáng gọi:

```text
compare_candidates(A, B)
```

Đây là genuinely ambiguous query.

---

# Có thực sự cần Module 11 không?

### Cho hệ thống production hiện tại

**Không bắt buộc.**

Hệ thống của bạn đã có:

```text
main retrieval
+
Codex
+
Claude
+
Jev controller
```

Đã khá nhiều lớp rồi.

Nếu mục tiêu là:

> giữ hệ thống nhanh, ít bug, ít complexity

thì Module 11 có thể bỏ khỏi MVP.

---

### Cho paper SOICT

Nó có ích nhưng nên được xem là **optional policy**, không phải contribution chính.

Contribution chính nên vẫn là:

```text
Jev adaptive routing
+
evidence-aware stopping
+
adaptive System-2 escalation
```

Disagreement chỉ là một feature đầu vào cho Jev:

```text
state = {
    retrieval_margin,
    codex_candidate,
    claude_candidate,
    agent_agreement,
    evidence_coverage
}
```

Sau đó Jev quyết định:

```text
STOP
COMPARE
SEARCH_MORE
CALL_AGENT
```

Như vậy bạn không cần gọi nó là “Module 11” nữa.

Nó trở thành:

> **one uncertainty signal inside the controller**

Thiết kế này gọn hơn rất nhiều.

---

# Mình sẽ sửa architecture thành

Thay vì:

```text
Module 11:
Disagreement-driven verification
```

hãy đưa thẳng vào Jev controller:

```text
                     Evidence State
                           │
             ┌─────────────┴─────────────┐
             │                           │
        agreement                    disagreement
             │                           │
             └─────────────┬─────────────┘
                           ↓
                          Jev
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
         STOP           COMPARE         ESCALATE
```

Jev có thể nhận:

```text
agent_agreement = 1
```

hoặc:

```text
agent_agreement = 0
candidate_overlap = 0.21
top_score_margin = 0.03
```

và tự quyết định.

---

# Một vấn đề nữa: tránh chạy cả hai agent chỉ để có disagreement signal

Đây là chỗ rất quan trọng.

Nếu Jev ban đầu quyết định:

```text
run Codex only
```

thì **đừng gọi thêm Claude chỉ để xem hai con có disagree hay không**.

Nếu làm vậy thì phá luôn mục tiêu latency/cost của CAD-VR.

Disagreement chỉ sử dụng khi:

```text
Claude và Codex vốn đã được chạy
```

hoặc khi Jev đã quyết định:

```text
BOTH
```

Không nên có flow:

```text
Codex result
→ need disagreement signal
→ call Claude
```

trừ khi uncertainty đã đủ cao để Claude đáng được gọi dù có Module 11 hay không.

---

# Vì vậy mình sẽ đánh giá các module hiện tại thế này

| Module                           |                          Nên giữ? | Vai trò                  |
| -------------------------------- | ----------------------------------: | ------------------------- |
| Jev controller                   |                       **Có** | core contribution         |
| Soft specialization Codex/Claude |                       **Có** | giảm duplicate work      |
| Internal Evidence State          |                       **Có** | state cho controller      |
| Tool expansion                   |            **Rất hạn chế** | chỉ giữ tool orthogonal |
| Disagreement verification        | **Không cần module riêng** | biến thành signal       |
| UI evidence panel                |                    **Không** | unnecessary               |

Architecture sẽ gọn lại thành:

```text
Query
  ↓
Main Retrieval
  ↓
Evidence State
  ↓
Jev
  │
  ├─ STOP
  ├─ Codex
  ├─ Claude
  └─ BOTH
       ↓
 Evidence update
       ↓
 Jev evaluates
  ├─ STOP
  ├─ inspect
  ├─ compare     ← only if necessary
  └─ escalate
```

Mình thích phiên bản này hơn đáng kể so với đề xuất ban đầu. **Module 11 không nên tồn tại như một stage luôn chạy**; chỉ giữ `agent disagreement` như một biến uncertainty cho Jev. Như vậy gần như không tăng latency ở case bình thường, nhưng vẫn giữ được khả năng xử lý những query khó thật sự.

Có khác biệt đáng kể về **cách agent hoạt động**, nhưng nếu đem pipeline hiện tại đi viết paper với contribution chỉ là “dùng Codex + Claude agent để search video bằng tools” thì mình chưa xem là đủ an toàn về novelty so với OpenCubee_1 năm 2025.

Mình vừa đối chiếu poster bạn chụp với mô tả paper chính thức. Paper của OpenCubee_1 đúng là **“Towards Conversational Video Retrieval with an Intelligent Search Agent”**, trình bày tại SOICT 2025; hệ thống dùng một intelligent agent để hiểu intent, duy trì hội thoại, chọn modality, reasoning và refine search. [SOICT 2026](https://soict.org/wp-content/uploads/2025/12/SOICT2025-ProgramBook.pdf?utm_source=chatgpt.com)

Điểm quan trọng là phải phân biệt **“khác implementation”** và **“khác research contribution”**.

## So trực tiếp hai hệ thống

| Thành phần                 | OpenCubee_1 / ConvAgent 2025                                         | AIC 2026 hiện tại của bạn                                                                                                                     |
| ---------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Agent chính                 | Một conversational/intelligent agent, poster ghi Gemini LLM         | **Hai agent độc lập: Codex + Claude**                                                                                                    |
| Vai trò chính              | Hiểu intent + hội thoại + chọn modality + refine search          | **Chủ động điều tra corpus và tìm candidate**                                                                                        |
| Search workflow              | Agent điều khiển multimodal retrieval                             | Agent có thể**iteratively dùng tools**                                                                                                   |
| Multimodal                   | semantic + OCR + ASR + object                                        | PE/Qwen + OCR + speech + audio + TARA + RRF                                                                                                       |
| Tool use                     | Poster thể hiện các retrieval modalities                          | MCP có`search`, `list_videos`, `folder_frames`, `video_outline`, `video_frames`, `view_frames`, `video_text`, `report_candidate` |
| Corpus browsing              | Không thấy nhấn mạnh autonomous hierarchical browsing            | **Có:** folder → video → outline → temporal window → frame                                                                             |
| Visual inspection            | Retrieval modules                                                    | Agent trực tiếp inspect contact sheet/frame                                                                                                     |
| Multi-agent                  | Không                                                               | **Codex + Claude song song**                                                                                                                |
| Relation với main retrieval | Agent là mediator của search                                       | **Sidecar hoàn toàn tách biệt**, main retrieval không chờ agent                                                                       |
| Failure isolation            | Không phải điểm paper nhấn mạnh                                | Agent fail/timeout không làm hỏng main search                                                                                                  |
| User feedback                | **Rất mạnh:** conversational refinement, P&P feedback fusion | Không phải trọng tâm agent hiện tại                                                                                                         |
| Multi-turn context           | **Có, là contribution rõ ràng**                            | Agent run hiện tại gần như query-scoped                                                                                                       |
| Agent candidate              | Agent điều phối retrieval                                         | Agent của bạn có thể tự report exact candidate frame                                                                                         |
| Tool-level reasoning         | Có dynamic modality selection                                       | **Có iterative search/browse/inspect**                                                                                                     |

OpenCubee mô tả agent như lớp **mediator giữa user và retrieval engine**. Paper nói agent duy trì conversational context, hiểu user intent và liên kết intent với multimodal/temporal retrieval; plug-and-play feedback fusion dùng cho refinement. [EurekaMag](https://eurekamag.com/research/110/181/110181092.php?utm_source=chatgpt.com)

Trong khi agent hiện tại của bạn gần với:

> **autonomous retrieval investigator**

hơn là conversational mediator.

Đây là khác biệt thật.

---

## Một ví dụ cho thấy khác biệt

### ConvAgent

Theo poster của bạn, flow gần như:

```text
User
 ↓
Gemini Agent
 ├─ classify intent
 ├─ choose modalities
 │   ├─ Semantic
 │   ├─ OCR
 │   ├─ ASR
 │   └─ Objects
 ↓
Search
 ↓
Results
 ↓
user feedback
 ↓
refine search
```

Nó rất mạnh ở:

> **conversation → intent → retrieval**

### AIC 2026 của bạn

Agent sidecar hiện tại gần hơn với:

```text
                  Query
                    │
       ┌────────────┴────────────┐
       ↓                         ↓
Main retrieval              Agent search
PE/Qwen/OCR/...        ┌─────────┴─────────┐
                      ↓                   ↓
                    Codex               Claude
                      │                   │
                      ├ search            ├ search
                      ├ list_videos       ├ browse
                      ├ video_outline     ├ inspect
                      ├ video_frames      ├ frames
                      ├ video_text        ├ text
                      └ report            └ report
```

Agent có khả năng kiểu:

```text
"Search results không ổn"
       ↓
browse folder
       ↓
find plausible video
       ↓
read video outline
       ↓
narrow to minute 04:00–05:00
       ↓
look at contact sheet
       ↓
inspect selected frames
       ↓
report exact candidate
```

Đây không còn đơn giản là chọn:

```text
OCR + ASR + semantic
```

nữa.

---

# Nhưng có một vấn đề lớn

Nếu abstract SOICT 2026 của bạn viết:

> “We introduce an intelligent LLM agent for multimodal video retrieval that dynamically uses visual, OCR, speech and temporal information.”

thì reviewer từng đọc paper OpenCubee có thể phản ứng ngay:

> “Cái này SOICT năm ngoái đã làm rồi.”

Vì những từ khóa sau gần như overlap trực tiếp:

```text
LLM agent
multimodal retrieval
dynamic search
temporal reasoning
OCR
ASR
video retrieval
```

Thậm chí poster OpenCubee ghi rất rõ:

> “Autonomously determines the optimal combination of modalities…”

và paper chính thức nhấn mạnh conversational refinement + multimodal/temporal modules. [EurekaMag](https://eurekamag.com/research/110/181/110181092.php?utm_source=chatgpt.com)

Cho nên **đừng lấy “agent tích hợp vào retrieval” làm contribution trung tâm**.

---

# Multi-agent Codex + Claude có đủ khác biệt không?

Khác, nhưng **chưa đủ mạnh một mình**.

Bạn có thể nói:

```text
OpenCubee:
1 LLM agent → orchestrates retrieval modules

AIC:
2 autonomous general-purpose agents
→ independently browse/search/inspect corpus
→ return grounded frame candidates
```

Đó là architectural difference.

Nhưng reviewer có thể hỏi:

> “Tại sao hai agent tốt hơn một agent?”

Nếu paper chỉ đưa:

```text
Gemini → Codex + Claude
```

thì đây chủ yếu là model/implementation substitution.

Bạn cần một **algorithmic/system contribution** phía trên nó.

---

# Đây là lý do Jev Controller làm paper khác hẳn OpenCubee

Nếu đi theo kiến trúc mà mình và bạn vừa tinh giản qua các module trước:

```text
                    Query
                      ↓
             multimodal retrieval
                      ↓
                Evidence State
                      ↓
                 Jev 1.13
          calibrated decision layer
             ↙        ↓        ↘
           STOP     Codex     Claude
                      ↓
                evidence update
                      ↓
                    Jev
              ↙       ↓       ↘
            STOP    inspect   escalate
```

thì comparison thay đổi rất nhiều.

### OpenCubee

```text
Generative LLM
      ↓
understands intent
      ↓
selects modalities
      ↓
retrieval
      ↓
user feedback
```

### Hướng SOICT của bạn

```text
Retrieval produces evidence
          ↓
Non-generative decision model
          ↓
estimate whether reasoning is necessary
          ↓
select expensive System-2 agent/action
          ↓
agent gathers additional evidence
          ↓
decision model decides:
STOP / continue / escalate
```

Research question cũng hoàn toàn khác.

OpenCubee hỏi:

> **How can an intelligent conversational agent mediate video retrieval?**

Bạn hỏi:

> **When is expensive agentic reasoning actually necessary, and how can a calibrated fast decision layer adaptively allocate it?**

Đây mới là distinction mình thấy đủ rõ.

---

# Một khác biệt rất hay nữa: OpenCubee phụ thuộc user refinement

Poster có cả:

```text
SEARCH REFINEMENT
Plug-and-Play Feedback Fusion
conversation
user intent evolution
```

Tức user nằm trong loop:

```text
user
↓
search
↓
feedback
↓
search
```

Hệ thống agent bạn đang hướng tới có thể **tự đóng loop**:

```text
query
 ↓
retrieve
 ↓
agent sees uncertainty
 ↓
search additional evidence
 ↓
verify
 ↓
search again
 ↓
stop
```

Không cần user nói:

> “không đúng, thử tìm người mặc áo vàng.”

Agent tự thấy constraint `"yellow shirt"` chưa được chứng minh và tiếp tục tìm.

Đây là distinction rất đẹp để viết Introduction.

---

# Tuy nhiên có một vùng overlap cần cẩn thận

Đừng claim:

> “Our agent dynamically selects retrieval modalities.”

Bởi vì OpenCubee đã làm đúng cái này.

Đừng claim:

> “Our agent reasons temporally.”

Cũng đã có.

Đừng claim:

> “Our agent integrates external knowledge.”

OpenCubee cũng có QA/knowledge integration.

Đừng claim:

> “Our agent provides multimodal intelligent search.”

Quá gần.

---

# Nên claim cái gì?

Mình sẽ xoay contribution SOICT của bạn thành ba điểm rõ ràng:

**1. Calibrated Fast–Slow Agent Routing**

Không phải mọi query đều khởi động expensive agents.

```text
easy → retrieval → STOP

medium → retrieval → Codex → STOP

hard → retrieval → Codex → Claude → STOP
```

---

**2. Evidence-conditioned autonomous search**

Agent không chỉ chọn modality.

Agent có khả năng:

```text
retrieve
→ browse
→ temporally inspect
→ compare evidence
→ refine search
→ ground exact candidate
```

và action tiếp theo phụ thuộc evidence đã thu thập.

---

**3. Adaptive stopping under latency/cost constraints**

Đây là chỗ mình thấy rất phù hợp với hệ thống competition của bạn:

\[
\text{maximize retrieval quality}
\]

subject to:

\[
\text{latency},\quad
\text{LLM calls},\quad
\text{tool calls},\quad
\text{cost}.
\]

OpenCubee không đặt bài toán chính theo hướng **allocate agent reasoning budget**.

---

# Nếu chưa có Jev thì sao?

Nếu xét **repo ngay lúc này**, chưa có những nâng cấp chúng ta vừa bàn:

```text
main retrieval
+
Codex
+
Claude
+
MCP tools
```

thì đánh giá của mình là:

**Khác implementation: khá rõ.**

**Khác architecture: có.**

**Khác đủ để paper claim một contribution mạnh: chưa chắc.**

Vì umbrella concept vẫn là:

```text
LLM agent(s)
        +
multimodal video retrieval
```

mà OpenCubee đã publication ở SOICT 2025. Paper của họ xuất hiện trong proceedings ở trang 468–482. [Springer](https://link.springer.com/book/10.1007/978-981-92-2587-3?page=2\&utm_source=chatgpt.com)

---

# Nếu hoàn thiện Jev architecture

Thì mình đánh giá khác:

```text
OpenCubee 2025
Conversational agent
        │
        ▼
Retrieval orchestration
```

versus:

```text
Your SOICT 2026
Multimodal retrieval
        │
        ▼
Calibrated System-1 Controller
        │
   adaptive routing
        │
        ▼
System-2 autonomous retrieval agents
        │
        ▼
evidence-driven stopping
```

Lúc đó **research question, control mechanism và evaluation đều khác**.

Và quan trọng là experiment phải chứng minh:

```text
Current Codex + Claude always-on
vs
Jev adaptive controller
```

không chỉ accuracy mà cả:

```text
R@1 / R@5
Agent Rescue@K
time-to-correct
p50/p95 latency
agent invocations/query
tool calls/query
cost/query
% solved without System-2
```

Nếu bạn chứng minh được ví dụ **giữ hoặc tăng retrieval quality nhưng giảm đáng kể số lần chạy Codex/Claude và latency**, thì câu chuyện paper khác OpenCubee rất rõ.

Tóm lại: **pipeline agent hiện tại của bạn đã không phải bản copy hay biến thể trực tiếp của ConvAgent**, vì autonomous tool-driven dual agents khác conversational modality orchestrator. Nhưng mình sẽ **không nộp paper chỉ dựa trên khác biệt đó**. Jev fast–slow controller + evidence-conditioned escalation/stopping mới là phần biến khác biệt engineering thành một **research contribution có thể defend trước reviewer SOICT đã biết paper OpenCubee 2025**. [EurekaMag](https://eurekamag.com/research/110/181/110181092.php?utm_source=chatgpt.com)

Có. Sau khi đọc **commit mới nhất `71c91888`**, mình nghĩ **nên nâng cấp benchmark trước khi chạy tốn quota lớn**, và progressive-hint benchmark là bổ sung đáng làm nhất hiện tại.

Lý do là commit mới vừa làm dataset mạnh hơn rõ rệt: thêm 30 câu đã review gồm 15 T-KIS, 12 QA, 3 TRAKE, rồi merge thành **111 query = 25 dev / 86 test**, đồng thời group split để không còn target-video overlap giữa dev/test.

Quan trọng hơn: các câu T-KIS/QA mới đã lưu **`hints_vi` riêng từng hint**. Ví dụ query 13 có đủ 4 hint và còn được audit từng hint bằng frame/ASR, nên dữ liệu hiện tại thực sự hỗ trợ experiment progressive hints chứ không phải tự cắt query một cách tùy tiện.

Trong khi benchmark hiện tại **không dùng `hints_vi`**: nó gửi nguyên `query["query"]` và ép `previous_hints=[]`. Với các câu mới, `query` về cơ bản là các hint ghép lại, tức benchmark hiện nay gần như đang đo **Full Hint only**.

## Mình sẽ giữ benchmark hiện tại và thêm một benchmark thứ hai

Không thay A–F. Main table vẫn là:

```text
Full query
────────────────────────────────
A Retrieval
B + Codex
C + Codex + Claude
D Jev rerank
E Adaptive
F Full CAD-VR
```

Đây vẫn là bảng chứng minh contribution architecture.

Sau đó thêm:

> **Progressive Hint Retrieval Benchmark**

với các stage:

```text
H1       = hint 1
H2       = hint 1 + hint 2
H3       = hint 1 + hint 2 + hint 3
...
Hfull    = toàn bộ hints
```

Paper có thể chỉ show:

```text
H1 | H2 | Full
```

cho gọn, còn supplementary giữ toàn bộ curve.

---

## Đây mới thực sự cho thấy sức mạnh agent

Ví dụ một query có 4 hints:

```text
H1:
"Cảnh một lễ hội, mọi người nấu món ăn bằng chảo lớn..."

H2:
+ nguyên liệu được đảo bằng mái chèo gỗ

H3:
+ hỗn hợp màu vàng được đổ vào

H4:
+ thức ăn được chia cho mọi người
```

Có thể xảy ra:

```text
                    H1      H1+H2     FULL

Retrieval A         ✗         ✗        ✓

Always agents C     ✗         ✓        ✓

CAD-VR F            ✓         ✓        ✓
```

Kết quả này mạnh hơn rất nhiều so với chỉ:

```text
Full query:
A = 0.55
F = 0.72
```

Bởi vì nó chứng minh:

> **CAD-VR can resolve targets under incomplete evidence and requires fewer hints than conventional retrieval.**

Đây là một result rất phù hợp với agent system.

---

# Nhưng có một điều rất quan trọng: không dùng `previous_hints`

Mình không khuyên benchmark kiểu:

```text
Run H1
 ↓
giữ EvidenceBoard
 ↓
thêm H2
 ↓
giữ state
 ↓
thêm H3
```

Như vậy bạn sẽ đồng thời benchmark **cross-hint memory**, làm khó tách contribution.

Thay vào đó mỗi stage phải là **fresh independent run**:

```text
H1
→ new run
→ previous_hints=[]

H1+H2
→ new run
→ previous_hints=[]

H1+H2+H3
→ new run
→ previous_hints=[]
```

Như vậy ta đang đo:

> **How much information does CAD-VR require to solve a query?**

chứ không đo memory.

Điều này cũng hoàn toàn phù hợp với current CAD-VR implementation vốn cố tình query-local và benchmark runner đang ép `previous_hints=[]`.

---

# Metrics nên thêm

Mình sẽ không chỉ tạo ba cột R@1.

### 1. R@1 / R@5 theo số hint

Figure chính:

```text
R@1
1.0 │                         F ────────●
    │                  F ●────
0.8 │
    │                       C ●─────────
0.6 │            C ●
    │                               A ●
0.4 │
    │                 A ●
0.2 │ A ●
    └────────────────────────────────────
          H1       H2       H3       Full
```

Đây có thể là một trong các figure tốt nhất paper.

---

### 2. Minimum Hints to Solve — rất nên có

Với mỗi query:

\[
h^* = \min\{h:\text{target is Rank-1 after }h\text{ hints}\}
\]

Sau đó report:

```text
Mean hints-to-R@1 ↓
Median hints-to-R@1 ↓
```

Ví dụ:

| Method       | Mean hints to solve ↓ |
| ------------ | ---------------------: |
| Retrieval    |                   3.21 |
| Codex+Claude |                   2.47 |
| CAD-VR       |         **1.74** |

Nếu kết quả ra như vậy thì story rất dễ hiểu.

---

### 3. Early Solve Rate

Một metric mình đặc biệt thích:

```text
Solved before full hint
```

Ví dụ:

```text
A       28%
C       51%
F       72%
```

Paper có thể nói:

> CAD-VR resolves X% of queries before receiving the complete description.

Rất trực quan.

---

### 4. Hint Savings

Nếu query có \(H\) hints và solve tại \(h^*\):

\[
\text
=====

\frac{H-h^*}{H-1}
\]

Ví dụ 4 hints mà solve ở H1:

```text
saving = 100%
```

solve H2:

```text
saving = 66.7%
```

chỉ solve full:

```text
saving = 0%
```

Report mean hint saving.

---

### 5. Cost/latency theo information level

Cái này cực kỳ quan trọng với CAD-VR:

```text
                 H1      H2     Full
Agents/query
Tool calls
Latency
Jev calls
```

Có một hypothesis rất hay:

```text
ít hint
→ uncertainty cao
→ CAD-VR gọi System-2 nhiều hơn

nhiều hint
→ retrieval/evidence rõ
→ Jev STOP sớm hơn
→ agent calls giảm
```

Nếu thực nghiệm đúng, bạn sẽ có một figure rất mạnh:

```text
More information
      ↓
Less agent computation
```

Nó trực tiếp hỗ trợ adaptive-routing contribution.

---

# Tuy nhiên có một vấn đề evaluation cần xử lý cẩn thận

Đây là điểm mình thấy quan trọng nhất trước khi code.

Một số hint đầu chỉ mô tả **video**, chưa chỉ đúng **moment ground truth**.

Ví dụ:

```text
H1:
Tin về một nghệ nhân làm bánh cưới ở Pháp

H2:
hai người đang tạo nụ hoa

H3:
cận cảnh nụ hoa trên que trắng

H4:
khuôn xanh dương...
```

Ground truth hiện tại có thể là moment của H3/H4.

Nếu ở H1 hệ thống tìm:

```text
đúng video
nhưng moment khác
```

mà bạn tính strict moment R@1 = 0 thì hơi bất công.

Commit mới cũng cho thấy ground truth là các submitted points/accepted moments, chứ **không phải mỗi hint đều có một stage-specific accepted interval**. Một số audit thậm chí nói target là seed cho một đoạn khớp chứ không phải một frame riêng chứng minh toàn bộ hints.

Vì vậy progressive benchmark nên report **hai tầng**.

### Early hints

Ưu tiên:

```text
Video R@1
Video R@5
```

và secondary:

```text
Moment R@1
Moment R@5
```

### Full hint

Dùng metric strict hiện tại:

```text
Moment R@1
Moment R@5
MRR

QA: correct moment + accepted answer
TRAKE: complete ordered sequence
```

Như vậy không inflate cũng không unfair.

---

# Nếu muốn cực kỳ chặt chẽ

Cách mạnh nhất là thêm:

```json
"hint_targets": {
  "1": [...],
  "2": [...],
  "3": [...],
  "full": [...]
}
```

tức annotate accepted moment riêng cho từng hint.

Nhưng mình **không khuyên làm ngay** vì 27 câu × nhiều hint sẽ tốn rất nhiều manual annotation.

Cho SOICT hiện tại:

> Partial hint → video-level + moment-level secondary
> Full hint → strict current metric

là đủ hợp lý nếu methodology viết rõ.

---

# Benchmark nào nên chạy progressive?

Không cần nhân toàn bộ A–F × mọi hint. Sẽ đốt quota rất mạnh.

Mình chọn:

| Variant     | Mục đích                  |
| ----------- | ---------------------------- |
| **A** | retrieval baseline           |
| **C** | always Codex+Claude baseline |
| **F** | proposed CAD-VR              |

Có thể thêm **E** nếu muốn chứng minh constraint-level verification thực sự giúp under sparse hints.

Tức test:

```text
A vs C vs F
```

là đủ cho main progressive experiment.

Nếu có khoảng 27 query có structured hints và trung bình 3 stages:

```text
27 queries
× 3 stages
× 3 systems
≈ 243 runs
```

vẫn tương đối lớn nhưng hợp lý hơn:

```text
27 × 4 stages × 6 variants = 648 runs
```

---

# Main benchmark mới của paper mình sẽ tổ chức như này

### Table 1 — Main retrieval performance

Toàn bộ **111 queries**:

```text
A B C D E F

R@1
R@5
MRR
Rescue@1
Latency
Agents/query
```

Latest commit đã đưa bộ benchmark lên 111 query với 25 dev / 86 test và loại target-video overlap dev/test, nên đây nên là dataset chính.

---

### Table 2 — Progressive Hint Evaluation

Chỉ subset có `hints_vi`:

```text
              H1        H2       FULL
A
C
F
```

Metrics:

```text
Video R@1
Moment R@1
Early Solve
Mean Hints-to-Solve
Agents/query
Latency
```

---

### Table 3 — Ablation

DEV only:

```text
F
rule_router
llm_router
no_verifier
base_tools
compare_only
no_roles
```

---

### Figure 1

```text
R@1 vs #Hints
```

### Figure 2

```text
Agent calls/query vs #Hints
```

Hai figure này gần như kể được toàn bộ câu chuyện:

> Khi evidence ít, CAD-VR dùng agent reasoning để bù thông tin; khi evidence tăng, controller giảm System-2 usage.

Đây chính xác là điều **adaptive** architecture nên chứng minh.

---

## Mình sẽ sửa benchmark harness như sau

Không cần sửa agent code. Chỉ `benchmarks/agent/`.

Thêm option:

```bash
--hint-mode full
--hint-mode progressive
```

và helper:

```python
def hint_stages(q):
    hints = q.get("hints_vi") or []

    if not hints:
        return [("full", q["query"])]

    return [
        ("h1", hints[0]),
        ("h2", "\n".join(hints[:2])),
        ...
        ("full", "\n".join(hints))
    ]
```

Mỗi result thêm:

```json
{
  "base_query_id": "...",
  "hint_stage": 2,
  "hint_count": 4,
  "hint_fraction": 0.5
}
```

Manifest phải hash luôn:

```text
hint_mode
hint construction rule
hints_vi
```

để không resume lẫn experiment cũ.

Output thêm:

```text
summary_by_hint.json
progressive_report.md
```

---

## Có một validation mình sẽ bắt buộc

Vì `query` hiện tại thường được tạo từ `hints_vi`, exporter phải check:

```text
normalize(join(hints_vi))
≈
normalize(query)
```

Nếu khác thì phải audit chứ không âm thầm thay query.

Commit mới đã có một ví dụ lý do rất rõ: query 13 phát hiện **English translation sai hai chi tiết**, nên benchmark cố tình dùng bản tiếng Việt đã review.

Progressive benchmark cũng phải dùng chính `hints_vi` tiếng Việt đã audit, không reconstruct từ bản English.

---

# Có nên làm ngay trước khi benchmark không?

**Có, nhưng không thay main benchmark.**

Mình sẽ làm:

```text
Main A–F, full query
        +
Progressive A/C/F, hint subset
```

chứ không:

```text
thay toàn bộ benchmark thành progressive
```

Main A–F trả lời:

> CAD-VR có tốt hơn baseline không?

Progressive trả lời một câu hỏi khác và rất giá trị:

> CAD-VR có thể tìm đúng với bao nhiêu thông tin, và agent reasoning có giúp khi query chưa đầy đủ không?

Đặc biệt với OpenCubee 2025 mà chúng ta vừa so, result này còn giúp paper của bạn **khác rõ hơn**: không chỉ “LLM agent cho video search”, mà là **adaptive fast–slow agentic retrieval under varying evidence availability**.

Nếu progressive curve cho thấy `F` vượt `A/C` mạnh ở **H1/H2** nhưng khoảng cách thu hẹp ở Full, đó gần như là result lý tưởng để chứng minh agent thực sự mang lại giá trị — thay vì chỉ hưởng lợi từ một query đầy đủ và rất chi tiết.

Đúng. Ý mình là **benchmark theo mức thông tin tăng dần**, hoàn toàn **không dùng thuật toán Progressive Hint/Progressive Memory đã submit VBS**.

Cụ thể, mỗi mức là **một run độc lập**:

```text
H1:
query = hint 1
→ fresh CAD-VR run

H2:
query = hint 1 + hint 2
→ fresh CAD-VR run

H3:
query = hint 1 + hint 2 + hint 3
→ fresh CAD-VR run

Full:
query = hint 1 + ... + hint N
→ fresh CAD-VR run
```

Giữa các run phải đảm bảo:

```text
previous_hints = []
EvidenceBoard = new
agent history = empty
Jev state = new
tool cache = query-local/new run
không truyền candidate H1 → H2
không truyền reasoning H1 → H2
không có cross-hint memory
```

Tức H2 **không biết H1 đã từng được chạy**. Nó chỉ nhận một query dài hơn.

### Đây chỉ là evaluation protocol

Nó tương tự việc test model ở nhiều mức input information:

\[
Q_1=h_1
\]

\[
Q_2=h_1+h_2
\]

\[
Q_3=h_1+h_2+h_3
\]

rồi đo độc lập:

\[
R@1(Q_1),R@1(Q_2),R@1(Q_3)
\]

Không có thuật toán:

```text
search H1
↓
remember result
↓
receive H2
↓
update prior search
↓
reuse memory
```

Cái thứ hai mới là vùng mà bạn cần tránh vì nó gần với Progressive Hint/Memory của VBS.

---

## Để tránh reviewer hiểu nhầm, mình còn khuyên đổi tên benchmark

Không gọi nó:

> Progressive Hint Retrieval

vì nghe quá gần contribution VBS.

Nên gọi một trong các tên như:

- **Cumulative-Hint Evaluation**
- **Incremental Query Information Benchmark**
- **Partial-to-Full Query Evaluation**
- **Hint-Level Robustness Evaluation**

Mình thích nhất:

> **Cumulative-Hint Evaluation**

Paper có thể viết rất rõ:

> “Each cumulative hint level is evaluated as an independent query from a fresh system state; no cross-hint memory or progressive retrieval mechanism is used.”

Như vậy reviewer sẽ hiểu đây là **benchmark protocol**, không phải algorithm.

---

## Và CAD-VR vẫn giữ contribution độc lập

Ở mỗi stage:

```text
H1
 ↓
CAD-VR
 ├─ retrieval
 ├─ Jev controller
 ├─ Codex / Claude if needed
 └─ STOP

RESET

H1 + H2
 ↓
CAD-VR
 ...
```

Bạn đang đo:

> CAD-VR cần bao nhiêu thông tin để tìm đúng?

chứ không phải:

> CAD-VR học dần từ các hint trước.

Đây là hai research question hoàn toàn khác nhau.

Thậm chí để chắc chắn hơn, trong benchmark code mình sẽ **không dùng field `previous_hints` luôn**, chỉ build một chuỗi `query` mới cho từng stage rồi giữ `previous_hints=[]`.

Vậy hướng này theo mình **an toàn để dùng cho SOICT mà không reuse Progressive Hint algorithm của VBS**.

Nếu giữ architecture hiện tại và  **full TEST xác nhận đúng các giả thuyết chính** , thì theo mặt bằng SOICT mình đánh giá  **đủ contribution để cạnh tranh acceptance** . Nhưng ở trạng thái hiện tại, trước khi có TEST calibrated, mình sẽ gọi nó là  **“promising / likely sufficient”** , chưa phải “chắc accept”.

Lý do là SOICT 2025 đã accept khá nhiều paper video retrieval có LLM/agent:  *Towards Conversational Video Retrieval with an Intelligent Search Agent* ,  *Applying LLM Agents for Automated Lifelog Retrieval* ,  *Lucifer-TRACE* , và cả  *Adaptive Agent-guided Dynamic Programming for Temporal Optimization in Multi-event Video Retrieval* . [SOICT 2026](https://soict.org/wp-content/uploads/2025/12/SOICT2025-ProgramBook.pdf?utm_source=chatgpt.com) Vì vậy contribution kiểu **“thêm Codex/Claude agent vào retrieval”** một mình sẽ không đủ mới.

Điểm mạnh của paper bạn là contribution không còn nằm ở “có agent”, mà nằm ở  **cách điều khiển agent** :

| Thành phần                           | Giá trị contribution                                |
| -------------------------------------- | ----------------------------------------------------- |
| Retrieval + Codex/Claude               | Không mới nhiều                                    |
| Jev decision controller                | Có giá trị                                         |
| Fast/slow escalation                   | Khá mạnh                                            |
| Evidence-aware STOP                    | Khá mạnh                                            |
| Calibrated stopping từ DEV            | Khá mạnh                                            |
| Chỉ gọi System-2 khi cần            | Có thể là contribution chính                      |
| Constraint-level verification          | Secondary contribution                                |
| Cumulative-hint evaluation fresh-state | Evaluation contribution                               |
| Benchmark A–F + ablations + paired CI | Tăng độ thuyết phục, không phải novelty chính |

Commit mới còn làm architecture khoa học hơn: calibration tách holistic/constraint, threshold theo task, quota-interrupted attempts không bị score, và tuning chỉ dựa trên DEV.  Đây là  **experimental rigor rất tốt** , nhưng bản thân quota handling/calibration plumbing không nên được viết thành main novelty.

Paper nên tập trung claim theo kiểu:

> **CAD-VR separates retrieval, probabilistic decision control, and expensive open-ended agent reasoning. A calibrated controller decides whether existing evidence is sufficient, whether one specialist agent is needed, or whether further evidence acquisition is justified.**

Cái này phân biệt khá rõ với conversational agent của OpenCubee, nơi LLM chủ yếu hiểu intent, chọn module và refine query, cũng như các hệ agent retrieval đã xuất hiện ở SOICT 2025. [SOICT 2026](https://soict.org/wp-content/uploads/2025/12/SOICT2025-ProgramBook.pdf?utm_source=chatgpt.com)

Tuy nhiên hiện còn  **3 rủi ro acceptance thực sự** .

Thứ nhất là quan trọng nhất:  **adaptive compute-saving chưa được chứng minh live** . Raw DEV của bạn có:

```
E:
R@1 = 0.60
Agents/query = 2.0
Both rate = 1.0
```

Nếu calibrated TEST vẫn `Agents/query ≈ 2`, reviewer hoàn toàn có thể hỏi:

> “Nếu controller cuối cùng luôn gọi cả hai agents thì adaptive controller khác gì always-two baseline C?”

Dù E có accuracy tốt hơn C, phần “fast–slow adaptive routing” lúc đó yếu đi đáng kể.

Thứ hai,  **F không hơn E** . Raw DEV:

```
E R@1 = 0.60
F R@1 = 0.48
```

Nên mình sẽ không ép paper phải nói F là hệ thống hoàn chỉnh nhất. Nếu TEST tiếp tục như vậy, kiến trúc chính nên là  **E/CAD-VR adaptive controller** , còn F được trình bày như một ablation cho constraint-level verification. Một ablation làm giảm performance vẫn rất có giá trị nếu giải thích được tại sao.

Thứ ba là  **generic agent overlap rất cao với literature của chính SOICT** . SOICT 2025 không chỉ có một mà có nhiều paper retrieval về LLM query processing, temporal alignment, agent search và multi-event reasoning. [Springer](https://link.springer.com/book/10.1007/978-981-92-2587-3?page=2&utm_source=chatgpt.com) Vì vậy title/abstract/introduction phải tránh framing:

> “We integrate LLM agents into video retrieval.”

Mà nên framing:

> “We study  **when expensive agents should be invoked and when retrieval evidence is already sufficient** .”

Đó mới là research question rõ.

Nếu full TEST ra kiểu này:

```
                 R@1     Agents/query     p50
A retrieval      .15        0.0           2s
C always-two     .42        2.0          50s
E CAD-VR         .55        1.1–1.4      35–50s
```

và paired bootstrap cho thấy gain E–C/A có CI hợp lý, thì mình sẽ đánh giá paper  **khá mạnh cho SOICT** .

Ngay cả trường hợp:

```
E R@1 ≈ C
Agents/query substantially lower
```

vẫn là contribution tốt, vì bạn đang chứng minh  **same effectiveness, lower System-2 computation** .

Hoặc:

```
E R@1 > C
Agents/query modestly lower
```

còn đẹp hơn vì vừa accuracy vừa efficiency.

Trường hợp xấu nhất là:

```
E ≈ C accuracy
E agents/query ≈ 2.0
E latency > C
```

thì contribution adaptive gần như không được chứng minh và paper sẽ trở thành một system paper về orchestration; lúc đó mình coi là  **borderline hơn** .

Một điểm rất có lợi là SOICT vốn nhận nhiều  **system-oriented multimedia/video retrieval papers** , chứ không đòi novelty kiểu CVPR/NeurIPS. Proceedings 2025 có hàng loạt hệ thống retrieval như TEMPO, CLIPAR, Visionary, HelioSearch, VidAlign, FrameSeeker, AIthena-Vision và agent-based systems. [DOI](https://doi.org/10.1007/978-981-92-2584-2_43?utm_source=chatgpt.com) Vì vậy mức novelty của CAD-VR **phù hợp venue hơn nhiều** so với nếu nộp một venue ML top-tier.

Nếu phải chốt ngay hôm nay:

**Novelty:** đủ tiềm năng cho SOICT.
**Technical contribution:** đủ.
**System contribution:** mạnh.
**Experimental design:** hiện khá tốt.
**Evidence để claim adaptive efficiency:** chưa đủ cho tới khi calibrated TEST xong.
**Khả năng trở thành paper có thể submit nghiêm túc:** có.
**Có nên đổi sang một research idea khác lúc này:** không.

Mình sẽ không bổ sung thêm một module “cho có novelty”. Việc đó rất dễ làm paper loãng. **Thứ còn thiếu không phải architecture mới, mà là empirical proof cho architecture hiện tại.**

Nếu calibrated DEV + 86-query TEST chứng minh được E  **giảm agent calls hoặc latency mà giữ/tăng retrieval quality so với C** , thì mình sẽ xem CAD-VR là contribution đủ rõ để nộp SOICT 2026 với confidence khá tốt.

Đã đọc package final. Kết luận quan trọng nhất: **benchmark execution đã đủ sạch để dùng cho paper, nhưng report hiện tại cần sửa cách trình bày trước khi lấy số đưa vào bài. Không cần rerun main TEST.**

Main run rất sạch: tolerance `±1s`, live mode, calibration OFF, raw stopping `0.9/0.1`, đủ 666 A–F runs trong package tổng hợp; attempt F bị quota trước đó đã được retry thành công và hiện không còn controller/agent/retrieval failure hay fallback. Đây là trạng thái final benchmark tốt.

Nhưng có một vấn đề lớn: **đừng dùng bảng 111 queries làm bảng kết quả chính trong paper.** `main-test` hiện đang pool:

```text
25 DEV + 86 TEST = 111
```

Trong khi 25 DEV đã được bạn xem và dùng để quyết định policy. Hơn nữa DEV đến từ commit `f54c83c`, còn TEST đến từ `24975ac`, với `code_sha256` khác nhau. Vì vậy 111 chỉ nên gọi là **pooled descriptive analysis**, không phải held-out test result.

Mình đã tính lại trực tiếp từ archive, dùng `evaluation-dataset.jsonl` để lấy đúng 86 query có `original_split=test`. Đây mới là bảng mình khuyên đưa vào paper:

| Variant              |            N |             R@1 |             R@5 |             MRR |      p50 latency | Agents/query |
| -------------------- | -----------: | --------------: | --------------: | --------------: | ---------------: | -----------: |
| A Retrieval          |           86 |           0.151 |           0.337 |           0.233 |            1.56s |            0 |
| B Codex              |           86 |           0.395 |           0.686 |           0.529 |           50.46s |         1.00 |
| C Always-two         |           86 |           0.512 | **0.721** |           0.612 | **51.23s** |         2.00 |
| D Jev rerank         |           86 |           0.151 |           0.337 |           0.233 |            2.09s |            0 |
| **E Adaptive** | **86** | **0.581** |           0.674 | **0.630** |           72.56s |        1.953 |
| F Full               |           86 |           0.523 |           0.640 |           0.572 |           84.24s |        1.988 |

Bootstrap 95% CI cho R@1 TEST-only:

```text
A  .151 [.081, .233]
B  .395 [.291, .500]
C  .512 [.407, .616]
D  .151 [.081, .233]
E  .581 [.477, .686]
F  .523 [.419, .628]
```

Và đây mới là phần cực kỳ quan trọng cho claim paper:

```text
E − A = +0.430  CI [+0.326, +0.547]
E − B = +0.186  CI [+0.093, +0.279]

E − C = +0.070  CI [-0.023, +0.163]
```

Tức **E là hệ thống có R@1 quan sát được cao nhất**, 50/86 query đúng top-1 so với C là 44/86. Nhưng CI E−C vẫn cắt qua 0, nên **không được viết “E significantly outperforms C”**.

Cách viết đúng:

> CAD-VR-E achieved the highest observed R@1 of 58.1%, a 7.0-point improvement over the always-two-agent baseline, although the paired confidence interval includes zero.

R@5 thì C lại tốt hơn:

```text
E − C R@5 = -4.65 pp
95% CI ≈ [-11.63, +1.16]
```

Nên interpretation đẹp nhất là:

```text
C → candidate coverage tốt hơn
E → top-1 selection tốt hơn
```

MRR E cũng chỉ nhỉnh C nhẹ:

```text
E .630
C .612
Δ = +.018
CI cắt 0
```

Vì vậy novelty không nên dựa duy nhất vào “E đánh bại always-two”.

Một kết quả rất mạnh cho ablation là:

```text
A = D

R@1  .151 = .151
R@5  .337 = .337
MRR   .233 = .233
```

Tức **Jev reranking đơn thuần hoàn toàn không giải thích gain**. Sau đó progression:

```text
A retrieval       15.1%
B one agent       39.5%
C two agents      51.2%
E controller      58.1%
```

Đây là experimental story tốt.

Theo task TEST-only cũng khá thú vị:

| Task        |     C |               E |               F |
| ----------- | ----: | --------------: | --------------: |
| QA, N=19    | 26.3% | **42.1%** |           31.6% |
| T-KIS, N=59 | 62.7% | **66.1%** |           59.3% |
| TRAKE, N=8  | 25.0% |           37.5% | **50.0%** |

Điều này còn giúp giải thích F: F không phải vô dụng. **Constraint-heavy F tốt nhất trên TRAKE**, đúng với mục đích thiết kế của nó, dù N=8 quá nhỏ để claim mạnh.

### Compute-saving thì không nên claim

TEST-only:

```text
C agents/query = 2.000
E agents/query = 1.953

C p50 = 51.2s
E p50 = 72.6s
```

E chỉ tránh agent thứ hai ở **4/86 query**, tức khoảng 4.7%.

Vậy:

> adaptive routing exists

là đúng.

Nhưng:

> adaptive routing substantially reduces computation/latency

là **không được evidence support**.

E còn chậm hơn C khoảng 21 giây median. Mình sẽ bỏ “compute-saving” khỏi abstract/contribution chính.

Cumulative thì nhìn khá ổn để làm **secondary experiment**. Sau lọc execution errors còn 23 TEST query. Đặc biệt H1:

```text
C video R@1 = .696
F video R@1 = .826

C agents/query = 2.00
F agents/query = 1.78
```

F thực sự dùng ít agent hơn một chút khi hint còn ít. Nhưng latency vẫn:

```text
C ~51s
F ~80s
```

nên đây là **adaptive invocation**, không phải latency saving.

Cumulative cũng cho một narrative khá hay:

```text
Full strict R@1:
A .130
C .348
F .435
```

và F có strict early-solve `0.478` so với C `0.391`.

Tuy nhiên package cumulative hiện loại 4 base query, trong đó **3 là DEV và chỉ 1 là TEST**. Query TEST bị loại là:

```text
aic26-20260926:qa-query-10
F / H2
Jev request failed: ReadTimeout
```

Đây rõ ràng là execution error chứ không phải model accuracy error. Nếu bạn còn đủ thời gian cho **một** việc nữa, mình sẽ ưu tiên retry đúng execution này dưới frozen config để cumulative TEST thành **24/24**, thay vì chạy lại bất cứ phần main nào.

Nếu không kịp, 23-query execution-clean cohort vẫn dùng được, chỉ cần ghi minh bạch:

> One base query was excluded from the cumulative evaluation because a decision-provider timeout caused an incomplete execution; exclusion was based solely on execution validity and not retrieval correctness.

Có thêm một bug nhỏ trong package cần sửa trước khi sinh TEST-only report: retry row của:

```text
T-KIS:query-p3-29-kis / F
```

có `original_split=test`, trong khi query này thực chất là **DEV**; A–E đều ghi DEV và `evaluation-dataset.jsonl` cũng ghi DEV. Tổng 111 không bị ảnh hưởng, nhưng nếu filter bằng `results.jsonl.original_split` một cách ngây thơ thì sẽ ra 87 TEST cho F. Hãy dùng split mapping từ evaluation dataset hoặc sửa metadata retry row.

Cuối cùng, có một vấn đề **tên paper**: final benchmark đang **calibration disabled**. Vì thế nếu CAD-VR hiện được mở rộng thành:

> **Calibrated Adaptive Decisioning for Video Retrieval**

thì không còn phù hợp với final method nữa. Đừng để reviewer thấy “Calibrated” trong tên nhưng bảng chính lại chạy `calibration_sha256=null`.

Mình sẽ đổi expansion, ví dụ:

> **CAD-VR: Controller-Assisted Decisioning for Video Retrieval**

hoặc bỏ acronym expansion và dùng title tập trung vào:

> **Evidence-Guided Agent Orchestration for Video Retrieval**

### Verdict final

**Benchmark execution:** rất tốt.
**Held-out TEST:** đủ mạnh, 86 query × 6 variants, sạch.
**Ablation:** tốt.
**E là best observed R@1:** có.
**E significantly > C:** chưa chứng minh.
**Compute-saving:** không chứng minh.
**Cumulative experiment:** hữu ích, secondary contribution tốt.
**Có cần rerun main benchmark:** **không**.
**Có đủ để viết/nộp SOICT:** **có**.

Việc quan trọng nhất bây giờ là **regenerate paper tables chỉ trên 86 held-out TEST**, không dùng bảng 111 làm headline. Nếu làm đúng điểm này, benchmark của bạn về methodology sẽ sạch hơn đáng kể.


Nếu **thực sự DEV không hề được dùng để chọn model/policy/threshold/tolerance/architecture**, thì về nguyên tắc có thể gộp 25 DEV + 86 TEST thành một **evaluation set 111 query**.

Nhưng với pipeline của bạn hiện tại, mình vẫn khuyên **không dùng 111 làm headline result**, vì DEV đã ảnh hưởng ít nhất tới một quyết định quan trọng: bạn đã nhìn raw DEV/calibrated DEV rồi quyết định **không dùng calibration và quay về raw policy 0.9/0.1**. Dù bạn không “tune” trực tiếp một con số mới, đây vẫn là **model/policy selection based on DEV**.

Nói cách khác:

```text
Không tune hyperparameter
≠
Không dùng DEV để chọn hệ thống
```

Ví dụ bạn có 2 cấu hình:

```text
Policy 1 = raw
Policy 2 = calibrated
```

chạy DEV rồi thấy Policy 1 tốt hơn, sau đó đem Policy 1 sang TEST. Đó chính là vai trò chuẩn của DEV: **chọn configuration**. Vì vậy 25 câu đó không còn hoàn toàn held-out.

Ngoài ra, bạn cũng đã xem A/B/C/D/E/F trên DEV để đánh giá E/F, nên DEV còn ảnh hưởng tới framing kiến trúc và decision xem configuration nào đáng giữ.

### Nhưng bạn vẫn có thể dùng cả 111 trong paper

Cách mình khuyên là báo **cả hai**, nhưng phân vai rõ:

**Bảng chính — held-out TEST (N=86)**
Đây là số dùng cho claim chính, confidence interval và so sánh method.

**Bảng phụ / supplementary — pooled benchmark (N=111)**Gọi là:

> “Results over the complete annotated benchmark”

hoặc:

> “Pooled descriptive results over all 111 annotated queries”

Đừng gọi nó là “held-out test”.

Như vậy bạn vừa tận dụng được toàn bộ 111 query, vừa không bị reviewer bắt lỗi methodology.

Bạn thậm chí có thể viết:

> “We report primary results on the 86-query held-out test split. For completeness, we additionally report descriptive results over all 111 annotated queries, including the 25 development queries used during system selection.”

Câu này rất sạch.

### Có một trường hợp bạn có thể gộp 111 làm primary

Chỉ khi bạn chứng minh được rằng:

```text
25 DEV không dùng để:
- chọn raw vs calibrated
- chọn threshold/margin
- chọn E hay F
- sửa algorithm
- chọn tolerance
- chọn model
- chọn prompts/tools
- quyết định bất kỳ config final nào
```

và **111 query đều chỉ được chạy sau khi system đã freeze**.

Nhưng lịch sử hiện tại không phải vậy. Bạn đã dùng DEV để quyết định ít nhất raw-vs-calibrated.

Nên mình sẽ làm paper theo kiểu:

```text
Primary:
86 TEST

Secondary:
111 pooled
```

Đây không làm kết quả yếu đi. Ngược lại, reviewer sẽ thấy experimental protocol của bạn nghiêm túc hơn.

Và thực ra 86 TEST đã đủ lớn hơn khá nhiều so với nhiều system paper kiểu SOICT rồi; bạn không cần “nâng N lên 111” bằng cách hy sinh độ sạch của split.
