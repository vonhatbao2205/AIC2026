# CAD-VR: chạy benchmark agent

### Bộ báo cáo cuối sau khi retry hoàn tất

Kết quả cuối nằm trong `benchmarks/agent/runs/soict-final-dev-test-pooled-tol1-v1/`:

| Cohort | Main | Cumulative |
|---|---|---|
| DEV | `dev/main/`: 25 câu, 150 lượt | `dev/cumulative/`: 3 câu, 33 lượt |
| TEST | `test/main/`: 86 câu, 516 lượt | `test/cumulative/`: 24 câu, 270 lượt |
| Pooled | `pooled/main/`: 111 câu, 666 lượt | `pooled/cumulative/`: 27 câu, 303 lượt |

Mỗi thư mục có báo cáo strict, bootstrap intervals, CSV và `video-report/`.
Cumulative có thêm bảng theo mức hint, hints-to-solve và biểu đồ PDF/PNG theo
cohort cố định. Không còn query bị loại sau khi các retry thành công. DEV/TEST
được khôi phục theo nhãn gốc, kể cả lượt retry đã chạy bằng dataset gộp TEST.
`README.md`, `SUITE_SUMMARY.json` và `ALL_PAPER_TABLES.csv` ở root tổng hợp cả ba
cohort. Đây là export offline; nguồn kết quả và archive lỗi được giữ để đối chiếu.

Bản báo cáo gọn được lưu trong Git tại
[`results/soict-final-tol1/`](results/soict-final-tol1/README.md), gồm bảng,
summary/manifest, CSV và biểu đồ. Raw rankings/traces đầy đủ nằm trong `runs/`
và không đưa vào Git. Mọi lệnh chạy/chấm cho bộ paper cuối dùng `--tolerance 1`
tường minh; default CLI vẫn là 5s cho các run khác.

Để xuất một phiên bản mới từ cùng nguồn đã hoàn tất, dùng output chưa tồn tại:

```bash
.venv/bin/python -m tools.finalize_soict_reports \
  --main-run benchmarks/agent/runs/soict-test-all111-tol1-retry-v1/main-test \
  --cumulative-run benchmarks/agent/runs/soict-test-all111-tol1-retry-v1/cumulative-all27-test \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/soict-final-dev-test-pooled-tol1-v1
```

### Chấm bổ sung theo video từ kết quả đã lưu

Không gọi model hoặc thay đổi fingerprint/resume của benchmark đang chạy:

```bash
.venv/bin/python -m tools.benchmark_video_report \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --run-dir benchmarks/agent/runs/soict-test-raw-tol1-v1/main-test
```

Xuất `video-report/REPORT.md`, `summary.json`, `query_results.csv`.
Video được xếp theo lần xuất hiện đầu tiên trong ranking frame; mỗi video chiếm
một slot. **Video R@k** chỉ kiểm tra đúng video. **Group moment R@k** yêu cầu
video top-k chứa ít nhất một frame đúng trong danh sách frame đã trả về, với
tolerance của run. **Group strict R@k** thêm yêu cầu answer đúng cho QA; TRAKE
vẫn cần chuỗi event đầy đủ, đúng thứ tự theo evaluator hiện tại.
Đây là đo coverage của candidate pool trong video, bên cạnh strict R@k cũ;
không thay thế accuracy của một frame/answer được submit. Báo cáo dùng cohort
chạy đủ các variant tại mỗi hint level, có breakdown theo task. Chạy lại lệnh
export sau khi có thêm kết quả để cập nhật; có thể áp dụng cho DEV hoặc cumulative.

### Gộp các kết quả DEV gốc và TEST để báo cáo trên 111 câu

```bash
.venv/bin/python -m tools.merge_benchmark_results \
  --dev-run benchmarks/agent/runs/soict-f54c83c/main-dev \
  --test-run benchmarks/agent/runs/soict-test-raw-tol1-v1/main-test \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/soict-test-all111-tol1-v1/main-test \
  --treat-dev-as-test
```

Chỉ dùng output mới chưa tồn tại. Lệnh kiểm tra cùng dataset, tolerance,
model/controller và đủ mọi query/variant trước khi gộp. Không gọi model.
Dataset trong output gán split `test`, giữ `original_split`; nguồn gốc và các
revision code được ghi trong manifest/provenance. Báo cáo gộp giữ mọi lượt lỗi
trong mẫu số, gồm lỗi quota cũ của DEV; kết quả nguồn không đổi.
Đây là thư mục báo cáo offline, không dùng làm output để resume `run_soict`.
`summary_by_split.json` vẫn có thống kê riêng theo nguồn DEV/TEST để đối chiếu.

### Cumulative theo video và gộp DEV/TEST

Exporter video tự nhận `hint_mode=cumulative`, xuất bảng từng mức hint và task,
cohort đủ mọi level/variant, strata theo tổng số hint, và hints-to-solve cho
Video / Group moment / Group strict. `--plots` xuất đường recall PDF/PNG theo
cohort có cùng tổng số hint; không trộn H1/H2/Full để tính một recall chung.

```bash
.venv/bin/python -m tools.benchmark_video_report \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --run-dir benchmarks/agent/runs/soict-test-raw-tol1-v1/cumulative-test \
  --plots

.venv/bin/python -m tools.merge_benchmark_results \
  --dev-run benchmarks/agent/runs/soict-f54c83c/cumulative-dev \
  --test-run benchmarks/agent/runs/soict-test-raw-tol1-v1/cumulative-test \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/soict-test-all111-tol1-v1/cumulative-test \
  --treat-dev-as-test
```

Bản gộp lịch sử có 27 base queries, 303 stage/variant runs, bao gồm các lượt lỗi
DEV cũ được ghi rõ trong report. Để xuất cohort sạch theo yêu cầu bỏ query lỗi:

```bash
.venv/bin/python -m tools.filter_benchmark_failures \
  --source benchmarks/agent/runs/soict-test-all111-tol1-v1/cumulative-test \
  --output benchmarks/agent/runs/soict-test-all111-tol1-retry-v1/cumulative-test
```

Lệnh loại **toàn bộ base query** nếu bất kỳ hint/variant nào có agent failure,
controller failure/fallback, retrieval error/degradation hoặc quota interruption;
không loại theo accuracy. Mọi level và A/C/F của query đó cùng bị loại để giữ
cohort so sánh. `EXCLUSIONS.json` ghi số lượng/lý do; `excluded_attempts.jsonl`
giữ các lượt bị loại. Dataset và manifest của report được cập nhật đúng cohort
còn lại. Không gọi model, không tạo retry. Các output phải là thư mục mới.

CAD-VR nằm trong `backend/app/agent/`. Mặc định `AGENT_POLICY=full`.
Codex dùng `gpt-6.1-sol`, reasoning `high`, fast. Jev gọi
`https://openrouter.ai/api/alpha/decisions` với `typesafe/jev-1.13`.
Key lấy từ `OPENROUTER_API_KEY` hoặc `API_KEY/openrouter.txt`; không gửi key cho CLI.

## Luồng hoạt động

1. Main retrieval trả kết quả ngay. Console chuyển `retrieval_id` do server cấp
   vào run để dùng lại đúng bằng chứng của query/profile/model/scope đó.
   Khi chạy benchmark hoặc thiếu seed hợp lệ, controller tự chạy retrieval.
   TRAKE dùng retrieval chuỗi hiện có; không biến thành tìm một frame đơn.
2. Board chỉ chứa dữ liệu của query hiện tại: constraints, score theo channel,
   OCR/speech/metadata, frame đã xem, quan sát của agent, lịch sử tool.
   `previous_hints` không đi vào prompt, compiler hoặc retrieval của run.
3. Jev đánh giá `supported / refuted / unknown`, code tổng hợp bằng geometric mean.
   Confidence agent và embedding similarity không được coi là xác suất đúng.
   Unknown giữ thứ tự RRF; chỉ bằng chứng hỗ trợ/phủ định chiếm ưu thế mới
   được dùng để nâng/hạ hạng. Việc được chấm không tự làm candidate lên hạng.
   Jev là text-only: chỉ các quan sát của agent sau khi ảnh thật được tải mới được
   dùng như mô tả hình ảnh. OCR/transcript vẫn có thể sai hoặc nói về cảnh khác.
4. Controller chọn STOP, Codex, Claude, BOTH hoặc thêm một bước search/inspect/compare.
   Codex ưu tiên exploration; Claude ưu tiên inspection. Đây là soft roles,
   cả hai vẫn được tự search và dùng mọi tool được bật.
5. STOP sớm yêu cầu xác suất candidate sau calibration vượt threshold và margin.
   Mỗi constraint còn phải có raw supported > 0.5 và lớn hơn unknown/refuted;
   đây là guard về bằng chứng, không phải xác suất constraint đã calibration.
   QA cần có answer; TRAKE cần đủ event cùng video và đúng thứ tự. AVS không dừng
   sớm chỉ vì tìm được một frame. Disagreement chỉ là tín hiệu, không có stage
   bắt buộc gọi thêm agent để tạo disagreement.
6. Kết quả hợp nhất ở `GET /api/agent/runs/{id}` → `ranking`, `baseline_ranking`,
   `controller`, `metrics`. Agent cards dùng thứ tự ranking này. Main search vẫn
   hiển thị độc lập, không bị controller âm thầm thay thế. Trace có ở
   `GET /api/agent/runs/{id}/trace`; không chứa base64 hay credentials.

Compiler hiện tại **deterministic**, giữ toàn bộ query, các mệnh đề/từng câu,
quoted OCR và các event E1…En. Các mẫu Việt/Anh còn tách chi tiết về trang phục,
hành động, quan hệ, vị trí và phủ định, có source span và dependency về query gốc.
Đây là decomposition bảo toàn câu gốc, chưa phải
semantic parser có khả năng tách chính xác mọi subject/action/relation tiếng Việt.

Hai tool mới: `compare_candidates` (2–4 cột, before/center/after + local text) và
`constraint_probe` (constraint trong một vùng thời gian của candidate, không global
search). `video_frames` nhận `center_time`, `before`, `after`, `density`.
Jev chỉ nhận action classes. Macro COMPARE_TOP2 của controller lấy local text;
việc so ảnh được giao cho CLI qua `compare_candidates`.

Board giữ tối đa 24 nguồn/candidate; quan sát hình ảnh đã xem được giữ khi OCR
lấp đầy bộ nhớ. Context chọn tối đa 8 nguồn, ưu tiên visual, cân bằng loại nguồn
và gộp text lặp; provenance gốc vẫn được lưu.

Cache/single-flight và evidence provenance chỉ dùng trong run. Quan sát trùng nguồn
không thành phiếu bầu độc lập. Run hoàn tất giữ trace để đọc trong TTL 1 giờ,
được thu hồi khi có run mới; board không bao giờ truyền sang query khác.

## Cấu hình và giới hạn

`AGENT_TIMEOUT_SECONDS=240` giới hạn toàn run, gồm retrieval, queue, Jev và CLI.
`AGENT_MAX_STEPS=6`, `AGENT_MAX_TOOL_CALLS=40`, `AGENT_VERIFY_TOP_K=5`.
Mỗi CLI được gọi tối đa một lần/run; `AGENT_MAX_CONCURRENT=2` giới hạn mỗi CLI trên
mọi tab. Lỗi Jev, response sai schema, timeout hoặc thiếu key được ghi vào
`controller.fallback`; controller thử agent còn khả dụng, không gắn nhãn "đã xác minh".

`AGENT_STOP_THRESHOLD=0.9` và `AGENT_STOP_MARGIN=0.1` là giá trị khởi đầu, **chưa
được fit trên corpus**. Không được gọi số liệu mặc định là calibrated accuracy.
`AGENT_CALIBRATION_PATH` nạp artifact temperature được fit trên dev; snapshot ghi
`calibrated=false` nếu chưa có artifact. Tune ngưỡng chỉ trên dev rồi khóa trước test.

## Dataset

Từ repo root, dùng môi trường có backend dependencies và `openpyxl`:

```bash
.venv/bin/python -m benchmarks.agent.export_dataset \
  --output benchmarks/agent/runs/workbooks-v2.jsonl
```

Exporter đọc TKIS/QA/TRAKE workbook hiện có, không đọc PHM/VBS annotations.
Không tạo nhãn giả; trường hợp thiếu answer/frame/chuỗi đúng bị ghi trong `.audit.json`.
Câu có marker temporal nhưng nằm trong workbook KIS/QA bị loại và ghi audit vì
thiếu ánh xạ event→frame. Loader từ chối dataset cũ gắn những câu đó thành KIS.
Bản export v2 hiện có 81 câu: 20 dev, 61 test; v1 từng có một câu TRAKE gắn sai loại.
Split dev/test cố định theo hash nội dung query (20/80 xấp xỉ). Nên kiểm tra thêm
near-duplicate queries và video overlap trước khi dùng số liệu cho paper.
Không đo được độ chính xác khi chưa có ground truth hợp lệ.

### Workbook AIC26 và submit ngày 26/9

[Bộ nhãn đã review](annotations/aic26-20260926/README.md) bổ sung **30 câu**:
15 T-KIS, 12 QA, 3 TRAKE. Workbook có 32 dòng, trong đó 21–23 là cùng một câu.
22 câu giữ CORRECT của DRES; 7 câu dùng override WRONG-only theo xác nhận chủ dữ liệu,
sau khi xem frame/clip và loại các lần submit không khớp. Câu TRAKE đua xe thiếu
submit được tìm bằng PE trong S01 và annotate bốn mốc trên video gốc S01-V011.
Nhãn tự tạo này ghi riêng nguồn và sai số, không được gán verdict DRES.

```bash
.venv/bin/python -m benchmarks.agent.export_submit_dataset \
  --output benchmarks/agent/runs/aic26-20260926-v3-audited.jsonl --verify-media

.venv/bin/python -m benchmarks.agent.export_dataset \
  --additional-dataset benchmarks/agent/annotations/aic26-20260926/ground_truth.jsonl \
  --group-video-splits \
  --output benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl
```

Trên máy hiện tại đã xuất `workbooks-v5-submit-audited.jsonl`: **111 câu, 25 dev,
86 test**, không trùng target video giữa dev/test. `--group-video-splits` chuyển
test sang dev khi cùng video với dev; câu đã dùng trong dev không chuyển sang test.
Audit vẫn giữ thông tin overlap trước khi nhóm và có `final_split_audit` sau khi nhóm.
Exporter từ chối ghi đè dataset khác; dùng tên phiên bản mới khi sửa nhãn.
Các lệnh xuất nhãn không chạy benchmark hay gửi submit. Evidence ảnh/clip là file
local bị git-ignore; JSON provenance và SHA-256 được giữ trong bộ annotations.

Có thể tự cung cấp JSONL:

```json
{"query_id":"kis-001","split":"test","query_type":"T-KIS","query":"người sửa xe đạp","retrieval_database":"infoshotpp","image_models":["pe"],"targets":[{"video_id":"L21_V001","frame_idx":1000,"fps":25}]}
```

Target nhận `submit_keyframe_id`, `frame_idx` hoặc `start_s` + `end_s`.
QA target cần `answers` (danh sách exact accepted answers, chuẩn hoá case/Unicode/space).
TRAKE cần `sequences` (danh sách alternatives; mỗi alternative là các target theo E1..En).
FPS dùng từ nhãn hoặc candidate, không tự giả định 25 nếu thiếu.

## Sáu cấu hình chính

### Ablation bổ sung: C+V

`C+V` (`parallel_verify`) chạy Codex và Claude song song như C, rồi gọi holistic
verifier của E một lần cuối. Không có verification trước agent, adaptive routing
hoặc early stopping. Snapshot giữ `pre_verification_ranking`, trace giữ event
`pre_verification`, để so sánh RRF với ranking sau verifier trên cùng evidence.

Chạy C/C+V/E mới trong cùng đợt, không gộp với kết quả A–F đã khóa của paper:

```bash
.venv/bin/python -m benchmarks.agent.run_benchmark \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/soict-cv-20261005-v1/dev \
  --split dev --variants 'C,C+V,E' --tolerance 1

.venv/bin/python -m benchmarks.agent.run_benchmark \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/soict-cv-20261005-v1/test \
  --split test --variants 'C,C+V,E' --tolerance 1
```

DEV có 75 lượt; TEST có 258 lượt. Kiểm tra execution/fallback trên DEV, giữ nguyên
code/config rồi mới chạy TEST. Chạy lại đúng lệnh để resume; quota interruption
được archive rồi thử lại từ state mới. Không tune theo accuracy TEST. C+V–E vẫn
là so sánh cả controller package, chưa cô lập riêng causal effect của call allocation.
Implementation đã qua kiểm thử offline; chưa có kết quả live hoàn chỉnh cho C+V.

### Cấu hình A–F

| ID | Policy | Hành vi |
|---|---|---|
| A | retrieval | Chỉ main retrieval |
| B | codex | Retrieval + luôn gọi Codex nếu khả dụng |
| C | parallel | Retrieval + Codex và Claude song song nếu khả dụng |
| D | rerank | Jev holistic reranking; không agent |
| E | adaptive | Jev holistic verification/routing + agent thích ứng |
| F | full | Constraint verification + adaptive agents + evidence actions |

B/C dùng reciprocal-rank fusion retrieval và đề xuất agent. Mọi cấu hình mặc định
có cùng soft roles/toolset để so sánh orchestration; `no_roles` / `base_tools` đo
ảnh hưởng riêng. C là baseline always-two **sau retrieval**; không phải đo nguyên
bản sidecar trước nâng cấp chạy agent cùng lúc với main retrieval.

```bash
# Backend đang chạy trên localhost:8000, các CLI đã đăng nhập.
.venv/bin/python -m benchmarks.agent.run_benchmark \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/dev-v5 --split dev --variants A,B,C,D,E,F

.venv/bin/python -m benchmarks.agent.run_benchmark \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/test-v5 --split test --variants A,B,C,D,E,F
```

Có `--limit`, `--url`, `--timeout`, `--poll`, `--tolerance` (giây, mặc định 5), `--seed`.
Raw DEV trước đây dùng tolerance=1; khi đối chiếu với run mới cần chấm lại cả hai
ở cùng tolerance. Không sửa ground truth để tăng tolerance; tham số chỉ áp dụng
trong evaluator và được khóa trong manifest. Target keyframe ID exact và accepted
QA answers vẫn giữ exact matching; TRAKE vẫn cần đủ events, cùng video, đúng thứ tự.
Lệnh này gọi model thật nếu backend chạy live. Không gọi endpoint DRES submit.
Mỗi query random thứ tự variants với seed cố định để giảm thiên lệch cache/độ nóng.
Từng run tự retrieval, dùng cùng cấu hình; không dùng ground truth để seed controller.

Output: `results.jsonl` (snapshot + trace + metrics), `manifest.json`,
`summary.json`, `summary_by_task.json` (tách loại task), `summary_paired.json`
(chỉ query đã đủ tất cả variants được cấu hình), `REPORT.md`. Chạy lại cùng lệnh để resume các cặp query/variant đã
hoàn tất; lỗi quota sẽ retry tự động khi resume, lỗi khác cần thư mục mới để retry. Manifest từ chối resume khi
đổi dataset, code agent/benchmark, model, ngưỡng hoặc calibration artifact.
Gặp quota Codex/Claude/Jev: controller ngắt run, hủy agent chạy song song và các
tool còn lại. Runner loại kết quả một phần khỏi `results.jsonl` và báo cáo chấm,
ghi trace/usage vào `quota_attempts.jsonl`, ghi `STOPPED.json` rồi dừng suite.
Khôi phục quota rồi chạy lại **cùng lệnh/cùng output**: retry chính
query/hint-stage/variant bị ngắt bằng fresh state; các lượt đã hoàn thành giữ nguyên.
Quota là lượt bị gián đoạn do tài khoản, được báo audit riêng, không tính thành
retrieval failure. Lỗi/timeout không phải quota vẫn nằm trong mẫu số. Khi resume,
runner cũng chuyển các row quota đã lưu sang audit nếu manifest vẫn tương thích.
Không xóa kết quả của query khác. Không tự đợi reset hoặc tự retry khi quota chưa hồi.

## Ablations và calibration

```bash
.venv/bin/python -m benchmarks.agent.run_benchmark \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/ablations-dev-v5 --split dev \
  --variants F,rule,rule_router,llm_router,no_verifier,claude_verifier,base_tools,compare_only,no_roles

.venv/bin/python -m benchmarks.agent.calibrate \
  --results benchmarks/agent/runs/dev-v5/results.jsonl \
  --variant F --output benchmarks/agent/runs/calibration.json
```

- `rule`: cascade Codex→Claude, không decision model.
- `rule_router`: verifier Jev, routing bằng rule.
- `llm_router`: generative LLM thay Jev router, verifier vẫn Jev.
- `no_verifier`: Jev router được phép kết thúc bằng quyết định chưa kiểm chứng.
- `claude_verifier`: generative Claude judge, Jev router.
- `base_tools`, `compare_only`: lần lượt 8 tool và 9 tool; F là 10 tool.
- `no_roles`: không strategy bias, giữ common prompt và cùng capabilities.

Generative ablations gọi OpenRouter chat completions với
`AGENT_LLM_JUDGE_MODEL=anthropic/claude-opus-5.5` (có thể cấu hình), dùng cùng state,
criteria và validation. Xác suất LLM tự sinh được đo riêng, không giả định tương
đương decision-head probability của Jev. Chỉ chạy ablation khi muốn đo chi phí này.

Sau fit, cấu hình `AGENT_CALIBRATION_PATH` rồi khởi động lại/reload backend và dùng
**output directory mới**. Evaluator từ chối test query xuất hiện trong dev fit.
Artifact không tự sửa threshold và không tự bật trên production. Calibration chỉ
fit candidate relevance; không áp dụng nó lên từng constraint như thể có nhãn
constraint. Raw majority-evidence guard vẫn bắt buộc kể cả khi probability cao.

## Metrics và cách đọc

R@1, R@5, MRR dùng moment; QA còn yêu cầu answer; TRAKE dùng rank video có chuỗi
hoàn chỉnh. `video_r1/video_r5` báo riêng, không thay thế moment recall.
Agent Rescue@k có mẫu số là query baseline thất bại ở k, trả null nếu không có;
chỉ tính khi đã gọi agent và kết quả đúng có provenance từ agent.
`improved_at_1/5` báo riêng cải thiện kể cả chỉ do reranking.
Time-to-first-correct là thời điểm ranking top-1 đúng đầu tiên (TRAKE: top video
có chuỗi đúng), báo cả số query quan sát được; query chưa đúng là censored.
Có p50/p95 wall latency, tool/model calls/query, tỷ lệ đúng không gọi System-2,
ECE/Brier trên final candidate relevance (không phải constraint-level labels).

Cost/tokens là **agent + decision layer**; không bao gồm chi phí hạ tầng retrieval.
CLI Codex có thể không báo cost: `cost_usd_mean=null` và `cost_usd_coverage` thể hiện
thiếu dữ liệu. `known_cost_usd_mean` chỉ là phần đã biết, không phải tổng chi phí.
Không dùng giá API để giả làm phí subscription CLI. Trace ghi raw usage để audit.
Lỗi/timeout vẫn ở mẫu số, đồng thời có `failed` và `fallback` rate.
`retrieval_degraded` ghi nhận channel unavailable; `retrieval_failed` nhận cả
trường hợp không có candidate do worker mất kết nối. `agent_failed` báo lỗi CLI
và baseline B/C thiếu CLI bắt buộc. Bảng lỗi hiển thị trực tiếp trong REPORT.
`no_cli_rate`, `codex_only_rate`, `claude_only_rate`, `both_agents_rate` cho biết
phân bố invocation; chúng đếm cả lần gọi thất bại và không tương đương solved rate.
Phải kiểm tra các tỷ lệ này trước khi diễn giải accuracy.

Pilot `dev-20261003-v1` giữ nguyên để audit: có lỗi mất visual evidence, ưu tiên
unknown trong ranking, một câu temporal sai loại và lỗi quota. Không trộn số liệu
pilot với phiên bản đã sửa; dùng dataset v2 và output directory mới. Chưa chạy lại
benchmark live hoặc fit calibration sau các bản sửa này.

Tests offline: `pytest backend/tests/test_agent*.py`. Smoke mock/HTTP không phải
kết quả chất lượng corpus. Cần chạy các lệnh live ở trên để có kết quả paper.

## SOICT: Full query + Cumulative-Hint Evaluation

Protocol này đo CAD-VR cần bao nhiêu thông tin đầu vào để tìm đúng. Mỗi prefix
`h1`, `h1+h2`, …, `h1+…+hN` là **một query độc lập**. Đây là evaluation protocol;
không dùng thuật toán Progressive Hint/Progressive Memory của VBS.

> Each cumulative hint level is evaluated as an independent query from a fresh
> system state; no cross-hint memory or progressive retrieval mechanism is used.

Mỗi query/level/variant gọi `POST /api/agent/runs` mới. Payload chỉ có query hiện
tại và cấu hình retrieval/policy; luôn `previous_hints=[]`, không có `retrieval_id`,
`replaces`, candidate, reasoning, ground truth hay các hint tương lai. Backend tạo
`AgentRun`, EvidenceBoard, Controller, Jev client, history và tool cache mới.
Codex dùng process ephemeral, Claude không lưu session. Runner xáo trộn các mức
và variants bằng seed cố định; H2 không phụ thuộc việc H1 đã chạy.
Connections, static corpus/model assets và provider infrastructure có thể vẫn
warm; đây là reset trạng thái retrieval/agent, không phải cold-start latency.

Giữ bảng chính Full A–F trên toàn dataset. Cumulative mặc định A/C/F, chỉ lấy
T-KIS/QA có ít nhất hai `hints_vi` đã review. Không tự chia câu workbook thành hint;
TRAKE/AVS chưa có protocol nhãn partial phù hợp nên được loại khỏi cumulative
và ghi lý do trong `run_plan.json`. Vẫn đánh giá các task đó trong Full.
Full chỉ chạy một lần tại mức cuối, không thêm HN và Full trùng nhau.
Loader và exporters từ chối hint rỗng/sai kiểu hoặc join(hints_vi) khác query;
chỉ chấp nhận khác biệt Unicode NFC/whitespace, giữ nguyên case, dấu câu và facts.
Merge dataset giữ các hint boundaries, từ chối hai cách chia hint mâu thuẫn.

Dataset v5 có **111 câu: 25 dev / 86 test**. Structured-hint subset có
**27 câu: 3 dev / 24 test**. Lịch chạy mặc định:

| Split | Full A–F | Cumulative A/C/F | Tổng |
|---|---:|---:|---:|
| dev | 150 | 33 | 183 |
| test | 516 | 270 | 786 |

### Kiểm tra kế hoạch trước khi dùng quota

```bash
.venv/bin/python -m benchmarks.agent.run_soict \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/soict-dev-v1 --split dev --dry-run

.venv/bin/python -m benchmarks.agent.run_soict \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/soict-test-v1 --split test --dry-run
```

`--dry-run` chỉ validate dữ liệu và ghi kế hoạch, không gọi network/model/CLI.
`--limit 2` giới hạn **base query đủ điều kiện**, sau khi lọc hint subset; vẫn chạy
mọi mức hint của hai query đó. Có thể smoke live bằng limit trước khi chạy đầy đủ.

### Chạy experiments

Với backend live đã chạy, dùng cùng lệnh phía trên và bỏ `--dry-run`.
Suite chạy `main-{split}` trước, rồi `cumulative-{split}`, mỗi experiment có
manifest/results riêng; lỗi quota dừng cả suite. Có `--experiment main`,
`--experiment cumulative` hoặc `--experiment ablations` (ablations dùng dev).
Tune/calibrate trên dev rồi khóa cấu hình trước test; suite không tự fit calibration.
Nếu bật artifact sau khi fit, dùng output directory mới vì runtime hash đã đổi.
Thứ tự code/data/config phải giữ cố định để resume. Quota attempts được retry;
failed pairs thuộc loại lỗi khác đã lưu không tự retry.

Có thể chạy trực tiếp chỉ cumulative, hoặc thêm E vào comparison:

```bash
.venv/bin/python -m benchmarks.agent.run_benchmark \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/cumulative-test-v1 \
  --split test --hint-mode cumulative --variants A,C,F

.venv/bin/python -m benchmarks.agent.run_soict \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/soict-ablations-v1 \
  --split dev --experiment ablations
```

Khóa resume là `(base query, hint stage, variant)`. Manifest hash dataset bao gồm
`hints_vi`, mode, version/construction rule, run plan, code, tolerance, seed,
timeout/poll và runtime model/controller/calibration. `evaluated_query` lưu đúng
chuỗi đã gửi; metadata gồm `base_query_id`, `hint_stage`, `hint_level` (1…H),
`hint_count` (H), `hint_fraction`, `is_full_hint`. Ground truth chỉ ở evaluator.

### Metrics và cohort

Hint đầu chưa chắc chỉ đúng moment được annotate, nên báo cáo ba tiêu chí:

- **Video R@1/R@5** là primary ở partial input, bỏ qua moment/answer.
- **Moment R@1/R@5** so với full-query moment labels, secondary ở partial input;
  QA chưa yêu cầu answer ở tiêu chí này.
- **Strict R@1/R@5/MRR** giữ evaluator chính: QA cần đúng moment + accepted answer,
  TRAKE cần đầy đủ chuỗi đúng thứ tự. Strict là primary ở Full.

Mỗi tiêu chí có `h* = min{h: Rank-1 correct}` từ các outcome độc lập. Báo mean/median
**trên các câu solved**, kèm solved N và unsolved N; câu không solved có h*=null.
Mean penalized gán H+1 cho unsolved. Early solve rate là tỷ lệ câu có h* < H trên
tất cả câu đủ lượt. Mean hint saving dùng `(H-h*)/(H-1)`, unsolved nhận 0;
chỉ lấy H≥2. Có regression rate vì thêm hint không đảm bảo success đơn điệu.

Hints-to-solve và bảng paper dùng **cùng cohort đã đủ mọi level và mọi variant**.
Failure/timeout vẫn ở mẫu số. Câu chưa chạy đủ do interruption được báo coverage,
không bị giả thành unsolved. JSON giữ thêm available/paired cohort riêng mỗi mức.
H1/H2/Full có thể so trên cùng cohort; H3 chỉ có các câu H≥4 vì H3 của câu ba hint
đã mang tên Full. Numeric curves tách theo tổng H để tránh đổi cohort giữa điểm.
Cost/tokens thiếu telemetry vẫn null; không thay phần phí chưa biết bằng zero.

Output cumulative có thêm `summary_by_hint.json`, `hint_solve.json`,
`hint_results.csv`, `CUMULATIVE_HINT_REPORT.md`. Summary/REPORT thông thường chỉ
lấy Full, không gộp các mức input vào một accuracy. Report ghi rõ runtime mock/live.

### Xuất bảng và figures cho paper (offline)

```bash
uv pip install --python .venv/bin/python -r benchmarks/agent/requirements.txt

.venv/bin/python -m benchmarks.agent.paper_report \
  --run-dir benchmarks/agent/runs/soict-test-v1/main-test

.venv/bin/python -m benchmarks.agent.paper_report \
  --run-dir benchmarks/agent/runs/soict-test-v1/cumulative-test --plots
```

Sinh `paper_tables.csv`, `paper_intervals.json`, `PAPER_REPORT.md`. Percentile
paired bootstrap 2.000 samples, seed 2026, lấy base query làm sampling unit;
báo 95% CI và paired differences so với A, tách từng task/hint level.
Các interval mang tính exploratory, chưa điều chỉnh multiple comparisons.
Không resample candidate như các mẫu độc lập. `--plots` xuất hai figures
`hint_recall` và `hint_computation` thành PDF/PNG: video/moment/strict success;
agent calls/tool calls/Jev calls/latency theo mức hint, tách tổng H, dùng cohort đủ lượt.
Main Table 1, Cumulative Table 2 và dev ablations nằm trong các thư mục experiment
riêng. Các artifacts này chỉ có giá trị thực nghiệm sau khi chạy backend live.

### Fit và sweep stopping từ raw DEV

```bash
.venv/bin/python -m benchmarks.agent.tune_stopping \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --run-dir benchmarks/agent/runs/soict-f54c83c/main-dev \
  --output benchmarks/agent/runs/soict-dev-tuning-v1
```

Không gọi model. Sinh `calibration.json` dạng bundle với profile `holistic` cho
E/D và `constraints` cho F; artifact kiểu temperature đơn cũ vẫn được hỗ trợ.
Không dùng một temperature cho cả hai aggregation. F vẫn giữ raw majority guard
cho mọi constraint; calibration không sửa nhãn constraint hoặc thay xác suất raw.
Profile có threshold/margin chung và `stopping_by_task`; controller ghi giá trị
hiệu lực/temperature cùng `stop_check` vào snapshot/trace.

Tuner chỉ nhận main DEV raw/live, kiểm tra hash dataset, loại lượt có agent/provider
failure khỏi fitting nhưng giữ failure audit. Không pool ba câu cumulative DEV
vào fitting. Sweep dùng temperature leave-target-video-component-out, giữ R@1
và R@5 trên observed prefixes, không chọn compute-saving STOP sai trên healthy DEV.
QA/TRAKE giữ threshold/margin gốc vì chỉ có hai câu mỗi loại. Nếu hạ threshold
không giảm agent calls hợp lệ thì giữ mặc định. `tuning.json` có toàn bộ sweep,
chẩn đoán probability/margin sau agent1, evidence guard, next action và lỗi quota.

Với DEV `soict-f54c83c`: E temperature≈5.4607, giữ threshold=.9/margin=.1;
F temperature≈5.7979, **T-KIS threshold=.5/margin=.1**, QA/TRAKE giữ .9/.1.
Trên 24 câu F không lỗi, prefix replay giữ R@1=.4583 và R@5=.6667, agents/query
từ 2 xuống≈1.8333. E giữ R@1=.60 nhưng vẫn 2 agents/query. Hạ E tới .5 chỉ tiết
kiệm agent trên các câu trả sai; không coi đó là evidence-aware stopping thành công.

Đây là **recorded-prefix replay**, chưa phải run live của policy mới. Calibration
có thể đổi routing Jev nên latency/accuracy sau tuning cần đọc từ TEST thực tế;
không trình bày replay như chất lượng benchmark mới. Người dùng chọn bỏ lần DEV
xác nhận do thời gian/quota và khóa artifact trước khi TEST. Sau đó không tune
theo TEST. Báo cáo này không chứng minh adaptive compute-saving của E.

Trỏ `AGENT_CALIBRATION_PATH` tới absolute path của bundle, **restart backend** để
nạp cả config và code quota/controller mới. Dùng output mới, ví dụ:

```bash
.venv/bin/python -m benchmarks.agent.run_soict \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/soict-test-tuned-v1 --split test
```

Chạy lại đúng lệnh này sau khi quota hồi để resume/retry lượt quota. Giữ nguyên
dataset/code/calibration và cấu hình trong suốt TEST. Raw DEV ban đầu giữ riêng.

### Run TEST ban đầu: raw DEV policy, tolerance 5 giây

Theo lựa chọn sau DEV confirmation, tắt calibration bundle để quay lại policy
raw DEV: `AGENT_CALIBRATION_PATH=` (rỗng), `AGENT_STOP_THRESHOLD=0.9`,
`AGENT_STOP_MARGIN=0.1`. A–F vẫn chạy các policy tương ứng; không đổi E thành F
hoặc thay architecture. Giữ budget/timeout/roles/toolset và cơ chế quota retry mới.
Restart backend để nạp `.env`. Không resume raw/tuned/tolerance=1 vào run mới.

```bash
.venv/bin/python -m benchmarks.agent.run_soict \
  --dataset benchmarks/agent/runs/workbooks-v5-submit-audited.jsonl \
  --output benchmarks/agent/runs/soict-test-raw-tol5-v1 \
  --split test --tolerance 5
```

Policy raw và tolerance=5 được khóa trước TEST. Không suy luận rằng calibration
làm giảm accuracy trên toàn corpus từ năm query confirmation; đây là lựa chọn
cấu hình DEV của người dùng. Dùng cùng tolerance khi báo cáo/so sánh các variants.
Kết quả paper cuối đã chấm lại ở tolerance=1, bổ sung retry và tách DEV/TEST/pooled
như phần đầu README; các đường dẫn tol5 ở đây là lịch sử run ban đầu.

Kiểm tra offline cho cả harness và state isolation:

```bash
.venv/bin/python -m pytest backend/tests/test_agent*.py \
  backend/tests/test_cumulative_hint_benchmark.py backend/tests/test_submit_dataset.py
```

Nguồn API đã đối chiếu:
- [OpenRouter Decisions](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
- [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
- [Codex configuration](https://learn.chatgpt.com/docs/config-file/config-reference)
