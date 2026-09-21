# Progressive benchmark

`run_benchmark.py` runs the production `ProgressiveEngine` directly. It never calls DRES or supplies labels to retrieval. Main configuration: PE-only, InfoShot++, all-corpus scope, global depth 200, fixed PHM defaults. Visual profiles and suites are locked independently:

| Profile | Default suite | Methods |
|---|---|---|
| `pe` | `main` | cumulative, full-union Hint-RRF, **dual_view**, PHM without rescue, PHM, deeper cumulative |
| `pe-qwen` | `replication` | cumulative, Hint-RRF, PHM without rescue, PHM |
| Either | `--suite extended` | All eight methods, including latest and arithmetic PHM |

`dual_view` fuses current delta and cumulative rank evidence with weights 0.5/0.5, without historical evidence or rescue. It separates the effect of two query views from persistent PHM memory. PHM scoring is unchanged.

## Workflow

Run from the repository root using the backend environment. The backend `.env` supplies credentials; output manifests never copy settings/secrets.

```sh
backend/.venv/bin/python benchmarks/progressive/annotations/validate_annotations.py --metadata-only
backend/.venv/bin/python benchmarks/progressive/run_benchmark.py smoke \
  --out benchmarks/progressive/runs/my-live-smoke
backend/.venv/bin/python benchmarks/progressive/run_benchmark.py prepare \
  --lock benchmarks/progressive/locks/pilot.json \
  --dataset-revision YOUR_INDEX_MANIFEST_REVISION --model-revision pe=YOUR_PE_REVISION \
  --deep-top-k 800
backend/.venv/bin/python benchmarks/progressive/run_benchmark.py run \
  --lock benchmarks/progressive/locks/pilot.json --split dev \
  --out benchmarks/progressive/runs/dev-pilot
```

`prepare` freezes heuristic parser output and translations for each delta/cumulative text. Translation failures abort preparation. No retrieval runs during preparation. Each method receives only its current prefix and has independent vectors/retrieval caches. Queries and method order are seeded; runs are sequential. No shared retrieval warm-up is included, so report measured cache/cold-start conditions, not isolated encoder latency.

The default seed 20260921 yields **15 dev / 60 eval queries** from 75 eligible owner-approved records. Connected components of target videos bind multi-target queries together. Approval/provenance remains owner-normalized, not independently video-verified. Run full annotation validation when local media integrity also needs attestation.

On dev, compare cumulative depths 200/400/800/1000 against PHM median latency. Use `--reuse-plans` so every pilot has byte-identical frozen parser/translation output. Keep revision labels and other options identical:

```sh
for depth in 200 400 800 1000; do
  backend/.venv/bin/python benchmarks/progressive/run_benchmark.py prepare \
    --profile pe --suite main --deep-top-k "$depth" \
    --reuse-plans benchmarks/progressive/locks/pilot.json \
    --dataset-revision YOUR_INDEX_MANIFEST_REVISION --model-revision pe=YOUR_PE_REVISION \
    --lock "benchmarks/progressive/locks/depth-$depth.json"
  backend/.venv/bin/python benchmarks/progressive/run_benchmark.py run \
    --lock "benchmarks/progressive/locks/depth-$depth.json" --split dev \
    --out "benchmarks/progressive/runs/dev-$depth"
done
backend/.venv/bin/python benchmarks/progressive/run_benchmark.py select-depth \
  --depth-runs benchmarks/progressive/runs/dev-200 benchmarks/progressive/runs/dev-400 \
               benchmarks/progressive/runs/dev-800 benchmarks/progressive/runs/dev-1000 \
  --out benchmarks/progressive/runs/depth-selection.json
```

The selector accepts only complete healthy live **dev** runs whose locks differ solely in `deep_top_k`. It compares each deep-cumulative p50 against the median of the four PHM p50s, breaking ties toward the shallower depth. It never uses effectiveness metrics to select depth. Keep the report, including the latency gap. Default 800 is only a pilot value, **not a claim of matched compute**.

Prepare `final.json` with the selected depth, the same revisions, and `--reuse-plans .../pilot.json`, then run a complete dev pass to `dev-final` before the following eval command. Do not tune on eval.

```sh
backend/.venv/bin/python benchmarks/progressive/run_benchmark.py run \
  --lock benchmarks/progressive/locks/final.json --split eval \
  --dev-run benchmarks/progressive/runs/dev-final \
  --out benchmarks/progressive/runs/eval-final
```

Eval requires a complete, healthy dev run with the same lock and source digest, live mode, and operator-attested dataset/model revision labels. Locks bind GT bytes, split, source, translations, method/configuration and collection names. Revision labels do not independently verify remote index/model bytes: freeze those services operationally and retain their manifests. Output directories are exclusive and cannot be overwritten. The workflow still relies on the operator to run eval once; creating another directory does not technically prevent repeated evaluation.

`--limit 1` is available for dev pipeline checks, which cannot qualify as the full dev prerequisite. `smoke` uses one dev query, PHM and three hints; it is not an effectiveness benchmark. Both visual profiles are supported. Every live run checks each selected encoder and its Milvus collection before retrieving queries. A failed preflight stops before creating a benchmark output directory. Every prefix also records delta/cumulative channel statuses; a missing/failed selected model marks that prefix degraded. Smoke writes a sanitized `preflight.json` and exits unsuccessfully on unhealthy/degraded results.

## PE+Qwen replication

```sh
backend/.venv/bin/python benchmarks/progressive/run_benchmark.py smoke \
  --profile pe-qwen --out benchmarks/progressive/runs/qwen-smoke
backend/.venv/bin/python benchmarks/progressive/run_benchmark.py prepare \
  --profile pe-qwen --suite replication \
  --dataset-revision YOUR_INDEX_MANIFEST_REVISION \
  --model-revision pe=YOUR_PE_REVISION --model-revision qwen3_vl=YOUR_QWEN_REVISION \
  --reuse-plans benchmarks/progressive/locks/final.json \
  --lock benchmarks/progressive/locks/pe-qwen.json
backend/.venv/bin/python benchmarks/progressive/run_benchmark.py run \
  --lock benchmarks/progressive/locks/pe-qwen.json --split dev \
  --out benchmarks/progressive/runs/qwen-dev
```

Then use that lock and `--dev-run .../qwen-dev` for a separate eval run. Eval requires a revision for **each selected model**. A bare `--model-revision REV` remains a PE shorthand. `run` derives profile/suite from the lock; explicitly supplied conflicting values are rejected. Version 1 locks must be regenerated because the source and method table changed.

These tables evaluate controlled visual PHM and stronger-visual-retriever replication, not the entire AIC2026 search stack. OCR/ASR/audio remain excluded (`hybrid=False`). TARA, reranker and expansion remain disabled. In particular, TARA clip-level evidence is not preserved in PHM trace/regroup; it needs separate integration before it can enter this benchmark.

## Artifacts and metric definitions

- `manifest.json`: code/lock provenance, effective configuration, frozen parser plans and split.
- `records.jsonl`: query/method/prefix metrics, status, retrieval wall latency, work counters and trace paths. Errors/timeouts stay in denominators as zero-hit prefixes. Degraded results are flagged and prevent a dev run from qualifying for eval.
- `traces/`: full engine snapshots, raw channel observations, local requests, frontiers, hint ledger and ranking before rescue.
- `summary.json`: mean prefix video MRR (average prefixes within each query first), per-turn Hit@1/5/10, first/stable-correct turns (null means unsolved), top-1 churn, rescue recovery/harm, p50/p95 latency, failures/timeouts, work totals and 10,000 paired target-video-group bootstrap samples versus cumulative.

Video ranking is truncated at the configured memory/output ranking cap (250 by default); missing videos have reciprocal rank zero. Hint-RRF scores its **entire observed union before this cap**, without historical evidence eviction or survivor protection. Moment Hit@K takes exactly the first preferred frame of each of the first K displayed video groups, converts seconds to milliseconds and checks `[start_ms,end_ms)`. Historical/support frames are not searched for an oracle match. This is a deterministic preferred-moment policy, not a simulation of human selection. Raw traces preserve other moments for separately declared analyses.

Rescue recovery counts absent-before / present-after ranking; harmful rescue counts a worse rank or disappearance. Stable-correct turn uses future turns only for retrospective analysis. Query-level failed attempts may be retried while rebuilding a later prefix, and their extra work is recorded in that later revision budget.

## Verified in this change

- Backend suite: 488 passed.
- Mock end-to-end replay: PE extended (8 methods × 3 hints) and PE+Qwen replication (4 methods × 3 hints) passed; these are infrastructure checks, not paper results.
- Live PE-only PHM smoke on `query-p1-1-kis` (dev): H1 2919 ms, H2 5390 ms, H3 8492 ms; all healthy. Global/local calls: 1/0, 2/12, 2/16. Uses `global_frontier_v1`.
- Live PE+Qwen PHM smoke (same dev query): H1 2236 ms, H2 12780 ms, H3 13452 ms; both selected channels healthy in delta/cumulative views at every turn, no degradation. Global/local calls: 2/0, 4/24, 4/32. These smoke latencies are not a controlled model-speed comparison.
- Metadata annotation validation: 81 records, 77 owner-approved, 75 eligible TKIS, 90 accepted intervals.

No final evaluation or effectiveness claim has been made.
