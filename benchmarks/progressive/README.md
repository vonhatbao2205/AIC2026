# Progressive benchmark

`run_benchmark.py` runs the production `ProgressiveEngine` directly. It never calls DRES or supplies labels to retrieval. Main configuration: PE-only, InfoShot++, all-corpus scope, global depth 200, fixed PHM defaults. Seven methods: cumulative, latest, full-union Hint-RRF, PHM without rescue, PHM, arithmetic PHM, and deeper cumulative.

## Workflow

Run from the repository root using the backend environment. The backend `.env` supplies credentials; output manifests never copy settings/secrets.

```sh
backend/.venv/bin/python benchmarks/progressive/annotations/validate_annotations.py --metadata-only
backend/.venv/bin/python benchmarks/progressive/run_benchmark.py smoke \
  --out benchmarks/progressive/runs/my-live-smoke
backend/.venv/bin/python benchmarks/progressive/run_benchmark.py prepare \
  --lock benchmarks/progressive/locks/pilot.json \
  --dataset-revision YOUR_INDEX_MANIFEST_REVISION --model-revision YOUR_PE_REVISION \
  --deep-top-k 800
backend/.venv/bin/python benchmarks/progressive/run_benchmark.py run \
  --lock benchmarks/progressive/locks/pilot.json --split dev \
  --out benchmarks/progressive/runs/dev-pilot
```

`prepare` freezes heuristic parser output and translations for each delta/cumulative text. Translation failures abort preparation. No retrieval runs during preparation. Each method receives only its current prefix and has independent vectors/retrieval caches. Queries and method order are seeded; runs are sequential. No shared retrieval warm-up is included, so report measured cache/cold-start conditions, not isolated encoder latency.

The default seed 20260921 yields **15 dev / 60 eval queries** from 75 eligible owner-approved records. Connected components of target videos bind multi-target queries together. Approval/provenance remains owner-normalized, not independently video-verified. Run full annotation validation when local media integrity also needs attestation.

On dev, compare cumulative depths 200/400/800/1000 against PHM median latency. Create a separate lock/output for each pilot. Choose the closest depth, then prepare the final lock with that depth and run a complete dev pass. Default 800 is only a pilot value, **not a claim of matched compute**. Archive the depth-selection evidence. Do not tune on eval.

```sh
backend/.venv/bin/python benchmarks/progressive/run_benchmark.py run \
  --lock benchmarks/progressive/locks/final.json --split eval \
  --dev-run benchmarks/progressive/runs/dev-final \
  --out benchmarks/progressive/runs/eval-final
```

Eval requires a complete, healthy dev run with the same lock and source digest, live mode, and operator-attested dataset/model revision labels. Locks bind GT bytes, split, source, translations, method/configuration and collection names. Revision labels do not independently verify remote index/model bytes: freeze those services operationally and retain their manifests. Output directories are exclusive and cannot be overwritten. The workflow still relies on the operator to run eval once; creating another directory does not technically prevent repeated evaluation.

`--limit 1` is available for dev pipeline checks, which cannot qualify as the full dev prerequisite. `smoke` uses one dev query, PHM and three hints; it is not an effectiveness benchmark. The runner currently targets PE-only visual retrieval; PE+Qwen/hybrid tables need separate configuration support and validation.

## Artifacts and metric definitions

- `manifest.json`: code/lock provenance, effective configuration, frozen parser plans and split.
- `records.jsonl`: query/method/prefix metrics, status, retrieval wall latency, work counters and trace paths. Errors/timeouts stay in denominators as zero-hit prefixes. Degraded results are flagged and prevent a dev run from qualifying for eval.
- `traces/`: full engine snapshots, raw channel observations, local requests, frontiers, hint ledger and ranking before rescue.
- `summary.json`: mean prefix video MRR (average prefixes within each query first), per-turn Hit@1/5/10, first/stable-correct turns (null means unsolved), top-1 churn, rescue recovery/harm, p50/p95 latency, failures/timeouts, work totals and 10,000 paired target-video-group bootstrap samples versus cumulative.

Video ranking is truncated at the configured memory/output ranking cap (250 by default); missing videos have reciprocal rank zero. Hint-RRF scores its **entire observed union before this cap**, without historical evidence eviction or survivor protection. Moment Hit@K takes exactly the first preferred frame of each of the first K displayed video groups, converts seconds to milliseconds and checks `[start_ms,end_ms)`. Historical/support frames are not searched for an oracle match. This is a deterministic preferred-moment policy, not a simulation of human selection. Raw traces preserve other moments for separately declared analyses.

Rescue recovery counts absent-before / present-after ranking; harmful rescue counts a worse rank or disappearance. Stable-correct turn uses future turns only for retrospective analysis. Query-level failed attempts may be retried while rebuilding a later prefix, and their extra work is recorded in that later revision budget.

## Verified in this change

- Backend suite: 480 passed.
- Mock end-to-end replay: all seven methods × three hints passed; these are infrastructure checks, not paper results.
- Live PE-only PHM smoke on `query-p1-1-kis` (dev): H1 2892 ms, H2 3996 ms, H3 4304 ms; all healthy. Global/local calls: 1/0, 2/12, 2/16. Uses `global_frontier_v1`.
- Metadata annotation validation: 81 records, 77 owner-approved, 75 eligible TKIS, 90 accepted intervals.

No final evaluation or effectiveness claim has been made.
