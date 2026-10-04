# Final SOICT benchmark: DEV, TEST and pooled

Tolerance 1s; raw policy; calibration disabled. All planned results are present with zero execution errors or fallback. No queries remain excluded. Reports are exported offline with no model calls.

| Cohort | Main queries | Main runs | Cumulative queries | Cumulative hint stages | Cumulative runs |
|---|---:|---:|---:|---:|---:|
| DEV | 25 | 150 | 3 | 11 | 33 |
| TEST | 86 | 516 | 24 | 90 | 270 |
| POOLED | 111 | 666 | 27 | 101 | 303 |

DEV and TEST retain the membership of the original reviewed dataset, including retries originally executed using the combined TEST dataset. Pooled is their descriptive combined result, not a relabelling of the source split.

## Reports

Each cohort contains main/ and cumulative/. Open PAPER_REPORT.md for strict paper tables and bootstrap intervals; video-report/REPORT.md for Video / Group moment / Group strict R@1/R@5/MRR.
Cumulative includes CUMULATIVE_HINT_REPORT.md, summary_by_hint.json, hint_results.csv and fixed-cohort PDF/PNG hint curves. All hint levels are independent fresh runs; no cross-hint memory is used.

## Main strict R@1

| Variant | DEV (25) | TEST (86) | Pooled (111) |
|---|---:|---:|---:|
| A | 12.00% | 15.12% | 14.41% |
| B | 28.00% | 39.53% | 36.94% |
| C | 44.00% | 51.16% | 49.55% |
| D | 12.00% | 15.12% | 14.41% |
| E | 60.00% | 58.14% | 58.56% |
| F | 48.00% | 52.33% | 51.35% |

## Cumulative Full strict R@1

| Variant | DEV (3) | TEST (24) | Pooled (27) |
|---|---:|---:|---:|
| A | 0.00% | 12.50% | 11.11% |
| C | 66.67% | 33.33% | 37.04% |
| F | 66.67% | 41.67% | 44.44% |

Original source outputs are preserved. Interrupted attempts are copied under provenance/ and excluded from the final accuracy denominator; completed successful retries are included. Source hashes and complete-run checks are in SUITE_SUMMARY.json and each VALIDATION.json.
DEV cumulative has only three base queries; bootstrap intervals there are exploratory. Combined CSV: ALL_PAPER_TABLES.csv.

## Versioned report snapshot

This directory is the compact Git snapshot of `benchmarks/agent/runs/soict-final-dev-test-pooled-tol1-v1/`. Raw `results.jsonl`, evaluation dataset JSONL copies and archived retry traces stay in the ignored local run directory. Use the local runs with the offline export tools to regenerate reports; this snapshot is not a live runner output.

`dataset.json` records the reviewed 111-query source labels and original split membership; its source JSONL SHA-256 is preserved. `SNAPSHOT_MANIFEST.json` records checksums of copied artifacts. Runtime and source paths inside frozen manifests refer to the original local export, not this report-only snapshot.
