# Video-group retrieval evaluation

Moment tolerance: 1s. Runtime mode: live.
Offline report from saved rankings; no new model calls. Original results and runtime manifests are preserved.

Videos are ordered by their first appearance in the frame ranking. Multiple frames from one video occupy one video slot.
Video R@k checks only video identity. Group moment R@k additionally requires a matching returned frame in a top-k video.
Group strict R@k also requires an accepted QA answer. TRAKE retains complete ordered sequence validation.
Grouped metrics inspect the full returned frame pool inside each video; they measure candidate coverage, not one-frame submission accuracy.
Tables compare only queries complete across all configured variants at each hint level. Failed agent attempts remain in the denominator if present in the source results.

Final TEST evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Cumulative complete cohort: 24/24 base queries complete across all hint levels and configured variants.
Main level tables use this common complete cohort. H3 can have fewer queries because three-hint queries end at Full; compare total-hint-count strata for fixed-cohort curves.

## H1

Matched queries: 24.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 24 | 0.0833 | 0.4167 | 0.6250 | 0.3750 | 0.5000 | 0.1667 | 0.2083 |
| C | 24 | 0.2500 | 0.6667 | 0.9167 | 0.6667 | 0.7917 | 0.3750 | 0.4167 |
| F | 24 | 0.2500 | 0.8333 | 0.8750 | 0.8333 | 0.8333 | 0.4167 | 0.4167 |

### QA

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 12 | 0.0000 | 0.5000 | 0.6667 | 0.4167 | 0.5833 | 0.0000 | 0.0000 |
| C | 12 | 0.0833 | 0.7500 | 0.9167 | 0.7500 | 0.9167 | 0.1667 | 0.1667 |
| F | 12 | 0.0833 | 0.9167 | 0.9167 | 0.9167 | 0.9167 | 0.0833 | 0.0833 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 12 | 0.1667 | 0.3333 | 0.5833 | 0.3333 | 0.4167 | 0.3333 | 0.4167 |
| C | 12 | 0.4167 | 0.5833 | 0.9167 | 0.5833 | 0.6667 | 0.5833 | 0.6667 |
| F | 12 | 0.4167 | 0.7500 | 0.8333 | 0.7500 | 0.7500 | 0.7500 | 0.7500 |

## H2

Matched queries: 24.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 24 | 0.1250 | 0.6250 | 0.7917 | 0.5833 | 0.6667 | 0.3750 | 0.3750 |
| C | 24 | 0.2917 | 0.7500 | 0.9583 | 0.7500 | 0.9167 | 0.5000 | 0.5417 |
| F | 24 | 0.3750 | 0.8333 | 0.9583 | 0.7917 | 0.8750 | 0.4583 | 0.5000 |

### QA

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 12 | 0.0000 | 0.4167 | 0.5833 | 0.4167 | 0.5833 | 0.0000 | 0.0000 |
| C | 12 | 0.0833 | 0.6667 | 0.9167 | 0.6667 | 0.9167 | 0.1667 | 0.1667 |
| F | 12 | 0.0833 | 0.8333 | 0.9167 | 0.7500 | 0.8333 | 0.0833 | 0.0833 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 12 | 0.2500 | 0.8333 | 1.0000 | 0.7500 | 0.7500 | 0.7500 | 0.7500 |
| C | 12 | 0.5000 | 0.8333 | 1.0000 | 0.8333 | 0.9167 | 0.8333 | 0.9167 |
| F | 12 | 0.6667 | 0.8333 | 1.0000 | 0.8333 | 0.9167 | 0.8333 | 0.9167 |

## H3

Matched queries: 18.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 18 | 0.2778 | 0.6667 | 0.7778 | 0.5556 | 0.6667 | 0.2778 | 0.3333 |
| C | 18 | 0.3333 | 0.7222 | 0.9444 | 0.7222 | 0.9444 | 0.4444 | 0.5556 |
| F | 18 | 0.4444 | 0.9444 | 1.0000 | 0.9444 | 1.0000 | 0.5556 | 0.6111 |

### QA

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 9 | 0.0000 | 0.6667 | 0.7778 | 0.5556 | 0.6667 | 0.0000 | 0.0000 |
| C | 9 | 0.1111 | 0.6667 | 0.8889 | 0.6667 | 0.8889 | 0.1111 | 0.1111 |
| F | 9 | 0.1111 | 0.8889 | 1.0000 | 0.8889 | 1.0000 | 0.1111 | 0.2222 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 9 | 0.5556 | 0.6667 | 0.7778 | 0.5556 | 0.6667 | 0.5556 | 0.6667 |
| C | 9 | 0.5556 | 0.7778 | 1.0000 | 0.7778 | 1.0000 | 0.7778 | 1.0000 |
| F | 9 | 0.7778 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## FULL

Matched queries: 24.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 24 | 0.1250 | 0.5833 | 0.7917 | 0.5417 | 0.7083 | 0.2917 | 0.4167 |
| C | 24 | 0.3333 | 0.7917 | 1.0000 | 0.7917 | 0.9167 | 0.5000 | 0.5000 |
| F | 24 | 0.4167 | 0.8750 | 1.0000 | 0.7917 | 0.8750 | 0.5000 | 0.5833 |

### QA

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 12 | 0.0000 | 0.5833 | 0.6667 | 0.5000 | 0.5833 | 0.0000 | 0.0000 |
| C | 12 | 0.1667 | 0.7500 | 1.0000 | 0.7500 | 1.0000 | 0.1667 | 0.1667 |
| F | 12 | 0.1667 | 0.9167 | 1.0000 | 0.7500 | 0.8333 | 0.1667 | 0.2500 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 12 | 0.2500 | 0.5833 | 0.9167 | 0.5833 | 0.8333 | 0.5833 | 0.8333 |
| C | 12 | 0.5000 | 0.8333 | 1.0000 | 0.8333 | 0.8333 | 0.8333 | 0.8333 |
| F | 12 | 0.6667 | 0.8333 | 1.0000 | 0.8333 | 0.9167 | 0.8333 | 0.9167 |

## Grouped hints-to-solve

Each level is an independent fresh run. Moment/strict success here is computed from the returned frame pool of the top video. Unsolved queries have null first-solved hint; the penalized mean assigns H+1. Regression means a later independent level failed after an earlier success.

| Criterion | Variant | N | Solved N | Mean h* (solved) | Mean h* (penalized) | Early solve | Full success | Regression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| video | A | 24 | 19 | 1.5789 | 2.2500 | 0.7500 | 0.5833 | 0.2083 |
| moment | A | 24 | 17 | 1.6471 | 2.5417 | 0.6250 | 0.5417 | 0.1667 |
| strict | A | 24 | 10 | 1.8000 | 3.5417 | 0.3750 | 0.2917 | 0.1250 |
| video | C | 24 | 22 | 1.3636 | 1.6667 | 0.8750 | 0.7917 | 0.1250 |
| moment | C | 24 | 22 | 1.3636 | 1.6667 | 0.8750 | 0.7917 | 0.1250 |
| strict | C | 24 | 12 | 1.2500 | 3.0417 | 0.5000 | 0.5000 | 0.0000 |
| video | F | 24 | 23 | 1.2174 | 1.3750 | 0.9583 | 0.8750 | 0.1250 |
| moment | F | 24 | 23 | 1.2174 | 1.3750 | 0.9583 | 0.7917 | 0.2083 |
| strict | F | 24 | 14 | 1.5000 | 2.8750 | 0.5417 | 0.5000 | 0.0833 |
This is a snapshot of the current run. Re-export after new results arrive to update the report.
Per-query ranks and candidate pool sizes are in query_results.csv. JSON includes available and paired cohorts for each hint level.
