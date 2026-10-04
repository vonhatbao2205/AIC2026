# Video-group retrieval evaluation

Moment tolerance: 1s. Runtime mode: live.
Offline report from saved rankings; no new model calls. Original results and runtime manifests are preserved.

Videos are ordered by their first appearance in the frame ranking. Multiple frames from one video occupy one video slot.
Video R@k checks only video identity. Group moment R@k additionally requires a matching returned frame in a top-k video.
Group strict R@k also requires an accepted QA answer. TRAKE retains complete ordered sequence validation.
Grouped metrics inspect the full returned frame pool inside each video; they measure candidate coverage, not one-frame submission accuracy.
Tables compare only queries complete across all configured variants at each hint level. Failed agent attempts remain in the denominator if present in the source results.

Final POOLED evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Cumulative complete cohort: 27/27 base queries complete across all hint levels and configured variants.
Main level tables use this common complete cohort. H3 can have fewer queries because three-hint queries end at Full; compare total-hint-count strata for fixed-cohort curves.

## H1

Matched queries: 27.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 27 | 0.0741 | 0.4444 | 0.6296 | 0.4074 | 0.5185 | 0.2222 | 0.2593 |
| C | 27 | 0.2222 | 0.6667 | 0.9259 | 0.6667 | 0.7778 | 0.4074 | 0.4444 |
| F | 27 | 0.2222 | 0.8148 | 0.8889 | 0.8148 | 0.8148 | 0.4444 | 0.4444 |

### QA

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 12 | 0.0000 | 0.5000 | 0.6667 | 0.4167 | 0.5833 | 0.0000 | 0.0000 |
| C | 12 | 0.0833 | 0.7500 | 0.9167 | 0.7500 | 0.9167 | 0.1667 | 0.1667 |
| F | 12 | 0.0833 | 0.9167 | 0.9167 | 0.9167 | 0.9167 | 0.0833 | 0.0833 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 15 | 0.1333 | 0.4000 | 0.6000 | 0.4000 | 0.4667 | 0.4000 | 0.4667 |
| C | 15 | 0.3333 | 0.6000 | 0.9333 | 0.6000 | 0.6667 | 0.6000 | 0.6667 |
| F | 15 | 0.3333 | 0.7333 | 0.8667 | 0.7333 | 0.7333 | 0.7333 | 0.7333 |

## H2

Matched queries: 27.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 27 | 0.1481 | 0.6296 | 0.8148 | 0.5926 | 0.7037 | 0.4074 | 0.4444 |
| C | 27 | 0.3704 | 0.7778 | 0.9630 | 0.7778 | 0.9259 | 0.5556 | 0.5926 |
| F | 27 | 0.4444 | 0.8519 | 0.9630 | 0.8148 | 0.8889 | 0.5185 | 0.5556 |

### QA

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 12 | 0.0000 | 0.4167 | 0.5833 | 0.4167 | 0.5833 | 0.0000 | 0.0000 |
| C | 12 | 0.0833 | 0.6667 | 0.9167 | 0.6667 | 0.9167 | 0.1667 | 0.1667 |
| F | 12 | 0.0833 | 0.8333 | 0.9167 | 0.7500 | 0.8333 | 0.0833 | 0.0833 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 15 | 0.2667 | 0.8000 | 1.0000 | 0.7333 | 0.8000 | 0.7333 | 0.8000 |
| C | 15 | 0.6000 | 0.8667 | 1.0000 | 0.8667 | 0.9333 | 0.8667 | 0.9333 |
| F | 15 | 0.7333 | 0.8667 | 1.0000 | 0.8667 | 0.9333 | 0.8667 | 0.9333 |

## H3

Matched queries: 20.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 20 | 0.3000 | 0.7000 | 0.8000 | 0.5500 | 0.6500 | 0.3000 | 0.3500 |
| C | 20 | 0.3500 | 0.7500 | 0.9500 | 0.7500 | 0.9500 | 0.5000 | 0.6000 |
| F | 20 | 0.4500 | 0.9500 | 1.0000 | 0.9500 | 1.0000 | 0.6000 | 0.6500 |

### QA

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 9 | 0.0000 | 0.6667 | 0.7778 | 0.5556 | 0.6667 | 0.0000 | 0.0000 |
| C | 9 | 0.1111 | 0.6667 | 0.8889 | 0.6667 | 0.8889 | 0.1111 | 0.1111 |
| F | 9 | 0.1111 | 0.8889 | 1.0000 | 0.8889 | 1.0000 | 0.1111 | 0.2222 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 11 | 0.5455 | 0.7273 | 0.8182 | 0.5455 | 0.6364 | 0.5455 | 0.6364 |
| C | 11 | 0.5455 | 0.8182 | 1.0000 | 0.8182 | 1.0000 | 0.8182 | 1.0000 |
| F | 11 | 0.7273 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## FULL

Matched queries: 27.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 27 | 0.1111 | 0.5556 | 0.8148 | 0.5185 | 0.7037 | 0.2963 | 0.4444 |
| C | 27 | 0.3704 | 0.8148 | 1.0000 | 0.8148 | 0.9259 | 0.5556 | 0.5556 |
| F | 27 | 0.4444 | 0.8889 | 1.0000 | 0.8148 | 0.8889 | 0.5556 | 0.6296 |

### QA

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 12 | 0.0000 | 0.5833 | 0.6667 | 0.5000 | 0.5833 | 0.0000 | 0.0000 |
| C | 12 | 0.1667 | 0.7500 | 1.0000 | 0.7500 | 1.0000 | 0.1667 | 0.1667 |
| F | 12 | 0.1667 | 0.9167 | 1.0000 | 0.7500 | 0.8333 | 0.1667 | 0.2500 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 15 | 0.2000 | 0.5333 | 0.9333 | 0.5333 | 0.8000 | 0.5333 | 0.8000 |
| C | 15 | 0.5333 | 0.8667 | 1.0000 | 0.8667 | 0.8667 | 0.8667 | 0.8667 |
| F | 15 | 0.6667 | 0.8667 | 1.0000 | 0.8667 | 0.9333 | 0.8667 | 0.9333 |

## Grouped hints-to-solve

Each level is an independent fresh run. Moment/strict success here is computed from the returned frame pool of the top video. Unsolved queries have null first-solved hint; the penalized mean assigns H+1. Regression means a later independent level failed after an earlier success.

| Criterion | Variant | N | Solved N | Mean h* (solved) | Mean h* (penalized) | Early solve | Full success | Regression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| video | A | 27 | 22 | 1.5455 | 2.1481 | 0.7778 | 0.5556 | 0.2593 |
| moment | A | 27 | 20 | 1.6000 | 2.4074 | 0.6667 | 0.5185 | 0.2222 |
| strict | A | 27 | 13 | 1.6923 | 3.2963 | 0.4444 | 0.2963 | 0.1852 |
| video | C | 27 | 25 | 1.3600 | 1.6296 | 0.8889 | 0.8148 | 0.1111 |
| moment | C | 27 | 25 | 1.3600 | 1.6296 | 0.8889 | 0.8148 | 0.1111 |
| strict | C | 27 | 15 | 1.2667 | 2.8519 | 0.5556 | 0.5556 | 0.0000 |
| video | F | 27 | 26 | 1.2308 | 1.3704 | 0.9630 | 0.8889 | 0.1111 |
| moment | F | 27 | 26 | 1.2308 | 1.3704 | 0.9630 | 0.8148 | 0.1852 |
| strict | F | 27 | 17 | 1.4706 | 2.7037 | 0.5926 | 0.5556 | 0.0741 |
This is a snapshot of the current run. Re-export after new results arrive to update the report.
Per-query ranks and candidate pool sizes are in query_results.csv. JSON includes available and paired cohorts for each hint level.
