# Video-group retrieval evaluation

Moment tolerance: 1s. Runtime mode: live.
Offline report from saved rankings; no new model calls. Original results and runtime manifests are preserved.

Videos are ordered by their first appearance in the frame ranking. Multiple frames from one video occupy one video slot.
Video R@k checks only video identity. Group moment R@k additionally requires a matching returned frame in a top-k video.
Group strict R@k also requires an accepted QA answer. TRAKE retains complete ordered sequence validation.
Grouped metrics inspect the full returned frame pool inside each video; they measure candidate coverage, not one-frame submission accuracy.
Tables compare only queries complete across all configured variants at each hint level. Failed agent attempts remain in the denominator if present in the source results.

Final DEV evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Cumulative complete cohort: 3/3 base queries complete across all hint levels and configured variants.
Main level tables use this common complete cohort. H3 can have fewer queries because three-hint queries end at Full; compare total-hint-count strata for fixed-cohort curves.

## H1

Matched queries: 3.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 3 | 0.0000 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |
| C | 3 | 0.0000 | 0.6667 | 1.0000 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |
| F | 3 | 0.0000 | 0.6667 | 1.0000 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 3 | 0.0000 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |
| C | 3 | 0.0000 | 0.6667 | 1.0000 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |
| F | 3 | 0.0000 | 0.6667 | 1.0000 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |

## H2

Matched queries: 3.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 3 | 0.3333 | 0.6667 | 1.0000 | 0.6667 | 1.0000 | 0.6667 | 1.0000 |
| C | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| F | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 3 | 0.3333 | 0.6667 | 1.0000 | 0.6667 | 1.0000 | 0.6667 | 1.0000 |
| C | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| F | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## H3

Matched queries: 2.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2 | 0.5000 | 1.0000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| C | 2 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| F | 2 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2 | 0.5000 | 1.0000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| C | 2 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| F | 2 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## FULL

Matched queries: 3.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 3 | 0.0000 | 0.3333 | 1.0000 | 0.3333 | 0.6667 | 0.3333 | 0.6667 |
| C | 3 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| F | 3 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 3 | 0.0000 | 0.3333 | 1.0000 | 0.3333 | 0.6667 | 0.3333 | 0.6667 |
| C | 3 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| F | 3 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Grouped hints-to-solve

Each level is an independent fresh run. Moment/strict success here is computed from the returned frame pool of the top video. Unsolved queries have null first-solved hint; the penalized mean assigns H+1. Regression means a later independent level failed after an earlier success.

| Criterion | Variant | N | Solved N | Mean h* (solved) | Mean h* (penalized) | Early solve | Full success | Regression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| video | A | 3 | 3 | 1.3333 | 1.3333 | 1.0000 | 0.3333 | 0.6667 |
| moment | A | 3 | 3 | 1.3333 | 1.3333 | 1.0000 | 0.3333 | 0.6667 |
| strict | A | 3 | 3 | 1.3333 | 1.3333 | 1.0000 | 0.3333 | 0.6667 |
| video | C | 3 | 3 | 1.3333 | 1.3333 | 1.0000 | 1.0000 | 0.0000 |
| moment | C | 3 | 3 | 1.3333 | 1.3333 | 1.0000 | 1.0000 | 0.0000 |
| strict | C | 3 | 3 | 1.3333 | 1.3333 | 1.0000 | 1.0000 | 0.0000 |
| video | F | 3 | 3 | 1.3333 | 1.3333 | 1.0000 | 1.0000 | 0.0000 |
| moment | F | 3 | 3 | 1.3333 | 1.3333 | 1.0000 | 1.0000 | 0.0000 |
| strict | F | 3 | 3 | 1.3333 | 1.3333 | 1.0000 | 1.0000 | 0.0000 |
This is a snapshot of the current run. Re-export after new results arrive to update the report.
Per-query ranks and candidate pool sizes are in query_results.csv. JSON includes available and paired cohorts for each hint level.
