# Video-group retrieval evaluation

Moment tolerance: 1s. Runtime mode: live.
Offline report from saved rankings; no new model calls. Original results and runtime manifests are preserved.

Videos are ordered by their first appearance in the frame ranking. Multiple frames from one video occupy one video slot.
Video R@k checks only video identity. Group moment R@k additionally requires a matching returned frame in a top-k video.
Group strict R@k also requires an accepted QA answer. TRAKE retains complete ordered sequence validation.
Grouped metrics inspect the full returned frame pool inside each video; they measure candidate coverage, not one-frame submission accuracy.
Tables compare only queries complete across all configured variants at each hint level. Failed agent attempts remain in the denominator if present in the source results.

Final POOLED evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

## FULL

Matched queries: 111.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 111 | 0.1441 | 0.4234 | 0.7387 | 0.3964 | 0.6306 | 0.3243 | 0.5405 |
| B | 111 | 0.3694 | 0.6396 | 0.9189 | 0.6126 | 0.8919 | 0.5405 | 0.7658 |
| C | 111 | 0.4955 | 0.7748 | 0.9189 | 0.7387 | 0.8559 | 0.6577 | 0.7568 |
| D | 111 | 0.1441 | 0.4234 | 0.7387 | 0.3964 | 0.6306 | 0.3243 | 0.5405 |
| E | 111 | 0.5856 | 0.8649 | 0.9279 | 0.8288 | 0.8649 | 0.7297 | 0.7568 |
| F | 111 | 0.5135 | 0.8108 | 0.9279 | 0.7748 | 0.8649 | 0.6847 | 0.7568 |

### QA

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 21 | 0.0000 | 0.4762 | 0.6667 | 0.3810 | 0.4762 | 0.0000 | 0.0000 |
| B | 21 | 0.1905 | 0.5714 | 0.9048 | 0.5714 | 0.9048 | 0.1905 | 0.2381 |
| C | 21 | 0.2381 | 0.7143 | 0.9524 | 0.6667 | 0.8571 | 0.2381 | 0.3333 |
| D | 21 | 0.0000 | 0.4762 | 0.6667 | 0.3810 | 0.4762 | 0.0000 | 0.0000 |
| E | 21 | 0.3810 | 0.9048 | 0.9524 | 0.9048 | 0.9524 | 0.3810 | 0.3810 |
| F | 21 | 0.2857 | 0.8095 | 0.9524 | 0.7619 | 0.8571 | 0.2857 | 0.2857 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 80 | 0.1875 | 0.4375 | 0.7750 | 0.4375 | 0.7250 | 0.4375 | 0.7250 |
| B | 80 | 0.4500 | 0.6875 | 0.9250 | 0.6875 | 0.9250 | 0.6875 | 0.9250 |
| C | 80 | 0.5875 | 0.8125 | 0.9250 | 0.8125 | 0.9125 | 0.8125 | 0.9125 |
| D | 80 | 0.1875 | 0.4375 | 0.7750 | 0.4375 | 0.7250 | 0.4375 | 0.7250 |
| E | 80 | 0.6625 | 0.8750 | 0.9250 | 0.8625 | 0.9000 | 0.8625 | 0.9000 |
| F | 80 | 0.5875 | 0.8250 | 0.9250 | 0.8250 | 0.9125 | 0.8250 | 0.9125 |

### TRAKE

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 10 | 0.1000 | 0.2000 | 0.6000 | 0.1000 | 0.2000 | 0.1000 | 0.2000 |
| B | 10 | 0.1000 | 0.4000 | 0.9000 | 0.1000 | 0.6000 | 0.1000 | 0.6000 |
| C | 10 | 0.3000 | 0.6000 | 0.8000 | 0.3000 | 0.4000 | 0.3000 | 0.4000 |
| D | 10 | 0.1000 | 0.2000 | 0.6000 | 0.1000 | 0.2000 | 0.1000 | 0.2000 |
| E | 10 | 0.4000 | 0.7000 | 0.9000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 |
| F | 10 | 0.4000 | 0.7000 | 0.9000 | 0.4000 | 0.5000 | 0.4000 | 0.5000 |

This is a snapshot of the current run. Re-export after new results arrive to update the report.
Per-query ranks and candidate pool sizes are in query_results.csv. JSON includes available and paired cohorts for each hint level.
