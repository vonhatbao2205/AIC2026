# Video-group retrieval evaluation

Moment tolerance: 1s. Runtime mode: live.
Offline report from saved rankings; no new model calls. Original results and runtime manifests are preserved.

Videos are ordered by their first appearance in the frame ranking. Multiple frames from one video occupy one video slot.
Video R@k checks only video identity. Group moment R@k additionally requires a matching returned frame in a top-k video.
Group strict R@k also requires an accepted QA answer. TRAKE retains complete ordered sequence validation.
Grouped metrics inspect the full returned frame pool inside each video; they measure candidate coverage, not one-frame submission accuracy.
Tables compare only queries complete across all configured variants at each hint level. Failed agent attempts remain in the denominator if present in the source results.

Final DEV evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

## FULL

Matched queries: 25.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 25 | 0.1200 | 0.3200 | 0.6800 | 0.2800 | 0.6000 | 0.2800 | 0.6000 |
| B | 25 | 0.2800 | 0.5600 | 0.8800 | 0.5600 | 0.8800 | 0.5200 | 0.8000 |
| C | 25 | 0.4400 | 0.7200 | 0.8800 | 0.7200 | 0.8000 | 0.6400 | 0.7200 |
| D | 25 | 0.1200 | 0.3200 | 0.6800 | 0.2800 | 0.6000 | 0.2800 | 0.6000 |
| E | 25 | 0.6000 | 0.8800 | 0.9200 | 0.8800 | 0.8800 | 0.8000 | 0.8000 |
| F | 25 | 0.4800 | 0.7200 | 0.8800 | 0.6800 | 0.8400 | 0.6000 | 0.7600 |

### QA

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B | 2 | 0.0000 | 0.5000 | 1.0000 | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| C | 2 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | 2 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E | 2 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| F | 2 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 21 | 0.0952 | 0.2857 | 0.7143 | 0.2857 | 0.6667 | 0.2857 | 0.6667 |
| B | 21 | 0.2857 | 0.5714 | 0.8571 | 0.5714 | 0.8571 | 0.5714 | 0.8571 |
| C | 21 | 0.4762 | 0.7143 | 0.8571 | 0.7143 | 0.8095 | 0.7143 | 0.8095 |
| D | 21 | 0.0952 | 0.2857 | 0.7143 | 0.2857 | 0.6667 | 0.2857 | 0.6667 |
| E | 21 | 0.6667 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 |
| F | 21 | 0.5714 | 0.7143 | 0.8571 | 0.7143 | 0.8571 | 0.7143 | 0.8571 |

### TRAKE

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| B | 2 | 0.5000 | 0.5000 | 1.0000 | 0.5000 | 1.0000 | 0.5000 | 1.0000 |
| C | 2 | 0.5000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| D | 2 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| E | 2 | 0.5000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| F | 2 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 |

This is a snapshot of the current run. Re-export after new results arrive to update the report.
Per-query ranks and candidate pool sizes are in query_results.csv. JSON includes available and paired cohorts for each hint level.
