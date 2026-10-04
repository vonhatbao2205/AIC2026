# Video-group retrieval evaluation

Moment tolerance: 1s. Runtime mode: live.
Offline report from saved rankings; no new model calls. Original results and runtime manifests are preserved.

Videos are ordered by their first appearance in the frame ranking. Multiple frames from one video occupy one video slot.
Video R@k checks only video identity. Group moment R@k additionally requires a matching returned frame in a top-k video.
Group strict R@k also requires an accepted QA answer. TRAKE retains complete ordered sequence validation.
Grouped metrics inspect the full returned frame pool inside each video; they measure candidate coverage, not one-frame submission accuracy.
Tables compare only queries complete across all configured variants at each hint level. Failed agent attempts remain in the denominator if present in the source results.

Final TEST evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

## FULL

Matched queries: 86.

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 86 | 0.1512 | 0.4535 | 0.7558 | 0.4302 | 0.6395 | 0.3372 | 0.5233 |
| B | 86 | 0.3953 | 0.6628 | 0.9302 | 0.6279 | 0.8953 | 0.5465 | 0.7558 |
| C | 86 | 0.5116 | 0.7907 | 0.9302 | 0.7442 | 0.8721 | 0.6628 | 0.7674 |
| D | 86 | 0.1512 | 0.4535 | 0.7558 | 0.4302 | 0.6395 | 0.3372 | 0.5233 |
| E | 86 | 0.5814 | 0.8605 | 0.9302 | 0.8140 | 0.8605 | 0.7093 | 0.7442 |
| F | 86 | 0.5233 | 0.8372 | 0.9419 | 0.8023 | 0.8721 | 0.7093 | 0.7558 |

### QA

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 19 | 0.0000 | 0.4737 | 0.6842 | 0.4211 | 0.5263 | 0.0000 | 0.0000 |
| B | 19 | 0.2105 | 0.5789 | 0.8947 | 0.5789 | 0.8947 | 0.2105 | 0.2632 |
| C | 19 | 0.2632 | 0.6842 | 0.9474 | 0.6316 | 0.8421 | 0.2632 | 0.3684 |
| D | 19 | 0.0000 | 0.4737 | 0.6842 | 0.4211 | 0.5263 | 0.0000 | 0.0000 |
| E | 19 | 0.4211 | 0.8947 | 0.9474 | 0.8947 | 0.9474 | 0.4211 | 0.4211 |
| F | 19 | 0.3158 | 0.7895 | 0.9474 | 0.7368 | 0.8421 | 0.3158 | 0.3158 |

### T-KIS

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 59 | 0.2203 | 0.4915 | 0.7966 | 0.4915 | 0.7458 | 0.4915 | 0.7458 |
| B | 59 | 0.5085 | 0.7288 | 0.9492 | 0.7288 | 0.9492 | 0.7288 | 0.9492 |
| C | 59 | 0.6271 | 0.8475 | 0.9492 | 0.8475 | 0.9492 | 0.8475 | 0.9492 |
| D | 59 | 0.2203 | 0.4915 | 0.7966 | 0.4915 | 0.7458 | 0.4915 | 0.7458 |
| E | 59 | 0.6610 | 0.8644 | 0.9322 | 0.8475 | 0.8983 | 0.8475 | 0.8983 |
| F | 59 | 0.5932 | 0.8644 | 0.9492 | 0.8644 | 0.9322 | 0.8644 | 0.9322 |

### TRAKE

| Variant | N | Strict R@1 | Video R@1 | Video R@5 | Group moment R@1 | Group moment R@5 | Group strict R@1 | Group strict R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 8 | 0.0000 | 0.1250 | 0.6250 | 0.0000 | 0.1250 | 0.0000 | 0.1250 |
| B | 8 | 0.0000 | 0.3750 | 0.8750 | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| C | 8 | 0.2500 | 0.6250 | 0.7500 | 0.2500 | 0.3750 | 0.2500 | 0.3750 |
| D | 8 | 0.0000 | 0.1250 | 0.6250 | 0.0000 | 0.1250 | 0.0000 | 0.1250 |
| E | 8 | 0.3750 | 0.7500 | 0.8750 | 0.3750 | 0.3750 | 0.3750 | 0.3750 |
| F | 8 | 0.5000 | 0.7500 | 0.8750 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

This is a snapshot of the current run. Re-export after new results arrive to update the report.
Per-query ranks and candidate pool sizes are in query_results.csv. JSON includes available and paired cohorts for each hint level.
