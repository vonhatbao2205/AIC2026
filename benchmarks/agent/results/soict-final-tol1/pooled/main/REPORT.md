# CAD-VR benchmark — Full query

Final POOLED evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Moment-level Recall; TRAKE requires a complete ordered sequence; QA requires the accepted answer.
Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.
Unknown CLI cost/token usage stays null. ECE/Brier use final candidate relevance, not individual constraint labels.
Queries complete across all configured variants: 111. See summary_paired.json for matched-query comparisons. Partial rows below may have different N.

| Variant | N | R@1 | R@5 | MRR | Rescue@1 | p50 s | p95 s | Agents/query | $/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 111 | 0.5856 | 0.6757 | 0.6331 | 0.5263 | 73.1972 | 166.3550 | 1.9640 | unknown |
| B | 111 | 0.3694 | 0.6757 | 0.5131 | 0.2632 | 50.4995 | 135.9125 | 1.0000 | unknown |
| D | 111 | 0.1441 | 0.3153 | 0.2255 | 0.0000 | 2.5767 | 7.1347 | 0.0000 | 0.0001 |
| F | 111 | 0.5135 | 0.6486 | 0.5729 | 0.4526 | 83.1261 | 173.5044 | 1.9910 | unknown |
| C | 111 | 0.4955 | 0.7027 | 0.5934 | 0.4105 | 49.4391 | 98.4465 | 2.0000 | unknown |
| A | 111 | 0.1441 | 0.3153 | 0.2255 | 0.0000 | 1.5521 | 5.0859 | 0.0000 | 0.0000 |

Failure and agent invocation rates (0–1); failed CLI runs remain in the denominator.

| Variant | Controller failed | Agent failed | Retrieval failed | Fallback | No CLI | Codex only | Claude only | Both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0360 | 0.9640 |
| B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| F | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0090 | 0.9910 |
| C | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
