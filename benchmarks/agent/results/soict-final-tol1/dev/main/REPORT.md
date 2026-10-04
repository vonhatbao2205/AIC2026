# CAD-VR benchmark — Full query

Final DEV evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Moment-level Recall; TRAKE requires a complete ordered sequence; QA requires the accepted answer.
Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.
Unknown CLI cost/token usage stays null. ECE/Brier use final candidate relevance, not individual constraint labels.
Queries complete across all configured variants: 25. See summary_paired.json for matched-query comparisons. Partial rows below may have different N.

| Variant | N | R@1 | R@5 | MRR | Rescue@1 | p50 s | p95 s | Agents/query | $/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 25 | 0.6000 | 0.6800 | 0.6439 | 0.5455 | 77.2403 | 149.0521 | 2.0000 | unknown |
| B | 25 | 0.2800 | 0.6400 | 0.4585 | 0.1818 | 50.9296 | 130.3760 | 1.0000 | unknown |
| D | 25 | 0.1200 | 0.2400 | 0.1992 | 0.0000 | 2.5930 | 6.8538 | 0.0000 | 0.0001 |
| F | 25 | 0.4800 | 0.6800 | 0.5772 | 0.4545 | 80.5924 | 156.7702 | 2.0000 | unknown |
| C | 25 | 0.4400 | 0.6400 | 0.5299 | 0.3636 | 47.9014 | 102.2067 | 2.0000 | unknown |
| A | 25 | 0.1200 | 0.2400 | 0.1992 | 0.0000 | 1.5507 | 6.5022 | 0.0000 | 0.0000 |

Failure and agent invocation rates (0–1); failed CLI runs remain in the denominator.

| Variant | Controller failed | Agent failed | Retrieval failed | Fallback | No CLI | Codex only | Claude only | Both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| F | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
