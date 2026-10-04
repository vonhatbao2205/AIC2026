# CAD-VR benchmark — Full query

Final TEST evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Moment-level Recall; TRAKE requires a complete ordered sequence; QA requires the accepted answer.
Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.
Unknown CLI cost/token usage stays null. ECE/Brier use final candidate relevance, not individual constraint labels.
Queries complete across all configured variants: 24. See summary_paired.json for matched-query comparisons. Partial rows below may have different N.

| Variant | N | R@1 | R@5 | MRR | Rescue@1 | p50 s | p95 s | Agents/query | $/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 24 | 0.1250 | 0.2917 | 0.1857 | 0.0000 | 2.0534 | 3.0803 | 0.0000 | 0.0000 |
| C | 24 | 0.3333 | 0.4583 | 0.3882 | 0.2381 | 52.0643 | 125.7897 | 2.0000 | unknown |
| F | 24 | 0.4167 | 0.5417 | 0.4590 | 0.3333 | 81.7801 | 140.8267 | 2.0000 | unknown |

Failure and agent invocation rates (0–1); failed CLI runs remain in the denominator.

| Variant | Controller failed | Agent failed | Retrieval failed | Fallback | No CLI | Codex only | Claude only | Both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| C | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| F | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
