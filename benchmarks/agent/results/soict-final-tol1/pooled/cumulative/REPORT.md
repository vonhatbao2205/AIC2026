# CAD-VR benchmark — Full query

Final POOLED evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Moment-level Recall; TRAKE requires a complete ordered sequence; QA requires the accepted answer.
Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.
Unknown CLI cost/token usage stays null. ECE/Brier use final candidate relevance, not individual constraint labels.
Queries complete across all configured variants: 27. See summary_paired.json for matched-query comparisons. Partial rows below may have different N.

| Variant | N | R@1 | R@5 | MRR | Rescue@1 | p50 s | p95 s | Agents/query | $/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F | 27 | 0.4444 | 0.5556 | 0.4823 | 0.3750 | 79.1299 | 138.6661 | 2.0000 | unknown |
| C | 27 | 0.3704 | 0.4815 | 0.4193 | 0.2917 | 51.2162 | 120.3967 | 2.0000 | unknown |
| A | 27 | 0.1111 | 0.2963 | 0.1794 | 0.0000 | 2.0550 | 3.0789 | 0.0000 | 0.0000 |

Failure and agent invocation rates (0–1); failed CLI runs remain in the denominator.

| Variant | Controller failed | Agent failed | Retrieval failed | Fallback | No CLI | Codex only | Claude only | Both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
