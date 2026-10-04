# CAD-VR benchmark — Full query

Final TEST evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Moment-level Recall; TRAKE requires a complete ordered sequence; QA requires the accepted answer.
Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.
Unknown CLI cost/token usage stays null. ECE/Brier use final candidate relevance, not individual constraint labels.
Queries complete across all configured variants: 86. See summary_paired.json for matched-query comparisons. Partial rows below may have different N.

| Variant | N | R@1 | R@5 | MRR | Rescue@1 | p50 s | p95 s | Agents/query | $/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 86 | 0.5814 | 0.6744 | 0.6300 | 0.5205 | 72.5583 | 171.1982 | 1.9535 | unknown |
| B | 86 | 0.3953 | 0.6860 | 0.5290 | 0.2877 | 50.4568 | 147.4901 | 1.0000 | unknown |
| D | 86 | 0.1512 | 0.3372 | 0.2332 | 0.0000 | 2.0933 | 7.1312 | 0.0000 | 0.0001 |
| F | 86 | 0.5233 | 0.6395 | 0.5716 | 0.4521 | 84.2375 | 173.6867 | 1.9884 | unknown |
| C | 86 | 0.5116 | 0.7209 | 0.6119 | 0.4247 | 51.2281 | 94.1107 | 2.0000 | unknown |
| A | 86 | 0.1512 | 0.3372 | 0.2332 | 0.0000 | 1.5574 | 4.8130 | 0.0000 | 0.0000 |

Failure and agent invocation rates (0–1); failed CLI runs remain in the denominator.

| Variant | Controller failed | Agent failed | Retrieval failed | Fallback | No CLI | Codex only | Claude only | Both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0465 | 0.9535 |
| B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| F | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0116 | 0.9884 |
| C | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
