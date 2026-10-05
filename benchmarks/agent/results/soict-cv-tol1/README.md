# Live C/C+V/E ablation — complete TEST cohort

86 queries; 258 completed live runs; tolerance 1 s. Original A–F results remain a separate round.

| Variant | R@1 (%) | R@5 (%) | MRR | p50 s | p95 s | Agents/query | Jev calls | kTokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 47.7 | 69.8 | 0.579 | 53.7 | 103.2 | 2.00 | 0.00 | 346 |
| C+V | 50.0 | 66.3 | 0.581 | 55.6 | 98.4 | 2.00 | 1.00 | 344 |
| E | 58.1 | 66.3 | 0.620 | 75.8 | 198.9 | 1.95 | 4.91 | 327 |

| Paired contrast | ΔR@1, pp [95% CI] | A-only / B-only | Exact McNemar p |
|---|---:|---:|---:|
| C+V - C | +2.3 [-3.5, 9.3] | 5 / 3 | 0.7266 |
| E - C+V | +8.1 [0.0, 16.3] | 10 / 3 | 0.0923 |
| E - C | +10.5 [2.3, 18.6] | 12 / 3 | 0.0352 |
| C+V - C+V pre | +3.5 [-1.2, 9.3] | 4 / 1 | 0.3750 |

| Paired contrast | ΔR@5, pp [95% CI] | ΔMRR [95% CI] |
|---|---:|---:|
| C+V - C | -3.5 [-10.5, 3.5] | +0.001 [-0.045, 0.050] |
| E - C+V | +0.0 [-5.8, 7.0] | +0.040 [-0.021, 0.100] |
| E - C | -3.5 [-9.3, 2.3] | +0.041 [-0.019, 0.102] |
| C+V - C+V pre | +0.0 [0.0, 0.0] | +0.019 [-0.004, 0.047] |

| Task | N | C R@1 (%) | C+V R@1 (%) | E R@1 (%) |
|---|---:|---:|---:|---:|
| QA | 19 | 26.3 | 26.3 | 31.6 |
| T-KIS | 59 | 57.6 | 61.0 | 74.6 |
| TRAKE | 8 | 25.0 | 25.0 | 0.0 |

All contrasts use 10,000 paired base-query resamples, seed 2026. Exploratory, unadjusted for multiplicity.
E exceeds C in this round, but E versus C+V is inconclusive; nonsignificance does not establish equivalence.
C+V versus C contains independently sampled agent trajectories. Only C+V final versus its pre-verification ranking holds acquired evidence fixed.
The controller-package contrast also changes scheduling, intermediate verification and agent context; it does not isolate call allocation.

C+V pre-verification R@1: 46.5%. All 86 traces have one final verification pass and no routing decisions.
Quota audit: 13 interrupted attempts, 314.6 s recorded wall time and 25 agent invocations outside completed-run metrics; pause time excluded.
Completed-run errors: {"C": {"failed": 0, "agent_failed": 0, "retrieval_failed": 0, "retrieval_degraded": 0, "fallback": 0}, "C+V": {"failed": 0, "agent_failed": 0, "retrieval_failed": 0, "retrieval_degraded": 0, "fallback": 0}, "E": {"failed": 0, "agent_failed": 0, "retrieval_failed": 0, "retrieval_degraded": 0, "fallback": 0}}

Token telemetry is complete; full monetary cost is unavailable because some CLI usage has no cost. Known cost is not total cost.
Per-task results, all R@1/R@5/MRR paired intervals, validation counts and SHA-256 provenance are in ablation_stats.json. Raw traces remain local.
