# CAD-VR paper tables

Final TEST evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.

Only queries complete across all configured variants are included. Failed attempts remain in the denominator.
95% percentile bootstrap intervals resample base queries; differences against A use matched queries. Intervals are exploratory, with no multiple-comparison correction.

| Variant | N | R@1 [95% CI] | R@5 [95% CI] | MRR [95% CI] | ΔR@1 vs A [95% CI] |
|---|---:|---|---|---|---|
| A | 86 | 0.1512 [0.0814, 0.2326] | 0.3372 [0.2442, 0.4419] | 0.2332 [0.1626, 0.3111] | — |
| B | 86 | 0.3953 [0.2907, 0.5000] | 0.6860 [0.5930, 0.7791] | 0.5290 [0.4428, 0.6189] | 0.2442 [0.1512, 0.3372] |
| C | 86 | 0.5116 [0.4070, 0.6163] | 0.7209 [0.6163, 0.8140] | 0.6119 [0.5188, 0.7002] | 0.3605 [0.2674, 0.4538] |
| D | 86 | 0.1512 [0.0814, 0.2326] | 0.3372 [0.2442, 0.4419] | 0.2332 [0.1606, 0.3043] | 0.0000 [0.0000, 0.0000] |
| E | 86 | 0.5814 [0.4767, 0.6860] | 0.6744 [0.5698, 0.7674] | 0.6300 [0.5344, 0.7249] | 0.4302 [0.3256, 0.5349] |
| F | 86 | 0.5233 [0.4186, 0.6279] | 0.6395 [0.5349, 0.7326] | 0.5716 [0.4750, 0.6642] | 0.3721 [0.2558, 0.4767] |

See paper_tables.csv and paper_intervals.json for matched task tables and intervals.
