# CAD-VR paper tables

Final DEV evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.

Only queries complete across all configured variants are included. Failed attempts remain in the denominator.
95% percentile bootstrap intervals resample base queries; differences against A use matched queries. Intervals are exploratory, with no multiple-comparison correction.

| Variant | N | R@1 [95% CI] | R@5 [95% CI] | MRR [95% CI] | ΔR@1 vs A [95% CI] |
|---|---:|---|---|---|---|
| A | 25 | 0.1200 [0.0000, 0.2400] | 0.2400 [0.0800, 0.4000] | 0.1992 [0.0805, 0.3381] | — |
| B | 25 | 0.2800 [0.1200, 0.4410] | 0.6400 [0.4400, 0.8400] | 0.4585 [0.3007, 0.6173] | 0.1600 [0.0400, 0.3200] |
| C | 25 | 0.4400 [0.2400, 0.6400] | 0.6400 [0.4400, 0.8400] | 0.5299 [0.3477, 0.7004] | 0.3200 [0.1600, 0.5200] |
| D | 25 | 0.1200 [0.0000, 0.2400] | 0.2400 [0.0800, 0.4400] | 0.1992 [0.0827, 0.3383] | 0.0000 [0.0000, 0.0000] |
| E | 25 | 0.6000 [0.4000, 0.8000] | 0.6800 [0.4800, 0.8400] | 0.6439 [0.4637, 0.8233] | 0.4800 [0.2800, 0.6800] |
| F | 25 | 0.4800 [0.2800, 0.6800] | 0.6800 [0.4800, 0.8400] | 0.5772 [0.4077, 0.7406] | 0.3600 [0.1600, 0.5600] |

See paper_tables.csv and paper_intervals.json for matched task tables and intervals.
