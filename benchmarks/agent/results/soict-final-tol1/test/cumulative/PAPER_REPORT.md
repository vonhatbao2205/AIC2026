# CAD-VR paper tables

Final TEST evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.

Only queries complete across all configured variants are included. Failed attempts remain in the denominator.
95% percentile bootstrap intervals resample base queries; differences against A use matched queries. Intervals are exploratory, with no multiple-comparison correction.

| Variant | N | R@1 [95% CI] | R@5 [95% CI] | MRR [95% CI] | ΔR@1 vs A [95% CI] |
|---|---:|---|---|---|---|
| A | 24 | 0.1250 [0.0000, 0.2917] | 0.2917 [0.1250, 0.4594] | 0.1857 [0.0671, 0.3278] | — |
| C | 24 | 0.3333 [0.1250, 0.5417] | 0.4583 [0.2917, 0.6667] | 0.3882 [0.1979, 0.5653] | 0.2083 [0.0833, 0.3750] |
| F | 24 | 0.4167 [0.2083, 0.6250] | 0.5417 [0.3333, 0.7500] | 0.4590 [0.2771, 0.6500] | 0.2917 [0.1250, 0.4583] |

See paper_tables.csv for matched full/task tables, paper_intervals.json for per-task and per-level intervals, and CUMULATIVE_HINT_REPORT.md for hints-to-solve definitions and censoring.

Figures: hint_recall.pdf, hint_recall.png, hint_computation.pdf, hint_computation.png
