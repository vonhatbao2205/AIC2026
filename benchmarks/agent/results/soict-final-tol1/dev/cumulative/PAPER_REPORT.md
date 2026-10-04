# CAD-VR paper tables

Final DEV evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.

Only queries complete across all configured variants are included. Failed attempts remain in the denominator.
95% percentile bootstrap intervals resample base queries; differences against A use matched queries. Intervals are exploratory, with no multiple-comparison correction.

| Variant | N | R@1 [95% CI] | R@5 [95% CI] | MRR [95% CI] | ΔR@1 vs A [95% CI] |
|---|---:|---|---|---|---|
| A | 3 | 0.0000 [0.0000, 0.0000] | 0.3333 [0.0000, 1.0000] | 0.1296 [0.0000, 0.3333] | — |
| C | 3 | 0.6667 [0.0000, 1.0000] | 0.6667 [0.0000, 1.0000] | 0.6680 [0.0040, 1.0000] | 0.6667 [0.0000, 1.0000] |
| F | 3 | 0.6667 [0.0000, 1.0000] | 0.6667 [0.0000, 1.0000] | 0.6682 [0.0045, 1.0000] | 0.6667 [0.0000, 1.0000] |

See paper_tables.csv for matched full/task tables, paper_intervals.json for per-task and per-level intervals, and CUMULATIVE_HINT_REPORT.md for hints-to-solve definitions and censoring.

Figures: hint_recall.pdf, hint_recall.png, hint_computation.pdf, hint_computation.png
