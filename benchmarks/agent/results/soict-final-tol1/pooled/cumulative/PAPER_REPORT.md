# CAD-VR paper tables

Final POOLED evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.

Only queries complete across all configured variants are included. Failed attempts remain in the denominator.
95% percentile bootstrap intervals resample base queries; differences against A use matched queries. Intervals are exploratory, with no multiple-comparison correction.

| Variant | N | R@1 [95% CI] | R@5 [95% CI] | MRR [95% CI] | ΔR@1 vs A [95% CI] |
|---|---:|---|---|---|---|
| A | 27 | 0.1111 [0.0000, 0.2231] | 0.2963 [0.1111, 0.4815] | 0.1794 [0.0710, 0.3077] | — |
| C | 27 | 0.3704 [0.1852, 0.5556] | 0.4815 [0.2963, 0.6667] | 0.4193 [0.2524, 0.5954] | 0.2593 [0.1111, 0.4074] |
| F | 27 | 0.4444 [0.2593, 0.6296] | 0.5556 [0.3704, 0.7407] | 0.4823 [0.2969, 0.6599] | 0.3333 [0.1481, 0.5185] |

See paper_tables.csv for matched full/task tables, paper_intervals.json for per-task and per-level intervals, and CUMULATIVE_HINT_REPORT.md for hints-to-solve definitions and censoring.

Figures: hint_recall.pdf, hint_recall.png, hint_computation.pdf, hint_computation.png
