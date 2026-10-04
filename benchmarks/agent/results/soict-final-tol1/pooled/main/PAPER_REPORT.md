# CAD-VR paper tables

Final POOLED evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.

Only queries complete across all configured variants are included. Failed attempts remain in the denominator.
95% percentile bootstrap intervals resample base queries; differences against A use matched queries. Intervals are exploratory, with no multiple-comparison correction.

| Variant | N | R@1 [95% CI] | R@5 [95% CI] | MRR [95% CI] | ΔR@1 vs A [95% CI] |
|---|---:|---|---|---|---|
| A | 111 | 0.1441 [0.0811, 0.2162] | 0.3153 [0.2342, 0.3964] | 0.2255 [0.1649, 0.2898] | — |
| B | 111 | 0.3694 [0.2793, 0.4595] | 0.6757 [0.5856, 0.7568] | 0.5131 [0.4337, 0.5905] | 0.2252 [0.1441, 0.3063] |
| C | 111 | 0.4955 [0.4052, 0.5946] | 0.7027 [0.6126, 0.7838] | 0.5934 [0.5100, 0.6698] | 0.3514 [0.2613, 0.4324] |
| D | 111 | 0.1441 [0.0811, 0.2072] | 0.3153 [0.2342, 0.3964] | 0.2255 [0.1646, 0.2909] | 0.0000 [0.0000, 0.0000] |
| E | 111 | 0.5856 [0.4865, 0.6757] | 0.6757 [0.5856, 0.7568] | 0.6331 [0.5466, 0.7151] | 0.4414 [0.3423, 0.5405] |
| F | 111 | 0.5135 [0.4232, 0.6036] | 0.6486 [0.5586, 0.7387] | 0.5729 [0.4853, 0.6553] | 0.3694 [0.2703, 0.4685] |

See paper_tables.csv and paper_intervals.json for matched task tables and intervals.
