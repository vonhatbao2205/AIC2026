# Cumulative-Hint Evaluation

Final TEST evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Each cumulative hint level is evaluated as an independent query from a fresh system state; no cross-hint memory or progressive retrieval mechanism is used.

Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.

Partial queries: video recall is primary; moment recall uses the full-query labels and is secondary. Strict success additionally requires the accepted QA answer. Full uses the existing strict task metric.

The table uses a cohort complete across every level and configured variant. Failed attempts remain in the denominator. Per-level available/paired cohorts, task breakdowns and total-hint-count strata are in summary_by_hint.json.
Complete base queries: 24/24.

| Level | Variant | N | Video R@1 | Video R@5 | Moment R@1 | Strict R@1 | Strict R@5 | Agents/query | Tools/query | Jev/query | p50 s | p95 s | $/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | A | 24 | 0.4167 | 0.6250 | 0.1667 | 0.0833 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 2.0588 | 2.9995 | 0 |
| H1 | C | 24 | 0.6667 | 0.9167 | 0.4583 | 0.2500 | 0.4167 | 2.0000 | 24.2083 | 0.0000 | 52.9908 | 107.0590 | unknown |
| H1 | F | 24 | 0.8333 | 0.8750 | 0.5417 | 0.2500 | 0.2917 | 1.7917 | 20.9167 | 9.2083 | 78.7065 | 179.2628 | unknown |
| H2 | A | 24 | 0.6250 | 0.7917 | 0.2500 | 0.1250 | 0.3750 | 0.0000 | 0.0000 | 0.0000 | 2.0532 | 3.0656 | 0 |
| H2 | C | 24 | 0.7500 | 0.9583 | 0.5833 | 0.2917 | 0.5417 | 2.0000 | 20.6250 | 0.0000 | 48.8595 | 101.8793 | unknown |
| H2 | F | 24 | 0.8333 | 0.9583 | 0.5417 | 0.3750 | 0.5000 | 1.9583 | 22.5833 | 10.6667 | 83.4493 | 175.8694 | unknown |
| H3 | A | 18 | 0.6667 | 0.7778 | 0.3889 | 0.2778 | 0.2778 | 0.0000 | 0.0000 | 0.0000 | 1.8375 | 3.1391 | 0 |
| H3 | C | 18 | 0.7222 | 0.9444 | 0.5000 | 0.3333 | 0.5000 | 2.0000 | 20.6111 | 0.0000 | 46.4969 | 81.9262 | unknown |
| H3 | F | 18 | 0.9444 | 1.0000 | 0.6667 | 0.4444 | 0.5000 | 2.0000 | 20.3333 | 12.1667 | 75.0275 | 120.4586 | unknown |
| FULL | A | 24 | 0.5833 | 0.7917 | 0.2500 | 0.1250 | 0.2917 | 0.0000 | 0.0000 | 0.0000 | 2.0534 | 3.0803 | 0 |
| FULL | C | 24 | 0.7917 | 1.0000 | 0.5417 | 0.3333 | 0.4583 | 2.0000 | 21.9583 | 0.0000 | 52.0643 | 125.7897 | unknown |
| FULL | F | 24 | 0.8750 | 1.0000 | 0.6250 | 0.4167 | 0.5417 | 2.0000 | 21.7917 | 12.3333 | 81.7801 | 140.8267 | unknown |

H3/other numeric columns can contain fewer queries because the last level is named Full. Use the total-hint-count strata for numeric curves with a fixed cohort.

Minimum hints are computed from independent outcomes and do not assume monotonic success. Unsolved queries are censored (null h*); conditional means show solved N. The penalized mean assigns H+1 to unsolved queries. Early solve and mean hint saving use all complete queries; unsolved queries receive zero saving. A regression means a later level failed after an earlier success.

| Criterion | Variant | N | Solved N | Mean h* (solved) | Median h* (solved) | Mean h* (penalized) | Early solve | Hint saving | Full success | Regression |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| video | A | 24 | 19 | 1.5789 | 1 | 2.2500 | 0.7500 | 0.6111 | 0.5833 | 0.2083 |
| moment | A | 24 | 11 | 2 | 2 | 3.4167 | 0.4167 | 0.2917 | 0.2500 | 0.2083 |
| strict | A | 24 | 6 | 2.1667 | 2.5000 | 4.0417 | 0.2500 | 0.1528 | 0.1250 | 0.1250 |
| video | C | 24 | 22 | 1.3636 | 1.0000 | 1.6667 | 0.8750 | 0.7917 | 0.7917 | 0.1250 |
| moment | C | 24 | 18 | 1.5000 | 1.0000 | 2.2917 | 0.7083 | 0.6042 | 0.5417 | 0.2917 |
| strict | C | 24 | 10 | 1.5000 | 1.0000 | 3.4167 | 0.3750 | 0.3194 | 0.3333 | 0.1667 |
| video | F | 24 | 23 | 1.2174 | 1 | 1.3750 | 0.9583 | 0.8889 | 0.8750 | 0.1250 |
| moment | F | 24 | 18 | 1.3889 | 1.0000 | 2.2083 | 0.7500 | 0.6528 | 0.6250 | 0.2083 |
| strict | F | 24 | 12 | 1.7500 | 1.5000 | 3.2500 | 0.4583 | 0.3611 | 0.4167 | 0.0833 |

Cost/tokens cover the agent and decision layers; missing CLI usage stays unknown. No empirical improvement is claimed until live runs complete.
