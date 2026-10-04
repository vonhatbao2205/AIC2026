# Cumulative-Hint Evaluation

Final DEV evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Each cumulative hint level is evaluated as an independent query from a fresh system state; no cross-hint memory or progressive retrieval mechanism is used.

Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.

Partial queries: video recall is primary; moment recall uses the full-query labels and is secondary. Strict success additionally requires the accepted QA answer. Full uses the existing strict task metric.

The table uses a cohort complete across every level and configured variant. Failed attempts remain in the denominator. Per-level available/paired cohorts, task breakdowns and total-hint-count strata are in summary_by_hint.json.
Complete base queries: 3/3.

| Level | Variant | N | Video R@1 | Video R@5 | Moment R@1 | Strict R@1 | Strict R@5 | Agents/query | Tools/query | Jev/query | p50 s | p95 s | $/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | A | 3 | 0.6667 | 0.6667 | 0 | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 2.0553 | 2.0562 | 0 |
| H1 | C | 3 | 0.6667 | 1.0000 | 0 | 0.0000 | 0.6667 | 2.0000 | 20.3333 | 0.0000 | 42.5525 | 50.9657 | unknown |
| H1 | F | 3 | 0.6667 | 1.0000 | 0 | 0.0000 | 0.6667 | 2.0000 | 22.6667 | 10.3333 | 86.4559 | 88.8836 | unknown |
| H2 | A | 3 | 0.6667 | 1.0000 | 0.3333 | 0.3333 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 3.0676 | 3.9928 | 0 |
| H2 | C | 3 | 1.0000 | 1.0000 | 1 | 1.0000 | 1.0000 | 2.0000 | 18.3333 | 0.0000 | 43.0609 | 52.4437 | unknown |
| H2 | F | 3 | 1.0000 | 1.0000 | 1 | 1.0000 | 1.0000 | 2.0000 | 20.0000 | 10.6667 | 76.3912 | 85.9856 | unknown |
| H3 | A | 2 | 1.0000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 2.3234 | 2.5606 | 0 |
| H3 | C | 2 | 1.0000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 2.0000 | 17.5000 | 0.0000 | 43.5052 | 44.3500 | unknown |
| H3 | F | 2 | 1.0000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 2.0000 | 18.0000 | 10.0000 | 66.6047 | 68.4897 | unknown |
| FULL | A | 3 | 0.3333 | 1.0000 | 0 | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 2.0722 | 2.5164 | 0 |
| FULL | C | 3 | 1.0000 | 1.0000 | 0.6667 | 0.6667 | 0.6667 | 2.0000 | 18.3333 | 0.0000 | 41.0581 | 42.2216 | unknown |
| FULL | F | 3 | 1.0000 | 1.0000 | 0.6667 | 0.6667 | 0.6667 | 2.0000 | 19.0000 | 10.3333 | 72.3243 | 75.4342 | unknown |

H3/other numeric columns can contain fewer queries because the last level is named Full. Use the total-hint-count strata for numeric curves with a fixed cohort.

Minimum hints are computed from independent outcomes and do not assume monotonic success. Unsolved queries are censored (null h*); conditional means show solved N. The penalized mean assigns H+1 to unsolved queries. Early solve and mean hint saving use all complete queries; unsolved queries receive zero saving. A regression means a later level failed after an earlier success.

| Criterion | Variant | N | Solved N | Mean h* (solved) | Median h* (solved) | Mean h* (penalized) | Early solve | Hint saving | Full success | Regression |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| video | A | 3 | 3 | 1.3333 | 1 | 1.3333 | 1.0000 | 0.8889 | 0.3333 | 0.6667 |
| moment | A | 3 | 2 | 2.5000 | 2.5000 | 3 | 0.6667 | 0.3333 | 0.0000 | 0.6667 |
| strict | A | 3 | 2 | 2.5000 | 2.5000 | 3 | 0.6667 | 0.3333 | 0.0000 | 0.6667 |
| video | C | 3 | 3 | 1.3333 | 1 | 1.3333 | 1.0000 | 0.8889 | 1.0000 | 0.0000 |
| moment | C | 3 | 3 | 2 | 2 | 2 | 1.0000 | 0.6111 | 0.6667 | 0.3333 |
| strict | C | 3 | 3 | 2 | 2 | 2 | 1.0000 | 0.6111 | 0.6667 | 0.3333 |
| video | F | 3 | 3 | 1.3333 | 1 | 1.3333 | 1.0000 | 0.8889 | 1.0000 | 0.0000 |
| moment | F | 3 | 3 | 2 | 2 | 2 | 1.0000 | 0.6111 | 0.6667 | 0.3333 |
| strict | F | 3 | 3 | 2 | 2 | 2 | 1.0000 | 0.6111 | 0.6667 | 0.3333 |

Cost/tokens cover the agent and decision layers; missing CLI usage stays unknown. No empirical improvement is claimed until live runs complete.
