# Cumulative-Hint Evaluation

Final POOLED evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. Pooled combines DEV+TEST; split-specific results are reported separately. All planned query/level/variant results are present; no base query exclusions remain. This directory is an offline report export, not an agent runner resume directory.

Each cumulative hint level is evaluated as an independent query from a fresh system state; no cross-hint memory or progressive retrieval mechanism is used.

Runtime mode: live. Mock runs validate the harness only and are not paper accuracy results.

Partial queries: video recall is primary; moment recall uses the full-query labels and is secondary. Strict success additionally requires the accepted QA answer. Full uses the existing strict task metric.

The table uses a cohort complete across every level and configured variant. Failed attempts remain in the denominator. Per-level available/paired cohorts, task breakdowns and total-hint-count strata are in summary_by_hint.json.
Complete base queries: 27/27.

| Level | Variant | N | Video R@1 | Video R@5 | Moment R@1 | Strict R@1 | Strict R@5 | Agents/query | Tools/query | Jev/query | p50 s | p95 s | $/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | A | 27 | 0.4444 | 0.6296 | 0.1481 | 0.0741 | 0.1852 | 0.0000 | 0.0000 | 0.0000 | 2.0563 | 2.9241 | 0 |
| H1 | C | 27 | 0.6667 | 0.9259 | 0.4074 | 0.2222 | 0.4444 | 2.0000 | 23.7778 | 0.0000 | 51.2872 | 106.2246 | unknown |
| H1 | F | 27 | 0.8148 | 0.8889 | 0.4815 | 0.2222 | 0.3333 | 1.8148 | 21.1111 | 9.3333 | 84.4728 | 176.9662 | unknown |
| H2 | A | 27 | 0.6296 | 0.8148 | 0.2593 | 0.1481 | 0.4074 | 0.0000 | 0.0000 | 0.0000 | 2.0550 | 3.0706 | 0 |
| H2 | C | 27 | 0.7778 | 0.9630 | 0.6296 | 0.3704 | 0.5926 | 2.0000 | 20.3704 | 0.0000 | 48.3214 | 99.1233 | unknown |
| H2 | F | 27 | 0.8519 | 0.9630 | 0.5926 | 0.4444 | 0.5556 | 1.9630 | 22.2963 | 10.6667 | 80.3942 | 172.2954 | unknown |
| H3 | A | 20 | 0.7000 | 0.8000 | 0.4000 | 0.3000 | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 2.0487 | 3.0889 | 0 |
| H3 | C | 20 | 0.7500 | 0.9500 | 0.5000 | 0.3500 | 0.5000 | 2.0000 | 20.3000 | 0.0000 | 46.1759 | 78.8622 | unknown |
| H3 | F | 20 | 0.9500 | 1.0000 | 0.6500 | 0.4500 | 0.5000 | 2.0000 | 20.1000 | 11.9500 | 72.2845 | 116.3556 | unknown |
| FULL | A | 27 | 0.5556 | 0.8148 | 0.2222 | 0.1111 | 0.2963 | 0.0000 | 0.0000 | 0.0000 | 2.0550 | 3.0789 | 0 |
| FULL | C | 27 | 0.8148 | 1.0000 | 0.5556 | 0.3704 | 0.4815 | 2.0000 | 21.5556 | 0.0000 | 51.2162 | 120.3967 | unknown |
| FULL | F | 27 | 0.8889 | 1.0000 | 0.6296 | 0.4444 | 0.5556 | 2.0000 | 21.4815 | 12.1111 | 79.1299 | 138.6661 | unknown |

H3/other numeric columns can contain fewer queries because the last level is named Full. Use the total-hint-count strata for numeric curves with a fixed cohort.

Minimum hints are computed from independent outcomes and do not assume monotonic success. Unsolved queries are censored (null h*); conditional means show solved N. The penalized mean assigns H+1 to unsolved queries. Early solve and mean hint saving use all complete queries; unsolved queries receive zero saving. A regression means a later level failed after an earlier success.

| Criterion | Variant | N | Solved N | Mean h* (solved) | Median h* (solved) | Mean h* (penalized) | Early solve | Hint saving | Full success | Regression |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| video | A | 27 | 22 | 1.5455 | 1.0000 | 2.1481 | 0.7778 | 0.6420 | 0.5556 | 0.2593 |
| moment | A | 27 | 13 | 2.0769 | 2 | 3.3704 | 0.4444 | 0.2963 | 0.2222 | 0.2593 |
| strict | A | 27 | 8 | 2.2500 | 2.5000 | 3.9259 | 0.2963 | 0.1728 | 0.1111 | 0.1852 |
| video | C | 27 | 25 | 1.3600 | 1 | 1.6296 | 0.8889 | 0.8025 | 0.8148 | 0.1111 |
| moment | C | 27 | 21 | 1.5714 | 1 | 2.2593 | 0.7407 | 0.6049 | 0.5556 | 0.2963 |
| strict | C | 27 | 13 | 1.6154 | 2 | 3.2593 | 0.4444 | 0.3519 | 0.3704 | 0.1852 |
| video | F | 27 | 26 | 1.2308 | 1.0000 | 1.3704 | 0.9630 | 0.8889 | 0.8889 | 0.1111 |
| moment | F | 27 | 21 | 1.4762 | 1 | 2.1852 | 0.7778 | 0.6481 | 0.6296 | 0.2222 |
| strict | F | 27 | 15 | 1.8000 | 2 | 3.1111 | 0.5185 | 0.3889 | 0.4444 | 0.1111 |

Cost/tokens cover the agent and decision layers; missing CLI usage stays unknown. No empirical improvement is claimed until live runs complete.
