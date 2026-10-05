# Claim and evidence map

Editorial support, outside the page budget. Revised 5 October 2026 against local
implementation files and the frozen benchmark; see `REVISION_NOTES.md`. "Snapshot" = `benchmarks/agent/results/soict-final-tol1/`;
"stats" = `data/paper_stats.json` from `tools/derive_stats.py` (raw TEST runs, no model calls).

## Initial round (TEST, N = 86, tolerance 1 s)

| Claim in the paper | Value | Source |
|---|---|---|
| Strict R@1 A/B/C/D/E/F | 15.1 / 39.5 / 51.2 / 15.1 / 58.1 / 52.3 | snapshot `test/main/summary.json`, `PAPER_REPORT.md` |
| R@1 95% CIs (Table 2) | e.g. E [47.7, 68.6] | snapshot `test/main/paper_intervals.json` (2,000 resamples) |
| E − C, E − B, E − F, F − C | +7.0 [−2.3, 16.3]; +18.6 [9.3, 27.9]; +5.8 [−3.5, 15.1]; R@5 F − C −8.1 [−15.1, −2.3] | stats `paired` (10,000 resamples, seed 2026) |
| McNemar E vs C / E vs B | 11 vs 5, p = 0.21 / 18 vs 2, p = 0.0004 | stats `mcnemar_r1` |
| D = A; 525/525 D assessments unknown | identical R@1/R@5/MRR; no attribution to post-agent verification | snapshot summary; raw D traces (checked in session) |
| DEV and pooled R@1 (E, C) | 60.0, 44.0; 58.6, 49.5 | snapshot `dev/main`, `pooled/main` summaries |
| By task (T-KIS/QA/TRAKE) | E 66.1/42.1/37.5; C 62.7/26.3/25.0; F 59.3/31.6/50.0 | snapshot `test/main/paper_tables.csv` |
| Rescue@1, p50/p95, agents, tokens | E 52.1%, 72.6/171.2 s, 1.95, 316k; C 42.5%, 51.2/94.1 s, 2.00, 323k | snapshot summary |
| Investigator first in 83/86 E runs | 83 CALL_CLAUDE, 3 CALL_CODEX | stats `controller.E.first_action` |
| Agent medians in E | Claude 30.7 s, Codex 41.3 s | stats `agents.E` |
| Jev overhead | 4.9 calls, 0.59 s median per call, 3.2 s/query, 3.8% median share, US$0.00082/query | stats `controller.E` |
| Router STOP after agent 1 | 57/82 above 0.5 (median 0.73); 31/57 vs 7/25 already correct; 57 overridden; 9 evidence-sufficient stops | stats `controller.E`, `post_agent_decisions.E` |
| E† replay | R@1 52.3%, 1.29 agents, 35.9 s p50 | stats `router_replay.E` — post-hoc trace analysis, not a deployed policy |
| Verifier reliability | E ECE 0.106, Brier 0.112; ≥0.8 bin 7/13 correct | snapshot summary `jev_calibration`; stats `reliability.E` |
| Group-by-video (A, E) | strict 15.1/58.1; group moment 43.0/81.4; video 45.3/86.0; QA E group 89.5 vs strict 42.1 | snapshot `test/main/video-report/` |
| Cumulative (24 queries) | H1 video R@1 A/C/F 41.7/66.7/83.3; full strict 12.5/33.3/41.7; F − C H1 video +16.7 [4.2, 33.3]; F − A full strict +29.2 [12.5, 50.0] | snapshot `test/cumulative/`; stats `cumulative_paired` |
| Hints-to-solve (Table 4) | as printed | snapshot `test/cumulative/CUMULATIVE_HINT_REPORT.md` |
| Channels routed in TEST | OCR 18, speech 11, audio 9, TARA 5 (PE in all 86) | stats `agents.retrieval_channels_queries` |

## Supplementary live C/C+V/E round (Table 3)

Source: `benchmarks/agent/results/soict-cv-tol1/`, derived offline with
`tools/benchmark_cv_report.py`; paper copy: `data/ablation_stats.json`.
All 258 saved metrics were recomputed exactly. All 86 C+V traces show both agents,
one final verifier pass, no routing, and pre-verification order matching both the
trace and reconstructed RRF, with candidate moments/answers unchanged.
Dataset, manifest, plan and raw results have SHA-256 provenance in the JSON.

| Claim | Value | JSON field |
|---|---|---|
| C/C+V/E strict R@1 | 47.7 / 50.0 / 58.1% | `summary` |
| R@5; MRR | 69.8/.579, 66.3/.581, 66.3/.620 | `summary` |
| C+V−C | +2.3 pp [−3.5, 9.3], 5/3 discordant, p=.7265625 | `contrasts` |
| E−C+V | +8.1 pp [0.0, 16.3], 10/3 discordant, p=.09228515625 | `contrasts` |
| E−C | +10.5 pp [2.3, 18.6], 12/3 discordant, p=.03515625 | `contrasts` |
| C+V final−pre | 43/86−40/86; +3.5 pp [−1.2, 9.3]; 4 repairs/1 harm; p=.375 | `contrasts`, `cv_pre_verification` |
| p50 C/C+V/E | 53.7 / 55.6 / 75.8 s | `summary` |
| p95 C/C+V/E | 103.2 / 98.4 / 198.9 s | `summary` |
| Agents C/C+V/E; Jev calls | 2.00/2.00/1.95; 0/1/4.91 | `summary` |
| Tokens C/C+V/E | 346k/344k/327k, complete token coverage; full cost unavailable | `summary` |
| T-KIS C/C+V/E | 57.6 / 61.0 / 74.6% | `by_task` |
| QA C/C+V/E | 26.3 / 26.3 / 31.6% | `by_task` |
| TRAKE C/C+V/E | 25.0 / 25.0 / 0.0%; initial E was 37.5% | `by_task`, initial snapshot |
| Completed errors/fallback | 0 across all variants | `errors` |
| Quota overhead outside completed results | 13 attempts, 314.6 s, 25 agent invocations; pauses excluded | `quota_audit` |

All new intervals use 10,000 paired query resamples, seed 2026. Paired R@5/MRR
intervals are in the report and JSON. C/C+V use independent agent trajectories;
the within-C+V contrast is the same-evidence test. The existing TEST corpus was
already analysed, so this is a supplementary experiment, not new held-out data.

## Implementation claims

| Claim | Code |
|---|---|
| Weighted RRF, k = 60 | `backend/app/fusion.py::reciprocal_rank_fusion` |
| Group score 0.75/0.15/0.10, 30 s support saturating at 4, ambiguity gap 30 s | `backend/app/fusion.py::group_by_video`, `_is_ambiguous` |
| TARA video-level RRF replaces the group score | `backend/app/tara_fusion.py`, `services/search_service.py` |
| TRAKE event DP and coverage-first ranking | `backend/app/trake.py::trake_video_score` and assembly |
| Query-local board, de-duplication by source, visual evidence only after a frame was loaded | `backend/app/agent/evidence/board.py` |
| Constraint compiler and weights (query 2, events 1.5, answer 2) | `backend/app/agent/evidence/compiler.py` |
| Typed verification, no-evidence ⇒ p_sup = 0, geometric aggregation, unknown = abstain; ranking plurality differs from stop majority | `agent/decision/verifier.py`, `calibration.py::geometric_score`, `board.py::ranked` |
| Action set, first-step modality/complexity, escalation order | `agent/decision/router.py` |
| Stop guard (τ, δ, majority, QA answer, TRAKE order, 8 s same-moment) | `agent/decision/stopping.py` |
| Budgets 6 steps / 40 tools / 240 s / Jev 12 s; each CLI once | `agent/controller.py`, `config.py` |
| Agents: Codex GPT-6.1 Sol and Claude Opus 5.5, high effort, fast; no shell; soft roles | `agent/cli.py`, `agent/prompt.py`, benchmark `manifest.json` |
| GLAP audio vectors in Milvus, text/tags in Elasticsearch | `README.md`; audio adapters in `backend/app/` |
| Ten MCP tools | `agent/tools.py` (`TOOL_SPECS`) |

## Corpus facts

| Claim | Source |
|---|---|
| 1,478 videos, 382,299 organiser keyframes | `elastic_staging/mapping_summary.json`; `Batch2/KEYFRAME_METADATA_LAYOUT.md` (873 L + 605 K) |
| 2,212,383 InfoShot++ keyframes (1,339,055 + 873,328) | `milvus_upload_pe_core_batch2_verification.json` |
| 298 traffic-camera recordings, 12 race recordings | `backend/app/traffic_cameras.json`; `Batch2/PE_Core_G14_448_Batch2_AIC2026_Retrieval_Handoff.md` |
| 168,536 TARA clips for 873 videos | `VideoRetrieval/TARA_INTEGRATION.md` |
| HunyuanOCR, PhoWhisper-large + WhisperX | `AIC2026_Batch2_OCR_HANDOFF.md`, `AIC2026_SPEECH_BATCH2_HANDOFF.md` |

## Citations added for this paper

Verified on 4–5 October 2026: the four SOICT 2025 entries now use official
Springer chapter pages and publisher BibTeX exports (checked 5 October), replacing
the provisional programme-book citations. Full author lists and name ordering,
CCIS volumes, page ranges, publisher and DOI are recorded in `references.bib`:

| Citation key | Official chapter | CCIS | Pages |
|---|---|---|---|
| `nguyen2025conversational` | [Conversational retrieval](https://link.springer.com/chapter/10.1007/978-981-92-2587-3_38) | 2913 | 483–497 |
| `hole2025lifelog` | [LLM lifelog agents](https://link.springer.com/chapter/10.1007/978-981-92-2600-9_40) | 2916 | 510–524 |
| `nguyen2025adaptivedp` | [Agent-guided dynamic programming](https://link.springer.com/chapter/10.1007/978-981-92-2587-3_46) | 2913 | 588–598 |
| `ho2025lucifer` | [Lucifer-TRACE](https://link.springer.com/chapter/10.1007/978-981-92-2590-3_5) | 2914 | 53–66 |

All four publisher exports and “Cite this paper” sections specify **2027** as the
citation year, although the conference is SOICT 2025 and the chapters first went
online in August–September 2026. The bibliography follows that citation year,
retains SOICT 2025 in the proceedings title, and preserves existing citation keys.

Other sources verified on 4–5 October 2026: [Deep Video Discovery](https://arxiv.org/abs/2505.18079),
[VideoMind](https://github.com/yeliudev/VideoMind), [LensWalk](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_LensWalk_Agentic_Video_Understanding_by_Planning_How_You_See_in_CVPR_2026_paper.pdf),
[Wu et al. 2026](https://arxiv.org/abs/2602.19040), [Jäckl et al. 2026](https://arxiv.org/abs/2609.07311),
[Lokoč et al. 2023](https://link.springer.com/article/10.1007/s00530-023-01143-5), and the
[Jev / OpenRouter listing](https://openrouter.ai/provider/typesafe). VBS 2026, PE, Qwen,
TARA, InfoShot, Milvus, RRF, DRES and bootstrap entries are reused from
`Paper/vbs2027/references.bib`, where they were verified. Well-known references
(ReAct, SwiftSage, FrugalGPT, RouteLLM, MT-Bench judge, Guo et al. calibration,
McNemar, Brier, Kahneman) were not re-fetched.

## What the paper does not claim

- Initial E−C is inconclusive; supplementary E−C is positive with unadjusted p=.035, but E−C+V remains inconclusive (p=.092). C has higher R@5 in both rounds.
- No compute or latency saving in the deployed configuration; E† is a replay.
- No benefit of constraint-level verification overall; the TRAKE advantage rests on 8 queries.
- No reuse of Progressive Hint Memory (VBS paper): cumulative hints are independent fresh runs.
- No user study or operator-effort measurement.

## Added fixed-trajectory diagnostic

`stats.final_ranking_diagnostic` reorders saved final candidates by the exact
unverified board key from `EvidenceBoard.ranked`: negative fusion score,
first-seen time, keyframe ID. The derivation asserts exact A/B/C/D order parity
on 344 controls and verifies logged strict R@1 for every run before reporting.

| Trajectory | Final RRF correct | Logged verification correct | Paired difference, pp |
|---|---:|---:|---|
| E | 51/86 (59.3%) | 50/86 (58.1%) | −1.2 [−5.8, 2.3] |
| F | 48/86 (55.8%) | 45/86 (52.3%) | −3.5 [−9.3, 1.2] |

This removes final verification **ordering only**, retaining candidate acquisition,
answers and prior decisions. It is neither live C+V nor a live no-verifier policy.
`stats.source.sha256` identifies the raw results and evaluation labels.

## Figure and format sources

Fig. 1 is a new native TikZ diagram. Official OpenAI, Claude and TypeSafe marks
are documented with exact download URLs in `figures/logos/SOURCES.md`; Jev is
identified by its provider's logo. No logo was generated or redrawn.

The [SOICT submission instructions](https://soict.org/submission/paper-submission/)
were checked on 5 October 2026: CCIS/LNCS, 12 content pages excluding references,
single-blind, no printed page numbers.

## Additional boundaries

- Live C+V is complete; E−C+V compares the controller package and does not isolate adaptive allocation. Nonsignificance is not evidence of equivalence.
- No independent benefit of group scoring or its heuristic weights is established.
- No equivalence of replay E† with C; no live replay latency measurement.
- ECE uses 10 bins and final candidate strict labels (excluding TRAKE); the plot
  uses five display bins. This is not calibration against human constraint labels.
- F changes verification granularity **and** the available evidence actions.
- First-success hint metrics do not imply monotonic success or online stopping.
