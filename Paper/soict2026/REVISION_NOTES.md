# Review revision — 5 October 2026

## Earlier review revision (initial A–F round)

- Reframed the abstract, introduction, contributions, discussion and conclusion
  around explicit evidence-guided control. Fast–slow remains an architectural
  description. Deployed computation is a negative result, not a main contribution.
- Kept the six live configurations and their frozen results unchanged. E–C remains
  inconclusive; the replay is explicitly post-hoc TEST analysis, not a live policy,
  an equivalence result, a measured operating curve or a promised saving.
- Added a reproducible **final-ranking diagnostic on fixed logged candidates**.
  Sorting each final snapshot by `(-fusion_score, first_seen_s, submit_keyframe_id)`
  implements the unverified board order. It reproduces all 344 A/B/C/D rankings.
  With E's own final candidates, RRF scores 51/86 and logged verification 50/86;
  with F's candidates, 48/86 versus 45/86. Verification changes earlier trajectories,
  so this diagnostic cannot establish the causal value of routing.
- Corrected metric definitions (top-k videos; QA answer ignored in group-moment;
  complete sequences still required for TRAKE), bootstrap counts (2,000 marginal,
  10,000 paired), ranking plurality versus stopping majority, scored competitors,
  shortlist size, budget termination, and GLAP vector storage (Milvus).
- Removed unsupported causal interpretations of QA gains, grouping, D=A and
  constraint-level verification. Specified heuristic group weights and the absence
  of a grouping/weight ablation. Distinguished current system capabilities from
  the narrower frozen benchmark configuration.
- Added limits concerning owner-reviewed labels, shared-video dependence,
  single-run variability, support-score calibration and retry-time exclusion.
- Redesigned Fig. 1 with separated processing regions, larger gutters, orthogonal
  connections and official provider marks. See `figures/logos/SOURCES.md`.
- Removed the connecting arrow between E and replay E† in the trade-off figure;
  the two observations do not establish an operating curve.

## New diagnostic, with paired uncertainty

| Logged trajectory | Final RRF | Logged verification | Difference (verification − RRF) | Repair / harm |
|---|---:|---:|---:|---:|
| E | 59.3% (51/86) | 58.1% (50/86) | −1.2 pp [−5.8, 2.3] | 1 / 2 |
| F | 55.8% (48/86) | 52.3% (45/86) | −3.5 pp [−9.3, 1.2] | 1 / 4 |

Intervals use 10,000 paired query resamples, seed 2026; exact McNemar p-values
are 1.0 and 0.375. These are exploratory diagnostics, not evidence of significant
harm. Per-query outcomes, control assertions and raw-source SHA-256 hashes are in
`data/paper_stats.json` and `tools/derive_stats.py`. No new model calls were made.

## Completed live C+V integration

The user completed the matched TEST experiment. Its full 86-query result differs
from the early partial snapshot: C/C+V/E score **47.7/50.0/58.1% R@1**.
The integration audits all 258 metrics, all 86 fixed-acquisition/final-verifier
traces, and the pre-verification order against saved RRF. The raw source code hash
matches the frozen protocol and current benchmark implementation. No new model
calls were made during this integration.

- E−C+V is +8.1 pp [0.0,16.3], exact McNemar p=.092; attribution to the controller
  package remains inconclusive. E−C is +10.5 pp [2.3,18.6], unadjusted p=.035.
- C+V−C is +2.3 pp [−3.5,9.3], p=.727. Independent agent trajectories mean this
  is a policy comparison, not a fixed-evidence comparison.
- C+V's own final verifier raises R@1 from 40/86 to 43/86, with four repairs and
  one harm (+3.5 pp [−1.2,9.3], p=.375). Moments and answers stay fixed.
- E has 0/8 TRAKE successes in this round, versus 3/8 initially; reported explicitly.
- C/C+V/E p50 is 53.7/55.6/75.8 s, with 2/2/1.95 agents. No practical savings.
- Completed attempts have zero errors/fallback. Thirteen quota interruptions add
  314.6 s of recorded wall time and 25 agent invocations; pauses are not counted.

The revised title is **CAD-VR: Evidence-Guided Control for Fast–Slow Agentic Video
Event Retrieval**. Contributions emphasise typed control, an explicit guard,
audited traces and evaluation. Grouping is an operator-facing design choice.
The initial A–F and cumulative results are kept in their own round; the follow-up
reuses already analysed TEST data and has no new DEV accuracy tuning.

The paper adds Table 3 for the ablation; hints-to-solve becomes Table 4. The
qualitative case figure was initially moved outside the main PDF for space, then
restored as Fig. 7 following the user's feedback. Repeated prose and captions
were shortened to retain the live ablation and all seven figures.
`tools/benchmark_cv_report.py`, its tests and the compact benchmark snapshot are
added to the repository. The paper bundles the exporter, full stats and outcomes.
This removes the missing-baseline limitation; it does not claim that scheduling
has been causally isolated or that a nonsignificant difference proves equivalence.

## Submission checks

The official [SOICT submission page](https://soict.org/submission/paper-submission/)
was rechecked: CCIS/LNCS, at most 12 content pages excluding references,
single-blind, and no page numbers. The manuscript suppresses page numbers.
Author names and affiliation are preserved; no email or ORCID is invented.

User follow-up: replaced both Jev icons with the official solid-pink TypeSafe
PNG matching the supplied reference; updated the standalone figure, manuscript
and source ZIP. Exact source is recorded in `figures/logos/SOURCES.md`.

Figure follow-up: added the existing Milvus/Elasticsearch logos, vector
functional pictograms and numbered stage badges. Kept the pink Jev marks,
spaced layout and original control/data flow.

Figure 7 restoration: reinstated the full-width original case-study image and
its qualitative explanation, explicitly labelled as the initial TEST round.
Kept its images, timeline, privacy blur, fonts and scaling intact; condensed text
and captions elsewhere instead of dropping an experiment or shrinking the figure.

Wording follow-up: replaced “open-ended agent search” with “tool-using agent
search” in the abstract and aligned the related-work terminology. Contribution 3
now reads “Controlled evaluation and diagnostics” and states that live C/C+V/E
and within-C+V same-evidence pre/post contrasts distinguish final-ranking effects
from the broader controller package. This describes the experiment's attribution
purpose without claiming that adaptive scheduling is isolated.

Submission preparation: added the direct GitHub benchmark-directory link and
“are available” wording, as the user will make the repository public at submission.
Repository visibility was checked as private during editing; this is prepared
submission wording. Competition media are not redistributed. Current local C+V
reports and exporter must be included in the public revision alongside the
already-versioned initial benchmark. Added a FloatBarrier before Section 8 so
Fig. 7 precedes its heading. Revised the abstract to “documented limits on ranking
gains, efficiency, and component attribution.” Condensed repeated prose while
preserving all results, uncertainty and limitations.

SOICT bibliography follow-up (5 October 2026): replaced four provisional
programme-book entries with Springer chapter metadata and publisher BibTeX
author lists. Added all four DOIs, CCIS volumes and page ranges; corrected
Le-Hinh, Van-Tu Ninh and the published first-author forms. The publisher citation
year is 2027 for these SOICT 2025 chapters, already online in 2026. Kept stable
citation keys and documented the official sources in `SOURCE_MAP.md`.
