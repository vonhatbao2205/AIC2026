# CAD-VR — SOICT 2026 manuscript

**CAD-VR: Evidence-Guided Control for Fast–Slow Agentic Video Event Retrieval**

[PDF](main.pdf) · [LaTeX](main.tex) · [Bibliography](references.bib) · [Claim and evidence map](SOURCE_MAP.md) · [Ghi chú tiếng Việt](GHI_CHU_VI.md) · [Review revisions](REVISION_NOTES.md) · [Live ablation report](ABLATION_REPORT.md) · [Fig. 1](figures/architecture.pdf)

Springer CCIS / LNCS format (`llncs`, `splncs04`), as required by the
[SOICT 2026 call](https://soict.org/submission/paper-submission/): 12 content pages
plus 3 reference pages, single-blind with author names. Contributions: typed evidence-guided control; an evidence board and explicit
stop guard with auditable traces; controlled component evaluation with the live C/C+V/E ablation and same-evidence
pre/post verification contrasts, plus group-level diagnostics and independent hints. Compute is a negative result and diagnostic, not
a claimed deployed saving. Page numbers are suppressed for submission.

## Build

```sh
cd Paper/soict2026
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Needs a TeX distribution with `llncs`, `splncs04`, TikZ, `fontawesome5`, `tgheros`
(TeX Gyre Heros Condensed for the diagrams) `algorithmicx` and `placeins`. The build makes no
network or model calls.

## Numbers and figures

Results distinguish two live TEST rounds on the same 86 queries (tolerance 1 s,
raw policy, calibration disabled). The initial six-configuration benchmark:

- committed snapshot: [`benchmarks/agent/results/soict-final-tol1/`](../../benchmarks/agent/results/soict-final-tol1/README.md)
- raw runs (git-ignored, local): `benchmarks/agent/runs/soict-final-dev-test-pooled-tol1-v1/`

The supplementary C/C+V/E round is complete (258 runs):

- [audited ablation report](../../benchmarks/agent/results/soict-cv-tol1/README.md)
- raw runs: `benchmarks/agent/runs/soict-cv-20261005-v1/test/`
- self-contained numbers: `data/ablation_stats.json` (including per-query outcomes)
- C/C+V/E R@1: **47.7 / 50.0 / 58.1%**; E−C+V +8.1 pp [0.0, 16.3], McNemar p=.092.
- Same-run C+V verification: **46.5 → 50.0%**, four repairs and one harm.
- E remains slower (75.8 s) than C+V (55.6 s) and C (53.7 s).

Regenerate the new report and audit all 258 metrics offline:

```sh
.venv/bin/python -m tools.benchmark_cv_report
cp benchmarks/agent/results/soict-cv-tol1/ablation_stats.json Paper/soict2026/data/ablation_stats.json
```

Regenerate the initial derived statistics and the four data figures from the repository root:

```sh
python3 Paper/soict2026/tools/derive_stats.py          # raw runs -> data/paper_stats.json
.venv/bin/python Paper/soict2026/tools/make_figures.py # -> figures/{tradeoff,grouping,cumulative,controller}.{pdf,png}
```

`derive_stats.py` computes what the snapshot does not contain: paired bootstrap
intervals for every pair of configurations (10,000 resamples, seed 2026, base query
as the unit), exact McNemar tests, controller trace statistics (first action, router
STOP proposals, guard overrides, Jev latency/cost), the post-hoc router-STOP replay
(E† in Fig. 3), verifier reliability bins, cumulative-hint paired contrasts, and a final-order
diagnostic using the same logged candidates (all 344 A/B/C/D rankings reproduced
exactly). Raw-source hashes and per-query diagnostic outcomes are saved too.
`make_figures.py` follows the house style of the scientific-figure skill (TeX Gyre
Heros, semantic palette: blues for CAD-VR, reds for agent baselines, greys for
agent-free runs), drawn at 2× and included at text width.

| Figure | Source |
|---|---|
| Fig. 1 `figures/architecture.tex` | Redesigned TikZ with official provider logos ([sources](figures/logos/SOURCES.md)), checked against `backend/app/fusion.py`, `tara_fusion.py`, `backend/app/agent/` |
| Fig. 2 `figures/console-groups.png` | crop of `Paper/vbs2027/figures/console-en.png` (illustrative candidates, real keyframes) |
| Fig. 3–6 | `tools/make_figures.py` |
| Fig. 7 `figures/example.tex` | Initial TEST query `T-KIS:query-p1-5-kis`; times, probabilities and tool counts copied from the logged A/C/E runs; keyframes `L27_V014/{855,1060,145}` downloaded from the project media server and downsized; the lower-third in frame 855 (a private person's name and address) is blurred |

Preview a TikZ figure on its own: `pdflatex -output-directory=/tmp "\def\figname{architecture}\input{tools/preview.tex}"` from this directory.

The source ZIP contains all pre-rendered figures and logo assets, so rebuilding
the manuscript requires no downloads, model calls or raw benchmark traces.
Recomputing statistics requires the local ignored raw runs and repository metrics
module; the ZIP does not contain those large files. `REVISION_NOTES.md` records the completed C+V integration and remaining
attribution limits. Fig. 7 is included in the main PDF alongside the live ablation. Repeated prose
and captions were shortened to retain seven figures and four tables within
the content-page budget. The ablation export script is also bundled under
`tools/benchmark_cv_report.py`; recomputation needs the repository and raw runs.

## Public reproducibility link

The submission PDF links directly to
https://github.com/vonhatbao2205/AIC2026/tree/main/benchmarks/agent .
The user will make the repository public when submitting. The prepared PDF uses
“are available”; publish the local C+V snapshot and its offline exporter with that
revision. Competition media are not redistributed. A section-boundary float
barrier keeps Fig. 7 before Discussion and Limitations.
