# VBS 2027 — method draft

`main.tex` uses Springer `llncs`. This is a development draft describing implemented PHM behavior and the completed locked evaluation. It includes the PE-only InfoShot++ live smoke screenshot, audited locked benchmark tables for PE and PE+Qwen, and explicit pending sections for remaining submission work. The experiments show mixed results: PHM exceeds Hint-RRF but does not demonstrate a mean-prefix MRR advantage over cumulative retrieval. It is not ready for submission. The source has been compiled locally with `pdflatex`.

Build with a TeX distribution that includes `llncs`:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Before submission: supply authors/affiliations, verify official page and reference limits, complete related-work citations, complete remaining pending markers and review the integrated experiments, add architecture artwork and a live screenshot, and check every claim against run artifacts. Keep synthetic hint provenance explicit. Do not use the mock UI smoke screenshot as a live system result.

Implementation and deferred modules: [PHM status](../../docs/PROGRESSIVE_HINT_MEMORY.md). Ground-truth preparation: [AI annotation guide](../../benchmarks/PHM_GROUND_TRUTH_GUIDE.md). The controlled PE and PE+Qwen benchmark is complete; see the audited artifacts below.

Benchmark artifacts: [report, CSV, LaTeX tables and plots](../../benchmarks/progressive/results/paper-20260921/REPORT.md). The updated draft compiles to six pages locally; author details, citations and final layout remain unfinished.
