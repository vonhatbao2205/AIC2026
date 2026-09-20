# VBS 2027 — method draft

`main.tex` uses Springer `llncs`. This is a development draft describing implemented PHM behavior and the intended evaluation protocol. It includes the PE-only InfoShot++ live smoke screenshot, explicit pending sections, no benchmark results and no claimed improvement. It is not ready for submission. The source has been compiled locally with `pdflatex`.

Build with a TeX distribution that includes `llncs`:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Before submission: supply authors/affiliations, verify official page and reference limits, complete related-work citations, replace pending markers with audited experiments, add architecture artwork and a live screenshot, and check every claim against run artifacts. Keep synthetic hint provenance explicit. Do not use the mock UI smoke screenshot as a live system result.

Implementation and deferred modules: [PHM status](../../docs/PROGRESSIVE_HINT_MEMORY.md). Ground-truth preparation: [AI annotation guide](../../benchmarks/PHM_GROUND_TRUTH_GUIDE.md). Benchmark execution is currently deferred by the user.
