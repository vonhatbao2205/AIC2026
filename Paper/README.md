# ClueScope Video Retrieval Paper

The SOICT 2026 paper on CAD-VR is in **[soict2026/](soict2026/README.md)**:
[PDF](soict2026/main.pdf) · [LaTeX](soict2026/main.tex) · [Overleaf ZIP](soict2026/cadvr-soict2026-source.zip) · [Claim map](soict2026/SOURCE_MAP.md) · [Ghi chú](soict2026/GHI_CHU_VI.md).
It presents controller-assisted fast–slow agent orchestration (main contribution) and group-by-video ranking and evaluation, evaluated on the frozen 86-query TEST benchmark.

The ClueScope VBS 2027 system paper is in **[vbs2027/](vbs2027/README.md)**:
[PDF](vbs2027/main.pdf) · [LaTeX](vbs2027/main.tex) · [Editorial notes](vbs2027/EDITORIAL_NOTES_VI.md).
It presents multimodal retrieval, video exploration, VQA assistance, and progressive hint memory, with a controlled component evaluation.

The current six-page English technical report is in **[technical-report/](technical-report/README.md)**:
[PDF](technical-report/main.pdf) · [LaTeX](technical-report/main.tex).
It focuses on video grouping, TRAKE, and the current interactive workspace.

The files directly in this directory are the earlier paper, retained as a reference.

This directory contains an IEEE-style LaTeX paper describing the AIC26 multi-modal retrieval system.

## Files

- `main.tex` - paper source, figures, tables, and appendix algorithms.
- `references.bib` - bibliography.

## Build

From this directory:

```bash
latexmk -pdf main.tex
```

Clean auxiliary files:

```bash
latexmk -C main.tex
```

The paper intentionally uses TikZ diagrams and UI placeholders so it can compile without external image assets. Replace the UI placeholder figure with real screenshots when preparing a submission.
