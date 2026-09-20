# AIC26 Retrieval Paper

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
