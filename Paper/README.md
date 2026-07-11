# AIC26 Retrieval Paper

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
