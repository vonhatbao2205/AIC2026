# ClueScope

**ClueScope: Evidence-Centric Multimodal Video Retrieval for Interactive Search**

[Read the paper](main.pdf) · [Overleaf source ZIP](cluescope-vbs2027-source.zip) · [LaTeX source](main.tex) · [Bibliography](references.bib) · [Source and claim audit](SOURCE_MAP.md) · [Editorial notes in Vietnamese](EDITORIAL_NOTES_VI.md)

The manuscript presents the complete implemented retrieval/inspection system, with PHM as one evaluated component. It covers multimodal retrieval, TARA clip context, video grouping, spatial KIS, VQA assistance, progressive evidence, and answer review. TRAKE is excluded at the author's request. The review version uses **Anonymous** in the author block, running header, and PDF author metadata, with no affiliation displayed. Citations, cross-references, and bibliography URLs/DOIs are blue, matching the supplied `Part4.pdf`.

The [VBS 2027 CFP](https://videobrowsershowdown.org/call-for-papers/), checked on 22 September 2026, specifies Springer LNCS, six content pages plus up to two reference pages, a screenshot, and a detailed interactive workflow. The manuscript uses the unmodified LNCS text size and margins. All twelve VBS 2026 system papers in the supplied Part IV are cited, together with the relevant representation and fusion sources.

The 23 September editorial revision expands the abstract to 206 words, separates **Introduction** from **Related Work**, and gives the three system contributions their own statements. The manuscript describes a collected video corpus without naming the earlier competition dataset; the internal source audit retains the original provenance. Fig. 1 is now a horizontal, editable vector diagram with Milvus/Elasticsearch logos and modality icons.

The local PDF has **8 pages: 6 content + 2 references**, with 29 references, no undefined citations, and no overfull boxes. The abstract, opening sections, architecture diagram, and final content page were visually inspected after compilation. The standard LNCS/amsmath `vec` warning is non-fatal. This revision changes paper assets only; application tests and the locked benchmark were not rerun.

The reference review adds nine directly cited sources: VBS 2025 task design, earlier Exquisitor interaction work, Milvus, Hungarian assignment, DRES, bootstrap resampling, and the V3C/MVK/GynSurg collections. Bibliographic details were checked against primary papers, author/project pages, and publisher DOI metadata. All 29 entries are cited in the text; collection citations remain in the deployment discussion and do not imply that those collections were used in the component benchmark.

Perception Encoder now cites its NeurIPS 2025 publication, and NVILA its CVPR 2025 publication, including published pages and DOIs. DeepSeek-V4.1-Flash retains its arXiv reference; the VQA paragraph attaches that citation to the model and attributes the separate verification workflow to ClueScope. External grounding depends on provider tool support and was not live-tested during this revision.

## Build

```sh
cd Paper/vbs2027
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

A TeX distribution with `llncs`, `splncs04`, TikZ, Font Awesome 5, and the standard packages in `main.tex` is required. The source ZIP contains the manuscript, bibliography, figures, and the installed LNCS class/bibliography style for convenient upload to Overleaf. A build needs no API keys, remote services, or benchmark rerun.

## Figure provenance

- `figures/architecture.tex`: horizontal TikZ diagram, checked against frame RRF, video grouping, TARA fusion, PHM, and the review workflow. TARA joins at video level; PHM consumes pre-TARA evidence. The amber dashed path issues local searches; the lower dashed path is operator refinement.
- [Architecture PDF](figures/architecture.pdf) / [PNG preview](figures/architecture.png): standalone exports. Rebuild with `pdflatex -output-directory=/tmp tools/architecture-preview.tex` from this directory, then copy `/tmp/architecture-preview.pdf` to `figures/architecture.pdf`.
- `figures/logos/`: unmodified Milvus and Elasticsearch logo files already present in the parent paper directory. Other icons use Font Awesome 5.
- `figures/console-en.png`: current React application, rendered with illustrative candidates from the earlier report and actual source keyframes. Ranking scores are demonstration values.
- `figures/progressive-ledger.png` and `figures/progressive-evidence.png`: current English UI, re-rendered with the two archived PE-only live smoke snapshots from 20 September 2026. Re-rendering performs no live retrieval. The original session predates the benchmark's global-frontier revision and is not a judged task.
- `tools/progressive-fixture.json`: archived retrieval responses used for that re-render; not a fresh run or modified evaluation result.
- `tools/capture-ui.cjs`: browser capture script. The image-only capture excludes the unrelated task selector and does not submit answers. Source-video text remains in its original language.

To reproduce screenshots, start isolated services from the repository root:

```sh
mkdir -p /tmp/aic-vbs-paper/config /tmp/aic-vbs-paper/data
AIC26_MOCK_MODE=true AIC26_CONFIG_DIR=/tmp/aic-vbs-paper/config \
 AIC26_SECRET_DIR=/tmp/aic-vbs-paper/config AIC26_DATA_DIR=/tmp/aic-vbs-paper/data \
 PYTHONPATH=backend backend/.venv/bin/python -m uvicorn app.main:app \
 --host 127.0.0.1 --port 8019
```

In a separate terminal:

```sh
cd frontend
VITE_SUPABASE_URL='' VITE_SUPABASE_PUBLISHABLE_KEY='' \
 VITE_API_TARGET=http://127.0.0.1:8019 npm run dev -- \
 --host 127.0.0.1 --port 5179 --strictPort --mode test
```

Then run the capture from the repository root, supplying your installed Playwright module and Chromium paths as needed:

```sh
PLAYWRIGHT_MODULE=/path/to/playwright CHROME_PATH=/path/to/chromium \
 node Paper/vbs2027/tools/capture-ui.cjs
```

The script uses the existing `Paper/technical-report/tools/demo.json` for the general console. Remote source keyframe URLs must still be accessible. It does not use the user's browser profile or enable Supabase/DRES writes.

## Evidence and remaining deployment work

The [locked benchmark report](../../benchmarks/progressive/results/paper-20260921/REPORT.md) supplies every table value. No new evaluation or tuning was performed during manuscript editing. The study establishes neither a PHM advantage over cumulative search nor full-system/VBS performance.

The implemented AVS basket and configurable DRES submission path are described in the manuscript. The console enables sending only when requested and the server is configured. Competition deployment remains unverified: collection ingestion, identity/capability adaptation, answer-format validation, live-server checks, and DRES interaction-log delivery still need work. This supersedes the earlier notes that described AVS as absent and direct submission as hard-disabled.
