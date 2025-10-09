# BSDE_14 Fix Plan

## Objectives
- Restore a clean pdflatex build for Tex/BSDE_14.tex.
- Keep the Continuous-Time Risk-Parity narrative and verification tone intact.

## Task Checklist
- [x] Run latexmk to capture the current failure signatures.
- [x] Isolate the LaTeX sources of the errors (packages, macros, citations, unicode).
- [x] Patch the document while preserving the existing style and structure.
- [x] Rebuild and log the clean output state.

## Running Log
- 2025-09-25 06:35 UTC: latexmk -pdf -interaction=nonstopmode -file-line-error -r Tex/latexmkrc Tex/BSDE_14.tex completed after biber; noted the minted/pythontex draft overrides still rely on minted being loaded.
- 2025-09-25 07:05 UTC: Adjusting the preamble to guard minted behind the draft toggle and preparing listings-based fallbacks for SymPy/Lean code boxes.
- 2025-09-25 07:18 UTC: Rebuilt with the listings fallback (latexmk -pdf -interaction=nonstopmode -file-line-error -r Tex/latexmkrc Tex/BSDE_14.tex); compilation now succeeds in 25s without the minted dependency in draft mode.
