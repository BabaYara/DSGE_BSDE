# BSDE_20 Fix Plan

## Objectives
- Restore a clean pdflatex build for Tex/BSDE_20.tex.
- Keep the prose and structure aligned with the existing Lucas-tree narrative.

## Task Checklist
- [x] Run latexmk to capture the current failure signatures (undefined references to ss:lucas, eq:consumption, eq:capm).
- [ ] Inspect theorem/label definitions and cross-references for mismatches.
- [ ] Adjust the TeX preamble/content to resolve the issues while preserving style.
- [ ] Rebuild and record the clean output state.

## Running Log
- 2025-09-25: latexmk -pdf -interaction=nonstopmode -file-line-error -r Tex/latexmkrc Tex/BSDE_20.tex flagged unresolved references and returned a non-zero status.
