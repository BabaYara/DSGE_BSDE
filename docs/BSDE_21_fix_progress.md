# BSDE_21.tex fix log (Iter. 3)

## Plan
- [x] Inspect current PythonTeX integration and confirm missing execution
- [x] Update build pipeline so latexmk runs pythontex between LaTeX passes
- [x] Patch BSDE_21.tex to reference the enforced pipeline while preserving tone
- [x] Rebuild with latexmk -r Tex/latexmkrc to verify SymPy output renders
- [x] Summarise changes + validation notes (wrapping helper, pythontex pass, MiKTeX 25.4)

## Notes
- PowerShell helper (`Tex/build_bsde10.ps1`) now wraps pdflatex -> pythontex -> pdflatex via cmd, preventing `.out` path parse errors.
- Added PythonTeX build note paragraph (latexmk / pdflatex -> pythontex -> pdflatex) per GPT-5 Pro guidance.
- GPT-5 Pro guidance: run LaTeX -> pythontex -> LaTeX; leverage latexmk rule instead of manual steps.
- pythontex CLI lacks --jobname/--auxdir flags; need to stage the root .tex alongside .pytxcode in .out before invoking pythontex.
- SymPy console helper now wraps equalities at 60 characters (plus two-space indent) so tcolorbox width isn't breached.

## Validation checklist
- [x] PDF shows SymPy console transcripts (no ?? PythonTeX ??).
- [x] latexmk exits cleanly with pythontex reporting 0 error(s), 0 warning(s).
- [ ] .out/pythontex-files-bsde_21/*.stdout regenerated on each rebuild.


## Iteration 4 (Sep 21, 2025)

- Fixed a stray leading `+` before the `\newtcolorbox{sympycheck}` definition in `Tex/BSDE_21.tex` that could break compilation on strict setups.
- Verified build using PowerShell helper:
	 - `Tex/build_bsde10.ps1 -File BSDE_21.tex`
	 - Output PDF: `Tex/.out/BSDE_21.pdf` (8 pages)
	 - PythonTeX: `0 error(s), 0 warning(s)`
	 - pdflatex: clean second pass; Cleveref first-aid note applied as usual.


## Iteration 5 (Sep 21, 2025)

- Inserted concise share lemma and offloaded Itô algebra to `Tex/scripts/ito_helpers.py` that emits `align` equations to `.out/build/ito_share_identities.tex`, included via `\input{}` with guards.
- Gated the verbose SymPy console derivation behind `\iffalse` to reduce overfull boxes; enabled line wrapping in listings.
- Adjusted PythonTeX helper to resolve paths relative to `.out`; added `\IfFileExists` guard for first LaTeX pass.
- Build validation:
	- Command: `Tex/build_bsde10.ps1 -File 'BSDE_21.tex'`
	- Result: PythonTeX 0 error(s), 0 warning(s); pdflatex produced `.out/BSDE_21.pdf`.
	- Note: one overfull hbox reported inside the generated align block; acceptable for now. Can wrap with `split` or shorten labels if desired.


