# BSDE_21.tex Fix Plan (Iteration 4)

## Context7 MCP Lookups
- SymPy pretty-print formatting parameters (wrap_line, num_columns) to shrink console output.
- PythonTeX pyconsole environment notes on prompt handling and string width.

## Tasks
- [x] Tighten PythonTeX verification cell formatting with helper for wrapped pretty-print output and broken long assignments.
- [x] Re-read surrounding exposition to ensure new helper description matches established style.
- [x] Remove stray leading `+` before `\newtcolorbox{sympycheck}` causing a LaTeX parse error.
- [x] Recompile `Tex/BSDE_21.tex` using `Tex/build_bsde10.ps1 -File BSDE_21.tex` to validate `pdflatex → pythontex → pdflatex`.
- [x] Update progress note once compilation passes (MiKTeX 25.4; PythonTeX: 0 error(s), 0 warning(s)).

## Notes

- Maintain existing tcolorbox structure and narrative tone.
- Keep console output deterministic so cached .pytxcode remains valid after reruns.
- Build path: outputs in `Tex/.out/BSDE_21.pdf`.
