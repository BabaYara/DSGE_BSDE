# BSDE_31 TeX Fix Plan

- [x] Review migration brief and current TeX draft
- [x] Catalogue structural or compilation issues in `Tex/BSDE_31.tex`
- [x] Implement corrective edits while preserving current narrative style
- [x] Rebuild PDF and confirm `Tex/.out/BSDE_31.pdf` refreshes without TeX errors
- [x] Note follow-ups for Lean blueprint linkage or further migration

Notes:
- Installed pygraphviz after unpacking Graphviz headers; commands now use %LOCALAPPDATA%\Graphviz-14.0.0.
- Added portable Graphviz 14.0.0 under %LOCALAPPDATA%\Graphviz-14.0.0 for pygraphviz builds.
- Logged Lean linkage gaps (Lemma C) and duplicate numbering in Appendix A.7 Lean boxes.
- `leanblueprint checkdecls` still fails because `lake` has no `checkdecls` executable (command `lake exe checkdecls blueprint/lean_decls`).
