## Lean blueprint migration (Lucas Orchard)

Purpose

- Turn Lean proofs in `lean/` into verified, navigable artifacts with dependency graphs, and link them to LaTeX statements.

Local requirements

- Lean + Lake in `lean/` (Mathlib configured in `lakefile.lean`)
- Python with `pip install leanblueprint` (Graphviz required by plasTeX dependency graph)

Daily workflow

1. Build Lean

  - `cd lean` then `lake update` (first run) and `lake build`.

1. Edit proofs

- Add/extend theorems in `lean/LucasNotes.lean` or `lean/Proofs/**`.

1. Map to blueprint

- Edit `lean/blueprint/src/content.tex`; add `\lean{Namespace.Decl}` to statements/proofs.
- Use `\leanok` when fully formalized; add `\uses{label1,label2}` for dependency graphs.

1. Preview (optional)

- `leanblueprint web` then `leanblueprint serve` to view HTML; `leanblueprint pdf` for print.

1. CI (optional)

- After running `leanblueprint new`, push to GitHub to publish to `<org>.github.io/<repo>/blueprint/`.

Current anchors

- `LucasNotes.sdf_drift_algebra` (algebraic identity used in SDF derivation) — linked in blueprint `prop:sdf` and marked `\leanok`.
- Next targets to formalize and map: Lemma C (aggregate consumption), Lemma s (share dynamics), Theorem `valuation_pde_reduced`.

Tips

- Keep algebraic identities small and composable; import Mathlib lemmas as needed.
- Run `leanblueprint checkdecls` after `lake build` to ensure all `\lean{...}` references resolve.


### 2025-09-24 08:18 - Codex update
- Captured migration tasks in PLAN.md (minted workflow, verification stamp).
- Refactored Tex/BSDE_31.tex preamble for minted-first Lean listings with listings fallback, added CI verification banner hooks, and inserted the minted/CI workflow subsection.
- Documented the minted-based verification pipeline in lean/blueprint_README.md.
- Verified latexmk -pdfxe Tex/BSDE_31.tex builds cleanly (XeLaTeX, fallback listings), noting only expected missing-glyph warnings from mono font.
### 2025-09-24 10:42 - Codex update
- Rewrote PLAN.md with the Way 3 minted + CI migration phases (audit, LaTeX refresh, Lean gate, blueprint sync, validation).
- Catalogued the Lean workspace (`LucasNotes.lean`, `Proofs/AffineMix.lean`, blueprint macros/src) and noted remaining formalization targets (Lemma C, Lemma s, `valuation_pde_reduced`).
- Confirmed XeLaTeX build: `latexmk -pdfxe Tex/BSDE_31.tex` succeeds; only expected Unicode glyph warnings from the listings fallback.

### 2025-09-24 12:05 - Codex update
- Lean inventory: `LucasNotes.aggregate_consumption_drift`, `LucasNotes.ito_quotient_rule_drift_algebra`, and `LucasNotes.sdf_drift_algebra` remain `sorry`; blueprint currently marks Lemma C and Proposition sdf as `\leanok` despite the gaps. The proven algebraic shim lives in `Tex/src/Lucas/Lucas/Identities.lean`. Outstanding target `thm:valuation_pde_reduced` still lacks a linked Lean declaration.
- Blueprint cross-check: `\lean{LucasNotes.aggregate_consumption_drift}` and `\lean{LucasNotes.ito_quotient_rule_drift_algebra}` recorded; added action item to swap in finalized names once proofs exist and to clear the premature `\leanok` tags.
- CI wiring: refreshed `.github/workflows/lean-minted-ci.yml` to run `lake build`, `lean4checker`, a hard `rg` check for `sorry|admit`, stamp `build/verified.tex`, and then build the PDF (XeLaTeX via latex-action).
- LaTeX validation: `latexmk -C -r Tex/latexmkrc Tex/BSDE_31.tex` followed by `latexmk -pdf -interaction=nonstopmode -file-line-error -r Tex/latexmkrc Tex/BSDE_31.tex` succeeds; noted Windows warning that `latexminted` is unavailable (MiKTeX minted fallback still produced the PDF).
### 2025-09-24 13:22 - Codex update
- Lean proofs: completed `LucasNotes.affine_mix`, `LucasNotes.ito_quotient_rule_drift_algebra`, `LucasNotes.sdf_drift_algebra`, and `LucasNotes.aggregate_consumption_drift`; `lake build` now clean.
- Blueprint: flagged `lem:s` algebra identity as fully formalized via `\lean{LucasNotes.ito_quotient_rule_drift_algebra}\leanok`.
- latexminted: vendored the Python package into `Tex/scripts/latexminted_lib`, added cross-platform wrappers (`Tex/scripts/latexminted`, `.bat`), and patched `Tex/latexmkrc` to prepend the scripts directory to `PATH`. Subsequent `latexmk` runs finish without the prior "latexminted not recognized" warning.
- Validation: repeated `latexmk -C -r Tex/latexmkrc Tex/BSDE_31.tex` followed by `latexmk -pdf -interaction=nonstopmode -file-line-error -r Tex/latexmkrc Tex/BSDE_31.tex`; build succeeds cleanly under the bundled latexminted tool.
### 2025-09-24 14:58 - Codex update
- Added `scripts/build_tex.ps1` so one PowerShell command runs `lake build`, optional `leanblueprint checkdecls`, and `latexmk` for any chosen TeX target.
- Documented the Lean + TeX verification workflow and script flags in README.md (section 2.4) for easy replication.
- Noted the helper as the preferred path to refresh the Lean verification stamp before pushing blueprint updates.
### 2025-09-24 15:21 - Codex update
- Hardened `scripts/build_tex.ps1` path resolution: now accepts bare filenames (auto-searches `Tex/`) and prints the attempted search paths before failing.
- Keeps the Lean/blueprint/latexmk steps unchanged so existing workflows continue to work.
### 2025-09-24 15:28 - Codex update
- Fixed `scripts/build_tex.ps1` path normalisation: `.TrimStart()` now uses character overloads so bare filenames no longer trigger cast errors when cleaning `TexFile` input.
### 2025-09-24 15:35 - Codex update
- README section 2.4 now notes that `build_tex.ps1` accepts bare filenames (auto-searches `Tex/`) and keeps the invocation example aligned with the script defaults.

### 2025-10-07 00:00 - Codex update
- Added `scripts/build_blueprint.ps1`, a wrapper that runs `lake build`, `lake exe checkdecls blueprint/lean_decls`, and (by default) the PDF build via `build_tex.ps1 -SkipLean`. Flags `-SkipLakeBuild`, `-SkipCheckDecls`, and `-SkipPdf` let you short-circuit stages; pass `-TexFile` to target a different TeX source.
- The script resolves the executable paths (Lean toolchain, helper script) relative to the repo root, so invoke it directly from anywhere inside the repo: `pwsh scripts/build_blueprint.ps1`.
