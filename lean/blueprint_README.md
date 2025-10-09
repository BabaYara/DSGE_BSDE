# Lean blueprint workflow (v1)

Goal: turn the Lean modules in `lean/` into verified, navigable artifacts with [PatrickMassot/leanblueprint](https://github.com/PatrickMassot/leanblueprint) while keeping the LaTeX narrative in sync.

## Tooling snapshot

- Lake project already lives in `lean/` (see `lakefile.lean`, `lake-manifest.json`).
- `lean/blueprint/` is committed; we keep source in `lean/blueprint/src/{content,web,print}.tex` and macros in `lean/blueprint/macros/`.
- Install once per machine: `pip install leanblueprint` (needs Graphviz); ensure Python is on PATH.
- Lean build sanity: from `lean/`, run `lake update` (first time) and `lake build`.

## Incremental loop (small delta changes)

1. **Touch Lean:** edit declarations in `lean/LucasNotes.lean` or `lean/Proofs/**`. Run `lake build LucasNotes` (or `lake build`) until it compiles.
2. **Map to blueprint:** in `lean/blueprint/src/content.tex`, add or update the matching statement. Use `\lean{Namespace.Decl}` inside the environment, `\leanok` when the Lean proof is complete, and `\uses{label_a,label_b}` to wire dependencies.
3. **Check glue:** from `lean/`, run `leanblueprint checkdecls` (ensures every `\lean{...}` resolves) and `leanblueprint web` to regenerate HTML; `leanblueprint pdf` gives the printable artifact. Use `leanblueprint serve` for a quick preview.
4. **Record progress:** update the roadmap or table in `content.tex` so reviewers see which LaTeX claims already have formal proofs. Commit both the Lean diff and the updated blueprint sources together.
5. **Optional CI:** hook a local pre-push script (`lake build && leanblueprint checkdecls`) or add a GitHub Action later. Until then, keep the commands in your local runbook.

## Mapping notes

- Source of truth in prose: `Tex/BSDE_31.tex` (Sections 3–9 plus Appendix A.7). Cross-reference labels directly.
- Current Lean anchors: `LucasNotes.sdf_drift_algebra` (`\leanok`) and `Proofs/AffineMix` (state simplex algebra). Next on deck: `Lemma C` (aggregate consumption), `Lemma s` (share dynamics), `valuation_pde_reduced`.
- When a proof is partial, omit `\leanok` and add a short TODO bullet under the statement.

## Useful commands

```powershell
cd lean
lake build                               # Compile Lean project
leanblueprint checkdecls                  # Ensure every \lean{...} matches
leanblueprint web && leanblueprint serve  # HTML preview (Ctrl+C to stop)
leanblueprint pdf                         # Print-friendly version
```

## Next actions

- [ ] Bring in Mathlib imports required for stochastic calculus lemmas (update `LucasNotes.lean`).
- [ ] Draft blueprint sections for `Lemma C` and `Lemma s`; mark them without `\leanok` until formalized.
- [ ] Add a short "How to contribute" snippet to the main README once the loop above feels stable.


## 2025 verification pipeline (minted + CI)

1. Run `lake build` (or `lake build LucasNotes`) under CI using `leanprover/lean-action@v1`; enable the `lean4checker` replay to detect environment hacks.
2. Enforce a `grep -nE "\\b(sorry|admit)\\b" lean/` guard so unfinished proofs fail the pipeline.
3. After a successful build, write a verification banner (commit, toolchain, timestamp, no-sorry flag) to `build/verified.tex`; this file is `\input` into `Tex/BSDE_31.tex` right after `\maketitle`.
4. Compile the paper with `latexmk -pdfxe Tex/BSDE_31.tex`. On TeX Live 2025 the bundled `latexminted` binary allows restricted shell escape for `minted`; fall back to `--shell-escape` on older images.
5. For arXiv-style uploads, switch the preamble to `\usepackage[frozencache,cachedir=minted-cache]{minted}` and include the populated `minted-cache/` directory in the tarball.

> The LaTeX file now defines `\leanfile`/`leanblock` macros that prefer `minted`'s Lean~4 lexer when available and degrade to the legacy `listings` setup otherwise.
