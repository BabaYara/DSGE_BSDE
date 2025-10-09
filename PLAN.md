# Lean Blueprint Migration Plan (Way 3 minted workflow)

## Strategic Goals
- Convert the Lucas Orchard blueprint to the 2025 "Way 3" toolchain: Lean verified first, minted (Lean4 lexer) typesetting, CI-gated PDF with verification banner.
- Keep BSDE_31.tex the primary LaTeX artifact, reflecting the new workflow without losing narrative tone.
- Ensure the Lean sources, blueprint content, and minted cache policy stay synchronized for local builds, CI, and arXiv delivery.

## Phase Breakdown
1. **Audit & Inventory**
   - Catalogue Lean projects under `lean/` (packages, blueprints, helper scripts) and capture unresolved declarations (Lemma C, Lemma s, `valuation_pde_reduced`).
   - Record current tooling assumptions (mathlib cache, leanblueprint, latexmk invocation) and highlight deltas from the Way 3 playbook.
2. **LaTeX Integration Refresh**
   - Confirm `Tex/BSDE_31.tex` preamble matches minted v3 guidance (restricted shell escape, `latexminted`, frozencache option).
   - Embed the verification banner hook (`\input{build/verified.tex}`), cache recommendations, and XeLaTeX requirement in-document.
3. **Lean Gate + CI Pipeline**
   - Standardize the workflow: `lake build` ? optional `lean4checker` replay ? grep for `sorry|admit` ? stamp `build/verified.tex`.
   - Document artifact handling (minted cache upload, arXiv tarball composition) and align GitHub Actions YAML with these steps.
4. **Blueprint Synchronization**
   - Map Lean declarations to blueprint labels using `\lean{...}`; add `\leanok` and `\uses{...}` once proofs harden.
   - Track outstanding stubs and schedule Lean work packages for each targeted lemma/theorem.
5. **Validation & Logging**
   - Rebuild BSDE_31 via `latexmk -pdfxe Tex/BSDE_31.tex`; capture warnings, ensure minted fallback remains functional.
   - Append results, open issues, and next checkpoints to `AGENTS.md` after each pass.

## Action Checklist
- [ ] Inventory Lean folders (`lean/`, `lean/Proofs/**`, blueprint scripts) and list pending formalization targets.
- [ ] Cross-check blueprint references (`\lean{...}`) against Lean declarations; flag missing links.
- [ ] Update CI/automation docs to reflect the Way 3 sequence (lake ? checker ? banner ? latexmk) and minted cache guidance.
- [ ] Validate BSDE_31.tex build under XeLaTeX with minted v3 assumptions; note cache behaviour.
- [ ] Log progress and outstanding risks in `AGENTS.md`.

## Deliverables
- Updated PLAN.md (this document) describing the Way 3 migration pipeline.
- Refined BSDE_31.tex aligned with minted-first workflow and verification banner.
- Lean inventory summary and blueprint linkage checklist recorded in `AGENTS.md`.
- Confirmed LaTeX build artifact demonstrating minted + Lean verification integration.

## Notes
- Preserve ASCII edits and existing prose tone in LaTeX narrative sections.
- Prefer restricted shell escape with `latexminted` (TeX Live 2025); retain guidance for legacy environments when necessary.
- Lean focus: keep proofs composable, rely on mathlib imports rather than re-deriving basics, and schedule formalization sprints for Lemma C, Lemma s, and the valuation PDE.