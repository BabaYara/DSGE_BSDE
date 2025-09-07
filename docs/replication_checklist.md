# Replication Checklist (Probab_01 – Multicountry)

Purpose

- Provide a crisp, actionable list to verify we reproduce Table 1 from `Probab_01.pdf` with notebooks, scripts, and tests.

Pre‑run

- Install deps and confirm environment: `make setup` then `make env-check` (prints python, jax/equinox/optax/jupyter/pytest status).
- Set `NOTEBOOK_FAST=1` for quick sanity passes; unset for publication‑quality figures.
- Ensure `data/probab01_table1.json` exists with transcribed calibration and targets.

Model wiring

- State definition: `Ω = (η, ζ)` with sizes J and J−1; confirm eq. mapping in `docs/research_notes.md`.
- Equations: BSDE driver h, FSDE drifts/vols in `bsde_dsgE/models/probab01_equations.py` mirror eqs (19)–(22) of `Probab_01.pdf`.
- Macro network: `MacroFinanceNet` outputs `(q, σ_q, r)` with market‑clearing embedding and σ_q sign constraints; symmetric evaluator exists.

Notebook outputs

- `notebooks/multicountry_probab01.ipynb` executes and displays:
  - Seed/device/calibration banner; sample paths; diffusion heatmap.
  - Mean ± 2SE per‑country bars; rolling correlation heatmap animation.
  - Summary stats and comparison to `table1_targets` via `compare_to_targets`.
- `notebooks/primitives_visuals.ipynb` displays:
  - Sobol vs Gaussian increments/paths; distribution animation.
  - Variance growth ~ t; antithetic pairing (ρ ≈ −1) and example path pair.
  - ResNetND feature maps over 2D grid.
  - QQ plots of increments (Sobol vs Gaussian) and lag‑1 autocorrelation bars.
- `notebooks/multicountry_probab01_solver.ipynb` displays Table‑1 style `(q, σ_q)` blocks at symmetric states; σ_q sign pattern ok.

CLI checks

- Moments: `make table1-check` prints per‑key max abs errors; set `STRICT_TABLE1=1` after final targets to gate CI.
- Solver comparisons: `python scripts/compare_table1_solver.py --calib data/probab01_table1.json` prints mean |q_pred − q_tgt| per η.
- Extract Table 1 from LaTeX: `python scripts/update_table1_from_tex.py --tex Tex/Model.tex --json data/probab01_table1.json --dry-run` (use `--write` to update JSON when needed).

Tests

- `pytest -q` with `NOTEBOOK_FAST=1` passes locally (JAX installed). Coverage includes:
  - ND shapes, matrix diffusion step; solver ND smoke with matrix Σ; metrics and compare utilities.
  - Calibration loader; macro network shapes and σ_q sign pattern.
  - Notebook metadata and content tags (Mean±2SE, rolling correlations, antithetic pairing, feature maps).
  - Scripts smoke: `check_table1.py`, `compare_table1_solver.py`.

Strict gating (enable when stable)

- Set `STRICT_TABLE1=1` in CI env to enforce moment checks. Optionally set `TABLE1_Q_TOL` for solver q matching tolerance in future tests.

Provenance

- Source of truth: `Probab_01.pdf` (equations and Table 1). Cross‑references in `docs/research_notes.md` and `docs/online_research.md` (add DOIs/links offline).
