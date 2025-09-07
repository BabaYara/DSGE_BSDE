# Research Notes: Multicountry BSDE/DSGE (Probab_01.pdf)

Purpose

- Track model assumptions, calibration sources, and links to references used to reproduce Table 1 in Probab_01.pdf.
- Serve as the working sheet to transcribe equations and values before codifying them into `data/probab01_table1.json`.

Source of truth

- The definitive model specification and Table 1 values come from `Probab_01.pdf` (repo root). Cite page/section numbers when transcribing.

Model summary (current mapping)

- States: η_t^i (experts’ net worth to asset value per country) and ζ_t^i (shares of asset value), with J countries. Symmetric analytic price under ζ_j=1/J is q = (aψ+1)/(ρψ+1) (eq. 19).
- Diffusions: Define σ^{qK}_{i,j} = 1{j=i}·σ + σ^q_{i,j} and σ^H = ζ·σ^{qK}.
- BSDE driver (eq. 20): for each i,
  h_i(η_i, q_i, σ^q_{i,·}) = (aψ+1)/ψ + (q_i/ψ)·ln q_i − q_i·(1/ψ + δ) + σ·q_i·σ^q_{i,i} − (q_i/η_i)·Σ_j (σ^{qK}_{i,j})² − q_i·r.
  Diffusion: Z_{i,j} = q_j·σ^q_{i,j}.
- Forward SDEs: (eq. 21) for η and (eq. 22) for ζ.
  - η drift: b_η^i = ([ (aψ+1)/(ψ·q_i) − 1/ψ − ρ ]·η_i) + ((1/η_i − 1)·η_i)·Σ_j (σ^{qK}_{i,j})²; vol: (1−η_i)·σ^{qK}_{i,·}.
  - ζ drift: b_ζ^i = ζ_i[ μ^{qK}_i − μ_H + Σ_l σ^H_l (σ^{qK}_{i,l} − σ^H_l) ], where μ^{qK}_i = −(aψ+1)/(ψ·q_i) + 1/ψ + (1/η_i)·Σ_l (σ^{qK}_{i,l})² + r and μ_H = ζ·μ^{qK}; vol: ζ_i·(σ^{qK}_i − σ^H).
- Terminal: use paper-specific condition; current notebooks use zero as a placeholder for path-sim scaffolds.

Table 1 – parameters (transcribed)

| Parameter | Symbol | Value | Units | Source (pdf page/eq) | Notes |
|-----------|--------|-------|-------|-----------------------|-------|
| Countries | J      | 5     |       | Table 1 (Symmetric)   | used by MacroFinanceNet |
| a, ψ, ρ   | a, ψ, ρ| 0.1, 5, 0.03 | | Eq. (19)              | q analytic sanity check |
| δ, σ      | δ, σ   | 0.05, 0.023 | | Eqs. (20)–(22)       | drift/driver coefficients |

See `data/probab01_table1.json:probab01_params`.

Table 1 – targets (symmetric states)

| Block            | Symbol     | Dimension | Value(s)                               | Source |
|------------------|------------|-----------|----------------------------------------|--------|
| Asset prices     | q_i        | 5         | See `table1_values.symmetric_states`   | Table 1 |
| Price volatility | σ^{q}_{i,j}| 5×5       | See `table1_values.symmetric_states`   | Table 1 |

Note: path‑moment targets (mean/std/cov/corr) are not used for Table 1; CLI prints simulated stats when present but are optional.

JSON calibration mapping

- After filling the tables above, create/edit `data/probab01_table1.json` with keys:
  - `dim`, `rho`, `gamma`, `kappa`, `theta`, and either `sigma` (scalar or list) or full `Sigma` matrix.
  - Optional names: `country_names_dim` (length=`dim`) for dim‑level plots; `country_names_J` (length=J) for Table‑1 (σ_q) heatmaps and symmetric‑state displays.
  - `table1_targets` with a nested `examples` dict, e.g.:

```json
{
  "dim": 2,
  "rho": 0.05,
  "gamma": 8.0,
  "kappa": 0.2,
  "theta": 1.0,
  "sigma": [0.3, 0.25],
  "table1_values": {"symmetric_states": [ {"eta": 0.3, "zeta": 0.2, "q": [...], "sigma_q": [[...],...]} , ... ]}
}
```

Verification flow

- Run `notebooks/multicountry_probab01.ipynb`; the “Comparison” cell will print per-key max absolute errors and an overall pass flag.
- For CLI: create `python scripts/check_table1.py` (optional) to load JSON and print comparisons (see README snippet).

Device + reproducibility notes

- Set NOTEBOOK_FAST=1 for CI/quick runs; unset for full figures.
- Fix seed: JAX PRNGKey(0) in notebooks/CLI. Echo seed, device, and hyperparams in the first cell of notebooks.
- Use deterministic Sobol paths for figures where possible (antithetic pairing included).

References (generic placeholders)

- Ma, J., Protter, P., Yong, J. (1994). Solving forward–backward SDEs via the four step scheme.
- Han, J., Jentzen, A., E, W. (2018). Solving high-dimensional PDEs using deep learning.
- Lucas, R. (1978). Asset Prices in an Exchange Economy.

Planned checks

- Moment matching: reproduce Table 1 statistics with assert-allclose tolerances via `compare_to_targets`.
- Sensitivity: compare outcomes under alternative σ, κ, γ grids (document any trade-offs).
- Diagnostics: seed robustness and fast/slow configurations; check that sample-path variance ~ t for Brownian increments.

Transcription guide (for Probab_01.pdf)

- Locate the exact model equations (state SDEs, generator f, terminal g). Copy equation numbers and page numbers here.
- Identify the calibration block that yields Table 1, including any correlation matrix Σ and parameter priors if applicable.
- Transcribe Table 1 values into `data/probab01_table1.json`:
  - Keys: `dim`, `rho`, `gamma`, `kappa`, `theta`, `sigma` (or `Sigma`).
  - Targets under `table1_targets.examples`: `mean_state`, `std_state`, optionally `cov` and `corr`.
- Re-run `scripts/check_table1.py` and the notebook comparison cell to verify tolerances.

Added notes

- `data/probab01_table1.json` includes `table1_values` with symmetric-state `q_i` and `σ_{q,i,j}` for η ∈ {0.3,0.4,0.5,0.6,0.7}, ζ_j = 1/J. Used for display/inspection and strict checks.
- `notebooks/multicountry_probab01_solver.ipynb` evaluates MacroFinanceNet at symmetric states, enforces the σ_q sign pattern by construction, and prints/plots comparable Table 1 blocks.
