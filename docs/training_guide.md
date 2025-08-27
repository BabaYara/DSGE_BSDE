# Training Guide: MacroFinanceNet (Probab_01)

Purpose

- Train the Equinox MacroFinanceNet to approximate `q`, `sigma_q`, and `r` consistent with the Probab_01 equations sketch, and evaluate Table 1 symmetric-state blocks.

Prerequisites

- Python 3.10+
- JAX (CPU or GPU), Equinox, Optax, NumPy

Quick Start

- Dry run symmetric evaluation in VS Code: open `notebooks/multicountry_probab01_solver.ipynb` and run the cells (untrained net).
- Train from CLI:
  - One-step loss: `python scripts/train_macro_solver.py --epochs 2000 --paths 4096 --dt 0.001 --J 5 --steps 0 --eval-symmetric`
  - Multi-step loss: `python scripts/train_macro_solver.py --epochs 2000 --paths 4096 --dt 0.001 --J 5 --steps 64 --eval-symmetric`
  - Save checkpoint: append `--save checkpoints/macro.eqx` and later evaluate via `python scripts/eval_macro_solver.py --model checkpoints/macro.eqx --J 5`.

Notebook integration tips

- Leverage helpers from `bsde_dsgE.utils.figures` to visualise progress:
  - `simulate_paths` to rollout states with your current SDE wiring.
  - `mean_and_2se` for error bars on per-dimension means.
  - `rolling_corr` to create heatmaps/animations for time-varying correlations.
  - `impulse_response` to show IRFs using shared noise across base vs shocked paths.

Analytic symmetric-state sanity check

```
from bsde_dsgE.models.probab01_equations import q_symmetric_analytic
params = {'a': 0.1, 'psi': 5.0, 'rho': 0.03}
print(q_symmetric_analytic(params['a'], params['psi'], params['rho']))
```

Model + Equations

- Network: `bsde_dsgE/models/macro_solver.py` (`MacroFinanceNet`) maps `Omega=(eta,zeta)` to `(q, sigma_q, r)`.
- Dynamics: `bsde_dsgE/models/probab01_equations.py` (`compute_dynamics`) encodes the Try.md equation sketch (to be updated with exact equations from Probab_01.pdf).
- Loss: one-step Euler consistency on `(q, Omega)` pairs; extend to multi-step scan if needed.

Evaluation

- Symmetric states: `eta ∈ {0.3,0.4,0.5,0.6,0.7}`, `zeta_j = 1/J`. See solver notebook for plots/heatmaps.
- Compare to `data/probab01_table1.json` values (transcribed) for visual and numeric checks.
  - Strict mode: set `STRICT_TABLE1=1` and `TABLE1_Q_TOL` to assert tolerances for q.

Reproducibility

- Fix seeds (default `--seed 42`).
- Use `NOTEBOOK_FAST=1` when demoing notebooks.
- Log training loss and snapshot key hyperparameters.

Next Steps

- Ensure exact generator/state dynamics match Probab_01 equations (19)–(22) (done). If needed, add boundary/terminal for scaffolds.
- Train with multi-step scans and checkpoint. Plot loss curves in the demo notebook and match q to paper values in strict mode.
- Tighten comparisons against Table 1 targets and enable strict asserts in CI.
