# DSGE_BSDE
A self-contained, progressively expanding GitHub repository that teaches the BSDE method for solving continuous-time DSGE models and physics-informed neural networks.

## Installation

```bash
pip install -e .
```

Example notebooks can be found in the `notebooks` directory demonstrating the KFAC solver.

The main optimizer lives in `bsde_seed/bsde_dsgE/optim/kfac.py` and integrates
with JAX and Equinox modules to support simple PINN training loops.
