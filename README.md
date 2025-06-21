# DSGE_BSDE

This repository provides a minimal yet extendable code base for experimenting
with **physics‑informed neural networks (PINNs)** and the
Kronecker‑Factored Approximate Curvature (KFAC) optimiser.
It grew out of a teaching project on solving continuous‑time DSGE models with
backward stochastic differential equations (BSDEs), but the tools here are
general purpose and can be used for many small‑scale PINN experiments.

The core KFAC implementation lives inside the :mod:`bsde_seed.kfac` package and
is used throughout the example notebooks.

## Installation

The project is a normal Python package.  Clone the repository and install it in
editable mode:

```bash
pip install -e .
```

The installation requires a working JAX and Equinox environment.  The
``pyproject.toml`` lists these dependencies explicitly.

## Basic Usage

Once installed you can import the `KFACPINNSolver` from
`bsde_seed.bsde_dsgE.optim` (or directly from `bsde_seed.kfac`).  A minimal
training loop only needs a neural network and a loss function::

```python
import jax.numpy as jnp
import equinox as eqx
from bsde_seed.bsde_dsgE.optim import KFACPINNSolver

def residual(net: eqx.Module, x: jnp.ndarray) -> jnp.ndarray:
    y = net(x)
    return jnp.mean(y ** 2)

net = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=jax.random.PRNGKey(0))
solver = KFACPINNSolver(net=net, loss_fn=residual, num_steps=100)
losses = solver.run(jnp.zeros((1, 1)), jax.random.PRNGKey(1))
```

Further tutorials and demonstrations are available in the `notebooks/`
directory.  The newly added `pinn_kfac_training.ipynb` walks through building a
simple PINN with JAX/Equinox, optimising it using the KFAC solver and plotting
its convergence.
