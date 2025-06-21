# KFAC-PINN

A minimal Python package implementing the Kronecker-Factored Approximate
Curvature (KFAC) optimizer for training physics-informed neural networks
(PINNs). The code uses JAX for automatic differentiation and supports
simple fully connected architectures. Two example notebooks illustrate how
to solve the Poisson equation and set up a placeholder for Burgers'
equation.

## Installation

```bash
pip install -e .
```

## Usage

See the notebooks in `kfac_pinn/notebooks/` for examples. Run the Poisson
demo with:

```bash
jupyter notebook kfac_pinn/notebooks/poisson_example.ipynb
```

Training uses a few steps of KFAC and prints the final loss.
