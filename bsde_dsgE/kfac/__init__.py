"""KFAC utilities for physics-informed neural networks.

This subpackage implements a lightweight Kronecker-Factored Approximate
Curvature update along with a simple training loop for PINNs.  Additional
helpers for common PDEs are provided for convenience.
"""

from .optimizer import kfac_update, _init_state
from .solver import KFACPINNSolver
from .pde import poisson_1d_residual, pinn_loss

__all__ = [
    "KFACPINNSolver",
    "kfac_update",
    "_init_state",
    "poisson_1d_residual",
    "pinn_loss",
]
