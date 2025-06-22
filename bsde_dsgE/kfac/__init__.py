"""KFAC utilities for physics-informed neural networks.

This subpackage implements a lightweight Kronecker-Factored Approximate
Curvature update along with a simple training loop for PINNs.  Additional
helpers for common PDEs are provided for convenience.
"""

from .optimizer import _init_state, kfac_update
from .pde import pinn_loss, poisson_1d_residual, poisson_nd_residual
from .solver import KFACPINNSolver

# Public alias for the optimiser state initialiser
init_state = _init_state

__all__ = [
    "KFACPINNSolver",
    "kfac_update",
    "init_state",
    "poisson_1d_residual",
    "poisson_nd_residual",
    "pinn_loss",
]
