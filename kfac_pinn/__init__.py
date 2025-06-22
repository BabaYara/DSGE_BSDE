"""Standalone KFAC utilities for physics-informed neural networks.

The :mod:`kfac_pinn` package contains a minimal diagonal KFAC optimiser,
a :class:`~kfac_pinn.solver.KFACPINNSolver` training loop and helpers for
toy Poisson equations.  It mirrors the functionality of
``bsde_dsgE.kfac`` but is packaged independently for clarity.
"""

from .optimizer import kfac_update, _init_state
from .solver import KFACPINNSolver
from .pde import poisson_1d_residual, pinn_loss

# Public alias for the optimiser state initialiser
init_state = _init_state

__all__ = [
    "KFACPINNSolver",
    "kfac_update",
    "init_state",
    "poisson_1d_residual",
    "pinn_loss",
]
