"""Convenience re-exports for the :mod:`bsde_dsgE` package.

All public optimisation utilities live in :mod:`bsde_dsgE.kfac`.  They are
imported here so users can simply ``import bsde_dsgE`` and access the KFAC
helpers without remembering the submodule layout.
"""

from .core.init import load_solver
from .kfac import (
    KFACPINNSolver,
    kfac_update,
    pinn_loss,
    poisson_1d_residual,
    poisson_nd_residual,
)
from .kfac import (
    init_state as init_kfac_state,
)

__all__ = [
    "kfac_update",
    "init_kfac_state",
    "KFACPINNSolver",
    "pinn_loss",
    "poisson_1d_residual",
    "poisson_nd_residual",
    "load_solver",
]
