"""Convenience imports for the :mod:`bsde_dsgE` package.

This module re-exports the key KFAC optimisation utilities and solver
constructors used throughout the repository.
"""

from .kfac import kfac_update, KFACPINNSolver as KFACPINNSolverFull
from .optim import init_kfac_state, KFACPINNSolver
from .core.init import load_solver

__all__ = [
    "kfac_update",
    "KFACPINNSolverFull",
    "init_kfac_state",
    "KFACPINNSolver",
    "load_solver",
]
