"""Optimization utilities for PINNs.

This module simply re-exports the canonical implementation from
``bsde_dsgE.kfac`` for backwards compatibility.
"""

from ..kfac import init_state as init_kfac_state, kfac_update, KFACPINNSolver

__all__ = ["init_kfac_state", "kfac_update", "KFACPINNSolver"]
