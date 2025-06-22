"""KFAC utilities for physics-informed neural networks."""
from .optimizer import init_state, kfac_update
from .solver import PINNSolver
from .pde import poisson_1d_residual, pinn_loss

__all__ = [
    "init_state",
    "kfac_update",
    "PINNSolver",
    "poisson_1d_residual",
    "pinn_loss",
]
