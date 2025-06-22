# (scalar Lucas tree, CRRA)
"""
Scalar Lucas-tree dividend with CRRA utility; minimal example.
"""

from __future__ import annotations
from typing import Callable
import jax.numpy as jnp
from bsde_dsgE.core.solver import BSDEProblem


def scalar_lucas(rho: float = 0.05, gamma: float = 10.0) -> BSDEProblem:
    def drift(x):
        return -0.2 * (x - 1.0)

    def diff(x):
        return 0.3 * x

    def generator(x, y, z):
        return rho * y - gamma * z ** 2 / (2 * y)

    def terminal(x):
        return 0.0

    return BSDEProblem(drift, diff, generator, terminal, 0.0, 1.0)

