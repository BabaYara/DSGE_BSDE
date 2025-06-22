"""Pareto weight search via bisection."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

__all__ = ["pareto_bisection"]


def pareto_bisection(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
    """Find a root of ``f`` using the bisection method.

    Parameters
    ----------
    f : Callable[[float], float]
        Monotonic function whose root is sought.
    lo, hi : float
        Bracketing interval with ``f(lo)`` and ``f(hi)`` of opposite sign.
    tol : float, default=1e-6
        Termination tolerance for the interval width.
    max_iter : int, default=50
        Maximum number of iterations.

    Returns
    -------
    float
        Approximate root of ``f``.
    """

    def body(state):
        lo, hi, it = state
        mid = 0.5 * (lo + hi)
        val = f(mid)
        lo = jnp.where(val > 0, lo, mid)
        hi = jnp.where(val > 0, mid, hi)
        return lo, hi, it + 1

    def cond(state):
        lo, hi, it = state
        return (it < max_iter) & (hi - lo > tol)

    lo, hi, _ = jax.lax.while_loop(cond, body, (lo, hi, 0))
    return float(0.5 * (lo + hi))
