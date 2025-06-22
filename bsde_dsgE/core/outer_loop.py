"""Pareto weight search via bisection."""

from __future__ import annotations

from typing import Callable

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

    while (hi - lo > tol) and (max_iter > 0):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        if val > 0:
            hi = mid
        else:
            lo = mid
        max_iter -= 1

    return 0.5 * (lo + hi)
