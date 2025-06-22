"""PDE utilities for PINNs.

This module provides helper functions to compute residuals for simple
partial differential equations. They are intended for small examples in
conjunction with :class:`~bsde_dsgE.kfac.KFACPINNSolver`.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

__all__ = ["poisson_1d_residual", "pinn_loss"]


def poisson_1d_residual(net: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray, f: Callable[[jnp.ndarray], jnp.ndarray] | None = None) -> jnp.ndarray:
    """Compute the residual of ``u''(x) = f(x)`` with zero boundary conditions.

    Parameters
    ----------
    net:
        Callable neural network approximating ``u(x)``.
    x:
        Interior points where the residual is evaluated.
    f:
        Right hand side function ``f(x)``. Defaults to zero.
    """
    if f is None:
        f = lambda x: jnp.zeros_like(x)

    d2u_dx2 = jax.vmap(jax.grad(jax.grad(net)))(x)
    return d2u_dx2 - f(x)


def pinn_loss(
    net: Callable[[jnp.ndarray], jnp.ndarray],
    interior_x: jnp.ndarray,
    bc_x: jnp.ndarray,
    f: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Return the squared residual loss for the Poisson problem."""
    res = poisson_1d_residual(net, interior_x, f)
    bc_res = net(bc_x)
    return jnp.mean(res ** 2) + jnp.mean(bc_res ** 2)
