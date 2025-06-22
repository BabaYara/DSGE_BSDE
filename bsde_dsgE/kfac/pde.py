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


def poisson_1d_residual(
    net: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    f: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
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
        def default_f(x: jax.Array) -> jax.Array:
            return jnp.zeros_like(x)

        f = default_f

    d2u_dx2 = jnp.asarray(jax.vmap(jax.grad(jax.grad(net)))(x))
    return d2u_dx2 - f(x)


def pinn_loss(
    net: Callable[[jax.Array], jax.Array],
    interior_x: jax.Array,
    bc_x: jax.Array,
    f: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    """Return the squared residual loss for the Poisson problem."""
    res = poisson_1d_residual(net, interior_x, f)
    bc_res = net(bc_x)
    return jnp.mean(res ** 2) + jnp.mean(bc_res ** 2)
