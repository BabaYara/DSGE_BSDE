"""PDE utilities for PINNs.

This module provides helper functions to compute residuals for simple
partial differential equations. They are intended for small examples in
conjunction with :class:`~bsde_dsgE.kfac.KFACPINNSolver`.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

__all__ = ["poisson_1d_residual", "poisson_nd_residual", "pinn_loss"]


def poisson_1d_residual(
    net: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    f: Callable[[jax.Array], jax.Array] | None = None,
    dirichlet_bc: Callable[[jax.Array], jax.Array] | None = None,
    neumann_bc: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    """Compute the residual of ``u''(x) = f(x)``.

    Optionally evaluate boundary conditions when ``x`` contains boundary
    locations. The function returns the PDE residual for the interior and, if a
    boundary condition is supplied, the boundary residual as well.  Dirichlet
    and Neumann conditions are mutually exclusive.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> def net(x):
    ...     return x * (1 - x)
    >>> x = jnp.array([0.25, 0.5, 0.75])
    >>> poisson_1d_residual(net, x)
    ...
    >>> bc = jnp.array([0.0, 1.0])
    >>> poisson_1d_residual(net, bc, dirichlet_bc=lambda z: 0.0)
    
    Parameters
    ----------
    net:
        Callable neural network approximating ``u(x)``.
    x:
        Interior points where the residual is evaluated.
    f:
        Right hand side function ``f(x)``. Defaults to zero.
    dirichlet_bc:
        Function enforcing ``u(x)=dirichlet_bc(x)`` at the boundary.  Defaults
        to ``None``.
    neumann_bc:
        Function enforcing ``u'(x)=neumann_bc(x)`` at the boundary.  Defaults to
        ``None``.
    """
    if dirichlet_bc is not None and neumann_bc is not None:
        msg = "Only one of `dirichlet_bc` or `neumann_bc` may be provided."
        raise ValueError(msg)

    if dirichlet_bc is not None:
        return net(x) - dirichlet_bc(x)

    if neumann_bc is not None:
        return jax.vmap(jax.grad(net))(x) - neumann_bc(x)

    if f is None:

        def default_f(x: jax.Array) -> jax.Array:
            return jnp.zeros_like(x)

        f = default_f

    d2u_dx2 = jnp.asarray(jax.vmap(jax.grad(jax.grad(net)))(x))
    return d2u_dx2 - f(x)


def poisson_nd_residual(
    net: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    f: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    """Compute the residual of ``\nabla^2 u(x) = f(x)`` for arbitrary dimension.

    Parameters
    ----------
    net:
        Callable neural network approximating ``u(x)``.
    x:
        Array of shape ``(n, d)`` of interior points where the residual is
        evaluated.
    f:
        Right hand side function ``f(x)``. Defaults to zero.
    """
    if f is None:

        def default_f(z: jax.Array) -> jax.Array:
            return jnp.zeros((), dtype=z.dtype)

        f = default_f

    def laplacian(z: jax.Array) -> jax.Array:
        return jnp.trace(jax.hessian(net)(z))

    res = jax.vmap(laplacian)(x)
    return res - jax.vmap(f)(x)


def pinn_loss(
    net: Callable[[jax.Array], jax.Array],
    interior_x: jax.Array,
    bc_x: jax.Array,
    f: Callable[[jax.Array], jax.Array] | None = None,
    dirichlet_bc: Callable[[jax.Array], jax.Array] | None = None,
    neumann_bc: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    """Return the squared residual loss for the Poisson problem.

    Either Dirichlet or Neumann boundary conditions can be specified using the
    ``dirichlet_bc`` or ``neumann_bc`` callables. By default zero Dirichlet
    conditions are enforced.

    Examples
    --------
    >>> bc = jnp.array([0.0, 1.0])
    >>> pinn_loss(net, interior, bc, dirichlet_bc=jnp.sin)
    >>> pinn_loss(net, interior, bc, neumann_bc=lambda x: 0.0)
    """

    if dirichlet_bc is not None and neumann_bc is not None:
        msg = "Specify at most one boundary condition type"
        raise ValueError(msg)

    res = poisson_1d_residual(net, interior_x, f)

    if dirichlet_bc is not None:
        bc_res = poisson_1d_residual(net, bc_x, dirichlet_bc=dirichlet_bc)
    elif neumann_bc is not None:
        bc_res = poisson_1d_residual(net, bc_x, neumann_bc=neumann_bc)
    else:
        bc_res = net(bc_x)

    return jnp.mean(res**2) + jnp.mean(bc_res**2)
