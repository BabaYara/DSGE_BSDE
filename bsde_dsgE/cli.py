"""Command-line helpers for :mod:`bsde_dsgE`."""

from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp

from typing import Callable, cast

from .kfac import KFACPINNSolver
from .kfac.pde import poisson_1d_residual, poisson_nd_residual


def pinn_demo() -> None:
    """Train a tiny 1-D Poisson PINN and print the final loss."""

    fast = os.environ.get("NOTEBOOK_FAST")
    num_steps = 3 if fast else 10
    num_points = 8 if fast else 16

    net = eqx.nn.MLP(
        in_size=1, out_size=1, width_size=16, depth=2, key=jax.random.PRNGKey(0)
    )

    def loss_fn(net_module: eqx.Module, interior: jax.Array) -> jax.Array:
        net_fn: Callable[[jax.Array], jax.Array] = cast(
            Callable[[jax.Array], jax.Array], net_module
        )

        def net_scalar(z: jax.Array) -> jax.Array:
            return net_fn(jnp.array([z]))[0]

        bc = jnp.array([0.0, 1.0])
        res = poisson_1d_residual(net_scalar, interior)
        bc_res = jax.vmap(net_scalar)(bc)
        return jnp.mean(res**2) + jnp.mean(bc_res**2)

    solver = KFACPINNSolver(net=net, loss_fn=loss_fn, lr=1e-2, num_steps=num_steps)
    xs = jnp.linspace(0.0, 1.0, num_points)
    losses = solver.run(xs, jax.random.PRNGKey(1))
    print("final loss", float(losses[-1]))


def pinn_poisson2d() -> None:
    """Train a tiny 2-D Poisson PINN and print the final loss."""

    fast = os.environ.get("NOTEBOOK_FAST")
    num_steps = 3 if fast else 10
    num_points = 8 if fast else 16

    net = eqx.nn.MLP(
        in_size=2, out_size=1, width_size=16, depth=2, key=jax.random.PRNGKey(0)
    )

    def loss_fn(net_module: eqx.Module, interior: jax.Array) -> jax.Array:
        net_fn: Callable[[jax.Array], jax.Array] = cast(
            Callable[[jax.Array], jax.Array], net_module
        )

        def net_scalar(z: jax.Array) -> jax.Array:
            return net_fn(z)[0]

        bc = jnp.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        res = poisson_nd_residual(net_scalar, interior)
        bc_res = jax.vmap(net_scalar)(bc)
        return jnp.mean(res**2) + jnp.mean(bc_res**2)

    solver = KFACPINNSolver(net=net, loss_fn=loss_fn, lr=1e-2, num_steps=num_steps)
    xs = jax.random.uniform(jax.random.PRNGKey(1), (num_points, 2))
    losses = solver.run(xs, jax.random.PRNGKey(2))
    print("final loss", float(losses[-1]))


__all__ = ["pinn_demo", "pinn_poisson2d"]
