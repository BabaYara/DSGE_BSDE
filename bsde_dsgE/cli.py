"""Command-line helpers for :mod:`bsde_dsgE`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .kfac import KFACPINNSolver
from .kfac.pde import poisson_1d_residual


def pinn_demo() -> None:
    """Train a tiny 1-D Poisson PINN and print the final loss."""

    net = eqx.nn.MLP(
        in_size=1, out_size=1, width_size=16, depth=2, key=jax.random.PRNGKey(0)
    )

    def loss_fn(model: eqx.Module, interior: jax.Array) -> jax.Array:
        def net_scalar(z: jax.Array) -> jax.Array:
            return model(jnp.array([z]))[0]

        bc = jnp.array([0.0, 1.0])
        res = poisson_1d_residual(net_scalar, interior)
        bc_res = jax.vmap(net_scalar)(bc)
        return jnp.mean(res**2) + jnp.mean(bc_res**2)

    solver = KFACPINNSolver(net=net, loss_fn=loss_fn, lr=1e-2, num_steps=10)
    xs = jnp.linspace(0.0, 1.0, 16)
    losses = solver.run(xs, jax.random.PRNGKey(1))
    print("final loss", float(losses[-1]))


__all__ = ["pinn_demo"]
