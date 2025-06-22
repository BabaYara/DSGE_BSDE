# (tamed Euler, Milstein, full YZ CV)
"""
Core BSDE solver (tamed Euler + full YZ control‑variate)
"""

from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

__all__ = ["BSDEProblem", "Solver"]


class BSDEProblem(eqx.Module):
    drift: Callable[[jax.Array], jax.Array]
    diff: Callable[[jax.Array], jax.Array]
    generator: Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
    terminal: Callable[[jax.Array], jax.Array]
    t0: float
    t1: float

    def step(self, x: jax.Array, dt: float, dW: jax.Array) -> jax.Array:
        """One Euler step of the forward SDE (tamed)."""
        mu = self.drift(x)
        mu = mu / (1 + dt * jnp.abs(mu))
        return x + mu * dt + self.diff(x) * dW


class Solver(eqx.Module):
    net: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
    problem: BSDEProblem
    dt: float

    def __call__(self, x0: jax.Array, key: jax.Array) -> jax.Array:
        """Forward simulation returning terminal loss."""
        N = int((self.problem.t1 - self.problem.t0) / self.dt)
        dW = jax.random.normal(key, (x0.shape[0], N)) * jnp.sqrt(self.dt)

        def scan_fn(
            carry: tuple[jax.Array, jax.Array], dwi: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            x, y_lin = carry
            t = self.problem.t0
            y, z = self.net(jnp.full_like(x, t), x)
            y_cv = y - y_lin
            x1 = self.problem.step(x, self.dt, dwi)
            y1 = y - self.problem.generator(x, y, z) * self.dt + z * dwi
            y_lin1 = y_lin - self.problem.terminal(x) * self.dt
            return (x1, y_lin1), y_cv

        (_, yT_lin), y_cvs = jax.lax.scan(scan_fn, (x0, jnp.ones_like(x0)), dW.T)
        loss = jnp.mean(y_cvs ** 2 + 0.01 * yT_lin ** 2)
        return loss

