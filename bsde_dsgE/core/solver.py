# (tamed Euler, Milstein, full YZ CV)
"""
Core BSDE solver (tamed Euler + full YZ control‑variate)
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["BSDEProblem", "Solver"]


class BSDEProblem(eqx.Module):
    drift: Callable[[jax.Array], jax.Array]
    diff: Callable[[jax.Array], jax.Array]
    generator: Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
    terminal: Callable[[jax.Array], jax.Array]
    t0: float
    t1: float

    def step(self, x: jax.Array, t: float, dt: float, dW: jax.Array) -> jax.Array:
        """One Euler step of the forward SDE (tamed).

        Supports both elementwise and matrix-valued diffusion:
        - If ``diff(x)`` has the same ndim as ``x``, uses elementwise product.
        - If ``diff(x)`` has one extra dim, treats it as a per-sample matrix and
          multiplies by ``dW`` via ``einsum``.
        """
        mu = self.drift(x)
        mu = mu / (1 + dt * jnp.abs(mu))
        sigma = self.diff(x)
        if sigma.ndim == x.ndim:
            inc = sigma * dW
        elif sigma.ndim == x.ndim + 1:
            # (batch, dim, dim) @ (batch, dim) -> (batch, dim)
            inc = jnp.einsum("bij,bi->bj", sigma, dW)
        else:
            raise ValueError("diff(x) must match x ndim or be one higher for matrix diffusion")
        return x + mu * dt + inc


class Solver(eqx.Module):
    net: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
    problem: BSDEProblem
    dt: float

    def __call__(self, x0: jax.Array, key: jax.Array) -> jax.Array:
        """Forward simulation returning terminal loss."""
        N = int((self.problem.t1 - self.problem.t0) / self.dt)
        dW = jax.random.normal(key, (x0.shape[0], N)) * jnp.sqrt(self.dt)

        def scan_fn(
            carry: tuple[float, jax.Array, jax.Array],
            dwi: jax.Array,
        ) -> tuple[tuple[float, jax.Array, jax.Array], jax.Array]:
            t, x, y_lin = carry
            y, z = self.net(jnp.full_like(x, t), x)
            y_cv = y - y_lin
            x1 = self.problem.step(x, t, self.dt, dwi)
            y_lin1 = y_lin - self.problem.terminal(x) * self.dt
            t1 = t + self.dt
            return (t1, x1, y_lin1), y_cv

        (_, _, yT_lin), y_cvs = jax.lax.scan(
            scan_fn,
            (self.problem.t0, x0, jnp.ones_like(x0)),
            dW.T,
        )
        loss = jnp.mean(y_cvs**2 + 0.01 * yT_lin**2)
        return loss


class SolverND(eqx.Module):
    """Vector-state BSDE solver with diagonal diffusion and per-dim controls.

    Expects a network that maps ``(t, x) -> (y, z)`` where ``x`` has shape
    ``(batch, dim)``, ``y`` has shape ``(batch,)`` and ``z`` has shape
    ``(batch, dim)``.
    """

    net: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
    problem: BSDEProblem
    dt: float

    def __call__(self, x0: jax.Array, key: jax.Array) -> jax.Array:
        x0_2d = x0 if x0.ndim == 2 else x0[:, None]
        batch, dim = x0_2d.shape
        N = int((self.problem.t1 - self.problem.t0) / self.dt)
        dW = jax.random.normal(key, (N, batch, dim)) * jnp.sqrt(self.dt)

        def scan_fn(
            carry: tuple[float, jax.Array, jax.Array],
            dwi: jax.Array,
        ) -> tuple[tuple[float, jax.Array, jax.Array], jax.Array]:
            t, x, y_lin = carry
            y, z = self.net(jnp.full((batch,), t), x)
            y_cv = y - y_lin
            x1 = self.problem.step(x, t, self.dt, dwi)
            y_lin1 = y_lin - self.problem.terminal(x) * self.dt
            t1 = t + self.dt
            return (t1, x1, y_lin1), y_cv

        (_, _, yT_lin), y_cvs = jax.lax.scan(
            scan_fn,
            (self.problem.t0, x0_2d, jnp.ones((batch,))),
            dW,
        )
        loss = jnp.mean(y_cvs**2 + 0.01 * yT_lin**2)
        return loss
