"""Kronecker-Factored Approximate Curvature utilities for PINNs.

This module exposes the same public API as :mod:`bsde_seed.kfac` while keeping a
simple solver implementation tailored for the teaching examples in this
repository.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from bsde_seed.kfac import kfac_update, _init_state


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

init_kfac_state = _init_state


class KFACPINNSolver(eqx.Module):
    """Simple optimisation loop using :func:`kfac_update`."""

    net: eqx.Module
    loss_fn: Callable[[eqx.Module, jnp.ndarray], jnp.ndarray]
    lr: float = 1e-3
    num_steps: int = 100

    def run(self, x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        params, static = eqx.partition(self.net, eqx.is_array)
        state = init_kfac_state(params)

        @jax.jit
        def step(params, state, x):
            net = eqx.combine(params, static)
            loss, grads = eqx.filter_value_and_grad(self.loss_fn)(net, x)
            params, state = kfac_update(params, grads, state, self.lr)
            return params, state, loss

        losses = []
        for _ in range(self.num_steps):
            params, state, loss = step(params, state, x)
            losses.append(loss)
        object.__setattr__(self, "net", eqx.combine(params, static))
        return jnp.stack(losses)


__all__ = ["init_kfac_state", "kfac_update", "KFACPINNSolver"]
