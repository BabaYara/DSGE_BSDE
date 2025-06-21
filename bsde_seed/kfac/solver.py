"""Minimal KFAC solver for physics-informed neural networks."""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx


class KFACPINNSolver(eqx.Module):
    """A stub KFAC solver for PINNs."""

    net: eqx.Module
    loss_fn: Callable[[eqx.Module, jnp.ndarray], jnp.ndarray]
    lr: float = 1e-3
    num_steps: int = 100

    def run(self, x0: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Performs a dummy optimization loop using SGD."""
        params, opt_state = eqx.partition(self.net, eqx.is_array)

        @jax.jit
        def step(params, opt_state, x):
            loss, grads = jax.value_and_grad(self.loss_fn)(eqx.combine(params, opt_state), x)
            params = jax.tree_util.tree_map(lambda p, g: p - self.lr * g, params, grads)
            return params, opt_state, loss

        loss_history = []
        for _ in range(self.num_steps):
            params, opt_state, loss = step(params, opt_state, x0)
            loss_history.append(loss)
        self.net = eqx.combine(params, opt_state)
        return jnp.stack(loss_history)
