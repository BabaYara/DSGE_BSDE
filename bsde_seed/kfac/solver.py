"""Minimal KFAC solver for physics-informed neural networks."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
from .optimizer import _init_state, kfac_update


class KFACPINNSolver(eqx.Module):
    """A stub KFAC solver for PINNs."""

    net: eqx.Module
    loss_fn: Callable[[eqx.Module, jnp.ndarray], jnp.ndarray]
    lr: float = 1e-3
    num_steps: int = 100

    def run(self, x0: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Performs a simple KFAC optimization loop."""
        params, opt_state = eqx.partition(self.net, eqx.is_array)
        fisher_state = _init_state(params)

        @jax.jit
        def step(params, opt_state, fisher_state, x):
            loss, grads = jax.value_and_grad(self.loss_fn)(eqx.combine(params, opt_state), x)
            # A small dampening factor stabilises optimisation on stiff problems
            params, fisher_state = kfac_update(params, grads, fisher_state, self.lr * 0.1)
            return params, fisher_state, loss

        loss_history = []
        for _ in range(self.num_steps):
            params, fisher_state, loss = step(params, opt_state, fisher_state, x0)
            loss_history.append(loss)
        self.net = eqx.combine(params, opt_state)
        return jnp.stack(loss_history)
