"""Minimal training loop for PINNs using a KFAC natural gradient."""
from __future__ import annotations
from typing import Any, Callable
import jax
import jax.numpy as jnp
import equinox as eqx
from .optimizer import init_state, kfac_update


class PINNSolver(eqx.Module):
    """Train ``net`` on ``loss_fn`` using a KFAC update."""

    net: eqx.Module
    loss_fn: Callable[[eqx.Module, jnp.ndarray], jnp.ndarray]
    lr: float = 1e-3
    num_steps: int = 100

    def run(self, x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        params, static = eqx.partition(self.net, eqx.is_array)
        state = init_state(params)

        @jax.jit
        def step(params: Any, state: Any, x: jnp.ndarray) -> tuple[Any, Any, jnp.ndarray]:
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


__all__ = ["PINNSolver"]
