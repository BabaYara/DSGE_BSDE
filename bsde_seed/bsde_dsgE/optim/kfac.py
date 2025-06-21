"""Kronecker-Factored Approximate Curvature optimizer utilities."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx


def init_kfac_state(params: Any) -> Any:
    """Initialise running estimates for the Fisher information."""
    return jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)


def kfac_update(
    params: Any,
    grads: Any,
    state: Any,
    lr: float,
    decay: float = 0.95,
    damping: float = 1e-3,
) -> Tuple[Any, Any]:
    """Applies a single KFAC-style update step."""
    new_state = jax.tree_util.tree_map(
        lambda s, g: decay * s + (1.0 - decay) * jnp.square(g), state, grads
    )
    nat_grads = jax.tree_util.tree_map(
        lambda g, s: g / (s + damping), grads, new_state
    )
    new_params = jax.tree_util.tree_map(lambda p, ng: p - lr * ng, params, nat_grads)
    return new_params, new_state


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
