"""Lightweight KFAC update utilities for PINNs."""
from __future__ import annotations
from typing import Any, Tuple
import jax
import jax.numpy as jnp


def init_state(params: Any) -> Any:
    """Return zeros matching ``params`` tree structure."""
    return jax.tree_util.tree_map(jnp.zeros_like, params)


def kfac_update(
    params: Any,
    grads: Any,
    state: Any,
    lr: float,
    decay: float = 0.95,
    damping: float = 1e-3,
) -> Tuple[Any, Any]:
    """Perform a single diagonal KFAC update step."""
    state = jax.tree_util.tree_map(
        lambda s, g: decay * s + (1.0 - decay) * jnp.square(g), state, grads
    )
    nat_grads = jax.tree_util.tree_map(lambda g, s: g / (s + damping), grads, state)
    params = jax.tree_util.tree_map(lambda p, ng: p - lr * ng, params, nat_grads)
    return params, state


__all__ = ["init_state", "kfac_update"]
