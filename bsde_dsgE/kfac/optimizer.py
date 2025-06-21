"""Simple KFAC optimizer for PINNs.

This is a minimal implementation that approximates the Fisher
information matrix using diagonal blocks and performs a natural
gradient update. It is intentionally lightweight to serve as a
starting point for a more complete implementation.
"""

from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp


def _init_state(params: Any) -> Any:
    """Initialise the running Fisher blocks with zeros matching ``params``."""
    return jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)


def kfac_update(
    params: Any,
    grads: Any,
    state: Any,
    lr: float,
    decay: float = 0.95,
    damping: float = 1e-3,
) -> Tuple[Any, Any]:
    """Performs a single KFAC-style update.

    Parameters
    ----------
    params:
        Current parameters.
    grads:
        Gradients of the loss with respect to ``params``.
    state:
        Running estimate of the Fisher information matrix blocks.
    lr:
        Learning rate.
    decay:
        Exponential decay factor for the running Fisher estimate.
    damping:
        Damping added to the Fisher blocks for numerical stability.
    """

    # Update Fisher blocks
    new_state = jax.tree_util.tree_map(
        lambda s, g: decay * s + (1 - decay) * jnp.square(g), state, grads
    )

    # Compute natural gradient step using diagonal Fisher approximation
    nat_grads = jax.tree_util.tree_map(
        lambda g, s: g / (s + damping), grads, new_state
    )

    new_params = jax.tree_util.tree_map(lambda p, ng: p - lr * ng, params, nat_grads)
    return new_params, new_state


__all__ = ["kfac_update", "_init_state"]

