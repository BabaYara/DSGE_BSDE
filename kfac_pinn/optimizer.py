"""Lightweight Kronecker-Factored Approximate Curvature utilities.

This module implements a minimal KFAC update suitable for
Physics-informed neural networks (PINNs). The Fisher information matrix
is approximated using diagonal blocks, resulting in a natural gradient
step that can be executed efficiently with :mod:`jax`.

Examples
--------
>>> import jax
>>> import jax.numpy as jnp
>>> from kfac_pinn import kfac_update, _init_state

Create a simple parameter tree and corresponding gradients
>>> params = {"w": jnp.ones((2, 2))}
>>> grads = jax.tree_util.tree_map(lambda p: 0.1 * jnp.ones_like(p), params)
>>> state = _init_state(params)

Update the parameters using a single KFAC step
>>> new_params, state = kfac_update(params, grads, state, lr=0.01)
"""

from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp


def _init_state(params: Any) -> Any:
    """Create an initial KFAC state.

    Parameters
    ----------
    params : Any
        PyTree of parameters whose structure and shapes will be mirrored in
        the returned state.

    Returns
    -------
    Any
        A PyTree of zeros with the same structure as ``params``.

    Examples
    --------
    >>> params = {"w": jnp.ones((2, 2))}
    >>> state = _init_state(params)
    >>> jax.tree_util.tree_map(jnp.shape, state)
    {'w': (2, 2)}
    """

    return jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)


def kfac_update(
    params: Any,
    grads: Any,
    state: Any,
    lr: float,
    decay: float = 0.95,
    damping: float = 1e-3,
) -> Tuple[Any, Any]:
    """Perform a single KFAC update step.

    Parameters
    ----------
    params : Any
        Current model parameters.
    grads : Any
        Gradients of the loss with respect to ``params``.
    state : Any
        Running estimate of the diagonal Fisher information blocks.
    lr : float
        Learning rate used for the natural gradient step.
    decay : float, default=0.95
        Exponential decay factor for the running Fisher estimate.
    damping : float, default=1e-3
        Small value added to the Fisher blocks for numerical stability.

    Returns
    -------
    Tuple[Any, Any]
        The updated ``params`` and Fisher state.

    Examples
    --------
    >>> params = {"w": jnp.ones((2, 2))}
    >>> grads = {"w": jnp.full((2, 2), 0.1)}
    >>> state = _init_state(params)
    >>> new_params, state = kfac_update(params, grads, state, lr=0.01)
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


__all__ = ["kfac_update"]

