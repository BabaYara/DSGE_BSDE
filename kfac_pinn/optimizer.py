"""Minimal KFAC optimizer for JAX networks."""

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple


class KFACState(NamedTuple):
    step: int
    q: jnp.ndarray
    p: jnp.ndarray


def init_kfac(params: jnp.ndarray) -> KFACState:
    q = jnp.zeros_like(params)
    p = jnp.zeros_like(params)
    return KFACState(step=0, q=q, p=p)


def kfac_update(
    grads: jnp.ndarray, params: jnp.ndarray, state: KFACState, lr: float = 1e-3
) -> tuple[jnp.ndarray, KFACState]:
    """Single KFAC update for fully-connected layers."""
    q = 0.95 * state.q + 0.05 * (grads ** 2)
    p = 0.95 * state.p + 0.05 * (params ** 2)
    precond = grads / (jnp.sqrt(q * p) + 1e-8)
    params = params - lr * precond
    return params, KFACState(step=state.step + 1, q=q, p=p)


class KFAC:
    """Wrapper class maintaining optimizer state."""

    def __init__(self, params: jnp.ndarray):
        self.params = params
        self.state = init_kfac(params)

    def step(self, loss_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        loss, grads = jax.value_and_grad(loss_fn)(self.params)
        self.params, self.state = kfac_update(grads, self.params, self.state)
        return loss
