"""Training loop for a simple 1D PINN using the KFAC optimizer."""

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Callable, Sequence

from .optimizer import KFAC, kfac_update, init_kfac


def feature_map(x: jnp.ndarray, m: int) -> jnp.ndarray:
    """Trigonometric features [sin(k*pi*x)] for k=1..m."""
    x = jnp.atleast_1d(x)
    k = jnp.arange(1, m + 1)
    return jnp.sin(jnp.pi * k * x[:, None])


def train_pinn(m: int,
               residual_fn: Callable[[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray], jnp.ndarray],
               x0: jnp.ndarray,
               steps: int = 500,
               lr: float = 1e-3) -> tuple[jnp.ndarray, list[float]]:
    """Train Fourier series coefficients with KFAC."""
    params = jnp.zeros(m)
    opt = KFAC(params)
    losses = []

    def u(x, p):
        return jnp.dot(feature_map(x, m), p).squeeze()

    def loss_fn(p):
        r = residual_fn(lambda x: u(x, p), x0)
        return jnp.mean(r ** 2)

    for _ in range(steps):
        loss, grads = jax.value_and_grad(loss_fn)(opt.params)
        opt.params, opt.state = kfac_update(grads, opt.params, opt.state, lr)
        losses.append(loss)
    return opt.params, losses
