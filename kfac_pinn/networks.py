"""Simple MLP for 1D PINNs."""

import jax
import jax.numpy as jnp
from typing import Sequence


def init_mlp(sizes: Sequence[int], key: jax.random.PRNGKey) -> list[jnp.ndarray]:
    params = []
    k1, *subkeys = jax.random.split(key, len(sizes) - 1)
    keys = [k1] + subkeys
    for in_dim, out_dim, k in zip(sizes[:-1], sizes[1:], keys):
        w_key, b_key = jax.random.split(k)
        w = jax.random.normal(w_key, (in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
        b = jnp.zeros((out_dim,))
        params.append((w, b))
    return params


def mlp_apply(params: list[tuple[jnp.ndarray, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b
