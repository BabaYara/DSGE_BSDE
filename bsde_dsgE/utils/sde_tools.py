# (Ito drift/vol wrappers, jacobian utils)
"""
Small helpers: Ito's lemma, jacobians, Sobol Brownian generator.
"""

from typing import Tuple

import jax
import jax.numpy as jnp


def sobol_brownian(dim: int, steps: int, batch: int, dt: float) -> jax.Array:
    """Sobol quasi-random Brownian paths with antithetic pairing."""
    sob = jax.random.sobol_sample(steps * dim, batch // 2)  # type: ignore[attr-defined]
    sob = jnp.clip(sob, 1e-6, 1 - 1e-6)
    g = jax.scipy.stats.norm.ppf(sob)
    g = jnp.concatenate([g, -g], 0).reshape(batch, steps, dim)
    return g * jnp.sqrt(dt)
