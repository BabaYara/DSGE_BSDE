# (Ito drift/vol wrappers, jacobian utils)
"""
Small helpers: Ito's lemma, jacobians, Sobol Brownian generator.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
from scipy import stats  # type: ignore
from scipy.stats import qmc  # type: ignore


def sobol_brownian(
    dim: int,
    steps: int,
    batch: int,
    dt: float | Sequence[float],
) -> jax.Array:
    """Sobol quasi-random Brownian paths with antithetic pairing."""
    if batch % 2 != 0:
        raise ValueError("batch must be even for antithetic pairing")

    engine = qmc.Sobol(d=dim * steps, scramble=False)
    sob = engine.random(batch // 2)
    sob = jnp.clip(jnp.asarray(sob), 1e-6, 1 - 1e-6)
    g = jnp.asarray(stats.norm.ppf(sob))
    g = jnp.concatenate([g, -g], axis=0).reshape(batch, steps, dim)

    dt_arr = jnp.asarray(dt)
    if dt_arr.ndim == 0:
        scale = jnp.sqrt(dt_arr)
    elif dt_arr.shape == (steps,):
        scale = jnp.sqrt(dt_arr)[:, None]
    else:
        raise ValueError("dt must be a scalar or a sequence of length `steps`")

    return g * scale
