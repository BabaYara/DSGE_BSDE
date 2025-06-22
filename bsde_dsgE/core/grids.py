"""Utilities for Brownian path generation."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from scipy import stats  # type: ignore
from scipy.stats import qmc  # type: ignore

__all__ = ["sobol_brownian"]


def sobol_brownian(
    *,
    steps: int,
    batch: int,
    dt: float | Sequence[float],
    dim: int = 1,
) -> jax.Array:
    """Sobol quasi-random Brownian increments.

    Parameters
    ----------
    steps : int
        Number of time steps in the Brownian path.
    batch : int
        Number of independent paths to generate. Must be even to enable
        antithetic pairing.
    dt : float or sequence of float
        Fixed time step or array of varying step sizes of length ``steps``.
    dim : int, default=1
        Dimension of the Brownian motion.

    Returns
    -------
    jnp.ndarray
        Array of shape ``(batch, steps, dim)`` containing Brownian
        increments scaled by ``sqrt(dt)``.
    """

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
