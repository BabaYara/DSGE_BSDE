"""
Multicountry (N-dim) continuous-time Lucas-style model skeleton.

This module provides a minimal, vector-valued dividend process intended as a
placeholder for the multicountry model described in Probab_01.pdf (Table 1).

It leverages the BSDEProblem interface and keeps the shapes explicit:
  - x: state vector of size ``dim`` per sample (dividends/log-dividends)
  - y: scalar value per sample
  - z: vector of size ``dim`` per sample

Notes
-----
The precise calibration and structure of the generator/terminal should be
adjusted to match the specification in Probab_01.pdf Table 1.  The functions
below follow a Lucas/CRRA-like template with OU drifts and diagonal volatilities,
which provides a reasonable starting point for empirical matching.
"""

from __future__ import annotations

from typing import Callable, Literal

import jax
import jax.numpy as jnp

from bsde_dsgE.core.solver import BSDEProblem
from bsde_dsgE.models.epstein_zin import EZParams, ez_generator


def _ensure_vector(x: jax.Array) -> jax.Array:
    """Ensure ``x`` is at least 2D (batch, dim)."""
    if x.ndim == 1:
        return x[:, None]
    return x


def multicountry_probab01(
    dim: int = 2,
    *,
    rho: float = 0.05,
    gamma: float = 8.0,
    kappa: float = 0.2,
    theta: float = 1.0,
    sigma: float | jax.Array = 0.3,
    Sigma: jax.Array | None = None,
    t0: float = 0.0,
    t1: float = 1.0,
    terminal_fn: Callable[[jax.Array], jax.Array] | None = None,
    preference: Literal["CRRA", "EZ"] = "CRRA",
    ez_params: EZParams | None = None,
    c_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> BSDEProblem:
    """Construct a vector-valued BSDEProblem for ``dim`` countries.

    Parameters
    ----------
    dim : int, default=2
        Number of countries (state dimensions).
    rho : float, default=0.05
        Impatience/discount rate in the generator.
    gamma : float, default=8.0
        Risk aversion (CRRA) coefficient.
    kappa : float, default=0.2
        Mean-reversion speed for the OU dividend processes.
    theta : float, default=1.0
        Long-run mean level for the OU processes.
    sigma : float or array, default=0.3
        Volatility level(s). If scalar, uses diagonal vol with this value.
        If array-like of shape ``(dim,)`` it sets a per-country diagonal vol.
    Sigma : array, optional
        Full ``(dim, dim)`` diffusion matrix (e.g., Cholesky or covariance). If
        provided, overrides ``sigma`` and is applied as a per-sample matrix.
    t0, t1 : float
        Time horizon.

    Returns
    -------
    BSDEProblem
        A BSDE problem with vector states. The solver must support vector
        Brownian increments (use ``SolverND``).
    """

    sig = jnp.asarray(sigma) * jnp.ones((dim,))
    Sigma_mat = None if Sigma is None else jnp.asarray(Sigma)

    def drift(x: jax.Array) -> jax.Array:
        x2 = _ensure_vector(x)
        return -kappa * (x2 - theta)

    def diff(x: jax.Array) -> jax.Array:
        x2 = _ensure_vector(x)
        if Sigma_mat is not None:
            # Return (batch, dim, dim)
            return jnp.broadcast_to(Sigma_mat, (x2.shape[0], dim, dim))
        # Else, diagonal elementwise diffusion (batch, dim)
        return jnp.broadcast_to(sig, x2.shape)

    def _default_c_fn(x: jax.Array) -> jax.Array:
        # Simple positive aggregator for dividends -> consumption
        x2 = _ensure_vector(x)
        return jnp.sum(jnp.exp(x2), axis=-1)

    if preference == "CRRA":
        def generator(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
            # Lucas/CRRA-like: f = rho*y - (gamma/2) * ||z||^2 / y
            # Shapes: y -> (batch,), z -> (batch, dim)
            z2 = jnp.sum(z**2, axis=-1)
            return rho * y - (gamma * z2) / (2.0 * jnp.maximum(y, 1e-6))
    elif preference == "EZ":
        assert ez_params is not None, "ez_params must be provided for preference='EZ'"
        gen_ez = ez_generator(ez_params, c_fn or _default_c_fn)

        def generator(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
            return gen_ez(x, y, z)
    else:
        raise ValueError(f"unknown preference type: {preference}")

    def terminal(x: jax.Array) -> jax.Array:
        if terminal_fn is not None:
            return terminal_fn(x)
        # Default: zero terminal value; customise to match the paper
        x2 = _ensure_vector(x)
        return jnp.zeros((x2.shape[0],))

    return BSDEProblem(drift, diff, generator, terminal, t0, t1)


__all__ = ["multicountry_probab01"]
