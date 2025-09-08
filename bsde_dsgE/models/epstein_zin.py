"""
Epstein–Zin aggregator and endogenous SDF scaffolding (continuous time).

This module provides a JAX-friendly implementation of a widely used
continuous-time Epstein–Zin (EZ) aggregator suitable for BSDE formulations
of recursive preferences. It exposes a small dataclass for parameters and a
factory producing a generator function compatible with ``BSDEProblem``.

Reference and notation
----------------------
- Preferences are characterised by the time-preference rate ``delta``, risk
  aversion ``gamma``, and elasticity of intertemporal substitution ``psi``.
- Define the canonical exponent ``theta = (1 - gamma) / (1 - 1/psi)``. This
  parameter governs the nonlinearity of the exposure term.
- An admissible continuous-time aggregator (Sauzet 2023-style normalisation)
  takes the form

    f(c, v, z) = \frac{\delta}{1 - 1/\psi} \Big( c^{1 - 1/\psi} v^{1/\psi} - v \Big)
                 - \tfrac{1}{2}\,\theta\, \frac{\|z\|^2}{v},

  where ``c`` is (aggregate) consumption, ``v`` is the continuation value
  (the BSDE ``y``), and ``z`` is the BSDE volatility (exposure) vector.

Notes
-----
- The exact normalisation conventions vary across the literature. We follow a
  form consistent with common continuous-time EZ BSDEs used in asset pricing,
  which aligns with Sauzet (2023) up to notational changes. The sign/convention
  is documented explicitly in the LaTeX file (Tex/BSDE_11.tex) and can be
  adjusted centrally here if needed.
- The SDF decomposition can be derived from this aggregator; we provide a
  helper returning the instantaneous SDF exposure implied by utility shocks
  under the EZ recursion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass
class EZParams:
    delta: float  # time preference rate
    gamma: float  # risk aversion (>0)
    psi: float    # EIS (>0, psi != 1)
    eps: float = 1e-8

    @property
    def theta(self) -> float:
        return (1.0 - self.gamma) / (1.0 - 1.0 / self.psi)


def ez_generator(params: EZParams, c_fn: Callable[[jax.Array], jax.Array]) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Return a BSDE generator implementing the EZ aggregator.

    Parameters
    - params: EZParams(delta, gamma, psi[, eps])
    - c_fn: function mapping state ``x`` (batch, dim) -> consumption ``c`` (batch,)

    Returns
    - generator(x, y, z) -> (batch,) array suitable for BSDEProblem

    Aggregator (Sauzet 2023 normalisation):
        f(c,v,z) = (delta / (1 - 1/psi)) * ( c^{1 - 1/psi} v^{1/psi} - v )
                   - 0.5 * theta * ||z||^2 / v

    with theta = (1 - gamma) / (1 - 1/psi).
    """

    delta, psi, theta, eps = params.delta, params.psi, params.theta, params.eps

    def _gen(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
        c = jnp.asarray(c_fn(x)).reshape(-1)
        y_safe = jnp.maximum(jnp.asarray(y).reshape(-1), eps)
        z2 = jnp.sum(jnp.asarray(z) ** 2, axis=-1)
        pref = (delta / (1.0 - 1.0 / psi)) * (
            jnp.power(jnp.maximum(c, eps), 1.0 - 1.0 / psi)
            * jnp.power(y_safe, 1.0 / psi)
            - y_safe
        )
        risk = -0.5 * theta * z2 / y_safe
        return pref + risk

    return _gen


def sdf_exposure_from_ez(y: jax.Array, z: jax.Array, params: EZParams) -> jax.Array:
    """Instantaneous SDF diffusion (market price of risk) implied by EZ recursion.

    Returns a vector ``lambda_t`` such that the pricing kernel satisfies
        dM_t / M_t = -r_t dt - lambda_t^\top dW_t.

    Under the EZ recursion used above, the continuation-utility shock exposure
    contributes proportionally to ``z/y`` with coefficient ``-theta``. This
    helper returns the diffusion component:
        lambda_util = -theta * z / y.

    Notes
    - Additional contributions from consumption shocks (when priced directly)
      should be added by the caller if ``c`` has its own Brownian exposure.
    - This function focuses on the utility-channel exposure that is generic to
      the EZ aggregator used here.
    """
    y_safe = jnp.maximum(jnp.asarray(y).reshape(-1, 1), params.eps)
    return -params.theta * jnp.asarray(z) / y_safe


__all__ = ["EZParams", "ez_generator", "sdf_exposure_from_ez"]

