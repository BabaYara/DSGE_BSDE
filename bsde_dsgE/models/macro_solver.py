"""
Macro‑finance deep solver (Try.md idea) for Table 1 evaluation.

This module provides a small Equinox network that maps state ``Omega = (eta, zeta)``
to per‑country asset prices ``q``, their volatility matrix ``sigma_q``, and the risk‑free
rate ``r``. It mirrors the structure sketched in Try.md with two practical tweaks:

1) Sign‑constrained volatility: diagonal elements are positive and off‑diagonals
   negative by construction via softplus transforms. This encodes the sign pattern
   described around Table 1 and makes tests deterministic without training.
2) Evaluation helpers: utilities to build symmetric states (eta grid, uniform zeta)
   so the notebook can display Table 1‑style blocks without a full training loop.

Note: Training loops and BSDE dynamics are intentionally omitted here to keep
this module lightweight for testing and notebook import. See Try.md for a full
training sketch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


@dataclass
class Config:
    J: int = 5  # number of countries
    # Model parameters (a, delta, sigma, psi, rho) included for completeness
    a: float = 0.1
    delta: float = 0.05
    sigma: float = 0.023
    psi: float = 5.0
    rho: float = 0.03

    # State layout: eta in R^J, zeta in R^(J-1)
    @property
    def N_ETA(self) -> int:
        return self.J

    @property
    def N_ZETA(self) -> int:
        return self.J - 1

    @property
    def N_STATE(self) -> int:
        return self.N_ETA + self.N_ZETA

    # Network size
    hidden: int = 128
    depth: int = 2


class _Sine(eqx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.sin(x)


class MacroFinanceNet(eqx.Module):
    """Equinox MLP head producing (q, sigma_q, r) from Omega.

    - Input: Omega in R^(J + (J-1)) per sample
    - Outputs per sample:
      * ``q`` in R^J (via market‑clearing embedding)
      * ``sigma_q`` in R^{J×J} with diag>0 and off‑diag<0 by construction
      * ``r`` in R (softplus to keep nonnegative)
    """

    mlp: eqx.nn.MLP
    cfg: Config = eqx.static_field()

    def __init__(self, cfg: Config, key: jax.Array):
        out_size = cfg.J + (cfg.J * cfg.J) + 1
        self.mlp = eqx.nn.MLP(
            in_size=cfg.N_STATE,
            out_size=out_size,
            width_size=cfg.hidden,
            depth=cfg.depth,
            activation=_Sine(),
            key=key,
        )
        self.cfg = cfg

    def _market_clearing_q(self, xi_tilde: jax.Array, zeta: jax.Array) -> jax.Array:
        """Map xi~ and zeta to q via the embedding described in Try.md.

        zeta has shape (B, J-1). We construct zeta_J and Xi, then rescale.
        """
        cfg = self.cfg
        zeta_J = 1.0 - jnp.sum(zeta, axis=1, keepdims=True)
        zeta_J = jnp.maximum(zeta_J, 1e-8)
        zeta_full = jnp.hstack([zeta, zeta_J])
        # Xi = sum_j xi_tilde^j * zeta^j
        Xi = jnp.sum(xi_tilde * zeta_full, axis=1, keepdims=True)
        xi = (cfg.rho / Xi) * xi_tilde
        APSI_PLUS_1 = cfg.a * cfg.psi + 1.0
        q = APSI_PLUS_1 / (cfg.psi * xi + 1.0)
        return q

    def __call__(self, Omega: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # Omega: (B, N_STATE)
        raw = jax.vmap(self.mlp)(Omega)
        J = self.cfg.J
        # xi_tilde: positive
        xi_raw = raw[:, :J]
        xi_tilde = jax.nn.softplus(xi_raw) + 1e-6
        # sigma_q: shape (B, J, J) with sign constraints
        sig_flat = raw[:, J : J + J * J]
        sig_raw = sig_flat.reshape((-1, J, J))
        # diag positive via softplus, off-diagonal negative via -softplus
        diag = jnp.diagonal(sig_raw, axis1=1, axis2=2)
        diag_pos = jax.nn.softplus(diag)
        off = sig_raw - jnp.einsum("bij->bji", jnp.zeros_like(sig_raw))  # no-op to keep shape
        # Build constrained sigma_q
        sigma_q = -jax.nn.softplus(sig_raw)
        sigma_q = sigma_q.at[:, jnp.arange(J), jnp.arange(J)].set(diag_pos)
        # r: nonnegative
        r = jax.nn.softplus(raw[:, -1:])
        # q from embedding
        zeta = Omega[:, self.cfg.N_ETA :]
        q = self._market_clearing_q(xi_tilde, zeta)
        return q, sigma_q, r


def symmetric_states(cfg: Config, etas=(0.3, 0.4, 0.5, 0.6, 0.7)) -> jax.Array:
    """Construct a batch of symmetric states for the given eta grid.

    Returns Omega of shape (len(etas), N_STATE) with zeta_j = 1/J (j=1..J-1).
    """
    J = cfg.J
    z = jnp.ones((len(etas), J)) / float(J)
    z = z[:, : J - 1]
    e = jnp.stack([jnp.ones((cfg.N_ETA,)) * float(val) for val in etas], axis=0)
    return jnp.hstack([e, z])


def evaluate_symmetric(cfg: Config, net: MacroFinanceNet, etas=(0.3, 0.4, 0.5, 0.6, 0.7)) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate (q, sigma_q, r) at symmetric states.

    Returns tensors stacked along the batch dimension corresponding to the
    provided eta grid.
    """
    Omega = symmetric_states(cfg, etas)
    q, sigma_q, r = net(Omega)
    return q, sigma_q, r


__all__ = [
    "Config",
    "MacroFinanceNet",
    "symmetric_states",
    "evaluate_symmetric",
]

