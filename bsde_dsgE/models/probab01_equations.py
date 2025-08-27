"""
Probab_01 equations (structured per Try.md) with JAX functions.

This module exposes a `Config` and `compute_dynamics` utility that mirror the
implementation idea in Try.md. These functions are imported by the training
script and can be unit-tested for shape consistency.

Note: Equation numbers and exact functional forms should be transcribed from
Probab_01.pdf and cross-referenced here (TODO). The current code reflects the
sketch in Try.md and is meant as a structurally faithful placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


@dataclass
class Config:
    # Model params (as per Try.md)
    J: int = 5
    a: float = 0.1
    delta: float = 0.05
    sigma: float = 0.023
    psi: float = 5.0
    rho: float = 0.03

    # State dims: eta in R^J, zeta in R^(J-1)
    @property
    def N_ETA(self) -> int:
        return self.J

    @property
    def N_ZETA(self) -> int:
        return self.J - 1

    @property
    def N_STATE(self) -> int:
        return self.N_ETA + self.N_ZETA


def compute_dynamics(
    cfg: Config,
    Omega: jax.Array,  # (B, N_STATE)
    q: jax.Array,      # (B, J)
    sigma_q: jax.Array,  # (B, J, J)
    r: jax.Array,        # (B, 1)
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute (drift_X, vol_X, h, Z) given state and controls.

    Shapes
    - drift_X: (B, N_STATE)
    - vol_X:   (B, N_STATE, J)
    - h:       (B, J)
    - Z:       (B, J, J)
    """
    J = cfg.J
    A, PSI, RHO, SIGMA, DELTA = cfg.a, cfg.psi, cfg.rho, cfg.sigma, cfg.delta
    APSI_PLUS_1 = A * PSI + 1.0

    eta = Omega[:, : cfg.N_ETA]
    zeta = Omega[:, cfg.N_ETA :]

    zeta_J = 1.0 - jnp.sum(zeta, axis=1, keepdims=True)
    zeta_J = jnp.maximum(zeta_J, 1e-8)
    zeta_full = jnp.hstack([zeta, zeta_J])

    q_safe = jnp.maximum(q, 1e-8)
    eta_safe = jnp.maximum(eta, 1e-8)

    I_J = jnp.eye(J)
    sigma_qK = I_J * SIGMA + sigma_q  # (B, J, J)
    sum_sq_sigma_qK = jnp.sum(jnp.square(sigma_qK), axis=2)  # (B, J)

    # BSDE driver h (Eq sketch)
    sigma_q_diag = jnp.diagonal(sigma_q, axis1=1, axis2=2)
    h_term1 = APSI_PLUS_1 / PSI
    h_term2 = (q / PSI) * jnp.log(q_safe)
    h_term3 = -q * (1.0 / PSI + DELTA)
    h_term4 = SIGMA * q * sigma_q_diag
    h_term5 = -(q / eta_safe) * sum_sq_sigma_qK
    h_term6 = -q * r
    h = h_term1 + h_term2 + h_term3 + h_term4 + h_term5 + h_term6

    # FSDE drifts
    b_eta_t1_inner = (APSI_PLUS_1 / (PSI * q_safe)) - (1.0 / PSI) - RHO
    b_eta_t1 = b_eta_t1_inner * eta
    # Ito correction (per eq. 21): proportional to ((1/eta) - 1) * eta * sum of squares
    # Per Model.tex (eq. 21): ((1/eta) - 1)^2 * eta * sum_j (sigma^{qK}_{i,j})^2
    b_eta_t2 = ((1.0 / eta_safe) - 1.0) ** 2 * eta * sum_sq_sigma_qK
    b_eta = b_eta_t1 + b_eta_t2

    mu_qK_t1 = -(APSI_PLUS_1 / (PSI * q_safe)) + (1.0 / PSI)
    mu_qK_t2 = (1.0 / eta_safe) * sum_sq_sigma_qK
    mu_qK = mu_qK_t1 + mu_qK_t2 + r
    mu_H = jnp.sum(zeta_full * mu_qK, axis=1, keepdims=True)
    sigma_H = jnp.einsum("bk,bkl->bl", zeta_full, sigma_qK)
    diff_vol_full = sigma_qK - sigma_H[:, None, :]
    cross_vol_term_full = jnp.einsum("bl,bil->bi", sigma_H, diff_vol_full)
    # Per Model.tex (eq. 22): minus sum_l sigma_H^l * (sigma_qK^{i,l} - sigma_H^l)
    mu_zeta_rate = mu_qK - mu_H - cross_vol_term_full
    b_zeta = mu_zeta_rate[:, : cfg.N_ZETA] * zeta
    drift_X = jnp.hstack([b_eta, b_zeta])

    # FSDE vol: eta uses (1-eta)*sigma_qK, zeta uses zeta*diff_vol
    vol_eta = (1.0 - eta)[:, :, None] * sigma_qK
    vol_zeta = zeta[:, :, None] * diff_vol_full[:, : cfg.N_ZETA, :]
    vol_X = jnp.hstack([vol_eta, vol_zeta])

    # BSDE vol (eq. 20): Z_{i,j} = q_j * sigma_{q,i,j}
    Z = q[:, None, :] * sigma_q
    return drift_X, vol_X, h, Z


__all__ = ["Config", "compute_dynamics"]


def q_symmetric_analytic(a: float, psi: float, rho: float) -> float:
    """Analytic symmetric-state price (eq. 19): q = (a*psi + 1) / (rho*psi + 1)."""
    return float((a * psi + 1.0) / (rho * psi + 1.0))

__all__.append("q_symmetric_analytic")
