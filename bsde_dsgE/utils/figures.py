"""Plot‑ready utilities for notebooks: rolling correlations, mean±2SE,
impulse responses, and path simulation helpers.

These functions are plotting‑library‑agnostic and return NumPy/JAX arrays
so notebooks can quickly build figures and animations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # Optional JAX dependency
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - environment without JAX
    jax = None  # type: ignore
    jnp = None  # type: ignore

from .sde_tools import sobol_brownian


def mean_and_2se(xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension mean and 2×standard error using a naive IID assumption.

    Parameters
    - xs: array shaped (T, P, D) or (N, D)

    Returns
    - mean: (D,)
    - two_se: (D,)
    """
    arr = np.asarray(xs)
    if arr.ndim == 3:
        N = arr.shape[0] * arr.shape[1]
        flat = arr.reshape(N, arr.shape[2])
    elif arr.ndim == 2:
        N = arr.shape[0]
        flat = arr
    else:
        raise AssertionError("xs must be (T,P,D) or (N,D)")
    mean = flat.mean(axis=0)
    se = flat.std(axis=0, ddof=1) / np.sqrt(max(N, 1))
    return mean, 2.0 * se


def rolling_corr(xs: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling correlation matrices over time.

    Parameters
    - xs: array shaped (T, P, D)
    - window: window length in time steps

    Returns
    - corrs: array shaped (T-window+1, D, D)
    """
    arr = np.asarray(xs)
    T, P, D = arr.shape
    assert 1 <= window <= T, "invalid window"
    frames = T - window + 1
    out = np.zeros((frames, D, D), dtype=float)
    for i in range(frames):
        chunk = arr[i : i + window]  # (window, P, D)
        flat = chunk.reshape(window * P, D)
        std = flat.std(axis=0, ddof=1)
        cov = np.cov(flat, rowvar=False)
        denom = np.outer(std, std)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = np.where(denom > 0, cov / denom, 0.0)
    return out


@dataclass
class SimResult:
    xs: np.ndarray  # (T+1, P, D)
    dWs: Optional[np.ndarray] = None  # (T, P, D)


def simulate_paths(
    problem,  # BSDEProblem‑like with .step(x,t,dt,dW)
    x0,
    *,
    steps: int,
    dt: float,
    key=None,
    use_sobol: bool = False,
) -> SimResult:
    """Simulate forward paths for a vector state SDE using problem.step.

    Returns xs shaped (T+1, P, D). Requires JAX when use_sobol=False.
    """
    if jax is None:
        raise RuntimeError("simulate_paths requires JAX runtime installed")
    x = jnp.asarray(x0)
    batch, dim = x.shape
    if use_sobol:
        dWs = sobol_brownian(dim=dim, steps=steps, batch=int(batch), dt=dt)
        dWs = np.array(dWs)
    else:
        if key is None:
            key = jax.random.PRNGKey(0)
        dWs = np.array(jax.random.normal(key, (steps, batch, dim)) * np.sqrt(dt))
    xs = [np.array(x)]
    for i in range(steps):
        x = problem.step(x, i * dt, dt, jnp.asarray(dWs[i]))
        xs.append(np.array(x))
    return SimResult(xs=np.stack(xs, axis=0), dWs=dWs)


def impulse_response(
    problem,
    x0,
    *,
    steps: int,
    dt: float,
    shock_dim: int,
    shock_size: float,
    key=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute impulse responses by shocking one state at t=0.

    Returns (mean_base, mean_shocked, irf) with shapes (T+1, D).
    """
    if jax is None:
        raise RuntimeError("impulse_response requires JAX runtime installed")
    x0 = jnp.asarray(x0)
    batch, dim = x0.shape
    key = jax.random.PRNGKey(0) if key is None else key
    dWs = np.array(jax.random.normal(key, (steps, batch, dim)) * np.sqrt(dt))

    # Base
    xb = x0
    xs_b = [np.array(xb)]
    for i in range(steps):
        xb = problem.step(xb, i * dt, dt, jnp.asarray(dWs[i]))
        xs_b.append(np.array(xb))
    Xb = np.stack(xs_b, axis=0)  # (T+1,P,D)

    # Shocked initial state
    xs = x0.at[:, shock_dim].add(shock_size)
    xs_s = [np.array(xs)]
    for i in range(steps):
        xs = problem.step(xs, i * dt, dt, jnp.asarray(dWs[i]))
        xs_s.append(np.array(xs))
    Xs = np.stack(xs_s, axis=0)

    mb = Xb.mean(axis=1)
    ms = Xs.mean(axis=1)
    irf = ms - mb
    return mb, ms, irf


__all__ = [
    "mean_and_2se",
    "rolling_corr",
    "simulate_paths",
    "impulse_response",
    "SimResult",
]

