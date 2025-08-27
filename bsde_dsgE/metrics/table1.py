"""Utilities for computing and comparing Table 1‑style summary statistics.

This module keeps dependencies minimal (NumPy/JAX only) and provides helpers to
compute summary moments from simulated state paths and to compare them against
targets with tolerances.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import jax.numpy as jnp
import numpy as np


def summary_stats(xs: np.ndarray | jnp.ndarray) -> Dict[str, Any]:
    """Compute per-dimension mean/std and full covariance/correlation.

    Parameters
    ----------
    xs : array
        Array of shape ``(T, P, D)`` where T = time steps, P = paths, D = dims.

    Returns
    -------
    dict
        Dictionary with keys: ``mean`` (D,), ``std`` (D,), ``cov`` (D,D),
        ``corr`` (D,D).
    """
    arr = np.asarray(xs)
    assert arr.ndim == 3, "xs must be (time, paths, dim)"
    T, P, D = arr.shape
    flat = arr.reshape(T * P, D)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0, ddof=1)
    cov = np.cov(flat, rowvar=False)
    # Corr from cov and std, guard zeros
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    return {"mean": mean, "std": std, "cov": cov, "corr": corr}


def compare_to_targets(
    stats: Mapping[str, Any],
    targets: Mapping[str, Any],
    tolerances: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Compare selected stats to targets with per-key absolute tolerances.

    Parameters
    ----------
    stats : mapping
        Output from :func:`summary_stats`.
    targets : mapping
        Dictionary of same structure as in calibration file. Supported keys:
        ``mean_state``, ``std_state``, ``cov``, ``corr``.
    tolerances : mapping, optional
        Mapping from key to absolute tolerance (default 1e-2 per key if
        unspecified).

    Returns
    -------
    dict
        Dictionary with per-key max absolute error and pass/fail flag.
    """
    tol_default = 1e-2
    result: Dict[str, Any] = {}
    for key, tgt in targets.items():
        tol = tol_default if tolerances is None else tolerances.get(key, tol_default)
        if key == "mean_state":
            err = np.max(np.abs(np.asarray(stats["mean"]) - np.asarray(tgt)))
        elif key == "std_state":
            err = np.max(np.abs(np.asarray(stats["std"]) - np.asarray(tgt)))
        elif key == "cov":
            err = np.max(np.abs(np.asarray(stats["cov"]) - np.asarray(tgt)))
        elif key == "corr":
            err = np.max(np.abs(np.asarray(stats["corr"]) - np.asarray(tgt)))
        else:
            continue
        result[key] = {"max_abs_err": float(err), "ok": bool(err <= tol)}
    # Global flag
    result["all_ok"] = all(v.get("ok", True) for k, v in result.items() if k != "all_ok")
    return result


__all__ = ["summary_stats", "compare_to_targets"]


def diag_pos_offdiag_neg(mat: np.ndarray | jnp.ndarray, tol: float = 0.0) -> Dict[str, Any]:
    """Check sign pattern: diagonal > tol and off-diagonal < -tol.

    Returns a dict with counts and ok flag.
    """
    arr = np.asarray(mat)
    assert arr.ndim == 2 and arr.shape[0] == arr.shape[1], "mat must be square"
    n = arr.shape[0]
    diag_vals = np.diag(arr)
    off_vals = arr[~np.eye(n, dtype=bool)]
    ok_diag = np.all(diag_vals > tol)
    ok_off = np.all(off_vals < -tol)
    return {
        "diag_positive": int(ok_diag),
        "offdiag_negative": int(ok_off),
        "ok": bool(ok_diag and ok_off),
    }

__all__.append("diag_pos_offdiag_neg")
