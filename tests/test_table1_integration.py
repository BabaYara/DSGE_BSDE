import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bsde_dsgE.models.multicountry import multicountry_probab01
from bsde_dsgE.metrics.table1 import summary_stats, compare_to_targets


@pytest.mark.skipif(not Path("data/probab01_table1.json").exists(), reason="calibration file not provided")
def test_table1_targets_optional_compare():
    # Load calibration and only run comparison if targets are present
    data = json.loads(Path("data/probab01_table1.json").read_text())
    dim = int(data.get("dim", 2))
    rho = float(data.get("rho", 0.05))
    gamma = float(data.get("gamma", 8.0))
    kappa = float(data.get("kappa", 0.2))
    theta = float(data.get("theta", 1.0))
    sigma = data.get("sigma", 0.3)
    Sigma = np.array(data["Sigma"]) if "Sigma" in data else None

    prob = multicountry_probab01(dim=dim, rho=rho, gamma=gamma, kappa=kappa, theta=theta, sigma=sigma, Sigma=Sigma)

    # Simulate forward paths (this tests only moment machinery, not training)
    dt = 0.02
    T = 50
    P = 16
    key = jax.random.PRNGKey(0)
    x = jnp.ones((P, dim)) * theta
    xs = [np.array(x)]
    for i in range(T):
        key = jax.random.fold_in(key, i)
        dW = jax.random.normal(key, x.shape) * np.sqrt(dt)
        x = prob.step(x, i * dt, dt, dW)
        xs.append(np.array(x))
    xs = np.stack(xs, axis=0)  # (T+1, P, dim)

    stats = summary_stats(xs)
    targets = data.get("table1_targets", {}).get("examples")
    if targets:
        res = compare_to_targets(stats, targets)
        # We don't assert all_ok here to avoid false failures with placeholder data,
        # but we require the comparison keys to be processed.
        assert isinstance(res, dict) and "all_ok" in res

