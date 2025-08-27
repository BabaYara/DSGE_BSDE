"""Quick check: simulate and compare against Table 1 targets.

Usage:
  python scripts/check_table1.py --calib data/probab01_table1.json --steps 100 --paths 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import jax
import jax.numpy as jnp
import numpy as np

from bsde_dsgE.models.multicountry import multicountry_probab01
from bsde_dsgE.metrics.table1 import summary_stats, compare_to_targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check Table 1 targets against simulation moments")
    p.add_argument("--calib", type=Path, default=Path("data/probab01_table1.json"))
    # Defaults adapt to NOTEBOOK_FAST if provided
    fast = bool(os.environ.get("NOTEBOOK_FAST", ""))
    p.add_argument("--steps", type=int, default=(30 if fast else 100))
    p.add_argument("--paths", type=int, default=(16 if fast else 64))
    p.add_argument("--dt", type=float, default=(0.05 if fast else 0.02))
    p.add_argument("--sobol", action="store_true", help="Use Sobol quasi-MC Brownian increments (antithetic)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.calib.read_text())
    dim = int(data["dim"]) ; rho = float(data["rho"]) ; gamma = float(data["gamma"]) ; kappa = float(data["kappa"]) ; theta = float(data["theta"]) ; sigma = data.get("sigma")
    Sigma = np.array(data["Sigma"]) if "Sigma" in data else None

    prob = multicountry_probab01(dim=dim, rho=rho, gamma=gamma, kappa=kappa, theta=theta, sigma=sigma, Sigma=Sigma)
    seed = 0
    key = jax.random.PRNGKey(seed)
    x = jnp.ones((args.paths, dim)) * theta
    xs = [np.array(x)]
    if args.sobol:
        from bsde_dsgE.utils.sde_tools import sobol_brownian  # lazy import; requires SciPy
        # Use Sobol Brownian paths with antithetic pairing
        dWs = np.array(sobol_brownian(dim=dim, steps=args.steps, batch=int(args.paths), dt=args.dt))  # (P, steps, D)
        dWs = np.transpose(dWs, (1, 0, 2))  # (steps, P, D)
    else:
        dWs = np.array(jax.random.normal(key, (args.steps, args.paths, dim)) * np.sqrt(args.dt))
    for i in range(args.steps):
        x = prob.step(x, i * args.dt, args.dt, jnp.asarray(dWs[i]))
        xs.append(np.array(x))
    xs = np.stack(xs, axis=0)
    stats = summary_stats(xs)
    targets = data.get("table1_targets", {}).get("examples")
    if not targets:
        print("No targets found in calibration; computed stats:")
        print({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in stats.items() if k in ("mean","std")})
        return
    res = compare_to_targets(stats, targets)
    print({
        "FAST": bool(os.environ.get("NOTEBOOK_FAST", "")),
        "device": jax.default_backend(),
        "seed": seed,
        "steps": args.steps,
        "paths": args.paths,
        "dt": args.dt,
        "sobol": bool(args.sobol),
        "comparison": res,
    })
    # Optional strict mode for CI
    if os.environ.get("STRICT_TABLE1", ""):
        assert res.get("all_ok", False), f"Table 1 moment comparison failed: {res}"


if __name__ == "__main__":
    main()
