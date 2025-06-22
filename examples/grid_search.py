"""Minimal grid‑search over the Lucas economy parameters.

This script sweeps the risk‑aversion coefficient ``gamma`` across a list of
values and records the terminal PDE residual for each model.  The results are
saved as a JSON file and can be visualised with ``dvc plots`` or other tooling.

Running the example prints a dictionary ``{gamma: residual}`` and writes the
same mapping to ``<output>/grid.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import jax
import jax.numpy as jnp
from bsde_dsgE.core import load_solver
from bsde_dsgE.models import ct_lucas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lucas model grid search")
    parser.add_argument(
        "--gammas",
        type=str,
        default="5,7,9",
        help="Comma separated list of risk aversion values",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("artifacts"),
        help="Directory to store the grid-search results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gammas = [float(g) for g in args.gammas.split(",")]
    if os.environ.get("NOTEBOOK_FAST"):
        gammas = gammas[:1]
    out = {}

    for gamma in gammas:
        model = ct_lucas.scalar_lucas(gamma=gamma)
        solver = load_solver(model, dt=0.1)
        key = jax.random.PRNGKey(1)
        loss = solver(jnp.ones((64,)) * 0.8, key)
        out[gamma] = float(loss)

    args.output.mkdir(exist_ok=True)
    json.dump(out, open(args.output / "grid.json", "w"))
    print("grid-search results", out)


if __name__ == "__main__":
    main()
