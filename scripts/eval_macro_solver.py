"""Load a trained MacroFinanceNet and evaluate symmetric-state q and sigma_q.

Usage:
  python scripts/eval_macro_solver.py --model path/to/checkpoint.eqx --J 5
"""

from __future__ import annotations

import argparse
import numpy as np
import jax

from bsde_dsgE.models.macro_solver import Config as NetCfg, MacroFinanceNet, evaluate_symmetric
import equinox as eqx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--J", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NetCfg(J=args.J)
    # Build a model with same structure then load leaves
    model = MacroFinanceNet(cfg, jax.random.PRNGKey(args.seed))
    model = eqx.tree_deserialise_leaves(args.model, model)
    etas = (0.3, 0.4, 0.5, 0.6, 0.7)
    q, sigma_q, r = evaluate_symmetric(cfg, model, etas)
    print("q@sym:", np.array(q))
    print("r@sym:", np.array(r).ravel())


if __name__ == "__main__":
    main()

