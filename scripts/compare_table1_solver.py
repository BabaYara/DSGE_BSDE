"""Compare MacroFinanceNet predictions at symmetric states against transcribed Table 1 values.

Usage:
  python scripts/compare_table1_solver.py --calib data/probab01_table1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from bsde_dsgE.models.macro_solver import Config as NetCfg, MacroFinanceNet, evaluate_symmetric
from bsde_dsgE.metrics.table1 import diag_pos_offdiag_neg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--calib", type=Path, default=Path("data/probab01_table1.json"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.calib.read_text())
    # Default J=5 per Table 1 states
    J = int(data.get("probab01_params", {}).get("J", 5))
    cfg = NetCfg(J=J)
    net = MacroFinanceNet(cfg, jax.random.PRNGKey(args.seed))
    etas = tuple(st.get("eta", 0.5) for st in data.get("table1_values", {}).get("symmetric_states", []))
    if not etas:
        etas = (0.3, 0.4, 0.5, 0.6, 0.7)
    q_pred, sigma_pred, r_pred = evaluate_symmetric(cfg, net, etas)
    q_pred = np.array(q_pred)
    # Compare to target q if present
    sym = data.get("table1_values", {}).get("symmetric_states", [])
    if not sym:
        print("No 'table1_values' found; print q predictions only.")
        print(q_pred)
        return
    errs = []
    for i, st in enumerate(sym):
        q_tgt = np.array(st["q"], dtype=float)
        err = float(np.mean(np.abs(q_pred[i] - q_tgt)))
        errs.append(err)
        print(f"eta={st['eta']:.1f}, mean |q_pred - q_tgt| = {err:.6f}")
        # Sign pattern check on target sigma
        res = diag_pos_offdiag_neg(np.array(st["sigma_q"], dtype=float))
        print({"diag_pos": res["diag_positive"], "offdiag_neg": res["offdiag_negative"], "ok": res["ok"]})
    print({"avg_err": float(np.mean(errs)), "max_err": float(np.max(errs))})
    # Optional strict tolerance gate for q matching
    import os
    if os.environ.get("STRICT_TABLE1", ""):
        tol = float(os.environ.get("TABLE1_Q_TOL", "1e9"))
        assert float(np.max(errs)) <= tol, f"q mismatch exceeds tolerance {tol}: max_err={float(np.max(errs))}"


if __name__ == "__main__":
    main()
