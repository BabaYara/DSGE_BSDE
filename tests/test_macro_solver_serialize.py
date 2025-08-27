import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from bsde_dsgE.models.macro_solver import Config, MacroFinanceNet, evaluate_symmetric


def test_macro_solver_serialize_roundtrip(tmp_path: Path):
    cfg = Config(J=5)
    key = jax.random.PRNGKey(0)
    model = MacroFinanceNet(cfg, key)
    # Save
    p = tmp_path / "macro.eqx"
    eqx.tree_serialise_leaves(str(p), model)
    assert p.exists()
    # Load
    model2 = MacroFinanceNet(cfg, jax.random.PRNGKey(1))
    model2 = eqx.tree_deserialise_leaves(str(p), model2)

    # Compare predictions on symmetric states
    etas = (0.3, 0.4, 0.5)
    q1, s1, r1 = evaluate_symmetric(cfg, model, etas)
    q2, s2, r2 = evaluate_symmetric(cfg, model2, etas)
    assert np.allclose(np.array(q1), np.array(q2))
    assert np.allclose(np.array(r1), np.array(r2))

