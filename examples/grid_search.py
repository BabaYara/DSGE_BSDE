"""
Grid‑search over risk‑aversion {5,7,9}.  Saves each PDE residual
to `dvc plots`.
"""
import itertools, json, pathlib, jax, jax.numpy as jnp
from bsde_dsgE.core import load_solver
from bsde_dsgE.models import ct_lucas


out = {}
for gamma in [5.0, 7.0, 9.0]:
    model = ct_lucas.scalar_lucas(gamma=gamma)
    solver = load_solver(model, dt=0.1)
    key = jax.random.PRNGKey(1)
    loss = solver(jnp.ones((64,)) * 0.8, key)
    out[gamma] = float(loss)

pathlib.Path("artifacts").mkdir(exist_ok=True)
json.dump(out, open("artifacts/grid.json", "w"))
print("grid‑search results", out)
