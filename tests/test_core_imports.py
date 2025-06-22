import jax
import jax.numpy as jnp

from bsde_dsgE.core import load_solver
from bsde_dsgE.models.ct_lucas import scalar_lucas


def test_core_load_solver_import(batch_x):
    solver = load_solver(scalar_lucas())
    loss = solver(batch_x, jax.random.PRNGKey(0))
    assert jnp.isfinite(loss)

