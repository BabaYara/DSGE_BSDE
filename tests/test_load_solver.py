import jax
import jax.numpy as jnp

from bsde_dsgE import load_solver
from bsde_dsgE.models.ct_lucas import scalar_lucas


def test_load_solver(batch_x):
    model = scalar_lucas()
    solver = load_solver(model)
    loss = solver(batch_x, jax.random.PRNGKey(0))
    assert jnp.isfinite(loss)
