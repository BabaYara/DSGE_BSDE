import jax
import jax.numpy as jnp

from bsde_dsgE.models.multicountry import multicountry_probab01
from bsde_dsgE.core import load_solver_nd


def test_solver_nd_with_matrix_sigma_runs():
    dim = 2
    Sigma = jnp.array([[0.3, 0.1], [0.1, 0.25]])
    prob = multicountry_probab01(dim=dim, Sigma=Sigma)
    solver = load_solver_nd(prob, dim=dim, dt=0.1, depth=2, width=8)
    x0 = jnp.zeros((4, dim))
    loss = solver(x0, jax.random.PRNGKey(0))
    assert jnp.isfinite(loss)

