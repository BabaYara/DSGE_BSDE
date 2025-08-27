import jax
import jax.numpy as jnp

from bsde_dsgE.models.multicountry import multicountry_probab01
from bsde_dsgE.core import load_solver_nd


def test_multicountry_problem_shapes():
    dim = 2
    prob = multicountry_probab01(dim=dim)
    # sample forward step
    x = jnp.zeros((4, dim))
    dW = jax.random.normal(jax.random.PRNGKey(0), (4, dim)) * 0.1
    x1 = prob.step(x, 0.0, 0.1, dW)
    assert x1.shape == x.shape


def test_multicountry_matrix_sigma():
    dim = 3
    Sigma = jnp.eye(dim)
    prob = multicountry_probab01(dim=dim, Sigma=Sigma)
    x = jnp.zeros((2, dim))
    dW = jnp.ones_like(x)
    x1 = prob.step(x, 0.0, 0.1, dW)
    assert x1.shape == x.shape


def test_solver_nd_runs():
    dim = 2
    prob = multicountry_probab01(dim=dim)
    solver = load_solver_nd(prob, dim=dim, dt=0.1, depth=2, width=16)
    x0 = jnp.zeros((8, dim))
    loss = solver(x0, jax.random.PRNGKey(1))
    assert jnp.isfinite(loss)
