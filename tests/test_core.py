# tests/test_core.py
from bsde_dsgE.core.solver import Solver
from bsde_dsgE.models.ct_lucas import scalar_lucas
import jax, jax.numpy as jnp

def test_forward_pass(batch_x):
    prob = scalar_lucas()
    dummy_net = lambda t,x: (jnp.ones_like(x), jnp.zeros_like(x))
    solver = Solver(dummy_net, prob, dt=0.05)
    out = solver(batch_x, jax.random.PRNGKey(0))
    assert jnp.isfinite(out)
