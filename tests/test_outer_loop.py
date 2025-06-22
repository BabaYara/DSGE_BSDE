import jax.numpy as jnp
from bsde_dsgE.core.outer_loop import pareto_bisection


def test_pareto_bisection():
    def f(x: float) -> float:
        return x**2 - 2.0

    root = pareto_bisection(f, 0.0, 2.0, tol=1e-6)
    assert jnp.allclose(root, jnp.sqrt(2.0), atol=1e-5)
