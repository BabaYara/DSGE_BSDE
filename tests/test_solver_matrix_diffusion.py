import jax.numpy as jnp
import jax

from bsde_dsgE.core.solver import BSDEProblem


def test_step_matrix_diffusion():
    # x in R^2, sigma is a per-sample 2x2 matrix
    def drift(x):
        return jnp.zeros_like(x)

    def diff(x):
        batch, dim = x.shape
        # Diagonal matrix with entries (2, 3)
        diag = jnp.array([2.0, 3.0])
        I = jnp.eye(dim)
        return jnp.broadcast_to(I * diag[None, :], (batch, dim, dim))

    def generator(x, y, z):
        return 0.0 * y

    def terminal(x):
        return jnp.zeros((x.shape[0],))

    prob = BSDEProblem(drift, diff, generator, terminal, 0.0, 1.0)
    x = jnp.zeros((1, 2))
    dW = jnp.array([[1.0, 2.0]])
    x1 = prob.step(x, 0.0, 0.1, dW)
    # Expected increment: (2*1, 3*2) = (2, 6)
    assert jnp.allclose(x1, jnp.array([[2.0, 6.0]]))

