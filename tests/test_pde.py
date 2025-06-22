import jax.numpy as jnp
from bsde_dsgE.kfac.pde import poisson_1d_residual, pinn_loss


def test_poisson_zero_residual():
    net = lambda x: jnp.zeros_like(x)
    x = jnp.linspace(0.0, 1.0, 4)
    res = poisson_1d_residual(net, x)
    assert jnp.allclose(res, 0)


def test_pinn_loss_zero_solution():
    net = lambda x: jnp.zeros_like(x)
    interior = jnp.linspace(0.0, 1.0, 4)
    bc = jnp.array([0.0, 1.0])
    loss = pinn_loss(net, interior, bc)
    assert loss == 0
