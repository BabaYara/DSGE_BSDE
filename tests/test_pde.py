import jax.numpy as jnp
from bsde_dsgE.kfac.pde import (
    pinn_loss,
    poisson_1d_residual,
    poisson_nd_residual,
)


def test_poisson_zero_residual() -> None:
    def net(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)

    x = jnp.linspace(0.0, 1.0, 4)
    res = poisson_1d_residual(net, x)
    assert jnp.allclose(res, 0)


def test_pinn_loss_zero_solution() -> None:
    def net(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)

    interior = jnp.linspace(0.0, 1.0, 4)
    bc = jnp.array([0.0, 1.0])
    loss = pinn_loss(net, interior, bc)
    assert loss == 0


def test_poisson_nd_zero_residual() -> None:
    def net(x: jnp.ndarray) -> float:
        return 0.0

    x = jnp.array(
        [
            [0.1, 0.2],
            [0.3, 0.7],
            [0.5, 0.5],
        ]
    )
    res = poisson_nd_residual(net, x)
    assert jnp.allclose(res, 0)
