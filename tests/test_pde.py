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


def test_poisson_dirichlet_bc() -> None:
    net = lambda x: x**2
    bc = jnp.array([0.0, 1.0])
    res = poisson_1d_residual(net, bc, dirichlet_bc=lambda z: z**2)
    assert jnp.allclose(res, 0)


def test_poisson_neumann_bc() -> None:
    net = lambda x: x**2
    bc = jnp.array([0.0, 1.0])
    res = poisson_1d_residual(net, bc, neumann_bc=lambda z: 2 * z)
    assert jnp.allclose(res, 0)


def test_pinn_loss_custom_bcs() -> None:
    net = lambda x: jnp.zeros_like(x)
    interior = jnp.linspace(0.0, 1.0, 4)
    bc = jnp.array([0.0, 1.0])
    loss_dir = pinn_loss(net, interior, bc, dirichlet_bc=lambda x: 0.0)
    loss_neu = pinn_loss(net, interior, bc, neumann_bc=lambda x: 0.0)
    assert loss_dir == 0
    assert loss_neu == 0


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
