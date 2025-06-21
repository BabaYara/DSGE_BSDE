import jax
import jax.numpy as jnp
from kfac_pinn.training import train_pinn
from kfac_pinn.pde import poisson_residual


def dummy_u(x, params):
    return jnp.zeros_like(x)


def test_train_pinn_runs():
    x0 = jnp.linspace(0, 1, 8)
    params, losses = train_pinn(m=3, residual_fn=poisson_residual, x0=x0, steps=2, lr=1e-3)
    assert len(losses) == 2
