import jax
import jax.numpy as jnp
import equinox as eqx

from bsde_seed.bsde_dsgE.optim import KFACPINNSolver


def pde_residual(net: eqx.Module, x: jnp.ndarray) -> jnp.ndarray:
    """Residual for y'(x) = 1 with unknown constant."""
    # net maps x -> y(x)
    def single_output(x_scalar):
        return net(jnp.array([x_scalar]))[0]

    dydx = jax.vmap(jax.grad(single_output))(x[:, 0])
    return jnp.mean((dydx - 1.0) ** 2)


def test_kfac_optimizer_toy_pde():
    key = jax.random.PRNGKey(0)
    net = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)
    solver = KFACPINNSolver(net=net, loss_fn=pde_residual, lr=1e-2, num_steps=50)
    x = jnp.linspace(0.0, 1.0, num=5).reshape(-1, 1)
    losses = solver.run(x, key)
    assert losses[-1] < losses[0]
