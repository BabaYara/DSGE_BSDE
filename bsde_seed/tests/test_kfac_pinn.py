import jax
import jax.numpy as jnp
import equinox as eqx

from bsde_seed.bsde_dsgE.optim import KFACPINNSolver


def residual(net: eqx.Module, x: jnp.ndarray) -> jnp.ndarray:
    y = net(x)
    return jnp.mean(y ** 2)


def test_kfac_optimizer_reduces_loss():
    key = jax.random.PRNGKey(0)
    net = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)
    solver = KFACPINNSolver(net=net, loss_fn=residual, lr=1e-2, num_steps=50)
    x = jnp.zeros((1, 1))
    losses = solver.run(x, key)
    assert losses[-1] < losses[0]
