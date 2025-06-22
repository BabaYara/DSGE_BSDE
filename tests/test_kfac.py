import jax
import jax.numpy as jnp
import equinox as eqx

from bsde_dsgE.kfac import KFACPINNSolver


def dummy_loss(net: eqx.Module, x: jnp.ndarray) -> jnp.ndarray:
    y = net(x)
    return jnp.mean(y**2)


def test_kfac_solver_reduces_loss():
    key = jax.random.PRNGKey(0)
    net = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)
    solver = KFACPINNSolver(net=net, loss_fn=dummy_loss, lr=1e-2, num_steps=15)
    x = jnp.zeros((1, 1))
    losses = solver.run(x, key)
    assert losses.shape == (15,)
    assert losses[-1] < losses[0]
