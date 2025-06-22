import jax
import jax.numpy as jnp
import equinox as eqx

from bsde_dsgE.kfac import (
    KFACPINNSolver as RefSolver,
    kfac_update as ref_update,
    init_state as ref_init,
)
from kfac_pinn import KFACPINNSolver, kfac_update, init_state


def dummy_loss(net: eqx.Module, x: jnp.ndarray) -> jnp.ndarray:
    y = net(x)
    return jnp.mean(y**2)


def test_kfac_update_equivalence():
    key = jax.random.PRNGKey(0)
    params = {"w": jax.random.normal(key, (2, 2))}
    grads = {"w": jnp.full((2, 2), 0.1)}

    state = init_state(params)
    ref_state = ref_init(params)

    new_params, new_state = kfac_update(params, grads, state, lr=1e-2)
    ref_params, ref_state = ref_update(params, grads, ref_state, lr=1e-2)

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: jnp.allclose(a, b), new_params, ref_params)
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: jnp.allclose(a, b), new_state, ref_state)
    )


def test_kfac_solver_equivalence():
    key = jax.random.PRNGKey(0)
    net_pkg = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)
    key = jax.random.PRNGKey(0)
    net_ref = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)

    solver_pkg = KFACPINNSolver(net=net_pkg, loss_fn=dummy_loss, lr=1e-2, num_steps=5)
    solver_ref = RefSolver(net=net_ref, loss_fn=dummy_loss, lr=1e-2, num_steps=5)

    x = jnp.zeros((1, 1))
    losses_pkg = solver_pkg.run(x, key)
    losses_ref = solver_ref.run(x, key)

    assert jnp.allclose(losses_pkg, losses_ref)
    assert eqx.tree_equal(solver_pkg.net, solver_ref.net)


def test_kfac_solver_reduces_loss_pkg():
    key = jax.random.PRNGKey(0)
    net = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)
    solver = KFACPINNSolver(net=net, loss_fn=dummy_loss, lr=1e-2, num_steps=15)
    x = jnp.zeros((1, 1))
    losses = solver.run(x, key)
    assert losses.shape == (15,)
    assert losses[-1] < losses[0]

