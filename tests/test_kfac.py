from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from bsde_dsgE.kfac import KFACPINNSolver, init_state, kfac_update


def dummy_loss(net: Any, x: jnp.ndarray) -> jnp.ndarray:
    y = net(x)
    return jnp.mean(y**2)


def test_kfac_solver_reduces_loss() -> None:
    key = jax.random.PRNGKey(0)
    net = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, key=key)
    solver = KFACPINNSolver(net=net, loss_fn=dummy_loss, lr=1e-2, num_steps=15)
    x = jnp.zeros((1, 1))
    losses = solver.run(x, key)
    assert losses.shape == (15,)
    assert losses[-1] < losses[0]


def test_kfac_update_jit_reduces_loss() -> None:
    params = {"w": jnp.array([1.0])}

    def loss_fn(p: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return jnp.sum(jnp.square(p["w"]))

    grads = jax.grad(loss_fn)(params)
    state = init_state(params)
    new_params, _ = kfac_update(params, grads, state, lr=0.1)
    assert loss_fn(new_params) < loss_fn(params)
