import jax.numpy as jnp
from kfac_pinn.optimizer import KFAC, kfac_update, init_kfac


def test_kfac_update_simple():
    params = jnp.array([1.0, -1.0])
    grads = jnp.array([0.1, -0.2])
    state = init_kfac(params)
    new_params, new_state = kfac_update(grads, params, state)
    assert new_state.step == 1
    assert jnp.all(jnp.isfinite(new_params))
