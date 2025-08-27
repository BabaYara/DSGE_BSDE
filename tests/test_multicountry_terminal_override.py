import pytest


@pytest.mark.skipif(pytest.importorskip("jax") is None, reason="requires jax")
def test_terminal_override_shapes():
    import jax.numpy as jnp
    from bsde_dsgE.models.multicountry import multicountry_probab01

    def custom_terminal(x):
        x2 = x if x.ndim == 2 else x[:, None]
        return jnp.sum(x2, axis=1)

    dim = 3
    prob = multicountry_probab01(dim=dim, terminal_fn=custom_terminal)
    x = jnp.ones((5, dim))
    yT = prob.terminal(x)
    assert yT.shape == (5,)
    assert jnp.allclose(yT, jnp.full((5,), dim))

