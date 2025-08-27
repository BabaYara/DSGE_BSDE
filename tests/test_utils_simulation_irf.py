import pytest


@pytest.mark.skipif(pytest.importorskip("jax") is None, reason="requires jax")
def test_simulate_and_irf_shapes():
    import jax
    import jax.numpy as jnp
    from bsde_dsgE.models.multicountry import multicountry_probab01
    from bsde_dsgE.utils.figures import simulate_paths, impulse_response

    dim = 2
    prob = multicountry_probab01(dim=dim)
    x0 = jnp.zeros((4, dim))
    sim = simulate_paths(prob, x0, steps=5, dt=0.1, key=jax.random.PRNGKey(0))
    assert sim.xs.shape == (6, 4, dim)
    assert sim.dWs is not None and sim.dWs.shape == (5, 4, dim)

    mb, ms, irf = impulse_response(prob, x0, steps=5, dt=0.1, shock_dim=0, shock_size=0.1, key=jax.random.PRNGKey(1))
    assert mb.shape == (6, dim)
    assert ms.shape == (6, dim)
    assert irf.shape == (6, dim)

