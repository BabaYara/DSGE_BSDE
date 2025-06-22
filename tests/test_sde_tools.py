import jax.numpy as jnp
import pytest

from bsde_dsgE.utils.sde_tools import sobol_brownian


def test_sobol_brownian_shape():
    out = sobol_brownian(dim=1, steps=3, batch=8, dt=0.1)
    assert out.shape == (8, 3, 1)


def test_sobol_brownian_stats():
    dt = 0.1
    out = sobol_brownian(dim=1, steps=50, batch=1000, dt=dt)
    mean = jnp.mean(out)
    var = jnp.var(out)
    assert jnp.allclose(mean, 0.0, atol=5e-2)
    assert jnp.allclose(var, dt, atol=5e-2)


def test_sobol_brownian_batch_must_be_even():
    with pytest.raises(ValueError, match="batch must be even"):
        sobol_brownian(dim=1, steps=2, batch=3, dt=0.1)
