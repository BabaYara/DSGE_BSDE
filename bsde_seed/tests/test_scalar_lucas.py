import jax.numpy as jnp
from bsde_seed.bsde_dsgE.models import scalar_lucas


def test_scalar_lucas_gbm_drift_diffusion():
    mu = 0.1
    sigma = 0.3
    drift, diffusion, _, _ = scalar_lucas(mu=mu, sigma=sigma)
    x = jnp.array([1.0, 2.0, 3.0])
    expected_drift = mu * x
    expected_diff = sigma * x
    assert jnp.allclose(drift(x), expected_drift)
    assert jnp.allclose(diffusion(x), expected_diff)
