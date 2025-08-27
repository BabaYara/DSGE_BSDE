import jax
import jax.numpy as jnp

from bsde_dsgE.models.probab01_equations import Config, compute_dynamics


def test_compute_dynamics_finite_outputs():
    cfg = Config(J=4)
    key = jax.random.PRNGKey(0)
    B = 5
    Omega = jax.random.uniform(key, (B, cfg.N_STATE), minval=0.2, maxval=0.8)
    q = jax.random.uniform(key, (B, cfg.J), minval=0.9, maxval=1.5)
    sigma_q = jax.random.normal(key, (B, cfg.J, cfg.J)) * 1e-3
    r = jax.nn.softplus(jax.random.normal(key, (B, 1)))

    drift_X, vol_X, h, Z = compute_dynamics(cfg, Omega, q, sigma_q, r)

    def finite(x):
        return jnp.isfinite(x).all()

    assert finite(drift_X) and finite(vol_X) and finite(h) and finite(Z)

