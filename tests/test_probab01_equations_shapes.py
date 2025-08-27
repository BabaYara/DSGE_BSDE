import jax
import jax.numpy as jnp

from bsde_dsgE.models.probab01_equations import Config, compute_dynamics
from bsde_dsgE.models.probab01_equations import q_symmetric_analytic


def test_compute_dynamics_shapes():
    cfg = Config(J=5)
    B = 8
    # Random state and controls
    key = jax.random.PRNGKey(0)
    Omega = jax.random.uniform(key, (B, cfg.N_STATE))
    q = jax.random.uniform(key, (B, cfg.J)) + 1e-3
    sigma_q = jax.random.normal(key, (B, cfg.J, cfg.J)) * 1e-3
    r = jax.nn.softplus(jax.random.normal(key, (B, 1)))

    drift_X, vol_X, h, Z = compute_dynamics(cfg, Omega, q, sigma_q, r)
    assert drift_X.shape == (B, cfg.N_STATE)
    assert vol_X.shape == (B, cfg.N_STATE, cfg.J)
    assert h.shape == (B, cfg.J)
    assert Z.shape == (B, cfg.J, cfg.J)


def test_z_definition_matches_eq20():
    cfg = Config(J=3)
    B = 2
    # Simple q and sigma_q so Z_{i,j} = q_j * sigma_{q,i,j}
    q = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    sigma_q = jnp.zeros((B, cfg.J, cfg.J)).at[:, 0, 1].set(0.5).at[:, 2, 0].set(-1.0)
    r = jnp.zeros((B, 1))
    Omega = jnp.zeros((B, cfg.N_STATE))
    _, _, _, Z = compute_dynamics(cfg, Omega, q, sigma_q, r)
    # Check two entries explicitly
    # i=0, j=1 => Z[0,0,1] == q[0,1]*sigma_q[0,0,1] = 2.0*0.5
    assert jnp.allclose(Z[0, 0, 1], 1.0)
    # i=2, j=0 => Z[1,2,0] == q[1,0]*sigma_q[1,2,0] = 4.0*(-1.0)
    assert jnp.allclose(Z[1, 2, 0], -4.0)


def test_q_symmetric_analytic_eq19():
    # Example params from data JSON: a=0.1, psi=5, rho=0.03
    q = q_symmetric_analytic(a=0.1, psi=5.0, rho=0.03)
    # (0.1*5 + 1) / (0.03*5 + 1) = 1.5 / 1.15 ≈ 1.304347826
    assert abs(q - 1.304347826) < 1e-9
