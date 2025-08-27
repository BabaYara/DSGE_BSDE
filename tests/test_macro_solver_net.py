import jax
import jax.numpy as jnp
import numpy as np

from bsde_dsgE.models.macro_solver import Config, MacroFinanceNet, evaluate_symmetric


def test_macro_solver_shapes_and_signs():
    cfg = Config(J=5)
    key = jax.random.PRNGKey(0)
    net = MacroFinanceNet(cfg, key)

    # Batch of random states
    B = 7
    Omega = jax.random.uniform(key, (B, cfg.N_STATE))
    q, sigma_q, r = net(Omega)

    assert q.shape == (B, cfg.J)
    assert sigma_q.shape == (B, cfg.J, cfg.J)
    assert r.shape == (B, 1)

    # Sign constraints by construction
    sigma = np.array(sigma_q)
    for b in range(B):
        diag = np.diag(sigma[b])
        off = sigma[b][~np.eye(cfg.J, dtype=bool)]
        assert np.all(diag > 0), "Diagonal must be positive"
        assert np.all(off < 0), "Off-diagonal must be negative"


def test_evaluate_symmetric_states_runs():
    cfg = Config(J=5)
    net = MacroFinanceNet(cfg, jax.random.PRNGKey(1))
    q, sigma_q, r = evaluate_symmetric(cfg, net)
    assert q.shape[0] == 5  # 5 eta grid points
    assert sigma_q.shape[1:] == (cfg.J, cfg.J)
    assert r.shape == (5, 1)

