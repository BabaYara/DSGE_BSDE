import numpy as np

from bsde_dsgE.utils.figures import qq_points, lag_autocorr


def test_qq_points_monotone_and_shapes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    qt, qe = qq_points(x)
    assert qt.shape == qe.shape
    # Sorted by construction
    assert np.all(np.diff(qe) >= 0)


def test_lag_autocorr_white_noise_near_zero():
    rng = np.random.default_rng(1)
    T, P, D = 50, 20, 3
    xs = rng.standard_normal((T, P, D))
    ac = lag_autocorr(xs, lag=1)
    assert ac.shape == (D,)
    # For white noise, lag-1 autocorr should be small in magnitude on average
    assert float(np.mean(np.abs(ac))) < 0.3

