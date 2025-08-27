import numpy as np

from bsde_dsgE.utils.figures import mean_and_2se, rolling_corr


def test_mean_and_2se_shapes():
    T, P, D = 10, 8, 3
    xs = np.random.randn(T, P, D)
    m, e = mean_and_2se(xs)
    assert m.shape == (D,)
    assert e.shape == (D,)


def test_rolling_corr_shapes():
    T, P, D = 20, 6, 2
    xs = np.random.randn(T, P, D)
    W = 5
    rc = rolling_corr(xs, window=W)
    assert rc.shape == (T - W + 1, D, D)

