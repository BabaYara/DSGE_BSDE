import numpy as np

from bsde_dsgE.metrics.table1 import summary_stats, compare_to_targets


def test_compare_to_targets_cov_corr():
    # Construct simple 2D data with known cov/corr
    rng = np.random.default_rng(0)
    x1 = rng.normal(0, 1.0, size=1000)
    x2 = 0.5 * x1 + rng.normal(0, 0.8660254, size=1000)  # approx corr 0.5
    xs = np.stack([x1, x2], axis=1)
    xs = xs.reshape(10, 100, 2)  # (T=10, P=100, D=2)

    s = summary_stats(xs)
    targets = {
        "cov": [[1.0, 0.5], [0.5, 1.0]],
        "corr": [[1.0, 0.5], [0.5, 1.0]],
    }
    res = compare_to_targets(s, targets, {"cov": 0.2, "corr": 0.2})
    assert res["all_ok"], res

