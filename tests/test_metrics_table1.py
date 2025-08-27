import numpy as np

from bsde_dsgE.metrics.table1 import summary_stats, compare_to_targets


def test_summary_stats_shapes():
    xs = np.zeros((5, 4, 3))
    s = summary_stats(xs)
    assert s["mean"].shape == (3,)
    assert s["std"].shape == (3,)
    assert s["cov"].shape == (3, 3)
    assert s["corr"].shape == (3, 3)


def test_compare_to_targets_pass_fail():
    # Construct xs with known mean std (zeros)
    xs = np.zeros((10, 8, 2))
    s = summary_stats(xs)
    # Targets exact zeros should pass
    res = compare_to_targets(s, {"mean_state": [0.0, 0.0], "std_state": [0.0, 0.0]}, {"mean_state": 1e-8, "std_state": 1e-8})
    assert res["all_ok"]
    # Targets off by 1 should fail
    res2 = compare_to_targets(s, {"mean_state": [1.0, 0.0]}, {"mean_state": 1e-3})
    assert not res2["all_ok"]

