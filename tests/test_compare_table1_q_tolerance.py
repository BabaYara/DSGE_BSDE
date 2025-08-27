import os
import sys
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(not Path("scripts/compare_table1_solver.py").exists(), reason="script missing")
def test_compare_table1_solver_tolerance_gate_allows_large_tol():
    # Skip if JAX not available in the environment
    pytest.importorskip("jax")

    cmd = [
        sys.executable,
        str(Path("scripts/compare_table1_solver.py")),
        "--calib",
        str(Path("data/probab01_table1.json")),
    ]
    env = os.environ.copy()
    env["STRICT_TABLE1"] = "1"
    env["TABLE1_Q_TOL"] = "1e3"  # generous tolerance for smoke
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # Should not assert with large tolerance
    assert proc.returncode == 0

