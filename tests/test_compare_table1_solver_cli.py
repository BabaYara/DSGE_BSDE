import os
import sys
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(not Path("scripts/compare_table1_solver.py").exists(), reason="script missing")
def test_compare_table1_solver_runs_smoke():
    # Skip if JAX not available in the environment
    pytest.importorskip("jax")

    cmd = [
        sys.executable,
        str(Path("scripts/compare_table1_solver.py")),
        "--calib",
        str(Path("data/probab01_table1.json")),
    ]
    env = os.environ.copy()
    env["NOTEBOOK_FAST"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    assert proc.returncode == 0

