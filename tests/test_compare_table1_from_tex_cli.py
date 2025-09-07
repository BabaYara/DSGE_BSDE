import sys
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(not Path("scripts/compare_table1_solver.py").exists(), reason="script missing")
def test_compare_table1_solver_from_tex_runs_smoke():
    pytest.importorskip("jax")
    cmd = [
        sys.executable,
        str(Path("scripts/compare_table1_solver.py")),
        "--calib",
        str(Path("data/probab01_table1.json")),
        "--from-tex",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0

