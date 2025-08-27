import os
import sys
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(not Path("scripts/check_table1.py").exists(), reason="script missing")
def test_check_table1_script_runs_smoke(tmp_path: Path):
    jax = pytest.importorskip("jax")  # skip if JAX not available in env
    # Use small steps/paths to keep runtime minimal
    cmd = [
        sys.executable,
        str(Path("scripts/check_table1.py")),
        "--calib",
        str(Path("data/probab01_table1.json")),
        "--steps",
        "5",
        "--paths",
        "4",
        "--dt",
        "0.05",
    ]
    env = os.environ.copy()
    env["NOTEBOOK_FAST"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    assert proc.returncode == 0

