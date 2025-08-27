import sys
import subprocess
from pathlib import Path
import json

import pytest


@pytest.mark.skipif(not Path("scripts/update_table1_from_tex.py").exists(), reason="script missing")
def test_update_table1_from_tex_dry_run_matches_extractor():
    # Dry-run prints updated JSON
    cmd = [
        sys.executable,
        str(Path("scripts/update_table1_from_tex.py")),
        "--tex",
        str(Path("Tex/Model.tex")),
        "--json",
        str(Path("data/probab01_table1.json")),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    # Ensure the symmetric_states appear in output and first eta matches 0.3
    out = proc.stdout
    data = json.loads(out)
    sym = data.get("table1_values", {}).get("symmetric_states", [])
    assert sym and abs(float(sym[0]["eta"]) - 0.3) < 1e-9

