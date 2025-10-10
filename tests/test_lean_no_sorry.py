from __future__ import annotations

import subprocess
from pathlib import Path


def test_no_sorry_or_admit_in_lean_sources():
    repo_root = Path(__file__).resolve().parents[1]
    lean_dir = repo_root / "lean"
    # `rg` returns exit code 1 when no matches are found; treat that as success.
    # We explicitly invoke it with `--files-with-matches` to avoid printing the
    # entire file contents in the test output if a violation slips in.
    for pattern in ("sorry", "admit"):
        completed = subprocess.run(
            [
                "rg",
                "--files-with-matches",
                "-g",
                "*.lean",
                pattern,
                str(lean_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert (
            completed.returncode == 1
        ), f"Found forbidden '{pattern}' in Lean sources:\n{completed.stdout}"
