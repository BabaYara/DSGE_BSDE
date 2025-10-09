from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_lean_workspace_builds():
    lake = shutil.which("lake")
    if lake is None:
        pytest.skip("lake executable is unavailable")

    subprocess.run([lake, "build"], check=True, cwd=Path("lean"))
