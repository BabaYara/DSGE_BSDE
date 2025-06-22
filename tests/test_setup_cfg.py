"""Tests for project configuration."""

from __future__ import annotations

import configparser
import tomllib
from pathlib import Path


def test_setup_cfg_has_tool_settings() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "setup.cfg"
    assert cfg_path.exists(), "setup.cfg missing"

    parser = configparser.ConfigParser()
    parser.read(cfg_path)

    assert parser["tool.ruff"]["src"] == "bsde_dsgE"
    assert parser["tool.ruff"]["exclude"] == '["notebooks/**"]'
    assert parser["tool.ruff.lint"]["select"] == "E, F, I, B"
    assert parser["tool.ruff.lint"]["ignore"] == '["E401", "E731", "I001", "E501"]'

    assert parser["mypy"]["python_version"] == "3.11"
    assert parser.getboolean("mypy", "strict")
    assert parser["mypy"]["files"] == "bsde_dsgE"


def test_pyproject_has_no_ruff_settings() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml missing"

    data = tomllib.loads(pyproject_path.read_text())
    assert "tool" not in data or "ruff" not in data.get("tool", {})

