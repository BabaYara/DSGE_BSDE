"""Tests for project configuration."""

from __future__ import annotations

import configparser
from pathlib import Path


def test_setup_cfg_has_tool_settings() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "setup.cfg"
    assert cfg_path.exists(), "setup.cfg missing"

    parser = configparser.ConfigParser()
    parser.read(cfg_path)

    assert parser["tool:ruff"]["src"] == "bsde_dsgE"
    assert parser["tool:ruff.lint"]["select"] == "E, F, I, B"

    assert parser["mypy"]["python_version"] == "3.11"
    assert parser.getboolean("mypy", "strict")
    assert parser["mypy"]["files"] == "bsde_dsgE"

