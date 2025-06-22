from pathlib import Path
import tomllib


def test_dev_optional_dependencies_include_sphinx() -> None:
    pyproject = Path("pyproject.toml")
    assert pyproject.exists(), "pyproject.toml missing"
    data = tomllib.loads(pyproject.read_text())
    dev_deps = data["project"]["optional-dependencies"]["dev"]
    assert "sphinx" in dev_deps, "'sphinx' not found in dev optional dependencies"


def test_wheel_includes_py_typed() -> None:
    pyproject = Path("pyproject.toml")
    assert pyproject.exists(), "pyproject.toml missing"
    data = tomllib.loads(pyproject.read_text())
    wheel_cfg = (
        data
        .get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
    )
    include = wheel_cfg.get("include", [])
    assert "bsde_dsgE/py.typed" in include


def test_py_typed_file_exists() -> None:
    assert Path("bsde_dsgE/py.typed").exists()

