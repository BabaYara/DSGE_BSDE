from pathlib import Path
import tomllib


def test_dev_optional_dependencies_include_sphinx() -> None:
    pyproject = Path("pyproject.toml")
    assert pyproject.exists(), "pyproject.toml missing"
    data = tomllib.loads(pyproject.read_text())
    dev_deps = data["project"]["optional-dependencies"]["dev"]
    assert "sphinx" in dev_deps, "'sphinx' not found in dev optional dependencies"

