import importlib.util
import pathlib
import sys
import pytest


@pytest.mark.skipif('marimo' not in sys.modules and not importlib.util.find_spec('marimo'),
                    reason="marimo not installed in this environment")
def test_book2_has_marimo_app():
    # Load module from file path without requiring package import
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / 'Marimo' / 'Book_2.py'
    spec = importlib.util.spec_from_file_location('book2_module', path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules['book2_module'] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    # Basic checks
    import marimo as mo  # type: ignore
    assert hasattr(mod, 'app'), "Book_2 module must define 'app'"
    assert isinstance(mod.app, mo.App), "'app' must be a marimo.App instance"