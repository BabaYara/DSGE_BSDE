import json
from types import SimpleNamespace

import jax.numpy as jnp

from examples import grid_search


def test_grid_search_fast(monkeypatch, tmp_path):
    args = SimpleNamespace(gammas="1,2,3", output=tmp_path)
    monkeypatch.setattr(grid_search, "parse_args", lambda: args)

    monkeypatch.setattr(grid_search.ct_lucas, "scalar_lucas", lambda gamma: None)

    def dummy_load_solver(model, dt):
        def solver(x, key):
            return jnp.array(0.0)

        return solver

    monkeypatch.setattr(grid_search, "load_solver", dummy_load_solver)
    monkeypatch.setenv("NOTEBOOK_FAST", "1")
    grid_search.main()

    data = json.load(open(tmp_path / "grid.json"))
    assert len(data) == 1


def test_grid_search_outputs(monkeypatch, tmp_path):
    args = SimpleNamespace(gammas="1,2,3", output=tmp_path)
    monkeypatch.setattr(grid_search, "parse_args", lambda: args)

    monkeypatch.setattr(grid_search.ct_lucas, "scalar_lucas", lambda gamma: None)

    def dummy_load_solver(model, dt):
        def solver(x, key):
            return jnp.array(0.0)

        return solver

    monkeypatch.setattr(grid_search, "load_solver", dummy_load_solver)

    grid_search.main()

    data = json.load(open(tmp_path / "grid.json"))
    assert set(data.keys()) == {"1.0", "2.0", "3.0"}
