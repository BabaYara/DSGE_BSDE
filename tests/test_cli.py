import jax.numpy as jnp

from bsde_dsgE import cli


def test_pinn_demo_runs(capsys):
    cli.pinn_demo()
    out = capsys.readouterr().out
    assert "final loss" in out


def test_pinn_poisson2d_runs(capsys):
    cli.pinn_poisson2d()
    out = capsys.readouterr().out
    assert "final loss" in out


def test_pinn_demo_fast_env(monkeypatch):
    captured = {}

    class DummySolver:
        def __init__(self, *args, num_steps: int, **kwargs):
            captured["steps"] = num_steps

        def run(self, xs, key):
            return jnp.zeros(captured["steps"])

    monkeypatch.setattr(cli, "KFACPINNSolver", DummySolver)
    monkeypatch.setenv("NOTEBOOK_FAST", "1")
    cli.pinn_demo()
    assert captured["steps"] == 3


def test_pinn_poisson2d_fast_env(monkeypatch):
    captured = {}

    class DummySolver:
        def __init__(self, *args, num_steps: int, **kwargs):
            captured["steps"] = num_steps

        def run(self, xs, key):
            return jnp.zeros(captured["steps"])

    monkeypatch.setattr(cli, "KFACPINNSolver", DummySolver)
    monkeypatch.setenv("NOTEBOOK_FAST", "1")
    cli.pinn_poisson2d()
    assert captured["steps"] == 3
