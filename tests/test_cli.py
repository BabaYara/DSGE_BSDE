from bsde_dsgE.cli import pinn_demo, pinn_poisson2d


def test_pinn_demo_runs(capsys):
    pinn_demo()
    out = capsys.readouterr().out
    assert "final loss" in out


def test_pinn_poisson2d_runs(capsys):
    pinn_poisson2d()
    out = capsys.readouterr().out
    assert "final loss" in out
