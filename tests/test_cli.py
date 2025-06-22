from bsde_dsgE.cli import pinn_demo


def test_pinn_demo_runs(capsys):
    pinn_demo()
    out = capsys.readouterr().out
    assert "final loss" in out
