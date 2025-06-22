from pathlib import Path
import yaml

from scripts.generate_dividend_draws import main as generate


def test_dvc_stage():
    spec = yaml.safe_load(Path("dvc.yaml").read_text())
    assert "prepare_data" in spec["stages"]
    assert spec["stages"]["prepare_data"]["outs"] == ["data/dividend_draws.csv"]


def test_generate_dividend_draws(tmp_path):
    out_file = tmp_path / "dividend_draws.csv"
    generate(out_file)
    text = out_file.read_text().strip().splitlines()
    assert text[0] == "period,dividend"
    assert text[1:] == ["1,1.0", "2,1.1", "3,0.9"]
