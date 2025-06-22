from pathlib import Path

from scripts.generate_dividend_draws import main as generate


def test_csv_dvc_exists() -> None:
    dvc_file = Path("data/dividend_draws.csv.dvc")
    assert dvc_file.exists(), ".dvc file missing"

    text = dvc_file.read_text().splitlines()[0]
    assert text.strip() == "outs:", ".dvc file format unexpected"


def test_readme_dvc_repro() -> None:
    readme = Path("README.md").read_text()
    assert "dvc repro fetch-data" in readme


def test_dvc_yaml_stage_exists() -> None:
    dvc_yaml = Path("dvc.yaml")
    assert dvc_yaml.exists(), "dvc.yaml missing"
    text = dvc_yaml.read_text()
    assert "fetch-data:" in text, "fetch-data stage missing"
    assert "data/dividend_draws.csv.dvc" in text, "dependency missing"
    assert "data/dividend_draws.csv" in text, "output missing"
    assert "cmd: dvc pull data/dividend_draws.csv.dvc" in text


def test_generate_dividend_draws(tmp_path: Path) -> None:
    out_file = tmp_path / "dividend_draws.csv"
    generate(out_file)
    text = out_file.read_text().strip().splitlines()
    assert text[0] == "period,dividend"
    assert text[1:] == ["1,1.0", "2,1.1", "3,0.9"]
