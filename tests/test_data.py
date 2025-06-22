
from pathlib import Path

from scripts.generate_dividend_draws import main as generate


def test_csv_exists() -> None:
    csv_file = Path("data/dividend_draws.csv")
    assert csv_file.exists(), "CSV file missing"
    text = csv_file.read_text().strip().splitlines()
    assert text[0] == "period,dividend"
    assert text[1:] == ["1,1.0", "2,1.1", "3,0.9"]


def test_readme_mentions_csv() -> None:
    readme = Path("README.md").read_text()
    assert "data/dividend_draws.csv" in readme
    assert "dvc" not in readme.lower()


def test_no_dvc_files() -> None:
    assert not Path("dvc.yaml").exists(), "dvc.yaml should be removed"
    assert not Path("data/dividend_draws.csv.dvc").exists(), ".dvc file should be removed"


def test_generate_dividend_draws(tmp_path: Path) -> None:
    out_file = tmp_path / "dividend_draws.csv"
    generate(out_file)
    text = out_file.read_text().strip().splitlines()
    assert text[0] == "period,dividend"
    assert text[1:] == ["1,1.0", "2,1.1", "3,0.9"]
