import json
from pathlib import Path


def _load_nb(path: Path) -> dict:
    return json.loads(path.read_text())


def _has_text(nb: dict, needle: str) -> bool:
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if needle in src:
            return True
    return False


def test_multicountry_notebook_mentions_analytic_q():
    p = Path("notebooks/multicountry_probab01.ipynb")
    nb = _load_nb(p)
    assert _has_text(nb, "Analytic q (eq. 19)"), "analytic-q check missing"

