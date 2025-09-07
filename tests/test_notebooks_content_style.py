import json
from pathlib import Path


def _has_text(nb: dict, needle: str) -> bool:
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if needle in src:
            return True
    return False


def test_notebooks_apply_style_present():
    for p in (
        Path("notebooks/multicountry_probab01.ipynb"),
        Path("notebooks/primitives_visuals.ipynb"),
        Path("notebooks/multicountry_probab01_solver.ipynb"),
        Path("notebooks/primitives_visuals_extra.ipynb"),
    ):
        nb = json.loads(p.read_text())
        assert _has_text(nb, "apply_notebook_style"), f"missing style cell in {p}"
