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


def test_multicountry_notebook_has_core_figures():
    p = Path("notebooks/multicountry_probab01.ipynb")
    nb = _load_nb(p)
    # Rolling correlation animation and mean ± 2SE should be mentioned
    assert _has_text(nb, "rolling correlation"), "rolling correlation section missing"
    assert _has_text(nb, "Mean ± 2SE"), "mean ± 2SE section missing"


def test_primitives_notebook_has_key_demos():
    p = Path("notebooks/primitives_visuals.ipynb")
    nb = _load_nb(p)
    # Antithetic pairing demo and feature maps should be present
    assert _has_text(nb, "Antithetic"), "antithetic pairing demo missing"
    assert _has_text(nb, "Feature Maps"), "ResNetND feature maps section missing"
    assert _has_text(nb, "QQ Plot"), "QQ plot section missing"
    assert _has_text(nb, "Autocorrelation"), "lagged autocorrelation section missing"
