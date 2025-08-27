import json
from pathlib import Path


def _load_nb(path: Path) -> dict:
    return json.loads(path.read_text())


def test_multicountry_notebook_metadata_present():
    p = Path("notebooks/multicountry_probab01.ipynb")
    nb = _load_nb(p)
    assert isinstance(nb.get("cells"), list) and len(nb["cells"]) > 0
    ks = nb.get("metadata", {}).get("kernelspec", {})
    assert ks.get("name") == "python3"


def test_primitives_notebook_metadata_present():
    p = Path("notebooks/primitives_visuals.ipynb")
    nb = _load_nb(p)
    assert isinstance(nb.get("cells"), list) and len(nb["cells"]) > 0
    ks = nb.get("metadata", {}).get("kernelspec", {})
    assert ks.get("name") == "python3"

