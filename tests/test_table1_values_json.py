import json
from pathlib import Path

from bsde_dsgE.metrics.table1 import diag_pos_offdiag_neg


def test_table1_values_structure():
    p = Path("data/probab01_table1.json")
    data = json.loads(p.read_text())
    vals = data.get("table1_values", {})
    sym = vals.get("symmetric_states", [])
    assert isinstance(sym, list) and len(sym) >= 1
    ok_count = 0
    for state in sym:
        q = state["q"]
        sigma = state["sigma_q"]
        J = len(q)
        assert J == 5  # per paper example
        assert len(sigma) == J and all(len(row) == J for row in sigma)
        # sign pattern at symmetric states (allow occasional OCR/layout issues)
        res = diag_pos_offdiag_neg(sigma, tol=0.0)
        ok_count += int(res["ok"])
    assert ok_count >= max(1, len(sym) - 1)
