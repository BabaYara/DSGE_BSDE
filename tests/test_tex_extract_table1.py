from pathlib import Path
import json
import numpy as np

from bsde_dsgE.utils.tex_extract import extract_symmetric_states


def test_tex_extract_matches_json_sym_states():
    tex_path = Path("Tex/Model.tex")
    assert tex_path.exists()
    states = extract_symmetric_states(tex_path)
    assert len(states) >= 5

    json_path = Path("data/probab01_table1.json")
    data = json.loads(json_path.read_text())
    sym_json = data.get("table1_values", {}).get("symmetric_states", [])
    assert sym_json, "JSON symmetric_states missing"

    # Compare first and last state to be robust to minor formatting
    for idx in (0, -1):
        s_tex = states[idx]
        s_json = sym_json[idx]
        assert abs(float(s_tex["eta"]) - float(s_json["eta"])) < 1e-9
        assert abs(float(s_tex["zeta"]) - float(s_json["zeta"])) < 1e-9
        assert np.allclose(np.array(s_tex["q"], float), np.array(s_json["q"], float))
        assert np.allclose(np.array(s_tex["sigma_q"], float), np.array(s_json["sigma_q"], float))

