from pathlib import Path
import json

from bsde_dsgE.utils.calibration import load_probab01_calibration


def test_load_probab01_calibration_roundtrip(tmp_path: Path):
    data = {
        "dim": 3,
        "rho": 0.04,
        "gamma": 7.0,
        "kappa": 0.25,
        "theta": 1.1,
        "sigma": [0.2, 0.25, 0.3],
        "table1_targets": {"foo": 1},
    }
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(data))
    calib = load_probab01_calibration(p)
    assert calib.dim == 3 and calib.gamma == 7.0


def test_load_probab01_calibration_missing_field(tmp_path: Path):
    data = {
        "dim": 2,
        # missing rho
        "gamma": 7.0,
        "kappa": 0.25,
        "theta": 1.1,
        "sigma": 0.3,
    }
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(data))
    try:
        load_probab01_calibration(p)
        assert False, "Expected ValueError for missing fields"
    except ValueError as e:
        assert "missing fields" in str(e).lower()

