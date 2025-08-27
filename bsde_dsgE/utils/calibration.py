"""Calibration file loader and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class Probab01Calibration:
    dim: int
    rho: float
    gamma: float
    kappa: float
    theta: float
    sigma: list[float] | float
    table1_targets: dict[str, Any] | None = None


REQUIRED_FIELDS = {"dim", "rho", "gamma", "kappa", "theta", "sigma"}


def load_probab01_calibration(path: str | Path) -> Probab01Calibration:
    p = Path(path)
    data = json.loads(p.read_text())
    missing = REQUIRED_FIELDS.difference(data.keys())
    if missing:
        raise ValueError(f"Calibration missing fields: {sorted(missing)}")
    return Probab01Calibration(
        dim=int(data["dim"]),
        rho=float(data["rho"]),
        gamma=float(data["gamma"]),
        kappa=float(data["kappa"]),
        theta=float(data["theta"]),
        sigma=data["sigma"],
        table1_targets=data.get("table1_targets"),
    )


__all__ = ["Probab01Calibration", "load_probab01_calibration"]

