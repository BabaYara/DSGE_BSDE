"""Utilities to extract Table 1 symmetric-state values from LaTeX (Tex/Model.tex).

Parses the 'Symmetric State' table and returns a list of dicts with keys
eta, zeta, q, sigma_q (rows=j, cols=i) matching the JSON structure used in
data/probab01_table1.json.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")


def _floats_in_line(line: str) -> List[float]:
    return [float(x) for x in _FLOAT_RE.findall(line)]


def extract_symmetric_states(tex_path: str | Path) -> List[Dict[str, Any]]:
    p = Path(tex_path)
    txt = p.read_text()
    # Locate the Symmetric State table block
    # Use a simple window around the caption to parse content
    cap_idx = txt.find("\\caption{Symmetric State}")
    if cap_idx == -1:
        raise ValueError("Could not find 'Symmetric State' caption in TeX file")
    window = txt[cap_idx : cap_idx + 5000]
    lines = [ln.strip() for ln in window.splitlines()]

    states: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("\\multicolumn{6}{l}{$") and "eta" in line or "\\eta" in line:
            # Example: \multicolumn{6}{l}{$\eta_{t}^{i}=0.3, \zeta_{t}^{j}=0.2$} \\
            floats = _floats_in_line(line)
            if len(floats) >= 2:
                eta, zeta = floats[0], floats[1]
            else:
                i += 1
                continue
            # Skip header line & midrule to q^i line
            # Typically: next lines: header, \midrule, then q^i
            # Scan ahead to find a line starting with q^i
            j = i + 1
            while j < len(lines) and not lines[j].startswith("$q^{i}") and not lines[j].startswith("q^{i}"):
                j += 1
            if j >= len(lines):
                break
            q_vals = _floats_in_line(lines[j])
            if len(q_vals) != 5:
                # malformed; skip
                i = j + 1
                continue
            # Next 5 lines are sigma rows for j=1..5
            sigma_rows: List[List[float]] = []
            k = j + 1
            for _ in range(5):
                if k >= len(lines):
                    break
                vals = _floats_in_line(lines[k])
                if len(vals) == 5:
                    sigma_rows.append(vals)
                k += 1
            if len(sigma_rows) == 5:
                states.append({
                    "eta": eta,
                    "zeta": zeta,
                    "q": q_vals,
                    "sigma_q": sigma_rows,
                })
            i = k
            continue
        i += 1
    if not states:
        raise ValueError("No symmetric state blocks parsed from TeX file")
    return states


__all__ = ["extract_symmetric_states"]

