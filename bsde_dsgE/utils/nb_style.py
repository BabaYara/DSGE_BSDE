"""Notebook visual style helpers (2025 standards).

Applies an accessible, modern Matplotlib style suitable for inline notebooks:
- Prefer Matplotlib's 'petroff10' (or 8/6) accessible color cycles if available
- Configure high-DPI, readable fonts, and subtle major/minor grids
- Enable inline HTML animations (jshtml)

Safe to call multiple times; uses best-effort fallbacks when styles are missing.
"""

from __future__ import annotations

from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


def _try_styles(*names: str) -> Optional[str]:
    """Return the first available style name from the given list, else None."""
    available = set(plt.style.available)
    for name in names:
        if name in available:
            return name
    return None


def apply_notebook_style() -> None:
    """Apply an accessible, modern style for notebook figures.

    Uses petroff color cycles when available (Matplotlib ≥3.10 adds petroff10,
    and Matplotlib next adds petroff8/petroff6). Falls back to seaborn v0.8
    whitegrid or the default style if not present.
    """
    # Choose an accessible style if available
    chosen = _try_styles("petroff10", "petroff8", "petroff6")
    if chosen is None:
        # Prefer seaborn v0.8 whitegrid if shipped; otherwise default
        chosen = _try_styles("seaborn-v0_8-whitegrid") or "default"
    plt.style.use(chosen)

    # High-DPI and figure size for readability
    mpl.rcParams["figure.dpi"] = 120
    mpl.rcParams["savefig.dpi"] = 150
    mpl.rcParams.setdefault("figure.figsize", (7.0, 4.0))

    # Typography and titles
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["axes.titleweight"] = "semibold"
    # Matplotlib ≥3.2 supports title location/color rcParams
    mpl.rcParams["axes.titlelocation"] = "left"
    mpl.rcParams["axes.titlecolor"] = "auto"

    # Grid: show both major/minor with subtle styling
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["axes.grid.which"] = "both"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    mpl.rcParams["grid.major.color"] = "#d9d9d9"
    mpl.rcParams["grid.major.linewidth"] = 0.8
    mpl.rcParams["grid.minor.linestyle"] = ":"
    mpl.rcParams["grid.minor.alpha"] = 0.5

    # Legends and errorbar caps
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["errorbar.capsize"] = 3

    # Animations inline in Jupyter
    mpl.rcParams["animation.html"] = "jshtml"


__all__ = ["apply_notebook_style"]

