import re
from pathlib import Path


def _get_literate_block(tex: str) -> str:
    match = re.search(r"\\lstset\{(?P<body>.*?)\n\}", tex, re.DOTALL)
    assert match, "\\lstset block missing"
    return match.group("body")


def test_bsde32_utf8_bom_removed():
    tex_path = Path("Tex/BSDE_32.tex")
    data = tex_path.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf"), "BOM should be removed for stable builds"


def test_bsde32_literate_maps_extended_unicode():
    tex = Path("Tex/BSDE_32.tex").read_text(encoding="utf-8")
    literate = _get_literate_block(tex)
    lines = [line.strip() for line in literate.splitlines() if line.strip()]
    latex_map = {
        "ℝ": "mathbb{R}",
        "∇": "nabla",
        "Σ": "Sigma",
        "σ": "sigma",
        "μ": "mu",
        "ρ": "rho",
        "γ": "gamma",
        "κ": "kappa",
    }
    for char, latex in latex_map.items():
        assert any(
            line.startswith(f"{{{char}}}") and ("\\" + latex) in line
            for line in lines
        ), f"Expected literate mapping for {char!r}"
