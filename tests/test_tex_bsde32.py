import re
import shutil
import subprocess
from pathlib import Path

import pytest


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


def test_bsde32_compiles_with_latexmk(tmp_path):
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        pytest.skip("latexmk executable is unavailable")

    outdir = tmp_path / "pdf"
    auxdir = tmp_path / "aux"
    outdir.mkdir()
    auxdir.mkdir()

    jobname = "bsde32_ci"
    cmd = [
        latexmk,
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-jobname={jobname}",
        f"-outdir={outdir}",
        f"-auxdir={auxdir}",
        "-r",
        "latexmkrc",
        "BSDE_32.tex",
    ]

    subprocess.run(cmd, check=True, cwd=Path("Tex"))
    pdf_path = outdir / f"{jobname}.pdf"
    log_path = auxdir / f"{jobname}.log"
    assert pdf_path.exists(), "latexmk should produce a PDF output"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    assert "undefined references" not in log_text.lower(), (
        "latexmk log reports undefined references; rerun or fix labels"
    )


def test_bsde32_lean_snippets_reference_lucasnotes():
    tex = Path("Tex/BSDE_32.tex").read_text(encoding="utf-8")
    checks = [
        "lean:lemmaC_drift",
        "lean:sdf_log_utility",
        "lean:prop:sdf_algebra",
        "lean:lemma_4_2_algebra",
        "lean:prop_b1_algebra",
        "lean:face_order",
    ]
    for label in checks:
        marker = f"label={{{label}}}"
        try:
            start = tex.index(marker)
        except ValueError as err:  # pragma: no cover - informative assertion below
            raise AssertionError(f"Lean proof block for {label} should be present") from err
        end = tex.index("\\end{leanproof}", start)
        segment = tex[start:end]
        assert "lean/LucasNotes.lean" in segment, (
            f"Lean proof block for {label} must include lucasnotes excerpt"
        )


def test_bsde32_main_text_lean_snippet_uses_lucasnotes():
    tex = Path("Tex/BSDE_32.tex").read_text(encoding="utf-8")
    snippet = re.search(
        r"Lean status: Aggregate consumption dynamics[\s\S]*?\\end{leanproof}", tex
    )
    assert snippet, "Expected aggregate consumption Lean snippet"
    assert "lean/LucasNotes.lean" in snippet.group(0), (
        "Aggregate consumption Lean snippet must input LucasNotes"
    )
