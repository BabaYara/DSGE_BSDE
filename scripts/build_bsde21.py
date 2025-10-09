#!/usr/bin/env python3
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX_DIR = ROOT / "Tex"
OUT_DIR = ROOT / ".out"
TEX_FILE = TEX_DIR / "BSDE_21.tex"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_paths() -> None:
    if not TEX_FILE.exists():
        raise FileNotFoundError(f"Missing {TEX_FILE}")
    OUT_DIR.mkdir(exist_ok=True)


def stage_pythontex_inputs() -> Path:
    pytx = OUT_DIR / "BSDE_21.pytxcode"
    if not pytx.exists():
        raise FileNotFoundError("pdflatex did not produce BSDE_21.pytxcode")
    staged = TEX_DIR / "BSDE_21.pytxcode"
    shutil.copy2(pytx, staged)
    return staged


def harvest_pythontex_outputs() -> None:
    src_dir = TEX_DIR / "pythontex-files-bsde_21"
    if not src_dir.exists():
        raise FileNotFoundError("pythontex did not create pythontex-files-bsde_21")
    dest_dir = OUT_DIR / "pythontex-files-BSDE_21"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)
    shutil.rmtree(src_dir)


def build() -> None:
    ensure_paths()
    first_pass = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={OUT_DIR}", str(TEX_FILE)]
    run(first_pass)

    staged = stage_pythontex_inputs()
    try:
        run(["pythontex", str(TEX_FILE)])
        harvest_pythontex_outputs()
    finally:
        if staged.exists():
            staged.unlink()

    second_pass = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={OUT_DIR}", str(TEX_FILE)]
    run(second_pass)
    pdf = OUT_DIR / "BSDE_21.pdf"
    if not pdf.exists():
        raise FileNotFoundError(f"Expected output {pdf} missing")
    print(f"Build complete: {pdf}")


def main() -> int:
    try:
        build()
    except subprocess.CalledProcessError as err:
        print(f"Command failed with exit code {err.returncode}: {err.cmd}", file=sys.stderr)
        return err.returncode or 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
