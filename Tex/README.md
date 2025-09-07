BSDE_10 build notes

- Prereqs: TeX Live/MiKTeX with pythontex, and Python 3 with SymPy installed.
- Install SymPy: `pip install sympy` (into the Python used by pythontex).
- VS Code (LaTeX Workshop) or `latexmk` picks up `Tex/latexmkrc` and runs PythonTeX automatically.

CLI options

- Fast path: `cd Tex && latexmk -pdf BSDE_10.tex` (auto-runs PythonTeX; outputs to `.out/`).
- Explicit path: `./build_bsde10.sh` or `./build_bsde10.ps1` (runs pdflatex → pythontex → pdflatex).

Appendix E

- Contains SymPy assertions that verify key derivations used in the paper.
- Any failed assertion stops the build, gating the PDF on algebraic correctness.

