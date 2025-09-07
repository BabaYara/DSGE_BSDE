#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

file="${1:-BSDE_10.tex}"

# First LaTeX pass to generate .pytxcode (call pdflatex directly to avoid premature pythontex)
pdflatex -interaction=nonstopmode -synctex=1 -shell-escape -output-directory=.out "$file"

# Run PythonTeX on generated .pytxcode
jobname=$(basename "$file" .tex)
pythontex ".out/${jobname}.pytxcode"

# Second LaTeX pass to include PythonTeX output
pdflatex -interaction=nonstopmode -synctex=1 -shell-escape -output-directory=.out "$file"
