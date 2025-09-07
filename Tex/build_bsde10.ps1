Param(
    [string]$File = "BSDE_10.tex"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Push-Location $PSScriptRoot
try {
    # First LaTeX pass to generate .pytxcode (call pdflatex directly to avoid premature pythontex)
    pdflatex -interaction=nonstopmode -synctex=1 -shell-escape -output-directory=.out $File

    # Run PythonTeX on generated .pytxcode
    $job = [System.IO.Path]::GetFileNameWithoutExtension($File)
    pythontex ".out/$job.pytxcode"

    # Second LaTeX pass to include PythonTeX output
    pdflatex -interaction=nonstopmode -synctex=1 -shell-escape -output-directory=.out $File
}
finally {
    Pop-Location
}
