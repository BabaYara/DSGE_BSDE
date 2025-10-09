Param(
    [string]$File = "BSDE_10.tex"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Step {
    param(
        [Parameter(Mandatory)][string]$Command
    )

    Write-Host "`n>>> $Command" -ForegroundColor Cyan
    & cmd /c $Command
    if (${LASTEXITCODE} -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command"
    }
}

Push-Location $PSScriptRoot
try {
    $resolved = Resolve-Path -Path $File
    $job = [System.IO.Path]::GetFileNameWithoutExtension($resolved)
    $outDir = Join-Path -Path (Get-Location) -ChildPath '.out'
    if (-not (Test-Path -Path $outDir)) {
        New-Item -ItemType Directory -Path $outDir | Out-Null
    }

    # First LaTeX pass to emit .pytxcode artefacts
    Invoke-Step "pdflatex -interaction=nonstopmode -synctex=1 -shell-escape -output-directory=.out `"$File`""

    $pytxSource = Join-Path -Path $outDir -ChildPath "$job.pytxcode"
    if (-not (Test-Path -Path $pytxSource)) {
        throw "PythonTeX source not found: $pytxSource"
    }

    # PythonTeX must run inside .out so the generated files land alongside LaTeX outputs
    Invoke-Step "cd /d `"$outDir`" && pythontex `"$job.pytxcode`""

    # Second LaTeX pass to pull in console output/macros
    Invoke-Step "pdflatex -interaction=nonstopmode -synctex=1 -shell-escape -output-directory=.out `"$File`""
}
finally {
    Pop-Location
}
