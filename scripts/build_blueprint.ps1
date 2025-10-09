[CmdletBinding()]
param(
    [string]$TexFile = 'Tex/BSDE_32.tex',
    [switch]$SkipLakeBuild,
    [switch]$SkipCheckDecls,
    [switch]$SkipPdf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path $scriptDir

function Invoke-Lake {
    param(
        [string[]]$Arguments,
        [string]$WorkDir
    )

    $lake = Get-Command -Name 'lake' -ErrorAction SilentlyContinue
    if (-not $lake) {
        throw "Could not find 'lake' on PATH. Install the Lean toolchain before running this script."
    }

    Write-Host "Running 'lake $($Arguments -join ' ')'..." -ForegroundColor Cyan
    Push-Location $WorkDir
    try {
        & $lake.Source @Arguments
    } finally {
        Pop-Location
    }
    Write-Host "Lake command completed." -ForegroundColor Green
}

Write-Host "Blueprint build root: $repoRoot" -ForegroundColor Cyan

if (-not $SkipLakeBuild) {
    Invoke-Lake -Arguments @('build') -WorkDir (Join-Path $repoRoot 'lean')
} else {
    Write-Host "Skipping 'lake build' step." -ForegroundColor Yellow
}

if (-not $SkipCheckDecls) {
    $declsPath = Join-Path $repoRoot 'lean/blueprint/lean_decls'
    if (-not (Test-Path -LiteralPath $declsPath)) {
        throw "Expected declaration list at '$declsPath'. Populate it before running this script."
    }
    Invoke-Lake -Arguments @('exe', 'checkdecls', 'blueprint/lean_decls') -WorkDir (Join-Path $repoRoot 'lean')
} else {
    Write-Host "Skipping declaration check step." -ForegroundColor Yellow
}

if (-not $SkipPdf) {
    $buildTexScript = Join-Path $scriptDir 'build_tex.ps1'
    if (-not (Test-Path -LiteralPath $buildTexScript)) {
        throw "Unable to locate helper script at '$buildTexScript'."
    }
    Write-Host "Building LaTeX target '$TexFile'..." -ForegroundColor Cyan
    & $buildTexScript -TexFile $TexFile -SkipLean
} else {
    Write-Host "Skipping LaTeX build." -ForegroundColor Yellow
}

Write-Host "Blueprint pipeline completed successfully." -ForegroundColor Green
