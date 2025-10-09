[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TexFile,
    [switch]$SkipLean,
    [switch]$SkipBlueprint,
    [string[]]$LatexmkArgs = @('-pdf', '-interaction=nonstopmode', '-file-line-error')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path $scriptDir

function Resolve-TexPath {
    param([string]$InputPath)

    if ([string]::IsNullOrWhiteSpace($InputPath)) {
        throw "Provide a LaTeX source path via -TexFile."
    }

    if ([System.IO.Path]::IsPathRooted($InputPath)) {
        return Resolve-Path -LiteralPath $InputPath -ErrorAction Stop
    }

    $texDir   = Join-Path $repoRoot 'Tex'
    $normalized = $InputPath.TrimStart('.','/','\')
    $candidates = [System.Collections.Generic.List[string]]::new()

    $candidates.Add((Join-Path $repoRoot $normalized))

    if ($normalized -notmatch '^[Tt]ex[\\/]') {
        $candidates.Add((Join-Path $texDir $normalized))
    }

    $leaf = Split-Path $normalized -Leaf
    if ($leaf -and $leaf -ne $normalized) {
        $candidateLeaf = Join-Path $texDir $leaf
        if (-not $candidates.Contains($candidateLeaf)) {
            $candidates.Add($candidateLeaf)
        }
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return Resolve-Path -LiteralPath $candidate -ErrorAction Stop
        }
    }

    $message = "Could not locate LaTeX source '$InputPath'. Tried:`n  " + ($candidates -join "`n  ") + "`nProvide a rooted path or a path relative to repo root or Tex/."
    throw [System.IO.FileNotFoundException]::new($message)
}

$texPath = Resolve-TexPath -InputPath $TexFile
Write-Host "LaTeX source: $texPath" -ForegroundColor Cyan

Push-Location $repoRoot
try {
    if (-not $SkipLean) {
        $lake = Get-Command -Name 'lake' -ErrorAction SilentlyContinue
        if (-not $lake) {
            throw "Could not find 'lake' on PATH. Install the Lean toolchain or pass -SkipLean."
        }

        Write-Host "Running 'lake build'..." -ForegroundColor Cyan
        Push-Location (Join-Path $repoRoot 'lean')
        try {
            & $lake.Source 'build'
            Write-Host "Lean build completed." -ForegroundColor Green

            try {
                Write-Host "Running 'lake exe lean4checker'..." -ForegroundColor Cyan
                & $lake.Source 'exe' 'lean4checker'
                Write-Host "lean4checker completed." -ForegroundColor Green
            } catch {
                Write-Warning "Skipping 'lake exe lean4checker' (target not available?)."
            }
        } finally {
            Pop-Location
        }
    } else {
        Write-Host "Skipping Lean build step as requested." -ForegroundColor Yellow
    }

    if (-not $SkipBlueprint) {
        $blueprint = Get-Command -Name 'leanblueprint' -ErrorAction SilentlyContinue
        if ($blueprint) {
            Write-Host "Running 'leanblueprint checkdecls'..." -ForegroundColor Cyan
            Push-Location (Join-Path $repoRoot 'lean')
            try {
                & $blueprint.Source 'checkdecls'
                Write-Host "leanblueprint checkdecls completed." -ForegroundColor Green
            } finally {
                Pop-Location
            }
        } else {
            Write-Warning "Could not find 'leanblueprint' on PATH; skipping blueprint checks."
        }
    }

    $latexmk = Get-Command -Name 'latexmk' -ErrorAction SilentlyContinue
    if (-not $latexmk) {
        throw "Could not find 'latexmk'. Install TeX Live / MiKTeX and ensure 'latexmk' is on PATH."
    }

    $latexmkrc = Join-Path $repoRoot 'Tex/latexmkrc'
    if (-not (Test-Path -LiteralPath $latexmkrc)) {
        Write-Warning "latexmkrc not found at $latexmkrc; proceeding without custom rc."
        $latexArgs = @($LatexmkArgs) + @($texPath)
    } else {
        $latexArgs = @($LatexmkArgs) + @('-r', $latexmkrc, $texPath)
    }

    Write-Host "Running latexmk $($latexArgs -join ' ')..." -ForegroundColor Cyan
    & $latexmk.Source @latexArgs
    Write-Host "LaTeX build completed successfully." -ForegroundColor Green
}
finally {
    Pop-Location
}
