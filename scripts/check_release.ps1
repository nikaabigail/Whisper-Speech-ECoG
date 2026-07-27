[CmdletBinding()]
param(
    [string]$PythonLauncher = "py",
    [string[]]$PythonPrefix = @("-3.10"),
    [switch]$SkipImports,
    [switch]$SkipHelp,
    [switch]$SkipPlotRefresh
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

if (-not (Get-Command $PythonLauncher -ErrorAction SilentlyContinue)) {
    throw "Python launcher not found: $PythonLauncher"
}

Push-Location -LiteralPath $repoRoot
try {
    # A reproducible local environment is intentionally allowed inside the checkout.
    # Inspect only files that Git would publish; scanning .venv would otherwise flag
    # dependency bytecode that is both expected and ignored.
    $bytecodeArtifacts = @()
    $releaseCandidates = $null
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $insideGit = (& git rev-parse --is-inside-work-tree 2>$null)
        if ($LASTEXITCODE -eq 0 -and $insideGit -eq "true") {
            $gitTop = (& git rev-parse --show-toplevel 2>$null)
            $resolvedGitTop = if ($gitTop) { (Resolve-Path -LiteralPath $gitTop).Path } else { $null }
            if ($resolvedGitTop -eq $repoRoot) {
                $releaseCandidates = @(& git ls-files --cached --others --exclude-standard)
                $bytecodeArtifacts = @(
                    $releaseCandidates |
                        Where-Object {
                            $_ -match '(^|[\\/])__pycache__([\\/]|$)' -or
                            [IO.Path]::GetExtension($_) -match '^\.py[co]$'
                        } |
                        ForEach-Object { Join-Path $repoRoot $_ }
                )
            }
        }
    }
    if ($null -eq $releaseCandidates) {
        $bytecodeArtifacts = @(
            Get-ChildItem -LiteralPath $repoRoot -Recurse -Force -ErrorAction Stop |
                Where-Object {
                    $_.FullName -notmatch '[\\/](?:\.git|\.venv)(?:[\\/]|$)' -and
                    (
                        ($_.PSIsContainer -and $_.Name -eq "__pycache__") -or
                        (-not $_.PSIsContainer -and $_.Extension -match '^\.py[co]$')
                    )
                } |
                ForEach-Object { $_.FullName }
        )
    }
    if ($bytecodeArtifacts.Count -gt 0) {
        $listed = $bytecodeArtifacts -join "`n"
        throw "Generated Python bytecode is present in the release tree. Remove it before packaging:`n$listed"
    }

    if (-not $SkipPlotRefresh) {
        Write-Host "[check] Regenerating curated PNG summaries from reported_metrics.json"
        & $PythonLauncher @PythonPrefix -B "results/plot_reported_metrics.py"
        if ($LASTEXITCODE -ne 0) {
            throw "Plot generation failed with exit code $LASTEXITCODE"
        }
    }

    $smokeArgs = @("-B", "tests/smoke_test.py")
    if ($SkipImports) { $smokeArgs += "--skip-imports" }
    if ($SkipHelp) { $smokeArgs += "--skip-help" }

    Write-Host "[check] Running offline release smoke checks"
    & $PythonLauncher @PythonPrefix @smokeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Release smoke checks failed with exit code $LASTEXITCODE"
    }

    $licenseFiles = @(Get-ChildItem -LiteralPath $repoRoot -File -ErrorAction Stop |
        Where-Object { $_.Name -match '^LICENSE(?:\.|$)' })
    if ($licenseFiles.Count -eq 0) {
        Write-Warning "No LICENSE file is present. Resolve code/checkpoint licensing before a public push."
    }

    if (Get-Command git -ErrorAction SilentlyContinue) {
        $insideGit = (& git rev-parse --is-inside-work-tree 2>$null)
        if ($LASTEXITCODE -eq 0 -and $insideGit -eq "true") {
            $gitTop = (& git rev-parse --show-toplevel 2>$null)
            $resolvedGitTop = if ($gitTop) { (Resolve-Path -LiteralPath $gitTop).Path } else { $null }
            if ($resolvedGitTop -eq $repoRoot) {
                Write-Host "[check] Git status"
                & git status --short
            }
            else {
                Write-Host "[check] Git status skipped: release folder is not its own Git worktree yet."
            }
        }
    }

    Write-Host "[done] Release checks passed."
}
finally {
    Pop-Location
}
