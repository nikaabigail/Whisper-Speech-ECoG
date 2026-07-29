[CmdletBinding()]
param(
    [string]$CacheDir = 'C:\WhisperECoG_Work\SWPD\matched_pca50_all_cache_v2\sub-01',
    [string]$RunDir = 'C:\WhisperECoG_Work\SWPD\runs\alternating50_sub01_v1',
    [int]$MaximumIterations = 25,
    [int]$Patience = 5
)
$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) { $PSNativeCommandUseErrorActionPreference = $false }
$ModuleRoot = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ModuleRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python environment not found: $Python" }
if (-not (Test-Path -LiteralPath $CacheDir -PathType Container)) { throw "SWPD sub-01 cache not found: $CacheDir" }
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = '1'
$Arguments = @(
    '-u', (Join-Path $ModuleRoot 'run_sub01_alternating.py'),
    '--cache-dir', $CacheDir, '--run-dir', $RunDir,
    '--maximum-iterations', [string]$MaximumIterations, '--patience', [string]$Patience,
    '--targets', 'mel80', 'L3', 'L4', 'L5', 'L345'
)
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "SWPD alternating50 run failed with exit code $LASTEXITCODE" }
