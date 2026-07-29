[CmdletBinding()]
param(
    [string]$CacheDir = 'C:\WhisperECoG_Work\SWPD\matched_pca50_all_cache_v2\sub-01',
    [string]$RunDir = 'C:\WhisperECoG_Work\SWPD\runs\clip50_sub01_v1',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [int]$Seed = 4,
    [int]$MaximumEpochs = 120,
    [int]$Patience = 18,
    [int]$BatchSize = 64
)

$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$ModuleRoot = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ModuleRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $CacheDir -PathType Container)) {
    throw "SWPD sub-01 cache not found: $CacheDir"
}
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& $Python (Join-Path $ModuleRoot 'clip_preflight.py') --device $Device
if ($LASTEXITCODE -ne 0) {
    throw "Python/PyTorch/CUDA preflight failed with exit code $LASTEXITCODE"
}
$Arguments = @(
    '-u', (Join-Path $ModuleRoot 'run_sub01_clip.py'),
    '--cache-dir', $CacheDir,
    '--run-dir', $RunDir,
    '--device', $Device,
    '--seed', [string]$Seed,
    '--maximum-epochs', [string]$MaximumEpochs,
    '--patience', [string]$Patience,
    '--batch-size', [string]$BatchSize,
    '--targets', 'mel80', 'L3', 'L4', 'L5', 'L345'
)
& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "SWPD sub-01 CLIP50 run failed with exit code $LASTEXITCODE"
}
