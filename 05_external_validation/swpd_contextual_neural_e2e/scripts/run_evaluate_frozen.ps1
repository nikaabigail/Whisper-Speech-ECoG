[CmdletBinding()]
param(
    [string]$CacheDir = 'C:\WhisperECoG_Work\SWPD\contextual_whisper_cache_v1\sub-01',
    [string]$ReferenceSummary = 'C:\WhisperECoG_Work\SWPD\runs\contextual_whisper_sub01_v1\summary.json',
    [string]$RunDir = 'C:\WhisperECoG_Work\SWPD\runs\contextual_neural_e2e_sub01_v1',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$ModuleRoot = Split-Path $PSScriptRoot -Parent
$ExternalRoot = Split-Path $ModuleRoot -Parent
$Python = Join-Path $ExternalRoot '.venv\Scripts\python.exe'
foreach ($Required in @($Python, $CacheDir, $ReferenceSummary, $RunDir, (Join-Path $ModuleRoot 'evaluate_frozen_sub01.py'))) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Missing required path: $Required" }
}
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& $Python -I (Join-Path $ModuleRoot 'preflight.py') `
    --device $Device `
    --cache-dir $CacheDir `
    --reference-summary $ReferenceSummary
if ($LASTEXITCODE -ne 0) { throw "Preflight failed with exit code $LASTEXITCODE" }
& $Python -I -u (Join-Path $ModuleRoot 'evaluate_frozen_sub01.py') `
    --cache-dir $CacheDir `
    --reference-summary $ReferenceSummary `
    --run-dir $RunDir `
    --device $Device
if ($LASTEXITCODE -ne 0) { throw "Frozen contextual neural E2E evaluation failed with exit code $LASTEXITCODE" }
