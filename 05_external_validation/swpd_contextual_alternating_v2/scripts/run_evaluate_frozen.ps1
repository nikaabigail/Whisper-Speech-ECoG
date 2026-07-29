[CmdletBinding()]
param(
    [string]$CacheDir = 'C:\WhisperECoG_Work\SWPD\contextual_whisper_cache_v1\sub-01',
    [string]$ReferenceSummary = 'C:\WhisperECoG_Work\SWPD\runs\contextual_whisper_sub01_v1\summary.json',
    [string]$RunDir = 'C:\WhisperECoG_Work\SWPD\runs\contextual_covariance_alternating_v2_sub01'
)
$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$ModuleRoot = Split-Path $PSScriptRoot -Parent
$ExternalRoot = Split-Path $ModuleRoot -Parent
$Python = Join-Path $ExternalRoot '.venv\Scripts\python.exe'
foreach ($Required in @($Python, $CacheDir, $ReferenceSummary, (Join-Path $RunDir 'fit_summary.json'))) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Missing frozen fit input: $Required" }
}
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = '1'
& $Python -u (Join-Path $ModuleRoot 'evaluate_frozen_sub01.py') `
    --cache-dir $CacheDir `
    --reference-summary $ReferenceSummary `
    --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "Frozen contextual evaluation failed with exit code $LASTEXITCODE" }
