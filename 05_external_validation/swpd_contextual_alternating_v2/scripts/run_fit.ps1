[CmdletBinding()]
param(
    [string]$CacheDir = 'C:\WhisperECoG_Work\SWPD\contextual_whisper_cache_v1\sub-01',
    [string]$ReferenceSummary = 'C:\WhisperECoG_Work\SWPD\runs\contextual_whisper_sub01_v1\summary.json',
    [string]$RunDir = 'C:\WhisperECoG_Work\SWPD\runs\contextual_covariance_alternating_v2_sub01',
    [int]$SearchDim = 128,
    [int]$MaxCycles = 10
)
$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$ModuleRoot = Split-Path $PSScriptRoot -Parent
$ExternalRoot = Split-Path $ModuleRoot -Parent
$Python = Join-Path $ExternalRoot '.venv\Scripts\python.exe'
foreach ($Required in @($Python, $CacheDir, $ReferenceSummary, (Join-Path $ModuleRoot 'fit_select_sub01.py'))) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Missing: $Required" }
}
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = '1'
& $Python (Join-Path $ModuleRoot 'preflight.py')
if ($LASTEXITCODE -ne 0) { throw "Preflight failed with exit code $LASTEXITCODE" }
& $Python -u (Join-Path $ModuleRoot 'fit_select_sub01.py') `
    --cache-dir $CacheDir `
    --reference-summary $ReferenceSummary `
    --run-dir $RunDir `
    --search-dim $SearchDim `
    --max-cycles $MaxCycles
if ($LASTEXITCODE -ne 0) { throw "Contextual alternating fit failed with exit code $LASTEXITCODE" }
