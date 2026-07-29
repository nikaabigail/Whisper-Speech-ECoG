[CmdletBinding()]
param(
    [string]$DataRoot = 'C:\WhisperECoG\SWPD\extracted',
    [string]$CacheRoot = 'C:\WhisperECoG_Work\SWPD\contextual_l4_frozen_cache_v1',
    [string]$RunRoot = 'C:\WhisperECoG_Work\SWPD\runs\contextual_l4_frozen_confirmatory_v1',
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [int]$ChannelBatchSize = 16
)
$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$ModuleRoot = Split-Path $PSScriptRoot -Parent
$ExternalRoot = Split-Path $ModuleRoot -Parent
$Python = Join-Path $ExternalRoot '.venv\Scripts\python.exe'
$Preflight = Join-Path $ModuleRoot 'preflight.py'
$Runner = Join-Path $ModuleRoot 'run_frozen.py'
foreach ($Required in @($Python, $Preflight, $Runner, $DataRoot)) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Required path is missing: $Required" }
}
if ($ChannelBatchSize -le 0) { throw 'ChannelBatchSize must be positive.' }
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = '1'
& $Python $Preflight --device $Device
if ($LASTEXITCODE -ne 0) { throw "Frozen contextual preflight failed with exit code $LASTEXITCODE" }
Write-Host ('=' * 78)
Write-Host 'FROZEN contextual extension | SWPD sub-02..sub-09'
Write-Host 'Control: direct MEL80 | selected system: Whisper L4 train-only PCA50'
Write-Host "Persistent cache: $CacheRoot"
Write-Host "Run directory:    $RunRoot"
Write-Host ('=' * 78)
& $Python -u $Runner --data-root $DataRoot --cache-root $CacheRoot --run-root $RunRoot `
    --device $Device --channel-batch-size $ChannelBatchSize
if ($LASTEXITCODE -ne 0) { throw "Frozen contextual run failed with exit code $LASTEXITCODE" }
