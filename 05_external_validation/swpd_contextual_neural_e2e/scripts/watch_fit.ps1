[CmdletBinding()]
param(
    [string]$LauncherReceipt = 'C:\WhisperECoG_Work\SWPD\runs\contextual_neural_e2e_sub01_v1\launcher\launcher.json',
    [switch]$Follow
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $LauncherReceipt)) {
    throw "Launcher receipt does not exist: $LauncherReceipt"
}
$Receipt = Get-Content -LiteralPath $LauncherReceipt -Raw -Encoding UTF8 | ConvertFrom-Json
$Process = Get-Process -Id ([int]$Receipt.pid) -ErrorAction SilentlyContinue
$Status = if ($Process) { 'RUNNING' } else { 'STOPPED' }
Write-Host "$Status | PID=$($Receipt.pid) | task=$($Receipt.task)"
Write-Host "Started UTC: $($Receipt.started_utc)"
Write-Host "Main log:  $($Receipt.stdout)"
Write-Host "Error log: $($Receipt.stderr)"
$FitSummary = Join-Path $Receipt.output_root 'fit_summary.json'
$FinalSummary = Join-Path $Receipt.output_root 'summary.json'
if (Test-Path -LiteralPath $FitSummary) { Write-Host "FIT COMPLETE | $FitSummary" }
if (Test-Path -LiteralPath $FinalSummary) { Write-Host "TEST COMPLETE | $FinalSummary" }
if (Test-Path -LiteralPath $Receipt.stderr) {
    Write-Host '--- recent stderr ---'
    Get-Content -LiteralPath $Receipt.stderr -Tail 60
}
if (Test-Path -LiteralPath $Receipt.stdout) {
    Write-Host '--- recent output ---'
    if ($Follow -and $Process) {
        Write-Host 'Ctrl+C stops only this watcher; the background process continues.'
        Get-Content -LiteralPath $Receipt.stdout -Tail 160 -Wait
    } else {
        Get-Content -LiteralPath $Receipt.stdout -Tail 160
    }
}
