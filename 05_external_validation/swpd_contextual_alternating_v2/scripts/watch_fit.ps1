[CmdletBinding()]
param(
    [string]$LauncherReceipt = 'C:\WhisperECoG_Work\SWPD\runs\contextual_covariance_alternating_v2_sub01\launcher\fit_launcher.json',
    [switch]$Follow
)
$ErrorActionPreference = 'Stop'
$Receipt = Get-Content -LiteralPath $LauncherReceipt -Raw -Encoding UTF8 | ConvertFrom-Json
$Process = Get-Process -Id ([int]$Receipt.pid) -ErrorAction SilentlyContinue
Write-Host "$(if ($Process) { 'RUNNING' } else { 'STOPPED' }) | PID=$($Receipt.pid) | task=$($Receipt.task)"
Write-Host "Main log:  $($Receipt.stdout)"
Write-Host "Error log: $($Receipt.stderr)"
$Summary = Join-Path $Receipt.output_root 'fit_summary.json'
if (Test-Path -LiteralPath $Summary) { Write-Host "FIT COMPLETE | $Summary | TEST NOT EVALUATED" }
if (Test-Path -LiteralPath $Receipt.stderr) {
    Write-Host '--- stderr ---'
    Get-Content -LiteralPath $Receipt.stderr -Tail 50
}
if (Test-Path -LiteralPath $Receipt.stdout) {
    Write-Host '--- output ---'
    if ($Follow -and $Process) {
        Write-Host 'Ctrl+C stops only this watcher.'
        Get-Content -LiteralPath $Receipt.stdout -Tail 120 -Wait
    } else {
        Get-Content -LiteralPath $Receipt.stdout -Tail 120
    }
}
