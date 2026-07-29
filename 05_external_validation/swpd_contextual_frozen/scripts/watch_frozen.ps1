[CmdletBinding()]
param(
    [string]$LauncherReceipt = 'C:\WhisperECoG_Work\SWPD\runs\contextual_l4_frozen_confirmatory_v1\launcher\launcher.json',
    [switch]$Follow
)
$ErrorActionPreference = 'Stop'
$Receipt = Get-Content -LiteralPath $LauncherReceipt -Raw -Encoding UTF8 | ConvertFrom-Json
$Process = Get-Process -Id ([int]$Receipt.pid) -ErrorAction SilentlyContinue
if ($null -eq $Process) { Write-Host "STOPPED | PID=$($Receipt.pid) | task=$($Receipt.task)" }
else { Write-Host "RUNNING | PID=$($Receipt.pid) | task=$($Receipt.task)" }
Write-Host "Started UTC: $($Receipt.started_utc)"
Write-Host "Main log:   $($Receipt.stdout)"
Write-Host "Error log:  $($Receipt.stderr)"
$Summary = Join-Path ([string]$Receipt.output_root) 'summary\population_summary.json'
if (Test-Path -LiteralPath $Summary -PathType Leaf) { Write-Host "LATEST SUMMARY | $Summary" }
$Queue = Join-Path ([string]$Receipt.output_root) 'queue_state.json'
if (Test-Path -LiteralPath $Queue -PathType Leaf) {
    $State = Get-Content -LiteralPath $Queue -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host "STATE=$($State.status) | current=$($State.current_subject) | completed=$(@($State.completed_subjects).Count)/8"
}
if (Test-Path -LiteralPath $Receipt.stderr -PathType Leaf) {
    Write-Host '--- recent stderr ---'; Get-Content -LiteralPath $Receipt.stderr -Tail 50
}
if (Test-Path -LiteralPath $Receipt.stdout -PathType Leaf) {
    Write-Host '--- main output ---'
    if ($Follow) {
        Write-Host 'Ctrl+C stops only this watcher; the background process continues.'
        Get-Content -LiteralPath $Receipt.stdout -Tail 140 -Wait
    } else { Get-Content -LiteralPath $Receipt.stdout -Tail 140 }
}
