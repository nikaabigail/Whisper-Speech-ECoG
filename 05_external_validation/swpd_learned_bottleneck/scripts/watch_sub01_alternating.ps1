[CmdletBinding()]
param(
    [string]$LauncherReceipt = 'C:\WhisperECoG_Work\SWPD\runs\alternating50_sub01_v1\launcher\launcher.json',
    [switch]$Follow
)
$ErrorActionPreference = 'Stop'
$Receipt = Get-Content -LiteralPath $LauncherReceipt -Raw -Encoding UTF8 | ConvertFrom-Json
$Process = Get-Process -Id ([int]$Receipt.pid) -ErrorAction SilentlyContinue
if ($null -eq $Process) { Write-Host "STOPPED | PID=$($Receipt.pid) | task=$($Receipt.task)" } else { Write-Host "RUNNING | PID=$($Receipt.pid) | task=$($Receipt.task)" }
Write-Host "Main log: $($Receipt.stdout)"
Write-Host "Error log: $($Receipt.stderr)"
$QueueState = Join-Path ([string]$Receipt.output_root) 'queue_state.json'
if (Test-Path -LiteralPath $QueueState) { $Queue=Get-Content -Raw -Encoding UTF8 $QueueState|ConvertFrom-Json; Write-Host "Queue: status=$($Queue.status) current=$($Queue.current_task) completed=$(@($Queue.completed_tasks).Count) remaining=$(@($Queue.remaining_tasks).Count)" }
if (Test-Path -LiteralPath $Receipt.stderr) { Write-Host '--- recent stderr ---'; Get-Content -LiteralPath $Receipt.stderr -Tail 40 }
if (Test-Path -LiteralPath $Receipt.stdout) { Write-Host '--- main output ---'; if ($Follow) { Write-Host 'Ctrl+C stops only this watcher; the background process continues.'; Get-Content -LiteralPath $Receipt.stdout -Tail 100 -Wait } else { Get-Content -LiteralPath $Receipt.stdout -Tail 100 } }
