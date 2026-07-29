[CmdletBinding()]
param(
    [string]$LauncherReceipt = 'C:\WhisperECoG_Work\SWPD\runs\contextual_fixed_q_neural_population_v1\launcher\launcher.json',
    [switch]$Follow
)
$ErrorActionPreference='Stop'
if(-not(Test-Path -LiteralPath $LauncherReceipt)){throw "Launcher receipt missing: $LauncherReceipt"}
$r=Get-Content -LiteralPath $LauncherReceipt -Raw -Encoding UTF8|ConvertFrom-Json;$p=Get-Process -Id ([int]$r.pid) -ErrorAction SilentlyContinue
Write-Host "$(if($p){'RUNNING'}else{'STOPPED'}) | PID=$($r.pid) | task=$($r.task)";Write-Host "Main log:  $($r.stdout)";Write-Host "Error log: $($r.stderr)"
if((Test-Path -LiteralPath $r.stderr)-and(Get-Item -LiteralPath $r.stderr).Length-gt 0){Write-Host '--- stderr ---';Get-Content -LiteralPath $r.stderr -Tail 80 -Encoding UTF8}
Write-Host '--- output ---';if($Follow -and $p){Write-Host 'Ctrl+C stops only this watcher.';Get-Content -LiteralPath $r.stdout -Tail 100 -Wait -Encoding UTF8}else{Get-Content -LiteralPath $r.stdout -Tail 100 -Encoding UTF8}
