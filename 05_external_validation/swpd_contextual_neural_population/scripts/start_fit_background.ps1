[CmdletBinding()]
param(
    [string]$CacheRoot = 'C:\WhisperECoG_Work\SWPD\contextual_l4_frozen_cache_v1',
    [string]$RunRoot = 'C:\WhisperECoG_Work\SWPD\runs\contextual_fixed_q_neural_population_v1',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [switch]$Diagnostic
)
$ErrorActionPreference='Stop'
$Wrapper=Join-Path $PSScriptRoot 'run_fit.ps1'; $Resolved=[IO.Path]::GetFullPath($RunRoot).TrimEnd('\')
$Launcher=Join-Path $Resolved 'launcher'; New-Item -ItemType Directory -Path $Launcher -Force | Out-Null
$ReceiptPath=Join-Path $Launcher 'launcher.json'
if(Test-Path -LiteralPath $ReceiptPath){$old=Get-Content -LiteralPath $ReceiptPath -Raw -Encoding UTF8|ConvertFrom-Json;if(Get-Process -Id ([int]$old.pid) -ErrorAction SilentlyContinue){throw "Run already active PID=$($old.pid)"};Move-Item -LiteralPath $ReceiptPath -Destination (Join-Path $Launcher "launcher_previous_$(Get-Date -Format yyyyMMdd_HHmmss).json")}
$stamp=Get-Date -Format yyyyMMdd_HHmmss; $stdout=Join-Path $Launcher "population_fit_$stamp.out.log"; $stderr=Join-Path $Launcher "population_fit_$stamp.err.log"
New-Item -ItemType File -Path $stdout -Force|Out-Null; New-Item -ItemType File -Path $stderr -Force|Out-Null
$Arguments=@('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$Wrapper`"",'-CacheRoot',"`"$CacheRoot`"",'-RunRoot',"`"$Resolved`"",'-Device',$Device);if($Diagnostic){$Arguments+='-Diagnostic'}
$Process=Start-Process powershell.exe -ArgumentList $Arguments -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$Receipt=[ordered]@{schema_version=1;task='swpd_fixed_q_neural_population_fit_only';pid=$Process.Id;started_utc=[DateTime]::UtcNow.ToString('o');run_root=$Resolved;stdout=$stdout;stderr=$stderr;subjects=@('sub-02','sub-03','sub-04','sub-05','sub-06','sub-07','sub-08','sub-09');seeds=@(1,2,3,4,42);folds=@(0,1,2,3,4);diagnostic=[bool]$Diagnostic}
$temp="$ReceiptPath.partial";$Receipt|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $temp -Encoding UTF8;Move-Item -LiteralPath $temp -Destination $ReceiptPath
Write-Host "Started hidden population fit PID=$($Process.Id)";Write-Host "Launcher receipt: $ReceiptPath";Write-Host 'Ctrl+C in watcher does not stop training.'
