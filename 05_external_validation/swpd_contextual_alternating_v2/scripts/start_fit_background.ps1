[CmdletBinding()]
param(
    [string]$CacheDir = 'C:\WhisperECoG_Work\SWPD\contextual_whisper_cache_v1\sub-01',
    [string]$ReferenceSummary = 'C:\WhisperECoG_Work\SWPD\runs\contextual_whisper_sub01_v1\summary.json',
    [string]$RunDir = 'C:\WhisperECoG_Work\SWPD\runs\contextual_covariance_alternating_v2_sub01',
    [int]$SearchDim = 128,
    [int]$MaxCycles = 10
)
$ErrorActionPreference = 'Stop'
$Wrapper = Join-Path $PSScriptRoot 'run_fit.ps1'
$ResolvedRun = [IO.Path]::GetFullPath($RunDir).TrimEnd('\')
$Launcher = Join-Path $ResolvedRun 'launcher'
New-Item -ItemType Directory -Path $Launcher -Force | Out-Null
$ReceiptPath = Join-Path $Launcher 'fit_launcher.json'
if (Test-Path -LiteralPath $ReceiptPath) {
    $Old = Get-Content -LiteralPath $ReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (Get-Process -Id ([int]$Old.pid) -ErrorAction SilentlyContinue) {
        throw "Fit process $($Old.pid) is already running"
    }
    Move-Item -LiteralPath $ReceiptPath -Destination (Join-Path $Launcher "fit_launcher_previous_$(Get-Date -Format yyyyMMdd_HHmmss).json")
}
$Stamp = Get-Date -Format yyyyMMdd_HHmmss
$Out = Join-Path $Launcher "contextual_alt_v2_fit_$Stamp.out.log"
$Err = Join-Path $Launcher "contextual_alt_v2_fit_$Stamp.err.log"
New-Item -ItemType File -Path $Out -Force | Out-Null
New-Item -ItemType File -Path $Err -Force | Out-Null
$Arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$Wrapper`"",
    '-CacheDir', "`"$CacheDir`"",
    '-ReferenceSummary', "`"$ReferenceSummary`"",
    '-RunDir', "`"$ResolvedRun`"",
    '-SearchDim', $SearchDim,
    '-MaxCycles', $MaxCycles
)
$Process = Start-Process powershell.exe -ArgumentList $Arguments `
    -RedirectStandardOutput $Out -RedirectStandardError $Err `
    -WindowStyle Hidden -PassThru
$Receipt = [ordered]@{
    schema_version = 1
    task = 'swpd_sub01_contextual_covariance_alternating_v2_fit_only'
    pid = $Process.Id
    started_utc = [DateTime]::UtcNow.ToString('o')
    output_root = $ResolvedRun
    stdout = $Out
    stderr = $Err
    test_evaluation = $false
}
$Partial = "$ReceiptPath.partial"
$Receipt | ConvertTo-Json | Set-Content -LiteralPath $Partial -Encoding UTF8
Move-Item -LiteralPath $Partial -Destination $ReceiptPath
Write-Host "Started hidden fit-only process PID=$($Process.Id)"
Write-Host "Launcher receipt: $ReceiptPath"
Write-Host 'Ctrl+C in watcher does not stop fitting. This stage never evaluates test.'
