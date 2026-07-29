[CmdletBinding()]
param(
    [string]$DataRoot = 'C:\WhisperECoG\SWPD\extracted',
    [string]$CacheRoot = 'C:\WhisperECoG_Work\SWPD\contextual_l4_frozen_cache_v1',
    [string]$RunRoot = 'C:\WhisperECoG_Work\SWPD\runs\contextual_l4_frozen_confirmatory_v1',
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [int]$ChannelBatchSize = 16
)
$ErrorActionPreference = 'Stop'
$Wrapper = Join-Path $PSScriptRoot 'run_frozen.ps1'
$ResolvedRun = [System.IO.Path]::GetFullPath($RunRoot).TrimEnd('\')
$LauncherDirectory = Join-Path $ResolvedRun 'launcher'
New-Item -ItemType Directory -Path $LauncherDirectory -Force | Out-Null
$ReceiptPath = Join-Path $LauncherDirectory 'launcher.json'
if (Test-Path -LiteralPath $ReceiptPath -PathType Leaf) {
    $Existing = Get-Content -LiteralPath $ReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (Get-Process -Id ([int]$Existing.pid) -ErrorAction SilentlyContinue) {
        throw "A recorded frozen process is still running with PID $($Existing.pid)."
    }
    Move-Item -LiteralPath $ReceiptPath -Destination (Join-Path $LauncherDirectory "launcher_previous_$(Get-Date -Format 'yyyyMMdd_HHmmss').json")
}
foreach ($Value in @($Wrapper, $DataRoot, $CacheRoot, $ResolvedRun)) {
    if ($Value.Contains('"')) { throw 'Double quotes are not allowed in launch paths.' }
}
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Stdout = Join-Path $LauncherDirectory "frozen_contextual_$Stamp.out.log"
$Stderr = Join-Path $LauncherDirectory "frozen_contextual_$Stamp.err.log"
New-Item -ItemType File -Path $Stdout -Force | Out-Null
New-Item -ItemType File -Path $Stderr -Force | Out-Null
$Arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$Wrapper`"",
    '-DataRoot', "`"$DataRoot`"", '-CacheRoot', "`"$CacheRoot`"", '-RunRoot', "`"$ResolvedRun`"",
    '-Device', $Device, '-ChannelBatchSize', $ChannelBatchSize
)
$Process = Start-Process -FilePath 'powershell.exe' -ArgumentList $Arguments `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Receipt = [ordered]@{
    schema_version = 1; kind = 'external_validation_background_launcher'
    task = 'swpd_contextual_l4_frozen_confirmatory_v1'; pid = $Process.Id
    started_utc = [DateTime]::UtcNow.ToString('o')
    data_root = [System.IO.Path]::GetFullPath($DataRoot)
    cache_root = [System.IO.Path]::GetFullPath($CacheRoot); output_root = $ResolvedRun
    stdout = $Stdout; stderr = $Stderr
}
$Temporary = "$ReceiptPath.partial"
$Receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Temporary -Encoding UTF8
Move-Item -LiteralPath $Temporary -Destination $ReceiptPath
Write-Host "Started hidden frozen process PID=$($Process.Id)"
Write-Host "Launcher receipt: $ReceiptPath"
Write-Host 'Ctrl+C in the watcher will not stop the experiment.'
