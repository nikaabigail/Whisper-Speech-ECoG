[CmdletBinding()]
param(
    [string]$DataRoot = 'C:\WhisperECoG\SWPD\extracted',
    [string]$CacheRoot = 'C:\WhisperECoG_Work\SWPD\matched_pca50_all_cache_v2',
    [string]$RunRoot = 'C:\WhisperECoG_Work\SWPD\runs\matched_pca50_all_v2',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$Wrapper = Join-Path $PSScriptRoot 'run_swpd_all_matched.ps1'
$ResolvedRun = [System.IO.Path]::GetFullPath($RunRoot).TrimEnd('\')
$LauncherDirectory = Join-Path $ResolvedRun 'launcher'
New-Item -ItemType Directory -Path $LauncherDirectory -Force | Out-Null
$ReceiptPath = Join-Path $LauncherDirectory 'launcher.json'
if (Test-Path -LiteralPath $ReceiptPath -PathType Leaf) {
    $Existing = Get-Content -LiteralPath $ReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (Get-Process -Id ([int]$Existing.pid) -ErrorAction SilentlyContinue) {
        throw "A recorded process is still running with PID $($Existing.pid)."
    }
    $ArchiveStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    Move-Item -LiteralPath $ReceiptPath -Destination (
        Join-Path $LauncherDirectory "launcher_previous_$ArchiveStamp.json"
    )
}

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Stdout = Join-Path $LauncherDirectory "swpd_matched_all_$Stamp.out.log"
$Stderr = Join-Path $LauncherDirectory "swpd_matched_all_$Stamp.err.log"
New-Item -ItemType File -Path $Stdout -Force | Out-Null
New-Item -ItemType File -Path $Stderr -Force | Out-Null
foreach ($Value in @($Wrapper, $DataRoot, $CacheRoot, $RunRoot)) {
    if ($Value.Contains('"')) { throw 'Double quotes are not allowed in launch paths.' }
}
$Arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$Wrapper`"",
    '-DataRoot', "`"$DataRoot`"",
    '-CacheRoot', "`"$CacheRoot`"",
    '-RunRoot', "`"$RunRoot`"",
    '-Device', $Device
)
$Process = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList $Arguments `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru
$Receipt = [ordered]@{
    schema_version = 1
    kind = 'external_validation_background_launcher'
    task = 'swpd_all_subjects_matched_pca50_v2'
    pid = $Process.Id
    started_utc = [DateTime]::UtcNow.ToString('o')
    data_root = [System.IO.Path]::GetFullPath($DataRoot)
    output_root = $ResolvedRun
    cache_root = [System.IO.Path]::GetFullPath($CacheRoot)
    stdout = $Stdout
    stderr = $Stderr
}
$Temporary = "$ReceiptPath.partial"
$Receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Temporary -Encoding UTF8
Move-Item -LiteralPath $Temporary -Destination $ReceiptPath
Write-Host "Started hidden background process PID=$($Process.Id)"
Write-Host "Launcher receipt: $ReceiptPath"
Write-Host 'Ctrl+C in the watcher will not stop the experiment.'
