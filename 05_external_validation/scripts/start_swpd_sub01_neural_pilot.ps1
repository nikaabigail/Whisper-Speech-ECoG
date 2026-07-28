[CmdletBinding()]
param(
    [string]$DataRoot = "C:\WhisperECoG\SWPD\extracted",
    [string]$CacheDir = "C:\WhisperECoG_Work\SWPD\cache_1000hz",
    [string]$RunDir = "C:\WhisperECoG_Work\SWPD\runs\seed4_v1",
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [switch]$FastSmoke,
    [switch]$SingleMelDevelopment
)

$ErrorActionPreference = "Stop"
$Wrapper = Join-Path $PSScriptRoot "run_swpd_sub01_neural_pilot.ps1"
$ResolvedRun = [System.IO.Path]::GetFullPath($RunDir).TrimEnd('\')
$LauncherDirectory = Join-Path $ResolvedRun "launcher"
New-Item -ItemType Directory -Path $LauncherDirectory -Force | Out-Null
$ReceiptPath = Join-Path $LauncherDirectory "launcher.json"
if (Test-Path -LiteralPath $ReceiptPath -PathType Leaf) {
    $Existing = Get-Content -LiteralPath $ReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (Get-Process -Id ([int]$Existing.pid) -ErrorAction SilentlyContinue) {
        throw "A recorded process is still running with PID $($Existing.pid)."
    }
    $ArchiveStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Move-Item -LiteralPath $ReceiptPath -Destination (
        Join-Path $LauncherDirectory "launcher_previous_$ArchiveStamp.json"
    )
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $LauncherDirectory "swpd_$Stamp.out.log"
$Stderr = Join-Path $LauncherDirectory "swpd_$Stamp.err.log"
New-Item -ItemType File -Path $Stdout -Force | Out-Null
New-Item -ItemType File -Path $Stderr -Force | Out-Null
foreach ($Value in @($Wrapper, $DataRoot, $CacheDir, $RunDir)) {
    if ($Value.Contains('"')) { throw "Double quotes are not allowed in launch paths." }
}
$Arguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$Wrapper`"",
    "-DataRoot", "`"$DataRoot`"",
    "-CacheDir", "`"$CacheDir`"",
    "-RunDir", "`"$RunDir`"",
    "-Device", $Device
)
if ($FastSmoke) { $Arguments += "-FastSmoke" }
if ($SingleMelDevelopment) { $Arguments += "-SingleMelDevelopment" }

$Process = Start-Process -FilePath "powershell.exe" `
    -ArgumentList $Arguments `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru
$Receipt = [ordered]@{
    schema_version = 1
    kind = "external_validation_background_launcher"
    task = "swpd_sub01_neural_development"
    pid = $Process.Id
    started_utc = [DateTime]::UtcNow.ToString("o")
    data_root = [System.IO.Path]::GetFullPath($DataRoot)
    output_root = $ResolvedRun
    cache_root = [System.IO.Path]::GetFullPath($CacheDir)
    fast_smoke = [bool]$FastSmoke
    stdout = $Stdout
    stderr = $Stderr
}
$Temporary = "$ReceiptPath.partial"
$Receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Temporary -Encoding UTF8
Move-Item -LiteralPath $Temporary -Destination $ReceiptPath
Write-Host "Started hidden background process PID=$($Process.Id)"
Write-Host "Launcher receipt: $ReceiptPath"
Write-Host "Use watch_background_run.ps1; Ctrl+C there does not stop training."
