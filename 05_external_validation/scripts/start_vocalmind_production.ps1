[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [Parameter(Mandatory = $true)][string]$CacheRoot,
    [ValidateSet("cuda")][string]$Device = "cuda",
    [ValidateRange(0, 100)][int]$MaxEpochsThisCall = 0
)

$ErrorActionPreference = "Stop"
$Wrapper = Join-Path $PSScriptRoot "run_vocalmind_production.ps1"
$ResolvedOutput = [System.IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$LauncherDirectory = "$ResolvedOutput.launcher"
New-Item -ItemType Directory -Path $LauncherDirectory -Force | Out-Null
$ReceiptPath = Join-Path $LauncherDirectory "launcher.json"
if (Test-Path -LiteralPath $ReceiptPath -PathType Leaf) {
    $Existing = Get-Content -LiteralPath $ReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (Get-Process -Id ([int]$Existing.pid) -ErrorAction SilentlyContinue) {
        throw "A recorded process is still running with PID $($Existing.pid)."
    }
    $ArchiveStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ArchivedReceipt = Join-Path $LauncherDirectory "launcher_previous_$ArchiveStamp.json"
    Move-Item -LiteralPath $ReceiptPath -Destination $ArchivedReceipt
    Write-Host "Previous stopped launcher receipt preserved: $ArchivedReceipt"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $LauncherDirectory "vocalmind_$Stamp.out.log"
$Stderr = Join-Path $LauncherDirectory "vocalmind_$Stamp.err.log"
New-Item -ItemType File -Path $Stdout -Force | Out-Null
New-Item -ItemType File -Path $Stderr -Force | Out-Null

foreach ($Value in @($Wrapper, $DataRoot, $OutputRoot, $CacheRoot)) {
    if ($Value.Contains('"')) { throw "Double quotes are not allowed in launch paths." }
}
$Arguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$Wrapper`"",
    "-DataRoot", "`"$DataRoot`"",
    "-OutputRoot", "`"$OutputRoot`"",
    "-CacheRoot", "`"$CacheRoot`"",
    "-Device", $Device
)
if ($MaxEpochsThisCall -gt 0) {
    $Arguments += @("-MaxEpochsThisCall", [string]$MaxEpochsThisCall)
}

$Process = Start-Process -FilePath "powershell.exe" `
    -ArgumentList $Arguments `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru
$Receipt = [ordered]@{
    schema_version = 1
    kind = "external_validation_background_launcher"
    task = "vocalmind_production"
    pid = $Process.Id
    started_utc = [DateTime]::UtcNow.ToString("o")
    data_root = [System.IO.Path]::GetFullPath($DataRoot)
    output_root = $ResolvedOutput
    cache_root = [System.IO.Path]::GetFullPath($CacheRoot)
    max_epochs_this_call = $MaxEpochsThisCall
    stdout = $Stdout
    stderr = $Stderr
}
$Temporary = "$ReceiptPath.partial"
$Receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Temporary -Encoding UTF8
Move-Item -LiteralPath $Temporary -Destination $ReceiptPath
Write-Host "Started hidden background process PID=$($Process.Id)"
Write-Host "Launcher receipt: $ReceiptPath"
Write-Host "Watching can be stopped with Ctrl+C without stopping training."
