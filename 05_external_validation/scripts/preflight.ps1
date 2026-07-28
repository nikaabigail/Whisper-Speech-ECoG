[CmdletBinding()]
param(
    [ValidateSet("swpd", "vocalmind", "all")]
    [Parameter(Mandatory = $true)]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [string]$CacheRoot,

    [string]$OutputRoot,

    [switch]$AllowCpu
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found. Run scripts\bootstrap_windows.ps1 first."
}

$arguments = @(
    "-m", "whisper_ecog_ext.preflight",
    "--dataset", $Dataset,
    "--data-root", $DataRoot,
    "--json-out", (Join-Path $projectRoot "artifacts\host_preflight.json")
)
if (-not [string]::IsNullOrWhiteSpace($CacheRoot)) {
    $env:HF_HOME = [System.IO.Path]::GetFullPath($CacheRoot)
    $arguments += @("--cache-root", $env:HF_HOME)
}
if (-not [string]::IsNullOrWhiteSpace($OutputRoot)) {
    $arguments += @("--output-root", $OutputRoot)
}
if ($AllowCpu) { $arguments += "--allow-cpu" }
& $python @arguments
exit $LASTEXITCODE
