#requires -Version 5.1

param(
    [ValidateSet("ivanova", "procenko")]
    [string]$Patient = "ivanova",

    [ValidateSet(3, 4, 5)]
    [int[]]$Layers = @(3, 4, 5),

    [ValidateRange(0, 2147483647)]
    [int]$Seed = 4,

    [switch]$FullTest,
    [switch]$Debug,

    [string]$SyncRoot = "",
    [string]$Python = "py",
    [string[]]$PythonPrefixArgs = @("-3.10")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($null -eq (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python command was not found: $Python"
}
if ([string]::IsNullOrWhiteSpace($SyncRoot)) {
    $SyncRoot = Join-Path (Split-Path -Parent $PSScriptRoot) "02_whisper_sync"
}
$SyncRoot = [System.IO.Path]::GetFullPath($SyncRoot)

$patients = Join-Path $SyncRoot "library\patients.json"
$modelDumps = Join-Path $SyncRoot "model_dumps"
if (-not (Test-Path -LiteralPath $patients -PathType Leaf)) {
    throw "Missing local patient configuration: $patients"
}
if (-not (Test-Path -LiteralPath $modelDumps -PathType Container)) {
    throw "Missing local checkpoints: $modelDumps. Train or restore matching L3/L4/L5 pairs first."
}

$Layers = @($Layers | Sort-Object -Unique)
if ($Layers.Count -ne 3 -or $Layers[0] -ne 3 -or $Layers[1] -ne 4 -or $Layers[2] -ne 5) {
    throw "The release evaluator is intentionally fixed to Layers 3,4,5."
}

$env:OSSADTCHI_SYNC_ROOT = $SyncRoot
$env:BENCH_SEED = "$Seed"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$arguments = @("ensemble_layers.py", $Patient, "--layers")
$arguments += @($Layers | ForEach-Object { "$_" })
$arguments += @("--seed", "$Seed")
if ($FullTest) { $arguments += "--full" }
if ($Debug) { $arguments += "--debug" }

Push-Location $PSScriptRoot
try {
    & $Python @PythonPrefixArgs @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Ensemble evaluation failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Ensemble evaluation completed successfully."
