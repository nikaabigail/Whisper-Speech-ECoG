#requires -Version 5.1

param(
    [ValidateSet("ivanova", "procenko")]
    [string]$Patient = "ivanova",

    [ValidateSet(3, 4, 5)]
    [int[]]$Layers = @(3, 4, 5),

    [ValidateRange(0, 2147483647)]
    [int]$Seed = 4,

    [ValidateSet("val", "test")]
    [string[]]$Splits = @("val", "test"),

    [ValidateRange(1, 1000)]
    [int]$StepFrames = 1,

    [ValidateRange(0.0, 10000.0)]
    [double]$SmoothMs = 200.0,

    [ValidateSet("centered", "causal")]
    [string]$Smoothing = "centered",

    [ValidateRange(0, 100000)]
    [int]$NullPermutations = 50,

    [switch]$PreflightOnly,
    [switch]$Debug,
    [switch]$NoSaveTimelines,
    [switch]$AllowNonzeroLead,

    [string]$BaseProject = "",
    [string]$Python = "py",
    [string[]]$PythonPrefixArgs = @("-3.10")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($null -eq (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python command was not found: $Python"
}
if ([string]::IsNullOrWhiteSpace($BaseProject)) {
    $BaseProject = Join-Path (Split-Path -Parent $PSScriptRoot) "02_whisper_sync"
}
$BaseProject = [System.IO.Path]::GetFullPath($BaseProject)

foreach ($required in @(
    (Join-Path $BaseProject "library\patients.json"),
    (Join-Path $BaseProject "model_dumps")
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing local input: $required. Patient paths and checkpoints are intentionally not bundled."
    }
}

$Layers = @($Layers | Sort-Object -Unique)
if ($Layers.Count -eq 0) {
    throw "At least one Whisper layer is required."
}
$uniqueSplits = @()
foreach ($split in $Splits) {
    if ($uniqueSplits -notcontains $split) { $uniqueSplits += $split }
}
$Splits = $uniqueSplits
if ($Splits.Count -eq 0) {
    throw "At least one split is required."
}

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$arguments = @(
    "async_replay.py",
    "--base-project", $BaseProject,
    "--patient", $Patient,
    "--seed", "$Seed",
    "--layers"
)
$arguments += @($Layers | ForEach-Object { "$_" })
$arguments += "--splits"
$arguments += $Splits
$arguments += @(
    "--step-frames", "$StepFrames",
    "--smooth-ms", "$SmoothMs",
    "--smoothing", $Smoothing,
    "--null-permutations", "$NullPermutations"
)
if ($PreflightOnly) { $arguments += "--preflight" }
if ($Debug) { $arguments += "--debug" }
if ($NoSaveTimelines) { $arguments += "--no-save-timelines" }
if ($AllowNonzeroLead) { $arguments += "--allow-nonzero-lead" }

Push-Location $PSScriptRoot
try {
    & $Python @PythonPrefixArgs @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Frozen asynchronous replay failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Frozen asynchronous replay completed successfully."
