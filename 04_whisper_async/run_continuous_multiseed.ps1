#requires -Version 5.1

[CmdletBinding()]
param(
    [int[]]$Seeds = @(1, 2, 3, 4, 42),

    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")]
    [string]$RunPrefix = "continuous_multiseed_v1",

    [ValidateRange(1, 1000)]
    [int]$Epochs = 30,

    [ValidateRange(1, 1000)]
    [int]$MinEpochs = 5,

    [ValidateRange(1, 1000)]
    [int]$Patience = 5,

    [ValidateRange(1, 100000)]
    [int]$BatchSize = 128,

    [switch]$VerifyCacheSha,
    [switch]$PreflightOnly,
    [switch]$AllowCpu,

    [string]$BaseProject = "",
    [string]$Archive = "",
    [string]$CacheRoot = "",
    [string]$RunsRoot = "",
    [string]$SummaryRoot = "",

    [string]$Python = "py",
    [string[]]$PythonPrefixArgs = @("-3.10")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Stage
    )

    Write-Host "`n===== $Stage ====="
    & $Python @PythonPrefixArgs @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE. Repeat the same command to resume completed cache/head stages."
    }
}

if ($MinEpochs -gt $Epochs) {
    throw "MinEpochs cannot exceed Epochs."
}
if ($null -eq (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python command was not found: $Python"
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($BaseProject)) {
    $BaseProject = Join-Path $repositoryRoot "02_whisper_sync"
}
if ([string]::IsNullOrWhiteSpace($Archive)) {
    $Archive = Join-Path $repositoryRoot "checkpoints\frozen_seed4"
}
if ([string]::IsNullOrWhiteSpace($CacheRoot)) {
    $CacheRoot = Join-Path $repositoryRoot "artifacts\async_hidden_cache"
}
if ([string]::IsNullOrWhiteSpace($RunsRoot)) {
    $RunsRoot = Join-Path $repositoryRoot "artifacts\continuous_multiseed_runs"
}
if ([string]::IsNullOrWhiteSpace($SummaryRoot)) {
    $SummaryRoot = Join-Path $repositoryRoot "artifacts\continuous_multiseed_summaries"
}

$BaseProject = [System.IO.Path]::GetFullPath($BaseProject)
$Archive = [System.IO.Path]::GetFullPath($Archive)
$CacheRoot = [System.IO.Path]::GetFullPath($CacheRoot)
$RunsRoot = [System.IO.Path]::GetFullPath($RunsRoot)
$SummaryRoot = [System.IO.Path]::GetFullPath($SummaryRoot)

$trainer = Join-Path $PSScriptRoot "train_continuous_heads.py"
$cacheBuilder = Join-Path $PSScriptRoot "build_continuous_cache.py"
$summarizer = Join-Path $PSScriptRoot "summarize_continuous_multiseed.py"

foreach ($requiredFile in @(
    $trainer,
    $cacheBuilder,
    $summarizer,
    (Join-Path $BaseProject "library\patients.json"),
    (Join-Path $Archive "MANIFEST.sha256.json")
)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw (
            "Missing required local input: $requiredFile`n" +
            "Raw data and the frozen seed-4 archive are intentionally not bundled; see checkpoints\release_manifest.json."
        )
    }
}

$Seeds = @($Seeds | Sort-Object -Unique)
if ($Seeds.Count -eq 0) {
    throw "At least one continuous-head seed is required."
}
if ($Seeds.Count -lt 2 -and -not $PreflightOnly) {
    throw "At least two distinct seeds are required for multiseed statistics."
}
foreach ($seed in $Seeds) {
    if ($seed -lt 0) { throw "Seeds must be non-negative: $seed" }
}

New-Item -ItemType Directory -Path $CacheRoot -Force | Out-Null
New-Item -ItemType Directory -Path $RunsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $SummaryRoot -Force | Out-Null

$env:BENCH_SEED = "4"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$common = @(
    "--base-project", $BaseProject,
    "--archive", $Archive,
    "--cache-root", $CacheRoot
)
$trainValFiles = @(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
$testFiles = @(10, 11)

if ($PreflightOnly) {
    $cachePreflight = @($cacheBuilder) + $common + @(
        "--layers", "3", "4", "5", "--files"
    )
    $cachePreflight += @($trainValFiles | ForEach-Object { "$_" })
    $cachePreflight += "--preflight"
    Invoke-PythonChecked -Arguments $cachePreflight -Stage "source/archive/cache preflight"

    $probe = @($trainer) + $common + @(
        "--runs-root", $RunsRoot,
        "--run-name", "${RunPrefix}_preflight",
        "--source-seed", "4",
        "--seed", "$($Seeds[0])",
        "--preflight"
    )
    Invoke-PythonChecked -Arguments $probe -Stage "continuous-head trainer preflight"
    Write-Host "Preflight completed; no head was trained and no test cache was opened."
    exit 0
}

if (-not $AllowCpu) {
    $cudaProbe = @(
        "-c",
        'import sys, torch; print("Torch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); sys.exit(0 if torch.cuda.is_available() else 1)'
    )
    Invoke-PythonChecked -Arguments $cudaProbe -Stage "CUDA check"
}

# Phase 1: build only training and validation features. Held-out recordings
# remain unopened until every requested seed has validation-fixed L3/L4/L5 heads.
$trainCache = @($cacheBuilder) + $common + @("--layers", "3", "4", "5", "--files")
$trainCache += @($trainValFiles | ForEach-Object { "$_" })
if ($VerifyCacheSha) { $trainCache += "--verify-existing-sha" }
Invoke-PythonChecked -Arguments $trainCache -Stage "shared train/validation hidden cache"

$trainerBase = @($trainer) + $common + @(
    "--runs-root", $RunsRoot,
    "--source-seed", "4",
    "--epochs", "$Epochs",
    "--min-epochs", "$MinEpochs",
    "--patience", "$Patience",
    "--batch-size", "$BatchSize"
)
if ($VerifyCacheSha) { $trainerBase += "--verify-cache-sha" }
if ($AllowCpu) { $trainerBase += "--allow-cpu" }

foreach ($seed in $Seeds) {
    $runName = "${RunPrefix}_seed${seed}"
    $arguments = $trainerBase + @(
        "--run-name", $runName,
        "--seed", "$seed",
        "--train-only"
    )
    Invoke-PythonChecked -Arguments $arguments -Stage "head seed $seed training (test gate closed)"
}

# Phase 2: only after all training stages returned zero may test features be built.
$testCache = @($cacheBuilder) + $common + @("--layers", "3", "4", "5", "--files")
$testCache += @($testFiles | ForEach-Object { "$_" })
if ($VerifyCacheSha) { $testCache += "--verify-existing-sha" }
Invoke-PythonChecked -Arguments $testCache -Stage "held-out test hidden cache"

# Phase 3: reuse each fixed head and evaluate once. Re-running is resumable.
foreach ($seed in $Seeds) {
    $runName = "${RunPrefix}_seed${seed}"
    $arguments = $trainerBase + @("--run-name", $runName, "--seed", "$seed")
    Invoke-PythonChecked -Arguments $arguments -Stage "head seed $seed fixed-test evaluation"
}

$summary = @(
    $summarizer,
    "--runs-root", $RunsRoot,
    "--run-prefix", $RunPrefix,
    "--seeds"
)
$summary += @($Seeds | ForEach-Object { "$_" })
$summary += @("--output-dir", $SummaryRoot)
Invoke-PythonChecked -Arguments $summary -Stage "continuous-head multiseed summary"

Write-Host "`nContinuous multiseed completed successfully."
Write-Host "Runs: $RunsRoot"
Write-Host "Summaries: $SummaryRoot"
