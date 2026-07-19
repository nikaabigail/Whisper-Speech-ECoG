#requires -Version 5.1

param(
    [ValidateSet("ivanova", "procenko")]
    [string]$Patient = "ivanova",

    [ValidateSet(3, 4, 5)]
    [int]$Layer = 4,

    [ValidateRange(0, 2147483647)]
    [int]$Seed = 4,

    [ValidateRange(1, 100)]
    [int]$RunsCount = 1,

    [switch]$SkipRegression,
    [switch]$RegressionOnly,
    [switch]$Debug,

    [string]$Python = "py",
    [string[]]$PythonPrefixArgs = @("-3.10")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-PythonChecked {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & $Python @PythonPrefixArgs @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

if ($SkipRegression -and $RegressionOnly) {
    throw "-SkipRegression and -RegressionOnly cannot be used together."
}
if ($null -eq (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python command was not found: $Python"
}

$patientConfig = Join-Path $PSScriptRoot "library\patients.json"
if (-not (Test-Path -LiteralPath $patientConfig -PathType Leaf)) {
    throw (
        "Missing local patient configuration: $patientConfig`n" +
        "Copy library\patients.example.json to library\patients.json and replace every placeholder path."
    )
}

$channelLayout = if ($Patient -eq "ivanova") { "8_16" } else { "6_12" }
$model = "SimpleNetBase_WithLSTM__CNANNELS_${channelLayout}__LAG_1000_0__WHISPER_BASE_L${Layer}"
$common = @("--patient", $Patient, "--model", $model)
if ($Debug) { $common += "--debug" }

$env:BENCH_SEED = "$Seed"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

Push-Location $PSScriptRoot
try {
    Write-Host "Synchronous Whisper experiment | patient=$Patient | layer=L$Layer | seed=$Seed"
    Write-Host "Model: $model"

    if (-not $SkipRegression) {
        Write-Host "`n===== Stage 1/2: ECoG -> 50D Whisper-PCA target ====="
        Invoke-PythonChecked (@("train_sync.py", "--mode", "regression", "--runs_count", "$RunsCount") + $common)
    }

    if (-not $RegressionOnly) {
        Write-Host "`n===== Stage 2/2: frozen encoder hidden state -> word ====="
        Invoke-PythonChecked (@("train_sync.py", "--mode", "hidden") + $common)
    }
}
finally {
    Pop-Location
}

Write-Host "`nCompleted successfully. Local results are under 02_whisper_sync\results."
