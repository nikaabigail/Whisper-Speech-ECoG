[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$ModuleRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ModuleRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ModuleRoot "configs\experiments\vocalmind_pilot.json"
$Source = Join-Path $ModuleRoot "src"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment is missing. Run .\scripts\bootstrap_windows.ps1 first: $Python"
}
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "VocalMind data root does not exist: $DataRoot"
}

$ResolvedDataRoot = (Resolve-Path -LiteralPath $DataRoot).Path
$ResolvedModuleRoot = (Resolve-Path -LiteralPath $ModuleRoot).Path
$OutputFullPath = [System.IO.Path]::GetFullPath($OutputRoot)

if ($OutputFullPath.StartsWith($ResolvedDataRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must be outside the read-only dataset."
}
if ($OutputFullPath.StartsWith($ResolvedModuleRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must be outside the source checkout."
}

$env:PYTHONPATH = $Source

Write-Host "Validating frozen pilot config..."
& $Python -B -m whisper_ecog_ext.vocalmind_primary validate-config --config $Config
if ($LASTEXITCODE -ne 0) {
    throw "Config validation failed with exit code $LASTEXITCODE"
}

Write-Host "Building read-only execution plan..."
$PlanPath = "$OutputFullPath.execution_plan.json"
$PlanCandidate = "$PlanPath.candidate-$PID"
& $Python -B -m whisper_ecog_ext.vocalmind_primary plan `
    --config $Config `
    --data-root $ResolvedDataRoot `
    --json-out $PlanCandidate
if ($LASTEXITCODE -ne 0) {
    throw "Pilot planning failed with exit code $LASTEXITCODE"
}
if (Test-Path -LiteralPath $PlanPath -PathType Leaf) {
    if ((Get-FileHash -LiteralPath $PlanPath -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath $PlanCandidate -Algorithm SHA256).Hash) {
        throw "Existing plan receipt differs from current source/config. Use a new OutputRoot."
    }
    Remove-Item -LiteralPath $PlanCandidate
    Write-Host "Validated immutable plan receipt: $PlanPath"
} else {
    Move-Item -LiteralPath $PlanCandidate -Destination $PlanPath
}
if ($PlanOnly) {
    Write-Host "Plan-only check completed. No training was started."
    Write-Host "Plan receipt: $PlanPath"
    exit 0
}

throw "VocalMind development config is intentionally PlanOnly. Repetitions 1-5 stay numerically closed until the production protocol/config/commit are frozen. Re-run with -PlanOnly; use a new OutputRoot only after freezing the production config."
