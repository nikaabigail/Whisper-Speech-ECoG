[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [Parameter(Mandatory = $true)]
    [string]$CacheRoot,

    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ModuleRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ModuleRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment is missing. Run bootstrap_windows.ps1 first."
}

$env:HF_HOME = [System.IO.Path]::GetFullPath($CacheRoot)
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTHONHASHSEED = "4"
New-Item -ItemType Directory -Path $env:HF_HOME -Force | Out-Null
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$SmokeReceipt = Join-Path ([System.IO.Path]::GetFullPath($OutputRoot)) "vocalmind_rep6_smoke_receipt.json"
if (Test-Path -LiteralPath $SmokeReceipt -PathType Leaf) {
    throw "Smoke receipt already exists; use a new OutputRoot to preserve it: $SmokeReceipt"
}
$PreflightReceipt = Join-Path ([System.IO.Path]::GetFullPath($OutputRoot)) "host_preflight.json"

& $Python -B -m whisper_ecog_ext.preflight `
    --dataset vocalmind `
    --data-root $DataRoot `
    --cache-root $env:HF_HOME `
    --output-root $OutputRoot `
    --json-out $PreflightReceipt
if ($LASTEXITCODE -ne 0) {
    throw "Host/storage/CUDA preflight failed with exit code $LASTEXITCODE"
}

& $Python -B (Join-Path $ModuleRoot "vocalmind_rep6_smoke.py") `
    --data-root $DataRoot `
    --output-dir $OutputRoot `
    --device $Device
if ($LASTEXITCODE -ne 0) {
    throw "VocalMind rep6 smoke failed with exit code $LASTEXITCODE"
}
