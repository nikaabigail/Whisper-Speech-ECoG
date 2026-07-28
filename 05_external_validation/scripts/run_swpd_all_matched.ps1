[CmdletBinding()]
param(
    [string]$DataRoot = 'C:\WhisperECoG\SWPD\extracted',
    [string]$CacheRoot = 'C:\WhisperECoG_Work\SWPD\matched_pca50_all_cache_v2',
    [string]$RunRoot = 'C:\WhisperECoG_Work\SWPD\runs\matched_pca50_all_v2',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Protocol = Join-Path $ProjectRoot 'configs\experiments\swpd_all_matched_pca50_v1.json'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "SWPD data root not found: $DataRoot"
}
if (-not (Test-Path -LiteralPath $Protocol -PathType Leaf)) {
    throw "Frozen protocol not found: $Protocol"
}

# Python 3.10 reads the existing editable .pth in the Windows ANSI code page.
# A parent PYTHONUTF8=1 would reinterpret it and fail before importing site.
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = '1'
New-Item -ItemType Directory -Path $CacheRoot -Force | Out-Null
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null

if (-not $PlanOnly) {
    $PreflightReceipt = Join-Path $RunRoot 'host_preflight.json'
    $PreflightArguments = @(
        '-B', '-m', 'whisper_ecog_ext.preflight',
        '--dataset', 'swpd',
        '--data-root', $DataRoot,
        '--output-root', $RunRoot,
        '--json-out', $PreflightReceipt
    )
    if ($Device -eq 'cpu') { $PreflightArguments += '--allow-cpu' }
    & $Python @PreflightArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python/PyTorch/CUDA preflight failed with exit code $LASTEXITCODE"
    }
}

$Arguments = @(
    '-u', (Join-Path $ProjectRoot 'swpd_matched_all.py'),
    '--data-root', $DataRoot,
    '--cache-root', $CacheRoot,
    '--run-root', $RunRoot,
    '--protocol-config', $Protocol,
    '--device', $Device,
    '--reducer-seed', '42'
)
if ($PlanOnly) { $Arguments += '--plan-only' }

Write-Host 'Starting frozen SWPD matched PCA50 queue.'
Write-Host "Data:     $DataRoot"
Write-Host "Cache:    $CacheRoot"
Write-Host "Run:      $RunRoot"
Write-Host 'Frozen planned cohort: sub-02..sub-10; sub-01 remains development-only.'
Write-Host 'Known source QC: sub-10 is incomplete and is excluded only by swpd_finalize_qc.py after sub-01..sub-09 complete.'
& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "SWPD all-subject matched run failed with exit code $LASTEXITCODE"
}
