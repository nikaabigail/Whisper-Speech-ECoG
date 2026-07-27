[CmdletBinding()]
param(
    [string]$DataRoot = 'C:\WhisperECoG\SWPD\extracted',
    [string]$CacheDir = 'C:\WhisperECoG_Work\SWPD\cache_1000hz',
    [string]$RunDir = 'C:\WhisperECoG_Work\SWPD\runs\seed4_v1',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [int]$MaxEpochs = 0,
    [int]$BatchSize = 0,
    [switch]$FastSmoke,
    [switch]$SingleMelDevelopment,
    [switch]$PrepareCacheOnly,
    [switch]$ForceCache,
    [switch]$NoVad
)

$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python. Run .\scripts\bootstrap_windows.ps1 first."
}
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "SWPD data root not found: $DataRoot"
}

New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$VadDir = Join-Path $RunDir 'audio_audit'
New-Item -ItemType Directory -Path $VadDir -Force | Out-Null
$CandidateTsv = Join-Path $VadDir 'audio_vad_candidates_unreviewed.tsv'

& $Python -c 'import torch; print("Torch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
if ($LASTEXITCODE -ne 0) {
    throw "Python/PyTorch preflight failed with exit code $LASTEXITCODE"
}
if ($Device -eq 'cuda') {
    & $Python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'
    if ($LASTEXITCODE -ne 0) {
        throw 'CUDA was requested but is unavailable in the project Python environment.'
    }
}

if (-not $NoVad) {
    Write-Host 'Creating or verifying deterministic audio-only VAD candidates (still unreviewed)...'
    & $Python (Join-Path $ProjectRoot 'swpd_audio_vad.py') `
        --data-root $DataRoot `
        --output-dir $VadDir
    if ($LASTEXITCODE -ne 0) {
        throw "Audio candidate generation failed with exit code $LASTEXITCODE"
    }
}

$Arguments = @(
    (Join-Path $ProjectRoot 'swpd_neural_pilot.py'),
    '--data-root', $DataRoot,
    '--cache-dir', $CacheDir,
    '--run-dir', $RunDir,
    '--device', $Device
)
if ($MaxEpochs -gt 0) {
    $Arguments += @('--max-epochs', [string]$MaxEpochs)
}
if ($BatchSize -gt 0) {
    $Arguments += @('--batch-size', [string]$BatchSize)
}
if ($FastSmoke) {
    $Arguments += '--fast-smoke'
}
if ($SingleMelDevelopment) {
    $Arguments += '--single-mel-development'
}
if ($PrepareCacheOnly) {
    $Arguments += '--prepare-cache-only'
}
if ($ForceCache) {
    $Arguments += '--force-cache'
}
if (-not $NoVad -and (Test-Path -LiteralPath $CandidateTsv -PathType Leaf)) {
    $Arguments += @('--audio-candidate-tsv', $CandidateTsv)
}

Write-Host 'Starting SWPD sub-01 full-neural regression pilot.'
Write-Host "Data:  $DataRoot"
Write-Host "Cache: $CacheDir"
Write-Host "Run:   $RunDir"
if ($FastSmoke) {
    Write-Warning 'FastSmoke is 50 Hz and diagnostic only; it is not a scientific result.'
} elseif ($SingleMelDevelopment) {
    Write-Warning 'Single MEL is development-only; production comparison requires fixed MEL x3.'
} else {
    Write-Host 'Production pilot controls: fixed MEL x3 plus Whisper L3/L4/L5.'
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "SWPD neural pilot failed with exit code $LASTEXITCODE"
}
Write-Host 'SWPD sub-01 regression pilot completed. The event/asynchronous gate is still closed.'
