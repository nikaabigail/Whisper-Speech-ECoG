[CmdletBinding()]
param(
    [string]$CacheDir = 'C:\WhisperECoG_Work\SWPD\contextual_whisper_cache_v1\sub-01',
    [string]$ReferenceSummary = 'C:\WhisperECoG_Work\SWPD\runs\contextual_whisper_sub01_v1\summary.json',
    [string]$RunDir = 'C:\WhisperECoG_Work\SWPD\runs\contextual_neural_e2e_sub01_v1',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [string]$SeedCsv = '1,2,3,4,42',
    [int]$MaxCycles = 5,
    [int]$EpochsPerCycle = 10,
    [int]$BatchSize = 256,
    [double]$LearningRate = 0.0002,
    [double]$WeightDecay = 0.0001,
    [double]$GradClip = 1.0,
    [switch]$DiagnosticSmoke
)
$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$ModuleRoot = Split-Path $PSScriptRoot -Parent
$ExternalRoot = Split-Path $ModuleRoot -Parent
$Python = Join-Path $ExternalRoot '.venv\Scripts\python.exe'
foreach ($Required in @(
    $Python,
    $CacheDir,
    $ReferenceSummary,
    (Join-Path $ModuleRoot 'core.py'),
    (Join-Path $ModuleRoot 'fit_select_sub01.py'),
    (Join-Path $ModuleRoot 'evaluate_frozen_sub01.py'),
    (Join-Path $ModuleRoot 'preflight.py')
)) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Missing required path: $Required" }
}

$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& $Python -I (Join-Path $ModuleRoot 'preflight.py') `
    --device $Device `
    --cache-dir $CacheDir `
    --reference-summary $ReferenceSummary
if ($LASTEXITCODE -ne 0) { throw "Preflight failed with exit code $LASTEXITCODE" }

$SeedTokens = @($SeedCsv.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($SeedTokens.Count -eq 0) { throw 'SeedCsv must contain at least one integer.' }
$Seeds = @($SeedTokens | ForEach-Object {
    $Parsed = 0
    if (-not [int]::TryParse($_, [ref]$Parsed)) { throw "Invalid seed in SeedCsv: $_" }
    $Parsed
})
if (($Seeds | Select-Object -Unique).Count -ne $Seeds.Count) {
    throw 'SeedCsv must contain unique integers.'
}
$NormalizedSeedCsv = ($Seeds | ForEach-Object { [string]$_ }) -join ','
$Arguments = @(
    '-I', '-u', (Join-Path $ModuleRoot 'fit_select_sub01.py'),
    '--cache-dir', $CacheDir,
    '--reference-summary', $ReferenceSummary,
    '--run-dir', $RunDir,
    '--device', $Device,
    '--seeds', $NormalizedSeedCsv,
    '--max-cycles', [string]$MaxCycles,
    '--epochs-per-cycle', [string]$EpochsPerCycle,
    '--batch-size', [string]$BatchSize,
    '--learning-rate', [string]$LearningRate,
    '--weight-decay', [string]$WeightDecay,
    '--grad-clip', [string]$GradClip
)
if ($DiagnosticSmoke) {
    $Arguments += @(
        '--folds', '0',
        '--seeds', '4',
        '--max-cycles', '2',
        '--epochs-per-cycle', '1',
        '--max-train-batches', '2',
        '--max-eval-batches', '2',
        '--diagnostic-smoke'
    )
}
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Contextual neural E2E fit failed with exit code $LASTEXITCODE" }
