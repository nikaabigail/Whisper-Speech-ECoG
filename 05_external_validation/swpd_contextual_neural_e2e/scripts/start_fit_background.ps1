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
$Wrapper = Join-Path $PSScriptRoot 'run_fit.ps1'
$ResolvedRun = [IO.Path]::GetFullPath($RunDir).TrimEnd('\')
$Launcher = Join-Path $ResolvedRun 'launcher'
New-Item -ItemType Directory -Path $Launcher -Force | Out-Null
$ReceiptPath = Join-Path $Launcher 'launcher.json'
if (Test-Path -LiteralPath $ReceiptPath) {
    $Old = Get-Content -LiteralPath $ReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (Get-Process -Id ([int]$Old.pid) -ErrorAction SilentlyContinue) {
        throw "An experiment process is already running: PID=$($Old.pid)"
    }
    $Previous = Join-Path $Launcher "launcher_previous_$(Get-Date -Format yyyyMMdd_HHmmss).json"
    Move-Item -LiteralPath $ReceiptPath -Destination $Previous
}
$Stamp = Get-Date -Format yyyyMMdd_HHmmss
$Stdout = Join-Path $Launcher "contextual_neural_e2e_fit_$Stamp.out.log"
$Stderr = Join-Path $Launcher "contextual_neural_e2e_fit_$Stamp.err.log"
New-Item -ItemType File -Path $Stdout -Force | Out-Null
New-Item -ItemType File -Path $Stderr -Force | Out-Null
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
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$Wrapper`"",
    '-CacheDir', "`"$CacheDir`"",
    '-ReferenceSummary', "`"$ReferenceSummary`"",
    '-RunDir', "`"$ResolvedRun`"",
    '-Device', $Device,
    '-SeedCsv', $NormalizedSeedCsv,
    '-MaxCycles', [string]$MaxCycles,
    '-EpochsPerCycle', [string]$EpochsPerCycle,
    '-BatchSize', [string]$BatchSize,
    '-LearningRate', [string]$LearningRate,
    '-WeightDecay', [string]$WeightDecay,
    '-GradClip', [string]$GradClip
)
if ($DiagnosticSmoke) { $Arguments += '-DiagnosticSmoke' }
$Process = Start-Process powershell.exe `
    -ArgumentList $Arguments `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru
$Receipt = [ordered]@{
    schema_version = 1
    task = 'swpd_sub01_contextual_neural_e2e_fit_only'
    pid = $Process.Id
    started_utc = [DateTime]::UtcNow.ToString('o')
    output_root = $ResolvedRun
    stdout = $Stdout
    stderr = $Stderr
    seeds = $Seeds
    max_cycles = $MaxCycles
    epochs_per_cycle = $EpochsPerCycle
    diagnostic_smoke = [bool]$DiagnosticSmoke
}
$Temporary = "$ReceiptPath.partial"
$Receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Temporary -Encoding UTF8
Move-Item -LiteralPath $Temporary -Destination $ReceiptPath
Write-Host "Started hidden background process PID=$($Process.Id)"
Write-Host "Launcher receipt: $ReceiptPath"
Write-Host 'Ctrl+C in the watcher will not stop training.'
