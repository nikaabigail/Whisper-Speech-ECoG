[CmdletBinding()]
param(
    [string]$CacheRoot = 'C:\WhisperECoG_Work\SWPD\contextual_l4_frozen_cache_v1',
    [string]$RunRoot = 'C:\WhisperECoG_Work\SWPD\runs\contextual_fixed_q_neural_population_v1',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference='Stop';$ModuleRoot=Split-Path $PSScriptRoot -Parent;$ExternalRoot=Split-Path $ModuleRoot -Parent;$Python=Join-Path $ExternalRoot '.venv\Scripts\python.exe'
$env:PYTHONNOUSERSITE='1';$env:PYTHONUTF8='1';$env:PYTHONIOENCODING='utf-8';$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& $Python -I -u (Join-Path $ModuleRoot 'run_population.py') evaluate --cache-root $CacheRoot --run-root $RunRoot --device $Device
if($LASTEXITCODE -ne 0){throw "Population evaluation failed with exit code $LASTEXITCODE"}
