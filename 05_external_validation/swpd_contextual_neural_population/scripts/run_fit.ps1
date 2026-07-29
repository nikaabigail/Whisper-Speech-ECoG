[CmdletBinding()]
param(
    [string]$CacheRoot = 'C:\WhisperECoG_Work\SWPD\contextual_l4_frozen_cache_v1',
    [string]$RunRoot = 'C:\WhisperECoG_Work\SWPD\runs\contextual_fixed_q_neural_population_v1',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [switch]$Diagnostic
)
$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) { $PSNativeCommandUseErrorActionPreference = $false }
$ModuleRoot = Split-Path $PSScriptRoot -Parent
$ExternalRoot = Split-Path $ModuleRoot -Parent
$Python = Join-Path $ExternalRoot '.venv\Scripts\python.exe'
$env:PYTHONNOUSERSITE='1'; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; $env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& $Python -I (Join-Path $ModuleRoot 'preflight.py') --cache-root $CacheRoot --device $Device
if ($LASTEXITCODE -ne 0) { throw "Population preflight failed with exit code $LASTEXITCODE" }
$Arguments = @('-I','-u',(Join-Path $ModuleRoot 'run_population.py'),'fit','--cache-root',$CacheRoot,'--run-root',$RunRoot,'--device',$Device)
if ($Diagnostic) { $Arguments += @('--subjects','sub-02','--seeds','4','--folds','0','--max-cycles','1','--epochs','1','--max-train-batches','2','--max-eval-batches','2','--diagnostic') }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Population fit failed with exit code $LASTEXITCODE" }
