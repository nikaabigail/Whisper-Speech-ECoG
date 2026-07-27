[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [Parameter(Mandatory = $true)]
    [string]$CacheRoot,

    [ValidateSet("cuda")]
    [string]$Device = "cuda",

    [ValidateRange(0, 100)]
    [int]$MaxEpochsThisCall = 0,

    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$ModuleRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ModuleRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ModuleRoot "configs\experiments\vocalmind_primary_production.json"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment is missing. Run bootstrap_windows.ps1 first."
}
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "VocalMind data root does not exist: $DataRoot"
}

$ResolvedDataRoot = (Resolve-Path -LiteralPath $DataRoot).Path.TrimEnd('\')
$ResolvedModuleRoot = (Resolve-Path -LiteralPath $ModuleRoot).Path.TrimEnd('\')
$ResolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$ResolvedCacheRoot = [System.IO.Path]::GetFullPath($CacheRoot).TrimEnd('\')
foreach ($pair in @(
    @($ResolvedOutputRoot, $ResolvedDataRoot, "OutputRoot inside dataset"),
    @($ResolvedOutputRoot, $ResolvedModuleRoot, "OutputRoot inside source"),
    @($ResolvedCacheRoot, $ResolvedModuleRoot, "CacheRoot inside source")
)) {
    if ($pair[0].Equals($pair[1], [System.StringComparison]::OrdinalIgnoreCase) -or
        $pair[0].StartsWith("$($pair[1])\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw $pair[2]
    }
}

$env:HF_HOME = $ResolvedCacheRoot
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTHONHASHSEED = "4"
New-Item -ItemType Directory -Path $ResolvedCacheRoot -Force | Out-Null

$OutputParent = Split-Path -Parent $ResolvedOutputRoot
if ([string]::IsNullOrWhiteSpace($OutputParent)) {
    throw "OutputRoot must have a writable parent directory"
}
$PreflightPath = "$ResolvedOutputRoot.host_preflight.json"
$PreflightCandidate = "$PreflightPath.candidate-$PID"
& $Python -B -m whisper_ecog_ext.preflight `
    --dataset vocalmind `
    --data-root $ResolvedDataRoot `
    --cache-root $ResolvedCacheRoot `
    --output-root $OutputParent `
    --json-out $PreflightCandidate
if ($LASTEXITCODE -ne 0) {
    throw "Current host/storage/CUDA preflight failed with exit code $LASTEXITCODE"
}
if (Test-Path -LiteralPath $PreflightPath -PathType Leaf) {
    $ExistingPreflight = Get-Content -LiteralPath $PreflightPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $CurrentPreflight = Get-Content -LiteralPath $PreflightCandidate -Raw -Encoding UTF8 | ConvertFrom-Json
    $ExistingIdentity = [ordered]@{
        python = $ExistingPreflight.python
        platform = $ExistingPreflight.platform
        machine = $ExistingPreflight.machine
        package_versions = $ExistingPreflight.package_versions
        accelerator = $ExistingPreflight.accelerator
    } | ConvertTo-Json -Depth 10 -Compress
    $CurrentIdentity = [ordered]@{
        python = $CurrentPreflight.python
        platform = $CurrentPreflight.platform
        machine = $CurrentPreflight.machine
        package_versions = $CurrentPreflight.package_versions
        accelerator = $CurrentPreflight.accelerator
    } | ConvertTo-Json -Depth 10 -Compress
    if ($ExistingIdentity -ne $CurrentIdentity) {
        throw "Host/runtime identity differs from the existing run. Use a new OutputRoot."
    }
    Remove-Item -LiteralPath $PreflightCandidate
} else {
    Move-Item -LiteralPath $PreflightCandidate -Destination $PreflightPath
}
$env:WHISPER_ECOG_PREFLIGHT_SHA256 = (
    Get-FileHash -LiteralPath $PreflightPath -Algorithm SHA256
).Hash.ToLowerInvariant()

& $Python -B -m whisper_ecog_ext.vocalmind_primary validate-config --config $Config
if ($LASTEXITCODE -ne 0) {
    throw "Production config validation failed with exit code $LASTEXITCODE"
}

$PlanPath = "$ResolvedOutputRoot.execution_plan.json"
$PlanCandidate = "$PlanPath.candidate-$PID"
& $Python -B -m whisper_ecog_ext.vocalmind_primary plan `
    --config $Config `
    --data-root $ResolvedDataRoot `
    --json-out $PlanCandidate
if ($LASTEXITCODE -ne 0) {
    throw "Production planning failed with exit code $LASTEXITCODE"
}
if (Test-Path -LiteralPath $PlanPath -PathType Leaf) {
    if ((Get-FileHash -LiteralPath $PlanPath -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath $PlanCandidate -Algorithm SHA256).Hash) {
        throw "Existing execution plan differs from current frozen source/config. Use a new OutputRoot."
    }
    Remove-Item -LiteralPath $PlanCandidate
} else {
    Move-Item -LiteralPath $PlanCandidate -Destination $PlanPath
}
if ($PlanOnly) {
    Write-Host "Plan-only check completed. No training was started."
    Write-Host "Plan receipt: $PlanPath"
    exit 0
}

$ConfigDocument = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
if ($ConfigDocument.status -ne "frozen_confirmatory") {
    throw "Production remains blocked: config status is '$($ConfigDocument.status)', expected 'frozen_confirmatory'."
}

$Arguments = @(
    "-u", "-B", "-m", "whisper_ecog_ext.vocalmind_primary", "run",
    "--config", $Config,
    "--data-root", $ResolvedDataRoot,
    "--output-root", $ResolvedOutputRoot,
    "--device", $Device
)
if ($MaxEpochsThisCall -gt 0) {
    $Arguments += @("--max-epochs-this-call", [string]$MaxEpochsThisCall)
}

Write-Host "Starting frozen VocalMind production; compatible checkpoints resume automatically."
Write-Host "Data:      $ResolvedDataRoot"
Write-Host "Artifacts: $ResolvedOutputRoot"
Write-Host "HF cache:  $ResolvedCacheRoot"
Write-Host "Preflight: $PreflightPath"
& $Python @Arguments
if ($LASTEXITCODE -eq 3) {
    Write-Host "Bounded stage saved. Re-run the identical command and OutputRoot to continue."
    exit 3
}
if ($LASTEXITCODE -ne 0) {
    throw "VocalMind production failed with exit code $LASTEXITCODE"
}
Write-Host "VocalMind production completed successfully."
