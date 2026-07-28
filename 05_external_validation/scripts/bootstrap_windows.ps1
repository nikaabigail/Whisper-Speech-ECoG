[CmdletBinding()]
param(
    [ValidateSet("swpd", "vocalmind", "all")]
    [string]$Dataset = "vocalmind",

    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [string]$CacheRoot,

    [switch]$InstallSystemTools,
    [switch]$AllowCpu,
    [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

function Test-ExternalCommand {
    param([Parameter(Mandatory = $true)][string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Install-WithWinget {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-ExternalCommand "winget")) {
        throw "winget is unavailable; install $Label manually and rerun this script."
    }
    Write-Host "Installing $Label..."
    & winget install --exact --id $Id --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed while installing $Label (exit $LASTEXITCODE)."
    }

    # winget updates the persistent PATH, but the already-open PowerShell
    # process does not always see it. Refresh both scopes before probing again.
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
}

if (-not (Test-ExternalCommand "git")) {
    if (-not $InstallSystemTools) {
        throw "Git is missing. Rerun with -InstallSystemTools or install Git for Windows."
    }
    Install-WithWinget -Id "Git.Git" -Label "Git for Windows"
}

$pyLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
if ($null -eq $pyLauncher) {
    if (-not $InstallSystemTools) {
        throw "Python launcher is missing. Rerun with -InstallSystemTools or install Python 3.10 x64."
    }
    Install-WithWinget -Id "Python.Python.3.10" -Label "Python 3.10 x64"
}

if ($null -eq (Get-Command "py.exe" -ErrorAction SilentlyContinue)) {
    throw "Python was installed but py.exe is not visible yet. Close PowerShell, open it again, and rerun this command."
}

$resolvedProject = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd('\')
$resolvedDataRoot = [System.IO.Path]::GetFullPath($DataRoot).TrimEnd('\')
if ($resolvedDataRoot.StartsWith("$resolvedProject\", [System.StringComparison]::OrdinalIgnoreCase) -or
    $resolvedDataRoot.Equals($resolvedProject, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "DataRoot must be outside the Git checkout. Use an ASCII path such as C:\WhisperECoG\VocalMind."
}

if ([string]::IsNullOrWhiteSpace($CacheRoot)) {
    $dataParent = Split-Path -Parent $resolvedDataRoot
    $CacheRoot = Join-Path $dataParent "model_cache\huggingface"
}
$resolvedCacheRoot = [System.IO.Path]::GetFullPath($CacheRoot).TrimEnd('\')
if ($resolvedCacheRoot.StartsWith("$resolvedProject\", [System.StringComparison]::OrdinalIgnoreCase) -or
    $resolvedCacheRoot.Equals($resolvedProject, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "CacheRoot must be outside the Git checkout."
}
New-Item -ItemType Directory -Path $resolvedCacheRoot -Force | Out-Null
$env:HF_HOME = $resolvedCacheRoot

& py.exe -3.10 -c "import sys; assert sys.version_info[:2] == (3, 10); print(sys.executable)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10 x64 is required. Install it and rerun the script."
}

if ($RecreateVenv -and (Test-Path -LiteralPath $venvRoot)) {
    $resolvedVenv = [System.IO.Path]::GetFullPath($venvRoot)
    if (-not $resolvedVenv.StartsWith($resolvedProject, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a virtual environment outside the project: $resolvedVenv"
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating Python 3.10 virtual environment..."
    & py.exe -3.10 -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }
}

& $venvPython -m pip install --upgrade -r (Join-Path $projectRoot "requirements\tooling-lock.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to update Python packaging tools." }

Write-Host "Installing the reproducible CUDA 12.8 PyTorch wheel..."
& $venvPython -m pip install --index-url "https://download.pytorch.org/whl/cu128" "torch==2.10.0"
if ($LASTEXITCODE -ne 0) { throw "Failed to install PyTorch 2.10.0+cu128." }

& $venvPython -m pip install -r (Join-Path $projectRoot "requirements\full-runtime-lock.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install the complete locked runtime." }

& $venvPython -m pip install --editable $projectRoot
if ($LASTEXITCODE -ne 0) { throw "Failed to install the external-validation package." }

$preflightArgs = @(
    "-m", "whisper_ecog_ext.preflight",
    "--dataset", $Dataset,
    "--data-root", $DataRoot,
    "--cache-root", $resolvedCacheRoot,
    "--json-out", (Join-Path $projectRoot "artifacts\host_preflight.json")
)
if ($AllowCpu) { $preflightArgs += "--allow-cpu" }

Write-Host "Running the final host and CUDA preflight..."
& $venvPython @preflightArgs
if ($LASTEXITCODE -ne 0) { throw "Preflight failed." }

Write-Host ""
Write-Host "Environment is ready."
Write-Host "Python: $venvPython"
Write-Host "Data root: $DataRoot"
Write-Host "Hugging Face cache: $resolvedCacheRoot"
Write-Host "No standalone CUDA Toolkit was installed or required."
