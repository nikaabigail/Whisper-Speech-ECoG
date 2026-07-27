[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,

    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$ModuleRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ModuleRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual environment is missing. Run bootstrap_windows.ps1 first."
}
if (-not (Test-Path -LiteralPath $RunRoot -PathType Container)) {
    throw "Completed VocalMind production root does not exist: $RunRoot"
}

$ResolvedRunRoot = (Resolve-Path -LiteralPath $RunRoot).Path.TrimEnd('\')
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $ResolvedOutputDirectory = Join-Path $ResolvedRunRoot "oof_aggregate"
}
else {
    $ResolvedOutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory).TrimEnd('\')
}
$ResolvedModuleRoot = (Resolve-Path -LiteralPath $ModuleRoot).Path.TrimEnd('\')
if ($ResolvedOutputDirectory.Equals($ResolvedModuleRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    $ResolvedOutputDirectory.StartsWith("$ResolvedModuleRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OOF output must stay outside the Git source tree."
}

Write-Host "Validating all five held-out gates and aggregating fixed OOF predictions..."
& $Python -B -m whisper_ecog_ext.vocalmind_oof `
    --run-root $ResolvedRunRoot `
    --output-dir $ResolvedOutputDirectory
if ($LASTEXITCODE -ne 0) {
    throw "VocalMind OOF aggregation failed with exit code $LASTEXITCODE"
}
Write-Host "Immutable OOF artifacts: $ResolvedOutputDirectory"
