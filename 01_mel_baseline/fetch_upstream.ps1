#requires -Version 5.1

[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot "upstream"),
    [string]$Repository = "https://github.com/pet67/ossadtchi-ml-test-bench-speech.git",
    [ValidatePattern("^[0-9a-fA-F]{40}$")]
    [string]$Commit = "6a2ee87957a7b15178c9ce4ca11efc5182f5dc59"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-GitChecked {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git failed with exit code ${LASTEXITCODE}: git $($Arguments -join ' ')"
    }
}

if ($null -eq (Get-Command "git" -ErrorAction SilentlyContinue)) {
    throw "Git was not found on PATH. Install Git for Windows and retry."
}

$Destination = [System.IO.Path]::GetFullPath($Destination)
$gitDirectory = Join-Path $Destination ".git"

if (-not (Test-Path -LiteralPath $Destination)) {
    $parent = Split-Path -Parent $Destination
    if ($parent) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Invoke-GitChecked @("clone", "--filter=blob:none", "--no-checkout", $Repository, $Destination)
}
elseif (-not (Test-Path -LiteralPath $gitDirectory -PathType Container)) {
    throw "Destination exists but is not a Git checkout: $Destination"
}

Invoke-GitChecked @("-C", $Destination, "fetch", "--no-tags", "origin", $Commit)
Invoke-GitChecked @("-C", $Destination, "checkout", "--detach", $Commit)

$resolved = (& git -C $Destination rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $resolved -ne $Commit.ToLowerInvariant()) {
    throw "Pinned revision verification failed: expected $Commit, got $resolved"
}

Write-Host "MEL baseline source is ready at: $Destination"
Write-Host "Pinned upstream commit: $resolved"
Write-Host "The upstream checkout is ignored by this repository and keeps its own history/license status."
