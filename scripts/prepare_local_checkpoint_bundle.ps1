#requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,

    [string]$DestinationRoot = "",
    [switch]$ArchiveOnly,
    [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..")
)
$manifestPath = Join-Path $repositoryRoot "checkpoints\release_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Release manifest is missing: $manifestPath"
}

$SourceRoot = [System.IO.Path]::GetFullPath($SourceRoot)
if ([string]::IsNullOrWhiteSpace($DestinationRoot)) {
    $DestinationRoot = Join-Path $repositoryRoot "checkpoints\frozen_seed4"
}
$DestinationRoot = [System.IO.Path]::GetFullPath($DestinationRoot)
$destinationPrefix = $DestinationRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
$syncRoot = Join-Path $repositoryRoot "02_whisper_sync"

function Get-CheckedDestination {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $candidate = [System.IO.Path]::GetFullPath((Join-Path $DestinationRoot $RelativePath))
    if (-not $candidate.StartsWith($destinationPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest path escapes the destination root: $RelativePath"
    }
    return $candidate
}

function Assert-ExpectedFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][long]$Bytes,
        [Parameter(Mandatory = $true)][string]$Sha256
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is missing: $Path"
    }
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -ne $Bytes) {
        throw "Size mismatch for $Path (expected $Bytes, got $($item.Length))"
    }
    $actualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $Sha256.ToLowerInvariant()) {
        throw "SHA256 mismatch for $Path"
    }
}

function Copy-VerifiedFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][long]$Bytes,
        [Parameter(Mandatory = $true)][string]$Sha256
    )

    Assert-ExpectedFile -Path $Source -Bytes $Bytes -Sha256 $Sha256
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        Assert-ExpectedFile -Path $Destination -Bytes $Bytes -Sha256 $Sha256
        Write-Host "[reuse] $Destination"
        return
    }
    Copy-Item -LiteralPath $Source -Destination $Destination
    Assert-ExpectedFile -Path $Destination -Bytes $Bytes -Sha256 $Sha256
    Write-Host "[copied] $Destination"
}

$releaseManifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([int]$releaseManifest.source_seed -ne 4) {
    throw "Unexpected source seed in release manifest: $($releaseManifest.source_seed)"
}

$runtimeEntries = @()
foreach ($entry in $releaseManifest.files) {
    $sourceDirectory = if ([string]$entry.role -eq "synchronous_result_and_split_provenance") {
        Join-Path $SourceRoot "results"
    }
    else {
        Join-Path $SourceRoot "model_dumps"
    }
    $source = Join-Path $sourceDirectory ([string]$entry.filename)
    $archiveDestination = Get-CheckedDestination -RelativePath ([string]$entry.expected_archive_relative_path)
    if ($VerifyOnly) {
        Assert-ExpectedFile `
            -Path $source `
            -Bytes ([long]$entry.bytes) `
            -Sha256 ([string]$entry.sha256)
        Write-Host "[verified] $source"
    }
    else {
        Copy-VerifiedFile `
            -Source $source `
            -Destination $archiveDestination `
            -Bytes ([long]$entry.bytes) `
            -Sha256 ([string]$entry.sha256)
    }

    if (-not $ArchiveOnly -and -not $VerifyOnly) {
        $syncDirectory = if ([string]$entry.role -eq "synchronous_result_and_split_provenance") {
            Join-Path $syncRoot "results"
        }
        else {
            Join-Path $syncRoot "model_dumps"
        }
        $syncDestination = Join-Path $syncDirectory ([string]$entry.filename)
        Copy-VerifiedFile `
            -Source $source `
            -Destination $syncDestination `
            -Bytes ([long]$entry.bytes) `
            -Sha256 ([string]$entry.sha256)
    }

    $runtimeEntries += [ordered]@{
        relative_path = ([string]$entry.expected_archive_relative_path).Replace('\', '/')
        bytes = [long]$entry.bytes
        sha256 = ([string]$entry.sha256).ToLowerInvariant()
    }
}

if ($VerifyOnly) {
    Write-Host "`nAll nine source payloads match checkpoints\release_manifest.json. Nothing was copied."
    exit 0
}

$runtimeManifest = [ordered]@{
    schema_version = 1
    kind = "frozen_upstream_sha256_manifest"
    seed = 4
    generated_utc = [DateTime]::UtcNow.ToString("o")
    files = $runtimeEntries
}
$runtimeManifestPath = Join-Path $DestinationRoot "MANIFEST.sha256.json"
$runtimeJson = ($runtimeManifest | ConvertTo-Json -Depth 6) + [Environment]::NewLine
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($runtimeManifestPath, $runtimeJson, $utf8NoBom)

Write-Host "`nLocal checkpoint bundle is ready: $DestinationRoot"
Write-Host "Runtime manifest: $runtimeManifestPath"
if (-not $ArchiveOnly) {
    Write-Host "Sync checkpoints/results were also materialized under: $syncRoot"
}
Write-Host "All copied payloads matched the release SHA256 manifest; source files were not modified."
