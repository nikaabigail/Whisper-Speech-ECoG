[CmdletBinding()]
param(
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $projectRoot "artifacts\host_inventory.json"
}

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$computer = Get-CimInstance Win32_ComputerSystem
$drives = Get-PSDrive -PSProvider FileSystem | ForEach-Object {
    [ordered]@{
        name = $_.Name
        root = $_.Root
        used_gib = [math]::Round($_.Used / 1GB, 2)
        free_gib = [math]::Round($_.Free / 1GB, 2)
    }
}

$gpuLines = @()
if ($null -ne (Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue)) {
    $gpuLines = @(& nvidia-smi.exe --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits 2>&1)
}

$pythonLines = @()
if ($null -ne (Get-Command "py.exe" -ErrorAction SilentlyContinue)) {
    $pythonLines = @(& py.exe -0p 2>&1)
}

$report = [ordered]@{
    schema_version = 1
    created_utc = [DateTime]::UtcNow.ToString("o")
    os = [ordered]@{
        caption = $os.Caption
        version = $os.Version
        build = $os.BuildNumber
        architecture = $os.OSArchitecture
    }
    cpu = [ordered]@{
        name = $cpu.Name
        logical_processors = $cpu.NumberOfLogicalProcessors
    }
    memory_gib = [math]::Round($computer.TotalPhysicalMemory / 1GB, 2)
    powershell = $PSVersionTable.PSVersion.ToString()
    drives = @($drives)
    nvidia_smi = @($gpuLines)
    python_launcher = @($pythonLines)
    git = if ($null -ne (Get-Command "git.exe" -ErrorAction SilentlyContinue)) { (& git.exe --version) } else { "MISSING" }
    winget = if ($null -ne (Get-Command "winget.exe" -ErrorAction SilentlyContinue)) { (& winget.exe --version) } else { "MISSING" }
}

$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$outputDirectory = Split-Path -Parent $resolvedOutput
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$temporary = "$resolvedOutput.partial"
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding UTF8
Move-Item -LiteralPath $temporary -Destination $resolvedOutput -Force

$report | ConvertTo-Json -Depth 6
Write-Host ""
Write-Host "Saved host inventory: $resolvedOutput"
