# Read-only build/hardware inventory. No install, service or firewall changes.
param(
    [string]$OutputPath = (Join-Path ([Environment]::GetFolderPath('Desktop')) 'qwenox-build-probe.json')
)
$ErrorActionPreference = 'Stop'

function Probe-Cim([string]$ClassName, [string[]]$Properties) {
    try { Get-CimInstance $ClassName | Select-Object -Property $Properties }
    catch { @{ error = $_.Exception.Message } }
}

$commands = @{}
foreach ($name in @('git.exe', 'bash.exe', 'hipcc.exe', 'clang++.exe', 'cl.exe', 'cmake.exe', 'ninja.exe')) {
    $commands[$name] = @(Get-Command $name -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
}
$rockRoots = @()
foreach ($drive in @('C:\', 'D:\')) {
    if (Test-Path -LiteralPath $drive) {
        $rockRoots += @(Get-ChildItem -LiteralPath $drive -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like '*therock*' } | Select-Object -ExpandProperty FullName)
    }
}
foreach ($name in @('THEROCK', 'HIP_PATH')) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ($value -and (Test-Path -LiteralPath $value)) { $rockRoots += $value }
}
$rock = @(foreach ($root in ($rockRoots | Select-Object -Unique)) {
    [ordered]@{
        root = $root
        hipcc = Test-Path -LiteralPath (Join-Path $root 'bin\hipcc.exe')
        clang = Test-Path -LiteralPath (Join-Path $root 'lib\llvm\bin\clang++.exe')
        rocblas_header = Test-Path -LiteralPath (Join-Path $root 'include\rocblas\rocblas.h')
        hipblaslt_header = Test-Path -LiteralPath (Join-Path $root 'include\hipblaslt\hipblaslt.h')
    }
})
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$visualStudio = @()
if (Test-Path -LiteralPath $vswhere) {
    $visualStudio = @(& $vswhere -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json | ConvertFrom-Json |
        Select-Object displayName, installationPath, installationVersion)
}
$sdkInclude = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\Include'
$report = [ordered]@{
    collected_at = (Get-Date).ToString('o')
    computer = Probe-Cim 'Win32_ComputerSystem' @('Manufacturer', 'Model', 'TotalPhysicalMemory')
    bios = Probe-Cim 'Win32_BIOS' @('Manufacturer', 'SMBIOSBIOSVersion', 'ReleaseDate')
    processor = Probe-Cim 'Win32_Processor' @('Name', 'NumberOfCores', 'NumberOfLogicalProcessors')
    graphics = Probe-Cim 'Win32_VideoController' @('Name', 'DriverVersion', 'DriverDate')
    operating_system = Probe-Cim 'Win32_OperatingSystem' @('Caption', 'Version', 'BuildNumber', 'LastBootUpTime')
    tools_on_path = $commands
    therock = $rock
    visual_studio = $visualStudio
    windows_sdk_versions = @(Get-ChildItem -LiteralPath $sdkInclude -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name)
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding UTF8
Write-Host "Saved inventory: $OutputPath"
Write-Host 'No system settings were changed. Share this report with the developer.'
