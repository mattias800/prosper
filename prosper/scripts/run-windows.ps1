[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Dump,

    [string]$BuildDir,
    [string]$GuestArgs = '-force-gfx-direct',
    [string]$SavedataDir,
    [string]$Record,
    [ValidateSet('flip', 'pad-read')]
    [string]$RecordAxis = 'flip',
    [int]$Frames = 0,
    [int]$Jobs = 8,
    [ValidateSet('fifo', 'mailbox', 'immediate')]
    [string]$PresentMode = 'fifo',
    [switch]$TestPattern,
    [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Program $($Arguments -join ' ')"
    }
}

function Find-WinLibsTool {
    param([string]$Name)

    $onPath = Get-Command $Name -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    $candidates = @()
    if ($env:LOCALAPPDATA) {
        $wingetRoot = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
        if (Test-Path -LiteralPath $wingetRoot) {
            $packages = Get-ChildItem -LiteralPath $wingetRoot -Directory -Filter 'BrechtSanders.WinLibs*' |
                Sort-Object LastWriteTime -Descending
            foreach ($package in $packages) {
                $candidates += Join-Path $package.FullName "mingw64\bin\$Name"
            }
        }
    }
    $candidates += "C:\msys64\ucrt64\bin\$Name"
    $candidates += "C:\msys64\mingw64\bin\$Name"

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

if (-not $IsWindows -and $PSVersionTable.PSEdition -eq 'Core') {
    throw 'run-windows.ps1 must be run from native Windows PowerShell, not WSL/Linux.'
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$sourceDir = Join-Path $repoRoot 'prosper'
if (-not $BuildDir) { $BuildDir = Join-Path $sourceDir 'build-mingw-app' }
$BuildDir = [IO.Path]::GetFullPath($BuildDir)

if (-not $TestPattern) {
    if (-not $Dump) { throw 'Pass an app0 dump directory, or use -TestPattern.' }
    $Dump = (Resolve-Path -LiteralPath $Dump).Path
    if (-not (Test-Path -LiteralPath $Dump -PathType Container)) {
        throw "Dump is not a directory: $Dump"
    }
}
if (-not $Record -and $RecordAxis -ne 'flip') {
    throw '-RecordAxis requires -Record.'
}

$cmake = (Get-Command cmake -ErrorAction Stop).Source
$cache = Join-Path $BuildDir 'CMakeCache.txt'
$app = Join-Path $BuildDir 'prosper-app.exe'

if (-not $NoBuild) {
    $configure = @(
        '-S', $sourceDir,
        '-B', $BuildDir,
        '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=RelWithDebInfo',
        '-DPROSPER_APP=ON',
        '-DPROSPER_AUDIO_SDL3=ON',
        '-DPROSPER_PAD_SDL3=ON'
    )
    if (-not (Test-Path -LiteralPath $cache)) {
        $cxx = Find-WinLibsTool 'g++.exe'
        $cc = if ($cxx) { Join-Path (Split-Path -Parent $cxx) 'gcc.exe' } else { $null }
        if (-not $cxx -or -not (Test-Path -LiteralPath $cc)) {
            throw 'MinGW-w64 UCRT gcc/g++ were not found. Install WinLibs with winget, or add its mingw64\bin directory to PATH.'
        }
        $configure += "-DCMAKE_C_COMPILER=$cc"
        $configure += "-DCMAKE_CXX_COMPILER=$cxx"
    }
    Invoke-Checked $cmake $configure
    Invoke-Checked $cmake @('--build', $BuildDir, '--target', 'prosper-app', '--parallel', "$Jobs")
}

if (-not (Test-Path -LiteralPath $app -PathType Leaf)) {
    throw "prosper-app.exe was not built at $app. Check the Vulkan SDK and rerun without -NoBuild."
}

$runArgs = @()
if ($TestPattern) {
    $runArgs += '--test-pattern'
} else {
    # PROSPER_GUEST_FS is not set: guest_tls.cpp reads it only under `#ifdef __APPLE__`.
    # On Windows (`:240`) guest TLS is ON by default; the real switch is the opt-OUT
    # PROSPER_NO_GUEST_FS (#2098).
    $env:PROSPER_GUEST_ARGS = $GuestArgs
    if ($SavedataDir) { $env:PROSPER_SAVEDATA_DIR = [IO.Path]::GetFullPath($SavedataDir) }
    $runArgs += @('--dump', $Dump)
}
if ($Frames -gt 0) { $runArgs += @('--frames', "$Frames") }
if ($PresentMode -ne 'fifo') { $runArgs += @('--present-mode', $PresentMode.ToLowerInvariant()) }
if ($Record) {
    $runArgs += @('--record', [IO.Path]::GetFullPath($Record))
    if ($RecordAxis -ne 'flip') {
        $runArgs += @('--record-axis', $RecordAxis.ToLowerInvariant())
    }
}

Write-Host "Starting $app"
& $app @runArgs
if ($LASTEXITCODE -ne 0) { throw "prosper-app exited with code $LASTEXITCODE" }

