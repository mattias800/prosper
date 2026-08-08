[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Dump,

    [string]$SavedataDir = (Join-Path $PSScriptRoot 'savedata'),
    [string]$GuestArgs = '-force-gfx-direct',
    [string]$Record,
    [ValidateSet('flip', 'pad-read')]
    [string]$RecordAxis = 'flip',
    [int]$Frames = 0,
    [ValidateSet('fifo', 'mailbox', 'immediate')]
    [string]$PresentMode = 'fifo',
    [switch]$TestPattern
)

$ErrorActionPreference = 'Stop'
$app = Join-Path $PSScriptRoot 'prosper-app.exe'

if (-not (Test-Path -LiteralPath $app -PathType Leaf)) {
    throw "prosper-app.exe must be beside this script: $app"
}
if (-not $Record -and $RecordAxis -ne 'flip') {
    throw '-RecordAxis requires -Record.'
}

$runArgs = @()
if ($TestPattern) {
    $runArgs += '--test-pattern'
} else {
    if (-not $Dump) { throw 'Pass an app0 dump directory, or use -TestPattern.' }
    $Dump = (Resolve-Path -LiteralPath $Dump).Path
    if (-not (Test-Path -LiteralPath $Dump -PathType Container)) {
        throw "Dump is not a directory: $Dump"
    }

    $SavedataDir = [IO.Path]::GetFullPath($SavedataDir)
    New-Item -ItemType Directory -Path $SavedataDir -Force | Out-Null
    # PROSPER_GUEST_FS is not set: guest_tls.cpp reads it only under `#ifdef __APPLE__`.
    # On Windows (`:240`) guest TLS is ON by default; the real switch is the opt-OUT
    # PROSPER_NO_GUEST_FS (#2098).
    $env:PROSPER_GUEST_ARGS = $GuestArgs
    $env:PROSPER_SAVEDATA_DIR = $SavedataDir
    $runArgs += @('--dump', $Dump)
}

if ($Frames -gt 0) { $runArgs += @('--frames', "$Frames") }
if ($PresentMode -ne 'fifo') { $runArgs += @('--present-mode', $PresentMode.ToLowerInvariant()) }
if ($Record) {
    $recordPath = [IO.Path]::GetFullPath($Record)
    $recordParent = Split-Path -Parent $recordPath
    if ($recordParent) { New-Item -ItemType Directory -Path $recordParent -Force | Out-Null }
    $runArgs += @('--record', $recordPath)
    if ($RecordAxis -ne 'flip') {
        $runArgs += @('--record-axis', $RecordAxis.ToLowerInvariant())
    }
}

& $app @runArgs
if ($LASTEXITCODE -ne 0) { throw "prosper-app exited with code $LASTEXITCODE" }
