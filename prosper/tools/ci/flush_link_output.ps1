#requires -Version 7.0
# Persist one completed link output so compressed CI storage need not wait for a cached flush.
# The caller supplies its own output, never a directory or an unrelated process's files.
param([string]$OutputFile)
$ErrorActionPreference = 'Stop'

function Flush-LinkedOutput {
    param([Parameter(Mandatory)][string]$LiteralPath)
    $file = Get-Item -LiteralPath $LiteralPath -Force
    if ($file.PSIsContainer -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Linked output must be a regular file'
    }
    $stream = [IO.File]::Open($file.FullName, [IO.FileMode]::Open, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite)
    try { $stream.Flush($true) } finally { $stream.Dispose() }
}

if ($MyInvocation.InvocationName -eq '.') { return }
if (-not $OutputFile) { throw 'Linked output path is required' }
Flush-LinkedOutput -LiteralPath $OutputFile
Write-Host 'Linked output flush complete'
