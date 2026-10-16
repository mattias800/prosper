#requires -Version 7.0
# Windows CI storage measurements; only explicit synthetic probes are flushed.
# No paths or file contents enter the report.
param(
    [ValidateSet('before', 'after')][string]$Phase = 'after',
    [string]$CompilerTempDirectory,
    [string]$ProbeFile,
    [ValidateSet('compressed', 'uncompressed')][string]$ExpectedProbe
)
$ErrorActionPreference = 'Stop'

function Initialize-StorageQuery {
    if ('ProsperCiStorage' -as [type]) { return }
    # GetCompressedFileSizeW reports storage bytes, including compressed/sparse files.
    # https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-getcompressedfilesizew
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class ProsperCiStorage {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, ExactSpelling = true, SetLastError = true)]
    private static extern uint GetCompressedFileSizeW(string path, out uint high);
    [DllImport("kernel32.dll", ExactSpelling = true, SetLastError = true)]
    private static extern void SetLastError(uint error);
    public static ulong Decode(uint low, uint high, int error) {
        // INVALID_FILE_SIZE is also a valid low word. Only a nonzero error means failure.
        if (low == UInt32.MaxValue && error != 0)
            throw new Win32Exception(error, "GetCompressedFileSizeW failed");
        return ((ulong)high << 32) | low;
    }
    public static string ExtendedPath(string fullPath) {
        // Read supplies an absolute path. Avoid depending on the host's long-path opt-in.
        if (fullPath.StartsWith(@"\\?\", StringComparison.Ordinal)) return fullPath;
        fullPath = fullPath.Replace('/', '\\');
        if (fullPath.StartsWith(@"\\", StringComparison.Ordinal))
            return @"\\?\UNC\" + fullPath.Substring(2);
        return @"\\?\" + fullPath;
    }
    public static ulong Read(string path) {
        path = ExtendedPath(System.IO.Path.GetFullPath(path));
        SetLastError(0);
        uint high;
        uint low = GetCompressedFileSizeW(path, out high);
        int error = Marshal.GetLastWin32Error();
        return Decode(low, high, error);
    }
}
'@
}

function Get-TreeStorage {
    param([string]$Label, [string]$Directory,
          [scriptblock]$SizeReader = { param($Path) [ProsperCiStorage]::Read($Path) })
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        return [ordered]@{ label = $Label; exists = $false }
    }
    $groups = [ordered]@{}
    $skipped = 0
    # PowerShell 7 does not recurse into directory links without -FollowSymlink.
    foreach ($file in Get-ChildItem -LiteralPath $Directory -Recurse -File -Force) {
        if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            ++$skipped
            continue
        }
        $kind = switch ($file.Extension.ToLowerInvariant()) {
            '.exe' { 'executables' }
            '.a' { 'archives' }
            '.o' { 'objects' }
            '.obj' { 'objects' }
            default { 'other' }
        }
        if (-not $groups.Contains($kind)) {
            $groups[$kind] = [ordered]@{ files = 0; logical_bytes = [uint64]0; storage_bytes = [uint64]0 }
        }
        $bytes = & $SizeReader $file.FullName
        # A failed query throws; never turn unavailable measurements into zero bytes.
        if ($null -eq $bytes) { throw 'Storage query returned no result' }
        ++$groups[$kind].files
        $groups[$kind].logical_bytes += [uint64]$file.Length
        $groups[$kind].storage_bytes += [uint64]$bytes
    }
    return [ordered]@{ label = $Label; exists = $true; skipped_links = $skipped; groups = $groups }
}

function Test-StorageProbe {
    param([string]$Path, [string]$Expected)
    $logical = (Get-Item -LiteralPath $Path).Length
    $beforeFlush = [ProsperCiStorage]::Read($Path)
    # Closing a cached writer does not establish persisted allocation. Flush only this
    # workflow-owned synthetic control, never the build/cache/temporary roots we observe.
    & "$PSScriptRoot/flush_link_output.ps1" -OutputFile $Path
    $storage = [ProsperCiStorage]::Read($Path)
    Write-Host "Storage control $Expected observed: logical=$logical before_flush=$beforeFlush storage=$storage"
    if ($logical -le 0) { throw 'Storage control must contain bytes' }
    if ($Expected -eq 'uncompressed' -and $storage -ne $logical) {
        throw 'Uncompressed storage control disagrees with logical size'
    }
    if ($Expected -eq 'compressed' -and $storage -ge $logical) {
        throw 'Compressible storage control did not shrink'
    }
    Write-Host "Storage control $Expected passed: logical=$logical storage=$storage"
}

if ($MyInvocation.InvocationName -eq '.') { return }
Initialize-StorageQuery
if ($ProbeFile) {
    if (-not $ExpectedProbe) { throw 'Probe requires its expected compression state' }
    Test-StorageProbe -Path $ProbeFile -Expected $ExpectedProbe
    return
}
if (-not $env:GITHUB_WORKSPACE) { throw 'GITHUB_WORKSPACE is required' }
$roots = [ordered]@{
    build = Join-Path $env:GITHUB_WORKSPACE 'prosper/build-ci'
    ccache = Join-Path $env:GITHUB_WORKSPACE '.ccache'
    runner_temp = $env:RUNNER_TEMP
    native_temp = $env:TEMP
    compiler_temp = $CompilerTempDirectory
}
$seen = [Collections.Generic.Dictionary[string,string]]::new([StringComparer]::OrdinalIgnoreCase)
$reports = @()
foreach ($label in $roots.Keys) {
    $directory = $roots[$label]
    if (-not $directory) {
        $reports += [ordered]@{ label = $label; configured = $false }
        continue
    }
    $canonical = [IO.Path]::GetFullPath($directory).TrimEnd([IO.Path]::DirectorySeparatorChar)
    if ($seen.ContainsKey($canonical)) {
        $reports += [ordered]@{ label = $label; alias_of = $seen[$canonical] }
        continue
    }
    $seen[$canonical] = $label
    if (Test-Path -LiteralPath $directory -PathType Container) {
        $drive = (Get-Item -LiteralPath $directory).PSDrive.Name
        Get-PSDrive -Name $drive | Select-Object Name, Used, Free | Format-Table | Out-Host
    }
    $reports += Get-TreeStorage -Label $label -Directory $directory
}
# Roots can overlap; these are separate observations, never a summed volume allocation.
[ordered]@{ phase = $Phase; roots = $reports } | ConvertTo-Json -Depth 8
