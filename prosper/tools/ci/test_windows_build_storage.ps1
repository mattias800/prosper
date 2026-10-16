#requires -Version 7.0
# Qualify storage accounting without Windows: invoke the real decoder and inject file sizes.
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/windows_build_storage.ps1"
Initialize-StorageQuery
$checks = 0
function Assert-Equal($Actual, $Expected) {
    if ($Actual -ne $Expected) { throw "Expected $Expected, got $Actual" }
    $script:checks++
}
function Assert-Throws([scriptblock]$Action, [string]$Message) {
    $caught = $null
    try { & $Action } catch { $caught = $_ }
    if ($null -eq $caught -or "$caught" -notlike "*$Message*") {
        throw "Expected failure containing: $Message"
    }
    $script:checks++
}
# A nonzero high word must survive; the sentinel low word is valid when error is zero.
Assert-Equal ([ProsperCiStorage]::Decode(17, 2, 0)) ([uint64]8589934609)
Assert-Equal ([ProsperCiStorage]::Decode([uint32]::MaxValue, 1, 0)) ([uint64]8589934591)
Assert-Equal ([ProsperCiStorage]::Decode(0, 0, 0)) 0
Assert-Throws { [ProsperCiStorage]::Decode([uint32]::MaxValue, 0, 5) } 'GetCompressedFileSizeW failed'

# Exercise Windows path spelling on every host, independently of its filesystem/opt-in.
$longDrivePath = 'C:\build\' + ('nested\' * 40) + 'output.obj'
$longUncPath = '\\server\share\' + ('nested\' * 40) + 'output.obj'
Assert-Equal ([ProsperCiStorage]::ExtendedPath($longDrivePath)) ('\\?\' + $longDrivePath)
Assert-Equal ([ProsperCiStorage]::ExtendedPath($longUncPath)) ('\\?\UNC\' + $longUncPath.Substring(2))
Assert-Equal ([ProsperCiStorage]::ExtendedPath('\\?\C:\build\output.obj')) '\\?\C:\build\output.obj'
Assert-Equal ([ProsperCiStorage]::ExtendedPath('\\?\UNC\server\share\output.obj')) '\\?\UNC\server\share\output.obj'
Assert-Equal ([ProsperCiStorage]::ExtendedPath('C:/build/output.obj')) '\\?\C:\build\output.obj'

$scratch = Join-Path ([IO.Path]::GetTempPath()) ('prosper-storage-' + [guid]::NewGuid())
try {
    New-Item -ItemType Directory -Path $scratch | Out-Null
    [IO.File]::WriteAllBytes((Join-Path $scratch 'a.exe'), [byte[]]::new(128))
    [IO.File]::WriteAllBytes((Join-Path $scratch 'b.obj'), [byte[]]::new(64))
    [IO.File]::WriteAllBytes((Join-Path $scratch 'c.o'), [byte[]]::new(32))
    [IO.File]::WriteAllBytes((Join-Path $scratch 'd.a'), [byte[]]::new(16))
    [IO.File]::WriteAllBytes((Join-Path $scratch '.hidden'), [byte[]]::new(8))
    $report = Get-TreeStorage -Label 'synthetic' -Directory $scratch -SizeReader {
        param($Path)
        # Deliberately different from Length: copying logical size cannot satisfy this control.
        [uint64]((Get-Item -LiteralPath $Path -Force).Length / 8)
    }
    Assert-Equal $report.label 'synthetic'
    Assert-Equal $report.exists $true
    Assert-Equal $report.groups.executables.logical_bytes 128
    Assert-Equal $report.groups.executables.storage_bytes 16
    Assert-Equal $report.groups.objects.files 2
    Assert-Equal $report.groups.objects.logical_bytes 96
    Assert-Equal $report.groups.objects.storage_bytes 12
    Assert-Equal $report.groups.archives.storage_bytes 2
    Assert-Equal $report.groups.other.storage_bytes 1
    Assert-Equal $report.skipped_links 0
    Assert-Throws { Get-TreeStorage -Label 'bad' -Directory $scratch -SizeReader { throw 'query failed' } } 'query failed'
    Assert-Throws { Get-TreeStorage -Label 'bad' -Directory $scratch -SizeReader { $null } } 'no result'
    $missing = Get-TreeStorage -Label 'missing' -Directory (Join-Path $scratch 'absent')
    Assert-Equal $missing.exists $false
    if ($missing.Contains('groups')) { throw 'Missing root must not claim measured zero groups' }
    ++$checks
} finally {
    Remove-Item -LiteralPath $scratch -Recurse -Force
}
Write-Host "Storage accounting controls: $checks passed"
