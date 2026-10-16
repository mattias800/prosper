#requires -Version 7.0
# Qualify storage accounting and the executable-link flush without needing Windows.
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/windows_build_storage.ps1"
. "$PSScriptRoot/flush_link_output.ps1"
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

    # The link hook must preserve actual bytes and timestamps, handle literal filenames,
    # and refuse an absent output instead of creating one. These run on every host.
    $linked = Join-Path $scratch '[linked output] space.exe'
    [IO.File]::WriteAllBytes($linked, [byte[]](0, 255, 13, 10, 42, 1, 2, 3))
    $hash = (Get-FileHash -LiteralPath $linked).Hash
    $modified = [IO.File]::GetLastWriteTimeUtc($linked)
    Flush-LinkedOutput -LiteralPath $linked
    Assert-Equal (Get-FileHash -LiteralPath $linked).Hash $hash
    Assert-Equal ([IO.File]::GetLastWriteTimeUtc($linked)) $modified
    Assert-Throws { Flush-LinkedOutput -LiteralPath $scratch } 'regular file'
    Assert-Throws { Flush-LinkedOutput -LiteralPath (Join-Path $scratch 'missing.exe') } 'Cannot find path'
    Assert-Throws { & "$PSScriptRoot/flush_link_output.ps1" -OutputFile (Join-Path $scratch 'missing.exe') } 'Cannot find path'
    Assert-Equal (Test-Path -LiteralPath (Join-Path $scratch 'missing.exe')) $false
    Assert-Throws { & "$PSScriptRoot/flush_link_output.ps1" } 'path is required'

    # Exercise the real compiler/linker rule, including paths with spaces. The failed-launcher
    # arm proves a post-link flush failure stops the build after an actual executable exists.
    $fixture = Join-Path $scratch 'link fixture'
    New-Item -ItemType Directory -Path $fixture | Out-Null
    $module = (Join-Path $PSScriptRoot '../../cmake/FlushLinkedOutput.cmake') -replace '\\', '/'
    $powershell = (Get-Process -Id $PID).Path -replace '\\', '/'
    $cmake = (Get-Command cmake).Source -replace '\\', '/'
    @'
cmake_minimum_required(VERSION 3.20)
project(link_flush_fixture LANGUAGES CXX)
option(FLUSH "Exercise the link flush" ON)
if(FLUSH)
  include("@MODULE@")
  prosper_flush_link_outputs("${LAUNCHER}")
endif()
add_executable(probe probe.cpp)
set_target_properties(probe PROPERTIES OUTPUT_NAME "probe [space]")
add_custom_command(TARGET probe POST_BUILD COMMAND "${CMAKE_COMMAND}" -E echo AFTER_LINK_FLUSH)
'@.Replace('@MODULE@', $module) | Set-Content -LiteralPath (Join-Path $fixture 'CMakeLists.txt')
    'int main() { return 0; }' | Set-Content -LiteralPath (Join-Path $fixture 'probe.cpp')
    foreach ($mode in @('on', 'off', 'failure')) {
        $build = Join-Path $scratch "build $mode"
        $launcher = if ($mode -eq 'failure') { $cmake } else { $powershell }
        $enabled = if ($mode -eq 'off') { 'OFF' } else { 'ON' }
        $configureLog = & cmake -S $fixture -B $build -G Ninja "-DFLUSH=$enabled" "-DLAUNCHER=$launcher" 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Link fixture configure failed: $configureLog" }
        $buildLog = (& cmake --build $build --parallel 1 2>&1) -join "`n"
        $buildExit = $LASTEXITCODE
        $filename = if ($IsWindows) { 'probe [space].exe' } else { 'probe [space]' }
        Assert-Equal (Test-Path -LiteralPath (Join-Path $build $filename) -PathType Leaf) $true
        if ($mode -eq 'failure') {
            Assert-Equal ($buildExit -ne 0) $true
            Assert-Equal ([regex]::IsMatch($buildLog, '(?m)^AFTER_LINK_FLUSH\r?$')) $false
        } else {
            if ($buildExit -ne 0) { throw "Link fixture build failed: $buildLog" }
            Assert-Equal ([regex]::IsMatch($buildLog, '(?m)^AFTER_LINK_FLUSH\r?$')) $true
            Assert-Equal $buildLog.Contains('Linked output flush complete') ($mode -eq 'on')
            if ($mode -eq 'on') {
                Assert-Equal ($buildLog.IndexOf('Linked output flush complete') -lt $buildLog.IndexOf('AFTER_LINK_FLUSH')) $true
            }
        }
    }
} finally {
    Remove-Item -LiteralPath $scratch -Recurse -Force
}
Write-Host "Storage accounting controls: $checks passed"
