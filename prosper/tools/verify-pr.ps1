[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('docs', 'core', 'renderer')]
    [string] $Profile,

    [string] $Base = 'origin/master',
    [string] $LinuxBuild = 'prosper/build-linux',
    [string] $WindowsBuild = 'prosper/build-windows',
    [string[]] $Snapshot = @(),
    [int] $Jobs = 8,
    [switch] $DryRun,
    [string] $EvidenceFile,
    [int] $Pr = 0,
    [int] $TestPauseAfterCaptureSeconds = 0
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Prosper = Join-Path $Repo 'prosper'
$Results = @()
$Notes = @()
$Succeeded = $false
$HeadSha = $null
$TreeSha = $null
$BaseSha = $null

function Invoke-Capture {
    param(
        [Parameter(Mandatory = $true)] [string] $Executable,
        [Parameter(Mandatory = $true)] [string[]] $Arguments,
        [Parameter(Mandatory = $true)] [string] $WorkingDirectory
    )

    Push-Location $WorkingDirectory
    try {
        $Output = & $Executable @Arguments 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "$Executable failed with exit code $LASTEXITCODE`: $($Output -join [Environment]::NewLine)"
        }
        return ($Output -join [Environment]::NewLine).Trim()
    }
    finally {
        Pop-Location
    }
}

function Format-Command {
    param([string] $Executable, [string[]] $Arguments)

    $Parts = @($Executable) + $Arguments
    return (($Parts | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        }
        else {
            $_
        }
    }) -join ' ')
}

function ConvertTo-BashLiteral {
    param([string] $Value)
    $Quote = [string][char]39
    $Replacement = $Quote + '"' + $Quote + '"' + $Quote
    return $Quote + $Value.Replace($Quote, $Replacement) + $Quote
}

function Invoke-AuthorCheck {
    param(
        [Parameter(Mandatory = $true)] [string] $Label,
        [Parameter(Mandatory = $true)] [string] $Executable,
        [Parameter(Mandatory = $true)] [string[]] $Arguments,
        [Parameter(Mandatory = $true)] [string] $WorkingDirectory
    )

    $Display = Format-Command $Executable $Arguments
    Write-Host "`n==> $Label"
    Write-Host "`$ $Display"
    if ($DryRun) {
        $script:Results += [pscustomobject]@{
            Label = $Label; Command = $Display; Status = 'DRY-RUN'; Seconds = 0; Detail = ''
        }
        return
    }

    $Watch = [Diagnostics.Stopwatch]::StartNew()
    Push-Location $WorkingDirectory
    try {
        $Lines = @(& $Executable @Arguments 2>&1 | ForEach-Object {
            Write-Host $_
            $_.ToString()
        })
        $ExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
        $Watch.Stop()
    }

    $Output = $Lines -join [Environment]::NewLine
    $Match = [regex]::Match($Output, '(\d+)% tests passed, (\d+) tests failed out of (\d+)')
    $Detail = ''
    if ($Match.Success) {
        $Passed = [int]$Match.Groups[3].Value - [int]$Match.Groups[2].Value
        $Detail = "$Passed/$($Match.Groups[3].Value) tests passed"
    }
    $Status = if ($ExitCode -eq 0) { 'PASS' } else { 'FAIL' }
    $script:Results += [pscustomobject]@{
        Label = $Label
        Command = $Display
        Status = $Status
        Seconds = $Watch.Elapsed.TotalSeconds
        Detail = $Detail
    }
    if ($ExitCode -ne 0) {
        throw "$Label failed with exit code $ExitCode"
    }
}

if ($Profile -eq 'renderer' -and $Snapshot.Count -eq 0) {
    throw 'renderer requires at least one -Snapshot NAME (or -Snapshot all)'
}
if ($Profile -ne 'renderer' -and $Snapshot.Count -ne 0) {
    throw '-Snapshot is only valid with the renderer profile'
}
if ($Snapshot -contains 'all' -and -not ($Snapshot.Count -eq 1 -and $Snapshot[0] -eq 'all')) {
    throw 'use -Snapshot all by itself'
}
if ($Pr -ne 0 -and $DryRun) {
    throw '-Pr cannot be combined with -DryRun'
}
if ($Jobs -lt 1) {
    throw '-Jobs must be positive'
}
if ($TestPauseAfterCaptureSeconds -lt 0) {
    throw '-TestPauseAfterCaptureSeconds cannot be negative'
}

try {
    $HeadSha = Invoke-Capture git @('rev-parse', 'HEAD') $Repo
    $TreeSha = Invoke-Capture git @('rev-parse', 'HEAD^{tree}') $Repo
    $BaseSha = Invoke-Capture git @('rev-parse', $Base) $Repo
    $Status = Invoke-Capture git @('status', '--porcelain') $Repo
    if ($Status) {
        throw 'commit or remove all nonignored changes before author verification'
    }

    try {
        $Upstream = Invoke-Capture git @('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}') $Repo
        $UpstreamSha = Invoke-Capture git @('rev-parse', '@{upstream}') $Repo
    }
    catch {
        throw 'push the PR branch and configure its upstream before verification'
    }
    if ($UpstreamSha -ne $HeadSha) {
        throw "pushed upstream $Upstream is $UpstreamSha, not local HEAD $HeadSha"
    }

    if ($TestPauseAfterCaptureSeconds) {
        Start-Sleep -Seconds $TestPauseAfterCaptureSeconds
    }

    Invoke-AuthorCheck 'base-to-head whitespace check' git @(
        'diff', '--check', "$BaseSha...$HeadSha"
    ) $Repo

    if ($Profile -in @('core', 'renderer')) {
        if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
            throw 'wsl.exe is required for the Linux verification on this Windows host'
        }
        $LinuxBuildPath = (Resolve-Path (Join-Path $Repo $LinuxBuild)).Path
        $QuotedWindowsLinuxBuild = ConvertTo-BashLiteral $LinuxBuildPath
        $LinuxBuildWsl = Invoke-Capture wsl.exe @(
            '-d', 'Ubuntu-24.04', '-u', 'root', '--',
            'bash', '-lc', "wslpath -a $QuotedWindowsLinuxBuild"
        ) $Repo
        $QuotedLinuxBuild = ConvertTo-BashLiteral $LinuxBuildWsl
        Invoke-AuthorCheck 'Linux build' wsl.exe @(
            '-d', 'Ubuntu-24.04', '-u', 'root', '--',
            'bash', '-lc', "cmake --build $QuotedLinuxBuild -j$Jobs"
        ) $Repo
        Invoke-AuthorCheck 'Linux ctest' wsl.exe @(
            '-d', 'Ubuntu-24.04', '-u', 'root', '--',
            'bash', '-lc', "ctest --test-dir $QuotedLinuxBuild --output-on-failure"
        ) $Repo

        $WindowsBuildPath = (Resolve-Path (Join-Path $Repo $WindowsBuild)).Path
        Invoke-AuthorCheck 'Windows build' cmake @(
            '--build', $WindowsBuildPath, '--parallel', $Jobs.ToString()
        ) $Repo
        Invoke-AuthorCheck 'Windows ctest' ctest @(
            '--test-dir', $WindowsBuildPath, '--output-on-failure'
        ) $Repo
    }

    if ($Profile -eq 'renderer') {
        $QuotedWindowsProsper = ConvertTo-BashLiteral $Prosper
        $ProsperWsl = Invoke-Capture wsl.exe @(
            '-d', 'Ubuntu-24.04', '-u', 'root', '--',
            'bash', '-lc', "wslpath -a $QuotedWindowsProsper"
        ) $Repo
        $SnapshotNames = if ($Snapshot.Count -eq 1 -and $Snapshot[0] -eq 'all') { @() } else { $Snapshot }
        $SnapshotArguments = @('python3', 'tools/snapshot/snapshot.py', 'check') + $SnapshotNames
        $SnapshotEnvironment = @(
            'PROSPER_BOOT_TRACE=' + (ConvertTo-BashLiteral "$LinuxBuildWsl/boot_trace")
            'PROSPER_SCREENSHOT=' + (ConvertTo-BashLiteral "$LinuxBuildWsl/screenshot")
        ) -join ' '
        $SnapshotShell = $SnapshotEnvironment + ' ' +
            (($SnapshotArguments | ForEach-Object { ConvertTo-BashLiteral $_ }) -join ' ')
        $SnapshotLabel = if ($SnapshotNames.Count -eq 0) {
            'snapshot guard: all'
        }
        else {
            'snapshot guard: ' + ($SnapshotNames -join ', ')
        }
        Invoke-AuthorCheck $SnapshotLabel wsl.exe @(
            '-d', 'Ubuntu-24.04', '-u', 'root', '--',
            'bash', '-lc', "cd $(ConvertTo-BashLiteral $ProsperWsl) && $SnapshotShell"
        ) $Prosper
    }

    $FinalHead = Invoke-Capture git @('rev-parse', 'HEAD') $Repo
    $FinalUpstream = Invoke-Capture git @('rev-parse', '@{upstream}') $Repo
    $FinalBase = Invoke-Capture git @('rev-parse', $Base) $Repo
    $FinalStatus = Invoke-Capture git @('status', '--porcelain') $Repo
    if ($FinalHead -ne $HeadSha -or $FinalUpstream -ne $HeadSha -or
        $FinalBase -ne $BaseSha -or $FinalStatus) {
        throw 'HEAD, pushed upstream, base ref, or nonignored worktree state changed during verification'
    }
    $Succeeded = $true
}
catch {
    Write-Host "`nERROR: $($_.Exception.Message)" -ForegroundColor Red
}

if (-not $HeadSha) {
    exit 1
}

$State = if (-not $Succeeded) { 'FAIL' } elseif ($DryRun) { 'DRY-RUN' } else { 'PASS' }
$Evidence = @(
    "### AUTHOR VERIFICATION - $State"
    ''
    "- Head: ``$HeadSha``"
    "- Tree: ``$TreeSha``"
    "- Compared with: ``$Base@$BaseSha``"
    ''
    'Checks:'
)
foreach ($Result in $Results) {
    $Elapsed = if ($Result.Seconds) { ", $([math]::Round($Result.Seconds, 1))s" } else { '' }
    $Detail = if ($Result.Detail) { ", $($Result.Detail)" } else { '' }
    $Evidence += "- **$($Result.Status)** - $($Result.Label) (``$($Result.Command)``$Detail$Elapsed)"
}
if ($Notes.Count -ne 0) {
    $Evidence += ''
    $Evidence += 'Notes:'
    $Evidence += $Notes | ForEach-Object { "- $_" }
}
$Evidence += ''
$Evidence += 'Reviewer note: this is author-owned verification; inspect its coverage but do not rerun it.'
$EvidenceText = ($Evidence -join [Environment]::NewLine) + [Environment]::NewLine
Write-Host "`n$EvidenceText"

if ($EvidenceFile) {
    $EvidenceParent = Split-Path -Parent $EvidenceFile
    if ($EvidenceParent) {
        New-Item -ItemType Directory -Force -Path $EvidenceParent | Out-Null
    }
    Set-Content -LiteralPath $EvidenceFile -Value $EvidenceText -Encoding utf8
    Write-Host "Wrote verification record to $EvidenceFile"
}

if ($Succeeded -and $Pr) {
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        throw 'gh is required to post the verification comment'
    }
    $PrHead = Invoke-Capture gh @('pr', 'view', $Pr.ToString(), '--json', 'headRefOid', '--jq', '.headRefOid') $Repo
    if ($PrHead -ne $HeadSha) {
        throw "PR #$Pr head is $PrHead, not verified HEAD $HeadSha"
    }
    $TemporaryEvidence = [IO.Path]::GetTempFileName()
    try {
        Set-Content -LiteralPath $TemporaryEvidence -Value $EvidenceText -Encoding utf8
        Push-Location $Repo
        try {
            & gh pr comment $Pr --body-file $TemporaryEvidence
            if ($LASTEXITCODE -ne 0) {
                throw "gh pr comment failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            Pop-Location
        }
    }
    finally {
        Remove-Item -LiteralPath $TemporaryEvidence -Force -ErrorAction SilentlyContinue
    }
}

if ($Succeeded) { exit 0 } else { exit 1 }
