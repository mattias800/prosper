[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Verifier = Join-Path $PSScriptRoot 'verify-pr.ps1'
$Passed = 0

function Assert-True {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) {
        throw $Message
    }
    $script:Passed++
    Write-Host "PASS: $Message"
}

function Invoke-Verifier {
    param([string[]] $Arguments)

    $PreviousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $Output = @(& powershell -NoProfile -ExecutionPolicy Bypass -File $Verifier @Arguments 2>&1 |
            ForEach-Object { $_.ToString() })
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorAction
    }
    return [pscustomobject]@{
        ExitCode = $ExitCode
        Output = $Output -join [Environment]::NewLine
    }
}

$Tokens = $null
$Errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    $Verifier,
    [ref]$Tokens,
    [ref]$Errors
) | Out-Null
Assert-True ($Errors.Count -eq 0) 'verifier parses without PowerShell syntax errors'

$HeadSha = (& git -C $Repo rev-parse HEAD).Trim()
$BaseSha = (& git -C $Repo rev-parse origin/master).Trim()
$ProvenanceBuildRelative = 'prosper/build-linux/verify-pr-provenance'
$ProvenanceBuild = Join-Path $Repo 'prosper\build-linux\verify-pr-provenance'
$PreviousBootTrace = $env:PROSPER_BOOT_TRACE
$PreviousScreenshot = $env:PROSPER_SCREENSHOT
try {
    New-Item -ItemType Directory -Path $ProvenanceBuild -Force | Out-Null
    $env:PROSPER_BOOT_TRACE = '/tmp/hostile-boot-trace'
    $env:PROSPER_SCREENSHOT = '/tmp/hostile-screenshot'
    $DryRun = Invoke-Verifier @(
        'renderer', '-Snapshot', 'messenger-scene',
        '-LinuxBuild', $ProvenanceBuildRelative, '-DryRun'
    )
}
finally {
    if ($null -eq $PreviousBootTrace) { Remove-Item Env:PROSPER_BOOT_TRACE -ErrorAction SilentlyContinue }
    else { $env:PROSPER_BOOT_TRACE = $PreviousBootTrace }
    if ($null -eq $PreviousScreenshot) { Remove-Item Env:PROSPER_SCREENSHOT -ErrorAction SilentlyContinue }
    else { $env:PROSPER_SCREENSHOT = $PreviousScreenshot }
    Remove-Item -LiteralPath $ProvenanceBuild -Recurse -Force -ErrorAction SilentlyContinue
}
Assert-True ($DryRun.ExitCode -eq 0) 'renderer dry run succeeds on a clean pushed head'
Assert-True ($DryRun.Output.Contains("git diff --check $BaseSha...$HeadSha")) `
    'diff check uses captured immutable base and head objects'
Assert-True ($DryRun.Output.Contains('wsl.exe -d Ubuntu-24.04 -u root -- bash -lc')) `
    'Linux build, test, and snapshot commands pin Ubuntu-24.04 as root'
Assert-True ($DryRun.Output.Contains('Windows build') -and
    $DryRun.Output.Contains('Windows ctest') -and
    $DryRun.Output.Contains('snapshot guard: messenger-scene')) `
    'renderer profile includes both platform checks and the selected snapshot'
Assert-True ($DryRun.Output.Contains('/prosper/build-linux/verify-pr-provenance/boot_trace') -and
    $DryRun.Output.Contains('/prosper/build-linux/verify-pr-provenance/screenshot') -and
    -not $DryRun.Output.Contains('/tmp/hostile-boot-trace') -and
    -not $DryRun.Output.Contains('/tmp/hostile-screenshot')) `
    'snapshot executables are pinned to the selected Linux build instead of inherited overrides'

$SkipAttempt = Invoke-Verifier @('core', '-SkipLinux', '-SkipWindows', '-DryRun')
Assert-True ($SkipAttempt.ExitCode -ne 0 -and
    $SkipAttempt.Output.Contains('parameter cannot be found')) `
    'removed skip switches cannot produce an incomplete PASS'

$UntrackedProbe = Join-Path $Repo 'prosper\src\verify_pr_untracked_probe.cpp'
try {
    [IO.File]::WriteAllText($UntrackedProbe, '// verifier cleanliness probe')
    $DirtyRun = Invoke-Verifier @('docs', '-DryRun')
    Assert-True ($DirtyRun.ExitCode -ne 0 -and
        $DirtyRun.Output.Contains('commit or remove all nonignored changes')) `
        'nonignored untracked source is rejected before verification'
}
finally {
    Remove-Item -LiteralPath $UntrackedProbe -Force -ErrorAction SilentlyContinue
}

$ProbeRef = "refs/codex/verify-pr-base-$PID"
$MoveJob = $null
try {
    & git -C $Repo update-ref $ProbeRef $BaseSha
    if ($LASTEXITCODE -ne 0) { throw 'failed to create verifier base probe ref' }
    $MoveJob = Start-Job -ScriptBlock {
        param($WorkingTree, $RefName, $NewValue)
        Start-Sleep -Seconds 1
        & git -C $WorkingTree update-ref $RefName $NewValue
    } -ArgumentList $Repo, $ProbeRef, $HeadSha
    $MovedBaseRun = Invoke-Verifier @(
        'docs', '-Base', $ProbeRef, '-DryRun', '-TestPauseAfterCaptureSeconds', '2'
    )
    Wait-Job $MoveJob | Out-Null
    Receive-Job $MoveJob | Out-Null
    Assert-True ($MovedBaseRun.ExitCode -ne 0 -and
        $MovedBaseRun.Output.Contains('base ref')) `
        'symbolic base movement after capture invalidates the evidence run'
}
finally {
    if ($MoveJob) { Remove-Job $MoveJob -Force -ErrorAction SilentlyContinue }
    & git -C $Repo update-ref -d $ProbeRef 2>$null
}

Write-Host "verify-pr self-test: $Passed assertions passed"
