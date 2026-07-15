[CmdletBinding()]
param(
    [string]$Dump = 'C:\Users\matti\repos\ps5ys\PPSA15552-app0',
    [string]$BuildDir,
    [string]$OutputDir,
    [ValidateSet('all', 'headless-full', 'app-full', 'headless-publish-10',
                 'headless-render-every-10', 'headless-no-graphics', 'headless-no-gpu-work')]
    [string]$Mode = 'all',
    [int]$FullTimeoutSeconds = 240,
    [int]$FastTimeoutSeconds = 90
)

$ErrorActionPreference = 'Stop'

if (-not $IsWindows -and $PSVersionTable.PSEdition -eq 'Core') {
    throw 'progression-matrix.ps1 must run in native Windows PowerShell.'
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$prosperRoot = Join-Path $repoRoot 'prosper'
if (-not $BuildDir) { $BuildDir = Join-Path $prosperRoot 'build-mingw-app' }
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
$Dump = (Resolve-Path -LiteralPath $Dump).Path

if (-not $OutputDir) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $OutputDir = Join-Path $env:TEMP "prosper-dead-cells-matrix-$stamp"
}
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$bootTrace = Join-Path $BuildDir 'boot_trace.exe'
$app = Join-Path $BuildDir 'prosper-app.exe'
$timelineTool = Join-Path $BuildDir 'gpu_timeline.exe'
foreach ($binary in @($bootTrace, $app)) {
    if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
        throw "Required binary was not found: $binary"
    }
}

$fullRoute = [IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot 'reach-first-gameplay-full-render.pad'))
$captureRoute = [IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot 'reach-first-gameplay-capture.pad'))

$controlledEnvironment = @(
    'PROSPER_CAPTURE_TITLE', 'PROSPER_GUEST_FS', 'PROSPER_GUEST_ARGS',
    'PROSPER_PAD_SCRIPT', 'PROSPER_PAD_SCRIPT_LOG', 'PROSPER_SAVEDATA_DIR',
    'PROSPER_PROGRESS', 'PROSPER_GPU_TIMELINE', 'PROSPER_RENDER',
    'PROSPER_NO_FRAME_DUMPS', 'PROSPER_PRESENT_EVERY', 'PROSPER_RENDER_EVERY',
    'PROSPER_NO_COMPUTE',
    'PROSPER_RENDER_EVERY_FOR_MS', 'PROSPER_RENDER_DELAY_MS', 'PROSPER_RENDER_FIRST',
    'PROSPER_RENDER_TIMING', 'PROSPER_FRAME_DIR'
)
$savedEnvironment = @{}
foreach ($name in $controlledEnvironment) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

function Reset-ControlledEnvironment {
    foreach ($name in $controlledEnvironment) {
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
}

function Set-RunEnvironment {
    param($Spec, [string]$RunDir, [string]$Timeline)

    Reset-ControlledEnvironment
    $common = @{
        PROSPER_CAPTURE_TITLE = 'PPSA15552'
        PROSPER_GUEST_FS = '1'
        PROSPER_GUEST_ARGS = ''
        PROSPER_PAD_SCRIPT = "@$($Spec.Route)"
        PROSPER_PAD_SCRIPT_LOG = '1'
        PROSPER_SAVEDATA_DIR = (Join-Path $RunDir 'savedata')
        PROSPER_PROGRESS = '2'
        PROSPER_GPU_TIMELINE = $Timeline
    }
    foreach ($entry in $common.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
    foreach ($entry in $Spec.Environment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
    New-Item -ItemType Directory -Path $common.PROSPER_SAVEDATA_DIR -Force | Out-Null
}

function Get-LastMatch {
    param([string[]]$Paths, [string]$Pattern)
    $matches = Select-String -Path $Paths -Pattern $Pattern -ErrorAction SilentlyContinue
    if ($matches) { return $matches[-1].Line.Trim() }
    return ''
}

$specs = @(
    [pscustomobject]@{
        Name = 'headless-full'; Program = $bootTrace; Route = $fullRoute
        Timeout = $FullTimeoutSeconds
        Environment = @{ PROSPER_RENDER = '1'; PROSPER_NO_FRAME_DUMPS = '1' }
        Meaning = 'Full GPU execution and publication through boot_trace.'
    },
    [pscustomobject]@{
        Name = 'app-full'; Program = $app; Route = $fullRoute
        Timeout = $FullTimeoutSeconds
        Environment = @{ PROSPER_RENDER = '1' }
        Meaning = 'Full GPU execution/publication through SDL video, audio, and input frontend.'
    },
    [pscustomobject]@{
        Name = 'headless-publish-10'; Program = $bootTrace; Route = $fullRoute
        Timeout = $FullTimeoutSeconds
        Environment = @{
            PROSPER_RENDER = '1'; PROSPER_NO_FRAME_DUMPS = '1'; PROSPER_PRESENT_EVERY = '10'
        }
        Meaning = 'Full GPU execution; publish only one in ten completed frames.'
    },
    [pscustomobject]@{
        Name = 'headless-render-every-10'; Program = $bootTrace; Route = $captureRoute
        Timeout = $FastTimeoutSeconds
        Environment = @{
            PROSPER_RENDER = '1'; PROSPER_NO_FRAME_DUMPS = '1'; PROSPER_RENDER_EVERY = '10'
        }
        Meaning = 'Diagnostic: skip graphics execution on nine in ten draw submits.'
    },
    [pscustomobject]@{
        Name = 'headless-no-graphics'; Program = $bootTrace; Route = $captureRoute
        Timeout = $FastTimeoutSeconds
        Environment = @{}
        Meaning = 'Diagnostic: no graphics backend; retained compute still executes.'
    },
    [pscustomobject]@{
        Name = 'headless-no-gpu-work'; Program = $bootTrace; Route = $captureRoute
        Timeout = $FastTimeoutSeconds
        Environment = @{ PROSPER_NO_COMPUTE = '1' }
        Meaning = 'Diagnostic: no graphics or compute execution; semantic dispatches remain indexed.'
    }
)
if ($Mode -ne 'all') { $specs = @($specs | Where-Object Name -eq $Mode) }

$results = @()
try {
    foreach ($spec in $specs) {
        $runDir = Join-Path $OutputDir $spec.Name
        New-Item -ItemType Directory -Path $runDir -Force | Out-Null
        $stdout = Join-Path $runDir 'stdout.log'
        $stderr = Join-Path $runDir 'stderr.log'
        $timeline = Join-Path $runDir 'run.prgtl'
        Set-RunEnvironment $spec $runDir $timeline

        $arguments = if ($spec.Program -eq $app) { @('--dump', $Dump) } else { @($Dump) }
        Write-Host "[$($spec.Name)] $($spec.Meaning)"
        $started = Get-Date
        $process = Start-Process -FilePath $spec.Program -ArgumentList $arguments -PassThru `
            -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $completed = $process.WaitForExit($spec.Timeout * 1000)
        if (-not $completed) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            $process.WaitForExit()
        } else {
            # Drain redirected output and populate ExitCode on all supported PowerShell runtimes.
            $process.WaitForExit()
        }
        $process.Refresh()
        $exitCode = $process.ExitCode
        $elapsed = [Math]::Round(((Get-Date) - $started).TotalSeconds, 1)
        $paths = @($stdout, $stderr)

        $selectorOutput = ''
        $selectorExit = $null
        if ((Test-Path -LiteralPath $timeline -PathType Leaf) -and
            (Test-Path -LiteralPath $timelineTool -PathType Leaf)) {
            $selectorStdout = Join-Path $runDir 'selector.stdout.log'
            $selectorStderr = Join-Path $runDir 'selector.stderr.log'
            $selectorProcess = Start-Process -FilePath $timelineTool -ArgumentList @(
                $timeline, '--select', '636x420', '77:85', '91:94', '8'
            ) -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $selectorStdout `
                -RedirectStandardError $selectorStderr
            $selectorExit = $selectorProcess.ExitCode
            $selectorLines = @()
            if (Test-Path -LiteralPath $selectorStdout) {
                $selectorLines += Get-Content -LiteralPath $selectorStdout
            }
            if (Test-Path -LiteralPath $selectorStderr) {
                $selectorLines += Get-Content -LiteralPath $selectorStderr
            }
            $selectorOutput = ($selectorLines -join [Environment]::NewLine).Trim()
            $selectorLines | Set-Content -LiteralPath (Join-Path $runDir 'selector.log')
        }

        $result = [pscustomobject]@{
            mode = $spec.Name
            meaning = $spec.Meaning
            timed_out = -not $completed
            termination = if ($completed) { 'process_exit' } else { 'timeout_killed' }
            exit_code = $exitCode
            elapsed_seconds = $elapsed
            loading_level = Get-LastMatch $paths 'Loading level PrisonStart'
            parseall = Get-LastMatch $paths 'PARSEALL TOOK'
            last_progress = Get-LastMatch $paths '^\[progress\]'
            semantic_selector_exit = $selectorExit
            semantic_selector = $selectorOutput
            run_dir = $runDir
        }
        $results += $result
        Write-Host "[$($spec.Name)] termination=$($result.termination) exit=$($result.exit_code) elapsed=${elapsed}s"
        if ($result.loading_level) { Write-Host "  $($result.loading_level)" }
        if ($result.parseall) { Write-Host "  $($result.parseall)" }
        if ($result.last_progress) { Write-Host "  $($result.last_progress)" }
        if ($selectorOutput) { Write-Host "  selector: $($selectorOutput -replace "`r?`n", ' | ')" }
    }
}
finally {
    Reset-ControlledEnvironment
    foreach ($name in $controlledEnvironment) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
}

$summaryPath = Join-Path $OutputDir 'summary.json'
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $summaryPath
Write-Host "Matrix complete: $summaryPath"
$results | Format-Table mode, timed_out, exit_code, elapsed_seconds, loading_level, parseall -AutoSize
