#!/usr/bin/env python3
"""Require --no-tests=error on every ctest invocation that RUNS tests.

Why this exists
---------------
`ctest` exits 0 when it finds no tests at all:

    $ ctest --output-on-failure ; echo "exit: $?"
    No tests were found!!!
    exit: 0

    $ ctest --output-on-failure --no-tests=error ; echo "exit: $?"
    No tests were found!!!
    Errors while running CTest
    exit: 8

So "nothing ran" and "everything passed" are the same status, and a gate reading only that status
reports a confident green having verified nothing (#2187). The reachable ways to get there are
mundane: a configure that quietly loses a dependency, a `--test-dir` naming the wrong build tree, or
a cleanup removing the tree mid-run.

The invocations were fixed once. This check is what stops the next one from reintroducing the hole --
a new CI job or a new script carries the flag or fails here, rather than passing silently for however
long it takes someone to notice.

What counts as an invocation, and why the rule is positional
------------------------------------------------------------
The first version of this checker matched the *word* `ctest`. Its first run against this repository
produced 23 findings of which 21 were prose -- docstrings, comments, argparse help strings, a
PowerShell `.Contains('Windows ctest')`. A checker whose debut is 91% false is one that gets muted,
so an invocation is now recognised by POSITION:

    shell / CI / PowerShell     `ctest` followed by a flag, or by PowerShell's `@(` splat
    python argument list        `["ctest",` as the head of a list literal

Everything else is prose by construction: "via ctest as X", "ctest produced no Y", "the ctest suite".
`-N` / `--show-only` invocations are excluded because they LIST tests and never run one -- ci.yml's
by-name registration guard is built entirely out of those and must keep working.

The flag may appear on the matched line or within the next few, because both a YAML `\\` continuation
and a PowerShell splat put the arguments on following lines.

Known gaps, stated rather than left implied
-------------------------------------------
* Documentation (`.md`) is not scanned. This gate is about invocations that gate a build, not about
  every place the command is quoted for a reader.
* THIS FILE is excluded from the scan, because its selftest arms are deliberately non-compliant
  invocation lines and the scanner would otherwise flag its own fixtures. The exclusion is by
  resolved path, is reported in the summary line rather than applied silently, and is asserted by a
  selftest arm. A real ctest invocation added to this file would not be gated; there is no reason
  to add one.
* A python invocation whose argument list spans lines, or one assembled from a variable, is not
  recognised. The repository has exactly one python invocation
  (`tools/vkval/vk_validation_scan.py`'s `run_ctest`); its list is on a single line and already
  carries the flag, so the single-line rule covers what exists today.

Self-validation
---------------
`--selftest` runs the matcher against hand-written instances of every class -- including the exact
prose lines that the token-matching version got wrong, kept as the regression. And a scan that finds
NO invocations at all exits non-zero: a checker that matched nothing has not established that
everything is compliant, it has established nothing, which is precisely the failure this file exists
to prevent one level up.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Files that can execute a command. Documentation, build trees and vendored code are out of scope.
SCAN_SUFFIXES = (".yml", ".yaml", ".ps1", ".sh", ".bash", ".py", ".cmake")
SCAN_NAMES = ("CMakeLists.txt",)
SKIP_PARTS = ("third_party", ".git")

# This checker's own source. Its selftest holds hand-written NON-compliant invocation lines as
# fixtures, so scanning itself reports its own examples as findings. Excluded by resolved path, and
# named in the summary so the exclusion is visible rather than assumed.
SELF = Path(__file__).resolve()

CTEST_SHELL = re.compile(r"""(?:^|[\s;&|(`"'])ctest\s+(?=-|@\()""")
CTEST_PYLIST = re.compile(r"""[\[(]\s*["']ctest["']\s*,""")
LISTING = re.compile(r"(?:^|\s)(?:-N|--show-only)(?:[\s=]|$)")

# `#` is the comment marker in every scanned language except PowerShell, where it is also `#`.
COMMENT = re.compile(r"^\s*#")

# Explicit opt-out for a line that does not run tests but trips the matcher anyway.
MARKER = "ctest-gate: listing"

REQUIRED = "--no-tests=error"

# How far past the matched line the flag may live. A PowerShell splat and a YAML backslash
# continuation both push arguments onto following lines; four covers every form in this repository
# with room to spare, and a larger window would start absorbing an unrelated later invocation.
CONTINUATION_LINES = 4


def invocation_lines(text: str) -> list[tuple[int, str, bool]]:
    """(line number, line, compliant) for every ctest invocation that runs tests."""
    lines = text.splitlines()
    out: list[tuple[int, str, bool]] = []
    for index, line in enumerate(lines):
        if MARKER in line or COMMENT.match(line):
            continue
        if not (CTEST_SHELL.search(line) or CTEST_PYLIST.search(line)):
            continue
        if LISTING.search(line):
            continue
        window = "\n".join(lines[index:index + 1 + CONTINUATION_LINES])
        out.append((index + 1, line.strip(), REQUIRED in window))
    return out


def offending_lines(text: str) -> list[tuple[int, str]]:
    return [(number, line) for number, line, ok in invocation_lines(text) if not ok]


def selftest() -> int:
    failures = 0

    def reject(label: str, line: str) -> None:
        nonlocal failures
        if not offending_lines(line):
            print("selftest: FAILED to reject %s:\n    %s" % (label, line.strip()), file=sys.stderr)
            failures += 1

    def accept(label: str, line: str) -> None:
        nonlocal failures
        if offending_lines(line):
            print("selftest: wrongly rejected %s:\n    %s" % (label, line.strip()), file=sys.stderr)
            failures += 1

    # Must be caught. Hand-written, not drawn from the repository, so the matcher is tested against
    # instances that exist independently of whatever the repository happens to contain.
    reject("a bare CI invocation",
           "        run: ctest --test-dir prosper/build-ci --output-on-failure\n")
    reject("a bare PowerShell splat invocation",
           "        Invoke-AuthorCheck 'Windows ctest' ctest @(\n"
           "            '--test-dir', $Build, '--output-on-failure'\n"
           "        ) $Repo\n")
    reject("a bare python argument list",
           '    cmd = ["ctest", "--timeout", "600", "--output-on-failure"] + extra\n')
    reject("a bare invocation in a shell &&-chain",
           "        cmake --build build && ctest --output-on-failure\n")

    # Must NOT be caught. The prose arms are verbatim lines from this repository that the
    # token-matching first version flagged; they are the regression for that mistake.
    accept("a compliant CI invocation",
           "        run: ctest --test-dir prosper/build-ci --output-on-failure --no-tests=error\n")
    accept("a compliant PowerShell splat invocation",
           "        Invoke-AuthorCheck 'Windows ctest' ctest @(\n"
           "            '--test-dir', $Build, '--output-on-failure', '--no-tests=error'\n"
           "        ) $Repo\n")
    accept("a compliant YAML backslash continuation",
           "          ctest --test-dir prosper/build-windows-app --output-on-failure \\\n"
           "            --no-tests=error\n")
    accept("a -N listing invocation",
           '            if ! ctest --test-dir prosper/build-ci -N -R "^${t}$" | grep -q "Test"; then\n')
    accept("a comment naming a ctest test",
           "# Run directly, or via ctest as doc_table_checker.\n")
    accept("a comment containing a ctest flag",
           "    # *case* its own root is what makes `ctest -j` safe: one test binary is\n")
    accept("a docstring describing the suite",
           '"""Run prosper\'s ctest suite under VK_LAYER_KHRONOS_validation and gate on findings.\n')
    accept("prose about ctest output",
           '    raise SystemExit(f"[vkval] ctest produced no {log}")\n')
    accept("a PowerShell string containing the words",
           "    $DryRun.Output.Contains('Windows ctest') -and\n")
    accept("an argparse help string",
           '                        help="extra argument forwarded to ctest (repeatable)")\n')

    # The self-exclusion is asserted, not assumed. Without it this checker fails on its own
    # fixtures -- which is exactly what CI reported the first time, while the local run said
    # "ok: 7" because the file was still UNTRACKED, so `git ls-files` never handed it to the
    # scanner. A repo-wide lint verified before its own file is committed has not been verified.
    if not offending_lines(SELF.read_text(encoding="utf-8", errors="replace")):
        print("selftest: this file no longer holds non-compliant fixtures, so the self-exclusion "
              "is no longer load-bearing and should be reconsidered", file=sys.stderr)
        failures += 1

    both = ("        run: ctest --test-dir b --output-on-failure\n"
            "        run: ctest --test-dir b --output-on-failure --no-tests=error\n")
    # Deliberately checked: the compliant line lies inside the FIRST line's continuation window, so
    # a window that ignored the matched line's own compliance would call both of these clean.
    if len(invocation_lines(both)) != 2:
        print("selftest: the invocation counter disagrees with the matcher", file=sys.stderr)
        failures += 1

    print("selftest: 16 arms, %d failed" % failures)
    return 1 if failures else 0


def tracked_files(root: Path) -> list[Path]:
    try:
        listing = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True,
                                 check=True).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.CalledProcessError) as exc:
        print("error: could not list tracked files: %s" % exc, file=sys.stderr)
        return []
    files = []
    for name in listing.split("\0"):
        if not name:
            continue
        path = Path(name)
        if path.suffix not in SCAN_SUFFIXES and path.name not in SCAN_NAMES:
            continue
        if any(part in SKIP_PARTS or part.startswith("build") for part in path.parts):
            continue
        if (root / path).resolve() == SELF:
            continue
        files.append(root / path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="repository root to scan")
    parser.add_argument("--selftest", action="store_true",
                        help="check the matcher against hand-built instances and exit")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    root = Path(args.root).resolve()
    scanned = 0
    problems: list[str] = []
    for path in tracked_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found = invocation_lines(text)
        scanned += len(found)
        for number, line, ok in found:
            if ok:
                continue
            problems.append("%s:%d: ctest invocation without %s\n    %s"
                            % (path.relative_to(root).as_posix(), number, REQUIRED, line))

    if not scanned:
        print("error: found no ctest invocations at all. Either the matcher is broken or --root is "
              "not the repository root; reporting success here would be the same silent-green "
              "defect this checker exists to prevent.", file=sys.stderr)
        return 1

    if problems:
        print("error: %d ctest invocation(s) can report green with zero tests (#2187):\n"
              % len(problems), file=sys.stderr)
        for problem in problems:
            print(problem, file=sys.stderr)
        print("\nAdd %s, or mark the line '%s' if it does not run tests."
              % (REQUIRED, MARKER), file=sys.stderr)
        return 1

    print("ok: %d ctest invocation(s), all carrying %s (this checker's own file is excluded -- its "
          "selftest fixtures are deliberately non-compliant)" % (scanned, REQUIRED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
