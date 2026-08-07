#!/usr/bin/env python3
"""Is the binary you are about to quote a measurement from built from the code you think it is?

THE FAILURE THIS EXISTS FOR, because it is silent and produces a *confident* wrong answer.

A lane checks out new work (or rebases onto a master that moved), runs an existing build directory,
and reports the result as a measurement of current master. The source tree IS current. The run
succeeds. Nothing in the output, the manifest, or the log says the executable predates the change
being tested. On 2026-08-06 that nearly turned "#2121 does not move Sonic Frontiers" into a false
negative on a title that gates on the exact NID #2121 fixed -- caught only because the binary's
provenance happened to be checked by hand before the number was quoted (instrument trap 81's shape).

WHY THIS CHECKS A BINARY'S MTIME AND NOT JUST THE GENERATED REVISION

`prosper_build_revision_refresh` is an **unconditional custom target that consumers depend on**
(`cmake/ProsperBuildRevision.cmake`), so `generated/prosper_build_revision/build_revision.cpp` is
rewritten at the **start** of a build. If that build then fails or is interrupted -- the single most
common state a lane's build directory is actually in -- the generated file records the NEW revision
while every executable still embeds the OLD one.

The first version of this tool read only that file, and therefore **certified a stale binary as
current**: exactly the failure it exists to prevent, with a green tick on top. It was caught in
review by a two-sided reproduction in a scratch project. The fix is to derive the verdict from
something that only exists once a **link succeeded**: an executable whose mtime is at least that of
the generated source. `configure_file` preserves the mtime when the revision is unchanged, so a
rebuild at the same revision does not raise a false alarm.

    python3 tools/revision/check_build_revision.py build-linux
    python3 tools/revision/check_build_revision.py build-linux --binary build-linux/screenshot
    python3 tools/revision/check_build_revision.py build-linux --against HEAD
    python3 tools/revision/check_build_revision.py --manifest ~/work/manifest.json

Exit status is the contract: **0 only when a named executable is certified**, 1 for a mismatch AND
for every case where the tool cannot establish provenance -- no recorded revision, no executable to
certify, not a build directory, an unresolvable ref. An unestablished provenance is not a match, and
a check that degrades to "pass" when it cannot answer is worse than no check: it converts "unknown"
into a green tick. (Usage errors exit 2, argparse's convention.)

`--allow-stale` downgrades any of those to exit 0 while still printing loudly, for deliberate pre-fix
A/B arms and bisects -- which must then say in the write-up which revision the numbers describe.

WHAT IT STILL CANNOT SEE, stated because a check that overstates its coverage is worse than none

* **Uncommitted changes.** The embedded revision is `HEAD`, so a build from a dirty tree reports the
  base commit. `--strict-dirty` additionally fails when tracked files under `prosper/` differ from
  HEAD -- a binary built from edits that exist in no commit cannot be reproduced by anyone.
* **Executables it was not asked about.** It certifies the ones named by `--binary`, or the known
  frontends it finds. A sibling target left stale by an interrupted build is invisible unless named.
  Name the executable you are about to run.
* **A binary copied in from elsewhere**, whose mtime says nothing about this build directory.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

GENERATED = pathlib.Path("generated") / "prosper_build_revision" / "build_revision.cpp"
_REV_IN_SOURCE = re.compile(r'return\s+"([0-9a-fA-F]{7,40}|unknown)"\s*;')
# The frontends a lane actually measures with. Certifying one of these is the common case; --binary
# overrides when the target of interest is something else.
_KNOWN_STEMS = ("screenshot", "boot_trace", "gpu_replay", "prosper-app", "gpu_timeline")
# `.exe` too: a Visual Studio / MinGW `build-windows` names every one of these with the
# suffix, so a stem-only list makes every default invocation there refuse for the wrong
# reason -- "nothing to certify" when the executables are sitting right there.
KNOWN_BINARIES = _KNOWN_STEMS + tuple(f"{stem}.exe" for stem in _KNOWN_STEMS)


def _git(*args: str, cwd: pathlib.Path | None = None) -> str | None:
    """stdout of a SUCCESSFUL git invocation, else None.

    Returning stdout regardless of the return code is the bug this signature exists to prevent: in a
    shallow clone `git rev-parse HEAD~1` fails *and* prints a plausible sha, so a caller that trusts
    stdout silently proceeds on a commit that is not in the repository.
    """
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def repo_root_for(path: pathlib.Path) -> pathlib.Path | None:
    """The checkout that OWNS `path` — not the one the tool happens to be invoked from.

    Under this project's mandatory-worktree rule the invoking cwd is routinely a different checkout
    from the one whose build directory is being certified, and resolving `--against` or a dirty-tree
    query against the wrong one answers a question nobody asked.
    """
    probe = (path if path.is_dir() else path.parent).resolve()
    # --show-cdup, not --show-toplevel: it answers with a path RELATIVE to the probe ("../../", or
    # "" at the root), so git never gets to choose how the absolute path is spelled.
    #
    # --show-toplevel does choose, and under MSYS2 -- which is what the Windows CI job runs -- it
    # prints a POSIX absolute path ("/d/a/prosper/prosper"). A native-Windows Python cannot use that
    # as a subprocess cwd, so EVERY later git call in this tool failed and the tool refused
    # everything it was asked to certify. The exit codes stayed plausible -- a refusal is still a
    # refusal -- which is why it read as a build failure rather than as a broken path.
    #
    # "" is the repo ROOT and is a success, so it must be distinguished from None, which means no
    # checkout owns this path. Testing the string for truthiness conflates the two.
    cdup = _git("rev-parse", "--show-cdup", cwd=probe)
    if cdup is None:
        return None
    return (probe / cdup).resolve() if cdup else probe


def revision_from_build_dir(build_dir: pathlib.Path) -> str | None:
    source = build_dir / GENERATED
    if not source.is_file():
        return None
    match = _REV_IN_SOURCE.search(source.read_text(encoding="utf-8", errors="replace"))
    return match.group(1).lower() if match else None


def revision_from_manifest(path: pathlib.Path) -> str | None:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("type") == "run":
                value = record.get("build_revision")
                return value.lower() if isinstance(value, str) else None
    return None


def certify_binaries(build_dir: pathlib.Path, named: list[pathlib.Path]) -> tuple[
        list[pathlib.Path], list[pathlib.Path], list[pathlib.Path]]:
    """Split candidate executables into (certified, stale) by mtime against the generated source.

    An executable older than the generated revision source cannot contain that revision: the source
    is written at the start of a build, so anything not relinked afterwards predates it.
    """
    source = build_dir / GENERATED
    if not source.is_file():
        return ([], [], [])
    cutoff = source.stat().st_mtime
    candidates = named or [build_dir / name for name in KNOWN_BINARIES]
    certified, stale, unexecutable = [], [], []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        # "Certified" has to mean an executable, not any file that happens to sit at that path: a
        # stray object file or log would otherwise satisfy the check. Reported separately rather
        # than skipped, so a file present-but-not-executable is not described as absent.
        if os.name == "posix" and not os.access(candidate, os.X_OK):
            unexecutable.append(candidate)
            continue
        (certified if candidate.stat().st_mtime >= cutoff else stale).append(candidate)
    return (certified, stale, unexecutable)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refuse to quote a measurement from a build that is not the revision you think.")
    parser.add_argument("build_dir", nargs="?", type=pathlib.Path,
                        help="a CMake build directory (e.g. build-linux)")
    parser.add_argument("--manifest", type=pathlib.Path,
                        help="check a screenshot manifest's recorded revision instead")
    parser.add_argument("--binary", type=pathlib.Path, action="append", default=[],
                        help="executable to certify (repeatable); defaults to the known frontends")
    parser.add_argument("--against", default="origin/master",
                        help="git ref the build must match (default: origin/master)")
    parser.add_argument("--allow-stale", action="store_true",
                        help="report loudly but exit 0 (deliberate pre-fix A/B arms)")
    parser.add_argument("--strict-dirty", action="store_true",
                        help="also fail when tracked files under prosper/ differ from HEAD")
    parser.add_argument("--repo", type=pathlib.Path,
                        help="checkout to resolve --against in (default: the one owning the target)")
    args = parser.parse_args()

    if (args.build_dir is None) == (args.manifest is None):
        parser.error("give exactly one of BUILD_DIR or --manifest")

    def refuse(message: str) -> int:
        print(message, file=sys.stderr)
        return 0 if args.allow_stale else 1

    target = args.manifest if args.manifest is not None else args.build_dir
    if not target.exists():
        return refuse(f"revision: UNKNOWN — {target} does not exist. Nothing was verified.")

    repo = args.repo or repo_root_for(target) or repo_root_for(pathlib.Path.cwd())
    if repo is None:
        return refuse("revision: UNKNOWN — no git checkout owns the target and none was given with\n"
                      "  --repo, so no ref can be resolved. Nothing was verified.")

    stale_binaries: list[pathlib.Path] = []
    if args.manifest is not None:
        built = revision_from_manifest(args.manifest)
        origin = f"manifest {args.manifest}"
        if built is None:
            return refuse(
                f"revision: UNKNOWN — {args.manifest} has no build_revision in its run header.\n"
                "  Manifests written before that field existed cannot be checked. Treat the\n"
                "  measurement's provenance as unestablished rather than as matching.")
    else:
        built = revision_from_build_dir(args.build_dir)
        origin = f"build dir {args.build_dir}"
        if built is None:
            return refuse(
                f"revision: UNKNOWN — no {GENERATED} under {args.build_dir}.\n"
                "  Either it is not a configured prosper build directory or it has never been\n"
                "  built. Nothing is being verified; do not read this as a match.")
        certified, stale_binaries, unexecutable = certify_binaries(args.build_dir, args.binary)
        if not certified and not stale_binaries:
            if unexecutable:
                return refuse(
                    f"revision: UNKNOWN — {args.build_dir} records revision {built[:12]}, and these\n"
                    "  files exist but are not executable, so nothing can be certified:\n"
                    + "".join(f"    {b} (mode {oct(b.stat().st_mode & 0o777)})\n"
                              for b in unexecutable)
                    + "  A file that cannot be executed cannot have produced a measurement. Fix the\n"
                      "  mode, or name the real executable with --binary.")
            wanted = [str(b) for b in args.binary] or list(KNOWN_BINARIES)
            listed = "".join(f"    {name}\n" for name in wanted[:8])
            more = f"    … and {len(wanted) - 8} more\n" if len(wanted) > 8 else ""
            return refuse(
                f"revision: UNKNOWN — {args.build_dir} records revision {built[:12]} but contains\n"
                "  none of the executables it looked for:\n" + listed + more
                + "  The generated revision source is written at the START of a build, so on its own\n"
                  "  it proves only that a build was attempted at that revision — not that anything\n"
                  "  was linked. Name the executable you are about to run with --binary.")

    if built == "unknown":
        return refuse("revision: UNKNOWN — this build recorded no revision (built outside a git\n"
                      "  checkout, or git was unavailable at build time). Provenance unestablished.")

    # `rev-list -n 1 <ref>`, not `rev-parse --verify <ref>^{commit}`. Both peel a ref to the commit
    # it names and both fail on a ref that is not commit-ish, so they are interchangeable here --
    # except that one of them contains braces, and an argument with braces does not survive the trip
    # to an MSYS2 git.
    #
    # MSYS2 binaries do not receive the argv their caller built. They re-parse the raw Windows
    # command line themselves and run it through glob expansion, which has brace expansion enabled,
    # so `HEAD^{commit}` arrives as `HEAD^commit` -- a ref that does not exist. This tool then could
    # not resolve --against and refused everything it was asked to certify, which is a plausible
    # enough outcome that the exit codes read as a build problem rather than a mangled argument.
    # Native-Windows git (Git for Windows is MinGW, not MSYS) does not do this, so it reproduces
    # only on a host with both an MSYS2 git and a native Python -- i.e. the Windows CI runner.
    #
    # The general rule, since the next brace would fail exactly as quietly: no argument this tool
    # hands to git may contain a glob or brace metacharacter. `test_check_build_revision` asserts it.
    want = _git("rev-list", "-n", "1", args.against, "--", cwd=repo)
    # Empty is unresolvable too, and this is the one place the two spellings genuinely differ:
    # `rev-parse --verify X^{commit}` FAILS on an object that is not commit-ish, while `rev-list`
    # succeeds and prints nothing. Testing only for None would leave want = "", which matches no
    # build revision and so still refuses -- but as a MISMATCH, telling the reader their build is
    # out of date when what actually happened is that they named a tree.
    if not want:
        return refuse(f"revision: UNKNOWN — cannot resolve --against {args.against!r} in {repo}.")
    want = want.lower()

    if stale_binaries:
        return refuse(
            "revision: STALE BINARY — " + origin + f" records revision {built[:12]}, but these\n"
            "  executables are OLDER than the generated revision source, so they cannot contain it:\n"
            + "".join(f"    {b}\n" for b in stale_binaries)
            + "  The revision source is written at the START of a build; an executable not relinked\n"
              "  afterwards predates it, which is what an interrupted or failed build leaves behind.\n"
              "  Rebuild before quoting any measurement from them.")

    matches = built == want or (len(built) >= 7 and want.startswith(built))
    if not matches:
        behind = _git("rev-list", "--count", f"{built}..{want}", cwd=repo)
        ahead = _git("rev-list", "--count", f"{want}..{built}", cwd=repo)
        relation = ""
        if behind is not None and ahead is not None:
            if ahead == "0" and behind != "0":
                relation = f" — {behind} commit(s) BEHIND it"
            elif behind == "0" and ahead != "0":
                relation = f" — {ahead} commit(s) ahead of it"
            elif ahead != "0" and behind != "0":
                relation = f" — diverged ({ahead} ahead, {behind} behind)"
        return refuse(
            f"revision: MISMATCH — {origin} is built from {built[:12]}, "
            f"but {args.against} is {want[:12]}{relation}.\n"
            f"  A measurement from this build is NOT a measurement of {args.against}. Rebuild, or\n"
            "  pass --allow-stale and say in the write-up which revision the numbers describe.")

    if args.strict_dirty:
        changed = _git("status", "--porcelain", "--untracked-files=no", "--", "prosper", cwd=repo)
        if changed:
            print("revision: DIRTY — tracked files under prosper/ differ from HEAD, so the binary\n"
                  "  was built from edits that exist in no commit and nobody else can reproduce:",
                  file=sys.stderr)
            for line in changed.splitlines()[:20]:
                print(f"    {line}", file=sys.stderr)
            return 0 if args.allow_stale else 1

    print(f"revision: OK — {origin} is built from {built[:12]} == {args.against}")
    if args.manifest is None:
        for binary in certified:
            print(f"  certified: {binary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
