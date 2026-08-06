#!/usr/bin/env python3
"""Regression test for check_build_revision.py.

What it guards: the tool's whole value is its EXIT STATUS, because that is what makes it usable as a
gate ahead of a measurement run rather than a warning nobody reads. Every arm asserts the status --
and each cannot-answer case asserts exit 1 specifically, because a check that degrades to "pass"
when it cannot answer is worse than no check: it converts unestablished provenance into a green tick.

Two arms exist because review found the tool doing exactly that, and they are the reason to keep this
file honest:

  * INTERRUPTED BUILD. The generated revision source is written at the START of a build, so a build
    that then fails leaves it recording the new revision while every executable still embeds the old
    one. The first version of the tool read only that file and certified the stale binary as current.
  * --strict-dirty FROM ANOTHER DIRECTORY. It ran git with the process cwd and a `prosper` pathspec,
    so from inside `prosper/` the pathspec became `prosper/prosper` and matched nothing -- a silent
    no-op exactly when invoked as documented.

Arms straddle the predicate in both directions on purpose: a test that only proved MISMATCH-fails
would pass against a tool that always failed.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

TOOL = pathlib.Path(__file__).resolve().parent / "check_build_revision.py"
CMAKE_DIR = pathlib.Path(__file__).resolve().parents[2] / "cmake"
REPO = pathlib.Path(__file__).resolve().parents[3]

fails = 0
ran = 0
skipped = 0


def check(condition: bool, message: str) -> None:
    global fails, ran
    ran += 1
    if condition:
        print(f"  [ok]   {message}")
    else:
        print(f"  [FAIL] {message}")
        fails += 1


def skip(arms: int, reason: str) -> None:
    """Record arms that did NOT run, and say how many.

    Without this the summary is byte-identical whether every arm ran or only some did, and plain
    ctest prints nothing at all for a passing test -- so a block silently vanishing looks exactly
    like a clean pass. The block most likely to vanish here is the scratch-repo one, which guards
    both defects review found. A suite that can drop the arms guarding its own bugs and still print
    `all ok` is a confident zero: precisely what the tool under test exists to refuse, turned on its
    own test suite. The count makes the suite's coverage falsifiable from a log.
    """
    global skipped
    skipped += arms
    print(f"  [skip] {reason} ({arms} arm{'s' if arms != 1 else ''})")


def run(*args: str, cwd: pathlib.Path | None = None,
        repo: pathlib.Path | None = REPO) -> subprocess.CompletedProcess:
    """Invoke the tool, pinned by default to THIS checkout.

    `--repo` is passed unless a caller explicitly asks for `repo=None`. Without it these arms depend
    on where `TMPDIR` points: repo resolution follows the *target*, so a fixture created under a
    tempdir that happens to live inside another checkout resolves `--against HEAD` to that
    checkout's HEAD and the arm flakes. `CLAUDE.md` instructs every agent to set
    `TMPDIR=<worktree>/build/tmpdir`, so that is the configuration a reader is most likely to have.
    The arms that deliberately exercise repo discovery pass `repo=None`.
    """
    extra = ["--repo", str(repo)] if repo is not None else []
    return subprocess.run([sys.executable, str(TOOL), *args, *extra], cwd=cwd or REPO,
                          capture_output=True, text=True, check=False)


def git(*args: str, cwd: pathlib.Path | None = None) -> str:
    """stdout of a SUCCESSFUL git call, else "".

    Not cosmetic: in a shallow clone (which is what CI checks out) `git rev-parse HEAD~1` fails and
    still prints a plausible sha. An earlier version of this helper returned stdout unconditionally,
    so the test proceeded on a commit not present in the repository and one arm failed in CI while
    passing on every full checkout.
    """
    try:
        out = subprocess.run(["git", *args], cwd=cwd or REPO, capture_output=True, text=True,
                             check=False)
    except OSError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def write_manifest(directory: pathlib.Path, name: str, revision: str | None) -> pathlib.Path:
    record: dict[str, object] = {"type": "run", "schema": 1, "title": "TEST"}
    if revision is not None:
        record["build_revision"] = revision
    path = directory / name
    path.write_text(json.dumps(record) + "\n" + json.dumps({"type": "sample", "index": 0}) + "\n",
                    encoding="utf-8")
    return path


def make_build_dir(root: pathlib.Path, revision: str, *, binary: bool = True,
                   binary_older: bool = False) -> pathlib.Path:
    """A build directory shaped like a real one: generated revision source plus an executable."""
    build = root / "build"
    generated = build / "generated" / "prosper_build_revision"
    generated.mkdir(parents=True, exist_ok=True)
    source = generated / "build_revision.cpp"
    source.write_text(
        '#include "build_revision.hpp"\n\nnamespace prosper {\n\n'
        'const char* embedded_build_revision() noexcept {\n'
        f'    return "{revision}";\n'
        '}\n\n} // namespace prosper\n', encoding="utf-8")
    if binary:
        exe = build / "screenshot"
        exe.write_bytes(b"\x7fELF fake")
        exe.chmod(0o755)
        stamp = source.stat().st_mtime
        os.utime(exe, (stamp - 60, stamp - 60) if binary_older else (stamp + 60, stamp + 60))
    return build


def main() -> int:
    head = git("rev-parse", "HEAD")
    if not head:
        print("test_check_build_revision: not a git checkout — skipping (0 arms, all skipped)")
        return 0
    parent = git("rev-parse", "HEAD~1")   # "" in a shallow clone; arms below are guarded

    with tempfile.TemporaryDirectory() as raw:
        tmp = pathlib.Path(raw)

        # --- the failure that reached review: an interrupted build ------------------------------
        interrupted = make_build_dir(tmp / "interrupted", head, binary_older=True)
        result = run(str(interrupted), "--against", "HEAD")
        check(result.returncode == 1,
              "an executable OLDER than the generated revision source FAILS (interrupted build)")
        check("STALE BINARY" in result.stderr,
              "the interrupted-build refusal names the stale binary rather than reporting a mismatch")
        check(run(str(interrupted), "--against", "HEAD", "--allow-stale").returncode == 0,
              "--allow-stale covers the stale-binary refusal too")

        good = make_build_dir(tmp / "good", head)
        result = run(str(good), "--against", "HEAD")
        check(result.returncode == 0,
              "an executable NEWER than the generated revision source is certified")
        check("certified:" in result.stdout,
              "a pass names which executable it certified, so the claim is auditable")

        # A build dir with the right revision but nothing linked must NOT pass.
        nolink = make_build_dir(tmp / "nolink", head, binary=False)
        result = run(str(nolink), "--against", "HEAD")
        check(result.returncode == 1,
              "a build dir with no executable to certify FAILS rather than trusting the source file")
        check("UNKNOWN" in result.stderr,
              "and says it is unknown rather than reporting a match")
        check(run(str(nolink), "--against", "HEAD",
                  "--binary", str(nolink / "screenshot")).returncode == 1,
              "naming a non-existent --binary does not manufacture a certification")

        # A file that cannot be executed cannot have produced a measurement -- and must be reported
        # as present-but-not-executable rather than as absent, which was the original misdirection.
        if os.name == "posix":
            notexec = make_build_dir(tmp / "notexec", head)
            (notexec / "screenshot").chmod(0o644)
            result = run(str(notexec), "--against", "HEAD")
            check(result.returncode == 1, "a non-executable candidate is not certified")
            check("not executable" in result.stderr and "screenshot" in result.stderr,
                  "and is named as present-but-not-executable rather than reported as absent")
            (notexec / "screenshot").chmod(0o755)
            check(run(str(notexec), "--against", "HEAD").returncode == 0,
                  "restoring the executable bit certifies it: the mode is the cause")
        else:
            skip(3, "not a POSIX host — the executable-bit arms")

        if not parent:
            skip(2, "shallow clone — the older-revision arms need HEAD~1")
        else:
            old = make_build_dir(tmp / "old", parent)
            result = run(str(old), "--against", "HEAD")
            check(result.returncode == 1, "a build dir built from an older revision FAILS")
            check("BEHIND" in result.stderr, "the mismatch says which direction it is out of date")

        # --- repo discovery: the tool must follow the TARGET, not the invoking checkout ----------
        # NB1 from review: both fixed behaviours were only ever exercised through an explicit
        # --repo, and every build-dir fixture sits in a non-repo tempdir where repo_root_for()
        # falls back to cwd -- so nothing pinned the discovery itself. These two arms are a
        # discriminator: the scratch repo's HEAD differs from this checkout's, so resolving against
        # the wrong one gives the opposite verdict.
        scratch = tmp / "scratchrepo"
        (scratch / "prosper" / "tools").mkdir(parents=True)
        (scratch / "prosper" / "tools" / "f.txt").write_text("one\n", encoding="utf-8")
        for cmd in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                    ["add", "-A"], ["commit", "-qm", "seed"]):
            subprocess.run(["git", *cmd], cwd=scratch, capture_output=True, check=False)
        scratch_head = git("rev-parse", "HEAD", cwd=scratch)
        if not scratch_head:
            skip(6, "scratch git repo unavailable — discovery and --strict-dirty arms")
        elif scratch_head == head:
            skip(6, "scratch HEAD equals this checkout's — discovery cannot discriminate")
        else:
            sbuild = make_build_dir(scratch, scratch_head)
            # No --repo, invoked from THIS checkout: only target-following resolution can pass.
            check(run(str(sbuild), "--against", "HEAD", cwd=REPO, repo=None).returncode == 0,
                  "repo resolution follows the target: a build dir inside another checkout is "
                  "certified against THAT checkout's HEAD")
            # The counter-arm proves the arm above is not vacuous: forced to this checkout, the
            # identical invocation must disagree.
            check(run(str(sbuild), "--against", "HEAD", cwd=REPO, repo=REPO).returncode == 1,
                  "and --repo overrides discovery: the same build dir MISMATCHes this checkout")

            # --- --strict-dirty must work from any cwd, which is how it is documented ------------
            # In the SCRATCH repo, not this one: dirtying a tracked file in the real checkout would
            # be visible to peer lanes, would survive a crash between write and restore, and would
            # make the arms depend on the developer's own tree being clean.
            check(run(str(sbuild), "--against", "HEAD", "--strict-dirty",
                      cwd=REPO, repo=scratch).returncode == 0,
                  "--strict-dirty passes on a clean tree")
            (scratch / "prosper" / "tools" / "f.txt").write_text("two\n", encoding="utf-8")
            check(run(str(sbuild), "--against", "HEAD", "--strict-dirty",
                      cwd=REPO, repo=scratch).returncode == 1,
                  "--strict-dirty fails on a dirty tree when run from the repo root")
            check(run(str(sbuild), "--against", "HEAD", "--strict-dirty",
                      cwd=scratch / "prosper", repo=scratch).returncode == 1,
                  "--strict-dirty fails identically when run from prosper/ — the pathspec bug that "
                  "made the flag a silent no-op exactly as documented")
            check(run(str(sbuild), "--against", "HEAD",
                      cwd=scratch / "prosper", repo=scratch).returncode == 0,
                  "and without --strict-dirty the same dirty tree passes: the flag is the cause")


        # --- the manifest reader ----------------------------------------------------------------
        exact = write_manifest(tmp, "exact.jsonl", head)
        check(run("--manifest", str(exact), "--against", "HEAD").returncode == 0,
              "a manifest recording HEAD matches HEAD")
        check(run("--manifest", str(write_manifest(tmp, "upper.jsonl", head.upper())),
                  "--against", "HEAD").returncode == 0,
              "an upper-case revision matches: comparison is case-insensitive")
        check(run("--manifest", str(write_manifest(tmp, "short.jsonl", head[:12])),
                  "--against", "HEAD").returncode == 0,
              "an abbreviated revision is accepted as a prefix of the full ref")
        wrong = head[:-1] + ("0" if head[-1] != "0" else "1")
        check(run("--manifest", str(write_manifest(tmp, "wrong.jsonl", wrong)),
                  "--against", "HEAD").returncode == 1,
              "a revision differing in one character is a mismatch, not a prefix match")

        # --- cannot-answer cases must FAIL, not pass ---------------------------------------------
        check(run("--manifest", str(write_manifest(tmp, "none.jsonl", None)),
                  "--against", "HEAD").returncode == 1,
              "a manifest with no build_revision fails: unestablished is not matching")
        check(run("--manifest", str(write_manifest(tmp, "unknown.jsonl", "unknown")),
                  "--against", "HEAD").returncode == 1,
              "a build that recorded no revision ('unknown') fails rather than passing")
        check(run(str(tmp), "--against", "HEAD").returncode == 1,
              "a directory that is not a configured build dir fails rather than passing")
        check(run("--manifest", str(tmp / "no-such-file.jsonl")).returncode == 1,
              "a missing manifest file fails cleanly rather than raising")
        check("Traceback" not in run("--manifest", str(tmp / "no-such-file.jsonl")).stderr,
              "and does so without a raw traceback")
        check(run("--manifest", str(exact), "--against", "no/such/ref").returncode == 1,
              "an unresolvable --against ref fails rather than passing")

        # --- argument contract -------------------------------------------------------------------
        check(run("--against", "HEAD").returncode == 2,
              "neither BUILD_DIR nor --manifest is a usage error (argparse exit 2)")
        check(run(str(tmp), "--manifest", str(exact)).returncode == 2,
              "both BUILD_DIR and --manifest is a usage error")

        # --- the fixture must match what CMake really generates ----------------------------------
        template = CMAKE_DIR / "build_revision.cpp.in"
        if not template.is_file():
            skip(3, f"{template} absent — template-drift arms")
        else:
            rendered = template.read_text(encoding="utf-8").replace(
                "@PROSPER_GENERATED_REVISION@", head)
            pattern = re.compile(r'return\s+"([0-9a-fA-F]{7,40}|unknown)"\s*;')
            check(bool(pattern.search(rendered)),
                  "the tool's regex matches the real cmake/build_revision.cpp.in template")
            # Both sides: the arm above would also pass against a regex that matched anything, so
            # pin that the pattern is actually discriminating. If the template ever stops emitting a
            # bare quoted literal, the arm above fails loudly instead of the tool silently reading
            # every build dir as UNKNOWN.
            check(not pattern.search(rendered.replace('return "', 'return get("')),
                  "and the regex rejects a template that no longer emits a bare quoted literal")
            drifted = make_build_dir(tmp / "drift", head)
            (drifted / "generated" / "prosper_build_revision" / "build_revision.cpp").write_text(
                'const char* embedded_build_revision() noexcept { return get("%s"); }\n' % head,
                encoding="utf-8")
            check(run(str(drifted), "--against", "HEAD").returncode == 1,
                  "a drifted template makes the tool refuse, not silently certify")

    coverage = f"{ran} arms" + (f", {skipped} skipped" if skipped else "")
    print(f"test_check_build_revision: "
          + (f"{fails} FAILURE(S) ({coverage})" if fails else f"all ok ({coverage})"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
