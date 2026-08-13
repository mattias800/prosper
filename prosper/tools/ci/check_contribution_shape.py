#!/usr/bin/env python3
"""Reject the two contribution shapes that cost review time without needing a reviewer.

Both rules here exist because a human (or an agent) spent an afternoon discovering something a
command could have reported in a second. Neither is a style preference; each encodes a defect that
CI could not otherwise see, and each fires on real merged history rather than on a hypothetical.

RULE 1 -- a new file outside the build tree.

  prosper's sources live under `prosper/`. The build collects exactly one thing:

      file(GLOB_RECURSE PROSPER_SRC CONFIGURE_DEPENDS src/*.cpp)   # relative to prosper/
      target_include_directories(prosper_core PUBLIC src)          # include root is prosper/src

  So a file added at the repository root -- `core/event_bus.hpp`, `plugins/foo_plugin.hpp` -- is not
  globbed, not on any include path, and not referenced by anything. It compiles nowhere. CI then goes
  green, because every job built prosper and prosper does not contain the file.

  Measured instance (#2507-#2510, 2026-08-12): four PRs, 4,541 added lines, six headers at the
  repository root. Three reported 8 SUCCESS. Not one line was compiled by any job. Green certified
  the absence of the code rather than its correctness -- the same family as instrument-trap 151, but
  further out: there a configuration was missing, here no configuration can reach the file at all.

  The allowlist is the set of top-level entries that actually exist, plus root-level Markdown and
  the usual repository metadata. A NEW top-level directory is the thing being rejected, so adding one
  deliberately means adding it here, in a commit that says why.

RULE 2 -- a new source file with no test touched anywhere.

  A new `prosper/src/**/*.cpp` is compiled automatically by the glob above, so it needs no build-system
  change and can arrive fully wired and completely unexercised. That is the #2492 shape: a subsystem
  with no call site and no test, where "it compiles" is the only claim anyone can check.

  Deliberately narrow, so the gate stays trusted:
    * ADDED .cpp only. Modifying an existing source file does not trip it -- a comment fix, a
      diagnostic, a one-line correction must not demand a test it cannot meaningfully have.
    * ANY file under prosper/tests/ satisfies it, added or modified. The rule asks whether the author
      considered the question, not whether a particular test exists; a reviewer judges the rest.

  A refactor that genuinely splits a file without touching tests will trip this. That is the intended
  cost: it is rare, and the fix is one line in the PR body explaining it, which is cheaper than the
  case the rule catches.

EXIT STATUS IS THE VERDICT. This exits non-zero on any violation, with or without --github.
Recorded explicitly because the author of this file publicly claimed, three times, that a sibling
gate reported problems and still exited 0 -- and then measured it and found it exits 1. The real
failure there was `> /dev/null` plus an unchecked `$?` (trap 40). A checker whose status can be
ignored is a checker that will be.

  --selftest runs both rules against hand-written violating AND compliant inputs. A gate that cannot
  demonstrate it fires is void; a gate that cannot demonstrate it stays quiet on good input gets
  disabled by the first person it annoys.
"""

import argparse
import subprocess
import sys

# Top-level entries that exist on master, plus metadata a contributor may legitimately add.
ALLOWED_TOP_LEVEL = {
    ".github",
    ".gitignore",
    ".gitattributes",
    "assets",
    "docs",
    "prosper",
    "scripts",
}

# Root-level files that are fine regardless of the set above (documentation and licensing).
ALLOWED_ROOT_SUFFIXES = (".md",)
ALLOWED_ROOT_NAMES = {"LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE"}


def top_level(path):
    return path.split("/", 1)[0]


def is_allowed_addition(path):
    """Whether a newly ADDED file may live at `path`."""
    head = top_level(path)
    if head in ALLOWED_TOP_LEVEL:
        return True
    if "/" not in path:  # a root-level file
        if path in ALLOWED_ROOT_NAMES:
            return True
        return path.endswith(ALLOWED_ROOT_SUFFIXES)
    return False


def check(changes):
    """`changes` is a list of (status, path). Returns a list of problem strings."""
    problems = []

    added = [p for s, p in changes if s == "A"]
    touched = [p for _, p in changes]

    # Rule 1 -- outside the build tree.
    for path in added:
        if not is_allowed_addition(path):
            problems.append(
                "%s: new file outside the build tree. prosper's sources live under `prosper/`; "
                "the build globs `prosper/src/*.cpp` and includes `prosper/src`, so nothing here "
                "is compiled or included by any job -- CI would go green without building it. "
                "Move it under `prosper/` (diagnostics belong in `prosper/src/diagnostics/`), or "
                "add its top-level directory to ALLOWED_TOP_LEVEL in %s with a reason."
                % (path, __file__.replace("\\", "/").split("/")[-1])
            )

    # Rule 2 -- a new source file with no test touched.
    new_sources = [
        p for p in added
        if p.startswith("prosper/src/") and p.endswith(".cpp")
    ]
    tests_touched = any(p.startswith("prosper/tests/") for p in touched)
    if new_sources and not tests_touched:
        problems.append(
            "adds %d new source file(s) under prosper/src/ but touches nothing under "
            "prosper/tests/: %s. A new .cpp is compiled automatically by the CONFIGURE_DEPENDS "
            "glob, so it can arrive fully built and completely unexercised -- 'it compiles' is "
            "then the only claim a reviewer can check. Add a test that fails without the change. "
            "If this is a pure refactor with no behaviour to assert, say so in the PR body."
            % (len(new_sources), ", ".join(sorted(new_sources)[:5]))
        )

    return problems


def parse_name_status(text):
    """Parse `git diff --name-status` output into (status, path) pairs."""
    changes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][:1]          # R100 -> R, etc.
        path = parts[-1]               # for renames, the destination
        changes.append((status, path))
    return changes


def selftest():
    """Both rules, against inputs that must fail and inputs that must not."""
    must_fail = [
        ("rule 1: header at the repo root (#2507)", [("A", "core/event_bus.hpp")]),
        ("rule 1: plugin dir at the repo root (#2508)", [("A", "plugins/boot_state_machine_plugin.hpp")]),
        ("rule 1: a new top-level directory", [("A", "framework/thing.hpp")]),
        ("rule 2: new source, no test touched", [("A", "prosper/src/diagnostics/foo.cpp")]),
        ("both rules at once", [("A", "core/x.hpp"), ("A", "prosper/src/y.cpp")]),
    ]
    must_pass = [
        ("source plus a test", [("A", "prosper/src/diagnostics/foo.cpp"),
                                ("A", "prosper/tests/test_foo.cpp")]),
        ("source plus a MODIFIED test", [("A", "prosper/src/foo.cpp"),
                                         ("M", "prosper/tests/test_gpu_capture.cpp")]),
        ("modifying an existing source, no test", [("M", "prosper/src/gpu/gpu_capture.cpp")]),
        ("docs only", [("M", "prosper/docs/GAME_COMPAT_ORCHESTRATION.md")]),
        ("root-level markdown", [("A", "CONTRIBUTING.md")]),
        ("workflow change", [("M", ".github/workflows/ci.yml")]),
        ("a new header under prosper/src (no .cpp)", [("A", "prosper/src/diagnostics/x.hpp")]),
        ("scripts and assets", [("A", "scripts/x.sh"), ("A", "assets/x.png")]),
        ("a deleted root file", [("D", "core/old.hpp")]),
    ]

    failures = 0
    for label, changes in must_fail:
        if not check(changes):
            print("selftest: FAILED to reject -- %s" % label, file=sys.stderr)
            failures += 1
    for label, changes in must_pass:
        problems = check(changes)
        if problems:
            print("selftest: wrongly rejected -- %s\n    %s" % (label, problems[0]), file=sys.stderr)
            failures += 1

    total = len(must_fail) + len(must_pass)
    print("selftest: %d arms (%d must-fail, %d must-pass), %d failed"
          % (total, len(must_fail), len(must_pass), failures))
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", help="git ref to diff against (e.g. origin/master)")
    ap.add_argument("--name-status-file", help="a file holding `git diff --name-status` output")
    ap.add_argument("--github", action="store_true",
                    help="emit ::error:: annotations so failures surface on the Actions summary")
    ap.add_argument("--selftest", action="store_true",
                    help="run both rules against hand-written violating and compliant inputs")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.name_status_file:
        with open(args.name_status_file, encoding="utf-8") as fh:
            text = fh.read()
    elif args.base:
        text = subprocess.run(
            ["git", "diff", "--name-status", "%s...HEAD" % args.base],
            capture_output=True, text=True, check=True).stdout
    else:
        ap.error("one of --base, --name-status-file or --selftest is required")

    changes = parse_name_status(text)
    problems = check(changes)

    if not problems:
        print("contribution shape: %d changed path(s), both rules satisfied" % len(changes))
        return 0

    for p in problems:
        if args.github:
            print("::error::%s" % p.replace("\n", " "))
        print("error: %s" % p, file=sys.stderr)
    print("\n%d problem(s) found." % len(problems), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
