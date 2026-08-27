#!/usr/bin/env python3
"""Self-checking tests for trap_number.py (exit code is truth).

Nothing here needs the network or authentication -- a test that needs either is a test that gets
disabled. `gh` is a stub on PATH, and the arms that exercise `git fetch` push to an `origin` that is
a directory in the same temporary tree. What IS exercised is every half that can be wrong silently:
which rows it reads out of a document, which base it reads them from, and whether its answer agrees
with the gate that will judge the row it helps you write.

That agreement is the point of the last section, and it is not decoration. The allocator and
`check_numbered_table.py` parse the same table with DIFFERENT code -- the checker builds table runs
and requires every body row to be numbered, the allocator scans linearly from a matching header. If
they ever disagree, the allocator hands out a number the gate rejects, and the failure appears on
someone else's PR hours later. Pinning them against the repository's real Instrument table is the
only arm that can see that. The arm asserts a floor rather than an exact count deliberately: an
exact one would need editing on every append, which is how prose figures go stale.

HOW TO TELL A REAL ARM FROM ONE THAT REDDENS NOTHING, without running a mutation:

    Ask "WHAT ELSE IN THIS OUTPUT COULD SATISFY THIS ASSERTION?"
    An arm discriminates only if it asserts on a string ONLY THE BRANCH UNDER TEST can produce.

Every void arm this suite has shipped failed exactly that question, and each was caught late and
expensively by a mutation run instead:

  * `expect="#11"` for the collision line -- the per-PR TABLE ROW prints `PR #11` too, so emptying
    the racing list satisfied it just as well.
  * `want_rc=1` for "an unauthenticated gh must not fall back" -- a silent fallback makes the NEXT
    `gh` call fail, so rc=1 either way. Assert WHICH call failed.
  * `expect="4"` for `--quiet` -- a bare substring any report containing a 4 satisfies. Assert the
    whole output.
  * (in the sibling suite) a `// s_trap 1` fixture asserting "not a citation", where 1 is a valid
    row, so it passed whether the mnemonic was excluded or matched-and-resolved.

The question is cheaper than a mutation run and catches the same class while you are writing, which
is where it is worth having. It is not a replacement for mutating -- an arm can name a unique string
and still test the wrong branch -- but nothing that fails this question is worth mutating.

AND THE OTHER HALF, WHICH THAT QUESTION CANNOT REACH (#2624). Hardening every arm you wrote says
nothing about a promise you never wrote one for, and the two failures are orthogonal: #2610 repaired
the selectivity of all twelve arms above and STILL left three advertised promises with no arm --
the fail-closed `gh pr diff`, `git fetch`, and the `no claim` state. So before adding arms, ask

    "WHAT SENTENCE DOES THIS TOOL'S DOCSTRING LEAD WITH, AND WHICH ARM GOES RED IF IT BECOMES FALSE?"

enumerated from the tool's own --help, docstrings and README claims, and only then compared against
the arms that exist. Starting from the existing arms cannot find this class, because a missing arm
leaves nothing to start from. It is the advertised faculty -- the thing everyone reasons about --
that nobody points an arm at.

Run directly, or via ctest as trap_number.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from check_numbered_table import Table, parse_tables  # noqa: E402
from trap_number import added_rows_from_patch, highest, run, table_numbers  # noqa: E402

FAILURES: list[str] = []


def case(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: expected {want!r}, got {got!r}")
        return
    print(f"  ok  {name}")


HEADER = "| # | Instrument | How it lied |\n|---|---|---|\n"

print("reading the right rows:")

case("a plain numbered table",
     table_numbers(HEADER + "| 1 | a | b |\n| 2 | c | d |\n", "Instrument"), [1, 2])

case("the highest is the allocation point, not the row count",
     highest(HEADER + "| 1 | a | b |\n| 5 | c | d |\n", "Instrument"), 5)

# Gaps are legal since #2089, so the tool must allocate from the MAXIMUM rather than from a count.
# An implementation returning len(rows) would pass the first case above and hand out 3 here, which
# is already taken.
case("a gapped table allocates above the maximum, not above the count",
     highest(HEADER + "| 1 | a | b |\n| 2 | c | d |\n| 9 | e | f |\n", "Instrument"), 9)

# The document holds several numbered tables; taking the wrong one hands out a number in the wrong
# space entirely. This is the case --table-header exists for.
TWO_TABLES = (
    "| # | Other | thing |\n|---|---|---|\n| 400 | x | y |\n\nprose\n\n"
    + HEADER + "| 1 | a | b |\n| 2 | c | d |\n"
)
case("the header selects the table", highest(TWO_TABLES, "Instrument"), 2)
case("...and the other table is reachable by its own header", highest(TWO_TABLES, "Other"), 400)

# This repository's docs paste tool output full of pipes, and the orchestration document quotes
# example table rows inside fences. Counting those would inflate the answer without a trace.
case("rows inside a fence are not rows",
     highest(HEADER + "| 1 | a | b |\n\n```\n| 999 | pasted | output |\n```\n", "Instrument"), 1)

# A blank line ends a table in Markdown, and rows after it belong to a different table -- the exact
# defect check_numbered_table's structure class exists for. The allocator must not read across it.
case("a blank line ends the table",
     highest(HEADER + "| 1 | a | b |\n\n| 700 | orphan | row |\n", "Instrument"), 1)

case("no matching table yields None", highest(HEADER + "| 1 | a | b |\n", "Nonexistent"), None)
case("an empty document yields None", highest("", "Instrument"), None)

print("reading a PR's claim out of its patch:")

D = "diff --git a/doc.md b/doc.md\n--- a/doc.md\n+++ b/doc.md\n"

case("a patch that appends a row claims it",
     added_rows_from_patch(D + "@@\n | 5 | old |\n+| 6 | new |\n", "doc.md"), [6])

# THE discriminating case. A row EDITED in place is a -/+ pair on one number, and trap rows are
# amended routinely as evidence accrues. Reading only the `+` side reports it as a fresh claim on a
# number the base already holds -- i.e. as a collision that does not exist, which would send a lane
# to renumber for nothing.
case("an amended row is not a new claim",
     added_rows_from_patch(D + "@@\n-| 6 | old text |\n+| 6 | extended text |\n", "doc.md"), [])

case("an amendment and an append together yield only the append",
     added_rows_from_patch(D + "@@\n-| 6 | old |\n+| 6 | new |\n+| 7 | appended |\n", "doc.md"), [7])

# Rows added to a DIFFERENT file in the same patch are not claims on this table.
case("another file's rows are not counted",
     added_rows_from_patch(D + "@@\n+| 6 | mine |\n"
                           "diff --git a/other.md b/other.md\n--- a/other.md\n+++ b/other.md\n"
                           "@@\n+| 99 | theirs |\n", "doc.md"), [6])

case("a patch touching nothing relevant claims nothing",
     added_rows_from_patch(D + "@@\n+some prose\n", "doc.md"), [])

print("main(), executed end to end against a stubbed gh:")

# EVERY defect this tool has had lived in a path no arm reached, because importing three pure
# functions is not coverage of a CLI. A 5-vs-6 tuple unpack in the collision branch survived two
# review rounds and then crashed live -- after the table had printed, so the run looked half
# successful, and with --quiet returning cleanly beforehand so a scripted caller saw nothing wrong.
# It was in the ONE path this tool exists for.
#
# So these run the real main() as a subprocess, with `gh` stubbed on PATH and a real git repository
# underneath. The collision arm is the discriminating one: red on the unpack bug, green after it.

GH_STUB = '''#!/usr/bin/env python3
import json, sys
a = sys.argv[1:]
DOC = "prosper/docs/GAME_COMPAT_ORCHESTRATION.md"
HDR = "| # | Instrument | How it lied |\\n|---|---|---|\\n"
def table(rows):
    return HDR + "".join("| %d | r%d | e |\\n" % (n, n) for n in rows) + "\\n### Next\\n"
def pr(n, title, oid, paths, draft=False):
    return {"number": n, "title": title, "headRefOid": oid, "isDraft": draft,
            "files": [{"path": p} for p in paths]}
def patch(path, lines):
    head = "diff --git a/%s b/%s\\n--- a/%s\\n+++ b/%s\\n@@\\n" % (path, path, path, path)
    return head + "".join(l + "\\n" for l in lines)
import os
if os.environ.get("STUB_MODE") == "unauth":
    sys.stderr.write("gh: not authenticated" + chr(10))
    sys.exit(4)
if a[:2] == ["repo", "view"]:
    print("o/r")
elif a[:2] == ["pr", "list"]:
    import os
    if os.environ.get("STUB_MODE") == "wild":
        print(json.dumps([
            {"number": 33, "title": "typo lane", "headRefOid": "ccc", "isDraft": False,
             "files": [{"path": DOC}]},
        ]))
        sys.exit(0)
    if os.environ.get("STUB_MODE") == "dupmaster":
        print(json.dumps([
            {"number": 44, "title": "dup lane", "headRefOid": "ddd", "isDraft": False,
             "files": [{"path": DOC}]},
        ]))
        sys.exit(0)
    if os.environ.get("STUB_MODE") == "capped":
        print(json.dumps([
            {"number": 11, "title": "lane A", "headRefOid": "aaa", "isDraft": False,
             "files": [{"path": "f%d.txt" % i} for i in range(100)]},
        ]))
        sys.exit(0)
    m = os.environ.get("STUB_MODE")
    if m == "diffail":
        print(json.dumps([pr(66, "unreadable patch lane", "eee", [DOC])]))
        sys.exit(0)
    if m == "noclaim":
        print(json.dumps([pr(77, "older base lane", "fff", [DOC])]))
        sys.exit(0)
    if m == "notable":
        print(json.dumps([pr(88, "rewrote the table lane", "ggg", [DOC])]))
        sys.exit(0)
    if m == "draft":
        print(json.dumps([pr(99, "draft lane", "hhh", [DOC], True)]))
        sys.exit(0)
    if m == "untouched":
        # One PR touches the document, one does not and is NOT at the files cap -- so the absent
        # path is real information rather than an unknown, and the tool must not fetch its copy.
        print(json.dumps([pr(11, "lane A", "aaa", [DOC]),
                          pr(55, "unrelated lane", "iii", ["prosper/src/foo.cpp"])]))
        sys.exit(0)
    print(json.dumps([
        {"number": 11, "title": "lane A", "headRefOid": "aaa", "isDraft": False,
         "files": [{"path": DOC}]},
        {"number": 22, "title": "lane B", "headRefOid": "bbb", "isDraft": False,
         "files": [{"path": DOC}]},
    ]))
elif a[0] == "api":
    m = os.environ.get("STUB_MODE")
    if m == "wild":
        print(table([1, 2, 900]))
    elif m == "dupmaster":
        print(table([1, 2]))
    elif m == "noclaim":
        # Exactly the base's rows: an older base that adds nothing.
        print(table([1, 2]))
    elif m == "notable":
        print("Just prose in this copy. No numbered table at all." + chr(10))
    else:
        print(table([1, 2, 3]))
elif a[:2] == ["pr", "diff"]:
    import os
    m = os.environ.get("STUB_MODE")
    if m == "diffail":
        sys.stderr.write("gh: could not read that pull request" + chr(10))
        sys.exit(3)
    if m in ("noclaim", "notable"):
        print(patch("other.txt", ["+unrelated"]))
        sys.exit(0)
    if m == "wild":
        print("diff --git a/%s b/%s" % (DOC, DOC))
        print("--- a/%s" % DOC); print("+++ b/%s" % DOC)
        print("@@"); print("+| 900 | r900 | e |")
        sys.exit(0)
    if m == "dupmaster":
        print("diff --git a/%s b/%s" % (DOC, DOC))
        print("--- a/%s" % DOC); print("+++ b/%s" % DOC)
        print("@@"); print("+| 2 | a DIFFERENT row 2 | e |")
        sys.exit(0)
    if m == "stacked" and a[2] == "22":
        print("diff --git a/other.txt b/other.txt\\n--- a/other.txt\\n+++ b/other.txt\\n@@\\n+unrelated\\n")
    else:
        print("diff --git a/%s b/%s\\n--- a/%s\\n+++ b/%s\\n@@\\n+| 3 | r3 | e |\\n" % (DOC, DOC, DOC, DOC))
else:
    sys.exit(1)
'''

DOC_REL = Path("prosper/docs/GAME_COMPAT_ORCHESTRATION.md")


def doc_text(rows: list[int]) -> str:
    """The seeded document, so an arm can say which rows a commit holds rather than paste a table."""
    return ("| # | Instrument | How it lied |\n|---|---|---|\n"
            + "".join(f"| {n} | r{n} | e |\n" for n in rows) + "\n### Next\n")


SEED = doc_text([1, 2])


def git(repo: Path, *args: str) -> str:
    """git in `repo`, FAILING LOUDLY -- the setup loop below can ignore rc, this cannot.

    The origin arms build the state their assertion reads: a push that silently failed would leave
    an upstream holding nothing, `git fetch` would still succeed against it, and the arm would then
    be measuring the seed it started from while looking like it had measured a fetch.
    """
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed (rc={proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout.strip()


def run_main(name: str, *, want_rc: int, expect: str | None = None, absent: str | None = None,
             exact: str | None = None, extra: list[str] | None = None,
             stub_env: dict[str, str] | None = None, no_gh: bool = False,
             origin_rows: list[int] | None = None, origin_missing: bool = False) -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        # The stub logic lives in a .py; what goes on PATH is a per-platform LAUNCHER.
        #
        # Windows resolves an executable through PATHEXT (.COM;.EXE;.BAT;.CMD;...), so a stub file
        # named `gh` with no extension matches nothing and `subprocess` reports the tool's first
        # `gh` call as a failure -- which is how eleven of these twelve arms went red on Windows
        # MinGW while passing on Linux and macOS. The arms written to close the "nothing executes
        # main()" gap were themselves POSIX-only (#2610 CI). Shipping a `.cmd` on Windows keeps
        # every arm running on all three platforms, which is the point: a skip here would leave the
        # tool's primary path untested on exactly the platform where PATHEXT and the path separator
        # differ.
        stub_py = bin_dir / "gh_stub.py"
        stub_py.write_text(GH_STUB, encoding="utf-8")
        if os.name == "nt":
            gh = bin_dir / "gh.cmd"
            gh.write_text(f'@"{sys.executable}" "{stub_py}" %*\n', encoding="utf-8")
        else:
            gh = bin_dir / "gh"
            gh.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stub_py}" "$@"\n', encoding="utf-8")
            gh.chmod(gh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        repo = root / "repo"
        (repo / DOC_REL.parent).mkdir(parents=True)
        (repo / DOC_REL).write_text(SEED, encoding="utf-8")
        for cmd in (["init", "-q", "-b", "master"],
                    ["config", "user.email", "t@example.invalid"],
                    # A developer with commit.gpgsign on globally would otherwise get a fixture
                    # whose commits fail, and every arm here reads a commit.
                    ["config", "commit.gpgsign", "false"],
                    ["config", "user.name", "t"], ["add", "-A"], ["commit", "-qm", "base"]):
            subprocess.run(["git", "-C", str(repo), *cmd], capture_output=True)

        # `--no-fetch --base master` unless an arm asks for a real remote, in which case it runs the
        # DEFAULT path: fetch, then read origin/master. Every other arm skips the fetch, which is
        # how the fetch went unexercised (#2624).
        base_args = ["--no-fetch", "--base", "master"]
        if origin_missing:
            base_args = ["--base", "origin/master"]
            git(repo, "remote", "add", "origin", str(root / "no-such-upstream"))
        elif origin_rows is not None:
            base_args = ["--base", "origin/master"]
            upstream = root / "upstream"
            init = subprocess.run(["git", "init", "--bare", "-q", str(upstream)],
                                  capture_output=True, text=True)
            assert init.returncode == 0, f"git init --bare failed: {init.stderr}"
            git(repo, "remote", "add", "origin", str(upstream))
            git(repo, "push", "-q", "origin", "master")
            seeded = git(repo, "rev-parse", "HEAD")
            # Another lane's push, landing on the shared origin after this checkout was made.
            (repo / DOC_REL).write_text(doc_text(origin_rows), encoding="utf-8")
            git(repo, "commit", "-qam", "another lane's row")
            git(repo, "push", "-q", "origin", "master")
            # Rewind the remote-tracking ref by hand rather than trusting push not to have moved it:
            # this is exactly the state of a checkout that has not fetched since that push, and it is
            # what makes the arm discriminate. Without a fetch the tool reads `seeded` and answers
            # from rows that are two commits stale.
            git(repo, "update-ref", "refs/remotes/origin/master", seeded)

        if no_gh:
            # PATH with NO gh at all -- not even the stub. `git` still has to resolve, so keep the
            # real PATH and simply do not prepend the stub directory... except the point is that gh
            # must be absent, and the runner has a real one. So: prepend a directory holding a git
            # launcher only, and drop the rest of PATH.
            git_only = root / "gitonly"
            git_only.mkdir()
            real_git = shutil.which("git")
            assert real_git, "git must be on PATH for these tests"
            if os.name == "nt":
                (git_only / "git.cmd").write_text(f'@"{real_git}" %*\n', encoding="utf-8")
            else:
                g = git_only / "git"
                g.write_text(f'#!/bin/sh\nexec "{real_git}" "$@"\n', encoding="utf-8")
                g.chmod(g.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            env = dict(os.environ, PATH=str(git_only))
        else:
            env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
        env.update(stub_env or {})
        proc = subprocess.run(
            [sys.executable, str(HERE / "trap_number.py"), *base_args, *(extra or [])],
            cwd=repo, capture_output=True, text=True, env=env,
        )
    out = proc.stdout + proc.stderr
    if proc.returncode != want_rc:
        FAILURES.append(f"{name}: expected rc={want_rc}, got {proc.returncode}: {out[:400]!r}")
    elif expect and expect not in out:
        FAILURES.append(f"{name}: expected {expect!r} in output, got {out[:400]!r}")
    elif absent and absent in out:
        FAILURES.append(f"{name}: did not expect {absent!r}, got {out[:400]!r}")
    elif exact is not None and out.strip() != exact:
        # `exact` exists because a bare substring is the weak form the whole batch kept tripping on:
        # "4" is satisfied by any report that happens to contain a 4, including the full table
        # --quiet is supposed to suppress. Whole-output equality is the only assertion here that
        # nothing but the intended branch can produce.
        FAILURES.append(f"{name}: expected the whole output to be {exact!r}, got {out.strip()[:200]!r}")
    else:
        print(f"  ok  {name}")


# THE arm. Two PRs whose patches both add row 3, so the collision branch actually runs. Under the
# 5-vs-6 unpack this exits 1 with a ValueError, and asserting on rc is what catches it.
run_main("a real collision reaches the advice instead of crashing",
         want_rc=0, expect="COLLISION: 3 claimed by more than one open PR")

# ASSERT ON THE COLLISION LINE, NOT ON "#11". The first draft did the latter and reddened nothing,
# because the per-PR table row above prints "PR #11" too -- so `racing = []` satisfied it just as
# well. That is the general rule the whole batch converged on: an arm discriminates only if it
# asserts on a string ONLY THE BRANCH UNDER TEST can produce. Asking "what else in this output could
# satisfy this assertion?" catches it while writing, with no mutation run (#2610 review).
run_main("...and the collision line names both racing PRs",
         want_rc=0, expect="open PR (#11, #22)")

run_main("...and suggests stepping clear rather than to the next free number",
         want_rc=0, expect="CONSIDER")

# --quiet returns before the report, which is exactly why the crash was invisible to a scripted
# caller. It must still be right, and it must not be the only path that works.
run_main("--quiet prints just the number and nothing else", want_rc=0, exact="4",
         extra=["--quiet"])

# THE arm that pins WHICH notion of "claim" the collision arithmetic uses. PR 22 here is STACKED:
# its file holds row 3, inherited from the branch underneath, while its patch adds nothing to this
# document. Computing `contested` from file-content-minus-master credits it with a claim it does not
# have and invents a collision; computing it from the patch does not. Reverting that one index makes
# this case, and only this case, fail -- which the collision arms above cannot see, because in their
# fixture the two notions agree.
run_main("a STACKED PR inherits a row without claiming it, so there is no collision",
         want_rc=0, expect="inherits 3 from its base branch",
         absent="COLLISION", stub_env={"STUB_MODE": "stacked"})

# "Fails closed if `gh` is not authenticated -- which is why that case is a hard error and not a
# fallback to master alone." That is the docstring's own sentence, and nothing executed it. A silent
# fallback would answer "master says N, take N+1" while LOOKING like it had worked, which is the
# defect the tool exists to prevent.
# NOTE the assertion is on WHICH call failed, not on rc. Asserting rc=1 alone is VOID here, and I
# shipped that first: with a silent fallback in place the very next `gh` call fails too, so the run
# exits 1 either way and the arm cannot tell the implementations apart. Naming `gh repo view` is what
# discriminates -- under a fallback the error would name `gh pr list` instead. Same void-arm shape as
# the `s_trap 1` and `WRITE-TRAP #1` fixtures, found here in the arms added to fix "no arm reaches
# this path", which is that problem one level up.
run_main("an unauthenticated gh is an error at the FIRST call, not a fallback to master alone",
         want_rc=1, expect="gh repo view", stub_env={"STUB_MODE": "unauth"})

# "A PR at the cap is SCANNED rather than trusted." The stub returns one PR with a 100-entry files
# array that does not list our path; skipping it would make its claim invisible, so the tool must
# fetch its copy anyway. It does, and the run completes -- under a `skip` the PR would simply vanish
# from the report.
run_main("a PR whose files array is at the 100 cap is scanned, not skipped",
         want_rc=0, expect="1 touching this file", stub_env={"STUB_MODE": "capped"})

# THE HEADLINE CAPABILITY, and it had no arm: a PR whose patch adds a number the base ALREADY holds.
# This is the report that found the live #2602 collision, and disabling the branch changed nothing in
# the suite. Master here holds rows 1 and 2; PR #44's patch adds a DIFFERENT row 2.
run_main("a PR adding a number the base already holds is reported as a duplicate",
         want_rc=0, expect="DUPLICATES 2 ALREADY ON master",
         stub_env={"STUB_MODE": "dupmaster"})

# The wild-claim report, added two rounds ago as the allocator/gate divergence guard, also had no
# arm. PR #33 claims 900 against a base whose max is 2 -- a mistyped digit, which
# check_numbered_table --baseline rejects, so the allocator must not hand out 901.
run_main("a claim far above the base is reported as a typo, not an allocation",
         want_rc=0, expect="IGNORING PR #33", stub_env={"STUB_MODE": "wild"})

# ...and the half that matters more: it must be EXCLUDED from the arithmetic. Without the MAX_JUMP
# filter on `sane` the answer becomes 901 and every later allocation inherits it. Asserting the
# number itself is what discriminates -- the IGNORING line above prints either way.
run_main("...and is excluded from the next free number, not merely mentioned",
         want_rc=0, expect="next free number: 3", stub_env={"STUB_MODE": "wild"})

# `gh` ABSENT ENTIRELY -- not failing, not unauthenticated, simply not installed. The tool promises
# to refuse rather than guess, and it did not: the missing executable escaped as an uncaught
# FileNotFoundError and printed a traceback on EVERY platform. A traceback is not a refusal to
# answer; it is a crash that also happens not to answer, and those read very differently to whoever
# is holding a number they are about to write.
#
# This is also the arm that closes the Windows failure. Python resolves a bare command name through
# CreateProcess, which appends only .EXE and does not walk PATHEXT, so the `gh.cmd` launcher these
# tests put on PATH was invisible to the tool. Resolving with shutil.which fixes both, and asserting
# "no traceback" is what pins it -- rc=1 alone is satisfied by the crash too.
run_main("an absent gh is a refusal with a message, not a traceback",
         want_rc=1, expect="is not on PATH", absent="Traceback", no_gh=True)

# The truncation guard, executed rather than reasoned about: the stub returns 2 PRs, so --limit 2
# cannot distinguish "there are 2" from "we stopped at 2".
run_main("a truncated PR list is an error, not a smaller answer",
         want_rc=1, expect="TRUNCATED", extra=["--limit", "2"])

# ---------------------------------------------------------------------------------------------
# Promises that had NO arm at all (#2624), found by listing what the tool's docstring and --help
# advertise and then asking which arm goes red if each sentence becomes false. That is a different
# question from "does the arm I wrote discriminate?", and #2610 proved the two are orthogonal:
# every existing arm's selectivity was repaired and these were still unguarded afterwards.
# ---------------------------------------------------------------------------------------------

# "RAISES rather than returning None on a failed `gh pr diff`, and that is the whole point."
# Returning None made the caller fall through to the set-difference branch -- the logic that
# function REPLACED -- so an unreadable patch silently restored the defect while the run looked
# successful. Nothing ran that path: the stub's `pr diff` always succeeded.
# The assertion names the failing call, and that is load-bearing. rc=1 alone is VOID here for the
# same reason it was void for the unauthenticated arm: a fallback would carry on and die at a later
# `gh` call, so the run exits 1 under either implementation. A traceback also exits 1 -- hence
# `absent="Traceback"`, since the tool promises a refusal with a message, not a crash.
run_main("an unreadable `gh pr diff` is a refusal that names that call, not a silent fallback",
         want_rc=1, expect="gh pr diff 66", absent="Traceback", stub_env={"STUB_MODE": "diffail"})

# "Without this the base is whatever the last fetch left behind, which is the stale-read half of the
# very defect being prevented." Every other arm passes --no-fetch, so the DEFAULT path -- the one a
# human actually runs -- was never executed. A wrong remote, a wrong ref, or a fetch failure treated
# as benign would be invisible to the suite and visible only as a stale answer.
#
# So: a real `origin` on the filesystem (no network, no authentication). Origin holds rows 1, 2 and
# 5; the checkout's remote-tracking ref is rewound to the commit holding 1 and 2. Reading the FETCHED
# base gives 5 and answers 6; reading the stale ref gives 2, and the two stubbed PRs' row 3 makes it
# answer 4. --quiet asserted whole-output, so nothing but the fetched read can satisfy it.
run_main("the default path fetches, so the base is the pushed one and not a stale local ref",
         want_rc=0, exact="6", extra=["--quiet"], origin_rows=[1, 2, 5])

# ...and the failure direction of the same promise: a fetch that CANNOT run must stop the tool. If
# it were swallowed, the answer would come from whatever the last fetch left behind and would look
# exactly like a successful run. Asserting the message names `git fetch` is what discriminates: the
# subsequent `git show origin/master:...` would fail too, so rc=1 alone proves nothing.
run_main("a fetch that fails is a hard error naming that call, not a fall back to the stale ref",
         want_rc=1, expect="git fetch", absent="next free number", origin_missing=True)

# "Every open PR is listed, including those whose highest row is at or below master's -- because
# that is NOT a claim, it is a branch on an older base that adds no row." Of the report's states,
# CLAIMS, DUPLICATES and `inherits` had arms and this one did not, so a change collapsing it into
# either would have passed. It is the branch that tells an author "this PR is not in your race".
# PR #77's copy holds exactly the base's rows and its patch touches another file entirely.
run_main("a PR on an older base that adds no row is reported as no claim, not as a claim",
         want_rc=0, expect="no claim (older base, adds no row)", absent="CLAIMS",
         stub_env={"STUB_MODE": "noclaim"})

# The fourth report state, likewise unguarded: a PR whose copy of the file has no numbered table at
# all (renamed header, table moved, file rewritten). It must be listed and excluded from the
# arithmetic rather than crash `max()` or be silently counted as a claim on nothing.
run_main("a PR whose copy has no numbered table is listed as that, not as a claim",
         want_rc=0, expect="no numbered table in its copy", absent="CLAIMS",
         stub_env={"STUB_MODE": "notable"})

# "The scan asks GitHub which files each PR changes and only fetches the ones that do." The cap arm
# above pins the other side -- a PR AT the 100-file cap is scanned because its file list proves
# nothing -- but nothing pinned that a PR below the cap which does not touch the file is skipped.
# Fetching every open PR would still produce a correct number, so only the counts and the absent
# row can see it. PR #55 changes one unrelated source file.
run_main("a PR that does not touch the file is not fetched or reported",
         want_rc=0, expect="scanned 2 open PR(s), 1 touching this file", absent="#55",
         stub_env={"STUB_MODE": "untouched"})

# Drafts are scanned like any other open PR -- a draft's row is on a branch and collides exactly as
# hard -- and are flagged so a reader can weigh the claim. The flag is the only thing distinguishing
# them in the report, and nothing asserted it.
run_main("a draft PR is scanned and flagged rather than dropped",
         want_rc=0, expect="[draft]", stub_env={"STUB_MODE": "draft"})

# "no numbered table whose header contains ... -- wrong path, or the table format changed". The
# same not-found condition on the BASE rather than on a PR, and the one that must stop the run:
# every number after it would be derived from an empty base.
run_main("no matching table in the base is an error, not an allocation from nothing",
         want_rc=1, expect="no numbered table whose header contains 'Nonexistent'",
         absent="next free number", extra=["--table-header", "Nonexistent"])

print("agreement with the gate, on the repository's real table:")

# ---------------------------------------------------------------------------------------------
# run() must decode subprocess output as UTF-8 regardless of the host locale.
#
# Not hygiene. text=True alone decodes with the LOCALE default, which on a Windows host is
# cp1252 -- and this tool's whole job is to pipe a UTF-8 Markdown document through a subprocess.
# The decode raised inside subprocess's reader THREAD, so proc.stdout became None while
# returncode stayed 0, and the tool then crashed 150 lines away in the table parser with
# "'NoneType' object has no attribute 'split'". trap_number.py was unusable on Windows, and
# unusable in a way that pointed the reader at the wrong file.
#
# Driven through `git show` of a real blob rather than a synthetic echo, so the arm exercises the
# same call the tool uses to read the base document.
# ---------------------------------------------------------------------------------------------
NON_ASCII = "trap row with an em dash \u2014 and typographic \u201cquotes\u201d\n"
with tempfile.TemporaryDirectory() as _d:
    _repo = Path(_d)
    _git = ["git", "-C", str(_repo)]
    for _args in (["init", "--quiet"],
                  ["config", "user.email", "t@example.com"],
                  ["config", "user.name", "t"]):
        subprocess.run([*_git, *_args], capture_output=True)
    (_repo / "doc.md").write_text(NON_ASCII, encoding="utf-8")
    subprocess.run([*_git, "add", "doc.md"], capture_output=True)
    subprocess.run([*_git, "commit", "--quiet", "-m", "x"],
                   capture_output=True)
    try:
        _got = run(["git", "-C", str(_repo), "show", "HEAD:doc.md"])
        case("run() decodes non-ASCII UTF-8 subprocess output", _got, NON_ASCII)
        case("...and returns a str, not the None that crashed the parser later",
             isinstance(_got, str), True)
    except Exception as exc:      # noqa: BLE001 - the point is that nothing escapes
        FAILURES.append(f"run() raised on non-ASCII UTF-8 output: {exc!r}")

DOC = HERE.parent.parent / "docs" / "GAME_COMPAT_ORCHESTRATION.md"
if not DOC.exists():  # fail closed: a moved document must not silently skip the only real-data arm
    FAILURES.append(f"the orchestration document is not at {DOC} -- this arm cannot run")
else:
    text = DOC.read_text(encoding="utf-8")
    tables, _, _ = parse_tables(text.split("\n"))
    gate: list[Table] = [t for t in tables if t.proper and t.all_rows_numbered
                         and "Instrument" in t.header]
    case("the gate finds exactly one Instrument table", len(gate), 1)
    if len(gate) == 1:
        gate_numbers = [n for _, n in gate[0].numbered_rows()]
        case("the allocator reads the same rows as the gate",
             table_numbers(text, "Instrument"), gate_numbers)
        # Stated as a floor rather than an exact number so the arm does not need editing on every
        # append -- but a floor still fails loudly if the parse collapses to a handful of rows.
        case("...and that is the whole table, not a truncated prefix",
             len(gate_numbers) > 150, True)

print()
if FAILURES:
    for f in FAILURES:
        print(f"FAIL: {f}", file=sys.stderr)
    print(f"\n{len(FAILURES)} case(s) failed.", file=sys.stderr)
    sys.exit(1)
print("all cases passed")
