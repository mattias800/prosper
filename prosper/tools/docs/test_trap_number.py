#!/usr/bin/env python3
"""Self-checking tests for trap_number.py (exit code is truth).

The network half of that tool (`gh pr list`, the contents API) is not exercised here -- a test that
needs authentication is a test that gets disabled. What IS exercised is the half that can be wrong
silently: which rows it reads out of a document, and whether its answer agrees with the gate that
will judge the row it helps you write.

That agreement is the point of the last section, and it is not decoration. The allocator and
`check_numbered_table.py` parse the same table with DIFFERENT code -- the checker builds table runs
and requires every body row to be numbered, the allocator scans linearly from a matching header. If
they ever disagree, the allocator hands out a number the gate rejects, and the failure appears on
someone else's PR hours later. Pinning them against the repository's real Instrument table is the
only arm that can see that. The arm asserts a floor rather than an exact count deliberately: an
exact one would need editing on every append, which is how prose figures go stale.

Run directly, or via ctest as trap_number.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from check_numbered_table import Table, parse_tables  # noqa: E402
from trap_number import added_rows_from_patch, highest, table_numbers  # noqa: E402

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
import os
if os.environ.get("STUB_MODE") == "unauth":
    sys.stderr.write("gh: not authenticated" + chr(10))
    sys.exit(4)
if a[:2] == ["repo", "view"]:
    print("o/r")
elif a[:2] == ["pr", "list"]:
    import os
    if os.environ.get("STUB_MODE") == "capped":
        print(json.dumps([
            {"number": 11, "title": "lane A", "headRefOid": "aaa", "isDraft": False,
             "files": [{"path": "f%d.txt" % i} for i in range(100)]},
        ]))
        sys.exit(0)
    print(json.dumps([
        {"number": 11, "title": "lane A", "headRefOid": "aaa", "isDraft": False,
         "files": [{"path": DOC}]},
        {"number": 22, "title": "lane B", "headRefOid": "bbb", "isDraft": False,
         "files": [{"path": DOC}]},
    ]))
elif a[0] == "api":
    print(table([1, 2, 3]))
elif a[:2] == ["pr", "diff"]:
    import os
    if os.environ.get("STUB_MODE") == "stacked" and a[2] == "22":
        print("diff --git a/other.txt b/other.txt\\n--- a/other.txt\\n+++ b/other.txt\\n@@\\n+unrelated\\n")
    else:
        print("diff --git a/%s b/%s\\n--- a/%s\\n+++ b/%s\\n@@\\n+| 3 | r3 | e |\\n" % (DOC, DOC, DOC, DOC))
else:
    sys.exit(1)
'''

DOC_REL = Path("prosper/docs/GAME_COMPAT_ORCHESTRATION.md")
SEED = ("| # | Instrument | How it lied |\n|---|---|---|\n"
        "| 1 | r1 | e |\n| 2 | r2 | e |\n\n### Next\n")


def run_main(name: str, *, want_rc: int, expect: str | None = None, absent: str | None = None,
             extra: list[str] | None = None, stub_env: dict[str, str] | None = None) -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        gh = bin_dir / "gh"
        gh.write_text(GH_STUB, encoding="utf-8")
        gh.chmod(gh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        repo = root / "repo"
        (repo / DOC_REL.parent).mkdir(parents=True)
        (repo / DOC_REL).write_text(SEED, encoding="utf-8")
        for cmd in (["init", "-q", "-b", "master"],
                    ["config", "user.email", "t@example.invalid"],
                    ["config", "user.name", "t"], ["add", "-A"], ["commit", "-qm", "base"]):
            subprocess.run(["git", "-C", str(repo), *cmd], capture_output=True)

        env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
        env.update(stub_env or {})
        proc = subprocess.run(
            [sys.executable, str(HERE / "trap_number.py"), "--no-fetch", "--base", "master",
             *(extra or [])],
            cwd=repo, capture_output=True, text=True, env=env,
        )
    out = proc.stdout + proc.stderr
    if proc.returncode != want_rc:
        FAILURES.append(f"{name}: expected rc={want_rc}, got {proc.returncode}: {out[:400]!r}")
    elif expect and expect not in out:
        FAILURES.append(f"{name}: expected {expect!r} in output, got {out[:400]!r}")
    elif absent and absent in out:
        FAILURES.append(f"{name}: did not expect {absent!r}, got {out[:400]!r}")
    else:
        print(f"  ok  {name}")


# THE arm. Two PRs whose patches both add row 3, so the collision branch actually runs. Under the
# 5-vs-6 unpack this exits 1 with a ValueError, and asserting on rc is what catches it.
run_main("a real collision reaches the advice instead of crashing",
         want_rc=0, expect="COLLISION: 3 claimed by more than one open PR")

run_main("...and it names the racing PRs", want_rc=0, expect="#11")

run_main("...and suggests stepping clear rather than to the next free number",
         want_rc=0, expect="CONSIDER")

# --quiet returns before the report, which is exactly why the crash was invisible to a scripted
# caller. It must still be right, and it must not be the only path that works.
run_main("--quiet prints just the number", want_rc=0, expect="4", extra=["--quiet"])

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

# The truncation guard, executed rather than reasoned about: the stub returns 2 PRs, so --limit 2
# cannot distinguish "there are 2" from "we stopped at 2".
run_main("a truncated PR list is an error, not a smaller answer",
         want_rc=1, expect="TRUNCATED", extra=["--limit", "2"])

print("agreement with the gate, on the repository's real table:")

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
