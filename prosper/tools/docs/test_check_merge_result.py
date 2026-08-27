#!/usr/bin/env python3
"""Self-checking tests for check_merge_result.py (exit code is truth).

Each case builds a throwaway git repository with two branches and asks the tool about the merge, so
the git plumbing is exercised for real rather than mocked -- the plumbing IS the tool, and a mocked
`merge-tree` would test nothing.

The case that matters most is `clean merge, duplicate in the result`: BOTH inputs merge without a
textual conflict, and the invalid table exists only in the merge. That is the state CI cannot
observe, because at the time each head was validated the other row did not exist (#2211).

Run directly, or via ctest as doc_merge_result_checker.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "check_merge_result.py"
DOC = "prosper/docs/GAME_COMPAT_ORCHESTRATION.md"

FAILURES: list[str] = []


def table(*rows: str) -> str:
    body = "".join(f"| {n} | {what} | evidence |\n" for n, what in (r.split(":", 1) for r in rows))
    return "| # | Instrument | How it lied |\n|---|---|---|\n" + body + "\n### Next\n\nprose\n"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def write(repo: Path, text: str) -> None:
    p = repo / DOC
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def scenario(base: str, ours: str, theirs: str, d: str) -> Path:
    """A repo whose `master` holds `ours` and whose `lane` holds `theirs`, both from `base`."""
    repo = Path(d) / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "master")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "t")
    # Windows runners default core.autocrlf=true, which would put CRLF in the worktree and LF in the
    # index. Nothing here would fail on that today, and pinning it costs one line versus a Windows-
    # only surprise later -- this suite's harness has already had exactly one of those (the
    # line-number extractor that vanished on `C:\...` paths, green on Linux, red only on MinGW).
    git(repo, "config", "core.autocrlf", "false")
    write(repo, base)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    git(repo, "branch", "lane")
    write(repo, ours)
    git(repo, "commit", "-qam", "master moves")
    git(repo, "checkout", "-q", "lane")
    write(repo, theirs)
    git(repo, "commit", "-qam", "lane appends")
    return repo


def case(name: str, base: str, ours: str, theirs: str, *, want_rc: int,
         expect_text: str | None = None) -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = scenario(base, ours, theirs, d)
        proc = subprocess.run(
            [sys.executable, str(TOOL), "--no-fetch", "--base", "master", "--head", "lane"],
            cwd=repo, capture_output=True, text=True,
        )
    out = proc.stdout + proc.stderr
    if proc.returncode != want_rc:
        FAILURES.append(f"{name}: expected rc={want_rc}, got {proc.returncode}. out={out[:400]!r}")
        return
    if expect_text and expect_text not in out:
        FAILURES.append(f"{name}: expected {expect_text!r} in output, got {out[:400]!r}")
        return
    print(f"  ok  {name}")


BASE = table("1:a", "2:b", "3:c")

print("what the merge result says:")

# Ordinary: master gained nothing this branch collides with.
case("a clean merge with a valid result passes",
     BASE, BASE, table("1:a", "2:b", "3:c", "4:lane's row"),
     want_rc=0, expect_text="the merge RESULT passes the gate")

# THE case. Master took 4; the lane took 4 too and placed it a few lines away, so the merge is
# textually CLEAN and the duplicate exists only in the result. Neither head-time check could have
# seen it. This is what #2581 was, and what a green Docs job did not warn about.
case("a clean merge whose RESULT holds a duplicate is caught",
     BASE,
     table("1:a", "2:b", "3:c", "4:master's row"),
     table("1:a", "4:the lane's row", "2:b", "3:c"),
     want_rc=1, expect_text="duplicate row number 4")

# ...and the message must say what to do, since the whole point is that the author is otherwise
# looking at a green check.
case("the duplicate message names the repair",
     BASE,
     table("1:a", "2:b", "3:c", "4:master's row"),
     table("1:a", "4:the lane's row", "2:b", "3:c"),
     want_rc=1, expect_text="renumbering YOUR")

# A textual conflict is a different outcome from an invalid clean merge, and must not be reported as
# a table defect -- the file in the merge result is not a table at that point, it is conflict
# markers, and telling the author their numbering is broken would send them to the wrong place.
case("a textual conflict is reported as a conflict",
     BASE,
     table("1:a", "2:b", "3:c", "4:master's row"),
     table("1:a", "2:b", "3:c", "4:the lane's row"),
     want_rc=1, expect_text="MERGE CONFLICT")

# The --baseline wiring is what catches a row that VANISHES in the merge. Without this case, the
# tool could be passing no baseline at all and every case above would still pass.
case("a row deleted by the merge is caught",
     table("1:a", "2:b", "3:c", "4:d"),
     table("1:a", "2:b", "3:c", "4:d"),
     table("1:a", "2:b", "3:c"),
     want_rc=1, expect_text="GONE from this table: 4")

# The checker-failure branch, reachable through the ENVIRONMENT rather than the arguments. My first
# claim was that no arm was possible without a test-only injection point in production code; the
# #2616 review rejected that and was right. Copy the tool ALONE into a tempdir and its sibling
# `check_numbered_table.py` is absent, so the subprocess exits non-zero without ever reporting a
# table problem -- which is exactly the state the branch classifies. Six lines, no injection point.
#
# Worth keeping as a lesson as well as a test: "this path is untestable" is a claim like any other,
# and it was wrong because I had only considered the argument surface.
def cli_missing_checker(name: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        repo = scenario(BASE, BASE, table("1:a", "2:b", "3:c", "4:lane"), d)
        lone = Path(d) / "lone"
        lone.mkdir()
        (lone / TOOL.name).write_text(TOOL.read_text(encoding="utf-8"), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(lone / TOOL.name), "--no-fetch", "--base", "master",
             "--head", "lane"],
            cwd=repo, capture_output=True, text=True,
        )
    out = proc.stdout + proc.stderr
    if proc.returncode == 0:
        FAILURES.append(f"{name}: expected a non-zero exit, got 0: {out[:250]!r}")
    elif "RESULT is invalid" in out:
        FAILURES.append(f"{name}: a tooling failure was reported as a table finding: {out[:250]!r}")
    elif "checker itself failed" not in out:
        FAILURES.append(f"{name}: expected the checker-failure message, got {out[:250]!r}")
    else:
        print(f"  ok  {name}")


cli_missing_checker("a missing checker is reported as a tooling failure, not a table defect")


# Nothing to check is not a pass-by-accident: it is stated, so a caller cannot mistake "already
# merged" for "validated".
case("a head already in the base says so",
     BASE, table("1:a", "2:b", "3:c", "4:d"), BASE,
     want_rc=0, expect_text="already contained in the base")


# NON-ASCII rows. This file contained no non-ASCII byte at all, so CI could not express the case
# that broke the tool on Windows: `git()` piped the UTF-8 document through subprocess with
# text=True and no encoding, the locale default (cp1252) failed to decode it, and stdout came back
# None with returncode 0 -- dying 36 lines later on write_text(None). The real document is full of
# em dashes and typographic quotes, so every real invocation hit it while every test passed.
#
# The probe that reported the tool as unaffected ran it with no arguments and took the
# already-contained-in-the-base early return above, 36 lines before the affected code: a control
# that could not reach the case it was validating. Found in review of #3071.
#
# Shaped like the clean-merge arm at the top (base == ours, the lane appends) so the only variable
# is the non-ASCII content. Both `git show` calls are covered: the lane's row is read from the
# merge result and the base copy is read separately, and the base carries non-ASCII too.
NON_ASCII_BASE = table("1:an instrument that lied \u2014 and its \u201ccontrol\u201d",
                       "2:b", "3:c")
case("a table carrying em dashes and typographic quotes still validates",
     NON_ASCII_BASE, NON_ASCII_BASE,
     table("1:an instrument that lied \u2014 and its \u201ccontrol\u201d",
           "2:b", "3:c", "4:a lane row, also with an em dash \u2014"),
     want_rc=0, expect_text="the merge RESULT passes the gate")

# And a non-ASCII table whose merge RESULT is defective must still be CAUGHT, not merely
# not-crash: a decode fix that silently truncated the document would make this pass by finding no
# duplicate. Shaped on the duplicate arm above -- master takes 4, the lane takes 4 a few lines
# away, so the merge is textually CLEAN and the duplicate exists only in the result.
#
# The first version of this arm was VACUOUS and I nearly shipped it: it produced a merge conflict
# rather than a clean merge, its rc=1 matched want_rc for the wrong reason, and its expect_text of
# "\"4\"" matched the substring inside "100644" in the conflict listing. It passed with the
# decode bug present. Asserting the full message is what makes it real.
NON_ASCII_MASTER = table("1:an instrument that lied \u2014 and its \u201ccontrol\u201d",
                         "2:b", "3:c", "4:master's row \u2014")
NON_ASCII_LANE = table("1:an instrument that lied \u2014 and its \u201ccontrol\u201d",
                       "4:the lane's row \u2014", "2:b", "3:c")
case("a duplicate in a non-ASCII table's merge result is still caught",
     NON_ASCII_BASE, NON_ASCII_MASTER, NON_ASCII_LANE,
     want_rc=1, expect_text="duplicate row number 4")


print()
if FAILURES:
    for f in FAILURES:
        print(f"FAIL: {f}", file=sys.stderr)
    print(f"\n{len(FAILURES)} case(s) failed.", file=sys.stderr)
    sys.exit(1)
print("all cases passed")
