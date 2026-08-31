#!/usr/bin/env python3
"""Self-checking tests for check_numbered_table.py (exit code is truth).

Every case here is a defect this repository actually produced, or a correct shape that an
earlier revision of the checker wrongly rejected. The second group matters as much as the
first: a check that fires on correct data gets deleted rather than heeded, so the false-positive
cases are the ones that keep it alive.

Run directly, or via ctest as doc_table_checker.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_numbered_table import check  # noqa: E402

CHECKER = Path(__file__).resolve().parent / "check_numbered_table.py"

FAILURES: list[str] = []
SKIPPED: list[str] = []


def run(name: str, body: str, *, ordered: bool = False, header: str | None = None,
        baseline: str | None = None,
        want_problems: bool, expect_text: str | None = None, want_count: int | None = None,
        want_lines: list[int] | None = None) -> None:
    """Check one document.

    `want_count`/`want_lines` exist because asserting mere truthiness cannot see a detector that
    finds the FIRST instance of a defect and stops -- a common shape (early return, comparing only
    against the previous element) that single-instance fixtures are blind to. An earlier draft of
    this checker had exactly that bug, and a truthiness-only version of this suite stayed green
    when it was reintroduced. Any case covering more than one defect must pin the count.

    `baseline` is the PRIOR text of the same document (--baseline), so a case can assert what the
    checker sees about a row that used to exist. `baseline="<missing>"` names a path that is not
    there, which is how the fail-closed behaviour is pinned.
    """
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "case.md"
        path.write_text(body, encoding="utf-8")
        base_path: Path | None = None
        if baseline == "<missing>":
            base_path = Path(d) / "no-such-baseline.md"
        elif baseline is not None:
            base_path = Path(d) / "baseline.md"
            base_path.write_text(baseline, encoding="utf-8")
        problems = check(path, ordered, header, base_path)
    if bool(problems) != want_problems:
        FAILURES.append(
            f"{name}: expected {'problems' if want_problems else 'clean'}, got "
            f"{problems if problems else 'clean'}"
        )
        return
    if want_count is not None and len(problems) != want_count:
        FAILURES.append(
            f"{name}: expected exactly {want_count} problem(s), got {len(problems)}: {problems}"
        )
        return
    if want_lines is not None:
        # Match ":<digits>: " rather than splitting on ":" -- a Windows path starts "C:\...", so
        # field 1 is the path remainder and every line number silently vanished. Found by CI on
        # Windows MinGW, which is the only job that runs this test.
        got = sorted({int(m.group(1)) for p in problems if (m := re.search(r":(\d+): ", p))})
        if got != sorted(want_lines):
            FAILURES.append(f"{name}: expected problems at lines {sorted(want_lines)}, got {got}")
            return
    if expect_text and not any(expect_text in p for p in problems):
        FAILURES.append(f"{name}: expected a message containing {expect_text!r}, got {problems}")
        return
    print(f"  ok  {name}")


# The line-number extractor must survive a Windows path. Splitting the message on ":" put the
# drive letter in field 0 and the path remainder in field 1, so every line number silently vanished
# and three want_lines cases reported "got []" -- green on Linux, red only on Windows MinGW, the
# one job that runs this test. Guarded here because the harness itself is not otherwise covered.
_win = r"C:\Users\runner\Temp\case.md:4: blank line splits the table that starts at line 1"
_posix = "/tmp/case.md:6: blank line splits the table that starts at line 1"
assert re.search(r":(\d+): ", _win).group(1) == "4", "extractor broken on Windows-shaped paths"
assert re.search(r":(\d+): ", _posix).group(1) == "6", "extractor broken on POSIX paths"

TABLE = "| # | What |\n|---|---|\n| 1 | a |\n| 2 | b |\n| 3 | c |\n"

print("defects this repo produced:")

# The 2026-08-01 break: a blank line mid-table orphans everything after it.
run("blank line splits a table",
    "| # | What |\n|---|---|\n| 1 | a |\n\n| 2 | b |\n",
    want_problems=True, expect_text="blank line splits the table")

# A blank between the delimiter and the first row orphans the entire body.
run("blank line after the delimiter",
    "| # | What |\n|---|---|\n\n| 1 | a |\n| 2 | b |\n",
    want_problems=True, expect_text="blank line splits the table")

# TWO breaks in one table. The second lives inside a fragment, not inside a proper table, so a
# checker that compares each run only against the run immediately above it reports the first and
# silently misses the second. The file that prompted this check had exactly this shape.
# N4: a THREE-blank gap. Every other case has a one-line gap, so per-fragment and per-line
# reporting are indistinguishable across them -- reverting the bound left the suite green while
# this shape went from 1 problem to 3. That bound is half of what fixed the 202-error explosion.
run("a multi-line blank gap is one problem, not one per line",
    "| # | What |\n|---|---|\n| 1 | a |\n\n\n\n| 2 | b |\n",
    want_problems=True, want_count=1)

run("two blank lines split the same table",
    "| # | What |\n|---|---|\n| 1 | a |\n\n| 2 | b |\n\n| 3 | c |\n",
    want_problems=True, want_count=2, want_lines=[4, 6],
    expect_text="blank line splits the table")

# R1 (SUPERSEDED by #3088; kept as the corrected case rather than deleted, since it is what
# proves the fix is not merely permissive). A row missing ONLY its leading pipe, whose interior
# pipes still land on the header's cell count, renders as a completely normal, unbroken 3-row
# table -- measured against GitHub's renderer (module docstring). The previous version of this
# test asserted the OPPOSITE ("ends the table... did it lose its leading '|'?"), which was a false
# positive on correct Markdown: a checker that fires on correct data gets deleted rather than
# heeded, and this shape is exactly the kind of thing #3088's fix corrects as a side effect of
# making the wrapped-row blind spot visible (parse_tables absorbs the row into the still-open
# table instead of ending it; ARITY finds nothing wrong because there is nothing wrong).
run("a row missing its leading pipe still renders correctly when its arity matches",
    "| # | What |\n|---|---|\n| 1 | a |\n2 | b |\n| 3 | c |\n",
    want_problems=False)

# The genuine defect hides in the arity, not in the missing pipe by itself: once the row's cell
# count actually disagrees with the header -- the realistic shape of a lost leading pipe, since a
# stray edit rarely preserves the exact column count -- ARITY catches it, on the absorbed row
# itself, the same class that already catches an unescaped pipe in a code span (#2108).
run("a row missing its leading pipe IS still a defect once its arity disagrees",
    "| # | Instrument | How it lied |\n|---|---|---|\n| 1 | x | y |\n2 | z |\n| 3 | a | b |\n",
    want_problems=True, want_count=1, want_lines=[4],
    expect_text="has 2 cells but the header")

# --ordered is a narrower residual case worth naming rather than hiding: a leading-pipe loss
# defeats LEADING_NUMBER (`^\s*\|\s*(\d+)\s*\|`) even when it does not defeat ARITY, so the
# absorbed row is never counted as numbered and the table fails all_rows_numbered -- reported as
# "no numbered table found" rather than silently accepted. This is a real behaviour change from R1
# above (whose --ordered form is want_problems=False, since the plain sweep this repository
# actually runs on every PR passes no --ordered flag at all), recorded so it is a decision and not
# a surprise.
run("--ordered still flags a leading-pipe loss, even when arity is fine",
    "| # | What |\n|---|---|\n| 1 | a |\n2 | b |\n| 3 | c |\n",
    ordered=True, want_problems=True, expect_text="no numbered table")

# Structural errors are reported first and the broken remainder leaves the sequence check, the way
# a compiler stops elaborating a malformed declaration. What matters is that the file CANNOT pass
# green while hiding a duplicate below the break -- the previous revision reported
# "contiguous and unbroken", exit 0, on exactly this input. Fix the structure, re-run, get the
# duplicate. (Uses a blank-line break, not a missing leading pipe: since #3088 the latter no longer
# ends an open table by itself -- see R1 above -- so it can no longer orphan a fragment this way.
# A blank line still can, and is this project's own real incident for exactly this shape.)
run("a duplicate hidden below an interruption cannot pass green",
    "| # | What |\n|---|---|\n| 1 | a |\n\n| 5 | c |\n| 5 | d |\n",
    ordered=True, want_problems=True, want_count=1, want_lines=[4],
    expect_text="blank line splits the table")

# The #1696 collision: two branches append the same number; the merge is textually clean. This is
# the check that survives the removal of gaplessness, and it is the one that protects the citation
# contract -- a duplicate makes every by-number reference ambiguous.
run("duplicate row number",
    "| # | What |\n|---|---|\n| 32 | a |\n| 33 | b |\n| 34 | c |\n| 35 | d |\n| 32 | e |\n",
    ordered=True, want_problems=True, expect_text="duplicate row number 32")

# Every duplicate is reported, not just the first: a detector that compares only against the
# previous element -- the shape this suite has already caught twice -- would find the second 32 and
# miss the second 33, and a single-duplicate fixture cannot tell the two implementations apart.
run("every duplicate is reported, not just the first",
    "| # | What |\n|---|---|\n| 32 | a |\n| 33 | b |\n| 32 | c |\n| 34 | d |\n| 33 | e |\n",
    ordered=True, want_problems=True, want_count=2, want_lines=[5, 7])

run("out of ascending order",
    "| # | What |\n|---|---|\n| 1 | a |\n| 3 | b |\n| 2 | c |\n",
    ordered=True, want_problems=True, expect_text="out of ascending order")

# The message must forbid the repair that breaks the citation contract. An author whose row is out
# of order because another lane's higher number landed first can fix it by MOVING the line; the one
# thing they must not do is renumber: 118 references on 110 lines in 46 files cite these rows by
# number (origin/master 7413647a, 2026-08-17), naming 63 distinct rows -- and 50 of those
# references are .cpp/.hpp/.py COMMENTS, not prose.
run("the out-of-order message forbids renumbering",
    "| # | What |\n|---|---|\n| 1 | a |\n| 3 | b |\n| 2 | c |\n",
    ordered=True, want_problems=True, expect_text="NOT renumber")
run("...and names the repair that is allowed",
    "| # | What |\n|---|---|\n| 1 | a |\n| 3 | b |\n| 2 | c |\n",
    ordered=True, want_problems=True, expect_text="MOVE the row")

# ---------------------------------------------------------------------------------------------
# GAPS ARE LEGAL (#2089). This case asserted the OPPOSITE until 2026-08-17, and flipping it is the
# whole behavioural change -- so it is the arm to read first if this suite ever disagrees with what
# CI does. Gaplessness made a lane's `Docs` job red BY CONSTRUCTION whenever a lower number was
# still in flight, which no edit of that lane's own branch could repair; the deleted-row job it was
# kept for now belongs to --baseline, which does it strictly better (see the two cases below that
# gaplessness could not see at all).
run("a gap is legal: the number belongs to a branch that has not landed",
    "| # | What |\n|---|---|\n| 1 | a |\n| 4 | b |\n",
    ordered=True, want_problems=False)

run("a permanent gap from an abandoned claim is legal",
    "| # | What |\n|---|---|\n| 184 | a |\n| 185 | b |\n| 186 | c |\n| 188 | d |\n| 189 | e |\n",
    ordered=True, want_problems=False)

# ---------------------------------------------------------------------------------------------
# PERSISTENCE (--baseline). A row that VANISHES leaves a perfectly well-formed file: uniqueness
# passes, ascent passes, structure passes, and every citation to it silently resolves to nothing.
BASE = "| # | What |\n|---|---|\n| 1 | a |\n| 2 | b |\n| 3 | c |\n| 4 | d |\n"

run("a deleted INTERIOR row is caught",
    "| # | What |\n|---|---|\n| 1 | a |\n| 2 | b |\n| 4 | d |\n",
    ordered=True, baseline=BASE, want_problems=True, want_count=1, expect_text="GONE from this table: 3")

# THE discriminating case, and the reason gaplessness was not merely relaxed but replaced. Deleting
# the HIGHEST row leaves 1,2,3 -- still gapless, still ascending, still unique. The old --sequential
# gate passed this green, measured on master's real Instrument table (186 rows at 0c268362;
# deleting row 186 -> rc=0).
run("a deleted TAIL row is caught -- gaplessness could not see this",
    "| # | What |\n|---|---|\n| 1 | a |\n| 2 | b |\n| 3 | c |\n",
    ordered=True, baseline=BASE, want_problems=True, want_count=1, expect_text="GONE from this table: 4")

# The instrument-trap-41 / #1701 shape at full size: `git checkout <old-branch> -- <file>` restores
# the file whole and silently drops every row added since. On master's real table that is 61 rows,
# and the old gate reported "contiguous and unbroken", rc=0.
run("a whole-file revert that truncates the tail is caught, with every lost row named",
    "| # | What |\n|---|---|\n| 1 | a |\n",
    ordered=True, baseline=BASE, want_problems=True, want_count=1,
    expect_text="3 row number(s) present in the baseline are GONE from this table: 2, 3, 4")

# A renumber is a deletion of the old key. Nothing before --baseline could see this at all, and it
# is the operation the table's maintenance note forbids outright.
run("a renumbered row is caught -- the old number is gone",
    "| # | What |\n|---|---|\n| 1 | a |\n| 2 | b |\n| 3 | c |\n| 9 | d |\n",
    ordered=True, baseline=BASE, want_problems=True, expect_text="GONE from this table: 4")

# The false-positive guard that keeps the check alive: appending is the ordinary operation and must
# never fire. Without this case, a rule as broken as "the row sets must be EQUAL" would pass the
# four cases above.
run("appending rows is not a deletion",
    BASE + "| 5 | e |\n| 6 | f |\n",
    ordered=True, baseline=BASE, want_problems=False)

# A gap in the SUBJECT that the baseline also had is not a deletion either -- the two rules are
# independent, and conflating them would reintroduce gaplessness through the back door.
run("a gap both files share is not a deletion",
    "| # | What |\n|---|---|\n| 1 | a |\n| 4 | d |\n| 7 | g |\n",
    ordered=True, baseline="| # | What |\n|---|---|\n| 1 | a |\n| 4 | d |\n",
    want_problems=False)

# A row that leaves the table by being STRUCTURALLY orphaned -- a blank line splitting the table
# above it -- is reported as gone, on top of the structure error naming the real cause. That is a
# decision, not an accident: rows below a break drop out of every numbered check, so suppressing
# persistence whenever structure complains would let a deletion hide behind a stray blank line,
# which is the shape "a duplicate hidden below an interruption cannot pass green" already guards.
# Two problems, and the structure one is listed first because it is the repair.
run("rows orphaned by a table split are reported gone, alongside the structure error",
    "| # | What |\n|---|---|\n| 1 | a |\n| 2 | b |\n\n| 3 | c |\n| 4 | d |\n",
    ordered=True, baseline=BASE, want_problems=True, want_count=2,
    expect_text="GONE from this table: 3, 4")

# N1 (#2610 review) -- a bound on how far a NEW number may sit above the baseline's maximum. Gaps are
# legal; a mistyped digit is not. `1189` for `189` passes uniqueness and ascent, reports a thousand
# unused numbers, and permanently poisons the space, because every later allocation comes off the new
# maximum. The old gapless rule caught this incidentally and nothing else does.
run("a wildly out-of-range number is a typo, not a gap",
    BASE + "| 1004 | typo |\n",
    ordered=True, baseline=BASE, want_problems=True, expect_text="more than 50 above")

# ...and the discriminating counter-arm: a DELIBERATE step clear of a contested band must still pass.
# Without this case, "reject any gap at all" would satisfy the case above and undo the whole change.
run("a deliberate step clear of a collision is still legal",
    BASE + "| 8 | stepped clear of a three-way race on 5 |\n",
    ordered=True, baseline=BASE, want_problems=False)

# The bound is measured against the BASE's maximum, never against another lane's timing -- so a
# number just inside it passes no matter what else is in flight.
run("a number just inside the bound passes",
    BASE + "| 54 | 50 above the baseline max of 4 |\n",
    ordered=True, baseline=BASE, want_problems=False)

# ...and a row the baseline ALREADY had is never re-judged by the bound, or every file whose table
# legitimately contains a historical jump would fail forever.
run("an existing far-out row is not re-judged",
    "| # | What |\n|---|---|\n| 1 | a |\n| 900 | historical |\n",
    ordered=True, baseline="| # | What |\n|---|---|\n| 1 | a |\n| 900 | historical |\n",
    want_problems=False)

# Fails closed. A baseline that cannot be read must be an ERROR: the caller asked for the check, and
# a silent skip would make a green run mean nothing -- exactly the "no tests ran and everything
# passed share an exit code" shape the charter warns about.
run("an unreadable baseline is an error, not a silent skip",
    BASE, ordered=True, baseline="<missing>", want_problems=True, expect_text="--baseline")

# ...and a baseline whose matching table cannot be found is an error too. This is the one that a
# same-source positive control cannot reach: if selection silently returned a DIFFERENT table, every
# deletion would read as intact and all six cases above would still pass.
run("a baseline with no matching numbered table is an error",
    "| # | Instrument | x |\n|---|---|---|\n| 1 | a | b |\n",
    ordered=True, header="Instrument", baseline="| # | Other | x |\n|---|---|---|\n| 1 | a | b |\n",
    want_problems=True, expect_text="--baseline")

# ---------------------------------------------------------------------------------------------
# ARITY (#2108). GFM splits a row into cells on `|` BEFORE parsing inline content, so a pipe
# inside a code span is a cell boundary and the excess cells are SILENTLY DISCARDED. Six rows of
# GAME_COMPAT_ORCHESTRATION.md rendered truncated on GitHub for months, including instrument trap
# 40 -- whose subject is `cmd \| tail`, and which was broken by writing `cmd | tail` unescaped.
# The raw file is correct-looking, so an editor never shows it; only the rendered page is wrong.

# The trap-40 shape, reduced. This is the whole defect in one row.
run("an unescaped pipe in a code span truncates the row",
    "| # | Instrument | How it lied |\n|---|---|---|\n| 1 | x | so `cmd | tail` discards it |\n",
    want_problems=True, want_count=1, want_lines=[3],
    expect_text="SILENTLY DISCARDS")

# The message must point at the offending CHARACTER, not merely report a count -- a 3,000-character
# trap row has no other way to be found, and locating it is this feature's whole ergonomic claim.
#
# The COLUMN NUMBER is what pins it. An excerpt substring does not: on a short fixture the 46-char
# window still contains the right text no matter which boundary it centres on, so mutating
# `bounds[expected - 1]` to `bounds[0]` left an excerpt-only assertion green (#2116 review). The
# stray pipe here is the one inside the code span, at 1-based column 19; `bounds[0]` would say 5.
run("the message locates the first stray pipe by column",
    "| # | Instrument | How it lied |\n|---|---|---|\n| 1 | x | so `cmd | tail` discards it |\n",
    want_problems=True, expect_text="First stray boundary at column 19: ")

# Same assertion on a row long enough that a wrong boundary choice cannot coincidentally land in
# the same excerpt window -- the short fixture above cannot distinguish them by excerpt alone.
run("the located column is the stray pipe, not the first pipe",
    "| # | Instrument | How it lied |\n|---|---|---|\n| 1 | x | "
    + "padding text that is comfortably wider than the excerpt window. " * 3
    + "tail `a | b` end |\n",
    want_problems=True, expect_text="at column 211: ")

# Trap 87's shape, and a DIFFERENT defect from the code-span one: a row that grows a genuine extra
# column the table's header never declared. Its `#1891, #1968` evidence cell was discarded whole.
run("a phantom extra column is discarded whole",
    "| # | Instrument | How it lied |\n|---|---|---|\n| 1 | x | text | #1891, #1968 |\n",
    want_problems=True, want_count=1, want_lines=[3])

# Multiple bad rows must ALL be reported. A detector that compares only until its first hit -- the
# shape this suite already caught once in the structure class -- would report 1 and pass the rest.
run("every short-changed row is reported, not just the first",
    "| # | Instrument | How it lied |\n|---|---|---|\n"
    "| 1 | x | `a | b` |\n| 2 | y | fine |\n| 3 | z | `c | d` |\n| 4 | w | `e | f` |\n",
    want_problems=True, want_count=3, want_lines=[3, 5, 6])

# The opposite direction: GFM pads a short row, so a dropped cell renders as a blank column rather
# than as anything visible. Trap 53 records this happening twice in one scripted edit.
run("a row that dropped a cell",
    "| # | Instrument | How it lied |\n|---|---|---|\n| 1 | x |\n",
    want_problems=True, want_count=1, want_lines=[3],
    expect_text="pads the row with empty cells")

# Worse than any row defect: GFM requires the delimiter to match the header, so this block does not
# render as a table AT ALL. Reported separately because the symptom is unrelated -- the whole table
# disappears into a paragraph of literal pipes.
run("a delimiter that disagrees with the header",
    "| # | Instrument | How it lied |\n|---|---|\n| 1 | x | y |\n",
    want_problems=True, expect_text="does NOT render as a table at all")

# ---------------------------------------------------------------------------------------------
# #3088: a row wrapped onto a second physical line. GFM does not require a table row to start
# with `|` -- it keeps consuming non-blank lines as rows of the same table regardless. Before this
# fix, parse_tables required every row to start with `|`, so a continuation line simply fell out
# of the run: never counted, never arity-checked, and -- when nothing pipe-prefixed happened to
# follow it -- entirely invisible. Reproduced by hand against current master before the fix
# (issue #3088): this exact fixture, with the fix reverted, prints
# "1 table(s), 1 rows, unbroken, every row matching its header" and exits 0.
#
# This is the PR #3049 shape reduced to a fixture: a two-column table whose answer cell wraps, so
# the qualifying clause lands on its own line with nothing to mark it as part of the row. Header
# and row count intentionally stay small so the mismatch (1 cell vs 2) is unambiguous.
run("a row wrapped onto a second physical line is caught (#3088)",
    "| Question | Answer |\n|---|---|\n| Does X leak? | no, bounded by the page count, not the "
    "wider\ncontext, per the #2790 fix |\n",
    want_problems=True, want_count=1, want_lines=[4],
    expect_text="has 1 cells but the header")

# The AGENTS.md:917 shape: a paragraph that forgot the blank line before it, landing directly
# under a table's last row. GFM absorbs the first line of that paragraph as another row with no
# pipes at all (a single cell), and every following non-blank line keeps being absorbed too until
# a real blank line appears -- both physical lines of the two-line paragraph below become rows.
run("a paragraph directly below a table with no blank line is caught (#3088)",
    "| # | What |\n|---|---|\n| 1 | a |\nThis paragraph forgot its blank line\nand runs two "
    "lines besides.\n",
    want_problems=True, want_count=2, want_lines=[4, 5])

# The discriminating counter-arm, and the reason the fix is not simply "absorb everything": a
# block construct that legitimately follows a table with no blank line must NOT be swallowed into
# it. Each shape below was checked against GitHub's renderer (module docstring) and renders as its
# own block, never as an absorbed row -- so a checker that instead absorbed it would report a
# false arity mismatch here (1 cell vs the header's 2), which is exactly what makes this arm
# falsifiable: deleting interrupts_table()'s effect turns every one of these red.
run("a heading right after a table is not absorbed",
    "| # | What |\n|---|---|\n| 1 | a |\n## A heading with several words\n",
    want_problems=False)
run("a bullet list right after a table is not absorbed",
    "| # | What |\n|---|---|\n| 1 | a |\n- a list item with several words\n",
    want_problems=False)
run("an ordered list right after a table is not absorbed",
    "| # | What |\n|---|---|\n| 1 | a |\n1. a list item with several words\n",
    want_problems=False)
run("a blockquote right after a table is not absorbed",
    "| # | What |\n|---|---|\n| 1 | a |\n> a quoted line with several words\n",
    want_problems=False)
run("a thematic break right after a table is not absorbed",
    "| # | What |\n|---|---|\n| 1 | a |\n---\n",
    want_problems=False)
run("an HTML block right after a table is not absorbed",
    "| # | What |\n|---|---|\n| 1 | a |\n<div>a whole html block</div>\n",
    want_problems=False)

# A conflict marker is handled by an earlier, separate scan (see below), but it is also a
# realistic "line right after a table" shape and must not be absorbed as a row either -- covered
# by the existing "a conflict marker after the last row is caught" case further down, which still
# passes unmodified because that scan runs and returns before parse_tables is ever reached.

# A stray conflict marker AFTER the last row. This is the shape that escaped every other class and
# reached a pushed branch: the marker is not a table row, so it merely ends the table, and with no
# orphaned rows below it there is no fragment to report. Without this arm the file below is reported
# "unique and ascending" -- over a line that is a merge artifact.
run("a conflict marker after the last row is caught",
    TABLE + ">>>>>>> 8f124852 (some commit subject)\n",
    ordered=True, want_problems=True, expect_text="unresolved merge conflict marker")

run("a conflict marker inside the table is caught",
    "| # | What |\n|---|---|\n| 1 | a |\n<<<<<<< HEAD\n| 2 | b |\n",
    want_problems=True, expect_text="unresolved merge conflict marker")

# THE ARM THE FENCE FIX NEEDED. The first draft of the conflict scan ran before parse_tables, with
# no fence tracking and always on -- so it rejected any document that PASTES an example conflict
# inside a ``` block, which is how this defect gets documented. This file's own header settled that
# question for every other class ("this file's own defect example would otherwise fail the check
# that documents it"); the scan had to make the same trade (#2610 review).
run("a conflict marker inside a fence is an example, not a defect",
    "```\n>>>>>>> 8f124852 (some commit)\n```\n\n" + TABLE,
    ordered=True, want_problems=False)

run("...and one inside a fence does not fire on the plain sweep either",
    "```\n<<<<<<< HEAD\n| 5 | a |\n>>>>>>> other\n```\n\n" + TABLE,
    want_problems=False)

# The counter-arm that keeps the fix honest: a marker AFTER the fence closes is still a defect, so
# "skip fenced" cannot decay into "skip everything once a fence is seen".
run("a marker after a closed fence is still caught",
    "```\nexample output\n```\n\n" + TABLE + ">>>>>>> 8f124852 (real leftover)\n",
    ordered=True, want_problems=True, expect_text="unresolved merge conflict marker")

# An UNCLOSED fence silences every check from that point on, so a real marker after it reads clean.
# That is the fenced-region skip decaying from a trade into a blind spot, and unlike the
# fenced-example case it has no legitimate shape (#2610 review). Measured safe: 0 of 101 tracked
# Markdown files end unbalanced.
run("an unclosed fence hiding a real marker is caught",
    TABLE + "```\nsome output\n>>>>>>> 8f124852 (real leftover)\n",
    ordered=True, want_problems=True, expect_text="opened and never closed")

run("an unclosed fence is caught on the plain sweep too",
    "| # | a |\n|---|---|\n| 1 | x |\n\n```\nunterminated\n",
    want_problems=True, expect_text="opened and never closed")

# ...and a BALANCED fence must not fire, or every document that quotes output would be rejected.
run("a balanced fence is not an unterminated one",
    TABLE + "```\nsome output\n```\n",
    ordered=True, want_problems=False)

# `=======` is a legal setext heading underline, so it must NOT be treated as a marker -- a checker
# that fires on correct Markdown gets deleted rather than heeded.
run("a setext heading underline is not a conflict marker",
    "Heading\n=======\n\n" + TABLE, ordered=True, want_problems=False)

print("correct shapes the arity check must not reject:")

# THE discriminating case for escape handling: remove it and this correct row is reported. It is
# also the fix this check prescribes, so a checker that rejected it would be self-contradictory.
run(r"an escaped pipe is content, not a boundary",
    "| # | Instrument | How it lied |\n|---|---|---|\n| 1 | x | so `cmd \\| tail` discards it |\n",
    want_problems=False)

# A backslash RUN of any length escapes the pipe: the rule is "unless the immediately preceding
# character is a backslash", with no parity counting. cmark-gfm splits cells in a pre-inline pass
# that inspects exactly one character; CommonMark's parity rule governs the inline pass, which
# never runs on a cell boundary.
#
# An earlier revision of this checker counted parity and pinned the OPPOSITE of this case, on the
# reasoning that `\\|` is a literal backslash plus a real delimiter. GitHub's renderer says
# otherwise -- 0 through 4 backslashes were put through it, and every run of length >= 1 keeps the
# row intact while 0 splits it. Reasoning about the case instead of measuring it is what produced
# the wrong rule; the oracle was already open for the single-backslash case (#2116 review).
run(r"a doubled backslash still escapes the pipe after it",
    "| x | y |\n|---|---|\n| a | b\\\\|c |\n",
    want_problems=False)

run(r"a single backslash escapes the pipe after it",
    "| x | y |\n|---|---|\n| a | b\\|c |\n",
    want_problems=False)

# ...and the unescaped form of the same row must still fail, or the two cases above would be
# satisfied by a checker that had simply stopped detecting anything.
run(r"the same row unescaped is still a defect",
    "| x | y |\n|---|---|\n| a | b|c |\n",
    want_problems=True, want_count=1, want_lines=[3])

# Leading and trailing pipes are optional decoration in GFM and do not create empty cells, so
# `| a | b | c |` is three cells and not five. Note WHY this needs the count pinned rather than a
# want_problems=False case: the comparison is relative, so a checker that counted the decorative
# leading pipe would shift header and row together, still detect every mismatch, and merely report
# both numbers one too high. A truthiness case for it cannot fail -- an earlier draft of this suite
# had exactly that void case, caught by mutating the rule and watching the suite stay green.
run("the reported cell counts are the ones a reader would count",
    "| # | Instrument | How it lied |\n|---|---|---|\n| 1 | x | so `cmd | tail` discards it |\n",
    want_problems=True, expect_text="has 4 cells but the header at line 1 has 3")

# An intentionally empty cell is still a cell: only the FIRST and LAST segments are decoration.
# Discriminating against the obvious shortcut of dropping every blank segment.
run("an empty interior cell still counts",
    "| a | b | c |\n|---|---|---|\n| 1 |  | 3 |\n",
    want_problems=False)

# A ``` block is not a table, and a checker that cries wolf on pasted tool output gets disabled
# rather than heeded -- which would cost the whole gate. The fenced block here is a COMPLETE,
# well-formed table with a bad row, so it is only invisible because fences are skipped; a block of
# bare `cmd | tail` lines would pass for the unrelated reason that they lack a leading pipe.
run("a mis-sized table inside a fence is not a defect",
    "| # | What |\n|---|---|\n| 1 | a |\n\n```\n| # | What |\n|---|---|\n| 1 | a | b |\n```\n",
    want_problems=False)

# KNOWN LIMIT, pinned so it is a decision rather than a surprise: a pipe run with no delimiter row
# has no header to be measured against, so its rows are not arity-checked. The structure class is
# what reports such a fragment when it abuts a real table; an isolated one is invisible to both.
run("a delimiter-less pipe block is not arity-checked",
    "prose\n\n| hand | written | with | four |\n| 1 | 2 |\n",
    want_problems=False)

print("fails closed (never vacuously green):")

# Absence of a table is an error only when we were pointed at ONE specific table (--ordered):
# then it means the path is wrong or the format changed, and passing would let the check go green
# forever on a file it can no longer see. In a plain structure sweep it is ordinary -- 35 of this
# repo's 77 Markdown files contain no table at all, and failing on those makes the tool unusable
# for the sweep it is meant to support.
run("no table at all, --ordered", "just prose, no table\n",
    ordered=True, want_problems=True, expect_text="no Markdown tables found")
run("empty file, --ordered", "", ordered=True, want_problems=True)
run("no table at all, structure sweep", "just prose, no table\n", want_problems=False)
run("empty file, structure sweep", "", want_problems=False)
run("--ordered with no numbered table",
    "| name | value |\n|---|---|\n| a | 1 |\n",
    ordered=True, want_problems=True, expect_text="no numbered table")
run("--ordered ambiguous between two numbered tables",
    "| # | a |\n|---|---|\n| 1 | x |\n\ntext\n\n| # | b |\n|---|---|\n| 1 | y |\n",
    ordered=True, want_problems=True, expect_text="ambiguous")

print("correct shapes an earlier revision wrongly rejected:")

# N1: backward attribution must be BOUNDED. Without an adjacency rule these produced 3 and 202
# errors respectively on correct documents -- and two of the three told the author to delete the
# blank lines that correctly ended the table. Wrong guidance is worse than a false positive,
# because an agent who follows it damages the file.
run("a pipe-leading paragraph after prose is not a table fragment",
    "| # | a |\n|---|---|\n| 1 | x |\n\nSome prose paragraph.\n\n| this line merely starts with a pipe\n",
    want_problems=False)

run("a delimiter-less table far below an unrelated one is not attributed to it",
    "| # | a |\n|---|---|\n| 1 | x |\n\n" + "prose line\n" * 20 + "\n| hand | written |\n| 1 | 2 |\n",
    want_problems=False)

# N2: the 4-space rule is load-bearing -- reverting it leaves a false positive on an indented code
# block, and without this case the suite stays green while that happens.
run("a TAB-indented code block after a table is not a fragment",
    "| # | a |\n|---|---|\n| 1 | x |\n\n\t| tabbed | code |\n\t| block | here |\n",
    want_problems=False)

run("a 4-space indented code block after a table is not a fragment",
    "| # | a |\n|---|---|\n| 1 | x |\n\n    | indented | code |\n    | block | here |\n",
    want_problems=False)


# Two properly delimited tables in one file are not a split table. This was a false positive in
# the ALWAYS-ON structure class, so it would have fired on correct documents.
run("two separate tables separated by prose",
    "| # | a |\n|---|---|\n| 1 | x |\n\nsome prose\n\n| # | b |\n|---|---|\n| 1 | y |\n",
    want_problems=False)

run("two separate tables separated only by a blank line",
    "| # | a |\n|---|---|\n| 1 | x |\n\n| # | b |\n|---|---|\n| 1 | y |\n",
    want_problems=False)

# The docs paste tool output containing pipes; the defect example in this very PR is fenced.
# Discriminating: the fenced run has no delimiter, so without fence handling it reads as a
# fragment continuing the table above and reports a break.
run("table-like lines inside a fenced block",
    TABLE + "\n```\n| 33 | 34 | 35 | 32 |\n| 9 | more output |\n```\n",
    want_problems=False)

# #1709: a fence opening IMMEDIATELY after a table — no blank line between. Before the
# fence-gap rule this reported a split, because the fence delimiter is a non-blank line
# between two pipe runs, which is exactly the lost-leading-pipe shape. Discriminating:
# reverting the rule makes this case, and only this case, fail.
run("a fenced block between a table and more rows is not a split",
    "| # | What |\n|---|---|\n| 1 | a |\n```\nsome output\n```\n| 2 | b |\n",
    want_problems=False)

run("fenced block between two tables",
    "| # | a |\n|---|---|\n| 1 | x |\n\n```\n| 9 | z |\n```\n\n| # | b |\n|---|---|\n| 1 | y |\n",
    want_problems=False)

# Frame indices, draw ordinals and submit numbers legitimately repeat and skip. Applying the
# sequence rules to these reported seven correct documents as broken.
run("non-sequential numbers pass structure",
    "| frame | note |\n|---|---|\n| 221 | a |\n| 349 | b |\n| 221 | c |\n",
    want_problems=False)

# A genuine numbered work list in this repo starts at 0, so "gapless from 1" is wrong.
run("sequence starting at zero",
    "| # | What |\n|---|---|\n| 0 | a |\n| 1 | b |\n| 2 | c |\n",
    ordered=True, want_problems=False)

run("--table-header selects among several numbered tables",
    "| # | Instrument |\n|---|---|\n| 1 | x |\n| 2 | y |\n\ntext\n\n| # | Other |\n|---|---|\n| 7 | z |\n",
    ordered=True, header="Instrument", want_problems=False)

run("clean ordered table", TABLE, ordered=True, want_problems=False)
run("indented table rows", "  | # | What |\n  |---|---|\n  | 1 | a |\n  | 2 | b |\n",
    ordered=True, want_problems=False)

print("command line (subprocess -- these live in main(), not in check()):")


def cli(name: str, args: list[str], *, want_rc: int, expect_text: str | None = None) -> None:
    """Run the checker as a process. main()'s argument handling has its own failure modes."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "case.md"
        path.write_text(TABLE, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(CHECKER)] + [a.replace("<FILE>", str(path)) for a in args],
            capture_output=True, text=True,
        )
    if proc.returncode != want_rc:
        FAILURES.append(
            f"{name}: expected rc={want_rc}, got {proc.returncode}. "
            f"stdout={proc.stdout[:200]!r} stderr={proc.stderr[:300]!r}"
        )
        return
    if expect_text and expect_text not in proc.stdout + proc.stderr:
        FAILURES.append(f"{name}: expected {expect_text!r} in output, got {proc.stderr[:300]!r}")
        return
    print(f"  ok  {name}")


# --sequential must FAIL, not be quietly accepted as an alias for --ordered. The distinction is the
# whole point: a caller who still passes it believes gaplessness is being enforced, and silently
# giving them the weaker check is exactly the "a gate's silence is not evidence about the class it
# no longer inspects" failure this file's own KNOWN LIMIT section is about. argparse.error exits 2.
cli("--sequential is an error naming its replacement",
    ["--sequential", "--table-header", "What", "<FILE>"],
    want_rc=2, expect_text="--ordered")

# ...and the message must be actionable rather than merely "unrecognized arguments".
cli("--sequential names the issue that removed it",
    ["--sequential", "<FILE>"], want_rc=2, expect_text="#2089")

cli("--ordered on a clean table exits 0", ["--ordered", "<FILE>"], want_rc=0)

# The success line reports the numbered range and any unused numbers, so an allocator can read the
# high-water mark from the gate's own output instead of grepping the file.
cli("the success line reports the numbered range",
    ["--ordered", "<FILE>"], want_rc=0, expect_text="numbered column 1..3")

# --baseline without --ordered is rejected rather than ignored: silently accepting it would mean a
# CI step could ask for the deletion check, get no check, and stay green.
cli("--baseline without --ordered is rejected",
    ["--baseline", "<FILE>", "<FILE>"], want_rc=2, expect_text="--baseline only applies")


# The success line must quote the BASELINE's row count, and this is not cosmetic. "no row has been
# deleted" prints identically whether 187 rows were compared or the subject was compared against
# ITSELF -- which is precisely what CI would do if a merge ref's parent order were the other way
# round and `HEAD^1` resolved to the head. Quoting the count makes the green run falsifiable: on a
# PR appending one row it must read one LESS than the subject's, and a reader who sees the two agree
# knows the comparison was void. Verified on the real CI run for this change: 188 rows in the
# subject, 187 in the baseline.
def cli_baseline(name: str, subject: str, baseline: str, *, expect_text: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        s, b = Path(d) / "case.md", Path(d) / "base.md"
        s.write_text(subject, encoding="utf-8")
        b.write_text(baseline, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(CHECKER), "--ordered", "--baseline", str(b), str(s)],
            capture_output=True, text=True,
        )
    if proc.returncode != 0:
        FAILURES.append(f"{name}: expected rc=0, got {proc.returncode}: {proc.stderr[:200]!r}")
    elif expect_text not in proc.stdout:
        FAILURES.append(f"{name}: expected {expect_text!r}, got {proc.stdout[:300]!r}")
    else:
        print(f"  ok  {name}")


cli_baseline("the success line quotes the BASELINE's row count, not the subject's",
             TABLE + "| 4 | d |\n", TABLE,
             expect_text="none of the 3 rows in")

# One baseline cannot describe several subjects.
cli("--baseline with several subjects is rejected",
    ["--ordered", "--baseline", "<FILE>", "<FILE>", "<FILE>"],
    want_rc=2, expect_text="exactly one subject")


# STDOUT specifically, and the distinction is the whole point of these arms. cli() above searches
# `stdout + stderr`, which cannot see the defect #2675 is about: this gate is quoted in CLAUDE.md as
# a copy-pasteable recipe, recipes get pasted into pipelines, and `check ... | tail` keeps stdout
# while discarding both stderr and the exit status. Every failure used to be written to stderr while
# only SUCCESS went to stdout, so a piped run printed a green summary or -- on a real failure, and
# on a refused flag -- nothing at all. Measured on master before this change: a document with two
# out-of-order rows piped through `| tail -3` produced ZERO bytes on stdout and a pipeline status
# of 0.
def cli_stdout(name: str, args: list[str], body: str, *, want_rc: int,
               present: str | None = None, absent: str | None = None) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "case.md"
        path.write_text(body, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(CHECKER)] + [a.replace("<FILE>", str(path)) for a in args],
            capture_output=True, text=True,
        )
    if proc.returncode != want_rc:
        FAILURES.append(f"{name}: expected rc={want_rc}, got {proc.returncode}. "
                        f"stdout={proc.stdout[:200]!r} stderr={proc.stderr[:200]!r}")
        return
    if present is not None and present not in proc.stdout:
        FAILURES.append(f"{name}: expected {present!r} on STDOUT, got {proc.stdout[:300]!r}")
        return
    if absent is not None and absent in proc.stdout:
        FAILURES.append(f"{name}: {absent!r} must NOT be on STDOUT, got {proc.stdout[:300]!r}")
        return
    print(f"  ok  {name}")


OUT_OF_ORDER = "| # | What |\n|---|---|\n| 3 | c |\n| 1 | a |\n| 2 | b |\n"

# The pair. The must-not-match half is what stops the positive from being satisfied by a marker
# printed unconditionally -- which would make every piped run look like a failure and get the line
# deleted rather than heeded.
cli_stdout("a real failure announces itself on stdout", ["--ordered", "<FILE>"], OUT_OF_ORDER,
           want_rc=1, present="problem(s) found")
cli_stdout("a clean run does NOT claim problems on stdout", ["--ordered", "<FILE>"], TABLE,
           want_rc=0, absent="problem(s) found")

# A refused flag is the sharper case: argparse writes usage to stderr and exits 2, so a stale recipe
# pasted into a pipeline was silent AND green. It must name itself on the stream the caller kept.
cli_stdout("a removed flag says on stdout that the check did not run",
           ["--sequential", "<FILE>"], TABLE, want_rc=2, present="THE CHECK DID NOT RUN")
cli_stdout("an unknown flag does too -- not only the one flag we happen to remember",
           ["--gapless", "<FILE>"], TABLE, want_rc=2, present="THE CHECK DID NOT RUN")
cli_stdout("a successful run does not claim it was refused",
           ["--ordered", "<FILE>"], TABLE, want_rc=0, absent="THE CHECK DID NOT RUN")

# --baseline takes a file path. Handed a git ref it fails with "cannot read", which reads as a
# problem with the DOCUMENT rather than with the invocation -- several briefings propagated the ref
# form (#2675). The message must name the flag's actual contract.
cli("--baseline handed a git ref names the contract, not just 'cannot read'",
    ["--ordered", "--baseline", "origin/master", "<FILE>"],
    want_rc=1, expect_text="takes a FILE PATH, not a git ref")


# ...and the hint must NOT appear for a baseline that EXISTS and is merely unusable, or it becomes
# noise on the one path where it is actively wrong -- pointing the reader at their invocation when
# the invocation was right and the FILE is the problem. Asserting the absence needs its own arm:
# every helper above searches for presence, and a message that always carries the hint satisfies
# the positive case above just as well as a correct one does.
#
# The baseline here is INVALID UTF-8, and that specific choice is what makes the arm able to fail.
# The first draft used a baseline containing prose -- which exists AND reads cleanly, so it never
# reaches the guarded branch at all: with the guard mutated to fire unconditionally the suite still
# passed, i.e. the arm was testing nothing. Undecodable bytes are the reachable shape that both
# exists and errors, and that is the only shape that separates "hint when the path is missing" from
# "hint on every --baseline failure".
def baseline_hint_absent(name: str, baseline_bytes: bytes) -> None:
    with tempfile.TemporaryDirectory() as d:
        s, b = Path(d) / "case.md", Path(d) / "base.md"
        s.write_text(TABLE, encoding="utf-8")
        b.write_bytes(baseline_bytes)
        problems = check(s, True, None, b)
    if not problems:
        FAILURES.append(f"{name}: expected a --baseline problem, got clean")
    elif any("not a git ref" in p for p in problems):
        FAILURES.append(f"{name}: the git-ref hint fired on a baseline that exists: {problems}")
    else:
        print(f"  ok  {name}")


baseline_hint_absent("an unreadable baseline that EXISTS gets no git-ref hint", b"\xff\xfe\x00bad")
# A SECOND, deliberately weaker arm: this baseline reads cleanly, so it never reaches the guarded
# branch and cannot fail mutation D. It is kept only to pin the no-numbered-table error path, and is
# labelled so nobody later reads it as the arm that proves the guard.
baseline_hint_absent("a baseline that exists but has no numbered table gets no git-ref hint",
                     b"Not a table, just prose.\n")


# A CLOSED STDOUT, which is what `check ... | head -1` leaves behind the moment head has its line.
# Routing the verdict to stdout is only an improvement if the process still exits with a status that
# MEANS something: an unguarded `print` raises BrokenPipeError, and the well-defined 1 or 2 becomes
# **120 plus a traceback**, which is a worse answer than the silence it replaced. Measured on master:
# the SUCCESS path already exited 120 this way while both failure paths exited cleanly, so without
# `say()` this PR would have extended the crash to exactly the paths it set out to make visible.
def cli_closed_stdout(name: str, args: list[str], body: str, *, want_rc: int) -> None:
    import os
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "case.md"
        path.write_text(body, encoding="utf-8")
        read_fd, write_fd = os.pipe()
        os.close(read_fd)                     # the reader is gone before the child writes a byte
        try:
            proc = subprocess.run(
                [sys.executable, str(CHECKER)] + [a.replace("<FILE>", str(path)) for a in args],
                stdout=write_fd, stderr=subprocess.DEVNULL,
            )
        finally:
            os.close(write_fd)
    if proc.returncode != want_rc:
        FAILURES.append(f"{name}: expected rc={want_rc} with stdout closed, got {proc.returncode}"
                        + (" (BrokenPipeError -- the status no longer names the outcome)"
                           if proc.returncode == 120 else ""))
        return
    print(f"  ok  {name}")


OUT_OF_ORDER_2 = "| # | What |\n|---|---|\n| 3 | c |\n| 1 | a |\n"
cli_closed_stdout("a clean run still exits 0 with stdout closed",
                  ["--ordered", "<FILE>"], TABLE, want_rc=0)
cli_closed_stdout("a real failure still exits 1 with stdout closed",
                  ["--ordered", "<FILE>"], OUT_OF_ORDER_2, want_rc=1)
cli_closed_stdout("a removed flag still exits 2 with stdout closed",
                  ["--sequential", "<FILE>"], TABLE, want_rc=2)
cli_closed_stdout("an unknown flag still exits 2 with stdout closed",
                  ["--gapless", "<FILE>"], TABLE, want_rc=2)


# The OTHER way a stdout write fails, and the one that must NOT be silent. A departed reader is
# routine and deserves no noise; a write that fails for any other reason means the summary was lost,
# and "could not say it" is otherwise indistinguishable from "there was nothing to say". `/dev/full`
# is the reachable instance (ENOSPC on every write). This also pins the deliberately BROAD
# `except (BrokenPipeError, OSError)`: narrowed to BrokenPipeError these two exit 120 again.
def cli_full_device(name: str, args: list[str], body: str, *, want_rc: int) -> None:
    if not os.path.exists("/dev/full"):
        # Announced AND counted. ctest captures stdout and shows it only on failure, so a printed
        # SKIP is invisible on a green Windows run -- the tail count below is what makes the
        # difference between "91 passed" and "89 passed, 2 skipped" legible in either place.
        SKIPPED.append(name)
        print(f"  SKIP {name} (no /dev/full on this platform)")
        return
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "case.md"
        path.write_text(body, encoding="utf-8")
        with open("/dev/full", "w") as full:
            proc = subprocess.run(
                [sys.executable, str(CHECKER)] + [a.replace("<FILE>", str(path)) for a in args],
                stdout=full, stderr=subprocess.PIPE, text=True,
            )
    if proc.returncode != want_rc:
        FAILURES.append(f"{name}: expected rc={want_rc} on a full device, got {proc.returncode}")
    elif "stdout could not be written" not in proc.stderr:
        FAILURES.append(f"{name}: a lost summary must say so on stderr, got {proc.stderr[:200]!r}")
    else:
        print(f"  ok  {name}")


cli_full_device("a lost summary says so on stderr and keeps rc=0",
                ["--ordered", "<FILE>"], TABLE, want_rc=0)
cli_full_device("a lost summary says so on stderr and keeps rc=1",
                ["--ordered", "<FILE>"], OUT_OF_ORDER_2, want_rc=1)


# The warning about a lost summary must not itself become the crash, and there are TWO ways it can
# be, which is why there are two helpers below rather than one.
#
#   BOTH STREAMS UNWRITABLE -- `check ... > log 2>&1` on a full or over-quota filesystem, which this
#   repository's own guidance warns recurs here. `print(..., file=sys.stderr)` BUFFERS: a failed
#   write raises nothing at the call, then raises at interpreter shutdown after main() returned the
#   right status -> rc=120.
#
#   FD 2 CLOSED -- `check ... > /dev/full 2>&-`. CPython sets `sys.stderr` to None, so
#   `sys.stderr.fileno()` raises AttributeError, which is not an OSError: it escapes say()'s except
#   clause, the dup2 never runs, and the poisoned stdout raises at shutdown -> rc=120.
#
# Both shipped, in successive drafts of the same one line (#2696 review rounds 1 and 2), and the
# second was opened by the fix for the first. `os.write(2, ...)` closes both: no buffer, and no
# attribute to be None. Neither arm is a substitute for the other -- draft 1 passes the fd-2-closed
# case by accident (`print` to a None file is a no-op) and draft 2 passes the both-full case.
def cli_both_streams_full(name: str, args: list[str], body: str, *, want_rc: int) -> None:
    if not os.path.exists("/dev/full"):
        SKIPPED.append(name)
        print(f"  SKIP {name} (no /dev/full on this platform)")
        return
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "case.md"
        path.write_text(body, encoding="utf-8")
        with open("/dev/full", "w") as full:
            proc = subprocess.run(
                [sys.executable, str(CHECKER)] + [a.replace("<FILE>", str(path)) for a in args],
                stdout=full, stderr=full,
            )
    if proc.returncode != want_rc:
        FAILURES.append(f"{name}: expected rc={want_rc} with BOTH streams unwritable, got "
                        f"{proc.returncode}"
                        + (" -- the stderr warning buffered and raised at shutdown"
                           if proc.returncode == 120 else ""))
        return
    print(f"  ok  {name}")


cli_both_streams_full("a clean run survives BOTH streams being unwritable",
                      ["--ordered", "<FILE>"], TABLE, want_rc=0)


def cli_stderr_closed(name: str, args: list[str], body: str, *, stdout_full: bool,
                      want_rc: int) -> None:
    """fd 2 CLOSED, not redirected. `stdout_full` is the control switch, and it is what makes the
    arm test the INTERACTION rather than merely "stderr was closed": with stdout healthy nothing
    reaches say()'s except branch at all, so that case is 0 on every draft and on master."""
    if stdout_full and not os.path.exists("/dev/full"):
        SKIPPED.append(name)
        print(f"  SKIP {name} (no /dev/full on this platform)")
        return
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "case.md"
        path.write_text(body, encoding="utf-8")
        out = open("/dev/full", "w") if stdout_full else subprocess.DEVNULL
        try:
            proc = subprocess.run(
                [sys.executable, str(CHECKER)] + [a.replace("<FILE>", str(path)) for a in args],
                stdout=out, preexec_fn=(lambda: os.close(2)) if hasattr(os, "fork") else None,
            )
        finally:
            if stdout_full:
                out.close()
    if proc.returncode != want_rc:
        FAILURES.append(f"{name}: expected rc={want_rc} with fd 2 closed, got {proc.returncode}"
                        + (" -- AttributeError from sys.stderr being None escaped say()"
                           if proc.returncode == 120 else ""))
        return
    print(f"  ok  {name}")


cli_stderr_closed("a lost summary with fd 2 CLOSED still exits 0",
                  ["--ordered", "<FILE>"], TABLE, stdout_full=True, want_rc=0)
# The control. Passes on every draft and on master, so on its own it proves nothing -- it is here to
# show the arm above fails for the INTERACTION and not merely for having closed fd 2.
cli_stderr_closed("fd 2 closed with a healthy stdout is unremarkable",
                  ["--ordered", "<FILE>"], TABLE, stdout_full=False, want_rc=0)

print()
if FAILURES:
    for f in FAILURES:
        print(f"FAIL: {f}", file=sys.stderr)
    print(f"\n{len(FAILURES)} case(s) failed.", file=sys.stderr)
    sys.exit(1)
# A count, not just a verdict: "all cases passed" reads identically whether every arm ran or half of
# them skipped, and two of these arms need /dev/full, which Windows does not have.
print(f"all cases passed ({len(SKIPPED)} skipped"
      + (": " + ", ".join(SKIPPED) if SKIPPED else "") + ")")
