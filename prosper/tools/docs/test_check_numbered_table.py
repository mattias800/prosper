#!/usr/bin/env python3
"""Self-checking tests for check_numbered_table.py (exit code is truth).

Every case here is a defect this repository actually produced, or a correct shape that an
earlier revision of the checker wrongly rejected. The second group matters as much as the
first: a check that fires on correct data gets deleted rather than heeded, so the false-positive
cases are the ones that keep it alive.

Run directly, or via ctest as doc_table_checker.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_numbered_table import check  # noqa: E402

FAILURES: list[str] = []


def run(name: str, body: str, *, sequential: bool = False, header: str | None = None,
        want_problems: bool, expect_text: str | None = None, want_count: int | None = None,
        want_lines: list[int] | None = None) -> None:
    """Check one document.

    `want_count`/`want_lines` exist because asserting mere truthiness cannot see a detector that
    finds the FIRST instance of a defect and stops -- a common shape (early return, comparing only
    against the previous element) that single-instance fixtures are blind to. An earlier draft of
    this checker had exactly that bug, and a truthiness-only version of this suite stayed green
    when it was reintroduced. Any case covering more than one defect must pin the count.
    """
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "case.md"
        path.write_text(body, encoding="utf-8")
        problems = check(path, sequential, header)
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

# R1: a row that lost its leading pipe ends the table just as a blank line does, and is the more
# likely lane accident. Everything below it silently leaves the sequence check too, so a duplicate
# number underneath would pass green -- which is the defect this whole check exists to prevent.
run("a row that lost its leading pipe ends the table",
    "| # | What |\n|---|---|\n| 1 | a |\n2 | b |\n| 3 | c |\n",
    want_problems=True, want_count=1, want_lines=[4],
    expect_text="lose its leading")

# Structural errors are reported first and the broken remainder leaves the sequence check, the way
# a compiler stops elaborating a malformed declaration. What matters is that the file CANNOT pass
# green while hiding a duplicate below the break -- the previous revision reported
# "contiguous and unbroken", exit 0, on exactly this input. Fix the structure, re-run, get the
# duplicate.
run("a duplicate hidden below an interruption cannot pass green",
    "| # | What |\n|---|---|\n| 1 | a |\n2 | b |\n| 5 | c |\n| 5 | d |\n",
    sequential=True, want_problems=True, want_count=1, want_lines=[4],
    expect_text="drops out of any sequence check")

# The #1696 collision: two branches append the same number; the merge is textually clean.
run("duplicate row number",
    "| # | What |\n|---|---|\n| 32 | a |\n| 33 | b |\n| 34 | c |\n| 35 | d |\n| 32 | e |\n",
    sequential=True, want_problems=True, expect_text="duplicate row number 32")

run("gap in the sequence",
    "| # | What |\n|---|---|\n| 1 | a |\n| 4 | b |\n",
    sequential=True, want_problems=True, expect_text="skip 2, 3")

run("out of ascending order",
    "| # | What |\n|---|---|\n| 1 | a |\n| 3 | b |\n| 2 | c |\n",
    sequential=True, want_problems=True)

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

# Absence of a table is an error only when we were pointed at ONE specific table (--sequential):
# then it means the path is wrong or the format changed, and passing would let the check go green
# forever on a file it can no longer see. In a plain structure sweep it is ordinary -- 35 of this
# repo's 77 Markdown files contain no table at all, and failing on those makes the tool unusable
# for the sweep it is meant to support.
run("no table at all, --sequential", "just prose, no table\n",
    sequential=True, want_problems=True, expect_text="no Markdown tables found")
run("empty file, --sequential", "", sequential=True, want_problems=True)
run("no table at all, structure sweep", "just prose, no table\n", want_problems=False)
run("empty file, structure sweep", "", want_problems=False)
run("--sequential with no numbered table",
    "| name | value |\n|---|---|\n| a | 1 |\n",
    sequential=True, want_problems=True, expect_text="no numbered table")
run("--sequential ambiguous between two numbered tables",
    "| # | a |\n|---|---|\n| 1 | x |\n\ntext\n\n| # | b |\n|---|---|\n| 1 | y |\n",
    sequential=True, want_problems=True, expect_text="ambiguous")

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
    sequential=True, want_problems=False)

run("--table-header selects among several numbered tables",
    "| # | Instrument |\n|---|---|\n| 1 | x |\n| 2 | y |\n\ntext\n\n| # | Other |\n|---|---|\n| 7 | z |\n",
    sequential=True, header="Instrument", want_problems=False)

run("clean sequential table", TABLE, sequential=True, want_problems=False)
run("indented table rows", "  | # | What |\n  |---|---|\n  | 1 | a |\n  | 2 | b |\n",
    sequential=True, want_problems=False)

print()
if FAILURES:
    for f in FAILURES:
        print(f"FAIL: {f}", file=sys.stderr)
    print(f"\n{len(FAILURES)} case(s) failed.", file=sys.stderr)
    sys.exit(1)
print("all cases passed")
