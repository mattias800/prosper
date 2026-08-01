#!/usr/bin/env python3
"""Self-checking tests for check_numbered_table.py (exit code is truth).

Every case here is a defect this repository actually produced, or a correct shape that an
earlier revision of the checker wrongly rejected. The second group matters as much as the
first: a check that fires on correct data gets deleted rather than heeded, so the false-positive
cases are the ones that keep it alive.

Run directly, or via ctest as doc_table_checker.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_numbered_table import check  # noqa: E402

FAILURES: list[str] = []


def run(name: str, body: str, *, sequential: bool = False, header: str | None = None,
        want_problems: bool, expect_text: str | None = None) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "case.md"
        path.write_text(body, encoding="utf-8")
        problems = check(path, sequential, header)
    got = bool(problems)
    if got != want_problems:
        FAILURES.append(
            f"{name}: expected {'problems' if want_problems else 'clean'}, got "
            f"{problems if problems else 'clean'}"
        )
        return
    if expect_text and not any(expect_text in p for p in problems):
        FAILURES.append(f"{name}: expected a message containing {expect_text!r}, got {problems}")
        return
    print(f"  ok  {name}")


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
run("two blank lines split the same table",
    "| # | What |\n|---|---|\n| 1 | a |\n\n| 2 | b |\n\n| 3 | c |\n",
    want_problems=True, expect_text="blank line splits the table")

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

# Two properly delimited tables in one file are not a split table. This was a false positive in
# the ALWAYS-ON structure class, so it would have fired on correct documents.
run("two separate tables separated by prose",
    "| # | a |\n|---|---|\n| 1 | x |\n\nsome prose\n\n| # | b |\n|---|---|\n| 1 | y |\n",
    want_problems=False)

run("two separate tables separated only by a blank line",
    "| # | a |\n|---|---|\n| 1 | x |\n\n| # | b |\n|---|---|\n| 1 | y |\n",
    want_problems=False)

# The docs paste tool output containing pipes; the defect example in this very PR is fenced.
run("table-like lines inside a fenced block",
    TABLE + "\nprose\n\n```\n| 33 | 34 | 35 | 32 |\nsome output\n```\n",
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
