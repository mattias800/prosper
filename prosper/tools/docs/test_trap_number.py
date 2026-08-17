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

import sys
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
