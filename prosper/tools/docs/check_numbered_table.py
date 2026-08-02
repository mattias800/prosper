#!/usr/bin/env python3
"""Validate Markdown tables that other documents cite by row number.

Two defects landed in docs/GAME_COMPAT_ORCHESTRATION.md on 2026-08-01, and no other check
in the pipeline could see either one:

  * A stray blank line between two rows. In Markdown a blank line TERMINATES a table, so the
    rows after it render as a separate table -- or, for a single trailing row, as a one-row
    table of its own. Master carried two at once, rendering 36 rows as three tables. Nothing
    is missing from the file, so the diff looks perfect and only a reader of the rendered
    page would notice.

  * Two branches independently claiming the same row number. Both merge CLEANLY, because they
    are separate lines with no textual conflict, producing a table reading "33, 34, 35, 32"
    at mergeable=CLEAN with every job green. Conflict detection is textual and line-local; it
    has no notion that a column is a unique key. Other documents cite these rows BY NUMBER, so
    a duplicate silently redirects every such citation to an unrelated entry.

HOW A BREAK IS TOLD FROM TWO LEGITIMATE TABLES. A proper Markdown table opens with a header
row followed by a delimiter row (`|---|---|`). A fragment left behind by a stray blank line has
no delimiter -- it is a continuation, not a table. So an all-pipe run WITHOUT a delimiter that
directly follows a proper table, separated only by blank lines, is a break; two properly
delimited tables in one file are not. This also catches a blank line between the delimiter and
the first data row, which orphans the entire body.

Fenced code blocks are skipped: this repository's docs paste tool output containing pipe
characters, and this file's own defect example would otherwise fail the check that documents it.

TWO CLASSES OF CHECK, deliberately separated:

  STRUCTURE (always on) -- no blank line may split a table. A fact about Markdown, true of
  every table in the repository.

  SEQUENCE (--sequential, opt-in) -- the numbered column is unique, strictly ascending and
  gapless. This is a CONVENTION of the trap table ("append, never renumber"), NOT a property of
  numbered tables in general. Measured over the 77 tracked Markdown files: all 77 satisfy
  structure, while of the 5 documents holding an all-numeric-first-column table only 2 satisfy
  sequence -- the rest lead with frame indices, draw ordinals and submit numbers, where gaps and
  repeats are the correct content. Applying sequence everywhere would report correct documents
  as broken, and a check that fires on correct data gets deleted rather than heeded.

  Two selector details, both from real documents here rather than from theory:
    * The sequence is checked from its own first value, not from 1, because a genuine numbered
      work list in this repo starts at 0 (EVERGATE_PERFORMANCE_HANDOFF_2026_07.md).
    * A table qualifies only if EVERY body row is numbered. That same work list ends with a
      `| Separate | ...` row, which is legitimate content and must not be read as a gap.

Validates structure, never content, so neither class can rot as a table grows.

WHAT THIS CANNOT SEE, and a real incident. It validates STRUCTURE. It does not read content, so it
cannot tell that a row still says what its author meant. On 2026-08-01 a rebase silently reverted an
entry's TEXT to a version asserting a conclusion its author had already retracted, while the
numbering stayed perfectly contiguous -- this check would have passed it without a word. The
boundary is sharp and worth knowing: numbering catches a row that VANISHED (the sequence gaps), and
nothing here catches a row that CHANGED ITS MIND. A green run means the table is well formed, never
that it is true. Diff the region for content too.

A RED `Docs` JOB THAT IS THE GATE WORKING, not a defect -- read this before "fixing" it. Two
concurrent PRs appending to a gapless table create a MERGE-ORDER DEPENDENCY. Master ends at 42;
both branches append; the earlier claimant keeps 43 and the later one renumbers to 44. The later
branch's table then reads `42, 44`, and --sequential correctly rejects the gap, so ITS `Docs` job is
red BY CONSTRUCTION until the earlier PR lands -- and merging it first would put that gap on master.
Neither PR is broken. Rebase after the earlier one merges: the gap closes itself, and nothing else
about either change is involved.

The temptation at that point is to weaken the gapless rule, and that would cost the one thing
numbering can see which content checking cannot. Gaplessness is what catches a DELETED row.
Uniqueness catches a duplicate and structure catches a split, but a row that simply vanishes -- in a
bad merge, a wholesale `git checkout` of the file, a careless rebase -- leaves a file that is
perfectly well formed and silently redirects nothing, because the citations to it still resolve to
"no such row". The sequence is the only signal. So the correct response to `42, 44` is to wait, not
to relax the check. First hit on 2026-08-02 (#1711/#1722, the fourth row-number collision that day);
#1729 tracks pre-empting the collision at claim time, which is what would remove the wait.

KNOWN LIMIT, recorded as a decision rather than a defect. The abutment rule keeps the
`blank line, prose, orphaned rows` shape -- a real table split -- and that shape is structurally
identical to a correct one: a pipe-leading line in the middle of a paragraph below an unrelated
table. No rule separates those two, so that half is a choice between two errors rather than a bug to
fix -- abutment keeps the genuine defect, and the shape it misreads has zero instances across the
tracked corpus and is bounded to a single message. If it starts appearing, narrowing to "gap
contains no blank line" trades it the other way at a known cost.

A fence opening immediately after a table is misread the same way, but that half IS separable by a
rule as principled as the blank-line one -- record fence line numbers and skip a gap containing one.
It is not implemented here only to keep this change scoped after five review rounds; tracked as
#1709 so the next reader looks for the rule that exists instead of taking this block as settled.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FENCE = re.compile(r"^\s*(```|~~~)")
DELIMITER = re.compile(r"^\s*\|[\s:|-]+\|?\s*$")
LEADING_NUMBER = re.compile(r"^\s*\|\s*(\d+)\s*\|")


class Table:
    """A maximal run of consecutive table lines, with 1-based source line numbers."""

    def __init__(self, start: int, lines: list[str]) -> None:
        self.start = start
        self.lines = lines

    @property
    def proper(self) -> bool:
        """True if this opens with a header + delimiter, i.e. it is a table rather than a fragment."""
        return len(self.lines) >= 2 and bool(DELIMITER.match(self.lines[1]))

    @property
    def body(self) -> list[tuple[int, str]]:
        """(line_no, text) for rows after the header and delimiter."""
        skip = 2 if self.proper else 0
        return [(self.start + i, l) for i, l in enumerate(self.lines)][skip:]

    def numbered_rows(self) -> list[tuple[int, int]]:
        out = []
        for line_no, text in self.body:
            m = LEADING_NUMBER.match(text)
            if m:
                out.append((line_no, int(m.group(1))))
        return out

    @property
    def all_rows_numbered(self) -> bool:
        rows = self.body
        return bool(rows) and len(self.numbered_rows()) == len(rows)

    @property
    def header(self) -> str:
        return self.lines[0].strip() if self.lines else ""


def parse_tables(lines: list[str]) -> tuple[list[Table], set[int]]:
    """Split into table runs, ignoring fenced code. Returns (tables, blank_line_numbers)."""
    tables: list[Table] = []
    blanks: set[int] = set()
    fenced = False
    run: list[str] = []
    run_start = 0

    for i, line in enumerate(lines, start=1):
        if FENCE.match(line):
            fenced = not fenced
            if run:
                tables.append(Table(run_start, run))
                run = []
            continue
        if fenced:
            continue
        # 4+ columns of indentation is a code block in Markdown, not a table row. Tabs count as
        # indentation too, so expand them first: a tab-indented pipe block previously read as a
        # table fragment and drew "Delete the blank line", which would merge a code block into a
        # table. A wrong message that instructs is worse than one that merely fires.
        indent = line.expandtabs(4)
        if line.startswith("|") or (indent[:4].strip() and indent.lstrip().startswith("|")):
            if not run:
                run_start = i
            run.append(line)
            continue
        if not line.strip():
            blanks.add(i)
        if run:
            tables.append(Table(run_start, run))
            run = []
    if run:
        tables.append(Table(run_start, run))
    return tables, blanks


def check(path: Path, sequential: bool, table_header: str | None) -> list[str]:
    """Return a list of problems; empty means the tables are sound."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: cannot read: {exc}"]
    except UnicodeDecodeError as exc:
        return [f"{path}: is not valid UTF-8: {exc}"]

    lines = text.split("\n")
    tables, blanks = parse_tables(lines)
    if not tables:
        # Under --sequential we were pointed at ONE specific table, so its absence means the path
        # is wrong or the format changed, and failing here is what stops this check passing
        # forever on a file it can no longer see. In a plain structure sweep across many
        # documents, a file with no table is ordinary and must not be an error -- 35 of this
        # repo's 77 Markdown files have none.
        if sequential:
            return [f"{path}: no Markdown tables found -- wrong path, or the table format changed"]
        return []

    problems: list[str] = []

    # `origin` is the proper table the current run of fragments descends from. It must survive a
    # fragment, because a table broken twice yields fragment-after-fragment: comparing each run
    # only against the run immediately above it would report the first break and miss every later
    # one.
    #
    # What separates the runs does NOT decide whether this is a break -- only whether the run
    # below is a proper table does. A blank line and a row that lost its leading pipe both end a
    # table, and the second is the likelier lane accident. Treating a non-blank gap as "these are
    # unrelated" loses that entirely: the rows below leave the sequence check with it, so a
    # duplicate number underneath passes green, and the success line reports the truncated row
    # count as though it were the whole table.
    origin: Table | None = None
    for previous, current in zip(tables, tables[1:]):
        if previous.proper:
            origin = previous
        if current.proper or origin is None:
            continue  # a table of its own, or a fragment with no proper table above it

        # What separates the runs does not decide whether this is a break, but it does decide
        # whether they are ADJACENT -- and without that bound, backward attribution is unbounded:
        # a delimiter-less table 200 lines below an unrelated one produced 202 errors, each
        # accusing a prose line, with the real cause pushed past GitHub's 10-annotation limit.
        # Two of them told the author to delete the blank lines that correctly ended the table,
        # and wrong guidance is worse than a false positive: an agent who follows it damages the
        # file. A fragment continues the table above only if it abuts it -- the whole gap is
        # blank, or the line immediately above the fragment is itself the interruption.
        gap_start = previous.start + len(previous.lines)
        gap = range(gap_start, current.start)
        all_blank = all(n in blanks for n in gap)
        interrupted_at = current.start - 1
        if not all_blank and interrupted_at in blanks:
            continue  # separated by prose that ends in a blank line: not this table's continuation

        # One problem per fragment, not one per gap line: the fragment is the defect, and a gap of
        # N blank lines is one mistake, not N.
        if all_blank:
            problems.append(
                f"{path}:{gap_start}: blank line splits the table that starts at line "
                f"{origin.start} -- in Markdown a blank line ends a table, so line {current.start} "
                f"onward renders as a separate table. Delete the blank line."
            )
        else:
            problems.append(
                f"{path}:{interrupted_at}: line is not a table row but sits inside the table that "
                f"starts at line {origin.start}, ending it -- so line {current.start} onward "
                f"renders as a separate table and drops out of any sequence check. Did it lose "
                f"its leading '|'? Line reads: {lines[interrupted_at - 1].strip()[:60]!r}"
            )

    if not sequential:
        return problems

    candidates = [t for t in tables if t.proper and t.all_rows_numbered]
    if table_header:
        candidates = [t for t in candidates if table_header in t.header]
    if not candidates:
        return problems + [
            f"{path}: --sequential found no numbered table"
            + (f" whose header contains {table_header!r}" if table_header else "")
        ]
    if len(candidates) > 1:
        # Ambiguous rather than guessed: checking the wrong table would report correct data as
        # broken, and a check that fires on correct data gets deleted rather than heeded.
        where = ", ".join(f"line {t.start}" for t in candidates)
        return problems + [
            f"{path}: --sequential is ambiguous -- {len(candidates)} numbered tables ({where}). "
            f"Select one with --table-header."
        ]

    rows = candidates[0].numbered_rows()
    seen: dict[int, int] = {}
    expected = rows[0][1]
    for line_no, num in rows:
        if num in seen:
            problems.append(
                f"{path}:{line_no}: duplicate row number {num} (first used at line {seen[num]}). "
                f"Two branches almost certainly appended the same number concurrently -- such a "
                f"merge is textually clean. Renumber the LATER row to {max(seen) + 1} or higher, "
                f"and re-check every by-number citation of it elsewhere in the docs."
            )
            continue
        seen[num] = line_no
        if num > expected:
            missing = ", ".join(str(n) for n in range(expected, num))
            problems.append(f"{path}:{line_no}: row numbers skip {missing} before reaching {num}")
        elif num < expected:
            problems.append(
                f"{path}:{line_no}: row number {num} is out of ascending order "
                f"(previous row was {expected - 1})"
            )
        expected = max(expected, num + 1)

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="Markdown files to check")
    ap.add_argument(
        "--sequential",
        action="store_true",
        help="also require the numbered column to be unique, ascending and gapless (a convention "
        "of the instrument-trap table, not of numbered tables generally)",
    )
    ap.add_argument(
        "--table-header",
        metavar="TEXT",
        help="with --sequential, select the table whose header row contains TEXT",
    )
    ap.add_argument(
        "--github",
        action="store_true",
        help="emit ::error:: annotations so failures surface on the GitHub Actions summary",
    )
    args = ap.parse_args()
    if args.table_header and not args.sequential:
        ap.error("--table-header only applies with --sequential")

    problems: list[str] = []
    for path in args.paths:
        problems.extend(check(path, args.sequential, args.table_header))

    for problem in problems:
        if args.github:
            # GitHub decodes these three in annotation text; leaving them raw truncates messages.
            escaped = problem.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::error::{escaped}")
        else:
            print(f"error: {problem}", file=sys.stderr)

    if problems:
        print(f"\n{len(problems)} problem(s) found.", file=sys.stderr)
        return 1

    for path in args.paths:
        tables, _ = parse_tables(path.read_text(encoding="utf-8").split("\n"))
        proper = [t for t in tables if t.proper]
        rows = sum(len(t.body) for t in proper)
        detail = "contiguous and unbroken" if args.sequential else "unbroken"
        print(f"{path}: {len(proper)} table(s), {rows} rows, {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
