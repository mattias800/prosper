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

TWO CLASSES OF CHECK, deliberately separated, because conflating them is a mistake this file
was written with and corrected:

  STRUCTURE (always on) -- no blank or non-table line may interrupt a table. This is a fact
  about Markdown and holds for every table in the repository.

  SEQUENCE (--sequential, opt-in) -- the first column is a unique, ascending, gapless sequence
  from 1. This is a CONVENTION of the trap table ("append, never renumber"), NOT a property of
  numbered tables in general. Most numbered tables here lead with frame indices, draw ordinals
  or submit numbers, where gaps and repeats are the correct content. Applying the sequence
  rules to those would report hundreds of failures that are not defects.

Validates structure, never content, so neither class can rot as a table grows.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROW = re.compile(r"^\|\s*(\d+)\s*\|")
SEPARATOR = re.compile(r"^\|[\s:|-]+\|?\s*$")


def check(path: Path, sequential: bool) -> list[str]:
    """Return a list of problems; empty means the table is sound."""
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError as exc:
        return [f"{path}: cannot read: {exc}"]

    numbered = [(i + 1, int(m.group(1))) for i, line in enumerate(lines) if (m := ROW.match(line))]
    if not numbered:
        # Not "the table is fine" -- the file we were pointed at has no numbered table at all,
        # so the path is wrong or the format changed under us. Failing here is what stops this
        # check from passing forever on a file it can no longer see.
        return [f"{path}: no numbered table rows found -- wrong path, or the table format changed"]

    problems: list[str] = []

    for (prev_line, prev_num), (this_line, this_num) in zip(numbered, numbered[1:]):
        for k in range(prev_line, this_line - 1):
            text = lines[k]
            if not text.strip():
                problems.append(
                    f"{path}:{k + 1}: blank line between rows {prev_num} and {this_num} terminates "
                    f"the Markdown table -- row {this_num} onward renders as a separate table. "
                    f"Delete the blank line."
                )
            elif not text.startswith("|") and not SEPARATOR.match(text):
                problems.append(
                    f"{path}:{k + 1}: non-table line between rows {prev_num} and {this_num} "
                    f"breaks the table: {text[:60]!r}"
                )

    if not sequential:
        return problems

    seen: dict[int, int] = {}
    highest = 0
    for line_no, num in numbered:
        if num in seen:
            problems.append(
                f"{path}:{line_no}: duplicate row number {num} (first used at line {seen[num]}). "
                f"Two branches almost certainly appended the same number concurrently -- such a "
                f"merge is textually clean. Renumber the LATER row to {max(seen) + 1} or higher, "
                f"and re-check every by-number citation of it elsewhere in the docs."
            )
            continue
        seen[num] = line_no
        if num > highest + 1:
            missing = ", ".join(str(n) for n in range(highest + 1, num))
            problems.append(f"{path}:{line_no}: row numbers skip {missing} before reaching {num}")
        elif num <= highest:
            problems.append(
                f"{path}:{line_no}: row number {num} is out of ascending order "
                f"(previous row was {highest})"
            )
        highest = max(highest, num)

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="Markdown files containing a numbered table")
    ap.add_argument(
        "--sequential",
        action="store_true",
        help="also require the first column to be unique, ascending and gapless from 1 "
        "(a convention of the instrument-trap table, not of numbered tables generally)",
    )
    ap.add_argument(
        "--github",
        action="store_true",
        help="emit ::error:: annotations so failures surface on the GitHub Actions summary",
    )
    args = ap.parse_args()

    problems: list[str] = []
    for path in args.paths:
        problems.extend(check(path, args.sequential))

    for problem in problems:
        print(f"::error::{problem}" if args.github else f"error: {problem}", file=sys.stderr)

    if problems:
        print(f"\n{len(problems)} problem(s) found.", file=sys.stderr)
        return 1

    mode = "contiguous and unbroken" if args.sequential else "unbroken"
    for path in args.paths:
        rows = sum(1 for line in path.read_text(encoding="utf-8").split("\n") if ROW.match(line))
        print(f"{path}: {rows} rows, {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
