#!/usr/bin/env python3
"""Every `trap NNN` reference in the repository must name a row that exists.

WHY. The instrument-trap table's whole value is that other documents, source comments and tests
point AT it by number -- `// instrument trap 104 records that a read-only probe's DURATION can
decide...` in `src/gpu/command_processor.cpp`, `See instrument-trap 41` in `CLAUDE.md`,
`Recorded as instrument trap 114` in a status doc. Measured on `origin/master` 2026-08-17: **95
lines across 50 files, naming 91 distinct rows** -- and roughly half are `.cpp`/`.hpp`/test comments
rather than prose, so this is a contract compiled files depend on, not a documentation nicety.

Nothing checked that those references resolve. `check_numbered_table.py` validates the TABLE --
structure, arity, uniqueness, ascent, and (with `--baseline`) that no row was deleted. It has no
idea anything cites it. So a reference to a row that never existed, or to one whose number was
mistyped, reads as perfectly correct: it is a plausible number in a plausible sentence, and the only
way to find out is to open the table and count. Nobody does that for a number in a code comment.

That is the failure this catches, and it is the more expensive half of the numbering problem: a
duplicate row is visible the moment you look at the table, while a stale by-number reference **reads
as correct** and quietly sends the next agent to an unrelated entry -- or to nothing at all.

WHAT COUNTS AS A REFERENCE. `trap 41`, `traps 55 and 56`, `instrument trap 104`, `instrument-trap
43`, `orchestration trap 68`, and comma lists like `traps 64, 116 and 121`. The word must not be
preceded by an identifier character, which is what keeps the RDNA2 shader tests' `s_trap 1` and
`v_trap` out -- those are instruction mnemonics and have nothing to do with this table. That
exclusion is measured rather than assumed: without it the corpus reports 4 spurious references, all
in `tests/test_recompile_coverage.cpp` and its neighbours.

WHAT IT CANNOT SEE, stated so silence is not read as coverage:
  * A reference that names a row which EXISTS but is the wrong one. Only a reader can tell that; a
    number that resolves is all this can check. This is the same boundary check_numbered_table draws
    between "the table is well formed" and "the table is true".
  * A reference written in a form this does not recognise -- "the trap about rate limits", "entry
    49", a bare "#49". Deliberately narrow: a citation checker that fires on prose gets deleted.
  * References in files git does not track, and in binary files, which are skipped.
  * A row that is cited and then DELETED is reported here as a dangling reference, but the deletion
    itself belongs to `check_numbered_table.py --baseline`. The two are complements: that one asks
    "did a row leave?", this one asks "does every pointer still land?".
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_numbered_table import parse_tables  # noqa: E402

# `(?<![A-Za-z0-9_])` is what excludes `s_trap 1` / `v_trap 3`: an RDNA2 mnemonic, not a citation.
# The separator after the word may be a space or a hyphen ("instrument-trap 43" appears in CLAUDE.md).
# The tail captures comma/and lists so "traps 64, 116 and 121" yields three references, not one.
CITATION = re.compile(
    r"(?<![A-Za-z0-9_])(?:instrument[- ]|orchestration )?traps?[- ]"
    r"(\d{1,3}(?:\s*,\s*\d{1,3})*(?:\s*(?:,\s*)?and\s+\d{1,3})?)",
    re.IGNORECASE,
)
NUMBER = re.compile(r"\d{1,3}")


def table_numbers(doc: Path, table_header: str) -> set[int] | None:
    try:
        text = doc.read_text(encoding="utf-8")
    except OSError:
        return None
    tables, _, _ = parse_tables(text.split("\n"))
    matching = [t for t in tables if t.proper and t.all_rows_numbered and table_header in t.header]
    if len(matching) != 1:
        return None
    return {n for _, n in matching[0].numbered_rows()}


def citations(path: Path) -> list[tuple[int, int, str]]:
    """(line_no, number, the matched text) for every trap reference in `path`."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []  # binary or unreadable: not a citation site
    out: list[tuple[int, int, str]] = []
    for line_no, line in enumerate(text.split("\n"), start=1):
        for m in CITATION.finditer(line):
            for num in NUMBER.findall(m.group(1)):
                out.append((line_no, int(num), m.group(0).strip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="files to scan for references")
    ap.add_argument("--table", type=Path, required=True, help="the document holding the table")
    ap.add_argument("--table-header", default="Instrument",
                    help="select the table whose header contains this (default 'Instrument')")
    ap.add_argument("--github", action="store_true",
                    help="emit ::error:: annotations so failures surface on the Actions summary")
    args = ap.parse_args()

    have = table_numbers(args.table, args.table_header)
    if have is None:
        # Fails closed. If the table cannot be found, every reference would trivially "not resolve"
        # or trivially resolve depending on the bug -- either way the run means nothing, so say so
        # instead of printing a number.
        print(f"error: no single numbered table whose header contains {args.table_header!r} in "
              f"{args.table} -- wrong path, or the table format changed", file=sys.stderr)
        return 2

    problems: list[str] = []
    scanned = cited = 0
    seen_numbers: set[int] = set()
    citing_files: set[Path] = set()
    citing_lines: set[tuple[Path, int]] = set()
    for path in args.paths:
        if path.resolve() == args.table.resolve():
            continue  # the table cites itself in row text; those are not pointers into it
        scanned += 1
        for line_no, num, text in citations(path):
            cited += 1
            seen_numbers.add(num)
            citing_files.add(path)
            citing_lines.add((path, line_no))
            if num not in have:
                problems.append(
                    f"{path}:{line_no}: cites row {num}, which is not in {args.table} "
                    f"(the table runs {min(have)}..{max(have)}). A by-number reference that does "
                    f"not resolve READS AS CORRECT -- it is a plausible number in a plausible "
                    f"sentence -- so nothing else will ever report it. Either the row was deleted "
                    f"or renumbered (which `check_numbered_table.py --baseline` forbids), or this "
                    f"is a typo. Reference reads: {text!r}"
                )

    for problem in problems:
        if args.github:
            escaped = problem.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::error::{escaped}")
        else:
            print(f"error: {problem}", file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} dangling reference(s) found.", file=sys.stderr)
        return 1

    # Quote the counts, not just the exit status: a scan that matched nothing at all -- a broken
    # regex, a wrong file list -- exits 0 exactly like a clean one, and the counts are the only way
    # to tell those apart.
    print(
        f"{args.table}: {len(have)} rows ({min(have)}..{max(have)}); "
        f"{cited} reference(s) on {len(citing_lines)} line(s) in {len(citing_files)} file(s) "
        f"(of {scanned} scanned), naming {len(seen_numbers)} distinct row(s), all resolving"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
