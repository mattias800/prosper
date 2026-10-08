#!/usr/bin/env python3
"""Which wave-width reason sets ADR 0028's route-2 rewrites could even apply to.

Route 2 of ADR 0028 grows the proven width-independent route with three exact rewrites: a
`readlane`/`readfirstlane` of a wave-uniform value (a move), a ballot that only feeds a count or a
compaction slot (a workgroup-scope prefix or reduction), and a vote over a wave-uniform predicate
(that predicate, whoever consumes it, including a scalar consumer). This module sorts the reason names `shader_inspect --wave-reasons` already
reports (see wave_reason_census.py) into those three CANDIDATE classes, and names what is left.

It is a candidate classifier, an UPPER bound, and it admits nothing. Each rewrite needs a uniformity
or order-insensitivity proof about the program's values, and a reason bit says only that the
operation is present, not that its operand is uniform or its consumer insensitive. A shader listed
here as `candidate` may still be refused by the proof; a shader listed as `other-route` cannot be
rescued by these three rewrites at all. It reuses the reason set the existing decoder produced; it
does not decode instructions.

    python wave_reason_census.py <shader_inspect> <dir> --csv census.csv
    python route_candidates.py census.csv
"""

from __future__ import annotations

import collections
import csv
from collections.abc import Iterable
from pathlib import Path

# reason name (shader_inspect's kWaveReasonNames) -> the route-2 rewrite class it is a candidate for.
CANDIDATE_CLASS = {
    "readlane64": "uniform-readlane",
    "wave-any": "uniform-vote",
    "wave-ballot": "compaction-ballot",
}
# A lane id (MBCNT) is the other half of a compaction slot allocation, so it is covered only when a
# ballot is present: on its own it is a lane-dependent value no rewrite here makes independent.
PAIRED_WITH_BALLOT = "lane-id"
# `scalar-reduce` is not a separate operation: the recompiler reports it for a WaveAny result that
# reaches a guest scalar-data consumer (rdna2_to_spirv.hpp, the reason's definition). The uniform-vote
# rewrite, "a vote over a wave-uniform predicate is that predicate", is exact whoever consumes the
# vote, so it is covered when a wave-any is present and is another route's problem on its own.
PAIRED_WITH_ANY = "scalar-reduce"

OTHER_ROUTE = "other-route"
CANDIDATE = "candidate"
NONE = "none"


def parse_names(names: str) -> list[str]:
    """`names=` field of a wave-reasons row: comma separated, or `none`/`absent`/empty."""
    names = (names or "").strip()
    if names in ("", "none", "absent"):
        return []
    return [n for n in names.split(",") if n]


def classify(names: Iterable[str]) -> tuple[str, list[str], list[str]]:
    """(verdict, candidate classes, reasons no route-2 rewrite covers) for one reason set.

    verdict is `none` for an empty set (nothing to rewrite), `candidate` when EVERY reason has a
    route-2 candidate rewrite, else `other-route`. Unknown reason bits are uncovered by design: a
    header change must make a shader fall out of `candidate`, not into it.
    """
    present = list(names)
    if not present:
        return NONE, [], []
    classes = sorted({CANDIDATE_CLASS[n] for n in present if n in CANDIDATE_CLASS})
    uncovered = []
    for n in present:
        if n in CANDIDATE_CLASS:
            continue
        if n == PAIRED_WITH_BALLOT and "wave-ballot" in present:
            continue
        if n == PAIRED_WITH_ANY and "wave-any" in present:
            continue
        uncovered.append(n)
    return (CANDIDATE if not uncovered else OTHER_ROUTE), classes, sorted(uncovered)


def tally(
    name_rows: Iterable[str],
) -> tuple[collections.Counter, collections.Counter, collections.Counter]:
    """Over `names=` strings of wide-wave shaders: verdict counts, candidate-class counts (one per
    class a candidate shader uses), and uncovered-reason counts (what keeps the rest out)."""
    verdicts: collections.Counter = collections.Counter()
    classes: collections.Counter = collections.Counter()
    blockers: collections.Counter = collections.Counter()
    for row in name_rows:
        verdict, cls, uncovered = classify(parse_names(row))
        verdicts[verdict] += 1
        if verdict == CANDIDATE:
            classes.update(cls)
        blockers.update(uncovered)
    return verdicts, classes, blockers


def report(name_rows: Iterable[str]) -> str:
    verdicts, classes, blockers = tally(list(name_rows))
    total = sum(verdicts.values())
    lines = [
        f"route-2 rewrite candidates among the {total} wide-wave shaders analysed:",
        f"  candidate (every reason has a rewrite class) = {verdicts[CANDIDATE]}",
        f"  other-route (some reason has none)           = {verdicts[OTHER_ROUTE]}",
        f"  none (no reason recorded)                    = {verdicts[NONE]}",
    ]
    for cls, n in sorted(classes.items()):
        lines.append(f"    candidate uses {cls:<18} {n:>4}")
    for reason, n in blockers.most_common():
        lines.append(f"    uncovered reason {reason:<16} {n:>4}")
    lines.append(
        "NOTE: an UPPER bound that admits nothing. Each rewrite still needs its uniformity or "
        "order-insensitivity proof, and a shader that could not be lowered without a resource "
        "table is in neither count."
    )
    return "\n".join(lines)


def resolve_census_csv(arg: str) -> Path:
    """The census file a command-line argument names: it must exist, be a regular file and be a
    `.csv`. Only the argument is ever turned into a path; nothing read from the file is."""
    path = Path(arg).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError(f"not a census CSV file: {arg}")
    return path


def names_from_census_csv(path: str) -> list[str]:
    """`names` of the wide-wave rows of a `wave_reason_census.py --csv` file: the shaders that were
    analysed (recompiled) and require more than 32 lanes, the same population the census reports."""
    with resolve_census_csv(path).open(newline="", encoding="utf-8") as fh:
        return [
            row["names"]
            for row in csv.DictReader(fh)
            if not row["error"]
            and int(row["recompiled"] or 0)
            and int(row["required_subgroup_size"] or 0) > 32
        ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("census_csv", help="output of wave_reason_census.py --csv")
    args = ap.parse_args(argv)
    try:
        names = names_from_census_csv(args.census_csv)
    except (OSError, ValueError) as err:
        print(f"cannot read the census: {err}")
        return 2
    if not names:
        print("no analysed wide-wave shaders in the census; nothing to classify")
        return 2
    print(report(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
