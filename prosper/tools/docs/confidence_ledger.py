#!/usr/bin/env python3
"""List every `CONFIDENCE: HIGH|MED|LOW` marker in the shipping code: the unverified-claims ledger.

The charter asks that genuinely uncertain code be marked `CONFIDENCE: HIGH/MED/LOW`. The markers are
written one at a time and never collected, so nobody can answer "what does prosper currently
believe on thin evidence?" without grepping and reading each hit. PortPS5 keeps the equivalent list
by hand in its technical-debt ledger (NIDs whose names or signatures are unconfirmed); this
generates it from the markers themselves, so it cannot fall out of date.

It is a REPORT, not a gate: a LOW marker is honest, and failing a build for one would teach people
to stop writing them. Each line is `path:line LEVEL  text`, where text is the rest of the marker's
comment line. `--level` filters (default LOW and MED), `--summary` prints counts per top-level area
instead, `--json` emits records.

Only C/C++ sources under prosper/src and prosper/frontends are read; documentation is not code
and carries its own `CONFIDENCE` statements in context. A marker must be written as
`CONFIDENCE: LEVEL` (colon, one space or more, upper-case level); anything else is not counted,
so a typo is missing from the ledger rather than misfiled in it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOTS = (Path("prosper/src"), Path("prosper/frontends"))
SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".inc", ".m", ".mm"}
LEVELS = ("HIGH", "MED", "LOW")
MARKER_RE = re.compile(r"CONFIDENCE:\s+(HIGH|MED|LOW)\b[\s.,:;)-]*(.*)")


@dataclass(frozen=True)
class Marker:
    path: str
    line: int
    level: str
    text: str


def scan(root: Path) -> list[Marker]:
    markers: list[Marker] = []
    for base in ROOTS:
        top = root / base
        if not top.is_dir():
            raise SystemExit(f"confidence_ledger: {base} not found under {root}")
        for path in sorted(top.rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            for n, line in enumerate(text.splitlines(), 1):
                for m in MARKER_RE.finditer(line):
                    note = m.group(2).strip().rstrip("*/").strip()
                    markers.append(Marker(rel, n, m.group(1), note))
    return markers


def area(path: str) -> str:
    """`prosper/src/gpu/x/y.cpp` -> `src/gpu`."""
    parts = path.split("/")
    return "/".join(parts[1:3]) if len(parts) > 3 else "/".join(parts[1:-1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."), help="repository root (default: .)")
    ap.add_argument(
        "--level",
        action="append",
        choices=LEVELS,
        help="level to list; repeatable (default: LOW and MED)",
    )
    output = ap.add_mutually_exclusive_group()
    output.add_argument("--summary", action="store_true", help="counts per area and level")
    output.add_argument("--json", action="store_true", help="emit JSON records")
    args = ap.parse_args(argv)
    levels = set(args.level or ("LOW", "MED"))
    markers = [m for m in scan(args.root.resolve()) if m.level in levels]
    if args.json:
        print(json.dumps([asdict(m) for m in markers], indent=1))
    elif args.summary:
        table: dict[str, dict[str, int]] = {}
        for m in markers:
            table.setdefault(area(m.path), {}).setdefault(m.level, 0)
            table[area(m.path)][m.level] += 1
        for name in sorted(table):
            counts = "  ".join(
                f"{lvl}={table[name].get(lvl, 0)}" for lvl in LEVELS if lvl in levels
            )
            print(f"{name:28} {counts}")
    else:
        for m in markers:
            print(f"{m.path}:{m.line} {m.level:4}  {m.text}")
    print(f"confidence_ledger: {len(markers)} marker(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
