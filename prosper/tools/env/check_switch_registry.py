#!/usr/bin/env python3
"""Every PROSPER_* switch the shipping code names is registered, and every new one is classified.

The charter asks each new `PROSPER_*` to be classified in its PR as a host-capability switch (what
this machine can do), a diagnostic (observes, changes nothing the guest sees) or a guest-behaviour
selector (changes what the guest sees, and needs an issue whose resolution settles the default and
deletes the switch). That rule had no instrument: nothing listed the switches, so nothing could
notice a new one arriving unclassified, and a selector could live forever with no issue to retire
it. This is the list, and the gate that keeps it true.

THE REGISTRY is `switch_registry.txt` beside this file, one row per switch:

    PROSPER_GFXLOG  diagnostic
    PROSPER_EXAMPLE_MODE  selector  #1234  # optional note

Classes: `host-capability`, `diagnostic`, `selector` (needs `#issue`), and `unclassified`, which
exists only for the switches that predate the registry. They are grandfathered as a ratchet: the
set of unclassified rows may shrink, never grow.

THE SCAN reads every string literal of the form "PROSPER_NAME" in prosper/src, prosper/frontends
and prosper/tests/fixtures (the shipping Vulkan backend lives there). A literal is a switch name
whether it is read through getenv, PROSPER_ENV_ON or a helper, which is the point: the reading
mechanism varies, the name does not.

CHECKS
  unregistered  A name in the tree with no row. Register it with a class.
  stale         A row whose name no longer appears. Delete the row.
  shape         An unknown class, a duplicate row, or a selector without `#NNN`.
  new-unclassified  With --base REV: an `unclassified` row that was not an unclassified row in the
                base's registry. New switches are classified when they are added.

WHAT IT CANNOT SEE: whether a class is TRUE. A selector filed as a diagnostic passes; only review
can tell. Names built at run time ("PROSPER_" + suffix) are invisible to a literal scan.

EXIT STATUS: 0 clean, 1 violation, 2 could not evaluate. `--update` rewrites the registry with
missing names added as `unclassified` and stale rows dropped, keeping every existing class; the
--base check then rejects the added rows unless they predate the registry.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EXIT_OK, EXIT_VIOLATION, EXIT_UNEVALUATED = 0, 1, 2

REGISTRY = Path("prosper/tools/env/switch_registry.txt")
SCAN_ROOTS = (Path("prosper/src"), Path("prosper/frontends"), Path("prosper/tests/fixtures"))
SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".inc"}
CLASSES = ("host-capability", "diagnostic", "selector", "unclassified")
NAME_RE = re.compile(r'"(PROSPER_[A-Z0-9_]*[A-Z0-9])"')
ROW_RE = re.compile(r"^(PROSPER_[A-Z0-9_]+)\s+(\S+)(?:\s+(#[0-9]+))?\s*(#\s.*)?$")

# Far below today's count. Tripping it means the scan is not seeing the tree (wrong --root), not
# that the switches went away; lower it in the same PR if they really do.
MIN_NAMES = 100

HEADER = """\
# PROSPER_* switch registry -- checked by check_switch_registry.py (see its docstring).
# NAME  CLASS  [#issue]  [# note]
# CLASS: host-capability | diagnostic | selector (needs #issue) | unclassified (pre-registry only)
"""


@dataclass(frozen=True)
class Row:
    name: str
    cls: str
    issue: str | None
    note: str | None
    line: int


class EvaluationError(Exception):
    """The gate could not establish an answer; exit 2, never a pass."""


def scan(root: Path, min_names: int = MIN_NAMES) -> set[str]:
    names: set[str] = set()
    for base in SCAN_ROOTS:
        top = root / base
        if not top.is_dir():
            raise EvaluationError(f"{base} not found under {root}")
        for path in top.rglob("*"):
            if path.suffix in SUFFIXES and path.is_file():
                names.update(NAME_RE.findall(path.read_text(encoding="utf-8", errors="ignore")))
    if len(names) < min_names:
        raise EvaluationError(
            f"only {len(names)} switch names found; the scan is not seeing the tree"
        )
    return names


def parse_registry(text: str) -> tuple[list[Row], list[str]]:
    rows, problems = [], []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = ROW_RE.match(line)
        if not m:
            problems.append(f"line {n}: not `NAME CLASS [#issue] [# note]`: {raw!r}")
            continue
        rows.append(Row(m.group(1), m.group(2), m.group(3), m.group(4), n))
    return rows, problems


def git_show(root: Path, rev: str, path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(root), "show", f"{rev}:{path.as_posix()}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode == 0:
        return proc.stdout
    probe = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{rev}^{{commit}}"], capture_output=True
    )
    if probe.returncode != 0:
        raise EvaluationError(f"base revision {rev} is not available")
    return None  # the base predates the registry


def evaluate(
    root: Path, base: str | None = None, min_names: int = MIN_NAMES
) -> tuple[list[str], dict[str, int]]:
    names = scan(root, min_names)
    path = root / REGISTRY
    if not path.is_file():
        raise EvaluationError(f"{REGISTRY} not found")
    rows, problems = parse_registry(path.read_text(encoding="utf-8"))
    seen: dict[str, Row] = {}
    for row in rows:
        if row.name in seen:
            problems.append(f"line {row.line}: {row.name} duplicates line {seen[row.name].line}")
        seen[row.name] = row
        if row.cls not in CLASSES:
            problems.append(f"line {row.line}: {row.name}: unknown class `{row.cls}`")
        elif row.cls == "selector" and not row.issue:
            problems.append(
                f"line {row.line}: {row.name}: a selector names the issue that retires it"
            )
    for name in sorted(names - seen.keys()):
        problems.append(f"unregistered: {name} -- add a row with its class")
    for name in sorted(seen.keys() - names):
        problems.append(
            f"stale: {name} (line {seen[name].line}) no longer appears -- delete the row"
        )
    if base:
        old = git_show(root, base, REGISTRY)
        if old is not None:
            old_rows, _ = parse_registry(old)
            grandfathered = {r.name for r in old_rows if r.cls == "unclassified"}
            for row in rows:
                if row.cls == "unclassified" and row.name not in grandfathered:
                    problems.append(
                        f"new-unclassified: {row.name} (line {row.line}) -- classify a new switch "
                        "as host-capability, diagnostic or selector #issue"
                    )
    counts = {c: sum(1 for r in seen.values() if r.cls == c) for c in CLASSES}
    counts["total"] = len(names)
    return problems, counts


def update(root: Path, min_names: int = MIN_NAMES) -> None:
    names = scan(root, min_names)
    path = root / REGISTRY
    rows, problems = (
        parse_registry(path.read_text(encoding="utf-8")) if path.is_file() else ([], [])
    )
    if problems:
        raise EvaluationError("registry does not parse; fix it before --update: " + problems[0])
    keep = {r.name: r for r in rows if r.name in names}
    lines = [HEADER]
    for name in sorted(names):
        row = keep.get(name)
        if row is None:
            lines.append(f"{name}  unclassified\n")
        else:
            issue = f"  {row.issue}" if row.issue else ""
            note = f"  {row.note}" if row.note else ""
            lines.append(f"{name}  {row.cls}{issue}{note}\n")
    path.write_text("".join(lines), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path("."), help="repository root (default: .)")
    ap.add_argument("--base", help="git revision whose registry defines the grandfathered rows")
    ap.add_argument("--update", action="store_true", help="add missing names, drop stale rows")
    ap.add_argument("--github", action="store_true", help="emit GitHub annotations")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.update:
            update(root)
        problems, counts = evaluate(root, args.base)
    except EvaluationError as exc:
        print(f"check_switch_registry: could not evaluate: {exc}", file=sys.stderr)
        return EXIT_UNEVALUATED
    for p in problems:
        print(f"::error file={REGISTRY.as_posix()}::{p}" if args.github else p)
    summary = ", ".join(f"{counts[c]} {c}" for c in CLASSES)
    print(f"check_switch_registry: {counts['total']} switches; {summary}")
    return EXIT_VIOLATION if problems else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
