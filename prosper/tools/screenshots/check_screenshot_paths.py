#!/usr/bin/env python3
"""Every committed screenshot lives at a path that says which title it shows and when it was taken.

`assets/screenshots/` grew as one flat folder of 261 files named however each PR's author chose:
by issue number (`3407-default-control-gta.webp`), by title spelled two ways (`alex-kidd.webp`,
`alexkidd-title-wave32-proved.webp`), with or without what the picture is evidence of, and never
with a date. A second root, `prosper/docs/screenshots/`, holds 53 more. Finding "every capture of
GTA V" or "what did Sonic look like in September" means grepping names that do not share a
convention. This gate fixes the convention for every new file (spec rule ASSET-1, ADR 0026):

    assets/screenshots/<GROUP>/<YYYY-MM-DD>-<what>[-i<issue>].webp

  GROUP   `<TITLE_ID>-<slug>` for one title (`PPSA04263-gta5`), or `cross-title` for a capture
          comparing several titles or showing a shared fix, or `app` for the frontend's own UI.
          One title id has exactly one folder: `PPSA04263-gta5` and `PPSA04263-gtav` cannot both
          exist.
  date    the day the capture was taken, ISO order, so a folder lists in time order.
  what    lowercase kebab-case: what the picture is evidence of (`prologue-bank-lobby`).
  issue   optional `-i<number>`, the issue or PR the capture belongs to.

Format is WebP: `tools/screenshots/shrink.py` re-encodes to a 1920-wide WebP, and the size gate in
the `Docs` job already enforces that.

GRANDFATHERING. Files committed before the convention are listed in `legacy_screenshot_paths.txt`
and pass as they are. The list may only shrink: a listed path that no longer exists fails as
stale, so moving a legacy file into the convention also deletes its line. Any image under either
root that neither conforms nor is listed fails.

REPORT-ONLY. While ADR 0026 is proposed, CI runs this with `--report-only`: violations print as
warnings and the step exits 0, so the convention is visible to every lane without being binding.
Accepting the ADR removes the flag. The flag never hides an evaluation failure (exit 2).

WHAT IT CANNOT SEE: whether the title id in a folder is the title the picture shows, or whether
the date is when it was taken.

EXIT STATUS: 0 clean, 1 violation, 2 could not evaluate.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

EXIT_OK, EXIT_VIOLATION, EXIT_UNEVALUATED = 0, 1, 2

ROOTS = ("assets/screenshots/", "prosper/docs/screenshots/")
IMAGE_SUFFIXES = (".webp", ".png", ".jpg", ".jpeg", ".gif", ".bmp")
LEGACY = Path("prosper/tools/screenshots/legacy_screenshot_paths.txt")
KEBAB = r"[a-z0-9]+(?:-[a-z0-9]+)*"
TITLE_GROUP_RE = re.compile(rf"(?P<id>(?:PPSA|CUSA)[0-9]{{5}})-{KEBAB}")
GROUP_RE = re.compile(rf"(?:(?:PPSA|CUSA)[0-9]{{5}}-{KEBAB}|cross-title|app)")
FILE_RE = re.compile(
    rf"[0-9]{{4}}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])-{KEBAB}(?:-i[0-9]+)?\.webp"
)


class EvaluationError(Exception):
    """The gate could not establish an answer; exit 2, never a pass."""


def tracked_images(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", *ROOTS],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise EvaluationError(f"git ls-files failed: {proc.stderr.strip()}")
    return sorted(p for p in proc.stdout.split("\0") if p and p.lower().endswith(IMAGE_SUFFIXES))


def conforms(path: str) -> bool:
    if not path.startswith("assets/screenshots/"):
        return False
    parts = path[len("assets/screenshots/") :].split("/")
    return (
        len(parts) == 2 and bool(GROUP_RE.fullmatch(parts[0])) and bool(FILE_RE.fullmatch(parts[1]))
    )


def read_legacy(root: Path) -> set[str]:
    path = root / LEGACY
    if not path.is_file():
        raise EvaluationError(f"{LEGACY} not found")
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def evaluate(root: Path) -> tuple[list[str], int, int]:
    images = tracked_images(root)
    legacy = read_legacy(root)
    problems = []
    for path in images:
        if path in legacy:
            if conforms(path):
                problems.append(f"{path}: conforms; delete its line from {LEGACY.name}")
            continue
        if not conforms(path):
            problems.append(
                f"{path}: not assets/screenshots/<TITLE_ID>-<slug>|cross-title|app/"
                "<YYYY-MM-DD>-<what>[-i<issue>].webp"
            )
    for path in sorted(legacy - set(images)):
        problems.append(f"{path}: listed in {LEGACY.name} but no longer exists; delete the line")
    folders: dict[str, set[str]] = {}
    for path in images:
        parts = path.split("/")
        if path.startswith("assets/screenshots/") and len(parts) == 4:
            m = TITLE_GROUP_RE.fullmatch(parts[2])
            if m:
                folders.setdefault(m.group("id"), set()).add(parts[2])
    for title_id, names in sorted(folders.items()):
        if len(names) > 1:
            problems.append(f"{title_id} has more than one folder: {', '.join(sorted(names))}")
    return problems, len(images), len(legacy & set(images))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."), help="repository root (default: .)")
    ap.add_argument("--github", action="store_true", help="emit GitHub annotations")
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="print violations as warnings and exit 0 (until ADR 0026 is accepted)",
    )
    args = ap.parse_args(argv)
    try:
        problems, total, legacy = evaluate(args.root.resolve())
    except EvaluationError as exc:
        print(f"check_screenshot_paths: could not evaluate: {exc}", file=sys.stderr)
        return EXIT_UNEVALUATED
    level = "warning" if args.report_only else "error"
    for p in problems:
        print(f"::{level}::{p}" if args.github else p)
    print(
        f"check_screenshot_paths: {total} image(s), {total - legacy} in the convention, "
        f"{legacy} grandfathered, {len(problems)} problem(s)"
        + (" (report-only)" if args.report_only else "")
    )
    if args.report_only:
        return EXIT_OK
    return EXIT_VIOLATION if problems else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
