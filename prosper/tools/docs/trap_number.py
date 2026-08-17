#!/usr/bin/env python3
"""Allocate the next free row number for a numbered doc table, against master AND every open PR.

WHY THIS EXISTS. The instrument-trap table in `docs/GAME_COMPAT_ORCHESTRATION.md` is appended to by
several lanes at once, and the obvious allocation -- read the highest row on `origin/master`, add
one -- is wrong the moment another lane pushes. It is not wrong *rarely*: #1729 records four
collisions in one day, and on 2026-08-17 #2574 and #2581 both read 181 and both wrote 182. Reading
the max more carefully cannot fix that, because the competing claim was never on master to be read;
it was sitting in an open PR.

So this reads the same table out of `origin/master` and out of the head of every open pull request,
and reports the highest number each one holds. The next free number is one above the maximum of all
of them.

AN ADVISOR, NOT A GATE, and the distinction matters enough to state twice. Two lanes running this in
the same minute both see the same free number, and both are right at the moment they look. It
shrinks the window in which a collision is possible; only merge order closes it. The backstop is
`check_numbered_table.py --ordered`, which rejects a duplicate wherever it came from -- and since
gaps became legal (#2089) losing a race costs one character rather than a re-derivation.

WHAT IT REPORTS AND WHY IT IS NOT JUST A NUMBER. Every open PR is listed, including those whose
highest row is at or below master's -- because "PR #1728: highest row 41" against a master of 42 is
NOT a claim, it is a branch on an older base that adds no row, and a bare "next = 45" hides which of
the two you are looking at. The difference is exactly what tells you whether you are in a race.

WHAT IT CANNOT SEE, stated so silence is not read as coverage:
  * A row written by a lane that has not pushed. Nothing can see that.
  * A PR whose diff does not touch the file. The scan asks GitHub which files each PR changes and
    only fetches the ones that do, so a PR that will *later* add a row is invisible until it pushes
    one.
  * A draft or a PR beyond --limit. Both are reported as scanned counts so the bound is visible
    rather than assumed.
  * Anything at all, if `gh` is not authenticated -- which is why that case is a hard error and not
    a fallback to master alone. Silently degrading to "master says 186, take 187" would reproduce
    the exact defect this tool exists to prevent, while looking like it had worked.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_FILE = "prosper/docs/GAME_COMPAT_ORCHESTRATION.md"
DEFAULT_HEADER = "Instrument"

# A leading-pipe row whose first cell is a bare integer. Deliberately the same shape as
# check_numbered_table.LEADING_NUMBER, so the allocator and the gate agree on what a numbered row
# is; a tool that disagreed with its own gate would hand out numbers the gate then rejects.
LEADING_NUMBER = re.compile(r"^\s*\|\s*(\d+)\s*\|")
FENCE = re.compile(r"^\s*(```|~~~)")
DELIMITER = re.compile(r"^\s*\|[\s:|-]+\|?\s*$")


class ScanError(RuntimeError):
    """Anything that would make the answer a guess rather than a measurement."""


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ScanError(f"{' '.join(cmd[:3])}... failed (rc={proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


def table_numbers(text: str, header_text: str) -> list[int]:
    """Row numbers of the numbered table whose header contains `header_text`.

    Fences are skipped for the same reason the checker skips them: this repository's docs paste tool
    output full of pipe characters, and a fenced example of the table itself would otherwise be read
    as the table.
    """
    numbers: list[int] = []
    fenced = False
    in_table = False
    for line in text.split("\n"):
        if FENCE.match(line):
            fenced = not fenced
            in_table = False
            continue
        if fenced:
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        if DELIMITER.match(line):
            continue
        if header_text in line and not LEADING_NUMBER.match(line):
            in_table = True
            continue
        if in_table and (m := LEADING_NUMBER.match(line)):
            numbers.append(int(m.group(1)))
    return numbers


def highest(text: str, header_text: str) -> int | None:
    nums = table_numbers(text, header_text)
    return max(nums) if nums else None


def open_prs(limit: int) -> list[dict]:
    out = run([
        "gh", "pr", "list", "--state", "open", "--limit", str(limit),
        "--json", "number,title,headRefOid,files,isDraft",
    ])
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:  # pragma: no cover - gh would have to emit non-JSON
        raise ScanError(f"could not parse `gh pr list` output: {exc}") from exc


def file_at(repo: str, path: str, ref: str) -> str:
    # The contents API, not `git fetch refs/pull/N/head`: one request per PR instead of a fetch, and
    # it works for a PR from a fork with no extra permissions or remotes.
    return run([
        "gh", "api", f"repos/{repo}/contents/{path}?ref={ref}",
        "-H", "Accept: application/vnd.github.raw",
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", default=DEFAULT_FILE, help=f"path in the repo (default {DEFAULT_FILE})")
    ap.add_argument("--table-header", default=DEFAULT_HEADER,
                    help=f"select the table whose header contains this (default {DEFAULT_HEADER!r})")
    ap.add_argument("--limit", type=int, default=60, help="how many open PRs to scan (default 60)")
    ap.add_argument("--base", default="origin/master", help="the merged baseline (default origin/master)")
    ap.add_argument("--no-fetch", action="store_true", help="skip `git fetch` (the base may be stale)")
    ap.add_argument("--quiet", action="store_true", help="print only the next free number")
    args = ap.parse_args()

    try:
        if not args.no_fetch:
            # Without this the base is whatever the last fetch left behind, which is the stale-read
            # half of the very defect being prevented.
            run(["git", "fetch", "--quiet", "origin"])

        repo = run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"]).strip()
        base_text = run(["git", "show", f"{args.base}:{args.file}"])
        base_max = highest(base_text, args.table_header)
        if base_max is None:
            raise ScanError(
                f"no numbered table whose header contains {args.table_header!r} in "
                f"{args.base}:{args.file} -- wrong path, or the table format changed"
            )

        claims: list[tuple[int, str, int | None, bool]] = []
        prs = open_prs(args.limit)
        touching = [
            pr for pr in prs
            if any(f.get("path") == args.file for f in pr.get("files") or [])
        ]
        for pr in touching:
            text = file_at(repo, args.file, pr["headRefOid"])
            claims.append((pr["number"], pr["title"], highest(text, args.table_header), pr["isDraft"]))
    except ScanError as exc:
        # Hard error. A fallback to "master alone" would answer the question this tool exists to
        # refuse to answer that way, and the caller could not tell the two apart.
        print(f"error: {exc}", file=sys.stderr)
        print("The answer would be a guess, so none is given. Fix the above and re-run.", file=sys.stderr)
        return 1

    overall = max([base_max] + [c[2] for c in claims if c[2] is not None])
    if args.quiet:
        print(overall + 1)
        return 0

    print(f"{args.file}  (table header contains {args.table_header!r})")
    print(f"  {args.base:<28} highest row {base_max}")
    for number, title, pr_max, draft in sorted(claims, key=lambda c: -(c[2] or 0)):
        if pr_max is None:
            note = "no numbered table in its copy"
        elif pr_max > base_max:
            note = f"CLAIMS {', '.join(str(n) for n in range(base_max + 1, pr_max + 1))}"
        else:
            note = "no claim (older base, adds no row)"
        flag = " [draft]" if draft else ""
        print(f"  PR #{number:<25} highest row {pr_max}   {note}{flag}")
    print(f"  scanned {len(prs)} open PR(s), {len(touching)} touching this file (--limit {args.limit})")
    print()
    print(f"next free number: {overall + 1}")
    print("Advisory only -- another lane can take it in the same minute. Push and merge promptly;")
    print("if you collide, renumber to any HIGHER number and merge (gaps are legal since #2089).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
