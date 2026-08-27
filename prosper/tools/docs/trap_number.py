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
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_FILE = "prosper/docs/GAME_COMPAT_ORCHESTRATION.md"
DEFAULT_HEADER = "Instrument"
# `gh pr list --json files` caps the array at 100 entries and gives NO "there is more" signal.
# CONFIRMED, and stated as fact rather than as a caveat (#2610 review). This repository's largest PR
# has 57 changed files, so the cap is unreachable here and had to be established elsewhere:
#   * server side -- `files(first: 101)` is refused outright: "Requesting 101 records on the `files`
#     connection exceeds the `first` limit of 100 records" (EXCESSIVE_PAGINATION), verbatim.
#   * in the wild -- `cli/cli#14082` reports changedFiles=1161, while `gh pr view --json files`
#     returns exactly 100 of them, silently, with no error and no indicator.
# So a PR at the cap that does NOT list our path proves nothing, and skipping it would make its claim
# invisible -- the same silent direction as the --limit truncation above. Scan it instead: one extra
# request against an invisible collision.
FILES_CAP = 100

# A leading-pipe row whose first cell is a bare integer. Deliberately the same shape as
# check_numbered_table.LEADING_NUMBER, so the allocator and the gate agree on what a numbered row
# is; a tool that disagreed with its own gate would hand out numbers the gate then rejects.
LEADING_NUMBER = re.compile(r"^\s*\|\s*(\d+)\s*\|")
FENCE = re.compile(r"^\s*(```|~~~)")
DELIMITER = re.compile(r"^\s*\|[\s:|-]+\|?\s*$")


class ScanError(RuntimeError):
    """Anything that would make the answer a guess rather than a measurement."""


def run(cmd: list[str]) -> str:
    """Run `cmd`, or raise ScanError. Never lets an exception escape as a traceback.

    Resolves the executable with `shutil.which` FIRST, for two reasons that turn out to be the same
    reason:

      * The tool promises to fail closed with a message. Without this it did not: a missing `gh`
        escaped as an uncaught FileNotFoundError and printed a traceback -- on every platform --
        instead of the ScanError the docstring advertises. A traceback is not a refusal to answer,
        it is a crash that happens to also not answer, and the two read very differently to whoever
        is holding a number they are about to write.
      * On Windows it is also load-bearing for CORRECTNESS. Python resolves a bare command name
        through CreateProcess, which appends only `.EXE` and does NOT walk PATHEXT -- that is
        cmd.exe's job. `shutil.which` does walk PATHEXT, and CreateProcess can execute a `.cmd`
        given its full path. Without this, anything named `gh.cmd` on PATH is invisible.
    """
    exe = shutil.which(cmd[0])
    if exe is None:
        raise ScanError(
            f"{cmd[0]!r} is not on PATH, so `{' '.join(cmd[:3])}...` cannot run. This tool answers "
            f"from `git` and `gh` and refuses to guess without them."
        )
    try:
        # encoding="utf-8" EXPLICITLY, not text=True alone. text=True decodes with the locale
        # default, which on a Windows host is cp1252 -- and this tool's whole job is to pipe a
        # UTF-8 Markdown document through a subprocess. The doc contains typographic quotes and
        # em dashes, so the decode raised UnicodeDecodeError inside subprocess's reader THREAD,
        # which left proc.stdout as None while returncode stayed 0. The tool then crashed 150
        # lines away in the table parser with "'NoneType' object has no attribute 'split'".
        # trap_number.py was therefore unusable on Windows, and unusable in a way that pointed
        # the reader at the wrong file.
        proc = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, encoding="utf-8")
    except OSError as exc:  # pragma: no cover - permissions, exec format, a full disk
        raise ScanError(f"{' '.join(cmd[:3])}... could not be run: {exc}") from exc
    if proc.returncode != 0:
        raise ScanError(f"{' '.join(cmd[:3])}... failed (rc={proc.returncode}): {proc.stderr.strip()}")
    # The backstop for the same class, and it is the half that matters: ANY future decode
    # failure in that reader thread yields None here with a zero returncode. Without this the
    # tool violates its own docstring -- "Never lets an exception escape as a traceback" -- by
    # crashing somewhere else entirely, which reads as a bug in the table rather than in the
    # read. Whoever is holding a number they are about to write needs to be able to tell those
    # two apart.
    if proc.stdout is None:
        raise ScanError(
            f"{' '.join(cmd[:3])}... produced output this tool could not decode as UTF-8, so "
            f"the table cannot be read. This tool refuses to guess a number from a partial read."
        )
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


# Mirrors check_numbered_table.MAX_JUMP. Without it a typo'd number in an OPEN PR (`1189` for `189`)
# propagates straight into the advice, and the allocator hands out a number the gate then rejects --
# the tool and its own backstop disagreeing, which is worse than either being wrong alone (#2610
# review). Kept as a literal rather than imported so this stays a standalone script.
MAX_JUMP = 50


def added_rows_from_patch(patch: str, path: str) -> list[int]:
    """Row numbers a unified patch ADDS to `path`. Pure, so it is testable without the network.

    Deliberately the patch and not a set difference over file contents. "Present here, present in the
    base, different text" is not a duplicate -- it is also what an AMENDED row looks like, and trap
    rows are amended routinely as evidence accrues. A first draft used the text comparison and would
    have reported an extended row as a duplicate. Only a line the patch introduces is a new claim,
    and a number appearing on BOTH sides of the patch is an amendment rather than an allocation.
    """
    added: set[int] = set()
    removed: set[int] = set()
    in_file = False
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            in_file = line[6:].strip() == path
            continue
        if line.startswith("diff --git"):
            in_file = False
            continue
        if not in_file:
            continue
        if m := re.match(r"^\+\s*\|\s*(\d+)\s*\|", line):
            added.add(int(m.group(1)))
        elif m := re.match(r"^-\s*\|\s*(\d+)\s*\|", line):
            removed.add(int(m.group(1)))
    return sorted(added - removed)


def added_rows_from_diff(repo: str, number: int, path: str) -> list[int]:
    """`added_rows_from_patch` over this PR's patch.

    RAISES rather than returning None on a failed `gh pr diff`, and that is the whole point. The
    first version returned None, the caller treated it as falsy, and the report fell through to the
    set-difference branch -- the exact logic this function was added to REPLACE, because set
    arithmetic cannot see a duplicate the base already holds. So an unreadable diff would have
    silently restored the defect while the run looked successful: B4's shape, one function over
    (#2610 review).
    """
    exe = shutil.which("gh")
    if exe is None:
        raise ScanError("'gh' is not on PATH, so this PR's patch cannot be read.")
    # Same explicit encoding as run(); a PR body or patch is as likely to carry non-ASCII as
    # the document itself.
    proc = subprocess.run([exe, "pr", "diff", str(number), "--repo", repo],
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise ScanError(
            f"`gh pr diff {number}` failed (rc={proc.returncode}): {proc.stderr.strip()}. Without "
            f"the patch this PR's claim cannot be distinguished from an amended row, and falling "
            f"back to comparing file contents would silently miss a duplicate already on "
            f"{repo}'s base."
        )
    return added_rows_from_patch(proc.stdout, path)


def added_numbers(text: str, base_text: str, header_text: str) -> list[int]:
    """The row numbers this version has that the base does not -- the PR's actual claim.

    Not `range(base_max + 1, pr_max + 1)`. Since gaps became legal (#2089) a branch's maximum says
    nothing about which numbers between it and the base are taken: this branch itself holds 191 and
    leaves 188-190 free. Printing the range would report three claims that do not exist and push the
    next lane needlessly high.
    """
    return sorted(set(table_numbers(text, header_text)) - set(table_numbers(base_text, header_text)))


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
        base_numbers = set(table_numbers(base_text, args.table_header))
        if base_max is None:
            raise ScanError(
                f"no numbered table whose header contains {args.table_header!r} in "
                f"{args.base}:{args.file} -- wrong path, or the table format changed"
            )

        claims: list[tuple[int, str, int | None, bool, list[int], list[int]]] = []
        prs = open_prs(args.limit)
        if len(prs) >= args.limit:
            # The one failure this tool must not have. `gh pr list --limit N` returns AT MOST N and
            # says nothing about what it dropped, so a truncated list yields a smaller maximum and a
            # confidently wrong "next free number" -- the exact defect this exists to prevent, wearing
            # the look of a successful run. Reproduced at --limit 3 while three open PRs held 188.
            raise ScanError(
                f"--limit {args.limit} returned {len(prs)} open PR(s), so the list is (or may be) "
                f"TRUNCATED and any answer would silently omit whatever was dropped. Re-run with a "
                f"higher --limit."
            )
        touching = []
        for pr in prs:
            paths = [f.get("path") for f in pr.get("files") or []]
            if args.file in paths:
                touching.append(pr)
            elif len(paths) >= FILES_CAP:
                # `gh pr list --json files` pages at 100 entries, so on a PR at the cap the absence
                # of our path proves nothing. Scan it rather than skip it: the cost is one request
                # and the alternative is an invisible claim.
                touching.append(pr)
        for pr in touching:
            text = file_at(repo, args.file, pr["headRefOid"])
            claims.append((pr["number"], pr["title"], highest(text, args.table_header),
                           pr["isDraft"], added_numbers(text, base_text, args.table_header),
                           added_rows_from_diff(repo, pr["number"], args.file)))
    except ScanError as exc:
        # Hard error. A fallback to "master alone" would answer the question this tool exists to
        # refuse to answer that way, and the caller could not tell the two apart.
        print(f"error: {exc}", file=sys.stderr)
        print("The answer would be a guess, so none is given. Fix the above and re-run.", file=sys.stderr)
        return 1

    # A claim more than MAX_JUMP above the base is a typo, not an allocation -- check_numbered_table
    # will reject it -- so it must not drag the advice up with it. Without this the allocator and its
    # own backstop disagree: the tool hands out 1190 and the gate then refuses the row (#2610 review).
    sane = [c[2] for c in claims if c[2] is not None and c[2] <= base_max + MAX_JUMP]
    wild = [(c[0], c[2]) for c in claims if c[2] is not None and c[2] > base_max + MAX_JUMP]
    overall = max([base_max] + sane)
    # Same notion as `contested` below: patch-derived, so a stacked PR's inherited rows are not
    # counted as its claims.
    claimed: set[int] = {n for c in claims for n in c[5] if n <= base_max + MAX_JUMP}
    if args.quiet:
        print(overall + 1)
        return 0

    print(f"{args.file}  (table header contains {args.table_header!r})")
    print(f"  {args.base:<28} highest row {base_max}")
    for number, title, pr_max, draft, added, dup in sorted(claims, key=lambda c: -(c[2] or 0)):
        if pr_max is None:
            note = "no numbered table in its copy"
        elif clash := [n for n in dup if n in base_numbers]:
            note = (f"DUPLICATES {', '.join(str(n) for n in clash)} ALREADY ON {args.base} "
                    f"-- its patch ADDS a row with that number")
        elif dup:
            # `dup` (what the PATCH adds), never `added` (this file's numbers minus master's). A
            # STACKED PR inherits its base's rows, so `added` credits it with claims that belong to
            # the branch underneath: #2616 is stacked on #2610 and was reported as claiming 189 for
            # #2610's row. One notion of "claim" throughout, or the report and the collision
            # arithmetic disagree (#2610 review).
            note = f"CLAIMS {', '.join(str(n) for n in dup)}"
        elif added:
            note = (f"inherits {', '.join(str(n) for n in added)} from its base branch, "
                    f"claims nothing of its own")
        else:
            note = "no claim (older base, adds no row)"
        flag = " [draft]" if draft else ""
        print(f"  PR #{number:<25} highest row {pr_max}   {note}{flag}")
    print(f"  scanned {len(prs)} open PR(s), {len(touching)} touching this file (--limit {args.limit})")
    for number, pr_max in wild:
        print(f"  IGNORING PR #{number}'s {pr_max}: more than {MAX_JUMP} above {args.base}'s "
              f"{base_max}, so it is a typo rather than an allocation -- check_numbered_table "
              f"--baseline rejects it. Not counted toward the next free number.")
    print()
    # How many lanes already hold the number a naive "master max + 1" would take. When more than one
    # does, "next free" is a trap: every loser steps to it simultaneously and collides again one
    # number up. Stepping CLEAR of the whole contested band is the cheap fix, and it is only safe
    # because gaps stopped being defects (#2089).
    #
    # TWO FIXES HERE, and only one of them is the crash. The collision arithmetic read `added`
    # (index 4: this PR's numbers minus master's) when the tool also carries `dup` (index 5:
    # patch-derived, what the PR's diff actually ADDS). Those disagree exactly where it matters. A
    # STACKED PR inherits its base's rows, so `added` reports them as its own claims -- #2616 is
    # stacked on #2610 and both showed `CLAIMS 189` for one row -- while `dup` correctly attributes
    # nothing to it. So the weaker notion invented a collision that did not exist.
    #
    # Switching to the patch-derived set removes that misreport, and it also removes the TRIGGER for
    # the crash below. It does NOT remove the crash: the unpack in `racing` still names five fields
    # of a six-field tuple, so a GENUINE collision would still die here -- in the one path this tool
    # exists for, after printing the table, with `--quiet` returning cleanly beforehand so a scripted
    # caller sees nothing wrong. Fixing only the trigger would have left that live and looked green.
    contested = sorted(n for n in claimed if sum(1 for c in claims if n in c[5]) > 1)
    if contested:
        racing = [f"#{n}" for n, _, _, _, _, dup in claims if set(dup) & set(contested)]
        band = ", ".join(str(n) for n in contested)
        print(f"COLLISION: {band} claimed by more than one open PR ({', '.join(racing)}).")
        print(f"next free number: {overall + 1}   --   CONSIDER {overall + 4} INSTEAD: if every")
        print("loser steps to the next free number they collide again one number up. Stepping clear")
        print("of the contested band costs a gap, legal since #2089, and takes you out of the race.")
    else:
        print(f"next free number: {overall + 1}")
    print("Advisory only -- another lane can take it in the same minute. Push and merge promptly;")
    print("if you collide, renumber to any HIGHER number and merge (gaps are legal since #2089).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
