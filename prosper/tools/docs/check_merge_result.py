#!/usr/bin/env python3
"""Run the numbered-table gate against the MERGE RESULT, against a freshly fetched base.

WHY. A `pull_request` CI job validates `refs/pull/N/merge`, which GitHub computes **when the event
fires** and never recomputes as the base moves. So a green required check is a statement about a
merge that may no longer exist, and unlike a red one nobody re-derives it -- a red check gets
investigated, a green check is *acted on*. That is instrument trap 189, and #2581 merged on exactly
it: its `Docs` job was green against a base predating #2574, so nothing warned either author that
both had written instrument-trap row 182 (#2211).

The residual hole is not fixable in the workflow file -- the remedies are repository settings
(branch protection's "require branches to be up to date", or a merge queue) that only the owner can
apply. This is the part a checkout CAN do: compute the merge yourself, immediately before merging,
and run the gate on the result. It is the manual procedure that caught the 182 collision, made into
one command.

HOW, and why it does not touch your tree. `git merge-tree --write-tree` performs the merge entirely
in the object database: no checkout, no index, no `MERGE_HEAD` to clean up, and nothing that can
collide with another agent working in the same repository. It reports conflicts by exit status and
names the conflicted paths, so a textual conflict is distinguished from a clean merge that is
nevertheless invalid -- which is the case the gate exists for, since two rows claiming one number on
lines a few apart merge CLEANLY (#1701, #2211).

`--base` DEFAULTS TO `origin/main`, WHICH IS WRONG FOR A STACKED PR. The honest pre-merge check is
against the branch this one will actually merge into, so pass `--base <that branch>` when your PR is
stacked. A green run against the wrong base is not a weaker signal -- it is a confident statement
about a merge that will never happen.

WHAT IT CANNOT TELL YOU. It answers about the base as of the moment it runs. Another lane can merge
in the second after it prints, so this shrinks the window exactly the way `trap_number.py` does and
closes it no more than that -- run it as the last step before merging, not as the first step of a
review. It also checks only the numbered-table classes: a semantically wrong row, a reverted
paragraph, or a stale by-number reference in prose all merge clean and pass here (that boundary is
the same one `check_numbered_table.py`'s own WHAT THIS CANNOT SEE section draws).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_FILE = "prosper/docs/GAME_COMPAT_ORCHESTRATION.md"
DEFAULT_HEADER = "Instrument"
CHECKER = Path(__file__).resolve().parent / "check_numbered_table.py"


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    # encoding="utf-8" EXPLICITLY, not text=True alone. text=True decodes with the locale
    # default, which on a Windows host is cp1252, and this tool pipes a UTF-8 Markdown document
    # through `git show`. The decode raised inside subprocess's reader THREAD, so stdout came
    # back None with returncode 0 and the tool died 36 lines later on
    # `write_text(None)` -- "TypeError: data must be str, not NoneType".
    #
    # So the check the charter tells you to run before merging a numbered-table PR could not be
    # run on Windows at all. It was reported as unaffected by #3071 because the obvious probe --
    # running it with no arguments -- takes the "already contained in the base" early return
    # below, 36 lines before the affected code. A control that cannot reach the case it is
    # validating; found in review of #3071.
    proc = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8")
    if check and proc.returncode != 0:
        # (proc.stderr or ""): under strict decoding a cp1252 git error message is None here,
        # and None.strip() would replace the real diagnosis with an AttributeError.
        print(f"error: git {' '.join(args[:3])}... failed: {(proc.stderr or '').strip()}",
              file=sys.stderr)
        raise SystemExit(2)
    # Fail closed on an undecodable read rather than handing None to a caller. Every caller here
    # treats .stdout as a str, so this is the one place that can say what actually went wrong.
    if proc.stdout is None:
        print(f"error: git {' '.join(args[:3])}... produced output that could not be decoded as "
              f"UTF-8; refusing to check a merge result from a partial read.", file=sys.stderr)
        raise SystemExit(2)
    return proc


def _make_own_output_utf8() -> None:
    """Reconfigure THIS process's own stdout/stderr to UTF-8, so what this tool prints is safe
    regardless of the ambient locale of whoever launched it.

    A THIRD locale boundary, found in review of #3079 by a test that runs this tool as a real
    subprocess on real Windows Python: the checker-invocation fix (encoding="utf-8" on the parent
    decode, PYTHONIOENCODING=utf-8 forced into the checker's own environment) makes what THIS
    process reads from the checker always well-formed text -- verified directly against a genuine
    Windows Python build, not reasoned about. But this process then re-prints that text to ITS OWN
    sys.stdout/sys.stderr (the loop a few lines below), and nothing made THOSE safe: on an
    unconfigured Windows host, absent PYTHONIOENCODING/PYTHONUTF8 in the environment of whoever
    ran THIS tool, they still fall back to the ANSI codepage (cp1252). Measured on real Windows
    Python (embeddable 3.12.7, run under Wine -- close enough to a genuine Win32 process to trust
    for this): printing an arrow (U+2192, unrepresentable in cp1252) to an unforced stdout raises
    UnicodeEncodeError (stdout's error handler is 'strict'); the identical content on stderr does
    NOT raise (stderr defaults to 'backslashreplace') but is silently rewritten to the literal
    7-character text `\\u2192` instead of the glyph -- content corruption with no error at all.

    Fixing the checker boundary alone therefore does not fix the whole pipeline: a message that
    survives decode intact can still be mangled or crash on the way back OUT. `reconfigure` is
    tolerant of streams that are not real TextIOWrapper objects (a test redirecting stdout/stderr
    to io.StringIO has no such method) so it is a no-op there rather than an AttributeError.
    """
    for name in ("stdout", "stderr"):
        reconfigure = getattr(getattr(sys, name), "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def main() -> int:
    _make_own_output_utf8()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="origin/main", help="branch to merge into (default origin/main)")
    ap.add_argument("--head", default="HEAD", help="what to merge (default HEAD)")
    ap.add_argument("--file", default=DEFAULT_FILE, help=f"file to check (default {DEFAULT_FILE})")
    ap.add_argument("--table-header", default=DEFAULT_HEADER,
                    help=f"select the table whose header contains this (default {DEFAULT_HEADER!r})")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip `git fetch` -- ONLY for a scripted caller that just fetched; a stale "
                         "base is the exact defect this tool exists to catch")
    args = ap.parse_args()

    if not args.no_fetch:
        git("fetch", "--quiet", "origin")

    base = git("rev-parse", args.base).stdout.strip()
    head = git("rev-parse", args.head).stdout.strip()
    print(f"merging {args.head} ({head[:8]}) into {args.base} ({base[:8]})")

    if git("merge-base", "--is-ancestor", head, base, check=False).returncode == 0:
        print("nothing to check: this head is already contained in the base.")
        return 0

    # --write-tree leaves the working tree and index untouched, which matters in a repository
    # several agents share. Exit 1 means conflicts; the tree oid is still the first stdout line and
    # the conflicted paths follow, so the message can name them.
    merged = git("merge-tree", "--write-tree", base, head, check=False)
    if merged.returncode not in (0, 1):
        # (merged.stderr or ""): a check=False caller reads .stderr where git()'s own guard
        # cannot reach it, and under strict decoding that is None. Same shape as the guard in
        # trap_number.run; review of #3071 flagged both.
        err = merged.stderr or ""
        print(f"error: git merge-tree failed: {err.strip()}", file=sys.stderr)
        if "--write-tree" in err or "usage:" in err.lower():
            # git 2.38 introduced this mode. Named explicitly because the fallback (`git merge` in a
            # scratch worktree) is the thing this tool deliberately avoids, so "just use git merge"
            # is the wrong lesson to draw from a usage message.
            print("`git merge-tree --write-tree` needs git >= 2.38; this git is "
                  f"{git('--version').stdout.strip()}", file=sys.stderr)
        return 2
    lines = merged.stdout.split("\n")
    tree = lines[0].strip()
    if merged.returncode == 1:
        conflicts = [l for l in lines[1:] if l.strip()]
        print("MERGE CONFLICT -- resolve it before merging. Conflicted paths / messages:")
        for line in conflicts[:20]:
            print(f"  {line}")
        print()
        print("Re-apply your own hunks onto the base's version rather than taking the file whole")
        print("(instrument trap 41 / #1701): `git checkout <branch> -- <file>` reverts every OTHER")
        print("lane's edit to that file, with no conflict and no failing check.")
        return 1

    with tempfile.TemporaryDirectory() as d:
        merged_copy = Path(d) / "merged.md"
        base_copy = Path(d) / "base.md"
        show = git("show", f"{tree}:{args.file}", check=False)
        if show.returncode != 0:
            print(f"error: {args.file} is not in the merge result: "
                  f"{(show.stderr or '').strip()}", file=sys.stderr)
            return 2
        merged_copy.write_text(show.stdout, encoding="utf-8")
        base_copy.write_text(git("show", f"{base}:{args.file}").stdout, encoding="utf-8")

        # encoding="utf-8" AND env=...PYTHONIOENCODING on the SAME call, deliberately: this is a
        # BOTH-SIDES fix, not two independent ones. `encoding="utf-8"` fixes how THIS process
        # decodes the checker's captured stdout/stderr; it says nothing about what bytes the
        # checker actually wrote. Absent an override, the checker's own `print()` encodes with
        # WHATEVER the ambient locale is (cp1252 on an unconfigured Windows host) -- and today,
        # with neither side forced, parent and child read that same ambient locale and agree BY
        # ACCIDENT. Adding encoding="utf-8" here alone would break that accidental agreement
        # into a real one: the checker would still emit cp1252 bytes, and this call would then
        # insist on decoding them as UTF-8. `PYTHONIOENCODING=utf-8` in the child's environment
        # is what makes the checker itself emit UTF-8 regardless of locale, so the two sides
        # agree on purpose instead of by luck. Neither half alone is a fix -- see #3079.
        proc = subprocess.run(
            [sys.executable, str(CHECKER), "--ordered", "--table-header", args.table_header,
             "--baseline", str(base_copy), str(merged_copy)],
            capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    if proc.stdout is None or proc.stderr is None:
        # Fail closed rather than let a discarded stream read as "nothing to report". A decode
        # failure inside subprocess's capture does not always raise here -- on a platform whose
        # pipe reader runs in a background thread (this file's own git() helper measured that on
        # Windows: the decode fails in the READER THREAD, and the affected stream comes back None
        # with returncode 0 rather than propagating), the exception never reaches this frame at
        # all. The `if stream:` guard below treats `None` exactly like "the checker printed
        # nothing", so without this check a lost read would report a CLEAN merge from a read that
        # never happened -- strictly worse than a crash, because nothing about it looks wrong.
        # encoding="utf-8" plus the child-side PYTHONIOENCODING above should make this
        # unreachable in ordinary operation; this is the backstop for when it somehow is not.
        print("error: the checker's output could not be decoded as UTF-8; refusing to report a "
              "verdict from a partial read.", file=sys.stderr)
        return 2
    # Rewrite the temporary path back to the real one: an error naming a file in /tmp that no longer
    # exists reads as a tooling failure rather than as a finding about your change.
    for stream, out in ((proc.stdout, sys.stdout), (proc.stderr, sys.stderr)):
        if stream:
            text = stream.replace(str(merged_copy), f"{args.file} (merged)")
            print(text.replace(str(base_copy), f"{args.base}:{args.file}").rstrip(), file=out)

    if proc.returncode == 0:
        print(f"\nclean merge, and the merge RESULT passes the gate against {args.base} as of now.")
        print("Merge promptly: another lane landing after this point invalidates it.")
    elif "problem(s) found" not in proc.stdout + proc.stderr:
        # Classified on the checker's own summary marker rather than on its exit STATUS, and the
        # difference matters: argparse exits 2, but an unhandled exception exits **1** -- the same
        # status as a real finding. So a crash in the checker would otherwise be announced as "your
        # table is invalid", sending the author into their own diff for a fault that is not theirs.
        # That is the same mis-attribution this file already avoids for a textual conflict (#2616
        # review). Only a run that reported problems is treated as a finding.
        print(f"\nThe checker itself failed (rc={proc.returncode}) without reporting any table",
              file=sys.stderr)
        print("problem, so this says NOTHING about the merge result. Fix the invocation above,",
              file=sys.stderr)
        print("then re-run.", file=sys.stderr)
    else:
        print("\nThe merge is textually clean but the RESULT is invalid -- this is the case CI",
              file=sys.stderr)
        print("cannot see, because at head time the other row did not exist. Fix on your branch,",
              file=sys.stderr)
        print("then re-run. Gaps are legal (#2089), so a duplicate is repaired by renumbering YOUR",
              file=sys.stderr)
        print("row to any number above the current highest -- nothing waits on anything.", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
