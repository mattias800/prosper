#!/usr/bin/env python3
"""Answer one question about a pull request: is it safe to merge RIGHT NOW?

Everything here is shaped by two merges that went wrong on 2026-09-02, both from mechanical
causes rather than carelessness (#3259).

**A check name is not a status.** `gh pr checks` prints tab-separated columns and this
repository's check names contain spaces -- `Windows App`, `Linux App Package`,
`macOS (x86_64 / Rosetta 2)`. Splitting that on whitespace puts the second *word of the name*
where the status should be, so `Windows App` reads as a status of `App`. A shell gate matching
the text this way merged #3234 with a red `Windows MinGW`; an `awk '$2=="pass"'` written while
filing #3259 counted 2 passing checks where 8 were passing. Both readings look plausible, which
is exactly what makes them dangerous. So this tool consumes `--json name,state,bucket` and never
the text form, and `parse_checks` rejects a payload that is not that JSON shape rather than
falling back to a looser parse.

**CI green can describe a commit that would not merge.** GitHub's recorded PR head can lag the
branch: on #3243 it froze while the branch advanced two commits, no run was ever queued for the
newer ones, and the merge carried code CI had never seen. `CLAUDE.md` records this detachment for
*reviews* ("a rebase or a new push detaches every review from head"); it applies identically to
*checks*, and that half was written down nowhere. So the head the checks describe is compared
against the real branch tip, and a mismatch refuses however green the checks are.

**An empty check list is VOID, not green.** Immediately after a push, `gh pr checks` can report
nothing at all. "No checks failed" and "no checks ran" are the same answer to a naive `fail == 0`
test, which is the same trap `CLAUDE.md` records for `ctest` (plain `ctest` on an empty build
directory exits 0). Hence the `pass > 0` requirement, and hence this tool reports COUNTS: a count
is falsifiable, an exit code alone is not.

Usage:
    pr_merge_gate.py <pr-number> [--repo-dir DIR] [--json]

Exit status: 0 only when every check has passed, none is pending, at least one passed, and the
head the checks describe is the branch tip. Any other state exits 1 and says which rule refused.
Network or tooling failure exits 2 -- distinguishable from a refusal, because "the gate could not
run" must never be confused with "the gate said no".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field

# gh classifies every check into one of these. Only `pass` and `skipping` are compatible with
# merging: `skipping` is a job whose own conditions excluded it (the Progress tracker job skips on
# most PRs), which is a decision CI made deliberately, not an absence of information.
BUCKET_PASS = "pass"
BUCKET_FAIL = "fail"
BUCKET_PENDING = "pending"
BUCKET_SKIPPING = "skipping"
BUCKET_CANCEL = "cancel"

# A cancelled check is treated as blocking, not as skipped. A cancellation usually means someone
# stopped the run or a newer push superseded it, and in both cases nothing was verified.
BLOCKING_BUCKETS = (BUCKET_FAIL, BUCKET_CANCEL)


class GateError(RuntimeError):
    """The gate could not reach a verdict (bad input, `gh` failure, malformed payload)."""


@dataclass
class GateResult:
    ok: bool
    counts: dict
    reasons: list = field(default_factory=list)
    blocking: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    pr_head: str = ""
    branch_tip: str = ""

    def render(self, pr: int) -> str:
        c = self.counts
        lines = [
            "GATE #%d: pass=%d fail=%d pending=%d skipping=%d cancel=%d"
            % (pr, c["pass"], c["fail"], c["pending"], c["skipping"], c["cancel"])
        ]
        for name, bucket in self.blocking:
            lines.append("  BLOCKING: %s (%s)" % (name, bucket))
        for name in self.pending:
            lines.append("  pending:  %s" % name)
        if self.pr_head:
            same = self.pr_head == self.branch_tip
            lines.append(
                "  head %s branch tip (pr=%s tip=%s)"
                % ("matches" if same else "DOES NOT MATCH", self.pr_head[:8], self.branch_tip[:8])
            )
        for r in self.reasons:
            lines.append("  refused: %s" % r)
        lines.append("  => %s" % ("GREEN" if self.ok else "NOT MERGEABLE"))
        return "\n".join(lines)


def parse_checks(payload):
    """Take the JSON `gh pr checks --json name,state,bucket` emits, and nothing looser.

    The point of being strict here is that the text form of the same data is the trap this whole
    tool exists for: a permissive parser that fell back to splitting lines would reintroduce it.
    """
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError) as exc:
            raise GateError("check payload is not JSON: %s" % exc)
    if not isinstance(payload, list):
        raise GateError("check payload is not a JSON list (got %s)" % type(payload).__name__)
    out = []
    for i, row in enumerate(payload):
        if not isinstance(row, dict):
            raise GateError("check #%d is not an object" % i)
        if "bucket" not in row or "name" not in row:
            raise GateError("check #%d lacks 'name'/'bucket' -- wrong --json fields?" % i)
        name, bucket = row["name"], row["bucket"]
        if not isinstance(name, str) or not isinstance(bucket, str):
            raise GateError("check #%d has a non-string name/bucket" % i)
        out.append((name, bucket))
    return out


def evaluate(checks, pr_head=None, branch_tip=None):
    """Decide, from already-fetched facts. Pure -- this is the half the tests drive."""
    counts = {b: 0 for b in (BUCKET_PASS, BUCKET_FAIL, BUCKET_PENDING, BUCKET_SKIPPING, BUCKET_CANCEL)}
    blocking, pending = [], []
    for name, bucket in checks:
        if bucket in counts:
            counts[bucket] += 1
        else:
            # An unrecognised bucket is blocking. A new gh bucket must not silently read as
            # success just because this tool predates it.
            counts.setdefault("unknown", 0)
            counts["unknown"] += 1
            blocking.append((name, bucket))
            continue
        if bucket in BLOCKING_BUCKETS:
            blocking.append((name, bucket))
        elif bucket == BUCKET_PENDING:
            pending.append(name)

    reasons = []
    if counts[BUCKET_PASS] == 0:
        # Covers both "no checks reported yet" and "everything skipped": in neither case has
        # anything actually been verified.
        reasons.append("no check has passed -- an empty or all-skipped result is VOID, not green")
    if blocking:
        reasons.append("%d check(s) failed or were cancelled" % len(blocking))
    if pending:
        reasons.append("%d check(s) still pending" % len(pending))
    if pr_head is not None and branch_tip is not None and pr_head != branch_tip:
        reasons.append(
            "the checks describe %s but the branch tip is %s -- CI never saw what would merge"
            % (pr_head[:8], branch_tip[:8])
        )

    return GateResult(
        ok=not reasons,
        counts=counts,
        reasons=reasons,
        blocking=blocking,
        pending=pending,
        pr_head=pr_head or "",
        branch_tip=branch_tip or "",
    )


def _run(cmd, cwd):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except OSError as exc:
        raise GateError("could not run %s: %s" % (cmd[0], exc))
    if p.returncode != 0:
        raise GateError("%s failed (%d): %s" % (" ".join(cmd[:3]), p.returncode, p.stderr.strip()[:300]))
    return p.stdout


def collect(pr, repo_dir):
    checks = parse_checks(_run(["gh", "pr", "checks", str(pr), "--json", "name,state,bucket"], repo_dir))
    view = _run(["gh", "pr", "view", str(pr), "--json", "headRefOid,headRefName"], repo_dir)
    try:
        info = json.loads(view)
        pr_head, branch = info["headRefOid"], info["headRefName"]
    except (ValueError, KeyError) as exc:
        raise GateError("could not read PR head: %s" % exc)
    ls = _run(["git", "ls-remote", "origin", "refs/heads/%s" % branch], repo_dir)
    tip = ls.split("\t")[0].strip() if ls.strip() else ""
    if not tip:
        raise GateError("branch %s has no remote tip -- deleted, or never pushed?" % branch)
    return checks, pr_head, tip


def main(argv=None):
    ap = argparse.ArgumentParser(description="Is this PR safe to merge right now?")
    ap.add_argument("pr", type=int)
    ap.add_argument("--repo-dir", default=".", help="repository to run gh/git in")
    ap.add_argument("--json", action="store_true", help="emit the verdict as JSON")
    args = ap.parse_args(argv)
    try:
        checks, head, tip = collect(args.pr, args.repo_dir)
    except GateError as exc:
        # Exit 2, never 1: "the gate could not run" must be distinguishable from "the gate said no".
        print("GATE #%d: COULD NOT EVALUATE -- %s" % (args.pr, exc), file=sys.stderr)
        return 2
    res = evaluate(checks, head, tip)
    if args.json:
        print(json.dumps({"ok": res.ok, "counts": res.counts, "reasons": res.reasons,
                          "blocking": res.blocking, "pending": res.pending,
                          "pr_head": res.pr_head, "branch_tip": res.branch_tip}, indent=2))
    else:
        print(res.render(args.pr))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
