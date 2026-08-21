#!/usr/bin/env python3
"""lanekill -- kill processes that belong to YOUR worktree, and refuse the ones that do not.

Several agents and the human run this repo concurrently. `pkill -x screenshot` means "every
process named screenshot on this box", which with three lanes running is three runs -- and on
2026-08-21 exactly that killed a concurrent lane's measurement sweep at 36 of 64 frames,
costing a result that was never recovered (instrument trap 213).

The failure is not that `pkill` is dangerous. It is that BOTH of its selectors answer *what* a
process is and NEITHER answers *whose* it is:

    pkill -f screenshot     over-matches substrings, and matches its own shell (trap 135)
    pkill -x screenshot     matches accurately -- and still kills every lane's copy

Swapping `-f` for `-x` fixes precision, which is a different property from ownership. This tool
adds the missing selector: it attributes each candidate to a git worktree and kills only the
ones inside yours.

    lanekill.py screenshot            # census; kills nothing
    lanekill.py screenshot --yes      # kill the ones in this worktree
    lanekill.py screenshot --yes --any-tree   # loud override; needs a stated reason

Attribution reuses `worktree_reclaim.py`'s process scanner rather than reimplementing it. That
matters more than code reuse: that scanner matches by (st_dev, st_ino) rather than by path
string, because `/home` is a symlink to `/var/home` on the host and a REAL bind mount inside the
ps5ys distrobox -- so a container process's `/proc/<pid>/cwd` reads back with a different
spelling for the same directory. A string-matching version of this tool would silently fail to
attribute every distrobox process, conclude they belong to nobody, and refuse to kill anything
(or worse, with --any-tree, kill everything). Hand-rolling a weaker guard than the one that
already exists is itself a recorded failure in this repo.

Exit status: 0 if it did what you asked, 1 if it refused something, 2 on a usage or platform
error. It FAILS CLOSED: with no readable /proc the attribution cannot be made at all, so nothing
is killed, because "I could not tell whose this is" must never render as "it is yours".
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from worktree_reclaim import Worktree, attach_holders, list_worktrees, run, scan_processes
except ImportError as exc:  # pragma: no cover - import shape is pinned by the selftest
    print(f"lanekill: cannot import worktree_reclaim.py ({exc})", file=sys.stderr)
    raise SystemExit(2) from exc


def repo_root(start: str) -> str | None:
    rc, out, _ = run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=start)
    if rc != 0 or not out.strip():
        return None
    common = out.strip()
    return os.path.dirname(common) if os.path.basename(common) == ".git" else common


def my_worktree(trees: list[Worktree], cwd: str) -> Worktree | None:
    """The tree containing `cwd`, by inode identity -- never by string prefix.

    A string prefix would get this wrong in the same distrobox case the scanner documents, and
    getting it wrong HERE is the dangerous direction: mis-identifying which tree is yours turns
    every other lane's process into a legitimate target.
    """
    try:
        here = os.stat(cwd)
    except OSError:
        return None
    cur = os.path.realpath(cwd)
    keys: dict[tuple[int, int], Worktree] = {}
    for t in trees:
        try:
            st = os.stat(t.real)
        except OSError:
            continue
        keys[(st.st_dev, st.st_ino)] = t
    del here
    while True:
        try:
            st = os.stat(cur)
        except OSError:
            return None
        hit = keys.get((st.st_dev, st.st_ino))
        if hit is not None:
            return hit
        parent = os.path.dirname(cur)
        if parent == cur or len(parent) <= 1:
            return None
        cur = parent


STRONG_KINDS = ("cwd", "exe")


def attribute(trees: list[Worktree]) -> tuple[dict[int, Worktree], dict[int, str]]:
    """Attribute each pid to at most one worktree, refusing every ambiguous case.

    ONLY `cwd` and `exe` constitute ownership. An `fd` or a memory mapping into a tree says the
    process READS something there -- a build output, a log, a dump -- which is not the same claim
    and must never authorise a kill.

    When cwd and exe name DIFFERENT trees the answer is *undecidable*, not "pick one". The first
    version of this function preferred whichever holder happened to be visited last, which made the
    verdict depend on `git worktree list` ORDER: a binary in wt-a run from wt-b and a binary in
    wt-b run from wt-a both attributed to wt-b. Same shape, opposite semantics, and the loser gets
    its run killed under a line that says MINE. Undecidable is the only honest answer, and this
    tool's whole purpose is that undecidable must not render as yours.
    """
    strong: dict[int, set[str]] = {}
    weak: set[int] = set()
    by_real: dict[str, Worktree] = {}
    for t in trees:
        by_real[t.real] = t
        for h in t.holders:
            if h.kind in STRONG_KINDS:
                strong.setdefault(h.pid, set()).add(t.real)
            else:
                weak.add(h.pid)

    owner: dict[int, Worktree] = {}
    why: dict[int, str] = {}
    for pid, reals in strong.items():
        if len(reals) == 1:
            owner[pid] = by_real[next(iter(reals))]
        else:
            why[pid] = f"cwd and exe disagree across {len(reals)} worktrees"
    for pid in weak - set(strong):
        why[pid] = "only an open fd or a mapping points into a worktree, which is not ownership"
    return owner, why


def classify(matches: list[int], owner_of: dict[int, Worktree], mine: Worktree
             ) -> tuple[list[int], list[tuple[int, Worktree]], list[int]]:
    """Split candidates into (ours, theirs, undecidable). Pure, so it can be tested directly.

    Extracted because the first version of this file had it inline in `main()`, where no test
    reached it: a reviewer mutated the tool six ways -- each turning it into a cross-lane killer --
    and every arm stayed green, because every arm called only `my_worktree()`. The decision that
    actually sends a signal now has arms of its own.
    """
    ours: list[int] = []
    theirs: list[tuple[int, Worktree]] = []
    undecidable: list[int] = []
    for pid in matches:
        t = owner_of.get(pid)
        if t is None:
            undecidable.append(pid)
        elif t.real == mine.real:
            ours.append(pid)
        else:
            theirs.append((pid, t))
    return ours, theirs, undecidable


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Kill processes matching a name that belong to YOUR worktree.",
        epilog="Census by default. Nothing is signalled without --yes.",
    )
    ap.add_argument("pattern", help="exact process name, as `pkill -x` matches it")
    ap.add_argument("--yes", action="store_true", help="actually signal the matches in your tree")
    # A REASON string that is only echoed is theatre: nothing validates it and nothing durable
    # records it, and `--any-tree ""` was falsy, so an empty reason silently degraded into a
    # refusal that looked like an override. A bare boolean states the same thing honestly.
    ap.add_argument(
        "--any-tree",
        action="store_true",
        help="also signal matches OUTSIDE your tree. Another lane is probably mid-run.",
    )
    ap.add_argument("--signal", default="TERM", help="signal name, default TERM")
    args = ap.parse_args()

    # `getattr(signal, "SIG" + name)` also resolves SIG_DFL and SIG_IGN, which are HANDLER
    # constants (0 and 1), not signals. `--signal _dfl` then called os.kill(pid, 0) -- the
    # existence probe -- and printed "signalled 1" while doing nothing at all. Match against the
    # real signal set instead.
    wanted = f"SIG{args.signal.upper().removeprefix('SIG')}"
    sig = next((s for s in signal.Signals if s.name == wanted), None)
    if sig is None:
        print(f"lanekill: unknown signal {args.signal!r}", file=sys.stderr)
        return 2

    cwd = os.getcwd()
    repo = repo_root(cwd)
    if repo is None:
        print("lanekill: not inside a git repository -- cannot attribute anything", file=sys.stderr)
        return 2

    trees = list_worktrees(repo)
    mine = my_worktree(trees, cwd)
    if mine is None:
        print(f"lanekill: {cwd} is not inside any registered worktree of {repo}", file=sys.stderr)
        print("  Attribution is undecidable here, so nothing is signalled.", file=sys.stderr)
        return 2

    refs, supported = scan_processes()
    if not supported:
        # macOS/Windows: no readable /proc. An absent instrument reads as "nobody owns anything",
        # which with --any-tree would authorise killing every lane's run. Fail closed.
        print("lanekill: no readable /proc -- ownership cannot be established on this platform.",
              file=sys.stderr)
        print("  Refusing rather than guessing. Scope the kill by PID yourself.", file=sys.stderr)
        return 2

    attach_holders(trees, refs)

    comm_of: dict[int, str] = {}
    for pid, _kind, comm, _path in refs:
        comm_of.setdefault(pid, comm)

    owner_of, why_of = attribute(trees)

    me = os.getpid()
    matches = sorted(p for p, c in comm_of.items() if c == args.pattern and p != me)
    if not matches:
        print(f"no live process named {args.pattern!r}")
        return 0

    def argv_of(pid: int) -> str:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                return " ".join(fh.read().decode("utf-8", "replace").split("\0")).strip()
        except OSError:
            return "(gone)"

    ours, theirs, unattributed = classify(matches, owner_of, mine)

    print(f"worktree: {mine.path}")
    print(f"{len(matches)} process(es) named {args.pattern!r}\n")

    for pid in ours:
        print(f"  MINE          pid {pid}  {argv_of(pid)}")
    for pid, t in theirs:
        print(f"  ANOTHER LANE  pid {pid}  [{os.path.basename(t.path)}]  {argv_of(pid)}")
    for pid in unattributed:
        print(f"  UNDECIDABLE   pid {pid}  {argv_of(pid)}")
        print(f"                {why_of.get(pid, 'no reference into any worktree')}")

    if theirs or unattributed:
        print()
        if theirs:
            print(f"  {len(theirs)} belong(s) to another worktree. Killing them takes a run that is")
            print("  probably mid-measurement, and the cost lands on a lane that cannot see why.")
        if unattributed:
            print(f"  {len(unattributed)} could not be attributed to any worktree. Undecidable is")
            print("  not the same as yours -- these are refused too.")

    if not args.yes:
        print(f"\n[census] nothing signalled. Re-run with --yes to signal the {len(ours)} in your tree.")
        return 0

    targets = list(ours)
    if args.any_tree:
        print("\n[--any-tree] overriding the ownership guard and signalling other lanes' matches.")
        targets += [p for p, _ in theirs] + unattributed

    sent, vanished, denied = 0, 0, 0
    for pid in targets:
        try:
            os.kill(pid, sig)
            sent += 1
        except ProcessLookupError:
            vanished += 1
        except PermissionError:
            print(f"  cannot signal pid {pid}: permission denied", file=sys.stderr)
            denied += 1

    print(f"\nsignalled {sent} with SIG{args.signal.upper().removeprefix('SIG')}"
          + (f", {vanished} had already exited" if vanished else ""))
    refused = len(theirs) + len(unattributed) if not args.any_tree else 0
    if refused:
        print(f"refused {refused} outside your tree")
    if denied:
        print(f"could not signal {denied} (permission denied)")
    return 1 if (refused or denied) else 0


if __name__ == "__main__":
    raise SystemExit(main())
