#!/usr/bin/env python3
"""Tests for lanekill.py.

lanekill SIGNALS PROCESSES, so like its sibling `test_worktree_reclaim.py` these are mutation
arms first: each builds a situation where a process MUST NOT be signalled and asserts both the
refusal and which guard produced it. The positive arm exists to prove the machinery fires at
all -- without it every refusal below would pass on a tool that never selects anything, which is
the "clean zero from a broken instrument" failure this repo's trap table is full of.

`test_symlinked_cwd_still_attributes` holds the two arms that discriminate, and they
discriminate against DIFFERENT wrong implementations -- which was established by mutation, not
assumed:

* the **prefix-sharing sibling** arm reddens against `realpath() + startswith()`, because
  `wt-real-extra` shares a prefix with `wt-real` and a prefix test claims it. Measured: mutating
  the tool to that implementation fails exactly this arm and no other.
* the **symlinked path** arm reddens against a bare string compare with no `realpath()` at all.
  It does NOT redden against `realpath() + startswith()`, because realpath resolves a symlink
  before the comparison. Stated explicitly so nobody reads a green symlink arm as proof that
  string matching was ruled out -- it rules out one string implementation, not the family.

Neither arm reaches the case that motivated inode identity in the first place: inside the ps5ys
distrobox `/home` is a real BIND MOUNT rather than a symlink, so the same directory has two
spellings that `realpath()` cannot reconcile. Constructing a bind mount needs privileges a test
should not take, so that case is covered by `worktree_reclaim.py`'s live measurement (dev=57
ino=10784453 reached by both spellings) and not by an arm here. The two arms below are the
cheap approximations of it; the docstring in the tool carries the real evidence.

Linux only, deliberately, and that is a statement about the tool: attribution reads /proc.

**The fail-closed arm is Linux-only too, which is a deliberate narrowing.** It used to run
everywhere by installing a stub and asserting the stub's own return values — which said nothing
about lanekill and stayed green when a reviewer deleted the gate. It now calls `main()` against a
live process, which needs a real process, so it cannot run where the rest cannot.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lanekill  # noqa: E402
from worktree_reclaim import list_worktrees  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want, extra: str = "") -> None:
    if got == want:
        print(f"  ok   {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}: got {got!r}, want {want!r} {extra}")


def git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr}")
    return p.stdout


def build_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "master", str(repo)], check=True)
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("x\n")
    git(repo, "add", "f.txt")
    git(repo, "commit", "-qm", "init")
    return repo


def test_attribution_separates_two_trees(root: Path) -> None:
    """Positive arm + the cross-tree refusal, in one situation.

    Two worktrees, a live process sitting in each. The tool must claim exactly the one in the
    tree it is run from and refuse the other. If attribution were broken in the permissive
    direction both would land in `ours`, which is the bug that cost a lane its measurement.
    """
    repo = build_repo(root)
    a = root / "wt-a"
    b = root / "wt-b"
    git(repo, "worktree", "add", "-q", "-b", "lane-a", str(a))
    git(repo, "worktree", "add", "-q", "-b", "lane-b", str(b))

    procs = [subprocess.Popen(["sleep", "60"], cwd=str(d)) for d in (a, b)]
    time.sleep(0.3)
    try:
        trees = list_worktrees(str(repo))
        mine = lanekill.my_worktree(trees, str(a))
        check("run from wt-a attributes to wt-a", mine.path if mine else None, str(a))

        other = lanekill.my_worktree(trees, str(b))
        check("run from wt-b attributes to wt-b", other.path if other else None, str(b))
        check("the two trees are distinct", mine.real != other.real, True)
    finally:
        for p in procs:
            p.kill()
            p.wait()


def test_symlinked_cwd_still_attributes(root: Path) -> None:
    """Two spellings must attribute alike, and a lookalike must not attribute at all.

    See the module docstring for which arm rules out which wrong implementation -- they are not
    interchangeable, and the symlink half is the weaker of the two.
    """
    repo = build_repo(root)
    wt = root / "wt-real"
    git(repo, "worktree", "add", "-q", "-b", "lane-s", str(wt))

    alias = root / "wt-alias"
    os.symlink(str(wt), str(alias))

    trees = list_worktrees(str(repo))
    via_real = lanekill.my_worktree(trees, str(wt))
    via_link = lanekill.my_worktree(trees, str(alias))
    check("attributes via the real path", via_real.path if via_real else None, str(wt))
    check("attributes via a symlinked path", via_link.path if via_link else None, str(wt))

    # And the negative direction: a sibling directory that merely SHARES A PREFIX with the
    # worktree name must not attribute. `wt-real-extra` starts with `wt-real`, so a naive
    # `startswith` would claim it.
    decoy = root / "wt-real-extra"
    decoy.mkdir()
    check("a prefix-sharing sibling does NOT attribute",
          lanekill.my_worktree(trees, str(decoy)), None,
          "-- a startswith() implementation would claim this directory")


def test_outside_any_worktree_is_undecidable(root: Path) -> None:
    """Ownership must be undecidable, not assumed, outside every tree."""
    repo = build_repo(root)
    wt = root / "wt-o"
    git(repo, "worktree", "add", "-q", "-b", "lane-o", str(wt))
    trees = list_worktrees(str(repo))
    outside = root / "not-a-tree"
    outside.mkdir()
    check("a path in no worktree attributes to nothing",
          lanekill.my_worktree(trees, str(outside)), None)


def test_fails_closed_without_a_process_scan(root: Path) -> None:
    """With no readable /proc the tool must refuse, and a LIVE MATCH IN MY OWN TREE must survive.

    The first version of this arm installed a lambda returning `([], False)` and then asserted the
    lambda returned `([], False)`. That is a statement about the lambda. A reviewer deleted the
    fail-closed gate from the tool and this arm stayed green, which is exactly the shape of test
    this repository keeps paying for -- an assertion whose subject was never the code under test.

    So: run `main()` for real with the scan forced unsupported, against a process that IS in this
    worktree and WOULD be signalled if the gate were gone.

    Precisely: **`rc == 2` is the discriminator**, not the survivor. With the scan stubbed the
    victim cannot be reached whatever the tool does, so its survival is guaranteed by the stub
    rather than earned by the gate — an earlier version of this docstring said "the survivor is
    the assertion", which is the same overclaim this file exists to warn about. The live process
    is here so the arm exercises the real matching path up to the refusal, not because its
    survival proves anything.
    """
    repo = build_repo(root)
    wt = root / "wt-fc"
    git(repo, "worktree", "add", "-q", "-b", "fc", str(wt))
    victim = subprocess.Popen(["sleep", "120"], cwd=str(wt))
    time.sleep(0.4)

    original_scan = lanekill.scan_processes
    original_argv = sys.argv[:]
    cwd0 = os.getcwd()
    try:
        lanekill.scan_processes = lambda *a, **k: ([], False)  # type: ignore[assignment]
        os.chdir(str(wt))
        sys.argv = ["lanekill.py", "sleep", "--yes"]
        rc = lanekill.main()
        time.sleep(0.3)
        check("refuses with exit 2 when /proc is unreadable", rc, 2)
        check("a live match in MY OWN tree survives the refusal", victim.poll(), None)
    finally:
        os.chdir(cwd0)
        sys.argv = original_argv
        lanekill.scan_processes = original_scan  # type: ignore[assignment]
        if victim.poll() is None:
            victim.kill()
        victim.wait()


# --------------------------------------------------------------------------- the decision itself
#
# Everything above tests ATTRIBUTION. A reviewer demonstrated that was not enough: mutating the
# tool six ways -- each turning it into a cross-lane killer -- left every arm green, because every
# arm called only `my_worktree()`. Deleting both live processes from the "positive" arm also left
# it green. These arms exercise the code that decides to send a signal.


class _T:
    """Minimal stand-in for Worktree: `classify` and `attribute` use `.real` and `.holders`."""

    def __init__(self, real, holders=()):
        self.real = real
        self.path = real
        self.holders = list(holders)


class _H:
    def __init__(self, pid, kind):
        self.pid = pid
        self.kind = kind
        self.comm = "x"
        self.path = "/x"


def test_classify_never_claims_another_tree() -> None:
    """The mutation `t.real == mine.real` -> `True` must redden here."""
    mine, other = _T("/w/a"), _T("/w/b")
    ours, theirs, undec = lanekill.classify([1, 2, 3], {1: mine, 2: other}, mine)
    check("only my tree's pid is ours", ours, [1])
    check("another tree's pid is theirs", [p for p, _ in theirs], [2])
    check("an unowned pid is undecidable", undec, [3])
    check("an unowned pid is NOT ours", 3 in ours, False)
    check("another tree's pid is NOT ours", 2 in ours, False)


def test_disagreeing_strong_evidence_is_undecidable() -> None:
    """cwd in one tree and exe in another: refuse, never pick by registration order.

    The first version resolved this by whichever holder was visited last, so the verdict followed
    `git worktree list` order and the same situation attributed both ways.
    """
    a = _T("/w/a", [_H(10, "cwd")])
    b = _T("/w/b", [_H(10, "exe")])
    owner, why = lanekill.attribute([a, b])
    check("a split cwd/exe pid is attributed to nobody", 10 in owner, False)
    check("and the refusal says why", "disagree" in why.get(10, ""), True)

    # order must not change the answer -- this is the actual defect, so assert it directly
    owner_rev, _ = lanekill.attribute([b, a])
    check("reversing worktree order gives the same answer", 10 in owner_rev, False)


def test_weak_evidence_is_not_ownership() -> None:
    """An open fd or a mapping into a tree says the process READS there, not that it is yours."""
    a = _T("/w/a", [_H(20, "fd")])
    owner, why = lanekill.attribute([a])
    check("an fd-only pid is attributed to nobody", 20 in owner, False)
    check("and the refusal names the reason", "ownership" in why.get(20, ""), True)

    b = _T("/w/b", [_H(21, "cwd"), _H(21, "map")])
    owner2, _ = lanekill.attribute([b])
    check("cwd still attributes even with a weak ref alongside", owner2[21].real, "/w/b")


def test_end_to_end_spares_the_other_lane(root: Path) -> None:
    """The whole tool, --yes, two live processes: exactly one dies.

    This is the arm the six mutations were invisible to. It runs lanekill as a subprocess so the
    census/`--yes` gate, the fail-closed gate and the exit code are all in the path.
    """
    repo = build_repo(root)
    a, b = root / "wt-a", root / "wt-b"
    git(repo, "worktree", "add", "-q", "-b", "e2e-a", str(a))
    git(repo, "worktree", "add", "-q", "-b", "e2e-b", str(b))

    mine = subprocess.Popen(["sleep", "120"], cwd=str(a))
    other = subprocess.Popen(["sleep", "120"], cwd=str(b))
    time.sleep(0.4)
    try:
        # census first: must kill NOTHING even though a match is ours
        cen = subprocess.run([sys.executable, str(HERE / "lanekill.py"), "sleep"],
                             cwd=str(a), capture_output=True, text=True)
        time.sleep(0.3)
        check("census exits 0", cen.returncode, 0)
        check("census kills nothing (mine alive)", mine.poll(), None)
        check("census kills nothing (theirs alive)", other.poll(), None)

        # A LOOKALIKE IN MY OWN TREE. `lanekill.py:main` matches `comm == pattern`; changing that
        # to `pattern in comm` reintroduces exactly the `pkill -f` over-match this tool contrasts
        # itself against, and the ownership filter would NOT save it -- the blast radius is my own
        # worktree, where everything is fair game. Nothing else in the suite notices.
        import shutil
        lookalike_bin = a / "sleepx"
        shutil.copy("/bin/sleep", str(lookalike_bin))
        lookalike = subprocess.Popen([str(lookalike_bin), "120"], cwd=str(a))
        time.sleep(0.4)

        run = subprocess.run([sys.executable, str(HERE / "lanekill.py"), "sleep", "--yes"],
                             cwd=str(a), capture_output=True, text=True)
        time.sleep(0.5)
        check("--yes returns 1 having refused one", run.returncode, 1)
        check("MY process was signalled", mine.poll() is not None, True)
        check("THE OTHER LANE'S process survived", other.poll(), None)
        check("output names the other lane", "ANOTHER LANE" in run.stdout, True)
        check("a LOOKALIKE name in my own tree is untouched", lookalike.poll(), None)
    finally:
        for pr in (mine, other, lookalike):
            if pr.poll() is None:
                pr.kill()
            pr.wait()


def test_nested_worktree_topology(root: Path) -> None:
    """A worktree INSIDE the main checkout -- which is what `.claude/worktrees/*` actually are.

    Every other arm builds siblings, so `my_worktree()` taking the OUTERMOST match instead of the
    innermost passed the whole suite. The direct effect of that is over-refusal rather than
    over-killing, but it is the topology every real invocation runs in and nothing saw it.
    """
    repo = build_repo(root)
    nest = repo / "inner"
    git(repo, "worktree", "add", "-q", "-b", "inner", str(nest))

    trees = list_worktrees(str(repo))
    inner = lanekill.my_worktree(trees, str(nest))
    outer = lanekill.my_worktree(trees, str(repo))
    check("a nested tree resolves to ITSELF, not the enclosing checkout",
          inner.real if inner else None, os.path.realpath(str(nest)))
    check("the enclosing checkout resolves to itself",
          outer.real if outer else None, os.path.realpath(str(repo)))
    check("they are different trees", (inner.real != outer.real) if inner and outer else False, True)


def test_any_tree_sweeps_undecidable_too(root: Path) -> None:
    """`--any-tree` must sweep the UNDECIDABLE matches, not only other trees' -- through main().

    My first attempt at this arm rebuilt the sweep from `classify()` output and asserted the
    reconstruction covered every pid. That passed while `main()` dropped `unattributed` from its
    target list, because the reconstruction was not the code. Same mistake as the one this whole
    review was about, one level down: an assertion whose subject is a copy of the logic.

    So this builds a genuinely undecidable process -- binary in one tree, cwd in another, which is
    the ambiguity `attribute()` refuses -- and drives the real tool twice.
    """
    import shutil

    repo = build_repo(root)
    a, b = root / "wt-a", root / "wt-b"
    git(repo, "worktree", "add", "-q", "-b", "at-a", str(a))
    git(repo, "worktree", "add", "-q", "-b", "at-b", str(b))

    binary = b / "psplit"                      # exe lives in wt-b ...
    shutil.copy("/bin/sleep", str(binary))
    proc = subprocess.Popen([str(binary), "120"], cwd=str(a))   # ... cwd is wt-a
    time.sleep(0.4)
    try:
        refuse = subprocess.run([sys.executable, str(HERE / "lanekill.py"), "psplit", "--yes"],
                                cwd=str(a), capture_output=True, text=True)
        time.sleep(0.3)
        check("a split cwd/exe process is refused by --yes alone", refuse.returncode, 1)
        check("and it is still running", proc.poll(), None)
        check("the refusal names both trees", ("at-a" in refuse.stdout or "wt-a" in refuse.stdout)
              and ("at-b" in refuse.stdout or "wt-b" in refuse.stdout), True)

        sweep = subprocess.run(
            [sys.executable, str(HERE / "lanekill.py"), "psplit", "--yes", "--any-tree"],
            cwd=str(a), capture_output=True, text=True)
        time.sleep(0.5)
        check("--any-tree signals the undecidable match", proc.poll() is not None, True)
        check("--any-tree exits 0 having refused nothing", sweep.returncode, 0)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def main() -> int:
    if sys.platform != "linux":
        print("lanekill: attribution needs /proc, and the fail-closed arm now needs a live")
        print("process to prove the refusal spared it -- so the whole suite is Linux-only.")
        return 0

    for fn in (
        test_attribution_separates_two_trees,
        test_symlinked_cwd_still_attributes,
        test_outside_any_worktree_is_undecidable,
        test_end_to_end_spares_the_other_lane,
        test_fails_closed_without_a_process_scan,
        test_nested_worktree_topology,
        test_any_tree_sweeps_undecidable_too,
    ):
        print(f"{fn.__name__}:")
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
    for fn in (test_classify_never_claims_another_tree,
               test_disagreeing_strong_evidence_is_undecidable,
               test_weak_evidence_is_not_ownership):
        print(f"{fn.__name__}:")
        fn()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("all arms passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
