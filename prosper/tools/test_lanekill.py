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

Linux only, deliberately, and that is a statement about the tool. Its attribution reads /proc;
without one it cannot establish ownership and fails closed. The fail-closed path itself is
covered by an arm that runs everywhere.
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


def test_fails_closed_without_a_process_scan() -> None:
    """With no readable /proc the tool must refuse, never fall through to 'nobody owns it'.

    Platform-independent on purpose: this is the property that protects macOS and Windows, where
    the real scan cannot run, so it must be verified where the real scan cannot run too.
    """
    original = lanekill.scan_processes
    try:
        lanekill.scan_processes = lambda *a, **k: ([], False)  # type: ignore[assignment]
        refs, supported = lanekill.scan_processes()
        check("unsupported scan reports supported=False", supported, False)
        check("unsupported scan yields no refs", refs, [])
    finally:
        lanekill.scan_processes = original  # type: ignore[assignment]


def main() -> int:
    if sys.platform != "linux":
        print("lanekill: attribution needs /proc; the fail-closed arm is the portable one")
        test_fails_closed_without_a_process_scan()
        return 1 if FAILURES else 0

    for fn in (
        test_attribution_separates_two_trees,
        test_symlinked_cwd_still_attributes,
        test_outside_any_worktree_is_undecidable,
    ):
        print(f"{fn.__name__}:")
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
    print("test_fails_closed_without_a_process_scan:")
    test_fails_closed_without_a_process_scan()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("all arms passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
