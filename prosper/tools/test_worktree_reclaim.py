#!/usr/bin/env python3
"""Self-checking tests for worktree_reclaim.py (exit code is truth).

The assertion that matters here is NOT "it removed the safe tree" -- a tool that removes
everything passes that. It is that each must-not-remove tree is refused, AND that the refusal
names the guard that fired. So every mutation arm below pins the specific blocker, not merely
that removal did not happen: a tree spared for the wrong reason is a tool that will delete
someone's work as soon as that accidental reason stops holding.

Each arm is built BY HAND from git primitives rather than from the tool's own view of the world,
because a fixture derived from the thing under test inherits its blind spots.

Run directly, or via ctest as worktree_reclaim_tool.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import worktree_reclaim as W  # noqa: E402

FAILURES: list[str] = []
OLD = 400 * 3600  # comfortably past the default 12 h idle threshold


def check(name: str, got, want, extra: str = "") -> None:
    if got == want:
        print(f"  ok   {name}")
    else:
        msg = f"{name}: got {got!r}, want {want!r}"
        if extra:
            msg += f" ({extra})"
        print(f"  FAIL {msg}")
        FAILURES.append(msg)


def git(repo: str | Path, *args: str, check_rc: bool = True) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check_rc and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout


def age(path: str | Path, admin: str | Path | None, seconds: int = OLD) -> None:
    """Backdate everything probe_idle looks at, so a fixture reads as idle."""
    when = time.time() - seconds
    targets = [Path(path)] + list(Path(path).iterdir())
    if admin and Path(admin).is_dir():
        targets += list(Path(admin).iterdir())
    for t in targets:
        try:
            os.utime(t, (when, when), follow_symlinks=False)
        except (OSError, NotImplementedError):
            pass


def spawn_holder(cwd: Path) -> subprocess.Popen:
    """A live process whose cwd is `cwd`, without depending on an external `sleep` binary."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=str(cwd))


def holder_scan_supported() -> bool:
    return W.scan_processes(scan_fds=False, scan_maps=False)[1]


def admin_of(repo: Path, name: str) -> Path:
    return repo / ".git" / "worktrees" / name


def build_repo(root: Path) -> Path:
    """A repo whose `origin/master` ref exists without needing a real remote."""
    repo = root / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "master")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "t")
    (repo / "file.txt").write_text("base\n")
    git(repo, "add", "file.txt")
    git(repo, "commit", "-qm", "base")
    head = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "update-ref", "refs/remotes/origin/master", head)
    return repo


def add_wt(repo: Path, name: str, branch: str | None = None) -> Path:
    path = repo.parent / name
    args = ["worktree", "add", "-q"]
    if branch:
        args += ["-b", branch]
    git(repo, *args, str(path), "origin/master")
    return path


def survey(repo: Path, **kw):
    opts = dict(base="origin/master", min_idle_hours=12.0, jobs=2, use_github=False,
                gh_limit=0, scan_fds=False, scan_maps=False, with_size=False)
    opts.update(kw)
    trees, meta = W.survey(repo=str(repo), **opts)
    return {t.name: t for t in trees}, meta


# --------------------------------------------------------------------------- arms


def test_positive_and_mutations() -> None:
    """One removable tree, and one mutation per guard that must refuse it."""
    print("\n[positive control + per-guard mutation arms]")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        repo = build_repo(root)

        safe = add_wt(repo, "safe", "feat/safe")
        dirty = add_wt(repo, "dirty", "feat/dirty")
        untracked = add_wt(repo, "untracked", "feat/untracked")
        unmerged = add_wt(repo, "unmerged", "feat/unmerged")
        inuse = add_wt(repo, "inuse", "feat/inuse")
        recent = add_wt(repo, "recent", "feat/recent")
        locked = add_wt(repo, "locked", "feat/locked")

        # --- mutations, each aimed at exactly one guard -------------------------------
        (dirty / "file.txt").write_text("modified\n")           # dirty: tracked change
        (untracked / "scratch.log").write_text("x\n")           # dirty: untracked only
        (unmerged / "new.txt").write_text("n\n")                # unmerged: real new commit
        git(unmerged, "add", "new.txt")
        git(unmerged, "commit", "-qm", "unmerged work")
        git(repo, "worktree", "lock", str(locked))              # locked

        for name in ("safe", "dirty", "untracked", "unmerged", "inuse", "locked"):
            age(root / name, admin_of(repo, name))              # `recent` is deliberately not aged

        holder = spawn_holder(inuse)  # in-use: a live cwd
        try:
            time.sleep(0.4)
            trees, _ = survey(repo)

            # Positive control: the arm that SHOULD be removable.
            check("safe is REMOVABLE", trees["safe"].state, "REMOVABLE")
            check("safe merged via ancestor", trees["safe"].merged_by, "ancestor")

            # Mutation arms: refused, and refused for the RIGHT reason.
            check("dirty refused", trees["dirty"].removable, False)
            check("dirty names guard", "dirty" in trees["dirty"].blockers, True,
                  str(trees["dirty"].blockers))
            check("untracked-only refused", trees["untracked"].removable, False)
            check("untracked names guard", "dirty" in trees["untracked"].blockers, True,
                  str(trees["untracked"].blockers))
            check("untracked counted as untracked", trees["untracked"].dirty_untracked, 1)
            # These arms run with use_github=False, so the squash-merge cross-check never ran and
            # the honest verdict is "could not tell", not "this is live work". Both refuse; the
            # label is the whole point. test_merge_evidence_labels pins the other branch.
            check("unmerged refused", trees["unmerged"].removable, False)
            check("unmerged names guard", "no-merge-evidence" in trees["unmerged"].blockers, True,
                  str(trees["unmerged"].blockers))
            check("in-use refused", trees["inuse"].removable, False)
            if holder_scan_supported():
                check("in-use names guard", "in-use" in trees["inuse"].blockers, True,
                      str(trees["inuse"].blockers))
                check("in-use identifies the pid",
                      holder.pid in [h.pid for h in trees["inuse"].holders], True)
            else:
                # No /proc here, so the guard cannot fire on its own merits. The tool must still
                # refuse -- via fail-closed -- and test_fails_closed_without_a_process_scan pins
                # that. Passing this arm silently would be the exact defect it exists to catch.
                print("  SKIP in-use guard needs a readable /proc; fail-closed arm covers this")
                check("in-use refused via fail-closed",
                      "holder-scan-unavailable" in trees["inuse"].blockers, True,
                      str(trees["inuse"].blockers))
            check("recent refused", trees["recent"].removable, False)
            check("recent names guard", "recent" in trees["recent"].blockers, True,
                  str(trees["recent"].blockers))
            check("locked refused", trees["locked"].removable, False)
            check("locked names guard", "locked" in trees["locked"].blockers, True,
                  str(trees["locked"].blockers))
            check("main refused", trees[repo.name].removable, False)
            check("main names guard", trees[repo.name].blockers, ["main"])
        finally:
            holder.kill()
            holder.wait()
            git(repo, "worktree", "unlock", str(locked), check_rc=False)


def test_merge_evidence_labels() -> None:
    """"Not merged" and "could not tell" must not share a label.

    The ancestor test is blind to squash merges, which is how this repository merges every PR. So
    without the GitHub cross-check a non-ancestor tree might be live work OR work that landed via
    a squash, and reporting a `gh` outage as a wall of live branches would be a confident wrong
    answer. Neither is removable; only the wording differs, and the wording is what a reader acts on.
    """
    print("\n[merge evidence: unmerged vs could-not-tell]")
    unproven = W.Worktree(path="/x", real="/x", merged_by=None, idle_hours=999.0)
    W.classify(unproven, 12.0, merge_evidence_available=True)
    check("complete evidence says unmerged", unproven.blockers, ["unmerged"])

    unknown = W.Worktree(path="/x", real="/x", merged_by=None, idle_hours=999.0)
    W.classify(unknown, 12.0, merge_evidence_available=False)
    check("incomplete evidence says no-merge-evidence", unknown.blockers, ["no-merge-evidence"])

    proven = W.Worktree(path="/x", real="/x", merged_by="squash-pr#1", idle_hours=999.0)
    W.classify(proven, 12.0, merge_evidence_available=True)
    check("proven merged has no blockers", proven.blockers, [])
    check("proven merged is removable", proven.removable, True)


def test_fails_closed_without_a_process_scan() -> None:
    """No readable /proc must mean "cannot tell", never "nobody is here".

    macOS and Windows have no procfs, so `scan_processes` finds nothing there. Read as "no
    holders", that turns the in-use guard off silently and a worktree with a live shell in it
    classifies as REMOVABLE -- the single most dangerous answer this tool can give, and one that
    would only ever be observed on the machine where it deleted someone's session. So the scan
    reports whether it could run at all, and an absent scan blocks every tree.

    CI caught this: the first revision passed on Linux and failed on the macOS and Windows jobs.
    """
    print("\n[fail closed when the process scan cannot run]")
    ok = W.Worktree(path="/x", real="/x", merged_by="ancestor", idle_hours=999.0)
    W.classify(ok, 12.0, merge_evidence_available=True, holder_scan_ok=True)
    check("scan available -> removable", ok.removable, True)

    blind = W.Worktree(path="/x", real="/x", merged_by="ancestor", idle_hours=999.0)
    W.classify(blind, 12.0, merge_evidence_available=True, holder_scan_ok=False)
    check("scan unavailable -> refused", blind.removable, False)
    check("scan unavailable names guard", blind.blockers, ["holder-scan-unavailable"])

    # The flag must reflect reality on THIS platform, not merely be plumbed through.
    _, supported = W.scan_processes(scan_fds=False, scan_maps=False)
    check("scan support matches the platform", supported, os.path.isdir("/proc"),
          f"platform={sys.platform}")


def test_nested_worktree_is_refused() -> None:
    """A worktree registered underneath another must not be removed out from under it."""
    print("\n[nested worktree]")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        repo = build_repo(root)
        outer = add_wt(repo, "outer", "feat/outer")
        inner = outer / "inner"
        git(repo, "worktree", "add", "-q", "-b", "feat/inner", str(inner), "origin/master")
        age(outer, admin_of(repo, "outer"))
        age(inner, admin_of(repo, "inner"))

        trees, _ = survey(repo)
        check("outer refused", trees["outer"].removable, False)
        check("outer names guard", "nested" in trees["outer"].blockers, True,
              str(trees["outer"].blockers))
        check("inner still removable", trees["inner"].state, "REMOVABLE")


def test_holder_attributed_to_most_specific_tree() -> None:
    """A process inside a nested tree must not be blamed on its parent, or vice versa."""
    print("\n[holder attribution]")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        repo = build_repo(root)
        outer = add_wt(repo, "outer", "feat/outer")
        inner = outer / "inner"
        git(repo, "worktree", "add", "-q", "-b", "feat/inner", str(inner), "origin/master")

        holder = spawn_holder(inner)
        try:
            time.sleep(0.4)
            trees, _ = survey(repo)
            if not holder_scan_supported():
                print("  SKIP holder attribution needs a readable /proc")
                return
            check("inner sees the holder", holder.pid in [h.pid for h in trees["inner"].holders],
                  True)
            check("outer does NOT see it", holder.pid in [h.pid for h in trees["outer"].holders],
                  False)
        finally:
            holder.kill()
            holder.wait()


def test_idle_is_not_measuring_our_own_footprint() -> None:
    """Regression: examining a tree must not make it look freshly touched.

    Two independent ways the first draft got this wrong, both of which reported every one of 278
    real worktrees as "0.00h idle" -- a result that reads as a busy machine rather than as a
    broken gauge, which is what makes it dangerous:

      1. it stat()ed the admin DIRECTORY, whose mtime `git status` bumps by creating and
         unlinking a lock entry inside it;
      2. it ran a plain `git status`, which REWRITES the index when the stat cache is stale,
         setting the mtime of a file idle legitimately reads.

    Fixes: read only the files inside the admin dir, and pass `--no-optional-locks`. This arm
    drives the tool's OWN probe rather than a hand-rolled git call, so it tests the thing that
    actually runs. Without both fixes it reads ~0 h instead of ~400 h.
    """
    print("\n[instrument: idle must not date our own footprints]")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        repo = build_repo(root)
        wt = add_wt(repo, "aged", "feat/aged")
        admin = admin_of(repo, "aged")
        # Backdating leaves the stat cache stale, which is exactly the condition under which a
        # plain `git status` rewrites the index. Without that the arm could not discriminate.
        age(wt, admin)

        before = W.Worktree(path=str(wt), real=os.path.realpath(wt))
        before.admin_dir = W.worktree_admin_dir(before)
        W.probe_idle(before)

        after = W.Worktree(path=str(wt), real=os.path.realpath(wt))
        after.admin_dir = W.worktree_admin_dir(after)
        W.probe_status(after)  # the tool's own probe, in the order that used to break
        admin_dir_mtime_now = time.time() - os.stat(admin).st_mtime
        W.probe_idle(after)
        check("status probe saw a clean tree", (after.dirty_tracked, after.dirty_untracked), (0, 0))

        # Only meaningful if git really did touch the directory on this filesystem; if it did
        # not, the arm cannot discriminate and says so rather than passing vacuously.
        if admin_dir_mtime_now > 60:
            print("  SKIP git status did not bump the admin dir mtime here; arm cannot discriminate")
            return
        check("idle survives our own status probe", after.idle_hours > 100, True,
              f"idle={after.idle_hours:.2f}h, admin dir looks {admin_dir_mtime_now:.0f}s old")
        check("idle unchanged by the probe", abs(after.idle_hours - before.idle_hours) < 1.0, True)


def test_dry_run_is_the_default_and_removes_nothing() -> None:
    """Default and --remove-without--yes must both leave every tree on disk."""
    print("\n[dry-run gating]")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        repo = build_repo(root)
        safe = add_wt(repo, "safe", "feat/safe")
        age(safe, admin_of(repo, "safe"))

        rc = W.main(["--repo", str(repo), "--no-github", "--no-fds", "--no-maps"])
        check("default exits 0", rc, 0)
        check("default kept the tree", safe.is_dir(), True)

        rc = W.main(["--repo", str(repo), "--no-github", "--no-fds", "--no-maps", "--remove"])
        check("--remove without --yes exits 2", rc, 2)
        check("--remove without --yes kept the tree", safe.is_dir(), True)


def test_remove_takes_only_the_safe_tree() -> None:
    """End to end: the safe tree goes, every mutation arm stays."""
    print("\n[--remove --yes end to end]")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        repo = build_repo(root)

        safe = add_wt(repo, "safe", "feat/safe")
        dirty = add_wt(repo, "dirty", "feat/dirty")
        unmerged = add_wt(repo, "unmerged", "feat/unmerged")
        recent = add_wt(repo, "recent", "feat/recent")

        (dirty / "file.txt").write_text("modified\n")
        (unmerged / "new.txt").write_text("n\n")
        git(unmerged, "add", "new.txt")
        git(unmerged, "commit", "-qm", "unmerged work")
        for name in ("safe", "dirty", "unmerged"):
            age(root / name, admin_of(repo, name))

        rc = W.main(["--repo", str(repo), "--no-github", "--no-fds", "--no-maps",
                     "--remove", "--yes"])
        check("exits 0", rc, 0)
        check("safe tree removed", safe.is_dir(), False)
        check("dirty tree kept", dirty.is_dir(), True)
        check("unmerged tree kept", unmerged.is_dir(), True)
        check("recent tree kept", recent.is_dir(), True)
        # The branch ref must survive: removal reclaims a directory, never history.
        check("branch ref kept", git(repo, "rev-parse", "--verify", "feat/safe").strip() != "", True)


def test_removal_rechecks_state_rather_than_trusting_the_census() -> None:
    """A tree that goes dirty after the census must still be refused at removal time."""
    print("\n[recheck beats a stale census]")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        repo = build_repo(root)
        wt = add_wt(repo, "racy", "feat/racy")
        age(wt, admin_of(repo, "racy"))

        trees, _ = survey(repo)
        check("census says removable", trees["racy"].state, "REMOVABLE")

        # Simulate the race the tool exists to survive: work appears after the scan.
        (wt / "file.txt").write_text("someone is mid-edit\n")

        removed = W.recheck_and_remove(str(repo), trees["racy"], "origin/master", 12.0,
                                       False, False, False, 0, False)
        check("recheck refuses", removed, False)
        check("tree still on disk", wt.is_dir(), True)


def main() -> int:
    for fn in (
        test_positive_and_mutations,
        test_merge_evidence_labels,
        test_fails_closed_without_a_process_scan,
        test_nested_worktree_is_refused,
        test_holder_attributed_to_most_specific_tree,
        test_idle_is_not_measuring_our_own_footprint,
        test_dry_run_is_the_default_and_removes_nothing,
        test_remove_takes_only_the_safe_tree,
        test_removal_rechecks_state_rather_than_trusting_the_census,
    ):
        fn()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all worktree_reclaim checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
