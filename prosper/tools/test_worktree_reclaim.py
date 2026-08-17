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

import json
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
    W.classify(unproven, 12.0, merge_evidence_available=True, holder_scan_ok=True)
    check("complete evidence says unmerged", unproven.blockers, ["unmerged"])

    unknown = W.Worktree(path="/x", real="/x", merged_by=None, idle_hours=999.0)
    W.classify(unknown, 12.0, merge_evidence_available=False, holder_scan_ok=True)
    check("incomplete evidence says no-merge-evidence", unknown.blockers, ["no-merge-evidence"])

    proven = W.Worktree(path="/x", real="/x", merged_by="squash-pr#1", idle_hours=999.0)
    W.classify(proven, 12.0, merge_evidence_available=True, holder_scan_ok=True)
    check("proven merged has no blockers", proven.blockers, [])
    check("proven merged is removable", proven.removable, True)


def test_github_merge_evidence() -> None:
    """The squash-merge cross-check, which is the SOLE merge evidence for most real candidates.

    On the live repo this path qualified 104 of 146 removal candidates while `ancestor` qualified
    41 -- so the majority of this tool's delete decisions rest here. Every other arm in this file
    runs with use_github=False, which left `probe_github`, `_apply_github`, the recycled-branch
    guard and `_GH_CACHE` at zero coverage: reverting the open-PR fail-closed fix kept all 54 arms
    green. A path that decides deletions cannot be the untested one.

    `W.run` is monkeypatched rather than calling a real `gh`, so these arms are hermetic and
    exercise the failure branches on demand.
    """
    print("\n[github merge evidence]")
    W._GH_CACHE.clear()

    def wt(name: str, head: str, branch: str | None) -> W.Worktree:
        t = W.Worktree(path=f"/{name}", real=f"/{name}", head=head, branch=branch,
                       detached=branch is None)
        return t

    # --- _apply_github: qualification, the recycled-branch guard, and detached trees -------
    merged_head = wt("merged", "a" * 40, "feat/landed")
    recycled = wt("recycled", "b" * 40, "feat/recycled")
    detached = wt("detached", "c" * 40, None)
    untouched = wt("other", "d" * 40, "feat/live")
    trees = [merged_head, recycled, detached, untouched]
    by_oid = {"a" * 40: 111, "b" * 40: 222, "c" * 40: 333}
    W._apply_github(trees, by_oid, open_branches={"feat/recycled"})

    check("sha match qualifies", merged_head.merged_by, "squash-pr#111")
    check("recycled branch name is refused", recycled.merged_by, None)
    check("detached qualifies on sha alone", detached.merged_by, "squash-pr#333")
    check("unmatched sha stays unqualified", untouched.merged_by, None)

    # A branch whose tip has MOVED past the merged PR is live work, not a stale tree. The branch
    # name here MUST be the merged PR's own `feat/landed`: with any other name this arm asserts
    # nothing that `untouched` above does not already cover, because the sha simply is not in
    # by_oid either way. Sharing the name is what makes it the one arm that fails if qualification
    # were ever keyed on branch name instead of sha -- which is the plausible refactor, since the
    # recycled-branch guard right beside it legitimately does key on the name.
    moved = wt("moved", "e" * 40, "feat/landed")
    W._apply_github([moved], {"a" * 40: 111}, open_branches=set())
    check("advanced tip of a merged branch stays unqualified", moved.merged_by, None)

    # --- probe_github: the open-PR query failing must fail the WHOLE cross-check closed ----
    real_run = W.run
    merged_json = json.dumps([{"number": 111, "headRefName": "feat/landed",
                               "headRefOid": "a" * 40, "mergedAt": "x"}])

    def fake_run(cmd, cwd=None, timeout=W.GIT_TIMEOUT):
        if "--state" in cmd and "merged" in cmd:
            return 0, merged_json, ""
        if "--state" in cmd and "open" in cmd:
            return 1, "", "API rate limit exceeded"
        return real_run(cmd, cwd=cwd, timeout=timeout)

    try:
        W.run = fake_run
        W._GH_CACHE.clear()
        victim = wt("victim", "a" * 40, "feat/landed")
        ok, note = W.probe_github("/repo", [victim], 10)
        check("open-PR failure -> evidence unavailable", ok, False)
        check("open-PR failure names the cause", "open-PR query failed" in note, True, note)
        check("and qualifies NOTHING", victim.merged_by, None)

        # --- _GH_CACHE: a failure must be remembered, or 145 candidates refetch it ---------
        calls = {"n": 0}

        def counting_run(cmd, cwd=None, timeout=W.GIT_TIMEOUT):
            if "pr" in cmd and "list" in cmd:
                calls["n"] += 1
            return fake_run(cmd, cwd=cwd, timeout=timeout)

        W.run = counting_run
        W._GH_CACHE.clear()
        W.probe_github("/repo", [wt("v1", "a" * 40, "feat/landed")], 10)
        after_first = calls["n"]
        W.probe_github("/repo", [wt("v2", "a" * 40, "feat/landed")], 10)
        check("failed lookup is cached, not refetched", calls["n"], after_first)

        # --- the success path, end to end through probe_github ----------------------------
        def ok_run(cmd, cwd=None, timeout=W.GIT_TIMEOUT):
            if "--state" in cmd and "merged" in cmd:
                return 0, merged_json, ""
            if "--state" in cmd and "open" in cmd:
                return 0, "[]", ""
            return real_run(cmd, cwd=cwd, timeout=timeout)

        W.run = ok_run
        W._GH_CACHE.clear()
        good = wt("good", "a" * 40, "feat/landed")
        ok, note = W.probe_github("/repo", [good], 10)
        check("success path reports available", ok, True)
        check("success path qualifies the tree", good.merged_by, "squash-pr#111")
        check("note states what it examined", "1 merged PRs" in note, True, note)

        # Truncation must announce itself rather than silently narrowing the evidence.
        W._GH_CACHE.clear()
        _, note = W.probe_github("/repo", [wt("t", "a" * 40, "feat/landed")], 1)
        check("hitting --gh-limit warns", "WARNING" in note, True, note)
    finally:
        W.run = real_run
        W._GH_CACHE.clear()


def test_unknown_idle_time_blocks_removal() -> None:
    """idle_hours=None means every stat failed, and must not be the one guard that permits.

    Unreachable in practice on a healthy tree, which is exactly why it needs an arm: it is the
    kind of fail-open default nobody notices until the day the stats fail.
    """
    print("\n[unknown idle time]")
    unknown = W.Worktree(path="/x", real="/x", merged_by="ancestor", idle_hours=None)
    W.classify(unknown, 12.0, merge_evidence_available=True, holder_scan_ok=True)
    check("unknown idle refuses", unknown.removable, False)
    check("unknown idle names recent", unknown.blockers, ["recent"])

    known = W.Worktree(path="/x", real="/x", merged_by="ancestor", idle_hours=999.0)
    W.classify(known, 12.0, merge_evidence_available=True, holder_scan_ok=True)
    check("known-old idle permits", known.removable, True)


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
    """A process that enters the tree AFTER the census must still stop the removal.

    The race is deliberately expressed as a live holder rather than as a dirty file. A dirty file
    is the wrong probe here: `git worktree remove` refuses dirty trees by itself, so that version
    of this arm passed even with the entire re-check deleted -- git produced the identical
    observable outcome and the arm could not tell "re-checked" from "trusted the census and got
    lucky". Verified: with `recheck_and_remove` reduced to a bare `git worktree remove`, the dirty
    form still reported ok.

    Nothing in git checks for live processes, so only our re-check can refuse this one. It is also
    the catastrophic case -- the tree wedges a running shell -- and it is exactly what the
    re-check exists for.
    """
    print("\n[recheck beats a stale census]")
    if not holder_scan_supported():
        print("  SKIP the discriminating form of this arm needs a readable /proc")
        return
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        repo = build_repo(root)
        wt = add_wt(repo, "racy", "feat/racy")
        age(wt, admin_of(repo, "racy"))

        trees, _ = survey(repo)
        check("census says removable", trees["racy"].state, "REMOVABLE")

        holder = spawn_holder(wt)  # an agent cd's in after the scan
        try:
            time.sleep(0.4)
            removed = W.recheck_and_remove(str(repo), trees["racy"], "origin/master", 12.0,
                                           False, False, False, 0, False)
            check("recheck refuses", removed, False)
            check("tree still on disk", wt.is_dir(), True)
        finally:
            holder.kill()
            holder.wait()


def test_holder_matching_survives_path_aliasing() -> None:
    """A holder reached through a different spelling of the same directory must still count.

    Regression for a demonstrated false negative. `/home` is a symlink to `/var/home` on this
    host, so a worktree's canonical path is the `/var/home` form -- but inside the ps5ys distrobox
    `/home` is a real bind mount, so a container process's `/proc/<pid>/cwd` reads back as
    `/home/...`. A string-keyed index missed it entirely: measured live, a
    `distrobox enter ps5ys -- bash -lc "cd <worktree> && ..."` process (the charter's own build
    command) was invisible while a host process in the same tree was seen.

    The fixture builds the alias by hand with a symlink rather than requiring a container, and
    feeds the aliased path in as a reference. Under string matching this arm fails; under
    (st_dev, st_ino) identity it passes.
    """
    print("\n[holder matching survives path aliasing]")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        repo = build_repo(root)
        wt = add_wt(repo, "aliased", "feat/aliased")

        alias = root / "alias"
        try:
            alias.symlink_to(root, target_is_directory=True)
        except (OSError, NotImplementedError):
            print("  SKIP cannot create a directory symlink on this platform")
            return
        aliased_cwd = str(alias / "aliased" / "prosper-ish-subdir")
        (Path(root / "aliased") / "prosper-ish-subdir").mkdir()

        trees = W.list_worktrees(repo)
        by_name = {t.name: t for t in trees}
        check("alias really is a different string", aliased_cwd.startswith(str(by_name["aliased"].real)),
              False, aliased_cwd)

        W.attach_holders(trees, [(4242, "cwd", "bash", aliased_cwd)])
        check("aliased holder is attributed", [h.pid for h in by_name["aliased"].holders], [4242])
        check("and not to the main checkout", [h.pid for h in by_name[repo.name].holders], [])


def test_fd_and_map_references_are_holders() -> None:
    """An open fd or a mapping inside a tree counts, not only cwd.

    Both are on by default in production and neither was covered: every other fixture passes
    scan_fds=False, scan_maps=False for speed. A compiler writing an object file, or a running
    binary mapped out of the tree, is the realistic holder that has no cwd there at all.
    """
    print("\n[fd and map references]")
    if not holder_scan_supported():
        print("  SKIP needs a readable /proc")
        return
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        repo = build_repo(root)
        wt = add_wt(repo, "fdheld", "feat/fdheld")

        # The artifact must be IGNORED, not merely untracked: an untracked file would make the
        # tree dirty, and then the control below would pass because of the dirty guard rather
        # than because the fd was released. Ignoring it means the open fd is the only difference
        # between the two arms, which is what makes this a control at all.
        (repo / ".git" / "info").mkdir(parents=True, exist_ok=True)
        (repo / ".git" / "info" / "exclude").write_text("artifact.bin\n")
        target = wt / "artifact.bin"
        target.write_bytes(b"x" * 4096)
        age(wt, admin_of(repo, "fdheld"))

        fh = open(target, "rb")  # a live fd, from a process whose cwd is elsewhere
        try:
            trees, _ = survey(repo, scan_fds=True, scan_maps=False)
            held = trees["fdheld"]
            check("fd holder refuses the tree", held.removable, False)
            check("fd holder names in-use", "in-use" in held.blockers, True, str(held.blockers))
            check("fd holder is this process",
                  os.getpid() in [h.pid for h in held.holders if h.kind == "fd"], True)
        finally:
            fh.close()

        # Control: with the same tree and no fd open, it is removable again -- so the arm above
        # measured the fd, not some unrelated property of the fixture.
        trees, _ = survey(repo, scan_fds=True, scan_maps=False)
        check("removable once the fd is closed", trees["fdheld"].state, "REMOVABLE")

        # --- the maps parser, which until now this arm's NAME claimed but did not cover -----
        # A mapping is the holder with no cwd and no fd: a running binary, or a compiler's mmap
        # of an object file. It also exercises a different parser -- /proc/<pid>/maps is parsed
        # by hand, unlike the fd and cwd symlinks -- so it can break independently.
        import mmap
        with open(target, "rb") as mf:
            mm = mmap.mmap(mf.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                trees, _ = survey(repo, scan_fds=False, scan_maps=True)
                held = trees["fdheld"]
                check("map holder refuses the tree", held.removable, False)
                check("map holder names in-use", "in-use" in held.blockers, True,
                      str(held.blockers))
                check("map holder is seen as a mapping",
                      os.getpid() in [h.pid for h in held.holders if h.kind == "map"], True,
                      str([(h.pid, h.kind) for h in held.holders]))
            finally:
                mm.close()

        trees, _ = survey(repo, scan_fds=False, scan_maps=True)
        check("removable once the mapping is gone", trees["fdheld"].state, "REMOVABLE")


def main() -> int:
    for fn in (
        test_positive_and_mutations,
        test_merge_evidence_labels,
        test_github_merge_evidence,
        test_unknown_idle_time_blocks_removal,
        test_fails_closed_without_a_process_scan,
        test_nested_worktree_is_refused,
        test_holder_attributed_to_most_specific_tree,
        test_idle_is_not_measuring_our_own_footprint,
        test_dry_run_is_the_default_and_removes_nothing,
        test_remove_takes_only_the_safe_tree,
        test_removal_rechecks_state_rather_than_trusting_the_census,
        test_holder_matching_survives_path_aliasing,
        test_fd_and_map_references_are_holders,
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
