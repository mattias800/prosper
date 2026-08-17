#!/usr/bin/env python3
"""worktree_reclaim.py -- classify this repo's git worktrees, and remove only provably dead ones.

Several agents and the human run this repository concurrently out of separate `git worktree`
checkouts, and nothing removes them when their work lands. They accumulate: issue #1735 measured
121 trees / 111 GB, and the count more than doubled inside a fortnight. The cost is not mainly
disk -- it is that an agent looking for prior work lands in a tree whose branch merged weeks ago
and reads it as current, and that routine `cannot delete branch ... used by worktree at ...`
noise hides the one instance where a lane really is still working in the tree you are about to
delete.

CONFIDENCE: HIGH that this is safe to run; the default does nothing but read.

    Removing a worktree that a live shell is cd'd into wedges that shell irrecoverably.

So this tool defaults to a dry-run census, and `--remove` is gated behind every guard below.
It prints what it EXAMINED, not merely what it would remove: a clean zero from an unarmed scan
is indistinguishable from a clean zero from a scan that found nothing, and this project has been
bitten by that often enough to name it. Each tree prints its state and every blocker that fired.

Guards -- ALL must pass before a tree is even a candidate:

  main        the primary worktree is never touched
  exists      a vanished directory is a `git worktree prune` job, not a `remove` job
  locked      `git worktree lock` is an explicit human "leave this alone"
  unmerged    the tree's work must be provably in origin/master (see "Merge evidence")
  dirty       `git status --porcelain --untracked-files=all` must be empty, untracked included
  in-use      no process may have cwd / exe / an open fd / a mapping inside the tree
  recent      the tree must have been idle for --min-idle-hours (default 12)
  nested      no other registered worktree may live underneath it

Merge evidence -- two independent primary sources, never a heuristic:

  ancestor    `git merge-base --is-ancestor <head> origin/master`. Exact, but blind to squash
              merges, which is how this repository merges every PR (checked: the last 20 commits
              on master each have one parent). After a squash the branch tip is NOT an ancestor,
              so this test alone under-reports badly -- it mostly catches branches that never
              committed anything.
  squash-pr   the tree's HEAD sha equals the `headRefOid` of a MERGED pull request, via `gh`.
              That is conclusive: this exact tree state is what GitHub merged. For a named
              branch we additionally require that no OPEN pull request uses it, so a branch name
              recycled for later work is never mistaken for the merged one.

Both are conservative in the same direction: they can fail to notice that a tree is dead, and
they cannot claim a live tree is dead. Unproven means "left alone", which is why the census
distinguishes "not merged" from "no evidence available" -- a `gh` outage must not silently turn
into a wall of ACTIVE.

What this deliberately does NOT protect, stated because each one looks covered:

  ignored files   `git status` does not list gitignored content, so a tree whose only content is
                  an ignored `build-linux/` or a run log reads as clean and goes with it. Mostly
                  that is the point -- but the charter also tells agents to put run artifacts in
                  "a gitignored worktree-local directory", so the dirty guard does not mean what
                  a reader may assume. Move what you want to keep, or `git worktree lock` the tree.
  detached trees  the recycled-branch cross-check keys on a branch name, so it cannot apply to a
                  detached worktree; those qualify on the HEAD sha alone. Sound, because the sha
                  matching a merged PR's headRefOid is what is being asserted -- but if that sha
                  is reachable only from this worktree's HEAD, removing the tree does make the sha
                  unreachable. The content is still on master via the squash.
  non-Linux       the in-use guard needs `/proc`. Without it the tool fails closed and removes
                  nothing at all, so on macOS and Windows this is a classifier only.

Usage:

    tools/worktree_reclaim.py                      # census, touches nothing
    tools/worktree_reclaim.py --json               # same, machine-readable
    tools/worktree_reclaim.py --redact             # census with host paths masked, for pasting
    tools/worktree_reclaim.py --remove --yes       # act; re-checks every guard per tree first

`--remove` re-runs the guards immediately before each individual removal rather than trusting the
census. A point-in-time scan is advisory, not a lock: an agent can cd into a tree between the scan
and the delete, and the whole point of the scan is the case where that matters.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

DEFAULT_MIN_IDLE_HOURS = 12.0
GIT_TIMEOUT = 120
STATUS_SAMPLE = 8  # porcelain lines kept per tree for the report

# Ordered most-severe first: a tree's headline state is its first blocker, but `blockers`
# always carries the complete list so the report never implies a single cause.
BLOCKER_ORDER = [
    "main",
    "missing",
    "locked",
    "in-use",
    "holder-scan-unavailable",
    "dirty",
    "unmerged",
    "no-merge-evidence",
    "nested",
    "recent",
    "status-unavailable",
]


def run(cmd: list[str], cwd: str | None = None, timeout: int = GIT_TIMEOUT) -> tuple[int, str, str]:
    """Run a command, never raising. Returns (rc, stdout, stderr); rc 124 on timeout."""
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except OSError as e:
        return 127, "", str(e)


@dataclass
class Holder:
    """A live process with a reference into a worktree -- the guard that matters most."""

    pid: int
    kind: str  # cwd | exe | root | fd | map
    comm: str
    path: str


@dataclass
class Worktree:
    path: str  # exactly as git reported it
    real: str  # realpath; /home and /var/home are the same volume here
    head: str | None = None
    branch: str | None = None  # short name, None when detached
    detached: bool = False
    locked: bool = False
    is_main: bool = False
    exists: bool = True
    admin_dir: str | None = None

    merged_by: str | None = None  # "ancestor" | "squash-pr#123" | None
    merge_evidence_available: bool = True
    dirty_tracked: int = 0
    dirty_untracked: int = 0
    dirty_sample: list[str] = field(default_factory=list)
    status_error: str | None = None
    holders: list[Holder] = field(default_factory=list)
    idle_hours: float | None = None
    nested: list[str] = field(default_factory=list)
    size_bytes: int | None = None

    blockers: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return os.path.basename(self.real) or self.real

    @property
    def removable(self) -> bool:
        return not self.blockers and self.merged_by is not None

    @property
    def state(self) -> str:
        if self.removable:
            return "REMOVABLE"
        for b in BLOCKER_ORDER:
            if b in self.blockers:
                return b.upper()
        return "UNKNOWN"


# --------------------------------------------------------------------------- inventory


def list_worktrees(repo: str) -> list[Worktree]:
    rc, out, err = run(["git", "-C", repo, "worktree", "list", "--porcelain"])
    if rc != 0:
        sys.exit(f"worktree_reclaim: `git worktree list` failed: {err.strip()}")

    trees: list[Worktree] = []
    cur: Worktree | None = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            p = line[len("worktree ") :]
            cur = Worktree(path=p, real=os.path.realpath(p))
            trees.append(cur)
        elif cur is None:
            continue
        elif line.startswith("HEAD "):
            cur.head = line[len("HEAD ") :].strip()
        elif line.startswith("branch "):
            ref = line[len("branch ") :].strip()
            cur.branch = ref[len("refs/heads/") :] if ref.startswith("refs/heads/") else ref
        elif line == "detached":
            cur.detached = True
        elif line.startswith("locked"):
            cur.locked = True
        elif line.startswith("prunable"):
            cur.exists = False

    for t in trees:
        t.exists = t.exists and os.path.isdir(t.real)
    if trees:
        trees[0].is_main = True

    # Nested registrations: removing a parent would orphan whatever lives beneath it.
    reals = {t.real for t in trees}
    for t in trees:
        if t.is_main:
            continue  # every .claude/worktrees/* sits under main by construction
        # os.sep, not "/": on Windows realpath yields backslashes and a "/" prefix never matches,
        # which silently disables this guard. Appending the separator is also what keeps
        # `/a/foo` from being read as a parent of `/a/foobar`.
        pfx = t.real.rstrip(os.sep) + os.sep
        t.nested = sorted(r for r in reals if r != t.real and r.startswith(pfx))

    for t in trees:
        t.admin_dir = worktree_admin_dir(t)
    return trees


def worktree_admin_dir(t: Worktree) -> str | None:
    """Resolve `<repo>/.git/worktrees/<name>` from the tree's `.git` pointer file."""
    dotgit = os.path.join(t.real, ".git")
    try:
        if os.path.isfile(dotgit):
            with open(dotgit, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("gitdir:"):
                        return os.path.realpath(line.split(":", 1)[1].strip())
        elif os.path.isdir(dotgit):
            return dotgit  # the main worktree
    except OSError:
        pass
    return None


# --------------------------------------------------------------------------- process scan


def _readlink(path: str) -> str | None:
    try:
        target = os.readlink(path)
    except OSError:
        return None
    # /proc marks unlinked targets; the prefix is still the directory we care about.
    return target[: -len(" (deleted)")] if target.endswith(" (deleted)") else target


def scan_processes(
    scan_fds: bool = True, scan_maps: bool = True
) -> tuple[list[tuple[int, str, str, str]], bool]:
    """Collect every (pid, kind, comm, path) reference a live process holds.

    Returns (refs, supported). `supported` is False where there is no readable `/proc` -- macOS
    and Windows, notably. That flag is load-bearing and must never be dropped: without it this
    function returns an empty list on those platforms, the in-use guard never fires, and a
    worktree with a live shell sitting in it classifies as REMOVABLE. An absent instrument would
    read as "nobody is using anything", which is the most dangerous answer this tool can give.
    Callers must fail closed on `supported == False` rather than treating it as "no holders".

    Deliberately scans ALL processes rather than a list of known binary names. The failure this
    guards against is a wedged interactive shell, and a shell is exactly what a name list misses.

    Unreadable processes (other users, or exited mid-scan) are skipped, so this is a LOWER bound
    on holders -- and that is the UNSAFE direction, not the safe one: under-detecting holders
    yields "nobody is here" and hence REMOVABLE. An upper bound would be safe. Nothing here can
    make the bound tight, which is precisely why `supported` exists and why the idle guard backs
    this one up; do not read a short holder list as reassurance.
    """
    refs: list[tuple[int, str, str, str]] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return refs, False
    if not any(n.isdigit() for n in entries):
        return refs, False  # something is mounted at /proc, but it is not a Linux procfs

    for name in entries:
        if not name.isdigit():
            continue
        pid = int(name)
        base = f"/proc/{name}"
        try:
            with open(f"{base}/comm", encoding="utf-8", errors="replace") as fh:
                comm = fh.read().strip()
        except OSError:
            comm = "?"

        for kind in ("cwd", "exe", "root"):
            tgt = _readlink(f"{base}/{kind}")
            if tgt and tgt != "/":
                refs.append((pid, kind, comm, tgt))

        if scan_fds:
            try:
                for fd in os.listdir(f"{base}/fd"):
                    tgt = _readlink(f"{base}/fd/{fd}")
                    if tgt and tgt.startswith("/"):
                        refs.append((pid, "fd", comm, tgt))
            except OSError:
                pass

        if scan_maps:
            try:
                seen: set[str] = set()
                with open(f"{base}/maps", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        idx = line.find(" /")
                        if idx == -1:
                            continue
                        tgt = line[idx + 1 :].rstrip("\n")
                        if tgt.endswith(" (deleted)"):
                            tgt = tgt[: -len(" (deleted)")]
                        if tgt not in seen:
                            seen.add(tgt)
                            refs.append((pid, "map", comm, tgt))
            except OSError:
                pass
    return refs, True


def attach_holders(trees: list[Worktree], refs: list[tuple[int, str, str, str]]) -> None:
    """Assign each reference to the most specific worktree containing it.

    Matching is by (st_dev, st_ino) identity, NOT by path string. String matching produced a
    demonstrated false negative here, which is the direction that deletes people's work:

        `/home` is a symlink to `/var/home` on this host, so `t.real` is always the `/var/home`
        spelling. Inside the ps5ys distrobox, however, `/home` is a real bind mount, so a
        container process's `/proc/<pid>/cwd` reads back as `/home/...` -- a different string for
        the same directory. Measured live: a `distrobox enter ps5ys -- bash -lc "cd <worktree> &&
        ..."` process, which is exactly the build command the charter prescribes, was invisible to
        the string index while a host process in the same tree was seen. Both paths stat to
        dev=57 ino=10784453.

    Inode identity closes that and any other aliasing (bind mounts, additional symlinks) in one
    step, because it asks the question the guard actually means: is this process inside THIS
    directory, however it happens to spell it.

    Walking up from the reference and taking the first hit is what makes "most specific" work:
    every `.claude/worktrees/X` is inside the main checkout, so matching the outermost container
    would blame main for a process that is really sitting in X.
    """
    by_key: dict[tuple[int, int], Worktree] = {}
    for t in trees:
        try:
            st = os.stat(t.real)
        except OSError:
            continue  # a vanished tree cannot hold anything; `missing` covers it
        by_key[(st.st_dev, st.st_ino)] = t

    # Memoised per directory: without this the ancestor walk stats the same handful of parents
    # tens of thousands of times over a full /proc sweep.
    cache: dict[str, Worktree | None] = {}

    def owner(path: str) -> Worktree | None:
        chain: list[str] = []
        cur = path
        found: Worktree | None = None
        while True:
            if cur in cache:
                found = cache[cur]
                break
            chain.append(cur)
            try:
                st = os.stat(cur)
                hit = by_key.get((st.st_dev, st.st_ino))
            except OSError:
                hit = None  # deleted fd target, or a path we may not stat; keep walking up
            if hit is not None:
                found = hit
                break
            parent = os.path.dirname(cur)
            if parent == cur or len(parent) <= 1:
                break
            cur = parent
        for c in chain:
            cache[c] = found
        return found

    for pid, kind, comm, path in refs:
        t = owner(path)
        if t is not None:
            t.holders.append(Holder(pid=pid, kind=kind, comm=comm, path=path))


# --------------------------------------------------------------------------- per-tree probes


def probe_status(t: Worktree) -> None:
    if not t.exists:
        return
    # `--no-optional-locks` is load-bearing, not tidiness. A plain `git status` REWRITES the index
    # whenever the stat cache is stale, which sets the index mtime to now -- and the index is one
    # of the files probe_idle reads to decide whether anyone has touched this tree. Measured on a
    # tree with a deliberately stale cache: plain status moved the index mtime to the current
    # second, `--no-optional-locks` left it at its true value. Without this flag the tool marks
    # trees "recent" because it just visited them.
    rc, out, err = run(
        ["git", "--no-optional-locks", "-C", t.real, "status", "--porcelain=v1",
         "--untracked-files=all"]
    )
    if rc != 0:
        t.status_error = (err.strip() or f"rc={rc}")[:200]
        return
    for line in out.splitlines():
        if line.startswith("?? "):
            t.dirty_untracked += 1
        elif line.strip():
            t.dirty_tracked += 1
        if len(t.dirty_sample) < STATUS_SAMPLE and line.strip():
            t.dirty_sample.append(line)


def probe_idle(t: Worktree) -> None:
    """Newest mtime over the tree root, its immediate children, and the files in its git admin dir.

    MUST run before probe_status, and MUST NOT stat the admin directory itself. Both rules are
    here because the first draft broke them and reported every one of 278 trees as touched
    "0.00h ago", which read as a plausible "everything is busy" rather than as a broken gauge:

      `git status` creates and unlinks a transient lock inside `.git/worktrees/<name>/`, and
      adding an entry to a directory updates THAT DIRECTORY's mtime. The files within keep their
      true age -- measured on an idle tree, `index` was 27 days old and `HEAD`/`gitdir`/`refs`
      31 days old, while the containing directory read as 4 seconds old, entirely because this
      tool had just run `git status` there. `git worktree list` does not do this; `git status`
      does.

    So the scan was dating its own footprints. Reading only the files inside the admin dir gives
    the real last-git-activity time, and probing idle first means no probe of ours has touched
    the tree yet either way.

    Still a proxy, not proof: a compile writing only inside `build-linux/obj/` updates none of
    these, so a busy tree can look idle. This is a backstop for the process scan -- an agent
    paused between tool calls holds no cwd -- not a substitute for it.
    """
    newest = 0.0

    def bump(path: str, follow: bool = True) -> None:
        nonlocal newest
        try:
            newest = max(newest, os.stat(path, follow_symlinks=follow).st_mtime)
        except OSError:
            pass

    if os.path.isdir(t.real):
        bump(t.real)  # a top-level add/remove is real activity, and git status does not cause it
        try:
            with os.scandir(t.real) as it:
                for e in it:
                    bump(e.path, follow=False)
        except OSError:
            pass

    if t.admin_dir and os.path.isdir(t.admin_dir):
        # Deliberately NOT os.stat(t.admin_dir) -- see the docstring.
        try:
            with os.scandir(t.admin_dir) as it:
                for e in it:
                    bump(e.path, follow=False)
        except OSError:
            pass

    t.idle_hours = (time.time() - newest) / 3600.0 if newest else None


def probe_size(t: Worktree) -> None:
    if not t.exists:
        return
    rc, out, _ = run(["du", "-sb", "--", t.real], timeout=300)
    if rc == 0 and out.strip():
        try:
            t.size_bytes = int(out.split()[0])
        except (ValueError, IndexError):
            pass


# --------------------------------------------------------------------------- merge evidence


def probe_ancestor(repo: str, trees: list[Worktree], base: str) -> None:
    for t in trees:
        if t.is_main or not t.head:
            continue
        rc, _, _ = run(["git", "-C", repo, "merge-base", "--is-ancestor", t.head, base])
        if rc == 0:
            t.merged_by = "ancestor"


_GH_CACHE: dict[tuple[str, int], tuple[bool, str, dict[str, int], set[str]]] = {}


def probe_github(repo: str, trees: list[Worktree], limit: int) -> tuple[bool, str]:
    """Cross-check against merged PRs. Returns (available, note).

    Two batched `gh` calls rather than one per tree; with hundreds of worktrees the per-tree form
    is both slow and rate-limited.
    """
    cached = _GH_CACHE.get((repo, limit))
    if cached is not None:
        ok, note, by_oid, open_branches = cached
        if ok:
            _apply_github(trees, by_oid, open_branches)
        return ok, note

    rc, out, err = run(
        [
            "gh", "pr", "list", "--state", "merged", "--limit", str(limit),
            "--json", "number,headRefName,headRefOid,mergedAt",
        ],
        cwd=repo,
        timeout=300,
    )
    if rc != 0:
        note = f"gh unavailable ({(err.strip() or 'rc=%d' % rc)[:120]})"
        _GH_CACHE[(repo, limit)] = (False, note, {}, set())
        return False, note
    try:
        merged = json.loads(out)
    except json.JSONDecodeError as e:
        return False, f"gh returned unparseable JSON ({e})"

    rc2, out2, err2 = run(
        ["gh", "pr", "list", "--state", "open", "--limit", str(limit), "--json", "number,headRefName"],
        cwd=repo,
        timeout=300,
    )
    # An empty open-PR set is the permissive answer -- it is what licenses squash-pr qualification
    # for a named branch. So a FAILED query must not produce it: "the query broke" and "there are
    # no open PRs" would otherwise be the same value, and a rate limit right after the paginated
    # merged fetch is a realistic way to get there. Fail closed on the whole cross-check instead.
    if rc2 != 0:
        return False, f"gh open-PR query failed ({(err2.strip() or 'rc=%d' % rc2)[:120]})"
    try:
        open_branches = {pr["headRefName"] for pr in json.loads(out2)}
    except json.JSONDecodeError as e:
        return False, f"gh open-PR query returned unparseable JSON ({e})"

    by_oid: dict[str, int] = {}
    for pr in merged:
        oid = pr.get("headRefOid")
        if oid:
            by_oid.setdefault(oid, pr["number"])

    note = f"gh: {len(merged)} merged PRs, {len(open_branches)} open branches"
    if len(merged) >= limit:
        # Newest-first, so merged-list truncation drops the OLDEST PRs -- the trees most worth
        # reclaiming. That direction is safe (they stay unqualified). Truncating the OPEN list
        # would be unsafe, so say so loudly rather than quietly qualifying more trees.
        note += f" -- WARNING: hit --gh-limit {limit}; raise it, results are truncated"
    _GH_CACHE[(repo, limit)] = (True, note, by_oid, open_branches)
    _apply_github(trees, by_oid, open_branches)
    return True, note


def _apply_github(trees: list[Worktree], by_oid: dict[str, int], open_branches: set[str]) -> None:
    for t in trees:
        if t.is_main or t.merged_by or not t.head:
            continue
        num = by_oid.get(t.head)
        if num is None:
            continue
        # A recycled branch name with a NEW open PR is live work; the sha check alone would not
        # see it, because the old merged sha is still what this stale tree has checked out.
        if t.branch and t.branch in open_branches:
            continue
        t.merged_by = f"squash-pr#{num}"


# --------------------------------------------------------------------------- classification


def classify(t: Worktree, min_idle_hours: float, merge_evidence_available: bool,
             holder_scan_ok: bool = True) -> None:
    t.blockers = []
    if t.is_main:
        t.blockers.append("main")
        return
    if not t.exists:
        t.blockers.append("missing")
        return
    if t.locked:
        t.blockers.append("locked")
    if not holder_scan_ok:
        # Fail closed. Without a readable /proc we cannot tell whether a live shell is sitting in
        # this tree, and "the instrument is absent" must never be reported as "nobody is here".
        # This makes the tool classify-only on macOS and Windows, which is the honest outcome.
        t.blockers.append("holder-scan-unavailable")
    elif t.holders:
        t.blockers.append("in-use")
    if t.status_error:
        t.blockers.append("status-unavailable")
    elif t.dirty_tracked or t.dirty_untracked:
        t.blockers.append("dirty")
    if t.merged_by is None:
        t.blockers.append("unmerged" if merge_evidence_available else "no-merge-evidence")
    if t.nested:
        t.blockers.append("nested")
    # None means every stat failed. Everything else in this list is fail-closed; an unknown
    # answer must not be the one place that quietly permits removal.
    if t.idle_hours is None or t.idle_hours < min_idle_hours:
        t.blockers.append("recent")


def survey(
    repo: str,
    base: str,
    min_idle_hours: float,
    jobs: int,
    use_github: bool,
    gh_limit: int,
    scan_fds: bool,
    scan_maps: bool,
    with_size: bool,
) -> tuple[list[Worktree], dict]:
    trees = list_worktrees(repo)

    # Idle FIRST: `git status` bumps the admin directory's mtime, so probing it afterwards would
    # measure this tool's own footprint. See probe_idle.
    for t in trees:
        probe_idle(t)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(probe_status, trees))
    if with_size:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(probe_size, trees))

    refs, holder_scan_ok = scan_processes(scan_fds=scan_fds, scan_maps=scan_maps)
    attach_holders(trees, refs)

    probe_ancestor(repo, trees, base)
    gh_available, gh_note = (False, "skipped (--no-github)")
    if use_github:
        gh_available, gh_note = probe_github(repo, trees, gh_limit)

    # "Not merged" and "could not tell" are different answers and must not share a label. The
    # ancestor test is local and always runs, but it is blind to squash merges, so on its own it
    # cannot distinguish live work from work that landed via a squash. Only a successful GitHub
    # cross-check licenses the confident UNMERGED verdict; otherwise the honest state is
    # NO-MERGE-EVIDENCE. Both are equally unremovable -- this is about not reporting a `gh`
    # outage, or `--no-github`, as a wall of live branches.
    evidence_complete = use_github and gh_available
    for t in trees:
        t.merge_evidence_available = evidence_complete
        classify(t, min_idle_hours, merge_evidence_available=evidence_complete,
                 holder_scan_ok=holder_scan_ok)

    meta = {
        "repo": repo,
        "base": base,
        "base_sha": run(["git", "-C", repo, "rev-parse", base])[1].strip(),
        "min_idle_hours": min_idle_hours,
        "holder_scan_supported": holder_scan_ok,
        "process_refs_scanned": len(refs),
        "pids_seen": len({r[0] for r in refs}),
        "scan_fds": scan_fds,
        "scan_maps": scan_maps,
        "github": gh_note,
        "github_available": gh_available,
    }
    return trees, meta


# --------------------------------------------------------------------------- reporting


def redact(path: str) -> str:
    home = os.path.expanduser("~")
    for prefix in (os.path.realpath(home), home):
        if path.startswith(prefix):
            return "~" + path[len(prefix) :]
    return path


def human(n: int | None) -> str:
    if n is None:
        return "-"
    v = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if v < 1024 or unit == "T":
            return f"{v:.0f}{unit}" if unit == "B" else f"{v:.1f}{unit}"
        v /= 1024
    return f"{v:.1f}T"


def report(trees: list[Worktree], meta: dict, do_redact: bool, verbose: bool) -> None:
    fmt = redact if do_redact else (lambda p: p)

    print("=" * 100)
    print("worktree_reclaim -- census")
    print("=" * 100)
    print(f"repo                 {fmt(meta['repo'])}")
    print(f"base                 {meta['base']} @ {meta['base_sha'][:12]}")
    print(f"worktrees registered {len(trees)}")
    if meta["holder_scan_supported"]:
        print(f"process refs scanned {meta['process_refs_scanned']} over {meta['pids_seen']} pids "
              f"(fds={meta['scan_fds']}, maps={meta['scan_maps']})")
    else:
        print("process refs scanned NONE -- no readable /proc on this platform, so the in-use "
              "guard cannot run;\n                     every tree is held back (fail closed) "
              "and this run is classify-only")
    print(f"merge evidence       ancestor-of-{meta['base']} + {meta['github']}")
    print(f"idle threshold       {meta['min_idle_hours']}h")
    print()

    order = {s: i for i, s in enumerate(["REMOVABLE"] + [b.upper() for b in BLOCKER_ORDER])}
    counts: dict[str, int] = {}
    reclaim = 0
    for t in trees:
        counts[t.state] = counts.get(t.state, 0) + 1
        if t.removable and t.size_bytes:
            reclaim += t.size_bytes

    print(f"{'STATE':<20} {'COUNT':>6}   meaning")
    print("-" * 100)
    meanings = {
        "REMOVABLE": "merged, clean, idle, unheld -- every guard passed",
        "MAIN": "the primary checkout, never touched",
        "MISSING": "directory gone; `git worktree prune` reclaims the registration",
        "LOCKED": "explicitly locked by a human",
        "IN-USE": "a live process holds cwd/exe/fd/map inside it",
        "HOLDER-SCAN-UNAVAILABLE": "no readable /proc: cannot prove nobody is inside (fail closed)",
        "DIRTY": "uncommitted or untracked files present",
        "UNMERGED": "work not provably in the base branch",
        "NO-MERGE-EVIDENCE": "could not establish merge state",
        "NESTED": "another worktree is registered underneath it",
        "RECENT": "touched inside the idle threshold",
        "STATUS-UNAVAILABLE": "`git status` failed here",
    }
    for state in sorted(counts, key=lambda s: order.get(s, 99)):
        print(f"{state:<20} {counts[state]:>6}   {meanings.get(state, '')}")
    print("-" * 100)
    print(f"{'TOTAL':<20} {len(trees):>6}")

    # The table above is mutually exclusive -- a tree is filed under its most severe blocker only,
    # so "UNMERGED 78" does NOT mean only 78 trees are unmerged. Quote this tally for per-guard
    # counts; a tree carrying three blockers appears in three rows here and one row above.
    tally: dict[str, int] = {}
    for t in trees:
        for b in t.blockers:
            tally[b] = tally.get(b, 0) + 1
    print()
    print("guards fired (independent; one tree can trip several -- these do NOT sum to the total)")
    print("-" * 100)
    for b in BLOCKER_ORDER:
        if tally.get(b):
            print(f"{b:<20} {tally[b]:>6}")
    if reclaim:
        print(f"\nreclaimable if every REMOVABLE tree is removed: {human(reclaim)}")
    print()

    groups: dict[str, list[Worktree]] = {}
    for t in trees:
        groups.setdefault(t.state, []).append(t)

    for state in sorted(groups, key=lambda s: order.get(s, 99)):
        if state == "MAIN" and not verbose:
            continue
        rows = groups[state]
        if state in ("UNMERGED", "RECENT", "IN-USE", "DIRTY") and not verbose:
            print(f"### {state} ({len(rows)}) -- not listed; pass --verbose")
            print()
            continue
        print(f"### {state} ({len(rows)})")
        for t in sorted(rows, key=lambda x: x.real):
            bits = [f"  {fmt(t.real)}"]
            bits.append(f"      branch   {t.branch or '(detached)'}  head {(t.head or '?')[:12]}")
            bits.append(
                f"      merged   {t.merged_by or 'NO'}"
                f"   dirty {t.dirty_tracked}+{t.dirty_untracked}u"
                f"   idle {('%.1fh' % t.idle_hours) if t.idle_hours is not None else '?'}"
                f"   size {human(t.size_bytes)}"
            )
            if t.blockers:
                bits.append(f"      blocked  {', '.join(t.blockers)}")
            if t.holders:
                shown = t.holders[:4]
                for h in shown:
                    bits.append(f"      holder   pid {h.pid} [{h.comm}] {h.kind} -> {fmt(h.path)}")
                if len(t.holders) > len(shown):
                    bits.append(f"      holder   ... and {len(t.holders) - len(shown)} more")
            if t.dirty_sample and verbose:
                for line in t.dirty_sample:
                    bits.append(f"      status   {line}")
            if t.nested:
                bits.append(f"      nested   {', '.join(fmt(n) for n in t.nested)}")
            print("\n".join(bits))
        print()


# --------------------------------------------------------------------------- removal


def recheck_and_remove(repo: str, t: Worktree, base: str, min_idle_hours: float,
                       scan_fds: bool, scan_maps: bool, use_github: bool,
                       gh_limit: int, do_redact: bool) -> bool:
    """Re-run every guard against fresh state, then remove. Returns True on removal.

    The census is deliberately not trusted here. An agent can cd into a tree between the scan and
    the delete, and that race is the entire reason the scan exists.
    """
    fmt = redact if do_redact else (lambda p: p)
    fresh = list_worktrees(repo)
    cur = next((w for w in fresh if w.real == t.real), None)
    if cur is None:
        print(f"  SKIP {fmt(t.real)}: no longer registered")
        return False

    probe_idle(cur)  # before probe_status, for the reason given in probe_idle
    probe_status(cur)
    fresh_refs, holder_scan_ok = scan_processes(scan_fds=scan_fds, scan_maps=scan_maps)
    attach_holders(fresh, fresh_refs)
    probe_ancestor(repo, fresh, base)
    # Cached after the census, so this is a dict lookup rather than another paginated fetch --
    # 145 candidates would otherwise mean 145 full merged-PR queries and a near-certain rate limit,
    # whose symptom would be trees refusing under a mislabelled verdict.
    gh_ok = probe_github(repo, fresh, gh_limit)[0] if use_github else False
    cur = next((w for w in fresh if w.real == t.real), cur)
    classify(cur, min_idle_hours, merge_evidence_available=(use_github and gh_ok),
             holder_scan_ok=holder_scan_ok)

    if not cur.removable:
        print(f"  REFUSE {fmt(cur.real)}: {', '.join(cur.blockers) or 'no merge evidence'}")
        return False
    if cur.merged_by != t.merged_by:
        print(f"  REFUSE {fmt(cur.real)}: merge evidence changed since census")
        return False

    # No --force, ever: git's own dirty check is a second independent line of defence, and the
    # branch ref is deliberately left in place so no commit becomes unreachable.
    rc, _, err = run(["git", "-C", repo, "worktree", "remove", cur.real])
    if rc != 0:
        print(f"  FAILED {fmt(cur.real)}: {err.strip()[:200]}")
        return False
    print(f"  REMOVED {fmt(cur.real)}  ({cur.merged_by}, idle {cur.idle_hours:.1f}h)")
    return True


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Classify git worktrees; remove only provably dead ones (opt-in).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--repo", default=None, help="repository root (default: containing this file)")
    ap.add_argument("--base", default="origin/master", help="branch work must have landed in")
    ap.add_argument("--min-idle-hours", type=float, default=DEFAULT_MIN_IDLE_HOURS)
    ap.add_argument("--jobs", type=int, default=4, help="parallel git probes (default 4)")
    ap.add_argument("--no-github", action="store_true", help="skip the merged-PR cross-check")
    ap.add_argument("--gh-limit", type=int, default=3000, help="merged PRs to fetch")
    ap.add_argument("--no-fds", action="store_true", help="skip /proc/*/fd scanning")
    ap.add_argument("--no-maps", action="store_true", help="skip /proc/*/maps scanning")
    ap.add_argument("--size", action="store_true", help="measure disk usage per tree (slow)")
    ap.add_argument("--json", action="store_true", help="machine-readable census")
    ap.add_argument("--redact", action="store_true", help="mask the home prefix in all paths")
    ap.add_argument("--verbose", action="store_true", help="list every tree, not just candidates")
    ap.add_argument("--remove", action="store_true", help="actually remove REMOVABLE trees")
    ap.add_argument("--yes", action="store_true", help="required alongside --remove")
    ap.add_argument("--prune", action="store_true", help="`git worktree prune` for MISSING trees")
    ap.add_argument("--limit", type=int, default=0, help="cap removals this run (0 = no cap)")
    ap.add_argument("--only", action="append", default=[],
                    help="restrict removal to trees whose path contains this substring")
    args = ap.parse_args(argv)

    repo = args.repo or os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
    rc, out, _ = run(["git", "-C", repo, "rev-parse", "--show-toplevel"])
    if rc != 0:
        sys.exit(f"worktree_reclaim: not a git repository: {repo}")
    repo = out.strip() or repo

    trees, meta = survey(
        repo=repo,
        base=args.base,
        min_idle_hours=args.min_idle_hours,
        jobs=max(1, args.jobs),
        use_github=not args.no_github,
        gh_limit=args.gh_limit,
        scan_fds=not args.no_fds,
        scan_maps=not args.no_maps,
        with_size=args.size,
    )

    if args.json:
        print(json.dumps(
            {
                "meta": ({**meta, "repo": redact(meta["repo"])} if args.redact else meta),
                "worktrees": [
                    {
                        "path": redact(t.real) if args.redact else t.real,
                        "branch": t.branch,
                        "head": t.head,
                        "state": t.state,
                        "merged_by": t.merged_by,
                        "blockers": t.blockers,
                        "dirty_tracked": t.dirty_tracked,
                        "dirty_untracked": t.dirty_untracked,
                        "idle_hours": t.idle_hours,
                        "size_bytes": t.size_bytes,
                        "holders": [
                            {"pid": h.pid, "kind": h.kind, "comm": h.comm,
                             "path": redact(h.path) if args.redact else h.path}
                            for h in t.holders
                        ],
                        "removable": t.removable,
                    }
                    for t in trees
                ],
            },
            indent=2,
        ))
    else:
        report(trees, meta, args.redact, args.verbose)

    # In --json mode every human line goes to stderr, so stdout stays a parseable document.
    say = (lambda *a: print(*a, file=sys.stderr)) if args.json else print

    if args.prune:
        missing = [t for t in trees if not t.exists and not t.is_main]
        if not (args.remove and args.yes):
            say(f"[dry-run] would `git worktree prune` for {len(missing)} MISSING registration(s)")
        else:
            rc, _, err = run(["git", "-C", repo, "worktree", "prune", "-v"])
            say(f"pruned {len(missing)} missing registration(s)"
                if rc == 0 else f"prune failed: {err.strip()[:200]}")

    candidates = [t for t in trees if t.removable]
    if args.only:
        candidates = [t for t in candidates if any(s in t.real for s in args.only)]
    if args.limit:
        candidates = candidates[: args.limit]

    if not args.remove:
        say(f"[dry-run] {len(candidates)} tree(s) pass every guard. "
            f"Nothing was modified. Re-run with --remove --yes to act.")
        return 0
    if not args.yes:
        say(f"[dry-run] --remove requires --yes as well. {len(candidates)} candidate(s); "
            f"nothing was modified.")
        return 2

    say(f"\nRemoving {len(candidates)} tree(s); every guard is re-checked per tree first.")
    removed = sum(
        1 for t in candidates
        if recheck_and_remove(repo, t, args.base, args.min_idle_hours,
                              not args.no_fds, not args.no_maps,
                              not args.no_github, args.gh_limit, args.redact)
    )
    say(f"\nremoved {removed} of {len(candidates)} candidate(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
