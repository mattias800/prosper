#!/usr/bin/env python3
"""gen_screenshot_feed.py -- build SCREENSHOTS.md, every checked-in screenshot newest-first.

WHY THIS EXISTS. Progress on this project is visual, and the visual record was spread across
COMPATIBILITY.md's per-title sections and 39 tracker issues. Finding "what is new since I last
looked" meant scrolling a 500-line chart and diffing it against memory. This file answers that
question directly: open it, read down from the top, stop when you reach something you have seen.

IT IS GENERATED. Do not hand-edit it -- edits are overwritten. There is nothing to maintain and
therefore nothing to forget: the ordering and the captions come from git history, so an image is
in the feed the moment its commit lands and it can never disagree with the tree.

    python3 prosper/tools/docs/gen_screenshot_feed.py           # rewrite SCREENSHOTS.md
    python3 prosper/tools/docs/gen_screenshot_feed.py --check   # fail if it is stale
    python3 prosper/tools/docs/gen_screenshot_feed.py --selftest # prove the generator discriminates

FAILURE MODE THIS IS BUILT AGAINST, because it is the one that would make the file worse than
useless: an image that is present in the tree but missing from the feed. A feed that silently
drops entries is read as "nothing new happened", which is exactly the wrong conclusion and is
indistinguishable from the truth. So every image in the tree MUST resolve to an add-commit, and
the tool ABORTS naming the offenders rather than emitting a partial file.

That is not hypothetical. The obvious implementation --

    git log --diff-filter=A --follow -- <path>          # WRONG

-- returns EMPTY for recently added files (measured 2026-08-20 on two images added that day, both
of which the non-follow query dates correctly). `--follow` is for tracking a path across renames
and does not compose reliably with `--diff-filter=A`. Using it would have silently omitted the
NEWEST entries -- the only ones anybody opens this file to see. The scan below is a single
`--diff-filter=A --name-only` pass over the whole history instead: no per-file queries, no
`--follow`, and O(1) git invocations rather than O(images).
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from collections import defaultdict

OUT = "SCREENSHOTS.md"
# Directories scanned, in the order their images are attributed. `assets/screenshots` is the
# user-facing compatibility gallery; `prosper/docs/screenshots` is issue-specific evidence.
DIRS = ("assets/screenshots", "prosper/docs/screenshots")
EXTS = (".png", ".jpg", ".jpeg")

BANNER = """<!--
    GENERATED FILE -- DO NOT EDIT BY HAND.
    Produced by prosper/tools/docs/gen_screenshot_feed.py from git history.
    Regenerate with:  python3 prosper/tools/docs/gen_screenshot_feed.py
    CI regenerates this file and fails if it differs from the committed copy.
-->"""


def run(args: list[str], repo: pathlib.Path) -> str:
    """Run git, and turn a failure into a usable message rather than a traceback.

    This tool is a CI gate. A Python traceback in a CI log tells the reader that something
    broke and nothing about what to do; the two failures that actually happen here -- not a
    repository, and a clone too shallow for a history scan -- both have one-line remedies.
    """
    proc = subprocess.run(
        ["git", "-C", str(repo)] + args, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        hint = ""
        if "not a git repository" in err:
            hint = "\n       --repo must point at a git repository (default: this file's repo root)."
        elif "shallow" in err or "unknown revision" in err:
            hint = ("\n       A shallow clone has no history to scan. Deepen it first:\n"
                    "         git fetch --unshallow --filter=blob:none origin <branch>")
        raise SystemExit(
            f"error: git {' '.join(args[:3])} ... failed (exit {proc.returncode})\n"
            f"       {err.splitlines()[0] if err else '(no stderr)'}{hint}")
    return proc.stdout


def tracked_images(repo: pathlib.Path, ref: str) -> set[str]:
    listing = run(["ls-tree", "-r", "--name-only", ref] + list(DIRS), repo)
    return {
        line for line in listing.splitlines()
        if line.lower().endswith(EXTS)
    }


def add_events(repo: pathlib.Path, ref: str) -> dict[str, tuple[int, str, str, str]]:
    """path -> (unix_time, iso_date, short_sha, subject) for the commit that ADDED it.

    One pass over history. A path can legitimately appear more than once (added, deleted,
    re-added); the LATEST addition wins, because that is the event a reader is looking for.
    """
    log = run([
        "log", ref, "--diff-filter=A", "--name-only", "--date=short",
        "--format=%x00%at%x00%ad%x00%h%x00%s", "--"] + list(DIRS), repo)

    found: dict[str, tuple[int, str, str, str]] = {}
    meta: tuple[int, str, str, str] | None = None
    for line in log.splitlines():
        if line.startswith("\x00"):
            _, at, ad, sha, subject = line.split("\x00", 4)
            meta = (int(at), ad, sha, subject)
            continue
        if not line or meta is None:
            continue
        if line.lower().endswith(EXTS):
            # `git log` walks newest-first, so the first sighting is the latest addition.
            found.setdefault(line, meta)
    return found


def build(repo: pathlib.Path, ref: str) -> str:
    images = tracked_images(repo, ref)
    events = add_events(repo, ref)

    # The load-bearing check. An image in the tree with no add-commit would be silently dropped,
    # and a feed that drops entries reads as "nothing new" -- the one wrong answer that looks fine.
    undated = sorted(images - events.keys())
    if undated:
        raise SystemExit(
            "error: {} image(s) in the tree have no add-commit, so the feed would be "
            "INCOMPLETE:\n  {}\nRefusing to write a partial file.".format(
                len(undated), "\n  ".join(undated)))

    by_date: dict[str, list[tuple[int, str, str, str, str]]] = defaultdict(list)
    for path in sorted(images):
        at, ad, sha, subject = events[path]
        by_date[ad].append((at, path, sha, subject, ad))

    total = len(images)
    dates = sorted(by_date, reverse=True)
    newest = dates[0] if dates else "never"

    out: list[str] = [
        "# Screenshots — newest first",
        "",
        BANNER,
        "",
        "**Every screenshot checked into this repository, most recent first.** Read down from the "
        "top and stop when you reach one you have already seen.",
        "",
        f"**{total} images**, most recent **{newest}**. Captions are the subject line of the commit "
        "that added each image, so they say what the change was rather than what the picture is.",
        "",
        "This file is generated from git history by "
        "[`prosper/tools/docs/gen_screenshot_feed.py`](prosper/tools/docs/gen_screenshot_feed.py) "
        "and is regenerated and diffed in CI, so it cannot drift from the tree. "
        "[`COMPATIBILITY.md`](COMPATIBILITY.md) remains the per-title overview and "
        "[`PROGRESS_TRACKER.md`](PROGRESS_TRACKER.md) the per-title rung table; this is only a "
        "reverse-chronological index of the images themselves.",
        "",
        "> A screenshot here is evidence of what rendered on the day its commit landed. It is not",
        "> a claim about the title's current state — for that, read the tracker. An image is never",
        "> removed from this feed when a title moves on, because the point of a feed is that it is",
        "> a record of *when* things happened.",
        "",
    ]

    for date in dates:
        entries = sorted(by_date[date], key=lambda e: (-e[0], e[1]))
        out.append(f"## {date}")
        out.append("")
        for _at, path, sha, subject, _ad in entries:
            name = pathlib.PurePosixPath(path).name
            out.append(f"### {name}")
            out.append("")
            label = pathlib.PurePosixPath(path).stem.replace("-", " ").replace("_", " ")
            out.append(f'<p align="center"><img src="{path}" alt="{label}"></p>')
            out.append("")
            # Two-space Markdown hard breaks are trailing whitespace, which `git diff --check`
            # rejects repo-wide. Separate paragraphs instead -- same rendering, no whitespace.
            out.append(subject)
            out.append("")
            out.append(f"`{sha}` · [`{path}`]({path})")
            out.append("")

    return "\n".join(out).rstrip() + "\n"


def selftest() -> int:
    """Prove the tool discriminates, rather than merely running.

    The property that matters is the abort on an undated image. A generator that emitted a
    partial file on missing metadata would fail exactly the way this feed must never fail, so
    the selftest asserts BOTH verdicts: a complete input builds, an incomplete one raises.
    """
    import types

    def fake(images, events):
        mod = types.SimpleNamespace()
        mod.tracked = lambda *a: images
        mod.evts = lambda *a: events
        return mod

    ok_images = {"assets/screenshots/a.png"}
    ok_events = {"assets/screenshots/a.png": (1, "2026-01-01", "abc1234", "feat: a")}

    g_tracked, g_add = globals()["tracked_images"], globals()["add_events"]
    failures = []
    try:
        globals()["tracked_images"] = lambda *a: ok_images
        globals()["add_events"] = lambda *a: ok_events
        text = build(pathlib.Path("."), "HEAD")
        if "a.png" not in text:
            failures.append("complete input did not render its image")

        # The violating arm: an image with no add-commit must ABORT, not emit a partial file.
        globals()["tracked_images"] = lambda *a: ok_images | {"assets/screenshots/orphan.png"}
        try:
            build(pathlib.Path("."), "HEAD")
            failures.append("an undated image did NOT abort -- the feed would silently omit it")
        except SystemExit as exc:
            if "orphan.png" not in str(exc):
                failures.append("abort message does not name the offending image")
    finally:
        globals()["tracked_images"], globals()["add_events"] = g_tracked, g_add

    for f in failures:
        print(f"selftest FAIL: {f}", file=sys.stderr)
    print(f"selftest: 2 arms, {len(failures)} failure(s)")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=None, help="repository root (default: infer from this file)")
    ap.add_argument("--ref", default="HEAD", help="git ref to read the tree and history from")
    ap.add_argument("--check", action="store_true", help="exit non-zero if the file is stale")
    ap.add_argument("--selftest", action="store_true", help="assert the generator still discriminates")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    repo = pathlib.Path(args.repo) if args.repo else pathlib.Path(__file__).resolve().parents[3]
    text = build(repo, args.ref)
    target = repo / OUT

    n = text.count("\n### ")
    if args.check:
        current = target.read_text() if target.exists() else ""
        if current != text:
            # Name BOTH sides. --check compares a file on disk against history read from --ref,
            # and those can legitimately differ (CI deliberately checks a PR branch's copy against
            # master's images). Reporting only "STALE" sent me chasing a phantom when the real
            # cause was a working tree that simply predated the file.
            state = "absent" if not target.exists() else f"{len(current.splitlines())} lines"
            print(
                f"error: {OUT} does not match the images in `{args.ref}`.\n"
                f"       compared: {target} ({state})\n"
                f"       against:  {n} image(s) in the tree at `{args.ref}`\n"
                f"       If those two are meant to describe the same commit, regenerate:\n"
                f"         python3 prosper/tools/docs/gen_screenshot_feed.py\n"
                f"       If the file is simply older than `{args.ref}` (a stale checkout), update\n"
                f"       the checkout first -- this is not a drift.", file=sys.stderr)
            return 1
        print(f"{OUT} is up to date with the tree ({n} images).")
        return 0

    target.write_text(text)
    print(f"{n} image(s) scanned")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
