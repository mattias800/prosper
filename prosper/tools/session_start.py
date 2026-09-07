#!/usr/bin/env python3
"""Read-only checkout/instruction freshness signal (#2710).

Exit 0: instructions match current origin/main and the checkout includes it.
Exit 1: stale/divergent checkout or changed instructions. Exit 2: unable to verify.
--hook always exits 0 and emits SessionStart context, including failures. It never
fetches, edits files, changes branches, or treats an unavailable remote as current.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

REFERENCE = "refs/remotes/origin/main"
INSTRUCTIONS = ("CLAUDE.md", "AGENTS.md", "CONTRIBUTING.md")


class Unverified(Exception):
    pass


def git(repo: Path, *args: str, allow_missing: bool = False, deadline: float) -> bytes:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise Unverified("startup check deadline exceeded")
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=min(5, remaining))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Unverified("Git unavailable or timed out while checking " + args[0]) from exc
    if result.returncode and not allow_missing:
        # Name the whole subcommand and its exit status, never git's stderr: the arguments are
        # this file's own literals, while stderr can carry a remote URL, credential-helper
        # output, or instruction contents. Reporting only args[0] was not enough to act on --
        # three distinct calls here are "rev-parse", and #3443 cost an investigation to tell
        # which one had failed.
        raise Unverified("Git could not verify " + " ".join(args)
                         + " (exit " + str(result.returncode) + ")")
    return result.stdout if result.returncode == 0 else b""


def inspect(repo: Path, offline: bool = False) -> tuple[int, str]:
    lines = []
    deadline = time.monotonic() + 12  # leave time to emit context before the 20-second hook limit
    def checked(*args, **kwargs):
        return git(*args, deadline=deadline, **kwargs)
    try:
        # Ask for the root RELATIVELY. Git answers path queries in its own build's path
        # flavour, and an absolute answer does not survive the trip back: the MSYS2 git the
        # Windows CI job installs replies to --show-toplevel in POSIX form (/c/Users/...),
        # which native Python reads as the drive-relative \c\Users\... -- a path neither
        # Python nor that same git can then resolve, so every later -C missed the repository
        # and the check reported UNVERIFIED on every platform pairing but its own (#3443).
        # --show-cdup is relative ("../.." or empty), so the absolute path stays the
        # caller's and never changes flavour.
        cdup = os.fsdecode(checked(repo, "rev-parse", "--show-cdup").strip())
        root = Path(repo, cdup).resolve()
        head = checked(root, "rev-parse", "HEAD").decode().strip()
        branch = checked(root, "symbolic-ref", "--quiet", "--short", "HEAD",
                     allow_missing=True).decode().strip() or "(detached)"
        lines.append(f"Checkout: branch={branch} HEAD={head[:12]}")
        reference = checked(root, "rev-parse", "--verify", REFERENCE).decode().strip()
        ahead, behind = map(int, checked(root, "rev-list", "--left-right", "--count",
                                    f"HEAD...{REFERENCE}").split())
        lines.append(f"origin/main={reference[:12]} ahead={ahead} behind={behind}")
        different = behind != 0
        if behind:
            lines.append("WARNING: checkout lacks commits from origin/main; "
                         "read current instructions before relying on this checkout.")
        for name in INSTRUCTIONS:
            # Missing AGENTS.md is legitimate on older references. Missing CLAUDE.md is not.
            exists = checked(root, "ls-tree", "--name-only", REFERENCE, "--", name).strip()
            if name == "CLAUDE.md" and not exists:
                raise Unverified("origin/main has no CLAUDE.md")
            expected = checked(root, "show", f"{REFERENCE}:{name}") if exists else None
            path = root / name
            actual = path.read_bytes() if path.exists() else None
            # Git checkout may translate LF to CRLF on Windows; this is not a charter change.
            normalize = lambda data: None if data is None else data.replace(b"\r\n", b"\n")
            matches = normalize(actual) == normalize(expected)
            status = checked(root, "status", "--porcelain", "--untracked-files=all", "--", name)
            dirty = status.startswith(b"??") or bool(
                checked(root, "diff", "--ignore-space-at-eol", "--name-only", "HEAD", "--", name) or
                checked(root, "diff", "--cached", "--ignore-space-at-eol", "--name-only", "--", name))
            if not matches or dirty:
                different = True
                lines.append(f"WARNING: {name}: reference={'matches' if matches else 'differs'}; "
                             f"local edits={'yes' if dirty else 'no'}.")
        if offline:
            raise Unverified("remote freshness unchecked (--offline); local comparison only")
        advertised = checked(root, "ls-remote", "--exit-code", "origin", "refs/heads/main")
        rows = [row.split() for row in advertised.splitlines()]
        tips = [row[0].decode() for row in rows
                if len(row) == 2 and row[1] == b"refs/heads/main"]
        if len(tips) != 1:
            raise Unverified("origin did not advertise exactly one main branch")
        if tips[0] != reference:
            lines.append("WARNING: origin/main tracking ref is stale; fetch origin main "
                         "in your own worktree and rerun this check.")
            return 1, "\n".join(lines)
        if different:
            lines.append("Compare intentional instruction edits with git diff origin/main -- "
                         "CLAUDE.md AGENTS.md CONTRIBUTING.md. Preserve other worktrees.")
            return 1, "\n".join(lines)
        lines.append("OK: checkout includes current origin/main; instruction files match.")
        return 0, "\n".join(lines)
    except (Unverified, OSError, ValueError) as exc:
        lines.append("UNVERIFIED: " + str(exc))
        return 2, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--hook", action="store_true")
    args = parser.parse_args()
    code, report = inspect(args.repo, args.offline)
    report = "[session-start] " + report
    if args.hook:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "SessionStart", "additionalContext": report}}))
        return 0
    print(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
