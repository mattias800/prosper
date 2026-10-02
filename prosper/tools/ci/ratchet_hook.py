#!/usr/bin/env python3
"""Claude Code PreToolUse hook: run the architecture ratchet before `git commit` / `git push`.

Why this exists
---------------
`check_arch_ratchet.py` is a baseline ratchet (per-file counts may fall, never rise). CI runs it,
but by the time CI reports, the commit exists and the PR is open. This hook runs the same check at
the moment an agent is about to commit or push, so a new violation is answered in the session that
made it. `.claude/settings.json` registers it for the Bash and PowerShell tools.

Contract (Claude Code hooks, PreToolUse)
----------------------------------------
stdin is one JSON object with `tool_name`, `tool_input.command` and `cwd`. Exit 0 lets the tool
call proceed; exit 2 blocks it and stderr is shown to the model.

This hook exits 2 ONLY when the checker ran and reported a violation (its exit status 1).
Everything else FAILS OPEN with exit 0: a command that is not a commit or push, a checkout without
the checker (it lands in its own PR, and older worktrees will never have it), a checker that
cannot be started or times out, or one that exits 2 ("could not evaluate") or any other status.
A fail-open on a commit or push prints one `ratchet UNVERIFIED` line as a JSON `systemMessage`, so
it is visible and never mistaken for a pass. The asymmetry is deliberate: a hook that wedged every
agent's `git` on a broken interpreter would be switched off within the hour, and then it protects
nothing.

What it does not do
-------------------
It checks the working tree of the repository the command will run in, not the index, so it can
disagree with exactly what a commit records. Command parsing is a heuristic over `;`, `&&`, `||`,
`|` and newlines, following `cd` / `Set-Location` / `pushd` and `git -C`; a commit hidden behind
an alias, a script or `sh -c` is not seen. CI is still the gate; this is the early warning.
"""

import json
import os
import re
import shlex
import subprocess
import sys

CHECKER = os.path.join("prosper", "tools", "ci", "check_arch_ratchet.py")
CHECKER_TIMEOUT_S = 45
GATED_SUBCOMMANDS = ("commit", "push")

_GIT_OPTS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
_CD_COMMANDS = {"cd", "set-location", "sl", "chdir", "pushd", "push-location"}
_SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|\n]")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _tokens(segment, posix):
    """Split one command segment into words.

    PowerShell commands are split non-POSIX so a Windows path's backslashes survive; the
    surrounding quotes that leaves on a token are stripped here.
    """
    try:
        toks = shlex.split(segment, posix=posix)
    except ValueError:
        toks = segment.split()
    if not posix:
        toks = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"" else t for t in toks]
    return toks


def _is_git(token):
    base = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base in ("git", "git.exe")


def _join(base, path):
    if not path:
        return base
    if base is None or os.path.isabs(path) or re.match(r"^[A-Za-z]:[\\/]", path):
        return path
    return os.path.join(base, path)


def gated_git_invocations(command, posix=True):
    """Return [(subcommand, directory_or_None)] for each `git commit|push` in a command line.

    The directory combines any earlier `cd X` with the invocation's own `git -C Y`, relative
    parts left relative so the caller can resolve them against the hook's `cwd`.
    """
    found = []
    cd_dir = None
    for segment in _SEGMENT_SPLIT.split(command or ""):
        toks = _tokens(segment.strip(), posix)
        while toks and _ENV_ASSIGNMENT.match(toks[0]):
            toks = toks[1:]
        if toks and toks[0] == "&":
            toks = toks[1:]
        if not toks:
            continue
        if toks[0].lower() in _CD_COMMANDS:
            if len(toks) >= 2:
                cd_dir = _join(cd_dir, toks[-1])
            continue
        if not _is_git(toks[0]):
            continue
        git_dir = cd_dir
        i = 1
        while i < len(toks):
            tok = toks[i]
            if tok in _GIT_OPTS_WITH_VALUE:
                if tok == "-C" and i + 1 < len(toks):
                    git_dir = _join(git_dir, toks[i + 1])
                i += 2
                continue
            if tok.startswith("-"):
                i += 1
                continue
            break
        if i < len(toks) and toks[i] in GATED_SUBCOMMANDS:
            found.append((toks[i], git_dir))
    return found


def repo_root(start):
    """Top level of the git checkout containing `start`, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", start, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def unverified(reason):
    """Fail open: allow the tool call, and say so where the user sees it."""
    print(json.dumps({"systemMessage": "ratchet UNVERIFIED: " + reason}))
    return 0


def run_checker(root, timeout_s=CHECKER_TIMEOUT_S):
    """Return (status, output) from the checker, or (None, reason) when it could not run."""
    checker = os.path.join(root, CHECKER)
    if not os.path.isfile(checker):
        return None, f"{CHECKER} is not in this checkout (the ratchet has not landed here)"
    try:
        proc = subprocess.run(
            [sys.executable, checker, "--root", root],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, f"the checker did not finish within {timeout_s} s"
    except OSError as error:
        return None, f"the checker could not be started ({error})"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def main(stdin_text):
    """Evaluate one PreToolUse payload; return the hook's exit status."""
    try:
        payload = json.loads(stdin_text or "{}")
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0
    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not isinstance(command, str):
        return 0
    posix = payload.get("tool_name") != "PowerShell"
    invocations = gated_git_invocations(command, posix=posix)
    if not invocations:
        return 0

    base = payload.get("cwd") or os.getcwd()
    subcommand, target = invocations[0]
    root = repo_root(_join(base, target))
    if root is None:
        return unverified(f"could not find the git checkout for `git {subcommand}`")

    status, output = run_checker(root)
    if status is None:
        return unverified(output)
    if status == 0:
        return 0
    if status == 1:
        sys.stderr.write(
            f"Blocked `git {subcommand}`: check_arch_ratchet.py reports a ratchet violation.\n"
            "Fix it, or raise the baseline row in the same PR with a `# note` a reviewer reads"
            f" (CLAUDE.md, 'Architecture and performance ratchets').\n\n{output}"
        )
        return 2
    return unverified(
        f"check_arch_ratchet.py exited {status} (2 = could not evaluate), which is not a verdict"
    )


if __name__ == "__main__":
    try:
        sys.exit(main(sys.stdin.read()))
    except Exception as error:  # noqa: BLE001 - a hook must never wedge the session
        sys.exit(unverified(f"the hook itself failed ({type(error).__name__}: {error})"))
