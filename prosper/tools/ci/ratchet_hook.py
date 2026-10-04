#!/usr/bin/env python3
"""Claude Code PreToolUse hook: run the architecture ratchet before `git commit` / `git push`.

Why this exists
---------------
`check_arch_ratchet.py` is a baseline ratchet (per-file counts may fall, never rise). CI runs it,
but by the time CI reports, the commit exists and the PR is open. This hook runs the same check at
the moment an agent is about to commit or push, so a new violation is answered in the session that
made it. `.claude/settings.json` registers it for the Bash and PowerShell tools.

It runs the checker in DELTA mode (`--base origin/main`): only the files this checkout changed
since its merge base with origin/main are judged, and only a count THIS change raised past its
baseline row is reported. Full mode would report every stale row on main -- and main's baseline
goes stale whenever a PR grows a capped file -- so every agent's commit would be flagged for
something it never touched.

Rollout: WARN-ONLY
------------------
While the rollout is young, a violation does NOT block. The hook exits 0 and shows the findings,
and how to fix them, as a JSON `systemMessage` (and on stderr). `BLOCK_ON_VIOLATION` is the one
switch that turns a violation into a blocking exit 2; the settings entry already passes a 2
through, so flipping it needs no other edit.

Contract (Claude Code hooks, PreToolUse)
----------------------------------------
stdin is one JSON object with `tool_name`, `tool_input.command` and `cwd`. Exit 0 lets the tool
call proceed (a JSON `systemMessage` on stdout is shown to the user); exit 2 blocks it and stderr
is shown to the model.

Everything that is not a verdict FAILS OPEN with exit 0: a command that is not a commit or push, a
checkout without the checker (it lands in its own PR, and older worktrees will never have it), a
checkout with no `origin/main` to take a merge base against, a checker that cannot be started or
times out, or one that exits 2 ("could not evaluate") or any other status. A fail-open on a commit
or push prints one `ratchet UNVERIFIED` line as a JSON `systemMessage`, so it is visible and never
mistaken for a pass. The asymmetry is deliberate: a hook that wedged every agent's `git` on a
broken interpreter would be switched off within the hour, and then it protects nothing.

What it does not do
-------------------
It checks the working tree of the repository the command will run in (committed, staged, unstaged
and untracked changes since the merge base), not the index, so it can disagree with exactly what a
commit records. Command parsing is a heuristic over `;`, `&&`, `||`,
`|` and newlines, following `cd` / `Set-Location` / `pushd` and `git -C`; a commit hidden behind
an alias, a script or `sh -c` is not seen. CI is still the gate; this is the early warning.
"""

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile

CHECKER = os.path.join("prosper", "tools", "ci", "check_arch_ratchet.py")
BASELINE = os.path.join("prosper", "tools", "ci", "arch_ratchet_baseline.txt")
# The ONLY checker code this hook executes: the reviewed copy on the project's own origin/main, read
# from the project repository's object store. Never the copy in the checkout a command names -- that
# may be a contributor's PR branch or an unrelated clone, and this hook runs before the permission
# prompt, so executing a file from it would run unreviewed code with no prompt.
# Fully spelled: git resolves `origin/main` through refs/tags/ and refs/heads/ before refs/remotes/, so a
# local tag or branch of that name (a fork fetch can bring one) would silently replace the checker.
TRUSTED_CHECKER_REF = "refs/remotes/origin/main:prosper/tools/ci/check_arch_ratchet.py"
GIT_TIMEOUT_S = 5
CHECKER_TIMEOUT_S = 45
BASE_REF = "refs/remotes/origin/main"
# The rollout switch. False: a violation is a warning (exit 0 + systemMessage). True: it blocks
# the commit or push (exit 2). Flip it in its own reviewed change once the warnings are trusted.
BLOCK_ON_VIOLATION = False
SYSTEM_MESSAGE_LIMIT = 4000
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


_MSYS_DRIVE_PATH = re.compile(r"^/(?:cygdrive/)?([A-Za-z])(/.*)?$")


def native_path(path, windows=None):
    """`path` as the host's own filesystem API spells it.

    An MSYS2 or Cygwin git (the MinGW CI runner's) prints POSIX drive paths such as `/d/a/x`.
    A native Windows Python cannot use those: as a subprocess `cwd` CreateProcess rejects them
    with WinError 267, and os.path.realpath turns them into `D:\\d\\a\\x`. Map the drive form
    back to `D:/a/x` on Windows; everywhere else, and for any other spelling, return it unchanged.
    """
    if windows is None:
        windows = os.name == "nt"
    if not windows or not path:
        return path
    match = _MSYS_DRIVE_PATH.match(path)
    if not match:
        return path
    return f"{match.group(1).upper()}:{match.group(2) or '/'}"


def repo_root(start):
    """Top level of the git checkout containing `start`, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", start, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return native_path(out.stdout.strip()) or None


def unverified(reason):
    """Fail open: allow the tool call, and say so where the user sees it."""
    print(json.dumps({"systemMessage": "ratchet UNVERIFIED: " + reason}))
    return 0


def has_base(root, ref=BASE_REF):
    """Whether `ref` resolves to a commit in the checkout at `root`."""
    try:
        # Resolve one object without the MSYS-mangled `^{commit}` suffix or a revision range.
        resolved = subprocess.run(
            ["git", "-C", root, "rev-parse", "--verify", "--end-of-options", ref],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
        )
        oid = resolved.stdout.strip()
        if resolved.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid):
            return False
        out = subprocess.run(
            ["git", "-C", root, "rev-list", "-n", "1", oid, "--"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and bool(
        re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", out.stdout.strip())
    )


def _git_out(path, *args):
    """stdout of `git -C path <args>`, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", path, *args], capture_output=True, text=True, timeout=GIT_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def common_dir(path):
    """The shared .git directory of the checkout at `path` (equal across its worktrees), or None."""
    out = _git_out(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return os.path.realpath(native_path(out.strip())) if out and out.strip() else None


def trusted_checker_source(project):
    """The reviewed checker as merged on the project's origin/main, or None."""
    return _git_out(project, "show", TRUSTED_CHECKER_REF)


def run_checker(root, project, timeout_s=CHECKER_TIMEOUT_S):
    """Return (status, output) from the checker, or (None, reason) when it could not run."""
    if not project:
        return None, "CLAUDE_PROJECT_DIR is not set, so there is no trusted checker to run"
    project_git, root_git = common_dir(project), common_dir(root)
    if project_git is None or root_git != project_git:
        return None, "the command targets a checkout that is not this project's repository"
    source = trusted_checker_source(project)
    if source is None:
        return None, f"{CHECKER} is not on origin/main yet (the ratchet has not landed)"
    if not has_base(root):
        return None, f"{BASE_REF} is not available here, so there is no merge base to judge against"
    # The checker locates its defaults from __file__, so it runs from a private temporary copy;
    # --root and --baseline point it at the target tree, whose baseline is data, not code.
    with tempfile.TemporaryDirectory(prefix="ratchet-hook-") as tmp:
        script = os.path.join(tmp, "check_arch_ratchet.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(source)
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    script,
                    "--root",
                    root,
                    "--base",
                    BASE_REF,
                    "--baseline",
                    os.path.join(root, BASELINE),
                ],
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

    status, output = run_checker(root, os.environ.get("CLAUDE_PROJECT_DIR"))
    if status is None:
        return unverified(output)
    if status == 0:
        return 0
    if status == 1:
        how = (
            "this change raised an architecture-ratchet count past its baseline row. Fix it, or "
            "raise the row in prosper/tools/ci/arch_ratchet_baseline.txt in the same PR with a "
            "`# note` a reviewer reads (CLAUDE.md, 'Architecture and performance ratchets'). "
            f"Re-check with: python3 {CHECKER.replace(os.sep, '/')} --root . --base {BASE_REF}"
        )
        if BLOCK_ON_VIOLATION:
            sys.stderr.write(f"Blocked `git {subcommand}`: {how}\n\n{output}")
            return 2
        message = f"ratchet WARNING (not blocking) before `git {subcommand}`: {how}\n\n{output}"
        if len(message) > SYSTEM_MESSAGE_LIMIT:
            message = message[:SYSTEM_MESSAGE_LIMIT] + "\n... (truncated; run the command above)"
        sys.stderr.write(message + "\n")
        print(json.dumps({"systemMessage": message}))
        return 0
    return unverified(
        f"check_arch_ratchet.py exited {status} (2 = could not evaluate), which is not a verdict"
    )


if __name__ == "__main__":
    try:
        sys.exit(main(sys.stdin.read()))
    except Exception as error:  # noqa: BLE001 - a hook must never wedge the session
        sys.exit(unverified(f"the hook itself failed ({type(error).__name__}: {error})"))
