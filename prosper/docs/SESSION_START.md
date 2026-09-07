# Checkout and instruction freshness

Issue [#2710](https://github.com/mattias800/prosper/issues/2710) reproduced with the shared
checkout on an old feature branch. Its tracked charter was clean but differed from `origin/main`.
The histories had diverged, so `git pull --ff-only` could not repair it. A clean working tree is
not evidence that a session loaded current instructions.

Run from the checkout root:

```sh
python3 prosper/tools/session_start.py
```

The check reports the branch and HEAD, commits ahead/behind `origin/main`, differences in the
working `CLAUDE.md`, `AGENTS.md` and `CONTRIBUTING.md`, and local instruction edits. It then checks
the advertised remote main tip against the tracking ref. A feature branch that includes current
main and keeps the same instructions is valid; detached HEAD is reported explicitly. Intentional
instruction changes still produce a difference to inspect. Windows CRLF checkout conversion is
not an instruction change.

Exit status is 0 for a verified current checkout, 1 for a difference, and 2 when verification
cannot finish. `--offline` performs the local comparison but returns 2: it cannot establish remote
freshness. Git operations have bounded timeouts, share a 12-second check deadline and disable
interactive credential prompts. The
tool does not fetch, change a branch, alter files, or print instruction contents or remote URLs.

Claude Code invokes the check using the checked-in `.claude/settings.json` SessionStart hook.
Its `--hook` mode returns exit 0 with JSON `additionalContext` for all verdicts, including errors,
so failures become visible session context instead of disappearing into hook stderr. This follows
the [SessionStart output contract](https://code.claude.com/docs/en/hooks#sessionstart-decision-control).
The exec-form hook passes paths as separate arguments, including paths containing spaces. It
requires Python 3 and Git on the agent host. Root `AGENTS.md` and `CLAUDE.md` instruct other agents
to run the same command at startup and after moving to another worktree.

When a tracking ref is stale, fetch in your own worktree and repeat the check. Compare instruction
changes against `git show origin/main:CLAUDE.md`; do not replace another lane's work just to clear
a warning. This guard reports the checkout being inspected, not the instructions already retained
inside an older running conversation. That session must read the current files again.

Refreshing a shared checkout is a separate, explicit maintenance action. Preserve its feature
branch and untracked files first. If another worktree owns `main`, a detached checkout of current
`origin/main` refreshes the instructions without moving that peer's branch. Do not force `main`
into two worktrees or reset the peer's branch. Future startup checks still detect when that shared
checkout falls behind again.

`test_session_start.py` uses local Git repositories to exercise the actual configured hook argv
and verify delivered context. Controls cover current and stale instructions, staged edits,
divergence, stale/missing remote refs, unreachable remotes, offline checks, linked worktrees,
detached HEAD, feature branches, Windows line endings and paths containing spaces. It is registered
as the `session_start` CTest case. The contribution-shape gate allows only `.claude/settings.json`,
keeping private settings and agent worktree contents outside the exception.
