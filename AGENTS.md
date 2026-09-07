# Session startup

Read `CLAUDE.md`, `LOCAL.md` if present, and `CONTRIBUTING.md` before working here.
Run `python3 prosper/tools/session_start.py` from the checkout root at the start of a session
and after changing worktrees. This read-only check compares the actual checkout and instruction
files with current `origin/main`; do not treat an unavailable remote as proof of freshness.
If it reports a difference, inspect current instructions with `git show origin/main:CLAUDE.md`
and preserve intentional local changes. Fetch stale refs in your own worktree and rerun the check.
Do not switch or reset someone else's worktree to silence a warning.

Follow the applicable nested `AGENTS.md` files for the area being changed. The startup check is
automatic in Claude Code through `.claude/settings.json`; other agents run the command above.
