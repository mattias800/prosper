---
name: start-task
description: Standard start for any prosper task: sync with origin/main, check the tracker and claim the issue, create an own worktree, read the area docs, and capture a test baseline.
---

# Start a task

1. **Sync and verify the charter.** `git fetch origin`, then `python3 prosper/tools/session_start.py` from the checkout root.
2. **Check the tracker.** `gh issue list --label bug-hunt` or by area. Re-verify the issue still exists on current `main`. Free means no `in-progress` label, no live claim comment, and `git ls-remote origin 'refs/heads/fix/issue-NN-*'` is empty.
3. **Claim it.** Add `in-progress` and comment `CLAIMING: <agent> | branch fix/issue-NN-<slug> | <UTC time>`; re-read comments for a race (earlier timestamp wins). Push the branch early.
4. **Own worktree, never the shared checkout:**
   ```bash
   git worktree add .claude/worktrees/<slug> -b <branch> origin/main
   ```
   Do not `git stash`; use `prosper/tools/wt_stash.py` or a temporary commit.
5. **Read before hypothesising.** The nearest `AGENTS.md`, the status doc named in the `CLAUDE.md` table, and its `## Ruled out` section. Game work starts with `prosper/docs/GAME_COMPAT_ORCHESTRATION.md`.
6. **Plan using the PR template headings:** Context, Higher Goal, Acceptance Criteria, Out of Scope, Summary of Changes.
7. **Baseline.** Configure in the worktree (`cmake -S prosper -B prosper/build-linux`), build, run `ctest --no-tests=error`, and note count and failures so later reds are attributable. Keep `TMPDIR` and run artifacts off `/tmp`.
8. **Tests come with the work:** every fix adds a test that fails without it (`prosper/tests/AGENTS.md`).
9. **Finish:** merge, delete the remote branch, remove the worktree (`CLAUDE.md`, teardown).
